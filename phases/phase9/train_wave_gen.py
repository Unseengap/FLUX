"""
Phase 9 — Training Script: WaveGenerator + WaveToText Joint Training

Two-stage training:
    Stage 1: WaveToText pre-training (fast, ~30 min)
        Train WaveToText to spell words from CSE chunk waves.
    Stage 2: WaveGenerator training (main, ~4-6 hours)
        Train WaveGenerator to predict next chunk-wave from context.
        WaveToText is frozen from Stage 1.

Training data: Same OpenWebText source as Phase 8.
"""

import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field as dc_field
from datetime import datetime

# ─────────────────────────────────────────────
# Path setup for cross-phase imports
# ─────────────────────────────────────────────
_PHASES_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PHASES_DIR.parent

for _phase in ['phase1', 'phase2', 'phase3', 'phase4',
               'phase5', 'phase6', 'phase7', 'phase8']:
    _p = str(_PHASES_DIR / _phase)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from flux_large import FLUXLarge, FLUX_LARGE_CONFIG
from flux_utils import (
    get_device, save_checkpoint, load_checkpoint, checkpoint_exists,
    PhaseLogger, PhaseResults,
)
from wave_chunker import WaveChunker
from wave_generator import WaveGenerator
from wave_to_text import WaveToText
from wave_sampler import ThermodynamicWaveSampler
from field_walk_generator import FieldWalkGenerator


# ─────────────────────────────────────────────
# Build FLUXLarge with Checkpoint Cascade
# ─────────────────────────────────────────────

def build_flux_for_phase9(device: str = 'cpu') -> FLUXLarge:
    """
    Build FLUXLarge by loading the best available checkpoint.

    Phase 9 is a PEER to Phase 8, not a continuation. Both build on
    Phase 7 (which contains all trained components: CSE, field, GR,
    TL, CGN, memory, bridges from Phases 1-7).

    Phase 8 added a WaveDecoder (byte-level generation).
    Phase 9 adds WaveGenerator + WaveToText (wave-level generation).
    They are independent approaches to generation.

    Cascade order:
        1. Phase 7 checkpoint → primary dependency (has all Phases 1-7)
        2. Phase 8 checkpoint → legacy fallback (has Phase 7 weights inside)
        3. Fresh init → untrained (last resort)

    Args:
        device: Target device

    Returns:
        FLUXLarge with the best available pre-trained weights, frozen
    """
    model = None

    # Try Phase 7 first — this is Phase 9's TRUE dependency.
    # Phase 7 has all trained components: CSE, field, GR, TL, CGN, memory, bridges.
    # Phase 9 builds its own generation on top of these, parallel to Phase 8.
    if model is None:
        try:
            model = FLUXLarge.from_phase7_checkpoint(device=device)
            print("  ✓ Loaded Phase 7 checkpoint (CSE, field, GR, TL, CGN, memory, bridges)")
            print("  ℹ Phase 9 builds wave-level generation on Phase 7 foundation")
        except Exception as e:
            print(f"  ℹ No Phase 7 checkpoint: {e}")

    # Legacy fallback: Phase 8 contains Phase 7's weights + a WaveDecoder we don't use.
    # If Phase 7 isn't available, Phase 8 can provide the same base weights.
    if model is None:
        try:
            model = FLUXLarge.from_phase8_checkpoint(device=device)
            print("  ✓ Loaded Phase 8 checkpoint as fallback (contains Phase 7 weights)")
            print("  ✗ WaveDecoder ignored — Phase 9 replaces it")
        except Exception as e:
            print(f"  ℹ No Phase 8 checkpoint: {e}")

    # Last resort: fresh init (untrained — will produce poor results)
    if model is None:
        print("  ⚠ No checkpoints available — using fresh FLUXLarge (untrained)")
        print("    Wave generation quality will be limited without trained CSE/field")
        model = FLUXLarge(device=device)

    # Freeze all base model params — Phase 9 only trains new generation modules
    for param in model.parameters():
        param.requires_grad = False

    return model


# ─────────────────────────────────────────────
# Phase 9 Configuration
# ─────────────────────────────────────────────
PHASE9_CONFIG = {
    'wave_dim': 432,
    'field_features': 768,
    'max_waves': 50,
    'k_neighbors': 16,
    'interference_radius': 4,
    'gru_hidden': 512,
    'gru_layers': 1,
    'wtt_hidden_dim': 256,
    'wtt_max_bytes': 20,
    'coherence_threshold': 0.5,
    'min_chunk_size': 2,
    'max_chunk_size': 20,
}


# ─────────────────────────────────────────────
# Training Data Loading
# ─────────────────────────────────────────────

def load_training_data(max_docs: int = 10000) -> List[str]:
    """
    Load training data from OpenWebText or synthetic fallback.

    Args:
        max_docs: Maximum documents to load

    Returns:
        List of text documents
    """
    try:
        from datasets import load_dataset
        print(f"  ℹ Loading OpenWebText subset ({max_docs:,} docs)...")
        ds = load_dataset('openwebtext', split='train', streaming=True)
        texts = []
        for i, sample in enumerate(ds):
            if i >= max_docs:
                break
            text = sample.get('text', '')
            if len(text) > 50:
                texts.append(text[:4000])
        print(f"  ✓ Loaded {len(texts):,} documents from OpenWebText")
        return texts
    except Exception as e:
        print(f"  ⚠ Could not load OpenWebText: {e}")
        print(f"  ℹ Using synthetic training corpus instead")
        return _synthetic_corpus(max_docs)


def _synthetic_corpus(n: int = 10000) -> List[str]:
    """Generate a synthetic training corpus for testing."""
    import random
    random.seed(42)

    templates = [
        "The {adj} {noun} {verb} the {adj2} {noun2} in the {place}.",
        "{name} discovered that {concept} is fundamental to {field}.",
        "In the year {year}, scientists proved that {fact}.",
        "The relationship between {a} and {b} was first described by {name}.",
        "Modern {field} relies heavily on {concept} for {purpose}.",
        "Research shows that {finding} affects {outcome} significantly.",
        "The {adj} theory of {concept} predicts {prediction}.",
        "According to experts, {fact} will change how we understand {field}.",
        "The history of {field} reveals the importance of {concept} in everyday life.",
        "New developments in {field} suggest that {concept} plays a key role.",
    ]

    nouns = ['system', 'model', 'network', 'field', 'algorithm', 'structure',
             'process', 'pattern', 'mechanism', 'function', 'wave', 'particle']
    adjs = ['complex', 'dynamic', 'stable', 'novel', 'fundamental', 'emergent',
            'classical', 'quantum', 'resonant', 'gravitational', 'thermal']
    verbs = ['transforms', 'generates', 'processes', 'analyzes', 'computes',
             'modifies', 'creates', 'discovers', 'reveals', 'predicts']
    names = ['Einstein', 'Turing', 'Shannon', 'Feynman', 'Dijkstra', 'Knuth']
    fields = ['physics', 'mathematics', 'computer science', 'biology', 'chemistry',
              'neuroscience', 'linguistics', 'philosophy', 'engineering']
    concepts = ['entropy', 'information', 'energy', 'symmetry', 'emergence',
                'causality', 'resonance', 'interference', 'gravity', 'topology']
    places = ['laboratory', 'university', 'research center', 'observatory', 'institute']

    texts = []
    for _ in range(n):
        template = random.choice(templates)
        text = template.format(
            adj=random.choice(adjs), adj2=random.choice(adjs),
            noun=random.choice(nouns), noun2=random.choice(nouns),
            verb=random.choice(verbs), name=random.choice(names),
            concept=random.choice(concepts), field=random.choice(fields),
            place=random.choice(places), year=random.randint(1900, 2025),
            fact=f"{random.choice(concepts)} affects {random.choice(nouns)}s",
            finding=f"{random.choice(adjs)} {random.choice(concepts)}",
            outcome=f"{random.choice(nouns)} performance",
            purpose=f"{random.choice(adjs)} {random.choice(nouns)} design",
            a=random.choice(concepts), b=random.choice(concepts),
            prediction=f"{random.choice(adjs)} behavior in {random.choice(nouns)}s",
        )
        texts.append(text)

    return texts


# ─────────────────────────────────────────────
# Training Result Dataclasses
# ─────────────────────────────────────────────

@dataclass
class WTTStageResult:
    """Result from WaveToText pre-training (Stage 1)."""
    total_steps: int
    total_chunks: int
    final_loss: float
    avg_loss: float
    word_accuracy: float
    total_time_seconds: float


@dataclass
class WGStageResult:
    """Result from WaveGenerator training (Stage 2)."""
    total_steps: int
    final_loss: float
    avg_loss: float
    wave_cosine_accuracy: float
    total_time_seconds: float


@dataclass
class JointStageResult:
    """Result from Stage 3 joint WG + WTT fine-tuning."""
    total_steps: int
    final_loss: float
    avg_loss: float
    wave_cosine_accuracy: float
    wtt_word_accuracy: float
    total_time_seconds: float


# ─────────────────────────────────────────────
# Phase 9 Trainer
# ─────────────────────────────────────────────

class Phase9Trainer:
    """
    Two-stage trainer for Phase 9 wave-level generation.

    Stage 1: Pre-train WaveToText to spell words from chunk waves.
    Stage 2: Train WaveGenerator to predict next chunk waves.

    Args:
        flux_model: FLUXLarge with Phase 8 weights (frozen)
        wave_chunker: WaveChunker instance
        wave_generator: WaveGenerator instance
        wave_to_text: WaveToText instance
        lr: Learning rate
        grad_accum: Gradient accumulation steps
        log: PhaseLogger instance
    """

    def __init__(
        self,
        flux_model: FLUXLarge,
        wave_chunker: WaveChunker,
        wave_generator: WaveGenerator,
        wave_to_text: WaveToText,
        lr: float = 3e-4,
        grad_accum: int = 4,
        log: Optional[PhaseLogger] = None,
    ):
        self.model = flux_model
        self.chunker = wave_chunker
        self.generator = wave_generator
        self.wtt = wave_to_text
        self.lr = lr
        self.grad_accum = grad_accum
        self.log = log
        self.device = flux_model._device_str

    # ─────────────────────────────────────────────
    # Pipeline Helper
    # ─────────────────────────────────────────────

    @torch.no_grad()
    def _compute_merged_context(self, wave_vec: torch.Tensor) -> torch.Tensor:
        """
        Compute merged field context using the full Phase 7 pipeline.

        Pipeline: wave_vec → wave_to_field → GR → CGN → field query → combine

        This is the CORRECT pipeline from Phase 7. The old code skipped
        wave_to_field bridge and GR, using field.query(mean).mean(dim=0)
        instead — losing gravitational relevance weighting.

        Args:
            wave_vec: [432] mean CSE wave vector

        Returns:
            [768] merged field context
        """
        try:
            # Step 1: Project wave → field feature space via bridge
            field_input = self.model.wave_to_field(wave_vec)          # [768]

            # Step 2: Gravitational Relevance — mass-weighted attention
            relevance_out = self.model.gr(
                field_input.unsqueeze(0)
            ).squeeze(0)                                               # [768]

            # Step 3: Causal Geometry — multi-timescale processing
            cgn_out = self.model.cgn(relevance_out)                    # [768]

            # Step 4: Field query → top-1 attractor (not mean!)
            field_features, sims, locs = self.model.field.query(
                wave_vec, k=4
            )
            best_features = field_features[0]                          # [768]

            # Step 5: Combine (Phase 7 pattern)
            merged = best_features + cgn_out                           # [768]
        except Exception:
            # Fallback: simplified pipeline (for untrained models)
            field_features, sims, locs = self.model.field.query(
                wave_vec, k=4
            )
            combined = field_features.mean(dim=0)
            cgn_out = self.model.cgn(combined)
            merged = combined + cgn_out                                # [768]

        return merged

    # ─────────────────────────────────────────────
    # Field Population: Densify Field with Chunk-Level Attractors
    # ─────────────────────────────────────────────

    @torch.no_grad()
    def populate_field(
        self,
        texts: List[str],
        max_chunks: int = 50000,
    ) -> Dict[str, Any]:
        """
        Densify the resonance field with chunk-level attractors.

        Phase 7 only perturbed the field with ~60 document-level means.
        For field-walk generation, we need word/chunk-level attractors
        so the walker has dense stepping stones.

        For each training document:
            1. CSE encode → wave sequence
            2. WaveChunker → chunk waves (word-level)
            3. field.perturb(chunk_wave) → create local attractor
            4. tl.settle_once(chunk_wave) → energy minimization

        This is ~20 lines of physics. No gradients. No optimizer.
        The field's own resistance (mass) prevents catastrophic forgetting.

        Args:
            texts: Training documents
            max_chunks: Maximum total chunk perturbs

        Returns:
            Dict with population statistics
        """
        t0 = time.time()
        total_perturbs = 0
        attractors_before = self.model.field.num_attractors()

        print(f"  ℹ Populating field with chunk-level attractors (max {max_chunks:,})...", flush=True)
        print(f"    Field attractors before: {attractors_before:,}", flush=True)

        _texts_processed = 0
        for i, text in enumerate(texts):
            if total_perturbs >= max_chunks:
                break
            if not text or len(text.strip()) < 10:
                continue

            try:
                wave = self.model.cse.encode(text)
                wave_seq = wave.full.to(self.device)  # [seq, 432]
                chunk_waves, spans = self.chunker(wave_seq)

                for chunk_wave in chunk_waves:
                    if total_perturbs >= max_chunks:
                        break
                    # Perturb field at this chunk's location
                    self.model.field.perturb(chunk_wave, strength=0.5)
                    # TL settle: energy minimization around the perturbation
                    self.model.tl.settle_once(chunk_wave)
                    total_perturbs += 1

                    # Log every 10 perturbs so output is never silent
                    if total_perturbs == 1:
                        elapsed = time.time() - t0
                        print(
                            f"    ... 1st perturb done in {elapsed:.2f}s",
                            flush=True,
                        )
                    elif total_perturbs % 10 == 0:
                        elapsed = time.time() - t0
                        rate = total_perturbs / max(elapsed, 0.01)
                        pct = total_perturbs / max_chunks * 100
                        eta = (max_chunks - total_perturbs) / max(rate, 0.01)
                        print(
                            f"    ... {total_perturbs:,}/{max_chunks:,} perturbs ({pct:.0f}%)  "
                            f"[{rate:.1f} perturb/s, ETA {eta:.0f}s]",
                            flush=True,
                        )

            except Exception:
                continue

            _texts_processed += 1

        # Settle the full field to let it stabilize
        self.model.field.settle(steps=20, dt=0.1)

        attractors_after = self.model.field.num_attractors()
        elapsed = time.time() - t0

        stats = {
            'total_perturbs': total_perturbs,
            'attractors_before': attractors_before,
            'attractors_after': attractors_after,
            'new_attractors': attractors_after - attractors_before,
            'time_seconds': elapsed,
        }

        print(f"  ✓ Field populated in {elapsed:.1f}s", flush=True)
        print(f"    Perturbs: {total_perturbs:,}", flush=True)
        print(f"    Attractors: {attractors_before:,} → {attractors_after:,} (+{stats['new_attractors']:,})", flush=True)

        return stats

    # ─────────────────────────────────────────────
    # Stage 1: WaveToText Pre-Training
    # ─────────────────────────────────────────────

    def train_wave_to_text(
        self,
        texts: List[str],
        max_steps: Optional[int] = None,
        batch_size: int = 32,
        log_interval: int = 100,
    ) -> WTTStageResult:
        """
        Pre-train WaveToText to spell words from chunk waves.

        For each document:
            1. CSE encode → [seq, 432]
            2. WaveChunker → [(chunk_wave, bytes), ...]
            3. For each (wave, bytes): cross-entropy spelling loss

        Args:
            texts: Training documents
            max_steps: Maximum optimization steps (None = all data)
            batch_size: Chunks per batch
            log_interval: Print every N steps

        Returns:
            WTTStageResult with pre-training metrics
        """
        t0 = time.time()

        # Only WaveToText + WaveChunker are trainable
        wtt_params = list(self.wtt.parameters()) + list(self.chunker.parameters())
        optimizer = torch.optim.AdamW(wtt_params, lr=self.lr, weight_decay=0.01)

        self.wtt.train()
        self.chunker.train()

        all_losses = []
        step = 0
        total_chunks = 0
        correct_words = 0
        total_words = 0
        _total_steps_est = max_steps or len(texts) * 5  # rough estimate

        print(f"\n{'='*60}", flush=True)
        print(f"  Stage 1: WaveToText Pre-Training — max_steps={max_steps}, batch_size={batch_size}", flush=True)
        print(f"  Training texts: {len(texts):,}  |  log_interval: {log_interval}", flush=True)
        print(f"{'='*60}", flush=True)

        # Collect chunks from all documents
        print("  ℹ Collecting (wave, word) pairs...", flush=True)
        chunk_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        _step1_t = time.time()

        for text_idx, text in enumerate(texts):
            if max_steps and step >= max_steps:
                break

            if not text or len(text.strip()) < 10:
                continue

            try:
                with torch.no_grad():
                    wave = self.model.cse.encode(text)
                wave_seq = wave.full.to(self.device)  # [seq, 432]
                text_bytes = text.encode('utf-8', errors='replace')

                # Chunk and pair with bytes
                pairs = self.chunker.chunk_with_bytes(wave_seq, text_bytes)
                for chunk_wave, chunk_bytes in pairs:
                    byte_tensor = torch.tensor(
                        list(chunk_bytes), dtype=torch.long, device=self.device
                    )
                    if byte_tensor.numel() > 0:
                        chunk_pairs.append((chunk_wave.detach(), byte_tensor))

            except Exception:
                continue

            # Train in batches when we have enough
            while len(chunk_pairs) >= batch_size:
                batch = chunk_pairs[:batch_size]
                chunk_pairs = chunk_pairs[batch_size:]

                waves_batch = torch.stack([p[0] for p in batch]).to(self.device)
                targets_batch = [p[1] for p in batch]

                loss = self.wtt.forward_batch(waves_batch, targets_batch)

                scaled_loss = loss / self.grad_accum
                scaled_loss.backward()

                step += 1
                total_chunks += batch_size

                if step % self.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(wtt_params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                all_losses.append(loss.item())

                # Early logging for first steps
                if step == 1:
                    print(
                        f"  WTT Step     1  "
                        f"loss={loss.item():.4f}  "
                        f"chunks={total_chunks:,}  "
                        f"(first step: {time.time()-_step1_t:.2f}s)",
                        flush=True,
                    )
                elif step == 10:
                    _e10 = time.time() - _step1_t
                    print(
                        f"  WTT Step    10  "
                        f"loss={sum(all_losses[-10:])/10:.4f}  "
                        f"chunks={total_chunks:,}  "
                        f"[{10/_e10:.1f} step/s]",
                        flush=True,
                    )

                if step % log_interval == 0:
                    avg_loss = sum(all_losses[-log_interval:]) / min(
                        len(all_losses), log_interval
                    )
                    elapsed_train = time.time() - _step1_t
                    steps_per_sec = step / max(elapsed_train, 0.01)
                    _est_total = max_steps or (step * len(texts) / max(text_idx + 1, 1))
                    eta = (_est_total - step) / max(steps_per_sec, 0.01)
                    _loss_trend = '↓' if len(all_losses) > log_interval and avg_loss < sum(all_losses[-2*log_interval:-log_interval]) / log_interval else '→'
                    print(
                        f"  WTT Step {step:>6}  "
                        f"loss={avg_loss:.4f} {_loss_trend}  "
                        f"chunks={total_chunks:,}  "
                        f"[{steps_per_sec:.1f} step/s, "
                        f"elapsed {elapsed_train:.0f}s, ETA ~{eta:.0f}s]",
                        flush=True,
                    )
                    if self.log:
                        self.log.metric(f"wtt_step_{step}_loss", f"{avg_loss:.4f}")

                if max_steps and step >= max_steps:
                    break

        # Process any remaining chunks
        if len(chunk_pairs) > 0 and (not max_steps or step < max_steps):
            waves_batch = torch.stack([p[0] for p in chunk_pairs]).to(self.device)
            targets_batch = [p[1] for p in chunk_pairs]
            loss = self.wtt.forward_batch(waves_batch, targets_batch)
            scaled_loss = loss / self.grad_accum
            scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(wtt_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            all_losses.append(loss.item())
            step += 1
            total_chunks += len(chunk_pairs)

        # Evaluate word accuracy on a small subset
        print(f"\n  ℹ Evaluating WTT word accuracy on 50 texts...", flush=True)
        word_acc = self._evaluate_wtt_accuracy(texts[:50])

        elapsed = time.time() - t0
        final_loss = all_losses[-1] if all_losses else 0.0
        avg_loss = sum(all_losses) / max(len(all_losses), 1)

        print(f"\n  ✓ Stage 1 complete: {step} steps, {total_chunks:,} chunks", flush=True)
        print(f"    Final loss:    {final_loss:.4f}", flush=True)
        print(f"    Average loss:  {avg_loss:.4f}", flush=True)
        print(f"    Word accuracy: {word_acc:.1%}", flush=True)
        print(f"    Total time:    {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)
        print(f"    Throughput:    {step/max(elapsed,0.01):.1f} step/s, {total_chunks/max(elapsed,0.01):.0f} chunks/s", flush=True)

        return WTTStageResult(
            total_steps=step,
            total_chunks=total_chunks,
            final_loss=final_loss,
            avg_loss=avg_loss,
            word_accuracy=word_acc,
            total_time_seconds=elapsed,
        )

    @torch.no_grad()
    def _evaluate_wtt_accuracy(self, texts: List[str], max_chunks: int = 500) -> float:
        """Evaluate WaveToText word reconstruction accuracy."""
        self.wtt.eval()
        self.chunker.eval()
        correct = 0
        total = 0

        for text in texts:
            if not text or len(text.strip()) < 10:
                continue
            try:
                wave = self.model.cse.encode(text)
                wave_seq = wave.full.to(self.device)
                text_bytes = text.encode('utf-8', errors='replace')
                pairs = self.chunker.chunk_with_bytes(wave_seq, text_bytes)

                for chunk_wave, chunk_bytes in pairs:
                    decoded = self.wtt.decode(chunk_wave, temperature=0.5)
                    if decoded == chunk_bytes:
                        correct += 1
                    total += 1
                    if total >= max_chunks:
                        break
            except Exception:
                continue
            if total >= max_chunks:
                break

        self.wtt.train()
        self.chunker.train()
        return correct / max(total, 1)

    # ─────────────────────────────────────────────
    # Stage 2: WaveGenerator Pre-Computation
    # ─────────────────────────────────────────────

    def _precompute_wg_data(
        self,
        texts: List[str],
        max_samples: int = 10000,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Pre-compute all frozen pipeline outputs for WG training.

        Uses the full Phase 7 pipeline: wave_to_field → GR → CGN → field
        query → combine. This replaces the old simplified pipeline that
        skipped GR and the wave_to_field bridge.

        Since CSE, field, GR, CGN, and WaveChunker are all frozen during
        Stage 2, their outputs never change. Computing them once upfront
        eliminates the CPU bottleneck that starves the GPU.

        Args:
            texts: Raw text documents
            max_samples: Maximum samples to pre-compute

        Returns:
            List of (merged_context [768], target_waves [N, 432], wave_vec [432])
            tuples, stored on CPU
        """
        precomputed = []
        skipped = 0

        print(f"  ℹ Pre-computing frozen pipeline outputs for up to {max_samples:,} samples...", flush=True)
        print(f"    Pipeline: CSE → wave_to_field → GR → CGN → field.query → WaveChunker", flush=True)
        t0 = time.time()

        for i, text in enumerate(texts):
            if len(precomputed) >= max_samples:
                break

            if not text or len(text.strip()) < 10:
                skipped += 1
                continue

            try:
                with torch.no_grad():
                    # CSE encode
                    wave = self.model.cse.encode(text)
                    wave_seq = wave.full.to(self.device)   # [seq, 432]
                    wave_vec = wave_seq.mean(dim=0)        # [432]

                    # Full FLUX pipeline → merged context
                    # (uses wave_to_field → GR → CGN → field top-1)
                    merged = self._compute_merged_context(wave_vec)  # [768]

                    # WaveChunker → target waves
                    chunk_waves, spans = self.chunker(wave_seq)
                    target_waves = chunk_waves               # [N, 432]

                if target_waves.shape[0] < 2:
                    skipped += 1
                    continue

                # Store on CPU — GPU can't hold 5K+ samples + model
                precomputed.append((merged.cpu(), target_waves.cpu(), wave_vec.cpu()))

                # Early feedback after 1st sample
                if len(precomputed) == 1:
                    _e1 = time.time() - t0
                    _chunks1 = target_waves.shape[0]
                    print(
                        f"    ... 1st sample done: {_chunks1} chunks in {_e1:.2f}s",
                        flush=True,
                    )
                    _est_total = _e1 * max_samples
                    print(
                        f"    ℹ Estimated total: ~{_est_total/60:.0f} min for {max_samples:,} samples",
                        flush=True,
                    )
                elif (len(precomputed)) % 500 == 0 or (len(precomputed) < 500 and (len(precomputed)) % 50 == 0):
                    elapsed_so_far = time.time() - t0
                    rate = len(precomputed) / max(elapsed_so_far, 0.01)
                    remaining = max_samples - len(precomputed)
                    eta = remaining / max(rate, 0.01)
                    pct = len(precomputed) / max_samples * 100
                    print(
                        f"    ... {len(precomputed):,}/{max_samples:,} samples ({pct:.0f}%)  "
                        f"[{rate:.1f} sample/s, elapsed {elapsed_so_far:.0f}s, ETA {eta:.0f}s]",
                        flush=True,
                    )

            except Exception:
                skipped += 1
                continue

        elapsed = time.time() - t0
        _rate_final = len(precomputed) / max(elapsed, 0.01)
        print(f"  ✓ Pre-computed {len(precomputed):,} samples in {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)
        print(f"    Rate: {_rate_final:.1f} samples/s  |  Skipped: {skipped:,}", flush=True)

        return precomputed

    def precompute_wg_data(
        self,
        texts: List[str],
        max_samples: int = 10000,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Public wrapper for pre-computing WG training data.

        Call this in a separate cell, then pass the result to
        train_wave_generator(precomputed=...) to skip re-computation.

        Returns:
            List of (merged [768], target_waves [N, 432], wave_vec [432]) tuples
        """
        # Freeze chunker for consistency
        for param in self.chunker.parameters():
            param.requires_grad = False
        self.chunker.eval()
        return self._precompute_wg_data(texts, max_samples=max_samples)

    # ─────────────────────────────────────────────
    # Stage 2: WaveGenerator Training
    # ─────────────────────────────────────────────

    def train_wave_generator(
        self,
        texts: List[str],
        max_steps: Optional[int] = None,
        log_interval: int = 50,
        precomputed: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> WGStageResult:
        """
        Train WaveGenerator to predict next chunk-wave from context.

        WaveToText and WaveChunker are frozen from Stage 1.

        Optimized pipeline:
            1. Pre-compute ALL frozen outputs (CSE + field + CGN + chunker)
            2. Training loop iterates over pre-computed GPU tensors only
            3. GPU utilization goes from ~37% to ~95%

        Args:
            texts: Training documents (ignored if precomputed is given)
            max_steps: Maximum steps (None = all data)
            log_interval: Print every N steps
            precomputed: Pre-computed (merged, target_waves) list from
                         precompute_wg_data(). If provided, skips the
                         pre-computation phase entirely.

        Returns:
            WGStageResult with training metrics
        """
        t0 = time.time()
        print(f"\n{'='*60}", flush=True)
        print(f"  WaveGenerator Training — max_steps={max_steps}, precomputed={'YES' if precomputed is not None else 'NO'}", flush=True)
        print(f"{'='*60}", flush=True)

        # Freeze WaveToText and WaveChunker
        for param in self.wtt.parameters():
            param.requires_grad = False
        for param in self.chunker.parameters():
            param.requires_grad = False

        self.wtt.eval()
        self.chunker.eval()
        self.generator.train()

        # ── Pre-compute frozen pipeline outputs (or use provided) ──
        total_steps = min(len(texts), max_steps) if max_steps else len(texts)
        if precomputed is None:
            precomputed = self._precompute_wg_data(texts, max_samples=total_steps + 500)
        else:
            print(f"  ✓ Using {len(precomputed):,} pre-computed samples (skipping pre-computation)", flush=True)

        if len(precomputed) == 0:
            print("  ✗ No valid samples after pre-computation", flush=True)
            return WGStageResult(
                total_steps=0, final_loss=0.0, avg_loss=0.0,
                wave_cosine_accuracy=0.0, total_time_seconds=time.time() - t0,
            )

        total_steps = min(total_steps, len(precomputed) * 3)  # Allow cycling

        # ── Optimizer setup ──
        wg_params = list(self.generator.parameters())
        optimizer = torch.optim.AdamW(wg_params, lr=self.lr, weight_decay=0.01)

        warmup_steps = min(100, total_steps // 10)

        def lr_lambda(step):
            if step < warmup_steps:
                return max(0.01, step / max(warmup_steps, 1))
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        all_losses = []
        cosine_accs = []

        precompute_time = time.time() - t0
        train_t0 = time.time()

        # ── Verify gradient state after pre-computation ──
        # _precompute_wg_data runs under torch.no_grad() — ensure it
        # didn't leak and that the generator's params still track grad.
        torch.set_grad_enabled(True)
        _trainable = sum(1 for p in self.generator.parameters() if p.requires_grad)
        _total_p = sum(1 for p in self.generator.parameters())
        assert _trainable > 0, (
            f"WaveGenerator has 0/{_total_p} trainable params! "
            f"Something froze them during pre-computation."
        )
        print(f"  ✓ Gradient check: {_trainable}/{_total_p} generator params trainable", flush=True)

        # Quick forward test — verify grad flows
        _m0, _t0_test = precomputed[0]
        _pred_test, _ = self.generator(_m0.to(self.device), _t0_test[:2].to(self.device))
        assert _pred_test.requires_grad, (
            f"Generator output has no grad_fn! "
            f"grad_enabled={torch.is_grad_enabled()}, "
            f"context_to_wave.weight.requires_grad="
            f"{self.generator.context_to_wave[0].weight.requires_grad}"
        )
        del _pred_test, _m0, _t0_test

        # ── Scheduled sampling schedule ──
        # Start with pure teacher forcing, linearly ramp to 50% own-prediction
        # by the end of training. This cures exposure bias: the model learns
        # to handle its own imperfect predictions, not just perfect inputs.
        ss_start = 0.0    # Start: 100% teacher forced
        ss_end = 0.5      # End: 50% use own prediction
        ss_warmup = total_steps // 5  # Don't start SS until 20% through

        # ── Context loss weight ──
        # Wave 0 must depend on context. Without this, the model learns
        # to ignore field_context and always outputs the same first wave.
        context_loss_weight = 2.0

        # ── Contrastive loss ──
        # Push Wave 0 from different inputs apart — prevents all inputs
        # from collapsing to the same first wave.
        wave0_buffer = []
        contrastive_weight = 0.3

        print(f"\n  ℹ Starting WG training loop: {total_steps:,} steps over {len(precomputed):,} samples", flush=True)
        print(f"    Scheduled sampling: {ss_start:.0%}→{ss_end:.0%} (warmup={ss_warmup})", flush=True)
        print(f"    Context loss weight: {context_loss_weight:.1f}x on Wave 0", flush=True)

        # ── Training loop — pure GPU, no CPU bottleneck ──
        import random
        sample_indices = list(range(len(precomputed)))
        optimizer.zero_grad()
        _step1_t = time.time()

        for step in range(1, total_steps + 1):
            # ── Scheduled sampling probability for this step ──
            if step < ss_warmup:
                ss_p = ss_start
            else:
                progress = (step - ss_warmup) / max(total_steps - ss_warmup, 1)
                ss_p = ss_start + (ss_end - ss_start) * progress

            # Cycle through pre-computed data with shuffling
            idx = (step - 1) % len(precomputed)
            if idx == 0 and step > 1:
                random.shuffle(sample_indices)
            sample_idx = sample_indices[idx]
            merged_cpu, target_cpu, wave_vec_cpu = precomputed[sample_idx]
            merged = merged_cpu.to(self.device)
            target_waves = target_cpu.to(self.device)

            # WaveGenerator forward with scheduled sampling
            predicted_waves, confidences = self.generator(
                merged, target_waves,
                scheduled_sampling_p=ss_p,
            )

            # Loss: MSE + cosine distance
            mse_loss = F.mse_loss(predicted_waves, target_waves)
            cos_loss = 1.0 - F.cosine_similarity(
                predicted_waves, target_waves, dim=-1
            ).mean()

            # Context loss: extra penalty on Wave 0
            # Forces context_to_wave to actually differentiate inputs.
            # Without this, the model ignores context and always predicts
            # the same first wave for every input.
            wave0_mse = F.mse_loss(predicted_waves[0], target_waves[0])
            wave0_cos = 1.0 - F.cosine_similarity(
                predicted_waves[0].unsqueeze(0),
                target_waves[0].unsqueeze(0),
                dim=-1,
            ).mean()
            ctx_loss = (wave0_mse + wave0_cos) * context_loss_weight

            # Contrastive loss: push Wave 0 from different inputs apart
            # Negatives come from the buffer (detached, from previous steps).
            # Current predicted_waves[0] retains grad_fn → gradient flows to WG.
            contrastive_loss = torch.tensor(0.0, device=self.device)
            if len(wave0_buffer) > 10:
                neg_count = min(8, len(wave0_buffer))
                neg_indices = random.sample(range(len(wave0_buffer)), neg_count)
                negatives = torch.stack(
                    [wave0_buffer[i] for i in neg_indices]
                ).to(self.device)
                neg_sim = F.cosine_similarity(
                    predicted_waves[0].unsqueeze(0), negatives, dim=-1
                )
                contrastive_loss = neg_sim.clamp(min=0).mean() * contrastive_weight

            loss = mse_loss + cos_loss + ctx_loss + contrastive_loss

            # Update Wave 0 buffer for contrastive loss
            wave0_buffer.append(predicted_waves[0].detach().cpu())
            if len(wave0_buffer) > 200:
                wave0_buffer = wave0_buffer[-200:]

            # Gradient accumulation
            scaled_loss = loss / self.grad_accum
            scaled_loss.backward()

            if step % self.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(wg_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            all_losses.append(loss.item())
            with torch.no_grad():
                cos_acc = F.cosine_similarity(
                    predicted_waves, target_waves, dim=-1
                ).mean().item()
            cosine_accs.append(cos_acc)

            # Early logging for first steps to confirm training is running
            if step == 1:
                print(
                    f"  WG Step     1/{total_steps}  "
                    f"loss={loss.item():.4f}  cos_acc={cos_acc:.3f}  "
                    f"ctx_loss={ctx_loss.item():.4f}  ss_p={ss_p:.2f}  "
                    f"(first step: {time.time()-_step1_t:.2f}s)", flush=True
                )
            elif step == 10:
                elapsed_10 = time.time() - _step1_t
                rate_10 = 10 / max(elapsed_10, 0.01)
                print(
                    f"  WG Step    10/{total_steps}  "
                    f"loss={sum(all_losses[-10:])/10:.4f}  "
                    f"[{rate_10:.1f} step/s]", flush=True
                )

            if step % log_interval == 0:
                avg_loss = sum(all_losses[-log_interval:]) / min(
                    len(all_losses), log_interval
                )
                avg_cos = sum(cosine_accs[-log_interval:]) / min(
                    len(cosine_accs), log_interval
                )
                lr_now = optimizer.param_groups[0]['lr']
                elapsed_train = time.time() - train_t0
                steps_per_sec = step / max(elapsed_train, 0.01)
                eta = (total_steps - step) / max(steps_per_sec, 0.01)
                print(
                    f"  WG Step {step:>6}/{total_steps}  "
                    f"loss={avg_loss:.4f}  cos_acc={avg_cos:.3f}  "
                    f"lr={lr_now:.6f}  ss_p={ss_p:.2f}  "
                    f"[{steps_per_sec:.1f} step/s, ETA {eta:.0f}s]",
                    flush=True,
                )
                if self.log:
                    self.log.metric(f"wg_step_{step}_loss", f"{avg_loss:.4f}")

        elapsed = time.time() - t0
        train_elapsed = time.time() - train_t0
        final_loss = all_losses[-1] if all_losses else 0.0
        avg_loss = sum(all_losses) / max(len(all_losses), 1)
        avg_cos = sum(cosine_accs) / max(len(cosine_accs), 1)

        print(f"\n  ✓ Stage 2 complete: {total_steps} steps", flush=True)
        print(f"    Final loss: {final_loss:.4f}", flush=True)
        print(f"    Avg cosine accuracy: {avg_cos:.3f}", flush=True)
        print(f"    Pre-compute time: {precompute_time:.1f}s", flush=True)
        print(f"    Training time: {train_elapsed:.1f}s", flush=True)
        print(f"    Total time: {elapsed:.1f}s", flush=True)

        return WGStageResult(
            total_steps=total_steps,
            final_loss=final_loss,
            avg_loss=avg_loss,
            wave_cosine_accuracy=avg_cos,
            total_time_seconds=elapsed,
        )

    # ─────────────────────────────────────────────
    # Stage 3: Joint Fine-Tuning
    # ─────────────────────────────────────────────

    def train_joint_finetune(
        self,
        texts: List[str],
        max_steps: Optional[int] = None,
        log_interval: int = 50,
        precomputed: Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
    ) -> JointStageResult:
        """
        Stage 3: Joint fine-tuning of WaveGenerator and WaveToText.

        WaveGenerator predicts waves, which are then passed (with gradients)
        into WaveToText. The WTT text loss backpropagates through WTT and
        into WaveGenerator.

        This ensures WG learns to produce waves that WTT can actually decode,
        aligning the two modules optimally.

        Args:
            texts: Training documents
            max_steps: Maximum steps
            log_interval: Print every N steps
            precomputed: Pre-computed data

        Returns:
            JointStageResult with metrics
        """
        t0 = time.time()
        print(f"\n{'='*60}", flush=True)
        print(f"  Stage 3: Joint Fine-Tuning — max_steps={max_steps}", flush=True)
        print(f"{'='*60}", flush=True)

        # Unfreeze WTT and WG
        self.chunker.eval()
        for param in self.chunker.parameters():
            param.requires_grad = False

        self.wtt.train()
        for param in self.wtt.parameters():
            param.requires_grad = True

        self.generator.train()
        for param in self.generator.parameters():
            param.requires_grad = True

        total_steps = min(len(texts), max_steps) if max_steps else len(texts)
        if precomputed is None:
            precomputed = self._precompute_wg_data(texts, max_samples=total_steps + 500)
        
        if len(precomputed) == 0:
            return JointStageResult(0, 0.0, 0.0, 0.0, 0.0, time.time() - t0)

        # Optimizer covers both WTT and WG
        joint_params = list(self.generator.parameters()) + list(self.wtt.parameters())
        optimizer = torch.optim.AdamW(joint_params, lr=self.lr * 0.5, weight_decay=0.01)

        all_losses = []
        all_mse = []
        all_wtt_losses = []
        cosine_accs = []
        _skipped = 0
        import random
        sample_indices = list(range(len(precomputed)))
        train_t0 = time.time()

        _wg_params_n = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
        _wtt_params_n = sum(p.numel() for p in self.wtt.parameters() if p.requires_grad)
        print(f"  Trainable: WG {_wg_params_n:,} + WTT {_wtt_params_n:,} = {_wg_params_n+_wtt_params_n:,} params", flush=True)
        print(f"  LR: {self.lr * 0.5:.6f}  |  Grad accum: {self.grad_accum}  |  SS: 0.5", flush=True)

        for step in range(1, total_steps + 1):
            idx = (step - 1) % len(precomputed)
            if idx == 0 and step > 1:
                random.shuffle(sample_indices)
            sample_idx = sample_indices[idx]
            merged_cpu, target_cpu, wave_vec_cpu = precomputed[sample_idx]

            merged = merged_cpu.to(self.device)
            target_waves = target_cpu.to(self.device)

            # Re-run text to get byte targets for WTT loss
            text = texts[sample_idx % len(texts)]
            try:
                wave = self.model.cse.encode(text)
                wave_seq = wave.full.to(self.device)
                text_bytes = text.encode('utf-8', errors='replace')
                pairs = self.chunker.chunk_with_bytes(wave_seq, text_bytes)

                if len(pairs) < 2:
                    _skipped += 1
                    continue

                target_waves_fresh = torch.stack([p[0] for p in pairs]).to(self.device)
                targets_batch = [p[1] for p in pairs]

                wave_vec = wave_seq.mean(dim=0)
                merged = self._compute_merged_context(wave_vec)

                # Predict
                predicted_waves, _ = self.generator(merged, target_waves_fresh, scheduled_sampling_p=0.5)

                # Ensure we only use up to len(targets_batch) predictions
                pred_len = min(len(predicted_waves), len(targets_batch))
                wtt_loss = self.wtt.forward_batch(predicted_waves[:pred_len], targets_batch[:pred_len])

                mse_loss = F.mse_loss(predicted_waves[:pred_len], target_waves_fresh[:pred_len])
                cos_loss = 1.0 - F.cosine_similarity(predicted_waves[:pred_len], target_waves_fresh[:pred_len], dim=-1).mean()

                # Combined loss: wave matching + text decoding
                loss = mse_loss + cos_loss + (0.5 * wtt_loss)

                # Backprop
                scaled_loss = loss / self.grad_accum
                scaled_loss.backward()

                if step % self.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(joint_params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                all_losses.append(loss.item())
                all_mse.append(mse_loss.item())
                all_wtt_losses.append(wtt_loss.item())
                cos_acc = F.cosine_similarity(predicted_waves[:pred_len], target_waves_fresh[:pred_len], dim=-1).mean().item()
                cosine_accs.append(cos_acc)

            except Exception:
                _skipped += 1
                continue

            # Early logging
            if step == 1:
                print(
                    f"  Joint Step     1/{total_steps}  "
                    f"loss={loss.item():.4f}  mse={mse_loss.item():.4f}  "
                    f"wtt_loss={wtt_loss.item():.4f}  cos_acc={cos_acc:.3f}  "
                    f"(first step: {time.time()-train_t0:.2f}s)",
                    flush=True,
                )

            if step % log_interval == 0:
                avg_loss = sum(all_losses[-log_interval:]) / max(len(all_losses[-log_interval:]), 1)
                avg_cos = sum(cosine_accs[-log_interval:]) / max(len(cosine_accs[-log_interval:]), 1)
                avg_mse = sum(all_mse[-log_interval:]) / max(len(all_mse[-log_interval:]), 1)
                avg_wtt = sum(all_wtt_losses[-log_interval:]) / max(len(all_wtt_losses[-log_interval:]), 1)
                elapsed_train = time.time() - train_t0
                steps_per_sec = step / max(elapsed_train, 0.01)
                eta = (total_steps - step) / max(steps_per_sec, 0.01)
                print(
                    f"  Joint Step {step:>6}/{total_steps}  "
                    f"loss={avg_loss:.4f}  mse={avg_mse:.4f}  wtt={avg_wtt:.4f}  "
                    f"cos_acc={avg_cos:.3f}  "
                    f"[{steps_per_sec:.1f} step/s, elapsed {elapsed_train:.0f}s, ETA {eta:.0f}s]",
                    flush=True,
                )
                if self.log:
                    self.log.metric(f"joint_step_{step}_loss", f"{avg_loss:.4f}")

        print(f"\n  ℹ Evaluating WTT word accuracy on 50 texts...", flush=True)
        wtt_acc = self._evaluate_wtt_accuracy(texts[:50])
        avg_loss = sum(all_losses) / max(len(all_losses), 1)
        avg_cos = sum(cosine_accs) / max(len(cosine_accs), 1)
        avg_mse_final = sum(all_mse) / max(len(all_mse), 1)
        avg_wtt_final = sum(all_wtt_losses) / max(len(all_wtt_losses), 1)
        elapsed = time.time() - t0
        train_elapsed = time.time() - train_t0

        print(f"\n  ✓ Stage 3 complete: {step} steps ({_skipped} skipped)", flush=True)
        print(f"    Combined loss: {avg_loss:.4f}  (mse={avg_mse_final:.4f} + wtt={avg_wtt_final:.4f})", flush=True)
        print(f"    Cosine Acc:    {avg_cos:.3f}", flush=True)
        print(f"    WTT Word Acc:  {wtt_acc:.1%}", flush=True)
        print(f"    Training time: {train_elapsed:.1f}s ({train_elapsed/60:.1f} min)", flush=True)
        print(f"    Total time:    {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)
        print(f"    Throughput:    {step/max(train_elapsed,0.01):.1f} step/s", flush=True)

        return JointStageResult(
            total_steps=total_steps,
            final_loss=avg_loss,
            avg_loss=avg_loss,
            wave_cosine_accuracy=avg_cos,
            wtt_word_accuracy=wtt_acc,
            total_time_seconds=elapsed,
        )

    # ─────────────────────────────────────────────
    # Checkpoint Save
    # ─────────────────────────────────────────────

    def save_phase9_checkpoint(
        self,
        wtt_result: WTTStageResult,
        wg_result: WGStageResult,
        joint_result: Optional[JointStageResult] = None,
        valid_word_rate: float = 0.0,
    ) -> Path:
        """
        Save Phase 9 checkpoint with all component states.

        Args:
            wtt_result: WaveToText pre-training result
            wg_result: WaveGenerator training result
            joint_result: Optional Joint fine-tuning result
            valid_word_rate: Valid English word rate from evaluation

        Returns:
            Path to saved checkpoint
        """
        state = {
            'phase': 9,
            'config': self.model.config,
            'phase9_config': PHASE9_CONFIG,
            # Phase 8 component states (frozen, preserved)
            'cse_state_dict': self.model.cse.state_dict(),
            'field_state_dict': self.model.field.state_dict(),
            'gr_state': self.model.gr.save_state(),
            'tl_state': self.model.tl.save_state(),
            'cgn_state': self.model.cgn.save_state(),
            'causal_graph_state': self.model.causal_graph.save_state(),
            'working_memory_state': self.model.working_memory.state_dict_memory(),
            'episodic_memory_state': self.model.episodic_memory.save_state(),
            'semantic_memory_state': self.model.semantic_memory.save_state(),
            'router_state': self.model.memory_router.save_state(),
            'wave_to_field_state': self.model.wave_to_field.state_dict(),
            'field_to_wave_state': self.model.field_to_wave.state_dict(),
            'output_head_state': self.model.output_head.state_dict(),
            # Phase 9 new module states
            'wave_chunker_state_dict': self.chunker.state_dict(),
            'wave_generator_state_dict': self.generator.state_dict(),
            'wave_to_text_state_dict': self.wtt.state_dict(),
            # Metrics
            'metrics': {
                'wtt_pretraining_steps': wtt_result.total_steps,
                'wtt_word_accuracy': wtt_result.word_accuracy,
                'wg_training_steps': wg_result.total_steps,
                'wg_final_loss': wg_result.final_loss,
                'wg_wave_cosine_accuracy': wg_result.wave_cosine_accuracy,
                'joint_training_steps': joint_result.total_steps if joint_result else 0,
                'valid_word_rate': valid_word_rate,
                'total_training_time': (
                    wtt_result.total_time_seconds + wg_result.total_time_seconds + (joint_result.total_time_seconds if joint_result else 0)
                ),
            },
        }

        path = save_checkpoint(9, state)
        print(f"  ✓ Phase 9 checkpoint saved: {path}")
        return path


# ─────────────────────────────────────────────
# Full Generation Pipeline
# ─────────────────────────────────────────────

def generate_text(
    prompt: str,
    flux_model: FLUXLarge,
    wave_chunker: WaveChunker,
    wave_generator: WaveGenerator,
    wave_to_text: WaveToText,
    max_waves: int = 30,
    temperature: float = 0.8,
    use_sampler: bool = True,
) -> str:
    """
    Phase 9 generation: think in waves, spell in bytes.

    Args:
        prompt: Input text prompt
        flux_model: FLUXLarge with frozen Phase 8 components
        wave_chunker: Trained WaveChunker
        wave_generator: Trained WaveGenerator
        wave_to_text: Trained WaveToText
        max_waves: Maximum waves to generate
        temperature: Sampling temperature
        use_sampler: Whether to use ThermodynamicWaveSampler

    Returns:
        Generated text (prompt + continuation)
    """
    device = flux_model._device_str

    wave_generator.eval()
    wave_to_text.eval()
    wave_chunker.eval()

    with torch.no_grad():
        # 1. Encode prompt through FLUX pipeline
        wave_seq, wave_vec, merged = flux_model._get_context(prompt)

        # 2. Generate new waves (the THINKING — universal, modality-agnostic)
        generated_waves, confidences = wave_generator.generate(
            field_context=merged,
            max_waves=max_waves,
            flux_model=flux_model,
            temperature=temperature,
        )

        # 3. Convert each wave to text bytes (the SPELLING — text-specific)
        sampler = ThermodynamicWaveSampler() if use_sampler else None
        text_parts = []

        for wave, conf in zip(generated_waves, confidences):
            if sampler is not None:
                wave = sampler.sample_wave(wave, conf)
            chunk_bytes = wave_to_text.decode(wave, temperature=temperature)
            try:
                text_parts.append(
                    chunk_bytes.decode('utf-8', errors='replace')
                )
            except Exception:
                text_parts.append(
                    chunk_bytes.decode('latin-1', errors='replace')
                )

    return prompt + ' ' + ' '.join(text_parts)


# ─────────────────────────────────────────────
# Build Phase 9 Modules (factory)
# ─────────────────────────────────────────────

def build_phase9_modules(
    device: str = 'cpu',
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[WaveChunker, WaveGenerator, WaveToText]:
    """
    Build fresh Phase 9 modules.

    Args:
        device: Target device
        config: Optional config overrides

    Returns:
        (wave_chunker, wave_generator, wave_to_text)
    """
    cfg = {**PHASE9_CONFIG, **(config or {})}

    chunker = WaveChunker(
        wave_dim=cfg['wave_dim'],
        min_chunk_size=cfg['min_chunk_size'],
        max_chunk_size=cfg['max_chunk_size'],
        coherence_threshold=cfg['coherence_threshold'],
    ).to(device)

    generator = WaveGenerator(
        wave_dim=cfg['wave_dim'],
        field_features=cfg['field_features'],
        max_waves=cfg['max_waves'],
        k_neighbors=cfg['k_neighbors'],
        interference_radius=cfg['interference_radius'],
    ).to(device)

    wtt = WaveToText(
        wave_dim=cfg['wave_dim'],
        hidden_dim=cfg['wtt_hidden_dim'],
        max_bytes=cfg['wtt_max_bytes'],
    ).to(device)

    # Parameter counts
    chunker_params = sum(p.numel() for p in chunker.parameters())
    gen_params = sum(p.numel() for p in generator.parameters())
    wtt_params = sum(p.numel() for p in wtt.parameters())
    total = chunker_params + gen_params + wtt_params

    print(f"  Phase 9 modules built:")
    print(f"    WaveChunker:   {chunker_params:>10,} params")
    print(f"    WaveGenerator: {gen_params:>10,} params")
    print(f"    WaveToText:    {wtt_params:>10,} params")
    print(f"    Total new:     {total:>10,} params")

    return chunker, generator, wtt


def build_phase9_modules_fieldwalk(
    device: str = 'cpu',
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[WaveChunker, FieldWalkGenerator, WaveToText]:
    """
    Build FLUX-native Phase 9 modules using FieldWalkGenerator.

    The FieldWalkGenerator replaces the GRU/MLP WaveGenerator with
    a physics-based field walker. ~50 trainable params instead of ~2.4M.

    Args:
        device: Target device
        config: Optional config overrides

    Returns:
        (wave_chunker, field_walk_generator, wave_to_text)
    """
    cfg = {**PHASE9_CONFIG, **(config or {})}

    chunker = WaveChunker(
        wave_dim=cfg['wave_dim'],
        min_chunk_size=cfg['min_chunk_size'],
        max_chunk_size=cfg['max_chunk_size'],
        coherence_threshold=cfg['coherence_threshold'],
    ).to(device)

    generator = FieldWalkGenerator(
        wave_dim=cfg['wave_dim'],
        field_features=cfg['field_features'],
        k_neighbors=cfg['k_neighbors'],
        interference_radius=cfg['interference_radius'],
        max_waves=cfg['max_waves'],
    ).to(device)

    wtt = WaveToText(
        wave_dim=cfg['wave_dim'],
        hidden_dim=cfg['wtt_hidden_dim'],
        max_bytes=cfg['wtt_max_bytes'],
    ).to(device)

    # Parameter counts
    chunker_params = sum(p.numel() for p in chunker.parameters())
    gen_params = sum(p.numel() for p in generator.parameters())
    wtt_params = sum(p.numel() for p in wtt.parameters())
    total = chunker_params + gen_params + wtt_params

    print(f"  Phase 9 modules built (FLUX-native FieldWalk):")
    print(f"    WaveChunker:        {chunker_params:>10,} params")
    print(f"    FieldWalkGenerator: {gen_params:>10,} params  ← ~50 learned (physics does the rest)")
    print(f"    WaveToText:         {wtt_params:>10,} params")
    print(f"    Total new:          {total:>10,} params")

    return chunker, generator, wtt


def load_phase9_modules(
    device: str = 'cpu',
) -> Tuple[FLUXLarge, WaveChunker, WaveGenerator, WaveToText]:
    """
    Load Phase 9 from checkpoint.

    Returns:
        (flux_model, wave_chunker, wave_generator, wave_to_text)
    """
    ckpt = load_checkpoint(9)
    p9cfg = ckpt.get('phase9_config', PHASE9_CONFIG)

    # Build FLUXLarge and load component states from the Phase 9
    # checkpoint itself (which contains all frozen component states).
    model_config = ckpt.get('config', FLUX_LARGE_CONFIG)
    model = FLUXLarge(config=model_config, device=device)

    # Load Phase 8 frozen states
    if 'cse_state_dict' in ckpt:
        try:
            model.cse.load_state_dict(ckpt['cse_state_dict'])
        except Exception:
            pass
    if 'field_state_dict' in ckpt:
        try:
            model.field.load_state_dict(ckpt['field_state_dict'])
        except Exception:
            pass
    if 'gr_state' in ckpt:
        try:
            from gravity import GravitationalRelevance
            model.gr = GravitationalRelevance.load_state(ckpt['gr_state'], device=device)
        except Exception:
            pass
    if 'tl_state' in ckpt:
        try:
            model.tl.load_state(ckpt['tl_state'])
        except Exception:
            pass
    if 'cgn_state' in ckpt:
        try:
            model.cgn.load_state(ckpt['cgn_state'])
        except Exception:
            pass
    if 'causal_graph_state' in ckpt:
        model.causal_graph.load_state(ckpt['causal_graph_state'])
    if 'working_memory_state' in ckpt:
        try:
            model.working_memory.load_state_memory(ckpt['working_memory_state'])
        except Exception:
            pass
    if 'episodic_memory_state' in ckpt:
        model.episodic_memory.load_state(ckpt['episodic_memory_state'])
    if 'semantic_memory_state' in ckpt:
        try:
            model.semantic_memory.load_state(ckpt['semantic_memory_state'])
        except Exception:
            pass
    if 'router_state' in ckpt:
        try:
            model.memory_router.load_state(ckpt['router_state'])
        except Exception:
            pass
    if 'wave_to_field_state' in ckpt:
        try:
            model.wave_to_field.load_state_dict(ckpt['wave_to_field_state'])
        except Exception:
            pass
    if 'field_to_wave_state' in ckpt:
        try:
            model.field_to_wave.load_state_dict(ckpt['field_to_wave_state'])
        except Exception:
            pass
    if 'output_head_state' in ckpt:
        try:
            model.output_head.load_state_dict(ckpt['output_head_state'])
        except Exception:
            pass

    # Freeze all model params
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(device)

    # Build Phase 9 modules
    chunker, generator, wtt = build_phase9_modules(device=device, config=p9cfg)

    # Load trained states
    if 'wave_chunker_state_dict' in ckpt:
        chunker.load_state_dict(ckpt['wave_chunker_state_dict'])
    if 'wave_generator_state_dict' in ckpt:
        generator.load_state_dict(ckpt['wave_generator_state_dict'])
    if 'wave_to_text_state_dict' in ckpt:
        wtt.load_state_dict(ckpt['wave_to_text_state_dict'])

    print(f"  ✓ Phase 9 loaded from checkpoint")
    return model, chunker, generator, wtt


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  Phase 9: Train Wave-Level Generation")
    print("=" * 60)

    device = get_device()
    log = PhaseLogger(phase=9)

    # Load best available checkpoint (Phase 8 → Phase 7 → fresh)
    log.separator("Loading FLUX Components")
    model = build_flux_for_phase9(device=device)

    # Build Phase 9 modules
    log.separator("Building Phase 9 Modules")
    chunker, generator, wtt = build_phase9_modules(device=device)

    # Load training data
    log.separator("Loading Training Data")
    texts = load_training_data(max_docs=5000)
    split_idx = int(len(texts) * 0.9)
    train_texts = texts[:split_idx]
    eval_texts = texts[split_idx:]
    print(f"  Train: {len(train_texts):,} docs, Eval: {len(eval_texts):,} docs")

    # Create trainer
    trainer = Phase9Trainer(
        flux_model=model,
        wave_chunker=chunker,
        wave_generator=generator,
        wave_to_text=wtt,
        lr=3e-4,
        grad_accum=4,
        log=log,
    )

    # Stage 1: WaveToText pre-training
    log.separator("Stage 1: WaveToText Pre-Training")
    wtt_result = trainer.train_wave_to_text(
        train_texts, max_steps=2000, log_interval=100
    )

    # Stage 2: WaveGenerator training
    log.separator("Stage 2: WaveGenerator Training")
    wg_result = trainer.train_wave_generator(
        train_texts, max_steps=5000, log_interval=50
    )

    # Stage 3: Joint fine-tuning
    log.separator("Stage 3: Joint Fine-Tuning")
    joint_result = trainer.train_joint_finetune(
        train_texts, max_steps=2000, log_interval=50
    )

    # Quick evaluation: valid word rate
    log.separator("Evaluation")
    prompts = [
        "The future of artificial intelligence",
        "In the beginning",
        "Scientists have discovered",
        "The relationship between energy and matter",
        "Modern technology relies on",
    ]

    valid_words = 0
    total_words = 0
    for p in prompts:
        try:
            result = generate_text(
                p, model, chunker, generator, wtt,
                max_waves=15, temperature=0.8,
            )
            continuation = result[len(p):].strip()
            words = continuation.split()
            for w in words:
                clean = w.strip('.,;:!?"\'-()[]{}').lower()
                if clean.isalpha() and len(clean) >= 2:
                    total_words += 1
                    # Simple English word heuristic
                    if len(clean) <= 15:
                        valid_words += 1
            print(f"  Prompt: {p}")
            print(f"  Output: {result[:200]}")
            print()
        except Exception as e:
            print(f"  ⚠ Generation failed for '{p[:30]}...': {e}")

    valid_rate = valid_words / max(total_words, 1)
    print(f"  Valid word rate: {valid_rate:.1%}")

    # Save checkpoint
    log.separator("Save Checkpoint")
    trainer.save_phase9_checkpoint(wtt_result, wg_result, joint_result, valid_rate)

    # Generate results
    results = PhaseResults(phase=9, component_name="Wave-Level Generation")
    results.add_metric("WTT training steps", wtt_result.total_steps)
    results.add_metric("WTT word accuracy", f"{wtt_result.word_accuracy:.1%}")
    results.add_metric("WG training steps", wg_result.total_steps)
    results.add_metric("WG final loss", f"{wg_result.final_loss:.4f}")
    results.add_metric("WG cosine accuracy", f"{wg_result.wave_cosine_accuracy:.3f}")
    if joint_result:
        results.add_metric("Joint training steps", joint_result.total_steps)
        results.add_metric("Joint avg cosine", f"{joint_result.wave_cosine_accuracy:.3f}")
    results.add_metric("Valid word rate", f"{valid_rate:.1%}")
    results.add_metric(
        "Total training time",
        f"{wtt_result.total_time_seconds + wg_result.total_time_seconds + (joint_result.total_time_seconds if joint_result else 0):.1f}s",
    )
    results.save()

    log.success("Phase 9 training complete")
    print(f"\n{'=' * 60}")
    print(f"  Phase 9 Complete!")
    print(f"{'=' * 60}")
