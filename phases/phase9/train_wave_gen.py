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

        # Collect chunks from all documents
        print("  ℹ Stage 1: Collecting (wave, word) pairs...")
        chunk_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []

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

                if step % log_interval == 0:
                    avg_loss = sum(all_losses[-log_interval:]) / min(
                        len(all_losses), log_interval
                    )
                    print(
                        f"  WTT Step {step:>6}  "
                        f"loss={avg_loss:.4f}  "
                        f"chunks={total_chunks:,}"
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
        word_acc = self._evaluate_wtt_accuracy(texts[:50])

        elapsed = time.time() - t0
        final_loss = all_losses[-1] if all_losses else 0.0
        avg_loss = sum(all_losses) / max(len(all_losses), 1)

        print(f"\n  ✓ Stage 1 complete: {step} steps, {total_chunks:,} chunks")
        print(f"    Final loss: {final_loss:.4f}")
        print(f"    Word accuracy: {word_acc:.1%}")
        print(f"    Time: {elapsed:.1f}s")

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
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Pre-compute all frozen pipeline outputs for WG training.

        Since CSE, field, CGN, and WaveChunker are all frozen during
        Stage 2, their outputs never change. Computing them once upfront
        eliminates the CPU bottleneck that starves the GPU.

        Args:
            texts: Raw text documents
            max_samples: Maximum samples to pre-compute

        Returns:
            List of (merged_context [768], target_waves [N, 432]) tuples,
            already on self.device
        """
        precomputed = []
        skipped = 0

        print(f"  ℹ Pre-computing frozen pipeline outputs for up to {max_samples:,} samples...")
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

                    # FLUX pipeline → merged context
                    field_features, sims, locs = self.model.field.query(
                        wave_vec, k=4
                    )
                    combined = field_features.mean(dim=0)
                    cgn_out = self.model.cgn(combined)
                    merged = (combined + cgn_out)           # [768]

                    # WaveChunker → target waves
                    chunk_waves, spans = self.chunker(wave_seq)
                    target_waves = chunk_waves               # [N, 432]

                if target_waves.shape[0] < 2:
                    skipped += 1
                    continue

                # Store on CPU — GPU can't hold 5K+ samples + model
                precomputed.append((merged.cpu(), target_waves.cpu()))

                if (i + 1) % 50 == 0:
                    elapsed_so_far = time.time() - t0
                    rate = (i + 1) / max(elapsed_so_far, 0.01)
                    remaining = max_samples - len(precomputed)
                    eta = remaining / max(rate, 0.01)
                    print(
                        f"    ... {i+1:,} texts → {len(precomputed):,} valid  "
                        f"[{rate:.0f} text/s, ETA {eta:.0f}s]"
                    )

            except Exception:
                skipped += 1
                continue

        elapsed = time.time() - t0
        print(f"  ✓ Pre-computed {len(precomputed):,} samples in {elapsed:.1f}s (skipped {skipped:,})")

        return precomputed

    def precompute_wg_data(
        self,
        texts: List[str],
        max_samples: int = 10000,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Public wrapper for pre-computing WG training data.

        Call this in a separate cell, then pass the result to
        train_wave_generator(precomputed=...) to skip re-computation.
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
            print(f"  ✓ Using {len(precomputed):,} pre-computed samples (skipping pre-computation)")

        if len(precomputed) == 0:
            print("  ✗ No valid samples after pre-computation")
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
        print(f"  ✓ Gradient check: {_trainable}/{_total_p} generator params trainable")

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

        print(f"\n  ℹ Starting WG training loop: {total_steps:,} steps over {len(precomputed):,} samples")

        # ── Training loop — pure GPU, no CPU bottleneck ──
        import random
        sample_indices = list(range(len(precomputed)))
        optimizer.zero_grad()

        for step in range(1, total_steps + 1):
            # Cycle through pre-computed data with shuffling
            idx = (step - 1) % len(precomputed)
            if idx == 0 and step > 1:
                random.shuffle(sample_indices)
            sample_idx = sample_indices[idx]
            merged_cpu, target_cpu = precomputed[sample_idx]
            merged = merged_cpu.to(self.device)
            target_waves = target_cpu.to(self.device)

            # WaveGenerator forward (teacher-forced) — all on GPU
            predicted_waves, confidences = self.generator(
                merged, target_waves
            )

            # Loss: MSE + cosine distance
            mse_loss = F.mse_loss(predicted_waves, target_waves)
            cos_loss = 1.0 - F.cosine_similarity(
                predicted_waves, target_waves, dim=-1
            ).mean()
            loss = mse_loss + cos_loss

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
                    f"lr={lr_now:.6f}  "
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

        print(f"\n  ✓ Stage 2 complete: {total_steps} steps")
        print(f"    Final loss: {final_loss:.4f}")
        print(f"    Avg cosine accuracy: {avg_cos:.3f}")
        print(f"    Pre-compute time: {precompute_time:.1f}s")
        print(f"    Training time: {train_elapsed:.1f}s")
        print(f"    Total time: {elapsed:.1f}s")

        return WGStageResult(
            total_steps=total_steps,
            final_loss=final_loss,
            avg_loss=avg_loss,
            wave_cosine_accuracy=avg_cos,
            total_time_seconds=elapsed,
        )

    # ─────────────────────────────────────────────
    # Checkpoint Save
    # ─────────────────────────────────────────────

    def save_phase9_checkpoint(
        self,
        wtt_result: WTTStageResult,
        wg_result: WGStageResult,
        valid_word_rate: float = 0.0,
    ) -> Path:
        """
        Save Phase 9 checkpoint with all component states.

        Args:
            wtt_result: WaveToText pre-training result
            wg_result: WaveGenerator training result
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
                'valid_word_rate': valid_word_rate,
                'total_training_time': (
                    wtt_result.total_time_seconds + wg_result.total_time_seconds
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
    trainer.save_phase9_checkpoint(wtt_result, wg_result, valid_rate)

    # Generate results
    results = PhaseResults(phase=9, component_name="Wave-Level Generation")
    results.add_metric("WTT training steps", wtt_result.total_steps)
    results.add_metric("WTT word accuracy", f"{wtt_result.word_accuracy:.1%}")
    results.add_metric("WG training steps", wg_result.total_steps)
    results.add_metric("WG final loss", f"{wg_result.final_loss:.4f}")
    results.add_metric("WG cosine accuracy", f"{wg_result.wave_cosine_accuracy:.3f}")
    results.add_metric("Valid word rate", f"{valid_rate:.1%}")
    results.add_metric(
        "Total training time",
        f"{wtt_result.total_time_seconds + wg_result.total_time_seconds:.1f}s",
    )
    results.save()

    log.success("Phase 9 training complete")
    print(f"\n{'=' * 60}")
    print(f"  Phase 9 Complete!")
    print(f"{'=' * 60}")
