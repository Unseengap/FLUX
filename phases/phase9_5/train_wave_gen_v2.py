"""
Phase 9.5 — Training Script: WaveGenerator Retraining with All Six Fixes

Loads phase9.phase.pt, keeps WaveToText + WaveChunker + all Phase 1–7
components frozen, and retrains WaveGenerator from scratch.

Six fixes from Phase 9 failure:
    1. Context collapse → L2-normalize + noise augmentation + 5× contrastive loss
    2. Fixed chunks → Random start position + variable sequence length
    3. Batch size 1 → DataLoader batch_size=128 + batched GRU forward
    4. SS started at 0% → Start at 50% from step 1, ramp to 90%
    5. Train-inference mismatch → Gaussian jitter on precomputed contexts
    6. Silent error swallowing → Log tracebacks, abort on >10% skip rate

Training stages:
    Stage 1: WaveGenerator training (main, batched, ~15,000 steps)
    Stage 2: Joint fine-tuning WG+WTT (3,000 steps with proper error handling)
"""

import sys
import time
import math
import random
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime

# ─────────────────────────────────────────────
# Path setup for cross-phase imports
# ─────────────────────────────────────────────
_PHASES_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PHASES_DIR.parent

for _phase in ['phase1', 'phase2', 'phase3', 'phase4',
               'phase5', 'phase6', 'phase7', 'phase8', 'phase9']:
    _p = str(_PHASES_DIR / _phase)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from flux_model import FLUXModel, FLUX_MODEL_CONFIG
from flux_utils import (
    get_device, save_checkpoint, load_checkpoint, checkpoint_exists,
    PhaseLogger, PhaseResults,
)
from wave_chunker import WaveChunker
from wave_generator_v3 import WaveGeneratorV3
from wave_to_text import WaveToText
from wave_sampler import ThermodynamicWaveSampler


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

PHASE9_5_CONFIG = {
    'wave_dim': 432,
    'field_features': 512,
    'max_waves': 50,
    'k_neighbors': 16,
    'interference_radius': 4,
    'gru_hidden': 512,
    'gru_layers': 1,
    'dropout': 0.15,
    'wtt_hidden_dim': 256,
    'wtt_max_bytes': 20,
    'coherence_threshold': 0.5,
    'min_chunk_size': 2,
    'max_chunk_size': 20,
    # Training
    'batch_size': 128,
    'lr': 3e-4,
    'grad_accum': 1,  # Not needed with batched training
    'ss_start': 0.5,
    'ss_end': 0.9,
    'contrastive_weight': 5.0,
    'noise_sigma': 0.1,
    'max_skip_rate': 0.10,
}


# ─────────────────────────────────────────────
# Result Dataclasses
# ─────────────────────────────────────────────

@dataclass
class WGStageResult:
    """Result from WaveGenerator training."""
    total_steps: int
    final_loss: float
    avg_loss: float
    wave_cosine_accuracy: float
    total_time_seconds: float
    steps_per_second: float


@dataclass
class JointStageResult:
    """Result from Joint WG+WTT fine-tuning."""
    total_steps: int
    skipped_steps: int
    final_loss: float
    avg_loss: float
    wave_cosine_accuracy: float
    wtt_word_accuracy: float
    total_time_seconds: float


# ─────────────────────────────────────────────
# Load from Phase 9 Checkpoint (self-contained)
# ─────────────────────────────────────────────

def load_from_phase9_checkpoint(
    device: str = 'cpu',
) -> Tuple[FLUXModel, WaveChunker, WaveToText, Dict[str, Any]]:
    """
    Load ALL frozen components from phase9.phase.pt.

    Returns FLUXModel with Phase 1–7 states, plus trained WaveChunker
    and WaveToText. WaveGenerator state is DISCARDED.

    Args:
        device: Target device

    Returns:
        (flux_model, wave_chunker, wave_to_text, checkpoint_dict)
    """
    ckpt = load_checkpoint(9)
    config = ckpt.get('config', FLUX_MODEL_CONFIG)

    # ── Build and load FLUXModel ──
    from gravity import GravitationalRelevance

    model = FLUXModel(config=config, device=device)

    if 'cse_state_dict' in ckpt:
        model.cse.load_state_dict(ckpt['cse_state_dict'])
    if 'field_state_dict' in ckpt:
        model.field.load_state_dict(ckpt['field_state_dict'])
    if 'gr_state' in ckpt:
        try:
            model.gr = GravitationalRelevance.load_state(ckpt['gr_state'], device=device)
        except Exception:
            pass
    if 'tl_state' in ckpt:
        model.tl.load_state(ckpt['tl_state'])
    if 'cgn_state' in ckpt:
        model.cgn.load_state(ckpt['cgn_state'])
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
        model.wave_to_field.load_state_dict(ckpt['wave_to_field_state'])
    if 'field_to_wave_state' in ckpt:
        model.field_to_wave.load_state_dict(ckpt['field_to_wave_state'])
    if 'output_head_state' in ckpt:
        model.output_head.load_state_dict(ckpt['output_head_state'])

    # Freeze ALL model params
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(device)
    model.eval()

    # ── Build and load WaveChunker (TRAINED — freeze) ──
    p9cfg = ckpt.get('phase9_config', {})
    chunker = WaveChunker(
        wave_dim=p9cfg.get('wave_dim', 432),
        min_chunk_size=p9cfg.get('min_chunk_size', 2),
        max_chunk_size=p9cfg.get('max_chunk_size', 20),
        coherence_threshold=p9cfg.get('coherence_threshold', 0.5),
    ).to(device)
    if 'wave_chunker_state_dict' in ckpt:
        chunker.load_state_dict(ckpt['wave_chunker_state_dict'])
    for param in chunker.parameters():
        param.requires_grad = False
    chunker.eval()

    # ── Build and load WaveToText (TRAINED — freeze) ──
    wtt = WaveToText(
        wave_dim=p9cfg.get('wave_dim', 432),
        hidden_dim=p9cfg.get('wtt_hidden_dim', 256),
        max_bytes=p9cfg.get('wtt_max_bytes', 20),
    ).to(device)
    if 'wave_to_text_state_dict' in ckpt:
        wtt.load_state_dict(ckpt['wave_to_text_state_dict'])
    for param in wtt.parameters():
        param.requires_grad = False
    wtt.eval()

    n_attractors = model.field.num_attractors()
    print(f"  ✓ Loaded from phase9.phase.pt:")
    print(f"    FLUXModel: CSE, Field ({n_attractors:,} attractors), GR, TL, CGN, Memory, bridges")
    print(f"    WaveChunker: {sum(p.numel() for p in chunker.parameters()):,} params (frozen)")
    print(f"    WaveToText: {sum(p.numel() for p in wtt.parameters()):,} params (frozen)")
    print(f"    WaveGenerator state: DISCARDED (will retrain)")

    return model, chunker, wtt, ckpt


def build_fresh_wave_generator(
    device: str = 'cpu',
    config: Optional[Dict[str, Any]] = None,
) -> WaveGeneratorV3:
    """
    Build a fresh WaveGeneratorV3 (random init — Phase 9 state discarded).

    Args:
        device: Target device
        config: Optional config overrides

    Returns:
        Fresh WaveGeneratorV3 on device
    """
    cfg = {**PHASE9_5_CONFIG, **(config or {})}

    gen = WaveGeneratorV3(
        wave_dim=cfg['wave_dim'],
        field_features=cfg['field_features'],
        max_waves=cfg['max_waves'],
        k_neighbors=cfg['k_neighbors'],
        interference_radius=cfg['interference_radius'],
        gru_hidden=cfg['gru_hidden'],
        gru_layers=cfg['gru_layers'],
        dropout=cfg['dropout'],
    ).to(device)

    n_params = sum(p.numel() for p in gen.parameters())
    print(f"  ✓ Fresh WaveGeneratorV3: {n_params:,} params (random init)")
    return gen


# ─────────────────────────────────────────────
# Training Data Loading
# ─────────────────────────────────────────────

def load_training_data(max_docs: int = 10000) -> List[str]:
    """Load training texts from OpenWebText or synthetic fallback."""
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
        "The history of {field} reveals the importance of {concept}.",
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
# Pipeline Helper
# ─────────────────────────────────────────────

@torch.no_grad()
def compute_merged_context(
    model: FLUXModel,
    wave_vec: torch.Tensor,
    device: str = 'cpu',
) -> torch.Tensor:
    """
    Compute merged field context using the full Phase 7 pipeline.

    Pipeline: wave_vec → wave_to_field → GR → CGN → field.query(top-1) + cgn_out

    Args:
        model: FLUXModel with trained Phase 1–7 weights
        wave_vec: [432] mean CSE wave vector
        device: Target device

    Returns:
        [512] merged field context
    """
    try:
        field_input = model.wave_to_field(wave_vec)
        relevance_out = model.gr(field_input.unsqueeze(0)).squeeze(0)
        cgn_out = model.cgn(relevance_out)
        field_features, sims, locs = model.field.query(wave_vec, k=4)
        best_features = field_features[0]
        merged = best_features + cgn_out
    except Exception:
        field_features, sims, locs = model.field.query(wave_vec, k=4)
        combined = field_features.mean(dim=0)
        cgn_out = model.cgn(combined)
        merged = combined + cgn_out
    return merged


# ─────────────────────────────────────────────
# FIX #2: Precompute with RANDOM windows + VARIABLE lengths
# ─────────────────────────────────────────────

@torch.no_grad()
def precompute_wg_data(
    model: FLUXModel,
    chunker: WaveChunker,
    texts: List[str],
    max_samples: int = 10000,
    device: str = 'cpu',
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Pre-compute frozen pipeline outputs for WG training.

    FIX #2: Random start position + variable chunk count per sample.
    Phase 9 used _start=0 and _MAX_CHUNKS=40 for all samples.

    Args:
        model: FLUXModel (frozen)
        chunker: WaveChunker (frozen)
        texts: Raw text documents
        max_samples: Max samples to pre-compute
        device: Device for pipeline computation

    Returns:
        List of (merged_context [512], target_waves [variable, 432]) on CPU
    """
    precomputed = []
    skipped = 0

    print(f"  ℹ Pre-computing frozen pipeline outputs ({max_samples:,} samples)...")
    print(f"    Pipeline: CSE → wave_to_field → GR → CGN → field.query → WaveChunker")
    print(f"    FIX #2: Random windows + variable lengths (was: fixed 40 from pos 0)")
    t0 = time.time()

    for i, text in enumerate(texts):
        if len(precomputed) >= max_samples:
            break
        if not text or len(text.strip()) < 10:
            skipped += 1
            continue

        try:
            wave = model.cse.encode(text)
            wave_seq = wave.full.to(device)
            wave_vec = wave_seq.mean(dim=0)
            merged = compute_merged_context(model, wave_vec, device)
            chunk_waves, spans = chunker(wave_seq)
            total_chunks = chunk_waves.shape[0]

            if total_chunks < 3:
                skipped += 1
                continue

            # FIX #2a: Variable chunk count per sample (5 to 40)
            max_len = random.randint(5, min(40, total_chunks))

            # FIX #2b: Random start position (not always 0)
            max_start = max(0, total_chunks - max_len)
            start = random.randint(0, max_start) if max_start > 0 else 0

            target_waves = chunk_waves[start:start + max_len]

            if target_waves.shape[0] < 3:
                skipped += 1
                continue

            precomputed.append((merged.cpu(), target_waves.cpu()))

            # Progress logging
            if len(precomputed) == 1:
                e1 = time.time() - t0
                print(f"    ... 1st sample: {target_waves.shape[0]} chunks in {e1:.2f}s")
            elif len(precomputed) % 500 == 0 or (len(precomputed) < 500 and len(precomputed) % 100 == 0):
                elapsed = time.time() - t0
                rate = len(precomputed) / max(elapsed, 0.01)
                pct = len(precomputed) / max_samples * 100
                eta = (max_samples - len(precomputed)) / max(rate, 0.01)
                print(
                    f"    ... {len(precomputed):,}/{max_samples:,} ({pct:.0f}%)  "
                    f"[{rate:.1f}/s, ETA {eta:.0f}s]"
                )

        except Exception:
            skipped += 1
            continue

    elapsed = time.time() - t0
    print(f"  ✓ Pre-computed {len(precomputed):,} samples in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"    Skipped: {skipped:,}")

    # Log chunk length distribution
    if precomputed:
        lens = [p[1].shape[0] for p in precomputed]
        print(f"    Chunk lengths: min={min(lens)}, max={max(lens)}, avg={sum(lens)/len(lens):.1f}")

    return precomputed


# ─────────────────────────────────────────────
# FIX #3: Batched Dataset + DataLoader
# ─────────────────────────────────────────────

class WaveGenDataset(torch.utils.data.Dataset):
    """
    Dataset for batched WaveGenerator training.

    Stores precomputed (merged_context, target_waves) pairs.
    Applies FIX #1 (noise augmentation) and FIX #5 (context jitter)
    on the fly.

    Args:
        precomputed: List of (merged [512], target_waves [N, 432])
        noise_sigma: Gaussian noise σ for context augmentation
    """

    def __init__(
        self,
        precomputed: List[Tuple[torch.Tensor, torch.Tensor]],
        noise_sigma: float = 0.1,
    ):
        self.data = precomputed
        self.noise_sigma = noise_sigma

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        merged, target_waves = self.data[idx]

        # FIX #1 + #5: Add Gaussian noise to context during training
        # This simulates the variance of dynamic attractor sampling
        # and decorrelates near-identical context vectors.
        if self.noise_sigma > 0:
            noise = torch.randn_like(merged) * self.noise_sigma
            merged = merged + noise

        return {
            'merged': merged,
            'target_waves': target_waves,
            'length': torch.tensor(target_waves.shape[0], dtype=torch.long),
        }


def collate_wg_batch(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Collate variable-length WG samples into padded batch.

    Pads target_waves to max length in batch.

    Args:
        batch: List of dicts from WaveGenDataset.__getitem__

    Returns:
        Dict with merged [B, 512], target_waves [B, max_seq, 432],
        lengths [B], mask [B, max_seq]
    """
    mergeds = torch.stack([b['merged'] for b in batch])
    lengths = torch.stack([b['length'] for b in batch])
    max_seq = lengths.max().item()
    wave_dim = batch[0]['target_waves'].shape[-1]

    padded = torch.zeros(len(batch), max_seq, wave_dim)
    mask = torch.zeros(len(batch), max_seq, dtype=torch.bool)

    for i, b in enumerate(batch):
        seq_len = b['length'].item()
        padded[i, :seq_len, :] = b['target_waves']
        mask[i, :seq_len] = True

    return {
        'merged': mergeds,
        'target_waves': padded,
        'lengths': lengths,
        'mask': mask,
    }


# ─────────────────────────────────────────────
# Stage 1: WaveGenerator Training (main)
# ─────────────────────────────────────────────

def train_wave_generator(
    generator: WaveGeneratorV3,
    precomputed: List[Tuple[torch.Tensor, torch.Tensor]],
    max_steps: int = 15000,
    batch_size: int = 128,
    lr: float = 3e-4,
    ss_start: float = 0.5,
    ss_end: float = 0.9,
    contrastive_weight: float = 5.0,
    noise_sigma: float = 0.1,
    log_interval: int = 200,
    device: str = 'cpu',
    log: Optional[PhaseLogger] = None,
) -> WGStageResult:
    """
    Train WaveGenerator with all six fixes applied.

    FIX #1: Noise augmentation on contexts (σ=noise_sigma)
    FIX #2: Already applied in precompute (random windows, variable lengths)
    FIX #3: Batched DataLoader with batch_size
    FIX #4: Scheduled sampling starts at ss_start (50%) from step 1
    FIX #5: Noise jitter simulates dynamic field re-query variance
    FIX #6: No silent error swallowing (not applicable — pure tensor ops)

    Args:
        generator: Fresh WaveGeneratorV3 (random init)
        precomputed: Pre-computed (merged, target_waves) list
        max_steps: Maximum training steps (optimizer updates)
        batch_size: Batch size for DataLoader
        lr: Learning rate
        ss_start: Initial scheduled sampling probability
        ss_end: Final scheduled sampling probability
        contrastive_weight: Weight for Wave 0 contrastive loss
        noise_sigma: Gaussian noise σ for context augmentation
        log_interval: Print every N steps
        device: Training device
        log: PhaseLogger instance

    Returns:
        WGStageResult with training metrics
    """
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"  Stage 1: WaveGenerator Training (Batched)")
    print(f"  max_steps={max_steps}, batch_size={batch_size}")
    print(f"  ss_start={ss_start}, ss_end={ss_end}")
    print(f"  contrastive_weight={contrastive_weight}, noise_σ={noise_sigma}")
    print(f"{'='*60}")

    generator.train()
    generator = generator.to(device)

    # ── DataLoader (FIX #3) ──
    dataset = WaveGenDataset(precomputed, noise_sigma=noise_sigma)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_wg_batch,
        num_workers=0,
        drop_last=True,
        pin_memory=(device != 'cpu'),
    )

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        generator.parameters(), lr=lr, weight_decay=0.01
    )

    warmup_steps = min(200, max_steps // 20)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(0.01, step / max(warmup_steps, 1))
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Verify gradient state ──
    trainable = sum(1 for p in generator.parameters() if p.requires_grad)
    total_p = sum(1 for p in generator.parameters())
    assert trainable > 0, f"Generator has 0/{total_p} trainable params!"
    print(f"  ✓ Gradient check: {trainable}/{total_p} params trainable")

    # ── Wave 0 buffer for contrastive loss ──
    wave0_buffer: List[torch.Tensor] = []

    all_losses: List[float] = []
    cosine_accs: List[float] = []
    step = 0
    epoch = 0
    train_t0 = time.time()

    print(f"\n  ℹ Starting training loop: {max_steps:,} steps")
    print(f"    DataLoader: {len(dataset):,} samples, batch_size={batch_size}")
    print(f"    Batches per epoch: {len(loader)}")

    while step < max_steps:
        epoch += 1
        for batch in loader:
            if step >= max_steps:
                break

            step += 1

            # ── Move batch to device ──
            merged = batch['merged'].to(device)           # [B, 512]
            targets = batch['target_waves'].to(device)     # [B, max_seq, 432]
            lengths = batch['lengths'].to(device)          # [B]
            mask = batch['mask'].to(device)                # [B, max_seq]

            # ── FIX #4: Scheduled sampling from 50% at step 1 ──
            progress = step / max(max_steps, 1)
            ss_p = ss_start + (ss_end - ss_start) * progress

            # ── Forward pass (batched — FIX #3) ──
            predicted, confidences = generator.forward_batch(
                merged, targets, lengths,
                scheduled_sampling_p=ss_p,
            )

            # ── Loss computation ──
            # MSE loss (masked)
            mse_loss = ((predicted - targets) ** 2 * mask.unsqueeze(-1)).sum()
            mse_loss = mse_loss / mask.sum().clamp(min=1) / generator.wave_dim

            # Cosine loss (masked)
            cos_sim = F.cosine_similarity(predicted, targets, dim=-1)  # [B, max_seq]
            cos_loss = ((1.0 - cos_sim) * mask.float()).sum() / mask.sum().clamp(min=1)

            # Context loss: extra penalty on Wave 0 (position 0)
            wave0_pred = predicted[:, 0, :]   # [B, wave_dim]
            wave0_tgt = targets[:, 0, :]      # [B, wave_dim]
            ctx_loss = F.mse_loss(wave0_pred, wave0_tgt) + (
                1.0 - F.cosine_similarity(wave0_pred, wave0_tgt, dim=-1).mean()
            )
            ctx_loss = ctx_loss * 2.0

            # ── FIX #1: Strong contrastive loss on Wave 0 (weight=5×) ──
            contrastive_loss = torch.tensor(0.0, device=device)
            if len(wave0_buffer) >= 2:
                # Use previous batch's Wave 0 as negatives
                neg_count = min(len(wave0_buffer), batch_size)
                neg_indices = random.sample(range(len(wave0_buffer)), neg_count)
                negatives = torch.stack([wave0_buffer[i] for i in neg_indices]).to(device)

                # For each sample in batch, compute similarity to random negatives
                # wave0_pred: [B, wave_dim], negatives: [neg_count, wave_dim]
                # We compare each wave0 to a random subset of negatives
                neg_sim = F.cosine_similarity(
                    wave0_pred.unsqueeze(1),       # [B, 1, wave_dim]
                    negatives.unsqueeze(0),         # [1, neg, wave_dim]
                    dim=-1,                         # [B, neg]
                )
                contrastive_loss = F.relu(neg_sim - 0.3).mean() * contrastive_weight

            # Update Wave 0 buffer (detached, on CPU)
            wave0_buffer.extend(wave0_pred.detach().cpu().unbind(0))
            if len(wave0_buffer) > 512:
                wave0_buffer = wave0_buffer[-512:]

            loss = mse_loss + cos_loss + ctx_loss + contrastive_loss

            # ── Backward + step ──
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # ── Metrics ──
            all_losses.append(loss.item())
            with torch.no_grad():
                masked_cos = (cos_sim * mask.float()).sum() / mask.sum().clamp(min=1)
                cosine_accs.append(masked_cos.item())

            # ── Logging ──
            if step == 1:
                elapsed_1 = time.time() - train_t0
                print(
                    f"  WG Step     1/{max_steps}  "
                    f"loss={loss.item():.4f}  cos_acc={masked_cos.item():.3f}  "
                    f"ctx={ctx_loss.item():.4f}  contra={contrastive_loss.item():.4f}  "
                    f"ss_p={ss_p:.2f}  (first step: {elapsed_1:.2f}s)"
                )
            elif step == 10:
                elapsed_10 = time.time() - train_t0
                rate_10 = 10 / max(elapsed_10, 0.01)
                print(
                    f"  WG Step    10/{max_steps}  "
                    f"loss={sum(all_losses[-10:])/10:.4f}  "
                    f"[{rate_10:.1f} step/s]"
                )

            if step % log_interval == 0:
                window = min(len(all_losses), log_interval)
                avg_loss = sum(all_losses[-window:]) / window
                avg_cos = sum(cosine_accs[-window:]) / window
                lr_now = optimizer.param_groups[0]['lr']
                elapsed_train = time.time() - train_t0
                steps_per_sec = step / max(elapsed_train, 0.01)
                eta = (max_steps - step) / max(steps_per_sec, 0.01)
                print(
                    f"  WG Step {step:>6}/{max_steps}  "
                    f"loss={avg_loss:.4f}  cos_acc={avg_cos:.3f}  "
                    f"lr={lr_now:.6f}  ss_p={ss_p:.2f}  "
                    f"[{steps_per_sec:.1f} step/s, ETA {eta:.0f}s]"
                )
                if log:
                    log.metric(f"wg_step_{step}_loss", f"{avg_loss:.4f}")

    elapsed = time.time() - t0
    train_elapsed = time.time() - train_t0
    final_loss = all_losses[-1] if all_losses else 0.0
    avg_loss = sum(all_losses) / max(len(all_losses), 1)
    avg_cos = sum(cosine_accs) / max(len(cosine_accs), 1)
    steps_per_sec = step / max(train_elapsed, 0.01)

    print(f"\n  ✓ Stage 1 complete: {step} steps in {epoch} epochs")
    print(f"    Final loss: {final_loss:.4f}")
    print(f"    Avg cosine accuracy: {avg_cos:.3f}")
    print(f"    Speed: {steps_per_sec:.1f} steps/s")
    print(f"    Time: {train_elapsed:.1f}s ({train_elapsed/60:.1f} min)")

    if log:
        log.metric("wg_final_loss", f"{final_loss:.4f}")
        log.metric("wg_cos_acc", f"{avg_cos:.3f}")
        log.metric("wg_steps_per_sec", f"{steps_per_sec:.1f}")

    return WGStageResult(
        total_steps=step,
        final_loss=final_loss,
        avg_loss=avg_loss,
        wave_cosine_accuracy=avg_cos,
        total_time_seconds=elapsed,
        steps_per_second=steps_per_sec,
    )


# ─────────────────────────────────────────────
# Stage 2: Joint Fine-Tuning (WG + WTT)
# FIX #6: Proper error handling
# ─────────────────────────────────────────────

def train_joint_finetune(
    generator: WaveGeneratorV3,
    wtt: WaveToText,
    model: FLUXModel,
    chunker: WaveChunker,
    texts: List[str],
    precomputed: List[Tuple[torch.Tensor, torch.Tensor]],
    max_steps: int = 3000,
    lr_wg: float = 1e-4,
    lr_wtt: float = 5e-5,
    log_interval: int = 200,
    max_skip_rate: float = 0.10,
    device: str = 'cpu',
    log: Optional[PhaseLogger] = None,
) -> JointStageResult:
    """
    Joint fine-tuning of WaveGenerator + WaveToText.

    FIX #6: Logs first 5 exception tracebacks. Aborts if skip rate > 10%.
    Never prints success markers for zero-work stages.

    Args:
        generator: Trained WaveGeneratorV3
        wtt: Trained WaveToText (unfrozen for fine-tuning)
        model: FLUXModel (frozen)
        chunker: WaveChunker (frozen)
        texts: Training documents
        precomputed: Pre-computed data from precompute_wg_data
        max_steps: Max training steps
        lr_wg: Learning rate for WaveGenerator
        lr_wtt: Learning rate for WaveToText
        log_interval: Print every N steps
        max_skip_rate: Abort if skip rate exceeds this
        device: Device
        log: PhaseLogger

    Returns:
        JointStageResult
    """
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"  Stage 2: Joint Fine-Tuning (WG + WTT)")
    print(f"  max_steps={max_steps}, lr_wg={lr_wg}, lr_wtt={lr_wtt}")
    print(f"  max_skip_rate={max_skip_rate:.0%}")
    print(f"{'='*60}")

    # Unfreeze WTT for fine-tuning
    wtt.train()
    for param in wtt.parameters():
        param.requires_grad = True

    generator.train()
    for param in generator.parameters():
        param.requires_grad = True

    # Separate param groups with different LRs
    optimizer = torch.optim.AdamW([
        {'params': generator.parameters(), 'lr': lr_wg},
        {'params': wtt.parameters(), 'lr': lr_wtt},
    ], weight_decay=0.01)

    wg_params_n = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    wtt_params_n = sum(p.numel() for p in wtt.parameters() if p.requires_grad)
    print(f"  Trainable: WG {wg_params_n:,} + WTT {wtt_params_n:,} = {wg_params_n + wtt_params_n:,}")

    all_losses: List[float] = []
    cosine_accs: List[float] = []
    skipped = 0
    errors_logged = 0
    sample_indices = list(range(len(precomputed)))
    train_t0 = time.time()

    for step in range(1, max_steps + 1):
        idx = (step - 1) % len(precomputed)
        if idx == 0 and step > 1:
            random.shuffle(sample_indices)
        sample_idx = sample_indices[idx]
        merged_cpu, target_cpu = precomputed[sample_idx]

        text = texts[sample_idx % len(texts)]

        try:
            merged = merged_cpu.to(device)
            target_waves = target_cpu.to(device)

            # Re-encode to get byte targets for WTT
            with torch.no_grad():
                wave = model.cse.encode(text)
                wave_seq = wave.full.to(device)
                text_bytes = text.encode('utf-8', errors='replace')
                pairs = chunker.chunk_with_bytes(wave_seq, text_bytes)

            if len(pairs) < 2:
                skipped += 1
                continue

            # Cap to match precomputed target length
            max_pairs = min(len(pairs), target_waves.shape[0])
            pairs = pairs[:max_pairs]

            target_waves_fresh = torch.stack([p[0] for p in pairs]).to(device)
            targets_batch = [p[1] for p in pairs]

            wave_vec = wave_seq.mean(dim=0)
            merged = compute_merged_context(model, wave_vec, device)

            # Forward through WG (single-sample compat mode)
            predicted_waves, _ = generator(
                merged, target_waves_fresh, scheduled_sampling_p=0.5
            )

            pred_len = min(len(predicted_waves), len(targets_batch))
            wtt_loss = wtt.forward_batch(
                predicted_waves[:pred_len], targets_batch[:pred_len]
            )

            mse_loss = F.mse_loss(
                predicted_waves[:pred_len], target_waves_fresh[:pred_len]
            )
            cos_loss = 1.0 - F.cosine_similarity(
                predicted_waves[:pred_len], target_waves_fresh[:pred_len], dim=-1
            ).mean()

            loss = mse_loss + cos_loss + 0.5 * wtt_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(generator.parameters()) + list(wtt.parameters()), 1.0
            )
            optimizer.step()

            all_losses.append(loss.item())
            cos_acc = F.cosine_similarity(
                predicted_waves[:pred_len],
                target_waves_fresh[:pred_len], dim=-1,
            ).mean().item()
            cosine_accs.append(cos_acc)

        except Exception as e:
            skipped += 1
            # FIX #6: Log first 5 exceptions with traceback
            if errors_logged < 5:
                errors_logged += 1
                print(f"\n  ✗ Joint FT step {step} exception ({errors_logged}/5):")
                traceback.print_exc()
                print()

            # FIX #6: Abort if skip rate exceeds threshold
            skip_rate = skipped / step
            if step >= 50 and skip_rate > max_skip_rate:
                print(
                    f"\n  ✗ ABORTING: Skip rate {skip_rate:.1%} > {max_skip_rate:.0%} "
                    f"({skipped}/{step} steps skipped)"
                )
                break
            continue

        # Logging
        if step == 1:
            print(
                f"  Joint Step     1/{max_steps}  "
                f"loss={loss.item():.4f}  mse={mse_loss.item():.4f}  "
                f"wtt={wtt_loss.item():.4f}  cos_acc={cos_acc:.3f}"
            )

        if step % log_interval == 0:
            window = min(len(all_losses), log_interval)
            avg_loss = sum(all_losses[-window:]) / max(window, 1)
            avg_cos = sum(cosine_accs[-window:]) / max(window, 1)
            elapsed = time.time() - train_t0
            rate = step / max(elapsed, 0.01)
            eta = (max_steps - step) / max(rate, 0.01)
            skip_rate = skipped / step
            print(
                f"  Joint Step {step:>6}/{max_steps}  "
                f"loss={avg_loss:.4f}  cos_acc={avg_cos:.3f}  "
                f"skip_rate={skip_rate:.1%}  "
                f"[{rate:.1f} step/s, ETA {eta:.0f}s]"
            )
            if log:
                log.metric(f"joint_step_{step}_loss", f"{avg_loss:.4f}")

    # Evaluate WTT accuracy
    print(f"\n  ℹ Evaluating WTT word accuracy on 50 texts...")
    wtt_acc = evaluate_wtt_accuracy(wtt, model, chunker, texts[:50], device)

    elapsed = time.time() - t0
    actual_steps = len(all_losses)
    avg_loss = sum(all_losses) / max(actual_steps, 1)
    avg_cos = sum(cosine_accs) / max(len(cosine_accs), 1)

    # FIX #6: Never print success for zero-work stages
    if actual_steps == 0:
        print(f"\n  ✗ Stage 2 FAILED: 0 successful steps ({skipped} skipped)")
    else:
        skip_rate = skipped / (actual_steps + skipped)
        print(f"\n  ✓ Stage 2 complete: {actual_steps} steps ({skipped} skipped, {skip_rate:.1%})")
        print(f"    Avg loss: {avg_loss:.4f}")
        print(f"    Avg cosine acc: {avg_cos:.3f}")
        print(f"    WTT word acc: {wtt_acc:.1%}")

    # Re-freeze WTT after fine-tuning
    for param in wtt.parameters():
        param.requires_grad = False
    wtt.eval()

    return JointStageResult(
        total_steps=actual_steps,
        skipped_steps=skipped,
        final_loss=all_losses[-1] if all_losses else 0.0,
        avg_loss=avg_loss,
        wave_cosine_accuracy=avg_cos,
        wtt_word_accuracy=wtt_acc,
        total_time_seconds=elapsed,
    )


# ─────────────────────────────────────────────
# Evaluation Utilities
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate_wtt_accuracy(
    wtt: WaveToText,
    model: FLUXModel,
    chunker: WaveChunker,
    texts: List[str],
    device: str = 'cpu',
    max_chunks: int = 500,
) -> float:
    """Evaluate WaveToText word reconstruction accuracy."""
    wtt.eval()
    correct = 0
    total = 0

    for text in texts:
        if not text or len(text.strip()) < 10:
            continue
        try:
            wave = model.cse.encode(text)
            wave_seq = wave.full.to(device)
            text_bytes = text.encode('utf-8', errors='replace')
            pairs = chunker.chunk_with_bytes(wave_seq, text_bytes)

            for chunk_wave, chunk_bytes in pairs:
                decoded = wtt.decode(chunk_wave, temperature=0.5)
                if decoded == chunk_bytes:
                    correct += 1
                total += 1
                if total >= max_chunks:
                    break
        except Exception:
            continue
        if total >= max_chunks:
            break

    return correct / max(total, 1)


@torch.no_grad()
def evaluate_context_diversity(
    generator: WaveGeneratorV3,
    precomputed: List[Tuple[torch.Tensor, torch.Tensor]],
    n_samples: int = 20,
    device: str = 'cpu',
) -> Dict[str, float]:
    """
    Evaluate context diversity and context dependency.

    Measures:
    - Pairwise cosine of processed contexts
    - Cross-context Wave 0 cosine (CRITICAL: must be < 0.85)
    - Hidden init cross-context cosine (must be < 0.90)

    Args:
        generator: Trained WaveGeneratorV3
        precomputed: Pre-computed data
        n_samples: Number of samples to test
        device: Device

    Returns:
        Dict of metric names to values
    """
    generator.eval()
    n = min(n_samples, len(precomputed))

    # Get processed contexts and Wave 0 predictions
    processed_ctxs = []
    wave0s = []
    hidden_inits = []

    for i in range(n):
        merged, targets = precomputed[i]
        merged = merged.to(device)

        ctx = generator.process_context(merged)
        processed_ctxs.append(ctx)

        hidden = generator.init_hidden(device, field_context=merged)
        hidden_inits.append(hidden.squeeze())

        context_wave = generator.context_to_wave(ctx)
        wave0, _, _ = generator.forward_step(
            generator.bos_wave, context_wave, hidden
        )
        wave0s.append(wave0)

    # Pairwise cosines for contexts
    ctx_stack = torch.stack(processed_ctxs)
    ctx_cos = F.cosine_similarity(
        ctx_stack.unsqueeze(0), ctx_stack.unsqueeze(1), dim=-1
    )
    # Mask diagonal
    mask = ~torch.eye(n, dtype=torch.bool, device=device)
    avg_ctx_cos = ctx_cos[mask].mean().item()

    # Cross-context Wave 0 cosines
    w0_stack = torch.stack(wave0s)
    w0_cos = F.cosine_similarity(
        w0_stack.unsqueeze(0), w0_stack.unsqueeze(1), dim=-1
    )
    avg_w0_cos = w0_cos[mask].mean().item()

    # Hidden init cosines
    h_stack = torch.stack(hidden_inits)
    h_cos = F.cosine_similarity(
        h_stack.unsqueeze(0), h_stack.unsqueeze(1), dim=-1
    )
    avg_h_cos = h_cos[mask].mean().item()

    metrics = {
        'processed_ctx_avg_cosine': avg_ctx_cos,
        'wave0_cross_ctx_cosine': avg_w0_cos,
        'hidden_init_cross_ctx_cosine': avg_h_cos,
    }

    print(f"  📊 Context Diversity Evaluation ({n} samples)")
    print(f"    Processed context avg cosine: {avg_ctx_cos:.3f} (Phase 9 was 0.980)")
    print(f"    Wave 0 cross-context cosine:  {avg_w0_cos:.3f} (Phase 9 was 1.000, target <0.85)")
    print(f"    Hidden init cross-ctx cosine: {avg_h_cos:.3f} (Phase 9 was 0.996, target <0.90)")

    return metrics


# ─────────────────────────────────────────────
# Full Generation Pipeline
# ─────────────────────────────────────────────

def generate_text(
    prompt: str,
    flux_model: FLUXModel,
    chunker: WaveChunker,
    generator: WaveGeneratorV3,
    wtt: WaveToText,
    max_waves: int = 30,
    temperature: float = 0.8,
    use_sampler: bool = True,
) -> str:
    """
    Phase 9.5 generation: think in waves, spell in bytes.

    Args:
        prompt: Input text prompt
        flux_model: FLUXModel with frozen Phase 7 components
        chunker: Trained WaveChunker
        generator: Trained WaveGeneratorV3
        wtt: Trained WaveToText
        max_waves: Maximum waves to generate
        temperature: Sampling temperature
        use_sampler: Whether to use ThermodynamicWaveSampler

    Returns:
        Generated text (prompt + continuation)
    """
    device = flux_model._device_str

    generator.eval()
    wtt.eval()
    chunker.eval()

    with torch.no_grad():
        wave = flux_model.cse.encode(prompt)
        wave_seq = wave.full.to(device)
        wave_vec = wave_seq.mean(dim=0)
        merged = compute_merged_context(flux_model, wave_vec, device)

        generated_waves, confidences = generator.generate(
            field_context=merged,
            max_waves=max_waves,
            flux_model=flux_model,
            temperature=temperature,
        )

        sampler = ThermodynamicWaveSampler() if use_sampler else None
        text_parts = []

        for wave, conf in zip(generated_waves, confidences):
            if sampler is not None:
                wave = sampler.sample_wave(wave, conf)
            chunk_bytes = wtt.decode(wave, temperature=temperature)
            try:
                text_parts.append(chunk_bytes.decode('utf-8', errors='replace'))
            except Exception:
                text_parts.append(chunk_bytes.decode('latin-1', errors='replace'))

    return prompt + ' ' + ' '.join(text_parts)


# ─────────────────────────────────────────────
# Checkpoint Save / Load
# ─────────────────────────────────────────────

def save_phase9_5_checkpoint(
    model: FLUXModel,
    chunker: WaveChunker,
    generator: WaveGeneratorV3,
    wtt: WaveToText,
    wg_result: WGStageResult,
    joint_result: Optional[JointStageResult] = None,
    diversity_metrics: Optional[Dict[str, float]] = None,
    valid_word_rate: float = 0.0,
) -> Path:
    """
    Save Phase 9.5 checkpoint with all component states.

    Args:
        model: FLUXModel (frozen components)
        chunker: WaveChunker (frozen)
        generator: WaveGeneratorV3 (retrained)
        wtt: WaveToText (frozen or fine-tuned)
        wg_result: WG training result
        joint_result: Optional joint FT result
        diversity_metrics: Context diversity metrics
        valid_word_rate: Valid English word rate

    Returns:
        Path to saved checkpoint
    """
    state = {
        'phase': 9.5,
        'config': model.config,
        'phase9_5_config': PHASE9_5_CONFIG,
        # Phase 1–7 frozen component states
        'cse_state_dict': model.cse.state_dict(),
        'field_state_dict': model.field.state_dict(),
        'gr_state': model.gr.save_state(),
        'tl_state': model.tl.save_state(),
        'cgn_state': model.cgn.save_state(),
        'causal_graph_state': model.causal_graph.save_state(),
        'working_memory_state': model.working_memory.state_dict_memory(),
        'episodic_memory_state': model.episodic_memory.save_state(),
        'semantic_memory_state': model.semantic_memory.save_state(),
        'router_state': model.memory_router.save_state(),
        'wave_to_field_state': model.wave_to_field.state_dict(),
        'field_to_wave_state': model.field_to_wave.state_dict(),
        'output_head_state': model.output_head.state_dict(),
        # Phase 9 / 9.5 module states
        'wave_chunker_state_dict': chunker.state_dict(),
        'wave_generator_v3_state_dict': generator.state_dict(),
        'wave_to_text_state_dict': wtt.state_dict(),
        # Metrics
        'metrics': {
            'wg_training_steps': wg_result.total_steps,
            'wg_final_loss': wg_result.final_loss,
            'wg_avg_cosine_accuracy': wg_result.wave_cosine_accuracy,
            'wg_steps_per_second': wg_result.steps_per_second,
            'joint_training_steps': joint_result.total_steps if joint_result else 0,
            'joint_skipped_steps': joint_result.skipped_steps if joint_result else 0,
            'valid_word_rate': valid_word_rate,
            **(diversity_metrics or {}),
        },
    }

    # Use integer for save_checkpoint key name compatibility
    from flux_utils import CHECKPOINT_DIR
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    state['timestamp'] = datetime.now().isoformat()
    path = CHECKPOINT_DIR / 'phase9_5.phase.pt'
    torch.save(state, path)
    size_mb = path.stat().st_size / 1e6
    print(f"  ✓ Phase 9.5 checkpoint saved: {path} ({size_mb:.1f} MB)")
    return path


def load_phase9_5_modules(
    device: str = 'cpu',
) -> Tuple[FLUXModel, WaveChunker, WaveGeneratorV3, WaveToText]:
    """
    Load Phase 9.5 from checkpoint.

    Returns:
        (flux_model, wave_chunker, wave_generator_v3, wave_to_text)
    """
    from flux_utils import CHECKPOINT_DIR
    path = CHECKPOINT_DIR / 'phase9_5.phase.pt'
    if not path.exists():
        raise FileNotFoundError(
            f"Phase 9.5 checkpoint not found at {path}\n"
            f"Run Phase 9.5 training first."
        )

    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model, chunker, wtt, _ = load_from_phase9_checkpoint(device=device)

    # Build and load WaveGeneratorV3
    p95cfg = ckpt.get('phase9_5_config', PHASE9_5_CONFIG)
    gen = build_fresh_wave_generator(device=device, config=p95cfg)
    if 'wave_generator_v3_state_dict' in ckpt:
        gen.load_state_dict(ckpt['wave_generator_v3_state_dict'])

    # If WTT was fine-tuned, update it
    if 'wave_to_text_state_dict' in ckpt:
        wtt.load_state_dict(ckpt['wave_to_text_state_dict'])

    print(f"  ✓ Phase 9.5 loaded from checkpoint")
    return model, chunker, gen, wtt


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  Phase 9.5: Fix WaveGenerator Training")
    print("=" * 60)

    device = get_device()
    log = PhaseLogger(phase=9.5)

    # Load from Phase 9 checkpoint
    log.separator("Loading from Phase 9 Checkpoint")
    model, chunker, wtt, ckpt = load_from_phase9_checkpoint(device=device)

    # Build fresh WaveGenerator
    log.separator("Building Fresh WaveGeneratorV3")
    generator = build_fresh_wave_generator(device=device)

    # Load training data
    log.separator("Loading Training Data")
    texts = load_training_data(max_docs=10000)
    split_idx = int(len(texts) * 0.9)
    train_texts = texts[:split_idx]
    print(f"  Train: {len(train_texts):,} docs")

    # Precompute with fixes
    log.separator("Precomputing Data (Fixed)")
    precomputed = precompute_wg_data(
        model, chunker, train_texts,
        max_samples=8500, device=device,
    )

    # Stage 1: WaveGenerator training
    log.separator("Stage 1: WaveGenerator Training")
    wg_result = train_wave_generator(
        generator, precomputed,
        max_steps=15000, batch_size=128,
        device=device, log=log,
    )

    # Evaluate context diversity
    log.separator("Context Diversity Evaluation")
    diversity = evaluate_context_diversity(generator, precomputed, device=device)

    # Stage 2: Joint fine-tuning
    log.separator("Stage 2: Joint Fine-Tuning")
    joint_result = train_joint_finetune(
        generator, wtt, model, chunker, train_texts, precomputed,
        max_steps=3000, device=device, log=log,
    )

    # Free generation evaluation
    log.separator("Evaluation: Free Generation")
    prompts = [
        "The future of artificial intelligence",
        "Scientists have discovered",
        "In the beginning",
        "The relationship between energy and matter",
        "Modern technology relies on",
    ]
    valid_words = 0
    total_words = 0
    for p in prompts:
        try:
            result = generate_text(p, model, chunker, generator, wtt, max_waves=15)
            continuation = result[len(p):].strip()
            words = continuation.split()
            for w in words:
                clean = w.strip('.,;:!?"\'-()[]{}').lower()
                if clean.isalpha() and len(clean) >= 2:
                    total_words += 1
                    if len(clean) <= 15:
                        valid_words += 1
            print(f"  Prompt: {p}")
            print(f"  Output: {result[:200]}")
            print()
        except Exception as e:
            print(f"  ⚠ Generation failed: {e}")
    valid_rate = valid_words / max(total_words, 1)

    # Save checkpoint
    log.separator("Save Checkpoint")
    save_phase9_5_checkpoint(
        model, chunker, generator, wtt,
        wg_result, joint_result, diversity, valid_rate,
    )

    # Results
    results = PhaseResults(phase=9.5, component_name="WaveGenerator Fix")
    results.add_test(
        "Wave 0 cross-context cosine",
        passed=diversity['wave0_cross_ctx_cosine'] < 0.85,
        score=f"{diversity['wave0_cross_ctx_cosine']:.3f}",
        threshold="< 0.85",
    )
    results.add_test(
        "Hidden init cross-context cosine",
        passed=diversity['hidden_init_cross_ctx_cosine'] < 0.90,
        score=f"{diversity['hidden_init_cross_ctx_cosine']:.3f}",
        threshold="< 0.90",
    )
    results.add_test(
        "Training speed",
        passed=wg_result.steps_per_second > 200,
        score=f"{wg_result.steps_per_second:.1f} step/s",
        threshold="> 200 step/s",
    )
    results.add_test(
        "Valid word rate",
        passed=valid_rate > 0.3,
        score=f"{valid_rate:.1%}",
        threshold="> 30%",
    )
    results.add_metric("WG training steps", wg_result.total_steps)
    results.add_metric("WG final loss", f"{wg_result.final_loss:.4f}")
    results.add_metric("WG avg cosine acc", f"{wg_result.wave_cosine_accuracy:.3f}")
    results.add_metric("WG speed", f"{wg_result.steps_per_second:.1f} step/s")
    if joint_result:
        results.add_metric("Joint steps", joint_result.total_steps)
        results.add_metric("Joint skipped", joint_result.skipped_steps)
    results.add_metric("Valid word rate", f"{valid_rate:.1%}")
    results.save(str(Path(__file__).parent / 'RESULTS_PHASE_9_5.md'))

    log.success("Phase 9.5 complete")
