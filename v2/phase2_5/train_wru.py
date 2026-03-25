"""
train_wru.py — Phase 2.5 v2: Wave Recurrent Unit — Single-Step Next-Wave Prediction

Trains:
  - WaveRecurrentUnit (WRU): field_context [512] → predicted_wave [432]
    Using FLUX-native physics: per-band interference gates, energy conservation,
    wave superposition — no GRU, no sigmoid gates.

Frozen (from Phase 1 + Phase 2 checkpoints):
  - ContinuousSemanticEncoder (Phase 1)
  - WaveChunker (Phase 1)
  - WaveToText (Phase 1)
  - WaveToField (Phase 2, bridge_wtf)
  - FieldToWave (Phase 2, bridge_ftw)

Training objective — NEXT-WAVE PREDICTION:
  Given chunk_waves[:n] (a prefix of word-level waves), predict chunk_waves[n].
  Context = WaveToField(chunk_waves[:n].mean(dim=0))  →  field_context [512]
  Target = chunk_waves[n]  →  ground-truth next wave [432]

Losses:
  wave_mse    — MSE(predicted, target) — raw reconstruction
  cosine_loss — 1 - cos_sim(predicted, target) — direction fidelity
  decode_loss — WTT(predicted) cross-entropy vs target_bytes — TEXT fidelity

Decode gate (Phase 2.5):
  text → CSE → chunk → prefix[:n] → mean → WTF → WRU → predicted_wave → WTT → bytes
  Must achieve: avg byte accuracy ≥ 60%, min ≥ 30%  (relaxed vs Phase 2 — this is generation)

Run: python train_wru.py [--steps N] [--device cpu/cuda/mps]
"""

import sys
import os
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import OrderedDict
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# v2/phase1 imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1'))
from wave_types import TOTAL_WAVE_DIM

# v2/phase2 imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase2'))
from wave_to_field import WaveToField
from field_to_wave import FieldToWave

# v2/phase2_5 imports
sys.path.insert(0, str(Path(__file__).parent))
from wave_recurrent_unit import WaveRecurrentUnit, FIELD_DIM

# Root imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from flux_utils import PhaseLogger, PhaseResults, get_device, save_checkpoint, upload_checkpoint_to_hf


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

PHASE2_5_CONFIG = {
    'wave_dim': TOTAL_WAVE_DIM,     # 432
    'field_dim': FIELD_DIM,          # 512
    'energy_cap': 50.0,
    'residual_scale': 0.3,
    'wave_mse_weight': 1.0,
    'cosine_weight': 0.5,
    'decode_weight': 0.3,
    'min_prefix_len': 1,            # Minimum prefix length for context
    'gate_avg_threshold': 0.60,     # Relaxed: generation, not reconstruction
    'gate_min_threshold': 0.30,
}


# ─────────────────────────────────────────────
# Corpus: same rich sources as Phase 1 + 2
# ─────────────────────────────────────────────

_FALLBACK_TEXTS: List[str] = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models translate patterns in data into actionable predictions.",
    "Physics describes the fundamental laws that govern the behavior of matter and energy.",
    "Neural networks approximate functions by composing linear transformations and nonlinearities.",
    "Language models have demonstrated emergent capabilities across diverse tasks.",
    "Attention mechanisms allow models to focus on relevant parts of the input sequence.",
    "Gradient descent optimizes parameters by following the direction of steepest descent.",
    "The transformer architecture relies on self-attention and feed-forward layers.",
    "Backpropagation computes gradients efficiently using the chain rule of calculus.",
    "Water is a polar molecule consisting of two hydrogen atoms bonded to one oxygen.",
    "Photosynthesis converts light energy into chemical energy stored as glucose.",
    "DNA encodes genetic information as sequences of nucleotide base pairs.",
    "The speed of light in vacuum is approximately 299,792,458 meters per second.",
    "Entropy measures the degree of disorder or randomness in a thermodynamic system.",
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "café résumé naïve coöperate",
    "∫₀^∞ e^(-x²) dx = √π/2",
    "The cat sat on the mat",
    "Water freezes at zero degrees Celsius",
    "Energy equals mass times the speed of light squared",
]


def build_corpus() -> List[str]:
    """Build training corpus using Phase 1 pipeline (rich HF datasets)."""
    try:
        from train_codec import build_training_corpus
        corpus = build_training_corpus(target_size=20_000)
        print(f"  ✓ Training corpus: {len(corpus):,} strings from HF datasets", flush=True)
        return corpus
    except Exception as e:
        print(f"  ⚠ build_training_corpus failed ({e}) — using {len(_FALLBACK_TEXTS)} fallback strings",
              flush=True)
        return list(_FALLBACK_TEXTS)


# ─────────────────────────────────────────────
# Precompute: text → (chunk_waves, chunk_bytes)
# ─────────────────────────────────────────────

def precompute_corpus(
    texts: List[str],
    cse,
    chunker,
    device: str = 'cpu',
    cache_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Precompute chunk_waves and chunk_bytes for each text.

    Each record stores the raw chunk sequence — NOT a mean-pooled summary.
    This is critical: we need per-position targets for next-wave prediction.

    Args:
        texts: Training corpus strings
        cse: Frozen ContinuousSemanticEncoder
        chunker: Frozen WaveChunker
        device: Device for computation
        cache_path: Optional path to save/load cache
    Returns:
        List of dicts with keys: 'chunk_waves', 'chunk_bytes', 'length'
    """
    # Try loading cache
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading corpus cache: {cache_path}", flush=True)
        try:
            cache = torch.load(cache_path, map_location='cpu', weights_only=False)
            print(f"  ✓ Loaded {len(cache)} cached records", flush=True)
            return cache
        except Exception as e:
            print(f"  ⚠ Cache load failed ({e}), recomputing...", flush=True)

    print(f"  Precomputing corpus ({len(texts)} texts)...", flush=True)
    records = []

    cse.eval()
    chunker.eval()

    with torch.no_grad():
        for i, text in enumerate(texts):
            byte_data = text.encode('utf-8')
            try:
                wave = cse.encode(text)
                pairs = chunker.chunk_with_bytes(wave.full, byte_data)
            except Exception:
                continue

            if len(pairs) < 2:
                # Need at least 2 chunks for prefix+target
                continue

            chunk_waves = torch.stack([p[0] for p in pairs]).cpu()  # [N, 432]
            chunk_bytes = [p[1] for p in pairs]                     # List[bytes]

            records.append({
                'chunk_waves': chunk_waves,   # [N, 432] — raw sequence
                'chunk_bytes': chunk_bytes,    # List[bytes] — per-chunk bytes
                'length': len(pairs),
            })

            if (i + 1) % 2000 == 0:
                print(f"    {i+1}/{len(texts)} texts processed, {len(records)} valid records",
                      flush=True)

    print(f"  ✓ Precomputed {len(records)} records (from {len(texts)} texts)", flush=True)

    # Save cache
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        torch.save(records, cache_path)
        mb = os.path.getsize(cache_path) / 1e6
        print(f"  ✓ Cache saved: {cache_path} ({mb:.1f} MB)", flush=True)

    return records


# ─────────────────────────────────────────────
# Batch Sampling — per-position next-wave targets
# ─────────────────────────────────────────────

def sample_batch(
    records: List[Dict],
    w2f: WaveToField,
    batch_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, List[bytes]]:
    """
    Sample a training batch for next-wave prediction.

    For each sample:
      1. Pick a random record
      2. Pick a random split position n ∈ [1, length-1]
      3. Context = w2f(chunk_waves[:n].mean(dim=0))  → [512]
      4. Target = chunk_waves[n]                     → [432]
      5. Target bytes = chunk_bytes[n]               → bytes

    Args:
        records: Precomputed corpus records
        w2f: Frozen WaveToField projection
        batch_size: Number of samples
        device: Device string
    Returns:
        (contexts [B, 512], target_waves [B, 432], target_bytes List[bytes])
    """
    contexts = []
    target_waves = []
    target_bytes = []

    for _ in range(batch_size):
        rec = random.choice(records)
        n = random.randint(1, rec['length'] - 1)

        # Prefix context: mean of first n chunk_waves → project to field space
        prefix = rec['chunk_waves'][:n].to(device)       # [n, 432]
        prefix_mean = prefix.mean(dim=0)                  # [432]

        with torch.no_grad():
            ctx = w2f(prefix_mean)                        # [512]

        contexts.append(ctx)
        target_waves.append(rec['chunk_waves'][n].to(device))  # [432]
        target_bytes.append(rec['chunk_bytes'][n])              # bytes

    contexts = torch.stack(contexts)          # [B, 512]
    target_waves = torch.stack(target_waves)  # [B, 432]

    return contexts, target_waves, target_bytes


# ─────────────────────────────────────────────
# Phase 2.5 Decode Gate
# ─────────────────────────────────────────────

DECODE_GATE_TEXTS = [
    "The future of artificial intelligence",
    "Energy equals mass times the speed of light squared",
    "Photosynthesis converts sunlight into chemical energy",
    "Water freezes at zero degrees Celsius",
    "The cat sat on the mat",
    "café naïve résumé",
    "def hello(): return 'world'",
    "∫₀^∞ e^(-x²) dx = √π/2",
]


@torch.no_grad()
def run_phase2_5_decode_gate(
    cse,
    chunker,
    w2f: WaveToField,
    wru: WaveRecurrentUnit,
    wtt,
    phase: float = 2.5,
    verbose: bool = True,
    temperature: float = 0.5,
) -> Tuple[bool, float, float]:
    """
    Phase 2.5 decode gate: predict next wave from prefix, decode it.

    Pipeline:
      text → CSE → chunker → chunk_waves
      prefix = chunk_waves[:n] → mean → WTF → field_context
      WRU(field_context) → predicted_wave
      WTT(predicted_wave) → decoded_bytes
      Compare decoded_bytes to chunk_bytes[n]

    We use n = length//2 (midpoint) as the split to test genuine prediction.

    Args:
        cse: Frozen CSE
        chunker: Frozen WaveChunker
        w2f: Frozen WaveToField
        wru: Trained WaveRecurrentUnit
        wtt: Frozen WaveToText
    Returns:
        (passed, avg_accuracy, min_accuracy)
    """
    from decode_gate import byte_accuracy

    cse.eval()
    chunker.eval()
    wru.eval()
    wtt.eval()

    results = []

    if verbose:
        print(f"\n  {'─'*60}")
        print(f"  Phase 2.5 v2 Decode Gate (WRU next-wave prediction)")
        print(f"  {'─'*60}")

    for text in DECODE_GATE_TEXTS:
        try:
            byte_data = text.encode('utf-8')
            wave = cse.encode(text)
            pairs = chunker.chunk_with_bytes(wave.full, byte_data)

            if len(pairs) < 2:
                # Too short — skip but count as 0
                results.append(0.0)
                if verbose:
                    print(f"  ⚠ [{0.0:.1%}] '{text[:45]}' — too few chunks ({len(pairs)})")
                continue

            chunk_waves = torch.stack([p[0] for p in pairs]).to(next(wru.parameters()).device)
            chunk_bytes = [p[1] for p in pairs]

            # Split at midpoint
            n = max(1, len(pairs) // 2)
            prefix = chunk_waves[:n]
            prefix_mean = prefix.mean(dim=0)

            # Predict next wave
            ctx = w2f(prefix_mean).unsqueeze(0)               # [1, 512]
            predicted_wave, _ = wru(ctx)                       # [1, 432]

            # Decode predicted wave
            decoded_bytes_raw = wtt.decode(predicted_wave.squeeze(0), temperature=temperature)
            decoded_text = decoded_bytes_raw.decode('utf-8', errors='replace')

            # Ground truth
            gt_bytes = chunk_bytes[n]
            gt_text = gt_bytes.decode('utf-8', errors='replace')

            acc = byte_accuracy(gt_text, decoded_text)

        except Exception as e:
            acc = 0.0
            decoded_text = f"<ERROR: {e}>"
            gt_text = "?"

        results.append(acc)
        if verbose:
            status = '✓' if acc >= 0.30 else '✗'
            print(f"  {status} [{acc:.1%}] '{text[:35]}' → gt='{gt_text[:15]}' pred='{decoded_text[:15]}'")

    avg_acc = sum(results) / max(len(results), 1)
    min_acc = min(results) if results else 0.0
    passed = avg_acc >= PHASE2_5_CONFIG['gate_avg_threshold'] and \
             min_acc >= PHASE2_5_CONFIG['gate_min_threshold']

    if verbose:
        print(f"  {'─'*60}")
        print(f"  Avg byte accuracy : {avg_acc:.1%}  (threshold: {PHASE2_5_CONFIG['gate_avg_threshold']:.0%})")
        print(f"  Min byte accuracy : {min_acc:.1%}  (threshold: {PHASE2_5_CONFIG['gate_min_threshold']:.0%})")
        if passed:
            print(f"  ✓ PHASE 2.5 DECODE GATE PASSED")
        else:
            print(f"  ⚠ PHASE 2.5 DECODE GATE: NOT YET (keep training)")
        print(f"  {'─'*60}\n")

    return passed, avg_acc, min_acc


# ─────────────────────────────────────────────
# Phase 1 + Phase 2 Checkpoint Loading
# ─────────────────────────────────────────────

def load_phase1_and_phase2(
    device: str = 'cpu',
    hf_token: str = '',
) -> Dict[str, Any]:
    """
    Load Phase 1 (CSE, Chunker, WTT) and Phase 2 (WaveToField, FieldToWave).
    All components are eval + frozen.

    Returns dict with keys: 'cse', 'chunker', 'wtt', 'w2f', 'f2w'
    """
    # ── Phase 1 ───────────────────────────────────────────────────────
    from train_codec import load_phase1_checkpoint
    codec = load_phase1_checkpoint(device=device, hf_token=hf_token)
    cse = codec.cse.eval()
    chunker = codec.chunker.eval()
    wtt = codec.wtt.eval()
    for p in cse.parameters():
        p.requires_grad_(False)
    for p in chunker.parameters():
        p.requires_grad_(False)
    for p in wtt.parameters():
        p.requires_grad_(False)

    # ── Phase 2 ───────────────────────────────────────────────────────
    ckpt_dir = Path(__file__).parent.parent.parent / 'checkpoints'
    p2_path = ckpt_dir / 'phase2_v2.phase.pt'

    # Try local, then HuggingFace
    if not p2_path.exists():
        try:
            from huggingface_hub import hf_hub_download
            print(f"  Downloading Phase 2 checkpoint from HuggingFace...", flush=True)
            dl = hf_hub_download(
                repo_id='UnseenGAP/FLUX',
                filename='v2/phase2_v2.phase.pt',
                token=hf_token,
                local_dir=str(ckpt_dir),
            )
            import shutil
            dl_real = os.path.realpath(dl)
            if dl_real != str(p2_path):
                shutil.copy2(dl_real, str(p2_path))
        except Exception as e:
            raise FileNotFoundError(
                f"Phase 2 checkpoint not found at {p2_path}\n"
                f"HuggingFace download failed: {e}\n"
                f"Run Phase 2 training first."
            )

    p2_state = torch.load(str(p2_path), map_location='cpu', weights_only=False)

    w2f = WaveToField().to(device)
    f2w = FieldToWave().to(device)

    w2f.load_state_dict(p2_state['state_dict']['bridge_wtf'])
    f2w.load_state_dict(p2_state['state_dict']['bridge_ftw'])

    w2f.eval()
    f2w.eval()
    for p in w2f.parameters():
        p.requires_grad_(False)
    for p in f2w.parameters():
        p.requires_grad_(False)

    print(f"  ✓ Phase 1 + Phase 2 loaded and frozen", flush=True)

    return {
        'cse': cse,
        'chunker': chunker,
        'wtt': wtt,
        'w2f': w2f,
        'f2w': f2w,
    }


# ─────────────────────────────────────────────
# Checkpoint Save / Load for Phase 2.5
# ─────────────────────────────────────────────

def save_phase2_5_checkpoint(
    wru: WaveRecurrentUnit,
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    path: str,
) -> str:
    """Save Phase 2.5 checkpoint with WRU state dict."""
    state = {
        'phase': 2.5,
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'state_dict': {
            'wru': wru.state_dict(),
        },
        'metrics': metrics,
    }
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save(state, path)
    mb = os.path.getsize(path) / 1e6
    print(f"  ✓ Phase 2.5 checkpoint saved: {path} ({mb:.1f} MB)", flush=True)
    return path


def load_phase2_5_checkpoint(
    device: str = 'cpu',
    hf_token: str = '',
) -> WaveRecurrentUnit:
    """Load Phase 2.5 WRU from checkpoint."""
    ckpt_dir = Path(__file__).parent.parent.parent / 'checkpoints'
    path = ckpt_dir / 'phase2_5_v2.phase.pt'

    if not path.exists():
        try:
            from huggingface_hub import hf_hub_download
            dl = hf_hub_download(
                repo_id='UnseenGAP/FLUX',
                filename='v2/phase2_5_v2.phase.pt',
                token=hf_token,
                local_dir=str(ckpt_dir),
            )
            import shutil
            dl_real = os.path.realpath(dl)
            if dl_real != str(path):
                shutil.copy2(dl_real, str(path))
        except Exception as e:
            raise FileNotFoundError(
                f"Phase 2.5 checkpoint not found at {path}\n"
                f"HuggingFace download failed: {e}\n"
                f"Run Phase 2.5 training first."
            )

    state = torch.load(str(path), map_location='cpu', weights_only=False)
    config = state.get('config', PHASE2_5_CONFIG)

    wru = WaveRecurrentUnit(
        wave_dim=config.get('wave_dim', TOTAL_WAVE_DIM),
        field_dim=config.get('field_dim', FIELD_DIM),
        energy_cap=config.get('energy_cap', 50.0),
        residual_scale=config.get('residual_scale', 0.3),
    ).to(device)
    wru.load_state_dict(state['state_dict']['wru'])
    return wru


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────

def train_wru(
    steps: int = 30_000,
    batch_size: int = 64,
    device: str = 'auto',
    lr: float = 5e-4,
    log_every: int = 500,
    save_every: int = 5_000,
    gate_check_every: int = 5_000,
    wave_mse_weight: float = 1.0,
    cosine_weight: float = 0.5,
    decode_weight: float = 0.3,
    upload_hf: bool = False,
    hf_token: str = '',
) -> WaveRecurrentUnit:
    """
    Train the Phase 2.5 Wave Recurrent Unit for single-step next-wave prediction.

    Phase 1 (CSE, Chunker, WTT) and Phase 2 (WaveToField, FieldToWave) are FROZEN.
    Only the WRU is trained.

    Args:
        steps: Training steps
        batch_size: Batch size
        device: Device string ('auto' → best available)
        lr: Learning rate
        log_every: Print metrics every N steps
        save_every: Save checkpoint every N steps
        gate_check_every: Run decode gate every N steps
        wave_mse_weight: Weight for MSE(predicted, target) loss
        cosine_weight: Weight for (1 - cosine_sim) loss
        decode_weight: Weight for WTT decode cross-entropy loss
        upload_hf: Upload checkpoint to HF after training
        hf_token: HuggingFace token
    Returns:
        Trained WaveRecurrentUnit
    """
    if device == 'auto':
        device = get_device()

    if hf_token:
        hf_token = hf_token.strip()

    log = PhaseLogger(phase=2)  # Logs to phase2.log (2.5 shares)
    log.separator("Phase 2.5 v2 — Wave Recurrent Unit (Next-Wave Prediction)")
    log.info(f"Device: {device}")
    log.info(f"Steps: {steps:,}  Batch size: {batch_size}")
    log.info(f"LR: {lr}  wave_mse={wave_mse_weight}  cosine={cosine_weight}  decode={decode_weight}")

    # ── Load frozen components ────────────────────────────────────────
    log.info("Loading Phase 1 + Phase 2 checkpoints (will be FROZEN)...")
    components = load_phase1_and_phase2(device=device, hf_token=hf_token)
    cse = components['cse']
    chunker = components['chunker']
    wtt = components['wtt']
    w2f = components['w2f']
    f2w = components['f2w']
    log.success("Phase 1 + Phase 2 loaded and frozen")

    # ── Build corpus ──────────────────────────────────────────────────
    log.info("Building training corpus...")
    texts = build_corpus()
    log.info(f"Corpus: {len(texts):,} strings")

    # ── Precompute chunk_waves and chunk_bytes ────────────────────────
    ckpt_dir = Path(__file__).parent.parent.parent / 'checkpoints'
    cache_path = str(ckpt_dir / 'phase2_5_v2_corpus_cache.pt')
    records = precompute_corpus(texts, cse, chunker, device='cpu', cache_path=cache_path)

    if len(records) < 10:
        log.error(f"Only {len(records)} valid records — not enough for training")
        raise RuntimeError("Corpus too small. Check Phase 1 checkpoint and text sources.")

    log.info(f"Corpus: {len(records):,} usable records")

    # ── Build WRU ─────────────────────────────────────────────────────
    wru = WaveRecurrentUnit(
        wave_dim=TOTAL_WAVE_DIM,
        field_dim=FIELD_DIM,
        energy_cap=PHASE2_5_CONFIG['energy_cap'],
        residual_scale=PHASE2_5_CONFIG['residual_scale'],
    ).to(device)

    n_params = wru.count_parameters()
    log.info(f"WRU parameters: {n_params:,}")

    param_summary = wru.parameter_summary()
    for component, count in param_summary.items():
        if component != 'total':
            log.info(f"  {component}: {count:,}")

    # ── Optimizer ─────────────────────────────────────────────────────
    optimizer = AdamW(wru.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.01)

    # ── Checkpoint path ───────────────────────────────────────────────
    ckpt_path = str(ckpt_dir / 'phase2_5_v2.phase.pt')

    # ── Training metrics ──────────────────────────────────────────────
    running_mse = 0.0
    running_cos = 0.0
    running_dec = 0.0
    running_total = 0.0
    best_gate_avg = 0.0
    gate_passed = False

    # ── Training loop ─────────────────────────────────────────────────
    wru.train()
    log.separator("Training started")

    for step in range(1, steps + 1):
        optimizer.zero_grad()

        # ── Sample batch ──────────────────────────────────────────────
        contexts, target_waves, target_bytes_list = sample_batch(
            records, w2f, batch_size, device,
        )

        # ── Forward: WRU single-step prediction ──────────────────────
        predicted_waves, _ = wru(contexts)  # [B, 432]

        # ── Loss 1: Wave MSE ─────────────────────────────────────────
        mse_loss = F.mse_loss(predicted_waves, target_waves)

        # ── Loss 2: Cosine direction loss ─────────────────────────────
        cos_sim = F.cosine_similarity(predicted_waves, target_waves, dim=-1)
        cosine_loss = (1.0 - cos_sim).mean()

        # ── Loss 3: Decode loss (WTT cross-entropy) ──────────────────
        # Feed predicted waves through WTT in teacher-forced mode
        decode_targets = [
            torch.tensor(list(b), dtype=torch.long, device=device)
            for b in target_bytes_list
        ]
        # Disable cuDNN for GRU backward (known compatibility issue)
        with torch.backends.cudnn.flags(enabled=False):
            decode_loss = wtt.forward_batch(predicted_waves, decode_targets)

        # ── Total loss ────────────────────────────────────────────────
        total_loss = (
            wave_mse_weight * mse_loss +
            cosine_weight * cosine_loss +
            decode_weight * decode_loss
        )

        # ── Backward + update ─────────────────────────────────────────
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(wru.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # ── Accumulate metrics ────────────────────────────────────────
        running_mse += mse_loss.item()
        running_cos += cosine_loss.item()
        running_dec += decode_loss.item()
        running_total += total_loss.item()

        # ── Log ───────────────────────────────────────────────────────
        if step % log_every == 0:
            n = log_every
            avg_mse = running_mse / n
            avg_cos = running_cos / n
            avg_dec = running_dec / n
            avg_tot = running_total / n
            avg_cosim = 1.0 - avg_cos  # higher is better
            current_lr = scheduler.get_last_lr()[0]

            log.info(
                f"Step {step:>6,}/{steps:,}  "
                f"loss={avg_tot:.4f}  mse={avg_mse:.4f}  "
                f"cos_sim={avg_cosim:.4f}  decode={avg_dec:.4f}  "
                f"lr={current_lr:.2e}"
            )

            running_mse = 0.0
            running_cos = 0.0
            running_dec = 0.0
            running_total = 0.0

        # ── Save checkpoint ───────────────────────────────────────────
        if step % save_every == 0:
            metrics = {
                'step': step,
                'total_loss': total_loss.item(),
                'mse_loss': mse_loss.item(),
                'cosine_loss': cosine_loss.item(),
                'decode_loss': decode_loss.item(),
                'best_gate_avg': best_gate_avg,
            }
            save_phase2_5_checkpoint(wru, metrics, PHASE2_5_CONFIG, ckpt_path)

        # ── Decode gate check ─────────────────────────────────────────
        if step % gate_check_every == 0:
            wru.eval()
            passed, avg_acc, min_acc = run_phase2_5_decode_gate(
                cse, chunker, w2f, wru, wtt,
                phase=2.5,
                verbose=True,
                temperature=0.5,
            )
            wru.train()

            if avg_acc > best_gate_avg:
                best_gate_avg = avg_acc
                # Save best checkpoint
                metrics = {
                    'step': step,
                    'total_loss': total_loss.item(),
                    'gate_avg': avg_acc,
                    'gate_min': min_acc,
                    'gate_passed': passed,
                }
                save_phase2_5_checkpoint(wru, metrics, PHASE2_5_CONFIG, ckpt_path)
                log.success(f"New best gate avg: {avg_acc:.1%}")

            if passed and not gate_passed:
                gate_passed = True
                log.success(f"DECODE GATE PASSED at step {step:,}")
                log.success(f"  avg={avg_acc:.1%}  min={min_acc:.1%}")

    # ── Final gate check ──────────────────────────────────────────────
    wru.eval()
    log.separator("Final Decode Gate Check")
    passed, avg_acc, min_acc = run_phase2_5_decode_gate(
        cse, chunker, w2f, wru, wtt,
        phase=2.5,
        verbose=True,
        temperature=0.5,
    )

    # ── Final save ────────────────────────────────────────────────────
    metrics = {
        'step': steps,
        'gate_avg': avg_acc,
        'gate_min': min_acc,
        'gate_passed': passed,
        'best_gate_avg': best_gate_avg,
    }
    save_phase2_5_checkpoint(wru, metrics, PHASE2_5_CONFIG, ckpt_path)

    # ── Upload to HuggingFace ─────────────────────────────────────────
    if upload_hf and hf_token:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            api.upload_file(
                path_or_fileobj=ckpt_path,
                path_in_repo='v2/phase2_5_v2.phase.pt',
                repo_id='UnseenGAP/FLUX',
                token=hf_token,
            )
            log.success("Checkpoint uploaded to HuggingFace")
        except Exception as e:
            log.warning(f"HF upload failed: {e}")

    log.separator("Phase 2.5 training complete")
    log.metric("Final gate avg", f"{avg_acc:.1%}")
    log.metric("Best gate avg", f"{best_gate_avg:.1%}")
    log.metric("WRU params", f"{n_params:,}")

    return wru


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 2.5 v2: WRU Training')
    parser.add_argument('--steps', type=int, default=30_000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--upload-hf', action='store_true')
    args = parser.parse_args()

    hf_token = os.environ.get('HF_TOKEN', '')
    train_wru(
        steps=args.steps,
        batch_size=args.batch_size,
        device=args.device,
        lr=args.lr,
        upload_hf=args.upload_hf,
        hf_token=hf_token,
    )
