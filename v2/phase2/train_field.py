"""
train_field.py — Phase 2 v2: Resonance Field with Decode Loss

Trains:
  - WaveToField:  wave [432] → field [512]  (NEW — must be decodable)
  - FieldToWave:  field [512] → wave [432]  (NEW — the crucial inverse)
  - ResonanceField.wave_to_location / wave_to_feature  (coordinate + feature projections)

Frozen (from Phase 1 v2 checkpoint):
  - ContinuousSemanticEncoder
  - WaveChunker
  - WaveToText

The critical fix vs original Phase 2:
  - **Trains WaveToField and FieldToWave from step 1 with a decode loss**
  - Original Phase 2 trained neither → they stayed random → Phase 9 failed

Phase 2 decode gate:
    text → CSE (frozen) → WaveChunker (frozen) → WaveToField → FieldToWave → WTT (frozen) → text
    Must achieve > 90% byte accuracy (same threshold as Phase 1 gate)

Run: python train_field.py [--steps N] [--device cpu/cuda/mps]
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
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
sys.path.insert(0, str(Path(__file__).parent))
from field import ResonanceField, FIELD_H, FIELD_W, FIELD_D, FIELD_FEATURES
from attractor import AttractorCatalog
from wave_to_field import WaveToField
from field_to_wave import FieldToWave, wave_field_reconstruction_loss

# Root imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from flux_utils import PhaseLogger, PhaseResults, get_device, save_checkpoint, upload_checkpoint_to_hf


# ─────────────────────────────────────────────
# Training Corpus (same as Phase 1 for consistency)
# ─────────────────────────────────────────────

TRAINING_TEXTS: List[str] = [
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


# ─────────────────────────────────────────────
# FieldBridge Container
# ─────────────────────────────────────────────

class FieldBridge(nn.Module):
    """
    Container for the trainable bridge components in Phase 2 v2.

    Holds WaveToField + FieldToWave as a single trainable unit.
    The ResonanceField itself is updated by physics (perturbation),
    but wave_to_location and wave_to_feature are trained with gradients.
    """

    def __init__(self):
        super().__init__()
        self.wtf = WaveToField(wave_dim=TOTAL_WAVE_DIM, field_dim=FIELD_FEATURES)
        self.ftw = FieldToWave(field_dim=FIELD_FEATURES, wave_dim=TOTAL_WAVE_DIM)

    def round_trip(self, wave: torch.Tensor) -> torch.Tensor:
        """
        Wave → field → wave round-trip.

        Args:
            wave: [..., 432] input wave(s)
        Returns:
            [..., 432] reconstructed wave(s)
        """
        field_vec = self.wtf(wave)
        return self.ftw(field_vec)


# ─────────────────────────────────────────────
# Phase 2 v2 Decode Gate
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
def run_phase2_decode_gate(
    cse, chunker, bridge: FieldBridge, wtt,
    phase: int = 2,
    verbose: bool = True,
    temperature: float = 0.5,
) -> tuple:
    """
    Phase 2 decode gate: text → CSE → Chunker → WaveToField → FieldToWave → WTT → text.

    The field bridge is in the loop now. This tests that WaveToField + FieldToWave
    are not destroying the decodable wave structure.

    Returns:
        (passed, avg_accuracy, min_accuracy)
    """
    from decode_gate import byte_accuracy

    cse.eval()
    chunker.eval()
    bridge.eval()
    wtt.eval()

    results = []

    if verbose:
        print(f"\n  {'─'*55}")
        print(f"  Phase 2 v2 Decode Gate (via field bridge)")
        print(f"  {'─'*55}")

    for text in DECODE_GATE_TEXTS:
        try:
            wave = cse.encode(text)
            chunk_waves, _ = chunker(wave.full)             # [N, 432]
            round_tripped = bridge.round_trip(chunk_waves)  # [N, 432]
            decoded_text  = wtt.decode_to_text(round_tripped, temperature=temperature)
            acc = byte_accuracy(text, decoded_text)
        except Exception as e:
            acc = 0.0
            decoded_text = f"<ERROR: {e}>"

        results.append(acc)
        if verbose:
            status = '✓' if acc >= 0.70 else '✗'
            print(f"  {status} [{acc:.1%}] '{text[:45]}'")

    avg_acc = sum(results) / len(results)
    min_acc = min(results)
    passed  = avg_acc >= 0.90 and min_acc >= 0.70

    if verbose:
        print(f"  {'─'*55}")
        print(f"  Avg byte accuracy : {avg_acc:.1%}  (threshold: 90%)")
        print(f"  Min byte accuracy : {min_acc:.1%}  (threshold: 70%)")
        print(f"  {'─'*55}")
        if passed:
            print(f"  ✓ PHASE 2 DECODE GATE PASSED")
        else:
            print(f"  ⚠ PHASE 2 DECODE GATE: NOT YET (keep training)")
        print()

    return passed, avg_acc, min_acc


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────

def train_field(
    steps: int = 20_000,
    device: str = 'auto',
    lr: float = 1e-3,
    log_every: int = 500,
    save_every: int = 5_000,
    gate_check_every: int = 5_000,
    recon_loss_weight: float = 1.0,
    decode_loss_weight: float = 0.5,
    field_settle_every: int = 1_000,
    upload_hf: bool = False,
    hf_token: str = None,
):
    """
    Train the Phase 2 v2 resonance field bridge.

    Phase 1 v2 components (CSE, WaveChunker, WTT) are FROZEN.
    We only train: WaveToField, FieldToWave, field.wave_to_location, field.wave_to_feature

    Args:
        steps: Training steps
        device: Device string
        lr: Learning rate
        log_every: Print interval
        save_every: Checkpoint interval
        gate_check_every: Decode gate check interval
        recon_loss_weight: Weight for wave reconstruction loss
        decode_loss_weight: Weight for decode (WTT) loss
        field_settle_every: Settle field every N steps
        upload_hf: Upload checkpoint to HuggingFace
        hf_token: HuggingFace token
    """
    if device == 'auto':
        device = get_device()

    log = PhaseLogger(phase=2)
    log.separator("Phase 2 v2 — Resonance Field with Decode Loss")
    log.info(f"Device: {device}")
    log.info(f"Steps: {steps:,}")
    log.info(f"recon_loss_weight={recon_loss_weight}  decode_loss_weight={decode_loss_weight}")

    # ── Load Phase 1 v2 checkpoint (frozen) ─────────────────────────
    log.info("Loading Phase 1 v2 checkpoint (will be FROZEN)...")
    try:
        from train_codec import load_phase1_checkpoint
        codec = load_phase1_checkpoint(device=device, hf_token=hf_token or '')
        cse     = codec.cse.eval()
        chunker = codec.chunker.eval()
        wtt     = codec.wtt.eval()

        # Freeze Phase 1 parameters
        for p in cse.parameters():
            p.requires_grad_(False)
        for p in chunker.parameters():
            p.requires_grad_(False)
        for p in wtt.parameters():
            p.requires_grad_(False)

        log.success("Phase 1 v2 loaded and frozen")
    except FileNotFoundError as _e:
        log.error(f"Phase 1 v2 checkpoint not found!\n  {_e}")
        raise

    # ── Build Phase 2 components ─────────────────────────────────────
    field  = ResonanceField().to(device)
    bridge = FieldBridge().to(device)

    # Also train field's spatial projection (wave_to_location) and
    # wave_to_feature with gradients during this phase
    trainable_params = (
        list(bridge.parameters()) +
        list(field.wave_to_location.parameters()) +
        list(field.wave_to_feature.parameters())
    )
    n_params = sum(p.numel() for p in trainable_params)
    log.info(f"Trainable parameters: {n_params:,}")

    # ── Optimizer ────────────────────────────────────────────────────
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.01)

    # ── AttractorCatalog ─────────────────────────────────────────────
    catalog = AttractorCatalog(field)

    # ── Metrics ──────────────────────────────────────────────────────
    running_recon    = 0.0
    running_decode   = 0.0
    running_total    = 0.0
    best_recon_loss  = float('inf')
    best_decode_loss = float('inf')

    # ── Training loop ─────────────────────────────────────────────────
    texts = TRAINING_TEXTS
    bridge.train()
    field.wave_to_location.train()
    field.wave_to_feature.train()

    for step in range(1, steps + 1):
        text = texts[(step - 1) % len(texts)]
        byte_data = text.encode('utf-8')

        optimizer.zero_grad()

        # ── Encode (frozen CSE + Chunker) ───────────────────────────
        with torch.no_grad():
            w       = cse.encode(text)
            pairs   = chunker.chunk_with_bytes(w.full, byte_data)

        if len(pairs) == 0:
            continue

        chunk_waves  = torch.stack([p[0] for p in pairs]).to(device)  # [N, 432]
        target_list  = [
            torch.tensor(list(p[1]), dtype=torch.long, device=device)
            for p in pairs
        ]

        # Remove empty targets
        valid = [(cw, tgt) for cw, tgt in zip(chunk_waves, target_list) if tgt.shape[0] > 0]
        if not valid:
            continue

        cw_batch   = torch.stack([v[0] for v in valid])
        tgt_list   = [v[1] for v in valid]

        # ── WaveToField → FieldToWave round-trip ───────────────────
        field_vecs    = bridge.wtf(cw_batch)          # [N, 512]
        reconstructed = bridge.ftw(field_vecs)         # [N, 432]

        # ── Losses ──────────────────────────────────────────────────
        # 1. Reconstruction: reconstructed wave ≈ original chunk wave
        recon_loss = F.mse_loss(reconstructed, cw_batch.detach())
        cos_loss   = (1.0 - F.cosine_similarity(reconstructed, cw_batch.detach())).mean()
        total_recon = recon_loss + 0.5 * cos_loss

        # 2. Decode: WTT(reconstructed_wave) ≈ original bytes
        # Reconstructed waves go through the FROZEN WTT
        decode_loss = wtt.forward_batch(reconstructed, tgt_list)

        total_loss = recon_loss_weight * total_recon + decode_loss_weight * decode_loss

        if total_loss.requires_grad:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

        scheduler.step()

        # ── Physics: perturb the field with the original wave ───────
        with torch.no_grad():
            wave_mean = w.full.mean(dim=0).to(device)
            field.perturb(wave_mean)

        running_recon  += total_recon.item()
        running_decode += decode_loss.item()
        running_total  += total_loss.item()

        # ── Logging ──────────────────────────────────────────────────
        if step % log_every == 0:
            avg_r = running_recon  / log_every
            avg_d = running_decode / log_every
            avg_t = running_total  / log_every
            lr_n  = scheduler.get_last_lr()[0]
            n_att = field.num_attractors()

            log.metric(
                f"step {step:6d}",
                f"recon={avg_r:.4f}  decode={avg_d:.4f}  total={avg_t:.4f}  "
                f"lr={lr_n:.6f}  attractors={n_att}"
            )

            if avg_r < best_recon_loss:
                best_recon_loss = avg_r
            if avg_d < best_decode_loss:
                best_decode_loss = avg_d

            running_recon  = 0.0
            running_decode = 0.0
            running_total  = 0.0

        # ── Periodic field settling ───────────────────────────────────
        if step % field_settle_every == 0:
            field.settle(steps=5)
            new_att = catalog.scan_and_update(mass_threshold=0.1)
            log.info(f"Field settled. {new_att} new attractors. Total: {catalog.count()}")

        # ── Periodic decode gate check ────────────────────────────────
        if step % gate_check_every == 0:
            bridge.eval()
            field.wave_to_location.eval()
            passed, avg_acc, min_acc = run_phase2_decode_gate(
                cse, chunker, bridge, wtt, phase=2, verbose=True,
            )
            log.metric(
                f"step {step:6d} P2 gate",
                f"{'✓' if passed else '⚠'} avg={avg_acc:.1%} min={min_acc:.1%}"
            )
            bridge.train()
            field.wave_to_location.train()
            field.wave_to_feature.train()

        # ── Periodic checkpoint ───────────────────────────────────────
        if step % save_every == 0:
            _save_phase2_checkpoint(field, bridge, catalog, step, best_recon_loss, best_decode_loss)
            log.success(f"Checkpoint saved at step {step}")

    # ── Final ─────────────────────────────────────────────────────────
    bridge.eval()
    field.wave_to_location.eval()
    field.wave_to_feature.eval()

    log.separator("Final Phase 2 v2 Decode Gate")
    passed, avg_acc, min_acc = run_phase2_decode_gate(
        cse, chunker, bridge, wtt, phase=2, verbose=True,
    )

    final_path = _save_phase2_checkpoint(
        field, bridge, catalog, steps, best_recon_loss, best_decode_loss, is_final=True
    )
    log.success(f"Final checkpoint saved: {final_path}")

    if upload_hf and hf_token:
        try:
            upload_checkpoint_to_hf(phase=2, hf_token=hf_token)
            log.success("Checkpoint uploaded to HuggingFace")
        except Exception as e:
            log.warning(f"HF upload failed: {e}")

    # ── Results ───────────────────────────────────────────────────────
    results = PhaseResults(phase=2, component_name="Resonance Field (v2 — with Decode Loss)")
    results.add_test("Phase 2 Decode Gate Avg", passed=(avg_acc >= 0.90), score=avg_acc, threshold=0.90)
    results.add_test("Phase 2 Decode Gate Min", passed=(min_acc >= 0.70), score=min_acc, threshold=0.70)
    results.add_test("Recon Loss Converged",    passed=(best_recon_loss < 0.5),  score=best_recon_loss, threshold=0.5)
    results.add_test("Decode Loss Converged",   passed=(best_decode_loss < 1.5), score=best_decode_loss, threshold=1.5)
    results.add_test("Attractors Formed",       passed=(catalog.count() > 5),    score=catalog.count(), threshold=5)
    results.save()

    log.separator("Phase 2 v2 Training Complete")
    log.info(f"Best recon loss  : {best_recon_loss:.4f}")
    log.info(f"Best decode loss : {best_decode_loss:.4f}")
    log.info(f"Gate avg         : {avg_acc:.1%}")
    log.info(f"Total attractors : {catalog.count()}")

    return field, bridge, catalog


# ─────────────────────────────────────────────
# Checkpoint Helpers
# ─────────────────────────────────────────────

def _save_phase2_checkpoint(
    field: ResonanceField,
    bridge: FieldBridge,
    catalog: AttractorCatalog,
    step: int,
    best_recon_loss: float,
    best_decode_loss: float,
    is_final: bool = False,
) -> Path:
    """Save Phase 2 v2 checkpoint."""
    checkpoint_dir = Path(__file__).parent.parent.parent / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / 'phase2_v2.phase.pt'

    state = {
        'phase': 2,
        'version': 'v2',
        'timestamp': datetime.now().isoformat(),
        'step': step,
        'is_final': is_final,
        'config': {
            'field_h': FIELD_H,
            'field_w': FIELD_W,
            'field_d': FIELD_D,
            'field_features': FIELD_FEATURES,
            'wave_dim': TOTAL_WAVE_DIM,
        },
        'state_dict': OrderedDict({
            'field':   field.state_dict(),
            'bridge_wtf': bridge.wtf.state_dict(),
            'bridge_ftw': bridge.ftw.state_dict(),
        }),
        'attractor_catalog': catalog.to_dict(),
        'metrics': {
            'best_recon_loss':  best_recon_loss,
            'best_decode_loss': best_decode_loss,
            'num_attractors':   catalog.count(),
        },
    }

    torch.save(state, path)
    size_mb = path.stat().st_size / (1024 ** 2)
    print(f"  ✓ Phase 2 v2 checkpoint saved: {path} ({size_mb:.1f} MB)")
    return path


def load_phase2_checkpoint(device: str = 'cpu'):
    """Load a saved Phase 2 v2 checkpoint."""
    checkpoint_dir = Path(__file__).parent.parent.parent / 'checkpoints'
    path = checkpoint_dir / 'phase2_v2.phase.pt'

    if not path.exists():
        raise FileNotFoundError(
            f"Phase 2 v2 checkpoint not found at {path}\n"
            f"Run: python train_field.py"
        )

    state = torch.load(path, map_location='cpu')

    field  = ResonanceField()
    bridge = FieldBridge()

    field.load_state_dict(state['state_dict']['field'])
    bridge.wtf.load_state_dict(state['state_dict']['bridge_wtf'])
    bridge.ftw.load_state_dict(state['state_dict']['bridge_ftw'])

    field  = field.to(device).eval()
    bridge = bridge.to(device).eval()

    catalog = AttractorCatalog(field)
    if state.get('attractor_catalog'):
        catalog.from_dict(state['attractor_catalog'])

    step = state.get('step', '?')
    m    = state['metrics']
    print(f"  ✓ Phase 2 v2 checkpoint loaded (step={step}, "
          f"recon={m.get('best_recon_loss','?'):.4f}, "
          f"decode={m.get('best_decode_loss','?'):.4f})")

    return field, bridge, catalog


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Phase 2 v2 Resonance Field')
    parser.add_argument('--steps',      type=int,   default=20_000)
    parser.add_argument('--device',     type=str,   default='auto')
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--log-every',  type=int,   default=500)
    parser.add_argument('--save-every', type=int,   default=5_000)
    parser.add_argument('--upload-hf',  action='store_true')
    parser.add_argument('--hf-token',   type=str,   default=None)
    args = parser.parse_args()

    train_field(
        steps=args.steps,
        device=args.device,
        lr=args.lr,
        log_every=args.log_every,
        save_every=args.save_every,
        upload_hf=args.upload_hf,
        hf_token=args.hf_token,
    )
