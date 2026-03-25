"""
train_codec.py — Phase 1 v2: Joint Wave Codec Training

Trains CSE + WaveChunker + WaveToText together from the very first step.
The wave space learns to be DECODABLE, not just encodable.

This is THE key difference from the original Phase 1:
- Original: train CSE alone → freeze wave space → WTT reverse-engineers it (Phase 9)
- v2: train CSE + WTT together → wave space is decodable from step 1

Loss = decode_loss + reconstruction_loss + coherence_loss

Run: python train_codec.py [--steps N] [--device cpu/cuda/mps]
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
sys.path.insert(0, str(Path(__file__).parent))
from wave_types import TOTAL_WAVE_DIM
from cse import ContinuousSemanticEncoder
from wave_chunker import WaveChunker
from wave_to_text import WaveToText
from decode_gate import run_decode_gate, byte_accuracy

# Root imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from flux_utils import (
    save_checkpoint, PhaseLogger, PhaseResults,
    get_device, upload_checkpoint_to_hf,
)


# ─────────────────────────────────────────────
# Training Corpus
# ─────────────────────────────────────────────

TRAINING_TEXTS: List[str] = [
    # English prose
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models translate patterns in data into actionable predictions.",
    "Physics describes the fundamental laws that govern the behavior of matter and energy.",
    "Neural networks approximate functions by composing linear transformations and nonlinearities.",
    "Language models have demonstrated emergent capabilities across diverse tasks.",
    "Attention mechanisms allow models to focus on relevant parts of the input sequence.",
    "Gradient descent optimizes parameters by following the direction of steepest descent.",
    "The transformer architecture relies on self-attention and feed-forward layers.",
    "Backpropagation computes gradients efficiently using the chain rule of calculus.",
    "Embeddings map discrete tokens to continuous vector representations.",
    # Scientific text
    "Water is a polar molecule consisting of two hydrogen atoms bonded to one oxygen.",
    "Photosynthesis converts light energy into chemical energy stored as glucose.",
    "DNA encodes genetic information as sequences of nucleotide base pairs.",
    "The speed of light in vacuum is approximately 299,792,458 meters per second.",
    "Entropy measures the degree of disorder or randomness in a thermodynamic system.",
    # Code
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "import torch; x = torch.randn(3, 4); y = x @ x.T",
    "for i in range(10): print(f'step {i}: loss={0.1/i+1:.4f}')",
    "class Model(nn.Module):\n    def forward(self, x):\n        return self.layers(x)",
    # Multi-byte UTF-8
    "café résumé naïve coöperate",
    "Привет мир — Hello world in Russian",
    "中文: 人工智能正在改变世界",
    "日本語: 機械学習は強力なツールです",
    "한국어: 딥러닝은 현대 AI의 핵심입니다",
    # Math
    "∫₀^∞ e^(-x²) dx = √π/2",
    "∑_{k=1}^{∞} 1/k² = π²/6",
    "E = mc² describes mass-energy equivalence",
    # Edge cases
    "a",
    "  leading and trailing spaces  ",
    "UPPERCASE AND lowercase MiXeD",
    "1234567890 !@#$%^&*()",
]


def get_training_texts(augment: bool = True) -> List[str]:
    """Return training texts, optionally augmented with sub-samples."""
    texts = list(TRAINING_TEXTS)
    if augment:
        # Add reversed, uppercased, and sub-string variants
        extras = []
        for t in TRAINING_TEXTS[:10]:
            if len(t) > 20:
                mid = len(t) // 2
                extras.append(t[:mid])          # First half
                extras.append(t[mid:])          # Second half
        texts.extend(extras)
    return texts


# ─────────────────────────────────────────────
# Codec Container
# ─────────────────────────────────────────────

class WaveCodec(nn.Module):
    """
    Container for CSE + WaveChunker + WaveToText.
    Holds all three components as submodules so a single optimizer
    can train them jointly.
    """

    def __init__(self, device: str = 'cpu'):
        super().__init__()
        self.cse     = ContinuousSemanticEncoder(device=device)
        self.chunker = WaveChunker()
        self.wtt     = WaveToText()

    def encode_and_decode_loss(self, text: str) -> torch.Tensor:
        """
        Full forward pass: text → CSE → chunker → WTT → decode_loss.
        This is the primary training signal.

        Args:
            text: Input string
        Returns:
            Scalar decode loss
        """
        byte_data = text.encode('utf-8')
        if len(byte_data) == 0:
            return torch.tensor(0.0)

        # Encode
        wave = self.cse.encode(text)             # SemanticWave
        wave_full = wave.full                     # [seq_len, 432]

        # Chunk with byte ground truth
        pairs = self.chunker.chunk_with_bytes(wave_full, byte_data)
        if len(pairs) == 0:
            return torch.tensor(0.0)

        # Decode loss: WTT must decode each chunk to its correct bytes
        chunk_waves = torch.stack([p[0] for p in pairs])     # [N, 432]
        target_tensors = [
            torch.tensor(list(p[1]), dtype=torch.long, device=chunk_waves.device)
            for p in pairs
        ]

        # Filter out empty targets
        valid = [(cw, tgt) for cw, tgt in zip(chunk_waves, target_tensors) if tgt.shape[0] > 0]
        if not valid:
            return torch.tensor(0.0, device=wave_full.device)

        cw_batch = torch.stack([v[0] for v in valid])
        tgt_list = [v[1] for v in valid]

        decode_loss = self.wtt.forward_batch(cw_batch, tgt_list)
        return decode_loss

    def coherence_loss(self, text: str) -> torch.Tensor:
        """
        Coherence loss: similar words should have similar waves.
        Ensures the wave space has meaningful geometry.

        Uses a positive pair (same text, minor aug) + random negative.

        Args:
            text: Input string
        Returns:
            Scalar coherence loss
        """
        wave1 = self.cse.encode(text).full.mean(dim=0)  # [432]

        # Positive: encode with slightly different padding (perturb bytes)
        byte_data = text.encode('utf-8')
        if len(byte_data) < 2:
            return torch.tensor(0.0, device=wave1.device)

        # Simple augmentation: trim 1 char from end
        aug_text = text[:-1] if len(text) > 1 else text
        wave2 = self.cse.encode(aug_text).full.mean(dim=0)

        # Positive pair: cosine should be > 0.8
        pos_sim = F.cosine_similarity(wave1.unsqueeze(0), wave2.unsqueeze(0))
        coherence_loss = F.relu(0.8 - pos_sim).mean()

        return coherence_loss


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────

def train_codec(
    steps: int = 30_000,
    device: str = 'auto',
    lr: float = 3e-4,
    log_every: int = 500,
    save_every: int = 5_000,
    decode_loss_weight: float = 1.0,
    coherence_loss_weight: float = 0.1,
    gate_check_every: int = 5_000,
    upload_hf: bool = False,
    hf_token: str = None,
) -> WaveCodec:
    """
    Joint training of CSE + WaveChunker + WaveToText.

    Args:
        steps: Total training steps
        device: Device ('auto', 'cuda', 'cpu', 'mps')
        lr: Learning rate
        log_every: Print loss every N steps
        save_every: Save checkpoint every N steps
        decode_loss_weight: Weight for WTT cross-entropy loss
        coherence_loss_weight: Weight for wave coherence loss
        gate_check_every: Run decode gate every N steps
        upload_hf: Whether to upload checkpoint to HuggingFace
        hf_token: HuggingFace token (optional)
    Returns:
        Trained WaveCodec
    """
    if device == 'auto':
        device = get_device()

    log = PhaseLogger(phase=1)
    log.separator("Phase 1 v2 — Wave Codec Joint Training")
    log.info(f"Device: {device}")
    log.info(f"Steps: {steps:,}")
    log.info(f"decode_loss_weight={decode_loss_weight}  coherence_loss_weight={coherence_loss_weight}")

    # ── Build model ──────────────────────────────────────────────
    codec = WaveCodec(device=device).to(device)
    n_params = sum(p.numel() for p in codec.parameters() if p.requires_grad)
    log.info(f"Model parameters: {n_params:,}")

    # ── Optimizer ────────────────────────────────────────────────
    optimizer = AdamW(codec.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.01)

    # ── Training texts ────────────────────────────────────────────
    texts = get_training_texts(augment=True)
    log.info(f"Training corpus: {len(texts)} texts")

    # ── Metrics tracking ─────────────────────────────────────────
    running_decode_loss   = 0.0
    running_coherence_loss = 0.0
    running_total_loss    = 0.0
    best_decode_loss      = float('inf')

    # ── Training loop ─────────────────────────────────────────────
    codec.train()
    for step in range(1, steps + 1):
        # Cycle through training texts
        text = texts[(step - 1) % len(texts)]

        optimizer.zero_grad()

        # Primary loss: decode fidelity
        d_loss = codec.encode_and_decode_loss(text)
        if not torch.is_tensor(d_loss) or d_loss.numel() == 0:
            d_loss = torch.tensor(0.0, device=device)

        # Secondary loss: wave coherence
        c_loss = codec.coherence_loss(text)
        if not torch.is_tensor(c_loss) or c_loss.numel() == 0:
            c_loss = torch.tensor(0.0, device=device)

        total_loss = decode_loss_weight * d_loss + coherence_loss_weight * c_loss

        if total_loss.requires_grad:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        running_decode_loss    += d_loss.item()
        running_coherence_loss += c_loss.item()
        running_total_loss     += total_loss.item()

        # ── Logging ───────────────────────────────────────────────
        if step % log_every == 0:
            avg_d   = running_decode_loss   / log_every
            avg_c   = running_coherence_loss / log_every
            avg_tot = running_total_loss    / log_every
            lr_now  = scheduler.get_last_lr()[0]

            log.metric(f"step {step:6d}", f"decode={avg_d:.4f}  coherence={avg_c:.4f}  total={avg_tot:.4f}  lr={lr_now:.6f}")

            if avg_d < best_decode_loss:
                best_decode_loss = avg_d

            running_decode_loss    = 0.0
            running_coherence_loss = 0.0
            running_total_loss     = 0.0

        # ── Periodic decode gate check ─────────────────────────────
        if step % gate_check_every == 0:
            codec.eval()
            try:
                passed, avg_acc, min_acc = run_decode_gate(
                    codec.cse, codec.chunker, codec.wtt,
                    phase=1, raise_on_fail=False,
                )
                status = '✓' if passed else '⚠ NOT YET'
                log.metric(f"step {step:6d} decode gate", f"{status}  avg={avg_acc:.1%}  min={min_acc:.1%}")
            except Exception as e:
                log.warning(f"Decode gate error at step {step}: {e}")
            codec.train()

        # ── Periodic checkpoint ────────────────────────────────────
        if step % save_every == 0:
            _save_phase1_checkpoint(codec, step, best_decode_loss)
            log.success(f"Checkpoint saved at step {step}")

    # ── Final checkpoint ──────────────────────────────────────────
    codec.eval()
    log.separator("Final Decode Gate Check")
    passed, avg_acc, min_acc = run_decode_gate(
        codec.cse, codec.chunker, codec.wtt,
        phase=1, raise_on_fail=False,
    )

    final_path = _save_phase1_checkpoint(codec, steps, best_decode_loss, is_final=True)
    log.success(f"Final checkpoint saved: {final_path}")

    if upload_hf and hf_token:
        try:
            upload_checkpoint_to_hf(phase=1, hf_token=hf_token)
            log.success("Checkpoint uploaded to HuggingFace Hub")
        except Exception as e:
            log.warning(f"HF upload failed: {e}")

    # ── Results ───────────────────────────────────────────────────
    results = PhaseResults(phase=1, component_name="Wave Codec (v2 — Joint CSE+WTT)")
    results.add_test("Decode Gate Avg Accuracy", passed=(avg_acc >= 0.90), score=avg_acc, threshold=0.90)
    results.add_test("Decode Gate Min Accuracy", passed=(min_acc >= 0.70), score=min_acc, threshold=0.70)
    results.add_test("Training Converged", passed=(best_decode_loss < 1.0), score=best_decode_loss, threshold=1.0)
    results.save()

    log.separator("Phase 1 v2 Training Complete")
    log.info(f"Best decode loss : {best_decode_loss:.4f}")
    log.info(f"Gate avg accuracy: {avg_acc:.1%}")
    log.info(f"Gate min accuracy: {min_acc:.1%}")

    return codec


# ─────────────────────────────────────────────
# Checkpoint Helpers
# ─────────────────────────────────────────────

def _save_phase1_checkpoint(
    codec: WaveCodec,
    step: int,
    best_loss: float,
    is_final: bool = False,
) -> Path:
    """Save phase 1 v2 checkpoint."""

    checkpoint_dir = Path(__file__).parent.parent.parent / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / 'phase1_v2.phase.pt'

    state = {
        'phase': 1,
        'version': 'v2',
        'timestamp': datetime.now().isoformat(),
        'step': step,
        'is_final': is_final,
        'config': {
            'wave_dim': TOTAL_WAVE_DIM,
            'chunker_min': 2,
            'chunker_max': 20,
            'wtt_hidden_dim': 256,
            'wtt_max_bytes': 20,
        },
        'state_dict': OrderedDict({
            'cse': codec.cse.state_dict(),
            'chunker': codec.chunker.state_dict(),
            'wtt': codec.wtt.state_dict(),
        }),
        'metrics': {
            'best_decode_loss': best_loss,
        },
    }

    torch.save(state, path)
    size_mb = path.stat().st_size / (1024 ** 2)
    print(f"  ✓ Checkpoint saved: {path} ({size_mb:.1f} MB)")
    return path


def load_phase1_checkpoint(device: str = 'cpu') -> WaveCodec:
    """
    Load a saved Phase 1 v2 checkpoint.

    Args:
        device: Target device
    Returns:
        WaveCodec with loaded weights
    """
    checkpoint_dir = Path(__file__).parent.parent.parent / 'checkpoints'
    path = checkpoint_dir / 'phase1_v2.phase.pt'

    if not path.exists():
        raise FileNotFoundError(
            f"Phase 1 v2 checkpoint not found at {path}\n"
            f"Run: python train_codec.py"
        )

    state = torch.load(path, map_location='cpu')
    codec = WaveCodec(device=device)

    codec.cse.load_state_dict(state['state_dict']['cse'])
    codec.chunker.load_state_dict(state['state_dict']['chunker'])
    codec.wtt.load_state_dict(state['state_dict']['wtt'])

    codec = codec.to(device)
    codec.eval()

    step = state.get('step', '?')
    loss = state['metrics'].get('best_decode_loss', '?')
    print(f"  ✓ Phase 1 v2 checkpoint loaded (step={step}, best_decode_loss={loss})")
    return codec


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Phase 1 v2 Wave Codec')
    parser.add_argument('--steps',   type=int,   default=30_000,  help='Training steps')
    parser.add_argument('--device',  type=str,   default='auto',  help='Device')
    parser.add_argument('--lr',      type=float, default=3e-4,    help='Learning rate')
    parser.add_argument('--log-every',  type=int, default=500)
    parser.add_argument('--save-every', type=int, default=5_000)
    parser.add_argument('--upload-hf',  action='store_true', help='Upload to HuggingFace Hub')
    parser.add_argument('--hf-token',   type=str, default=None)
    args = parser.parse_args()

    train_codec(
        steps=args.steps,
        device=args.device,
        lr=args.lr,
        log_every=args.log_every,
        save_every=args.save_every,
        upload_hf=args.upload_hf,
        hf_token=args.hf_token,
    )
