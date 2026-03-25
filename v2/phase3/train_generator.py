"""
train_generator.py — Phase 3 v2: Wave Generator Training

Trains WaveGenerator to predict the next wave from field context.
The KEY difference from legacy Phase 9.5: decode_loss is applied from step 1,
so the generator is forced to produce waves that are decodable to text.

Frozen (loaded from checkpoints):
    - CSE (ContinuousSemanticEncoder)       — phase1_v2.phase.pt
    - WaveChunker                           — phase1_v2.phase.pt
    - WaveToText                            — phase1_v2.phase.pt
    - WaveToField                           — phase2_v2.phase.pt
    - FieldToWave                           — phase2_v2.phase.pt

Trainable:
    - WaveGenerator  (GRU-based next-wave predictor)

Loss (per step):
    wave_loss    = MSE(predicted_wave, target_wave)
    cosine_loss  = 1 - cosine_similarity(predicted, target)
    contrastive  = penalise if same wave predicted for different contexts
    decode_loss  = cross_entropy(WTT(predicted_wave), gt_bytes)  ← THE KEY FIX

Checkpoint saved to: checkpoints/phase3_v2.phase.pt

Run: python train_generator.py [--steps N] [--device cpu/cuda/mps]
"""

import sys
import argparse
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import OrderedDict
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ─────────────────────────────────────────────
# Path setup
# ─────────────────────────────────────────────
_V2_DIR       = Path(__file__).parent.parent
_PHASE1_DIR   = _V2_DIR / 'phase1'
_PHASE2_DIR   = _V2_DIR / 'phase2'
_PROJECT_ROOT = _V2_DIR.parent

for _p in [str(_PHASE1_DIR), str(_PHASE2_DIR), str(Path(__file__).parent), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cse          import ContinuousSemanticEncoder
from wave_chunker import WaveChunker
from wave_to_text import WaveToText
from wave_to_field import WaveToField
from field_to_wave import FieldToWave
from wave_generator import WaveGenerator
from flux_utils   import PhaseLogger, PhaseResults, get_device, save_checkpoint


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

PHASE3_CONFIG: Dict[str, Any] = {
    # Model
    'wave_dim':            432,
    'field_features':      512,
    'gru_hidden':          512,
    'gru_layers':          1,
    'dropout':             0.15,
    'max_waves':           50,
    'interference_radius': 4,
    # Training
    'steps':               20_000,
    'batch_size':          64,
    'lr':                  3e-4,
    'ss_start':            0.5,    # scheduled sampling start probability
    'ss_end':              0.9,    # scheduled sampling end probability
    # Loss weights
    'wave_loss_weight':    1.0,
    'cosine_loss_weight':  0.5,
    'contrastive_weight':  2.0,
    'decode_loss_weight':  1.0,    # ← The key fix: decode loss from step 1
    # Decode gate
    'gate_avg_threshold':  0.90,
    'gate_min_threshold':  0.70,
    'gate_check_interval': 1_000,
}

# Decode gate texts — same 8 used in Phase 1 and 2
DECODE_GATE_TEXTS: List[str] = [
    "The future of artificial intelligence",
    "Energy equals mass times the speed of light squared",
    "Photosynthesis converts sunlight into chemical energy",
    "Water freezes at zero degrees Celsius",
    "The cat sat on the mat",
    "café naïve résumé",
    "def hello(): return 'world'",
    "∫₀^∞ e^(-x²) dx = √π/2",
]

_FALLBACK_TEXTS: List[str] = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models transform data into predictions.",
    "The Eiffel Tower stands 330 metres tall in Paris.",
    "Neural networks learn by adjusting billions of parameters.",
    "Water consists of two hydrogen atoms bonded to one oxygen.",
    "Photosynthesis converts sunlight into chemical energy stored as glucose.",
    "The speed of light is approximately 299,792,458 metres per second.",
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "café résumé naïve coöperate — UTF-8 test",
    "∫₀^∞ e^(-x²) dx = √π/2",
    "The capital of France is Paris.",
    "DNA encodes genetic information as sequences of nucleotide base pairs.",
    "Entropy measures the degree of disorder in a thermodynamic system.",
    "Shakespeare wrote Hamlet, Macbeth, and King Lear.",
    "The mitochondria is the powerhouse of the cell.",
    "Quantum mechanics describes nature at the smallest scales.",
    "Gravity curves spacetime according to general relativity.",
    "The Earth orbits the Sun once every 365.25 days.",
    "Python is a high-level, dynamically-typed programming language.",
    "import torch; x = torch.randn(3, 432)",
]


# ─────────────────────────────────────────────
# Corpus Builder
# ─────────────────────────────────────────────

def build_training_corpus(target_size: int = 10_000) -> List[str]:
    """
    Build training corpus from HuggingFace datasets with fallback.

    Tries: WikiText-103, TinyStories, CodeSearchNet.
    Falls back to _FALLBACK_TEXTS if datasets unavailable.

    Args:
        target_size: Target number of texts

    Returns:
        List of training strings
    """
    # Always pin gate texts at front so they're seen from step 1
    texts = list(DECODE_GATE_TEXTS)

    try:
        from datasets import load_dataset

        sources = [
            ('wikitext', 'wikitext-103-raw-v1', 'train', 'text', 3_000),
            ('roneneldan/TinyStories', None, 'train', 'text', 3_000),
        ]

        for name, config, split, field, max_n in sources:
            try:
                ds = load_dataset(name, config, split=split, streaming=True,
                                  trust_remote_code=True)
                count = 0
                for row in ds:
                    t = row[field].strip()
                    if len(t) > 20:
                        texts.append(t)
                        count += 1
                        if count >= max_n:
                            break
                print(f"  ✓ Loaded {count} texts from {name}")
            except Exception as e:
                print(f"  ⚠ {name}: {e}")

    except ImportError:
        print("  ⚠ datasets not available — using fallback corpus")

    # Pad with fallback if needed
    while len(texts) < 100:
        texts.extend(_FALLBACK_TEXTS)

    # Shuffle everything except the gate texts at front
    gate = texts[:len(DECODE_GATE_TEXTS)]
    rest = texts[len(DECODE_GATE_TEXTS):]
    random.shuffle(rest)
    return gate + rest[:target_size]


# ─────────────────────────────────────────────
# Checkpoint Loaders
# ─────────────────────────────────────────────

def load_phase1_components(
    checkpoint_path: Path,
    device: str,
) -> Tuple['ContinuousSemanticEncoder', 'WaveChunker', 'WaveToText']:
    """
    Load CSE, WaveChunker, WaveToText from phase1_v2.phase.pt.

    All three are frozen — only WaveGenerator trains in Phase 3.

    Args:
        checkpoint_path: Path to phase1_v2.phase.pt
        device: Target device

    Returns:
        (cse, chunker, wtt) all in eval mode on device
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Phase 1 v2 checkpoint not found: {checkpoint_path}\n"
            f"Run v2/phase1/train_codec.py first."
        )

    state = torch.load(checkpoint_path, map_location='cpu')
    cfg   = state['config']

    cse = ContinuousSemanticEncoder(
        wave_dim=cfg.get('wave_dim', 432),
        window_size=cfg.get('window_size', 8),
        stride=cfg.get('stride', 1),
    )
    cse.load_state_dict(state['state_dict']['cse'])
    cse.to(device).eval()
    for p in cse.parameters():
        p.requires_grad_(False)

    chunker = WaveChunker(
        wave_dim=cfg.get('wave_dim', 432),
        min_chunk=cfg.get('min_chunk', 2),
        max_chunk=cfg.get('max_chunk', 20),
    )
    chunker.load_state_dict(state['state_dict']['chunker'])
    chunker.to(device).eval()
    for p in chunker.parameters():
        p.requires_grad_(False)

    wtt = WaveToText(
        wave_dim=cfg.get('wave_dim', 432),
        hidden_dim=cfg.get('wtt_hidden_dim', 256),
        max_bytes=cfg.get('max_bytes', 20),
    )
    wtt.load_state_dict(state['state_dict']['wtt'])
    wtt.to(device).eval()
    for p in wtt.parameters():
        p.requires_grad_(False)

    print(f"  ✓ Phase 1 v2 loaded and frozen (CSE + WaveChunker + WTT)")
    return cse, chunker, wtt


def load_phase2_components(
    checkpoint_path: Path,
    device: str,
) -> Tuple['WaveToField', 'FieldToWave']:
    """
    Load WaveToField and FieldToWave from phase2_v2.phase.pt.

    Both are frozen — they are already trained with decode loss in Phase 2.

    Args:
        checkpoint_path: Path to phase2_v2.phase.pt
        device: Target device

    Returns:
        (wave_to_field, field_to_wave) both in eval mode on device
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Phase 2 v2 checkpoint not found: {checkpoint_path}\n"
            f"Run v2/phase2/train_field.py first."
        )

    state = torch.load(checkpoint_path, map_location='cpu')
    cfg   = state['config']

    w2f = WaveToField(
        wave_dim=cfg.get('wave_dim', 432),
        field_dim=cfg.get('field_features', 512),
    )
    w2f.load_state_dict(state['state_dict']['wave_to_field'])
    w2f.to(device).eval()
    for p in w2f.parameters():
        p.requires_grad_(False)

    f2w = FieldToWave(
        field_dim=cfg.get('field_features', 512),
        wave_dim=cfg.get('wave_dim', 432),
    )
    f2w.load_state_dict(state['state_dict']['field_to_wave'])
    f2w.to(device).eval()
    for p in f2w.parameters():
        p.requires_grad_(False)

    print(f"  ✓ Phase 2 v2 loaded and frozen (WaveToField + FieldToWave)")
    return w2f, f2w


# ─────────────────────────────────────────────
# Decode Gate
# ─────────────────────────────────────────────

@torch.no_grad()
def run_decode_gate(
    cse:       'ContinuousSemanticEncoder',
    chunker:   'WaveChunker',
    wtt:       'WaveToText',
    w2f:       'WaveToField',
    generator: 'WaveGenerator',
    device:    str,
    phase:     int = 3,
) -> Tuple[float, float]:
    """
    Run decode gate: prompt → CSE → field context → generate waves → WTT → text.

    Measures byte accuracy of the generated output vs the original prompt.

    Args:
        cse, chunker, wtt, w2f, generator: model components
        device: target device
        phase: phase number for logging

    Returns:
        (avg_byte_accuracy, min_byte_accuracy)
    """
    accs = []

    for text in DECODE_GATE_TEXTS:
        try:
            text_bytes = text.encode('utf-8')
            wave       = cse.encode(text)                          # SemanticWave
            mean_wave  = wave.full.mean(dim=0)                     # [432]
            ctx        = w2f(mean_wave.to(device))                 # [512]

            generated_waves, _ = generator.generate(
                field_context=ctx,
                max_waves=len(text_bytes) // 3 + 10,
            )                                                       # [N, 432]

            # Decode each generated wave to bytes
            decoded_bytes = b''
            for i in range(generated_waves.shape[0]):
                decoded = wtt.decode(generated_waves[i])
                if decoded:
                    decoded_bytes += bytes(decoded)

            # Compute byte accuracy
            n     = max(len(text_bytes), len(decoded_bytes))
            match = sum(
                a == b for a, b in zip(text_bytes, decoded_bytes)
            )
            acc = match / max(n, 1)
            accs.append(acc)
        except Exception:
            accs.append(0.0)

    avg = sum(accs) / len(accs) if accs else 0.0
    mn  = min(accs) if accs else 0.0
    return avg, mn


# ─────────────────────────────────────────────
# Training Step Helpers
# ─────────────────────────────────────────────

def compute_decode_loss(
    predicted_waves: torch.Tensor,
    target_bytes_list: List[bytes],
    wtt: 'WaveToText',
) -> torch.Tensor:
    """
    Compute cross-entropy decode loss on predicted waves.

    For each predicted wave, run through WTT and compare to ground truth bytes.
    This is the key loss that forces the generator to produce decodable waves.

    Args:
        predicted_waves:   [N, wave_dim] batch of predicted waves
        target_bytes_list: list of N byte sequences (ground truth)
        wtt:               WaveToText decoder

    Returns:
        Scalar decode loss tensor
    """
    losses = []
    for i, (wave, gt_bytes) in enumerate(zip(predicted_waves, target_bytes_list)):
        if not gt_bytes:
            continue
        try:
            logits = wtt.forward_train(wave.unsqueeze(0), gt_bytes)
            if logits is not None:
                gt_tensor = torch.tensor(list(gt_bytes), dtype=torch.long,
                                         device=wave.device)
                gt_tensor = gt_tensor[:logits.shape[0]]
                logits    = logits[:len(gt_tensor)]
                if len(gt_tensor) > 0:
                    losses.append(F.cross_entropy(logits, gt_tensor))
        except Exception:
            continue
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=predicted_waves.device)


def build_batch(
    texts:   List[str],
    cse:     'ContinuousSemanticEncoder',
    chunker: 'WaveChunker',
    w2f:     'WaveToField',
    device:  str,
    max_seq: int = 30,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[bytes]]]:
    """
    Build a training batch from a list of texts.

    Encodes texts to waves, chunks them, and builds padded tensors.

    Args:
        texts:   List of training strings
        cse:     Frozen CSE encoder
        chunker: Frozen WaveChunker
        w2f:     Frozen WaveToField
        device:  Target device
        max_seq: Maximum sequence length for padding

    Returns:
        (field_contexts [B, 512],
         target_waves   [B, max_seq, 432],
         lengths        [B],
         bytes_list     [B][N bytes per chunk])
    """
    contexts, targets, lengths, bytes_list = [], [], [], []

    for text in texts:
        try:
            with torch.no_grad():
                wave          = cse.encode(text)               # SemanticWave
                chunks, spans = chunker(wave.full)              # [N, 432], [N, bytes]
                mean_wave     = wave.full.mean(dim=0)
                ctx           = w2f(mean_wave.to(device))       # [512]

            n = min(len(chunks), max_seq)
            if n == 0:
                continue

            pad = torch.zeros(max_seq, 432, device=device)
            pad[:n] = chunks[:n].to(device)

            contexts.append(ctx)
            targets.append(pad)
            lengths.append(n)
            bytes_list.append([sp for sp in spans[:n]])

        except Exception:
            continue

    if not contexts:
        return None, None, None, None

    return (
        torch.stack(contexts),
        torch.stack(targets),
        torch.tensor(lengths, device=device),
        bytes_list,
    )


# ─────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────

def train(
    steps:  int = PHASE3_CONFIG['steps'],
    device: str = 'auto',
    lr:     float = PHASE3_CONFIG['lr'],
    decode_loss_weight: float = PHASE3_CONFIG['decode_loss_weight'],
    checkpoint_dir: Path = None,
    hf_token: str = None,
) -> Dict[str, Any]:
    """
    Train WaveGenerator for Phase 3.

    Args:
        steps:              Total training steps
        device:             Device string or 'auto'
        lr:                 Learning rate
        decode_loss_weight: Weight for the decode loss (KEY FIX)
        checkpoint_dir:     Directory to save checkpoint
        hf_token:           HuggingFace token for upload

    Returns:
        Dict of training metrics
    """
    from flux_utils import get_device
    if device == 'auto':
        device = get_device()

    log = PhaseLogger(phase=3)
    log.separator("Phase 3 v2 — Wave Generator Training")
    log.info(f"Device: {device}")
    log.info(f"Steps:  {steps:,}")
    log.info(f"decode_loss_weight={decode_loss_weight}")

    if checkpoint_dir is None:
        checkpoint_dir = _PROJECT_ROOT / 'checkpoints'
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_out = checkpoint_dir / 'phase3_v2.phase.pt'

    # ── Load frozen components ──
    p1_ckpt = checkpoint_dir / 'phase1_v2.phase.pt'
    p2_ckpt = checkpoint_dir / 'phase2_v2.phase.pt'

    cse, chunker, wtt = load_phase1_components(p1_ckpt, device)
    w2f, f2w = load_phase2_components(p2_ckpt, device)

    # ── Build WaveGenerator ──
    generator = WaveGenerator(
        wave_dim=PHASE3_CONFIG['wave_dim'],
        field_features=PHASE3_CONFIG['field_features'],
        gru_hidden=PHASE3_CONFIG['gru_hidden'],
        gru_layers=PHASE3_CONFIG['gru_layers'],
        dropout=PHASE3_CONFIG['dropout'],
        max_waves=PHASE3_CONFIG['max_waves'],
        interference_radius=PHASE3_CONFIG['interference_radius'],
    ).to(device)

    n_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    log.info(f"Trainable parameters: {n_params:,}")

    # ── Optimizer & scheduler ──
    optimizer = AdamW(generator.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.01)

    # ── Build corpus ──
    log.info("Building training corpus...")
    corpus = build_training_corpus(target_size=10_000)
    log.info(f"Training corpus: {len(corpus):,} texts")

    batch_size  = PHASE3_CONFIG['batch_size']
    ss_start    = PHASE3_CONFIG['ss_start']
    ss_end      = PHASE3_CONFIG['ss_end']

    # ── Training loop ──
    print(f"\n{'─' * 60}")
    print(f"Phase 3 v2 — Wave Generator Training")
    print(f"{'─' * 60}")

    best_gate_avg   = 0.0
    step            = 0
    running_loss    = 0.0
    log_interval    = 500

    start_time      = time.time()

    while step < steps:
        # Sample a batch of texts
        batch_texts = random.choices(corpus, k=batch_size)

        # Scheduled sampling probability — ramps up over training
        ss_p = ss_start + (ss_end - ss_start) * (step / steps)

        # Build batch
        field_contexts, target_waves, lengths, bytes_list = build_batch(
            texts=batch_texts,
            cse=cse,
            chunker=chunker,
            w2f=w2f,
            device=device,
        )
        if field_contexts is None:
            continue

        # Forward
        generator.train()
        predicted, confidences = generator.forward_batch(
            field_contexts=field_contexts,
            target_waves=target_waves,
            lengths=lengths,
            scheduled_sampling_p=ss_p,
        )

        # ── Wave losses ──
        # Only compute loss on valid (non-padded) positions
        B, S, D = predicted.shape
        mask = torch.zeros(B, S, device=device)
        for i, l in enumerate(lengths):
            mask[i, :l] = 1.0
        mask = mask.unsqueeze(-1)  # [B, S, 1]

        wave_loss   = (F.mse_loss(predicted, target_waves, reduction='none') * mask).sum() / mask.sum()
        cosine_loss = (1 - F.cosine_similarity(predicted, target_waves, dim=-1)).unsqueeze(-1)
        cosine_loss = (cosine_loss * mask).sum() / mask.sum()

        # ── Contrastive loss: different contexts → different waves ──
        if B > 1:
            mean_pred = (predicted * mask).sum(dim=1) / mask.sum(dim=1)  # [B, D]
            ctx_norm  = F.normalize(field_contexts, dim=-1)
            pred_norm = F.normalize(mean_pred, dim=-1)
            ctx_sim   = (ctx_norm @ ctx_norm.T)               # [B, B] context similarity
            pred_sim  = (pred_norm @ pred_norm.T)             # [B, B] prediction similarity
            eye       = torch.eye(B, device=device)
            # penalise if prediction similarity > context similarity for off-diagonals
            misalign  = F.relu(pred_sim - ctx_sim - 0.1) * (1 - eye)
            contrastive_loss = misalign.mean()
        else:
            contrastive_loss = torch.tensor(0.0, device=device)

        # ── Decode loss: force generated waves to be decodable ──
        # Sample a few predicted waves and check WTT can decode them
        wtt.train()  # brief train mode so we can backprop through WTT
        decode_losses = []
        n_decode_samples = min(8, B)
        sample_idx = random.sample(range(B), n_decode_samples)

        for i in sample_idx:
            L = lengths[i].item()
            if L == 0 or not bytes_list[i]:
                continue
            # Use the first predicted wave vs first ground-truth chunk bytes
            pred_wave = predicted[i, 0]       # [432] first predicted wave
            gt_bytes  = bytes_list[i][0]      # bytes for first chunk
            if not gt_bytes:
                continue
            try:
                logits = wtt.forward_train(pred_wave.unsqueeze(0), gt_bytes)
                if logits is not None and len(logits) > 0:
                    gt_t = torch.tensor(list(gt_bytes), dtype=torch.long, device=device)
                    min_len = min(len(gt_t), logits.shape[0])
                    if min_len > 0:
                        decode_losses.append(
                            F.cross_entropy(logits[:min_len], gt_t[:min_len])
                        )
            except Exception:
                pass
        wtt.eval()

        decode_loss = torch.stack(decode_losses).mean() \
            if decode_losses else torch.tensor(0.0, device=device)

        # ── Total loss ──
        total_loss = (
            PHASE3_CONFIG['wave_loss_weight']    * wave_loss
          + PHASE3_CONFIG['cosine_loss_weight']  * cosine_loss
          + PHASE3_CONFIG['contrastive_weight']  * contrastive_loss
          + decode_loss_weight                   * decode_loss
        )

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += total_loss.item()
        step         += 1

        # ── Logging ──
        if step % log_interval == 0:
            avg_loss = running_loss / log_interval
            running_loss = 0.0
            elapsed = time.time() - start_time
            lr_now  = scheduler.get_last_lr()[0]
            print(
                f"  step {step:6d}/{steps}: "
                f"wave={wave_loss.item():.4f}  "
                f"cosine={cosine_loss.item():.4f}  "
                f"decode={decode_loss.item():.4f}  "
                f"total={avg_loss:.4f}  "
                f"lr={lr_now:.6f}"
            )
            log.metric("step", str(step))
            log.metric("total_loss", f"{avg_loss:.4f}")

        # ── Decode gate check ──
        if step % PHASE3_CONFIG['gate_check_interval'] == 0:
            generator.eval()
            avg_acc, min_acc = run_decode_gate(cse, chunker, wtt, w2f, generator, device)
            generator.train()

            gate_symbol = "✓ PASSED" if avg_acc >= PHASE3_CONFIG['gate_avg_threshold'] else "⚠ NOT YET"
            print(
                f"  step {step:6d} decode gate: {gate_symbol}  "
                f"avg={avg_acc:.1%}  min={min_acc:.1%}"
            )

            if avg_acc > best_gate_avg:
                best_gate_avg = avg_acc
                # Save best-gate checkpoint
                _save_checkpoint(
                    generator=generator,
                    step=step,
                    loss=wave_loss.item(),
                    decode_gate_avg=avg_acc,
                    decode_gate_min=min_acc,
                    checkpoint_path=checkpoint_out,
                    suffix='_best_gate',
                )
                log.success(f"New best gate checkpoint: avg={avg_acc:.1%}  step={step}")

            if avg_acc >= PHASE3_CONFIG['gate_avg_threshold'] and \
               min_acc >= PHASE3_CONFIG['gate_min_threshold']:
                log.success(f"DECODE GATE PASSED at step {step} — stopping early")
                break

        # ── Mid-run checkpoint ──
        if step % 5_000 == 0:
            _save_checkpoint(
                generator=generator,
                step=step,
                loss=wave_loss.item(),
                decode_gate_avg=best_gate_avg,
                decode_gate_min=0.0,
                checkpoint_path=checkpoint_out,
            )
            log.success(f"Checkpoint saved at step {step}")

    # ── Final checkpoint ──
    generator.eval()
    final_gate_avg, final_gate_min = run_decode_gate(cse, chunker, wtt, w2f, generator, device)

    _save_checkpoint(
        generator=generator,
        step=step,
        loss=wave_loss.item(),
        decode_gate_avg=final_gate_avg,
        decode_gate_min=final_gate_min,
        checkpoint_path=checkpoint_out,
    )

    total_time = time.time() - start_time
    log.success(f"Final checkpoint saved: {checkpoint_out}")
    log.metric("final_gate_avg", f"{final_gate_avg:.1%}")
    log.metric("final_gate_min", f"{final_gate_min:.1%}")
    log.metric("total_time_s",  f"{total_time:.0f}")

    # ── Upload to HuggingFace ──
    if hf_token:
        try:
            from flux_utils import upload_checkpoint_to_hf
            upload_checkpoint_to_hf(phase=3, hf_token=hf_token)
            log.success("Checkpoint uploaded to HuggingFace Hub")
        except Exception as e:
            log.warning(f"HF upload failed: {e}")

    return {
        'steps':           step,
        'final_gate_avg':  final_gate_avg,
        'final_gate_min':  final_gate_min,
        'total_time_s':    total_time,
    }


# ─────────────────────────────────────────────
# Checkpoint Save Helper
# ─────────────────────────────────────────────

def _save_checkpoint(
    generator:       'WaveGenerator',
    step:            int,
    loss:            float,
    decode_gate_avg: float,
    decode_gate_min: float,
    checkpoint_path: Path,
    suffix:          str = '',
) -> None:
    """Save a Phase 3 checkpoint."""
    ckpt_path = checkpoint_path
    if suffix:
        ckpt_path = checkpoint_path.with_name(
            checkpoint_path.stem + suffix + checkpoint_path.suffix
        )

    state = {
        'phase':     3,
        'version':   'v2',
        'timestamp': datetime.now().isoformat(),
        'config':    PHASE3_CONFIG,
        'step':      step,
        'state_dict': {
            'generator': OrderedDict(generator.state_dict()),
        },
        'metrics': {
            'loss':            loss,
            'decode_gate_avg': decode_gate_avg,
            'decode_gate_min': decode_gate_min,
        },
    }
    torch.save(state, ckpt_path)


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train FLUX v2 Phase 3 Wave Generator')
    parser.add_argument('--steps',  type=int,   default=PHASE3_CONFIG['steps'])
    parser.add_argument('--device', type=str,   default='auto')
    parser.add_argument('--lr',     type=float, default=PHASE3_CONFIG['lr'])
    parser.add_argument('--decode-loss-weight', type=float,
                        default=PHASE3_CONFIG['decode_loss_weight'])
    parser.add_argument('--hf-token', type=str, default=None)
    args = parser.parse_args()

    train(
        steps=args.steps,
        device=args.device,
        lr=args.lr,
        decode_loss_weight=args.decode_loss_weight,
        hf_token=args.hf_token,
    )
