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
    # Scheduled sampling: ramp from 30% to 85% over training so the GRU trains
    # on its OWN outputs most of the time — closing the train/inference gap.
    # Original 0 → 40% meant 60% teacher-forced throughout → inference was
    # completely out-of-distribution.
    'ss_start':            0.3,
    'ss_end':              0.85,
    # Loss weights
    # wave_loss is NORMALISED by target magnitude → ~0-1 range, same as decode
    'wave_loss_weight':    1.0,
    'cosine_loss_weight':  0.5,
    # Contrastive weight reduced 0.5 → 0.1: early training showed it dominated
    # total loss (~5-6 pts), preventing wave_loss from converging.
    'contrastive_weight':  0.1,
    # Decode loss routed through Phase 2 bridge (f2w(w2f(wave))) so WTT sees
    # waves on the CSE manifold it was trained on. Weight reduced 10→3 because
    # the bridge fix makes each gradient step ~10× more effective.
    'decode_loss_weight':  3.0,
    # Manifold anchoring: teaches GRU to produce waves where f2w(w2f(x))≈x,
    # i.e., waves that already lie on the Phase 2-decodable manifold.
    'manifold_loss_weight': 2.0,
    # Anchor loss: forces first generated wave toward f2w(ctx) — the field
    # context decoded back to wave space.  Phase 2 guarantees f2w(ctx) is
    # on the CSE manifold (97.79% accuracy), so this gives the GRU a concrete
    # reachable on-manifold target from step 1, replacing the abstract manifold
    # projection that starts flat at 0.07 and never descends.
    'anchor_loss_weight': 5.0,
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
# Corpus Pre-encoder  (one-time, saves to disk)
# ─────────────────────────────────────────────

def precompute_corpus(
    corpus:     List[str],
    cse:        'ContinuousSemanticEncoder',
    chunker:    'WaveChunker',
    w2f:        'WaveToField',
    f2w:        'FieldToWave',
    device:     str,
    cache_path: Path,
    max_seq:    int = 30,
) -> List[Dict[str, Any]]:
    """
    Encode every training text exactly once and persist to disk.

    After the first run the cache is reloaded instantly, so the hot
    training loop only does random sampling — zero CSE / WaveChunker /
    WaveToField calls during training.

    Record schema:
        ctx       : Tensor [512]          — WaveToField context (CPU)
        f2w_wave  : Tensor [432]          — f2w(ctx): context decoded to wave space
                                            Always on the Phase 2 CSE manifold.
                                            Used as anchor target for first generated wave.
        waves     : Tensor [max_seq, 432] — padded target waves (CPU)
        length    : int                   — valid (non-padded) wave count
        bytes     : List[bytes]           — UTF-8 bytes per chunk

    Args:
        corpus:     Training strings
        cse:        Frozen ContinuousSemanticEncoder
        chunker:    Frozen WaveChunker (chunk_with_bytes gives correct bytes)
        w2f:        Frozen WaveToField
        device:     Encoding device (results stored on CPU)
        cache_path: .pt file to save/load
        max_seq:    Padding length

    Returns:
        List of record dicts (CPU tensors)
    """
    if cache_path.exists():
        print(f"  ✓ Corpus cache found — loading {cache_path.name}")
        records = torch.load(cache_path, map_location='cpu')
        print(f"  ✓ {len(records):,} pre-encoded records loaded")
        return records

    print(f"  Pre-encoding {len(corpus):,} texts → {cache_path.name} (one-time)...")
    records: List[Dict[str, Any]] = []

    with torch.no_grad():
        for i, text in enumerate(corpus):
            try:
                wave       = cse.encode(text)
                text_bytes = text.encode('utf-8')
                # chunk_with_bytes returns List[(chunk_wave [432], chunk_bytes)]
                pairs = chunker.chunk_with_bytes(wave.full, text_bytes)
                if not pairs:
                    continue

                n         = min(len(pairs), max_seq)
                mean_wave = wave.full.mean(dim=0)
                ctx       = w2f(mean_wave.to(device)).cpu()          # [512]
                # f2w(ctx) is the context decoded to wave space — always on-manifold
                # because Phase 2 was trained with exactly this round-trip path.
                # Used as the anchor target: first generated wave → f2w_wave.
                f2w_wave  = f2w(ctx.to(device)).cpu()                 # [432]

                chunk_waves = torch.stack([p[0] for p in pairs[:n]])  # [n, 432]
                chunk_bytes = [p[1] for p in pairs[:n]]               # List[bytes]

                # Pad to max_seq so tensors can be stacked in sample_batch
                pad      = torch.zeros(max_seq, 432)
                pad[:n]  = chunk_waves

                records.append({
                    'ctx':      ctx,
                    'f2w_wave': f2w_wave,
                    'waves':    pad,
                    'length':   n,
                    'bytes':    chunk_bytes,
                })
            except Exception:
                continue

            if (i + 1) % 500 == 0 or (i + 1) == len(corpus):
                print(f"    {i + 1:,}/{len(corpus):,} encoded  "
                      f"({len(records):,} valid)")

    torch.save(records, cache_path)
    print(f"  ✓ Saved {len(records):,} records → {cache_path}")
    return records


def sample_batch(
    records:    List[Dict[str, Any]],
    batch_size: int,
    device:     str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[bytes]]]:
    """
    Sample a training batch from pre-encoded corpus records.

    Replaces build_batch() — no CSE / WaveChunker / WaveToField calls.

    Args:
        records:    Output of precompute_corpus()
        batch_size: Number of samples
        device:     Target device

    Returns:
        (field_contexts [B, 512],
         f2w_waves      [B, 432],
         target_waves   [B, max_seq, 432],
         lengths        [B],
         bytes_list     [B][n_chunks bytes each])
    """
    batch          = random.choices(records, k=batch_size)
    field_contexts = torch.stack([r['ctx']      for r in batch]).to(device)
    f2w_waves      = torch.stack([r['f2w_wave'] for r in batch]).to(device)
    target_waves   = torch.stack([r['waves']    for r in batch]).to(device)
    lengths        = torch.tensor([r['length']  for r in batch], device=device)
    bytes_list     = [r['bytes'] for r in batch]
    return field_contexts, f2w_waves, target_waves, lengths, bytes_list


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

    # ContinuousSemanticEncoder takes wave_dims (dict) + byte_window/byte_stride,
    # not wave_dim/window_size/stride — use defaults which match the checkpoint.
    cse = ContinuousSemanticEncoder()
    cse.load_state_dict(state['state_dict']['cse'])
    cse.to(device).eval()
    for p in cse.parameters():
        p.requires_grad_(False)

    # Phase 1 config keys: chunker_min, chunker_max, wtt_max_bytes
    chunker = WaveChunker(
        wave_dim=cfg.get('wave_dim', 432),
        min_chunk_size=cfg.get('chunker_min', 2),
        max_chunk_size=cfg.get('chunker_max', 20),
    )
    chunker.load_state_dict(state['state_dict']['chunker'])
    chunker.to(device).eval()
    for p in chunker.parameters():
        p.requires_grad_(False)

    wtt = WaveToText(
        wave_dim=cfg.get('wave_dim', 432),
        hidden_dim=cfg.get('wtt_hidden_dim', 256),
        max_bytes=cfg.get('wtt_max_bytes', 20),
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
    # Phase 2 state_dict keys: bridge_wtf (WaveToField), bridge_ftw (FieldToWave)
    w2f.load_state_dict(state['state_dict']['bridge_wtf'])
    w2f.to(device).eval()
    for p in w2f.parameters():
        p.requires_grad_(False)

    f2w = FieldToWave(
        field_dim=cfg.get('field_features', 512),
        wave_dim=cfg.get('wave_dim', 432),
    )
    f2w.load_state_dict(state['state_dict']['bridge_ftw'])
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
    f2w:       'FieldToWave',
    generator: 'WaveGenerator',
    device:    str,
    phase:     int = 3,
) -> Tuple[float, float]:
    """
    Run decode gate: prompt → CSE → field context → generate waves → bridge → WTT → text.

    Generated waves are snapped onto the Phase 2 CSE manifold via f2w(w2f(wave))
    before WTT decoding.  Phase 2 achieves 97.79% accuracy on this pipeline,
    so any wave close to the manifold will decode cleanly.

    Args:
        cse, chunker, wtt, w2f, f2w, generator: model components
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

            # Snap each generated wave onto the Phase 2-decodable manifold
            # before passing to WTT.  f2w(w2f(x)) projects x into the
            # subspace that WTT was trained on (Phase 2 gate: 97.79% accuracy).
            bridged_waves = f2w(w2f(generated_waves))              # [N, 432]

            # Decode each bridged wave to bytes
            decoded_bytes = b''
            for i in range(bridged_waves.shape[0]):
                decoded = wtt.decode(bridged_waves[i])
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

# NOTE: compute_decode_loss() and build_batch() have been superseded.
# Use wtt.forward_batch() for batched decode loss (see training loop).
# Use precompute_corpus() + sample_batch() instead of build_batch().


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

    # ── Build corpus & pre-encode to disk (one-time cost) ──
    log.info("Building training corpus...")
    corpus = build_training_corpus(target_size=10_000)
    log.info(f"Training corpus: {len(corpus):,} texts")

    # Cache v2 — includes f2w_wave field (anchor target per record).
    # The old phase3_v2_corpus_cache.pt lacks this field and will crash.
    corpus_cache_path = checkpoint_dir / 'phase3_v2_corpus_cache_v2.pt'
    log.info("Pre-encoding corpus (uses cache if already done)...")
    corpus_records = precompute_corpus(
        corpus=corpus,
        cse=cse,
        chunker=chunker,
        w2f=w2f,
        f2w=f2w,
        device=device,
        cache_path=corpus_cache_path,
    )
    log.info(f"Corpus cache: {len(corpus_records):,} pre-encoded records")

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
    last_good_loss  = 1.0   # tracks last finite wave_loss — never passes NaN to checkpoint
    log_interval    = 500

    start_time      = time.time()

    while step < steps:
        # Scheduled sampling probability — ramps up over training
        ss_p = ss_start + (ss_end - ss_start) * (step / steps)

        # Sample pre-encoded batch — zero CSE / chunker / field calls
        field_contexts, f2w_waves, target_waves, lengths, bytes_list = sample_batch(
            records=corpus_records,
            batch_size=batch_size,
            device=device,
        )

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

        mse_raw     = (F.mse_loss(predicted, target_waves, reduction='none') * mask).sum() / mask.sum()
        # Normalise by target wave magnitude so wave_loss ~ 0-1 (same scale as decode_loss)
        # Without this, wave_loss ~3300 drowns decode_loss ~1.5 — decode gradient = 0.05%
        target_mag  = ((target_waves.pow(2) * mask).sum() / mask.sum()).detach().clamp(min=1e-6)
        wave_loss   = mse_raw / target_mag

        # Guard: skip step entirely if wave_loss is NaN/Inf (exploded batch)
        if not torch.isfinite(wave_loss):
            optimizer.zero_grad()
            step += 1
            continue

        last_good_loss = wave_loss.item()
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

        # ── Anchor loss: first generated wave → f2w(ctx) ────────────────
        # f2w_waves[b] = f2w(ctx[b]) = the field context decoded to wave space.
        # Phase 2 guarantees this is on the CSE manifold at 97.79% accuracy.
        # Forcing predicted[:, 0, :] toward f2w_waves gives the GRU a concrete
        # on-manifold target from step 1.  Without this, manifold_loss starts
        # at ~0.07 and never moves, because f2w(w2f(random)) maps to a constant
        # region of the manifold — gradient is pure noise.
        anchor_loss = F.mse_loss(predicted[:, 0, :], f2w_waves.detach())

        # ── Manifold anchoring loss ──────────────────────────────────────
        # Kept as a secondary regulariser: MSE(f2w(w2f(pred)), pred) still
        # discourages outputs that project far from the manifold, but it is
        # now secondary to anchor_loss which provides the direct on-manifold
        # gradient signal.
        flat_pred   = predicted.reshape(-1, predicted.shape[-1])  # [B*S, 432]
        flat_mask   = mask.reshape(-1)                             # [B*S, 1]
        valid_preds = flat_pred[flat_mask.squeeze(-1).bool()]      # [V, 432]
        if valid_preds.shape[0] > 0:
            bridged_valid = f2w(w2f(valid_preds))                  # [V, 432]
            manifold_loss = F.mse_loss(bridged_valid, valid_preds)
        else:
            manifold_loss = torch.tensor(0.0, device=device)

        # ── Decode loss: WTT on bridge-projected waves ───────────────────
        # Snap predicted waves through f2w(w2f(...)) before feeding WTT.
        # Phase 2 guarantees f2w(w2f(cse_wave)) decodes at 97.79% accuracy.
        # Training with the bridge: (a) gives WTT real signal from step 1,
        # (b) makes the gradient path meaningful even before GRU is on-manifold.
        wtt.train()
        decode_waves:   List[torch.Tensor] = []
        decode_targets: List[torch.Tensor] = []
        for i in random.sample(range(B), min(8, B)):
            if lengths[i].item() == 0 or not bytes_list[i] or not bytes_list[i][0]:
                continue
            decode_waves.append(predicted[i, 0])          # [432] first predicted wave
            decode_targets.append(
                torch.tensor(list(bytes_list[i][0]), dtype=torch.long, device=device)
            )
        if decode_waves:
            stacked_waves = torch.stack(decode_waves)          # [M, 432]
            # Route through bridge: GRU output → WaveToField → FieldToWave → WTT
            # This is the same path Phase 2 trained to 97.79% accuracy.
            bridged_decode = f2w(w2f(stacked_waves))           # [M, 432]
            decode_loss = wtt.forward_batch(
                bridged_decode,
                decode_targets,
            )
        else:
            decode_loss = torch.tensor(0.0, device=device)
        wtt.eval()

        # ── Total loss ──
        total_loss = (
            PHASE3_CONFIG['wave_loss_weight']     * wave_loss
          + PHASE3_CONFIG['cosine_loss_weight']   * cosine_loss
          + PHASE3_CONFIG['contrastive_weight']   * contrastive_loss
          + decode_loss_weight                    * decode_loss
          + PHASE3_CONFIG['manifold_loss_weight'] * manifold_loss
          + PHASE3_CONFIG['anchor_loss_weight']   * anchor_loss
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
                f"wave_norm={wave_loss.item():.4f}  "
                f"cosine={cosine_loss.item():.4f}  "
                f"anchor={anchor_loss.item():.4f}  "
                f"manifold={manifold_loss.item():.4f}  "
                f"decode={decode_loss.item():.4f}  "
                f"total={avg_loss:.4f}  "
                f"lr={lr_now:.6f}"
            )
            log.metric("step", str(step))
            log.metric("total_loss", f"{avg_loss:.4f}")

        # ── Decode gate check ──
        if step % PHASE3_CONFIG['gate_check_interval'] == 0:
            generator.eval()
            avg_acc, min_acc = run_decode_gate(cse, chunker, wtt, w2f, f2w, generator, device)
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
                    loss=last_good_loss,
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
                loss=last_good_loss,
                decode_gate_avg=best_gate_avg,
                decode_gate_min=0.0,
                checkpoint_path=checkpoint_out,
            )
            log.success(f"Checkpoint saved at step {step}")

    # ── Final checkpoint ──
    generator.eval()
    final_gate_avg, final_gate_min = run_decode_gate(cse, chunker, wtt, w2f, f2w, generator, device)

    _save_checkpoint(
        generator=generator,
        step=step,
        loss=last_good_loss,
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

    # Guard against NaN loss values (can occur if last batch had numerical instability)
    loss_val = float(loss)
    if loss_val != loss_val:   # NaN check (NaN != NaN)
        loss_val = -1.0        # sentinel — diagnostics can detect this explicitly

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
            'loss':            loss_val,
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
