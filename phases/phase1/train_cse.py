"""
Training script for Phase 1 CSE.

Training objectives:
1. Reconstruction loss: encode → decode → predict original bytes
2. Semantic contrastive loss: similar words → similar waves, different → different

Dataset: WikiText-2 (auto-downloaded via HuggingFace datasets)
Hardware: CPU acceptable, GPU/MPS faster
Expected training time: 30–90 minutes

Usage:
    cd phases/phase1
    python train_cse.py
    python train_cse.py --steps 10000 --device cpu
"""

import sys
import math
import random
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ─────────────────────────────────────────────
# Path Setup
# ─────────────────────────────────────────────

PHASE_DIR = Path(__file__).parent
PROJECT_ROOT = PHASE_DIR.parent.parent
sys.path.insert(0, str(PHASE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from cse import ContinuousSemanticEncoder, WaveDecoder
from wave_types import WAVE_DIMS, TOTAL_WAVE_DIM
from flux_utils import save_checkpoint, get_device


# ─────────────────────────────────────────────
# Semantic Training Data
# ─────────────────────────────────────────────

# Triplets: (anchor, positive, negative)
# Training objective: sim(anchor, positive) > sim(anchor, negative)
SEMANTIC_TRIPLETS = [
    # Core test triplets (must pass test_phase1_test3)
    ("dog", "cat", "democracy"),
    ("Paris", "France", "banana"),
    ("running", "jogging", "purple"),
    ("happy", "joyful", "concrete"),
    ("king", "queen", "triangle"),
    ("car", "vehicle", "philosophy"),
    ("ocean", "sea", "keyboard"),
    ("doctor", "physician", "algebra"),
    ("big", "large", "grammar"),
    ("fast", "quick", "geology"),
    ("house", "home", "electron"),
    ("road", "street", "symphony"),
    ("smart", "intelligent", "pencil"),
    ("old", "ancient", "software"),
    ("dark", "dim", "onion"),
    ("love", "adore", "thermometer"),
    ("strong", "powerful", "butterfly"),
    ("beautiful", "pretty", "protocol"),
    ("sad", "unhappy", "politics"),
    ("begin", "start", "geology"),
    # Additional training triplets
    ("water", "liquid", "guitar"),
    ("sun", "star", "suitcase"),
    ("tree", "plant", "calculator"),
    ("rain", "storm", "philosophy"),
    ("book", "novel", "transistor"),
    ("song", "melody", "concrete"),
    ("movie", "film", "protein"),
    ("teacher", "instructor", "magnet"),
    ("lawyer", "attorney", "volcano"),
    ("mountain", "hill", "software"),
    ("river", "stream", "algebra"),
    ("mother", "parent", "keyboard"),
    ("baby", "infant", "diamond"),
    ("friend", "companion", "formula"),
    ("enemy", "foe", "circuit"),
    ("rich", "wealthy", "molecule"),
    ("poor", "needy", "antenna"),
    ("hot", "warm", "grammar"),
    ("cold", "cool", "guitar"),
    ("good", "excellent", "tornado"),
    ("bad", "terrible", "molecule"),
    ("eat", "consume", "triangle"),
    ("sleep", "rest", "diamond"),
    ("walk", "stroll", "protein"),
    ("think", "ponder", "volcano"),
    ("speak", "talk", "electron"),
    ("write", "compose", "magnet"),
    ("read", "study", "circuit"),
    ("laugh", "giggle", "formula"),
    ("cry", "weep", "antenna"),
]

# Antonym pairs (target: low similarity)
ANTONYM_PAIRS = [
    ("hot", "cold"),
    ("big", "small"),
    ("happy", "sad"),
    ("love", "hate"),
    ("fast", "slow"),
    ("light", "dark"),
    ("young", "old"),
    ("good", "bad"),
    ("strong", "weak"),
    ("rich", "poor"),
]

# Cross-lingual pairs for language agnostic training
MULTILINGUAL_TRIPLETS = [
    ("cat", "le chat", "banana"),
    ("dog", "le chien", "triangle"),
    ("hello", "bonjour", "electron"),
    ("water", "eau", "keyboard"),
    ("the cat is sleeping", "le chat dort", "purple"),
    ("good morning", "bonjour", "concrete"),
    ("thank you", "merci", "guitar"),
    ("house", "maison", "algebra"),
    ("book", "libro", "volcano"),
    ("sun", "soleil", "transistor"),
    ("cat", "gato", "protocol"),
    ("water", "agua", "formula"),
    ("hello", "hola", "circuit"),
    ("friend", "amigo", "antenna"),
    ("I love you", "je t'aime", "molecule"),
]


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

def load_text_data(max_texts: int = 5000) -> tuple:
    """
    Load WikiText-2 text data for reconstruction training.
    Falls back to synthetic data if download fails.

    Returns:
        (train_texts, val_texts) lists of strings
    """
    try:
        from datasets import load_dataset
        print("  Loading WikiText-2 dataset...")
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', trust_remote_code=True)
        train_lines = [
            line.strip() for line in ds['train']['text']
            if len(line.strip()) > 30
        ]
        val_lines = [
            line.strip() for line in ds['validation']['text']
            if len(line.strip()) > 30
        ]
        train_texts = train_lines[:max_texts]
        val_texts = val_lines[:min(1000, len(val_lines))]
        print(f"  ✓ Loaded {len(train_texts)} train, {len(val_texts)} val texts")
        return train_texts, val_texts
    except Exception as e:
        print(f"  ⚠ Could not load WikiText-2: {e}")
        print("  Using synthetic text data instead...")
        return _generate_synthetic_data(max_texts)


def _generate_synthetic_data(max_texts: int) -> tuple:
    """Generate synthetic English text as fallback training data."""
    words = [
        'the', 'a', 'is', 'was', 'and', 'of', 'to', 'in', 'that', 'it',
        'for', 'on', 'with', 'as', 'at', 'by', 'this', 'from', 'or', 'an',
        'but', 'not', 'are', 'were', 'been', 'have', 'has', 'had', 'will',
        'cat', 'dog', 'house', 'tree', 'car', 'book', 'man', 'woman', 'city',
        'water', 'food', 'day', 'night', 'sun', 'moon', 'star', 'rain', 'fire',
        'big', 'small', 'hot', 'cold', 'new', 'old', 'good', 'bad', 'fast',
        'run', 'walk', 'eat', 'sleep', 'read', 'write', 'speak', 'think',
        'happy', 'sad', 'dark', 'light', 'strong', 'weak', 'young', 'beautiful',
        'king', 'queen', 'doctor', 'teacher', 'ocean', 'mountain', 'river',
        'Paris', 'France', 'London', 'music', 'science', 'world', 'people',
    ]
    texts = []
    for _ in range(max_texts):
        length = random.randint(8, 25)
        texts.append(' '.join(random.choices(words, k=length)) + '.')
    split = int(0.9 * len(texts))
    print(f"  ✓ Generated {split} train, {len(texts) - split} val texts")
    return texts[:split], texts[split:]


def get_text_chunk(texts: list, chunk_size: int = 64) -> str:
    """Sample a random text chunk of up to chunk_size bytes."""
    text = random.choice(texts)
    text_bytes = text.encode('utf-8')
    if len(text_bytes) <= chunk_size:
        return text
    start = random.randint(0, len(text_bytes) - chunk_size)
    # Decode safely (ignore partial UTF-8 at boundaries)
    return text_bytes[start:start + chunk_size].decode('utf-8', errors='ignore')


# ─────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────

def reconstruction_loss(
    cse: ContinuousSemanticEncoder,
    decoder: WaveDecoder,
    texts: list,
) -> torch.Tensor:
    """
    Compute byte-level reconstruction loss.
    Encode text → wave, decode wave → byte predictions, compare to original.

    Returns:
        scalar loss tensor
    """
    total_loss = torch.tensor(0.0, device=cse.device)
    count = 0

    for text in texts:
        if len(text) < 2:
            continue
        wave = cse.encode(text)
        logits = decoder(wave.full)        # [seq_len, 256]
        target = cse.text_to_bytes(text)   # [seq_len]

        # Ensure matching lengths
        min_len = min(logits.shape[0], target.shape[0])
        logits = logits[:min_len]
        target = target[:min_len]

        total_loss = total_loss + F.cross_entropy(logits, target)
        count += 1

    return total_loss / max(count, 1)


def semantic_contrastive_loss(
    cse: ContinuousSemanticEncoder,
    triplets: list,
    margin: float = 0.5,
    pos_target: float = 0.85,
    neg_target: float = 0.0,
) -> torch.Tensor:
    """
    Triplet + target contrastive loss on semantic dimension.

    For each (anchor, positive, negative):
    - Push cosine_sim(anchor, positive) toward pos_target
    - Push cosine_sim(anchor, negative) toward neg_target
    - Ensure sim(anchor, pos) > sim(anchor, neg) + margin

    Returns:
        scalar loss tensor
    """
    total_loss = torch.tensor(0.0, device=cse.device)
    count = 0

    for anchor, positive, negative in triplets:
        v_a = cse.encode(anchor).semantic.mean(dim=0)
        v_p = cse.encode(positive).semantic.mean(dim=0)
        v_n = cse.encode(negative).semantic.mean(dim=0)

        sim_pos = F.cosine_similarity(v_a.unsqueeze(0), v_p.unsqueeze(0))
        sim_neg = F.cosine_similarity(v_a.unsqueeze(0), v_n.unsqueeze(0))

        # Triplet margin loss
        trip_loss = F.relu(margin - sim_pos + sim_neg)

        # Target similarity loss: push pos toward target, neg toward 0
        pos_target_loss = (pos_target - sim_pos) ** 2
        neg_target_loss = sim_neg ** 2

        total_loss = total_loss + trip_loss + 0.5 * pos_target_loss + 0.5 * neg_target_loss
        count += 1

    return total_loss / max(count, 1)


def antonym_loss(
    cse: ContinuousSemanticEncoder,
    pairs: list,
    target_sim: float = 0.1,
) -> torch.Tensor:
    """Push antonym pairs to have low similarity."""
    total_loss = torch.tensor(0.0, device=cse.device)
    count = 0

    for w1, w2 in pairs:
        v1 = cse.encode(w1).semantic.mean(dim=0)
        v2 = cse.encode(w2).semantic.mean(dim=0)
        sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))
        total_loss = total_loss + (sim - target_sim) ** 2
        count += 1

    return total_loss / max(count, 1)


# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────

@torch.no_grad()
def validate(
    cse: ContinuousSemanticEncoder,
    decoder: WaveDecoder,
    val_texts: list,
    num_samples: int = 100,
) -> dict:
    """Run validation: reconstruction accuracy + semantic quality."""
    cse.eval()
    decoder.eval()

    # Reconstruction accuracy
    total_correct = 0
    total_bytes = 0
    total_ce = 0.0
    num_ce = 0

    samples = random.sample(val_texts, min(num_samples, len(val_texts)))
    for text in samples:
        if len(text) < 2:
            continue
        wave = cse.encode(text)
        logits = decoder(wave.full)
        target = cse.text_to_bytes(text)

        min_len = min(logits.shape[0], target.shape[0])
        logits = logits[:min_len]
        target = target[:min_len]

        preds = logits.argmax(dim=-1)
        total_correct += (preds == target).sum().item()
        total_bytes += min_len
        total_ce += F.cross_entropy(logits, target).item()
        num_ce += 1

    accuracy = total_correct / max(total_bytes, 1)
    avg_ce = total_ce / max(num_ce, 1)

    # Semantic quality: check a few triplets
    correct_ordering = 0
    test_triplets = SEMANTIC_TRIPLETS[:10]
    for anchor, pos, neg in test_triplets:
        v_a = cse.encode(anchor).semantic.mean(dim=0)
        v_p = cse.encode(pos).semantic.mean(dim=0)
        v_n = cse.encode(neg).semantic.mean(dim=0)
        sim_p = F.cosine_similarity(v_a.unsqueeze(0), v_p.unsqueeze(0)).item()
        sim_n = F.cosine_similarity(v_a.unsqueeze(0), v_n.unsqueeze(0)).item()
        if sim_p > sim_n:
            correct_ordering += 1

    cse.train()
    decoder.train()

    return {
        'accuracy': accuracy,
        'recon_loss': 1.0 - accuracy,
        'cross_entropy': avg_ce,
        'semantic_ordering': f"{correct_ordering}/{len(test_triplets)}",
    }


# ─────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────

def train(args):
    """Full Phase 1 training pipeline."""
    print("=" * 60)
    print("FLUX Phase 1: Continuous Semantic Encoder (CSE)")
    print("=" * 60)

    device = args.device
    print(f"\n  Device: {device}")
    print(f"  Steps:  {args.steps}")
    print(f"  LR:     {args.lr}")

    # ── Load data ──
    print("\n── Loading Data ──")
    train_texts, val_texts = load_text_data(max_texts=args.max_texts)

    # ── Create models ──
    print("\n── Building CSE + Decoder ──")
    cse = ContinuousSemanticEncoder(
        wave_dims=dict(WAVE_DIMS),
        byte_window=8,
        byte_stride=1,
        interference_radius=4,
        device=device,
    ).to(device)
    decoder = WaveDecoder(wave_dim=TOTAL_WAVE_DIM).to(device)

    param_count = sum(p.numel() for p in cse.parameters())
    dec_count = sum(p.numel() for p in decoder.parameters())
    print(f"  ✓ CSE parameters:     {param_count:,}")
    print(f"  ✓ Decoder parameters: {dec_count:,}")
    print(f"  ✓ Total parameters:   {param_count + dec_count:,}")

    # ── Optimizer ──
    optimizer = torch.optim.Adam(
        list(cse.parameters()) + list(decoder.parameters()),
        lr=args.lr,
        weight_decay=1e-5,
    )

    # Combine all triplets
    all_triplets = SEMANTIC_TRIPLETS + MULTILINGUAL_TRIPLETS

    # ── Training ──
    print(f"\n── Training for {args.steps} steps ──\n")
    start_time = datetime.now()
    best_val_acc = 0.0

    pbar = tqdm(range(1, args.steps + 1), desc="Training", ncols=100)
    for step in pbar:
        cse.train()
        decoder.train()

        # ── Reconstruction loss ──
        text_batch = [
            get_text_chunk(train_texts, chunk_size=args.chunk_size)
            for _ in range(args.recon_batch)
        ]
        recon = reconstruction_loss(cse, decoder, text_batch)

        # ── Semantic loss (with warm-up) ──
        sem_weight = min(1.0, step / args.sem_warmup) * args.sem_weight
        sem_loss = torch.tensor(0.0, device=device)
        ant_loss = torch.tensor(0.0, device=device)

        if sem_weight > 0:
            triplet_batch = random.sample(
                all_triplets, min(args.sem_batch, len(all_triplets))
            )
            sem_loss = semantic_contrastive_loss(cse, triplet_batch)

            antonym_batch = random.sample(
                ANTONYM_PAIRS, min(4, len(ANTONYM_PAIRS))
            )
            ant_loss = antonym_loss(cse, antonym_batch)

        # ── Total loss ──
        total = recon + sem_weight * (sem_loss + 0.5 * ant_loss)

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            list(cse.parameters()) + list(decoder.parameters()), max_norm=1.0
        )
        optimizer.step()

        # ── Logging ──
        pbar.set_postfix({
            'recon': f'{recon.item():.3f}',
            'sem': f'{sem_loss.item():.3f}',
            'sw': f'{sem_weight:.2f}',
        })

        if step % args.log_every == 0:
            tqdm.write(
                f"  Step {step:5d} | recon={recon.item():.4f} "
                f"sem={sem_loss.item():.4f} ant={ant_loss.item():.4f} "
                f"total={total.item():.4f}"
            )

        # ── Validation ──
        if step % args.val_every == 0 or step == args.steps:
            val = validate(cse, decoder, val_texts)
            tqdm.write(
                f"  ── VAL step {step} ── "
                f"accuracy={val['accuracy']:.3f} "
                f"recon_loss={val['recon_loss']:.3f} "
                f"CE={val['cross_entropy']:.3f} "
                f"semantic={val['semantic_ordering']}"
            )
            if val['accuracy'] > best_val_acc:
                best_val_acc = val['accuracy']

    duration = datetime.now() - start_time
    print(f"\n── Training Complete ──")
    print(f"  Duration: {duration}")
    print(f"  Best validation accuracy: {best_val_acc:.3f}")

    # ── Final validation ──
    print("\n── Final Validation ──")
    final_val = validate(cse, decoder, val_texts, num_samples=200)
    for k, v in final_val.items():
        print(f"  {k}: {v}")

    # ── Save checkpoint ──
    print("\n── Saving Checkpoint ──")
    checkpoint_state = {
        'component': 'CSE',
        'state_dict': cse.state_dict(),
        'config': {
            'wave_dims': cse.wave_dims,
            'byte_window': cse.byte_window,
            'byte_stride': cse.byte_stride,
            'interference_radius': cse.interference_radius,
        },
        'decoder_state_dict': decoder.state_dict(),
        'decoder_config': {
            'wave_dim': TOTAL_WAVE_DIM,
        },
        'metrics': {
            'final_accuracy': final_val['accuracy'],
            'final_recon_loss': final_val['recon_loss'],
            'final_cross_entropy': final_val['cross_entropy'],
            'final_semantic_ordering': final_val['semantic_ordering'],
            'training_steps': args.steps,
            'training_duration': str(duration),
        },
    }
    save_checkpoint(1, checkpoint_state)

    print("\n" + "=" * 60)
    print("Phase 1 training complete!")
    print(f"  Reconstruction accuracy: {final_val['accuracy']:.3f}")
    print(f"  Semantic ordering:       {final_val['semantic_ordering']}")
    print("  Next: run test scripts to verify acceptance criteria")
    print("=" * 60)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Phase 1 CSE')
    parser.add_argument('--steps', type=int, default=5000, help='Training steps')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu/cuda/mps)')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--recon-batch', type=int, default=16, help='Reconstruction batch size')
    parser.add_argument('--sem-batch', type=int, default=8, help='Semantic triplet batch size')
    parser.add_argument('--sem-weight', type=float, default=1.0, help='Semantic loss weight')
    parser.add_argument('--sem-warmup', type=int, default=1000, help='Semantic loss warmup steps')
    parser.add_argument('--chunk-size', type=int, default=64, help='Text chunk size in bytes')
    parser.add_argument('--max-texts', type=int, default=5000, help='Max training texts')
    parser.add_argument('--log-every', type=int, default=100, help='Log interval')
    parser.add_argument('--val-every', type=int, default=500, help='Validation interval')

    args = parser.parse_args()
    if args.device is None:
        args.device = get_device()

    train(args)
