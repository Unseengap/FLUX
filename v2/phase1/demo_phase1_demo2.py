"""
demo_phase1_demo2.py — Wave Interference Patterns + Semantic Similarity

Shows how wave interference creates semantic structure:
  1. Similar words → constructive interference → similar waves
  2. Opposite words → destructive interference → dissimilar waves
  3. The similarity is preserved through chunking and decoding

Run: python demo_phase1_demo2.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F


WORD_GROUPS = [
    {
        'label': 'Animals',
        'words': ['cat', 'kitten', 'dog', 'puppy', 'bird', 'elephant'],
    },
    {
        'label': 'Emotions',
        'words': ['happy', 'joyful', 'sad', 'melancholy', 'angry', 'furious'],
    },
    {
        'label': 'Opposites',
        'words': ['hot', 'cold', 'fast', 'slow', 'light', 'dark'],
    },
    {
        'label': 'Science',
        'words': ['quantum', 'photon', 'entropy', 'gravity', 'momentum'],
    },
]

INTERFERENCE_DEMO = [
    # (base_text, modifier_text, expected_effect)
    ("The cat sat on the mat", "The dog sat on the mat", "SIMILAR — same structure, one word changed"),
    ("I love this book", "I hate this book", "DIFFERENT — opposite sentiment"),
    ("Apple iPhone", "Samsung Galaxy", "SIMILAR — both phone products"),
    ("Python code", "binary stars", "DIFFERENT — unrelated domains"),
]


def print_bar(width: int = 60, char: str = '─'):
    print(char * width)


def cosine_sim(v1: torch.Tensor, v2: torch.Tensor) -> float:
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()


@torch.no_grad()
def run_demo():
    print()
    print_bar(60, '═')
    print("  FLUX v2 Phase 1 — Wave Interference + Semantic Similarity")
    print_bar(60, '═')

    # ── Load checkpoint ───────────────────────────────────────────
    try:
        from train_codec import load_phase1_checkpoint
        codec = load_phase1_checkpoint(device='cpu')
        print("  ✓ Phase 1 v2 checkpoint loaded\n")
    except FileNotFoundError:
        print("  ⚠ No checkpoint — UNTRAINED model (similarity not meaningful yet)")
        print("  Train first: python train_codec.py --steps 30000\n")
        from train_codec import WaveCodec
        codec = WaveCodec(device='cpu')

    cse = codec.cse
    chunker = codec.chunker

    # ── Part 1: Word groups ────────────────────────────────────────
    print_bar()
    print("  Part 1 — Semantic Similarity Within Word Groups")
    print_bar()
    print("  (Within-group cosine should be higher than cross-group)")
    print()

    group_vectors = {}
    for group in WORD_GROUPS:
        vecs = []
        for word in group['words']:
            wave = cse.encode(word)
            v = wave.full.mean(dim=0)
            vecs.append(v)
        group_vectors[group['label']] = (group['words'], vecs)

        # Within-group average similarity
        sims = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sims.append(cosine_sim(vecs[i], vecs[j]))
        avg_sim = sum(sims) / len(sims) if sims else 0.0
        print(f"  [{group['label']:12s}]  avg cosine = {avg_sim:.3f}  words: {group['words']}")

    print()

    # ── Part 2: Cross-group separation ─────────────────────────────
    print_bar()
    print("  Part 2 — Cross-Group Separation")
    print_bar()
    print("  (Cross-group cosine should be lower than within-group)")
    print()

    labels = list(group_vectors.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            words_i, vecs_i = group_vectors[labels[i]]
            words_j, vecs_j = group_vectors[labels[j]]

            cross_sims = []
            for vi in vecs_i:
                for vj in vecs_j:
                    cross_sims.append(cosine_sim(vi, vj))

            avg_cross = sum(cross_sims) / len(cross_sims)
            print(f"  [{labels[i]:12s}] ↔ [{labels[j]:12s}]  avg cross-cosine = {avg_cross:.3f}")

    print()

    # ── Part 3: Interference demo ──────────────────────────────────
    print_bar()
    print("  Part 3 — Interference Effects on Full Sentences")
    print_bar()
    print()

    for text1, text2, description in INTERFERENCE_DEMO:
        w1 = cse.encode(text1).full.mean(dim=0)
        w2 = cse.encode(text2).full.mean(dim=0)
        sim = cosine_sim(w1, w2)

        expected = 'SIMILAR' in description
        actual_sim = sim >= 0.5
        match_expectation = expected == actual_sim
        marker = '✓' if match_expectation else '⚠'

        print(f"  {marker} sim = {sim:.3f}  [{description}]")
        print(f"    '{text1}'")
        print(f"    '{text2}'")
        print()

    # ── Part 4: Wave shape inspection ─────────────────────────────
    print_bar()
    print("  Part 4 — Wave Dimension Analysis")
    print_bar()

    inspect_text = "The transformation of information through wave physics"
    wave = cse.encode(inspect_text)

    print(f"  Text: '{inspect_text}'")
    print(f"  Full wave shape: {list(wave.full.shape)}")
    print(f"  Phonetic  [64]:  mean={wave.phonetic.mean():.3f}   std={wave.phonetic.std():.3f}")
    print(f"  Syntactic [64]:  mean={wave.syntactic.mean():.3f}   std={wave.syntactic.std():.3f}")
    print(f"  Semantic  [256]: mean={wave.semantic.mean():.3f}   std={wave.semantic.std():.3f}")
    print(f"  Temporal  [32]:  mean={wave.temporal.mean():.3f}   std={wave.temporal.std():.3f}")
    print(f"  Intensity [16]:  mean={wave.intensity.mean():.3f}   std={wave.intensity.std():.3f}")

    # Chunk
    chunks, spans = chunker(wave.full)
    print(f"\n  WaveChunker output:")
    print(f"  Chunks: {chunks.shape[0]}  spans: {spans[:5]}")
    print(f"  Chunk wave mean norm: {chunks.norm(dim=-1).mean():.3f}")

    print()
    print_bar(60, '═')
    print("  Demo complete.")
    print_bar(60, '═')


if __name__ == '__main__':
    run_demo()
