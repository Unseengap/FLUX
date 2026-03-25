"""
test_phase2_5_test2.py — Test: Different contexts produce different predictions

Loads Phase 2.5 checkpoint from HuggingFace.
Feeds different prefix contexts to the WRU and verifies:
  1. Different prompts → different predicted waves (cosine < 0.90)
  2. Same prompt → same predicted wave (deterministic)

This tests that the WRU is NOT collapsing all contexts to the same output.

Threshold: avg pairwise cosine between different contexts < 0.90
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase2'))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F
from flux_utils import PhaseResults

from train_wru import load_phase1_and_phase2, load_phase2_5_checkpoint


DIVERSITY_TEXTS = [
    "The cat sat on the mat",
    "Quantum mechanics describes",
    "def fibonacci(n):",
    "Water freezes at zero degrees",
    "The stock market crashed",
    "Photosynthesis converts sunlight",
]


@torch.no_grad()
def main():
    print("\n" + "=" * 60)
    print("  TEST 2: Different contexts → different predictions")
    print("=" * 60)

    device = 'cpu'
    hf_token = os.environ.get('HF_TOKEN', '')

    # Load components
    print("\n  Loading Phase 1 + 2 (frozen)...")
    components = load_phase1_and_phase2(device=device, hf_token=hf_token)
    cse = components['cse']
    chunker = components['chunker']
    w2f = components['w2f']

    print("  Loading Phase 2.5 WRU checkpoint...")
    wru = load_phase2_5_checkpoint(device=device, hf_token=hf_token)
    wru.eval()

    # Generate predictions from different contexts
    predictions = []
    for text in DIVERSITY_TEXTS:
        wave = cse.encode(text)
        chunk_waves, _ = chunker(wave.full)

        if chunk_waves.shape[0] < 2:
            continue

        n = max(1, chunk_waves.shape[0] // 2)
        prefix_mean = chunk_waves[:n].mean(dim=0)
        ctx = w2f(prefix_mean).unsqueeze(0)  # [1, 512]

        pred, _ = wru(ctx)
        predictions.append((text, pred.squeeze(0)))

    # ── Check diversity: pairwise cosine between different contexts ────
    print(f"\n  Pairwise cosine similarities:")
    cosines = []
    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            cos = F.cosine_similarity(
                predictions[i][1].unsqueeze(0),
                predictions[j][1].unsqueeze(0),
                dim=-1,
            ).item()
            cosines.append(cos)
            t1 = predictions[i][0][:25]
            t2 = predictions[j][0][:25]
            status = '✓' if cos < 0.90 else '✗'
            print(f"  {status} cos({t1}, {t2}) = {cos:.4f}")

    avg_cos = sum(cosines) / max(len(cosines), 1)
    max_cos = max(cosines) if cosines else 1.0
    print(f"\n  Avg pairwise cosine: {avg_cos:.4f}  (want < 0.90)")
    print(f"  Max pairwise cosine: {max_cos:.4f}  (want < 0.95)")

    # ── Check determinism: same input → same output ───────────────────
    print(f"\n  Determinism check:")
    text = DIVERSITY_TEXTS[0]
    wave = cse.encode(text)
    chunk_waves, _ = chunker(wave.full)
    n = max(1, chunk_waves.shape[0] // 2)
    ctx = w2f(chunk_waves[:n].mean(dim=0)).unsqueeze(0)

    pred1, _ = wru(ctx)
    pred2, _ = wru(ctx)
    self_cos = F.cosine_similarity(pred1, pred2, dim=-1).item()
    print(f"  Same input cosine: {self_cos:.6f}  (want = 1.0)")

    # ── Record results ────────────────────────────────────────────────
    diversity_pass = avg_cos < 0.90
    determinism_pass = self_cos > 0.999

    results = PhaseResults(phase=2, component_name="WRU Context Diversity (Phase 2.5)")
    results.add_test(
        "Avg Pairwise Cosine (different contexts)",
        passed=diversity_pass,
        score=avg_cos,
        threshold=0.90,
    )
    results.add_test(
        "Determinism (same input → same output)",
        passed=determinism_pass,
        score=self_cos,
        threshold=0.999,
    )
    results.save()

    assert diversity_pass, f"FAILED: avg cosine {avg_cos:.4f} ≥ 0.90 — context collapse!"
    assert determinism_pass, f"FAILED: self-cosine {self_cos:.6f} < 0.999 — non-deterministic!"
    print("  ✓ TEST 2 PASSED\n")


if __name__ == '__main__':
    main()
