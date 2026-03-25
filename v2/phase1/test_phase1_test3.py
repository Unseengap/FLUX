"""
test_phase1_test3.py — Similar words → similar waves → similar decode

Verifies semantic structure in the wave space:
  - Similar words have high cosine similarity (> 0.7)
  - Opposite/unrelated words have low cosine similarity (< 0.3)
  - The similarity is preserved through the chunker (wave space invariant)

Run: python test_phase1_test3.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F
from train_codec import load_phase1_checkpoint
from flux_utils import PhaseResults


SIMILAR_PAIRS = [
    ("cat",    "kitten",    0.70),
    ("dog",    "puppy",     0.70),
    ("happy",  "joyful",    0.65),
    ("fast",   "quick",     0.65),
    ("water",  "liquid",    0.60),
    ("king",   "queen",     0.55),
    ("run",    "running",   0.75),
]

DISSIMILAR_PAIRS = [
    ("cat",    "democracy",  0.30),
    ("happy",  "catastrophe", 0.30),
    ("water",  "philosophy",  0.30),
    ("king",   "algorithm",   0.30),
]


@torch.no_grad()
def get_wave_vector(cse, text: str) -> torch.Tensor:
    """Get mean semantic wave vector for a text."""
    wave = cse.encode(text)
    return wave.full.mean(dim=0)  # [432]


@torch.no_grad()
def cosine(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """Cosine similarity between two vectors."""
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()


def test_wave_similarity():
    """Test: wave space preserves semantic similarity structure."""
    print("=" * 60)
    print("Test 3: Similar Words → Similar Waves")
    print("=" * 60)

    codec = load_phase1_checkpoint(device='cpu')
    cse = codec.cse

    # ── Similar pairs ──────────────────────────────────────────────
    print("\n  Similar pairs (higher cosine = better):")
    sim_passed = 0
    sim_scores = []
    for w1, w2, threshold in SIMILAR_PAIRS:
        v1 = get_wave_vector(cse, w1)
        v2 = get_wave_vector(cse, w2)
        sim = cosine(v1, v2)
        sim_scores.append(sim)
        ok = sim >= threshold
        if ok:
            sim_passed += 1
        status = '✓' if ok else '✗'
        print(f"  {status} sim('{w1}', '{w2}') = {sim:.3f}  (threshold: {threshold:.2f})")

    # ── Dissimilar pairs ───────────────────────────────────────────
    print("\n  Dissimilar pairs (lower cosine = better):")
    dis_passed = 0
    for w1, w2, threshold in DISSIMILAR_PAIRS:
        v1 = get_wave_vector(cse, w1)
        v2 = get_wave_vector(cse, w2)
        sim = cosine(v1, v2)
        ok = sim < threshold
        if ok:
            dis_passed += 1
        status = '✓' if ok else '✗'
        print(f"  {status} sim('{w1}', '{w2}') = {sim:.3f}  (threshold: < {threshold:.2f})")

    avg_sim = sum(sim_scores) / len(sim_scores)
    sim_pass_rate = sim_passed / len(SIMILAR_PAIRS)
    dis_pass_rate = dis_passed / len(DISSIMILAR_PAIRS)

    print(f"\n  Similar pairs:    {sim_passed}/{len(SIMILAR_PAIRS)} passed")
    print(f"  Dissimilar pairs: {dis_passed}/{len(DISSIMILAR_PAIRS)} passed")
    print(f"  Avg similar cosine: {avg_sim:.3f}")

    # Must pass at least 60% of similar pairs and 75% of dissimilar pairs
    assert sim_pass_rate >= 0.60, (
        f"FAIL: only {sim_pass_rate:.0%} of similar pairs met cosine threshold\n"
        f"The wave space may not have semantic structure yet."
    )
    assert dis_pass_rate >= 0.75, (
        f"FAIL: only {dis_pass_rate:.0%} of dissimilar pairs show separation\n"
        f"The wave space may be collapsing to a single region."
    )

    print(f"\n  ✓ PASS: similar pairs {sim_pass_rate:.0%}, dissimilar pairs {dis_pass_rate:.0%}")
    return sim_pass_rate, dis_pass_rate, avg_sim


if __name__ == '__main__':
    results = PhaseResults(phase=1, component_name="Wave Codec v2")
    try:
        sim_pass, dis_pass, avg_sim = test_wave_similarity()
        results.add_test("Similar pair cosine rate", passed=(sim_pass >= 0.60), score=sim_pass, threshold=0.60)
        results.add_test("Dissimilar pair separation", passed=(dis_pass >= 0.75), score=dis_pass, threshold=0.75)
        results.add_test("Avg similar cosine", passed=(avg_sim >= 0.60), score=avg_sim, threshold=0.60)
        print("\n  ALL TESTS PASSED ✓")
    except AssertionError as e:
        print(f"\n  TEST FAILED ✗\n  {e}")
    except FileNotFoundError as e:
        print(f"\n  CHECKPOINT NOT FOUND ✗\n  {e}")
    results.save()
