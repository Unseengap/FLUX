"""
PHASE 1 TEST 3: Semantic proximity ordering.

Test that: sim(word_A, word_B) > sim(word_A, word_C)
when word_B is semantically closer to word_A than word_C.

20 test triples. 18/20 must pass.

Run: python test_phase1_test3.py
"""

import sys
from pathlib import Path

PHASE_DIR = Path(__file__).parent
PROJECT_ROOT = PHASE_DIR.parent.parent
sys.path.insert(0, str(PHASE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from cse import ContinuousSemanticEncoder
from flux_utils import load_checkpoint, PhaseResults


def compute_semantic_similarity(cse, text1: str, text2: str) -> float:
    """Compute cosine similarity of semantic mean vectors."""
    with torch.no_grad():
        w1 = cse.encode(text1)
        w2 = cse.encode(text2)
    v1 = w1.semantic.mean(dim=0)
    v2 = w2.semantic.mean(dim=0)
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()


def main():
    print("=" * 60)
    print("FLUX Phase 1 Test 3: Semantic Proximity Ordering")
    print("=" * 60)

    results = PhaseResults(phase=1, component_name="Continuous Semantic Encoder")

    # ── Load checkpoint ──
    print("\n  Loading Phase 1 checkpoint...")
    checkpoint = load_checkpoint(1)
    config = checkpoint['config']
    cse = ContinuousSemanticEncoder(**config, device='cpu')
    cse.load_state_dict(checkpoint['state_dict'])
    cse.eval()

    # ── 20 Test Triplets ──
    # Format: (anchor, closer_word, farther_word)
    # Requirement: sim(anchor, closer) > sim(anchor, farther)
    test_triplets = [
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
    ]

    # ── Evaluate ──
    print("\n  ── Evaluating semantic proximity ──")
    print(f"  {'#':>3}  {'Anchor':<12} {'Closer':<14} {'Farther':<14} {'sim(A,B)':>8} {'sim(A,C)':>8} {'Pass':>5}")
    print("  " + "-" * 70)

    passed_count = 0
    for i, (anchor, closer, farther) in enumerate(test_triplets):
        sim_close = compute_semantic_similarity(cse, anchor, closer)
        sim_far = compute_semantic_similarity(cse, anchor, farther)
        passed = sim_close > sim_far
        if passed:
            passed_count += 1
        status = "✓" if passed else "✗"
        print(f"  {i+1:3d}  {anchor:<12} {closer:<14} {farther:<14} {sim_close:>8.4f} {sim_far:>8.4f} {status:>5}")

    # ── Record results ──
    print(f"\n  ── Score: {passed_count}/{len(test_triplets)} ──")

    results.add_test(
        "Semantic Proximity Ordering",
        passed=passed_count >= 18,
        score=f"{passed_count}/{len(test_triplets)}",
        threshold="≥ 18/20",
    )

    # Also check absolute similarity levels
    print("\n  ── Similarity Levels (informational) ──")
    similar_sims = []
    unrelated_sims = []
    for anchor, closer, farther in test_triplets:
        similar_sims.append(compute_semantic_similarity(cse, anchor, closer))
        unrelated_sims.append(compute_semantic_similarity(cse, anchor, farther))

    avg_similar = sum(similar_sims) / len(similar_sims)
    avg_unrelated = sum(unrelated_sims) / len(unrelated_sims)
    print(f"    Average similarity (related words):   {avg_similar:.4f}")
    print(f"    Average similarity (unrelated words):  {avg_unrelated:.4f}")
    print(f"    Gap (related - unrelated):             {avg_similar - avg_unrelated:.4f}")

    results.add_test(
        "Similar Words > 0.5 avg",
        passed=avg_similar > 0.5,
        score=f"{avg_similar:.4f}",
        threshold="> 0.5 average",
        notes="Informational",
    )

    results.add_metric("Semantic proximity score", f"{passed_count}/{len(test_triplets)}")
    results.add_metric("Average similar-word similarity", f"{avg_similar:.4f}")
    results.add_metric("Average unrelated-word similarity", f"{avg_unrelated:.4f}")

    results.add_demo("test_phase1_test3 (Semantic Proximity)", True, "Automated")
    results.save()

    # ── Summary ──
    print(f"\n  All tests passed: {results.all_tests_passed()}")
    if results.all_tests_passed():
        print("  ✓ SEMANTIC PROXIMITY TEST PASSED")
    else:
        print("  ✗ SEMANTIC PROXIMITY TEST FAILED")

    return results.all_tests_passed()


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
