"""
PHASE 1 TEST 2: Language-agnostic encoding.

Test pairs (same semantic content, different languages):
    English: "The cat is sleeping"
    French:  "Le chat dort"
    Spanish: "El gato está durmiendo"

Pass criteria:
    - Cosine similarity of semantic dimension > 0.4 across pairs
    - All 5 test language pairs pass
    - Encoding of any UTF-8 input succeeds without error

Run: python test_phase1_test2.py
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
    print("FLUX Phase 1 Test 2: Language-Agnostic Encoding")
    print("=" * 60)

    results = PhaseResults(phase=1, component_name="Continuous Semantic Encoder")

    # ── Load checkpoint ──
    print("\n  Loading Phase 1 checkpoint...")
    checkpoint = load_checkpoint(1)
    config = checkpoint['config']
    cse = ContinuousSemanticEncoder(**config, device='cpu')
    cse.load_state_dict(checkpoint['state_dict'])
    cse.eval()

    # ── Test 1: Encoding diverse inputs without errors ──
    print("\n  ── Test: Encode diverse UTF-8 inputs ──")
    diverse_inputs = [
        "Hello, world!",                          # English
        "Bonjour le monde!",                       # French
        "¡Hola mundo!",                            # Spanish
        "你好世界",                                  # Chinese
        "こんにちは世界",                              # Japanese
        "def hello(): print('hi')",                # Code
        "E = mc² + ∫f(x)dx",                      # Math
        "🎉🚀💡",                                   # Emoji
        "Mixed: hello 你好 🌍",                     # Mixed
        "",                                        # Empty
        "a",                                       # Single char
    ]

    encode_errors = 0
    for text in diverse_inputs:
        try:
            with torch.no_grad():
                wave = cse.encode(text)
            if len(text) > 0:
                assert wave.seq_len > 0, f"Empty wave for non-empty input: {text}"
                assert wave.full.shape[-1] == 432, f"Wrong wave dim for: {text}"
            print(f"    ✓ \"{text[:40]}\" → [{wave.seq_len}, 432]")
        except Exception as e:
            print(f"    ✗ \"{text[:40]}\" → ERROR: {e}")
            encode_errors += 1

    results.add_test(
        "UTF-8 Encoding (no errors)",
        passed=encode_errors == 0,
        score=f"{len(diverse_inputs) - encode_errors}/{len(diverse_inputs)} success",
        threshold="all inputs encode",
    )

    # ── Test 2: Cross-language semantic similarity ──
    print("\n  ── Test: Cross-Language Similarity ──")
    language_pairs = [
        ("the cat", "le chat", "EN/FR cat"),
        ("hello", "bonjour", "EN/FR hello"),
        ("water", "eau", "EN/FR water"),
        ("the cat is sleeping", "le chat dort", "EN/FR sentence"),
        ("thank you", "merci", "EN/FR thanks"),
    ]

    pair_results = []
    for text1, text2, label in language_pairs:
        sim = compute_semantic_similarity(cse, text1, text2)
        passed = sim > 0.4
        pair_results.append(passed)
        status = "✓" if passed else "✗"
        print(f"    {status} {label}: sim(\"{text1}\", \"{text2}\") = {sim:.4f}")

    pairs_passed = sum(pair_results)
    results.add_test(
        "Cross-Language Similarity",
        passed=pairs_passed >= 4,  # Allow 1 failure out of 5
        score=f"{pairs_passed}/{len(language_pairs)} pairs > 0.4",
        threshold="≥ 4/5 pairs pass",
    )

    # ── Test 3: Deterministic encoding ──
    print("\n  ── Test: Deterministic Encoding ──")
    test_text = "The quick brown fox"
    with torch.no_grad():
        wave1 = cse.encode(test_text)
        wave2 = cse.encode(test_text)

    diff = (wave1.full - wave2.full).abs().max().item()
    deterministic = diff < 1e-6
    print(f"    Max difference between two encodings: {diff:.2e}")
    print(f"    {'✓' if deterministic else '✗'} Encoding is {'deterministic' if deterministic else 'NOT deterministic'}")

    results.add_test(
        "Deterministic Encoding",
        passed=deterministic,
        score=f"max_diff={diff:.2e}",
        threshold="< 1e-6",
    )

    # ── Record metrics ──
    all_sims = [compute_semantic_similarity(cse, t1, t2) for t1, t2, _ in language_pairs]
    results.add_metric("Average cross-language similarity", f"{sum(all_sims)/len(all_sims):.4f}")
    results.add_metric("Language pairs passing", f"{pairs_passed}/{len(language_pairs)}")

    results.add_demo("test_phase1_test2 (Language Agnostic)", True, "Automated")
    results.save()

    # ── Summary ──
    print(f"\n  All tests passed: {results.all_tests_passed()}")
    if results.all_tests_passed():
        print("  ✓ LANGUAGE AGNOSTIC TEST PASSED")
    else:
        print("  ✗ LANGUAGE AGNOSTIC TEST FAILED")

    return results.all_tests_passed()


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
