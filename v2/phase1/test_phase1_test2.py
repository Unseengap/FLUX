"""
test_phase1_test2.py — Language-agnostic encode+decode

Verifies the codec works across languages, scripts, code, math, and
edge-case byte patterns — not just ASCII English.

Run: python test_phase1_test2.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from train_codec import load_phase1_checkpoint
from decode_gate import byte_accuracy
from flux_utils import PhaseResults


MULTILINGUAL_TESTS = {
    'english':    ("The cat sat on the mat", 0.90),
    'french':     ("café résumé naïve coöperate", 0.85),
    'russian':    ("Привет мир", 0.80),
    'chinese':    ("人工智能", 0.75),
    'japanese':   ("機械学習", 0.75),
    'arabic':     ("مرحبا بالعالم", 0.70),
    'math':       ("∫₀^∞ e^(-x²) dx = √π/2", 0.80),
    'code':       ("def f(x): return x**2", 0.90),
    'emoji':      ("Hello 👋 World 🌍", 0.70),
    'mixed':      ("Python 3.11 支援 unicode 🐍", 0.70),
    'numbers':    ("1234567890.00 €£¥", 0.85),
    'empty_like': ("a", 0.90),
}


@torch.no_grad()
def test_multilingual():
    """Test: codec handles diverse languages and scripts."""
    print("=" * 60)
    print("Test 2: Language-Agnostic Encode + Decode")
    print("=" * 60)

    codec = load_phase1_checkpoint(device='cpu')

    passed_count = 0
    total_count = 0
    failures = []

    for lang, (text, threshold) in MULTILINGUAL_TESTS.items():
        try:
            wave = codec.cse.encode(text)
            chunk_waves, _ = codec.chunker(wave.full)
            decoded = codec.wtt.decode_to_text(chunk_waves, temperature=0.3)
            acc = byte_accuracy(text, decoded)

            status = '✓' if acc >= threshold else '✗'
            print(f"  {status} [{lang:12s}] {acc:.1%} (threshold: {threshold:.0%})  '{text[:30]}'")

            if acc >= threshold:
                passed_count += 1
            else:
                failures.append((lang, text, acc, threshold))
        except Exception as e:
            print(f"  ✗ [{lang:12s}] ERROR: {e}")
            failures.append((lang, text, 0.0, threshold))
        total_count += 1

    pass_rate = passed_count / total_count
    print(f"\n  Pass rate: {passed_count}/{total_count} ({pass_rate:.0%})")

    if failures:
        print(f"\n  Failures:")
        for lang, text, acc, thr in failures:
            print(f"    [{lang}] {acc:.1%} < {thr:.0%}: '{text[:30]}'")

    # Must pass at least 75% of language tests
    assert pass_rate >= 0.75, (
        f"FAIL: only {pass_rate:.0%} of language tests passed (need 75%)\n"
        f"Failures: {[f[0] for f in failures]}"
    )

    print(f"\n  ✓ PASS: {pass_rate:.0%} language/script coverage")
    return pass_rate, failures


if __name__ == '__main__':
    results = PhaseResults(phase=1, component_name="Wave Codec v2")
    try:
        pass_rate, failures = test_multilingual()
        results.add_test("Multilingual pass rate", passed=(pass_rate >= 0.75), score=pass_rate, threshold=0.75)
        results.add_test("Number of language failures", passed=(len(failures) <= 3), score=len(failures), threshold=3)
        print("\n  ALL TESTS PASSED ✓")
    except AssertionError as e:
        print(f"\n  TEST FAILED ✗\n  {e}")
        results.add_test("Multilingual pass rate", passed=False, score=0.0, threshold=0.75)
    except FileNotFoundError as e:
        print(f"\n  CHECKPOINT NOT FOUND ✗\n  {e}")
    results.save()
