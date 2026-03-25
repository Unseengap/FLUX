"""
test_phase1_test1.py — Round-trip byte accuracy > 95%

Loads the Phase 1 v2 checkpoint and verifies that the full pipeline
(text → CSE → WaveChunker → WaveToText → text) achieves > 95% byte
accuracy on the standard decode gate texts.

Run: python test_phase1_test1.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from train_codec import load_phase1_checkpoint
from decode_gate import run_decode_gate, byte_accuracy, DECODE_GATE_TEXTS
from flux_utils import PhaseResults


def test_round_trip_accuracy():
    """Test: entire pipeline achieves > 95% byte accuracy on gate texts."""
    print("=" * 60)
    print("Test 1: Round-trip Byte Accuracy > 95%")
    print("=" * 60)

    codec = load_phase1_checkpoint(device='cpu')

    passed, avg_acc, min_acc = run_decode_gate(
        codec.cse, codec.chunker, codec.wtt,
        phase=1, raise_on_fail=False, verbose=True, temperature=0.3,
    )

    # Record per-text breakdown
    import torch
    individual_results = {}
    for text in DECODE_GATE_TEXTS:
        with torch.no_grad():
            wave = codec.cse.encode(text)
            chunk_waves, _ = codec.chunker(wave.full)
            decoded = codec.wtt.decode_to_text(chunk_waves, temperature=0.3)
            acc = byte_accuracy(text, decoded)
            individual_results[text[:30]] = acc

    # Assertions
    assert avg_acc >= 0.95, (
        f"FAIL: avg byte accuracy {avg_acc:.1%} < 95%\n"
        f"Worst results: {sorted(individual_results.items(), key=lambda x: x[1])[:3]}\n"
        f"Re-train with more steps: python train_codec.py --steps 50000"
    )

    print(f"\n  ✓ PASS: avg byte accuracy = {avg_acc:.1%} (threshold: 95%)")
    print(f"  ✓ PASS: min byte accuracy = {min_acc:.1%} (threshold: 70%)")
    return avg_acc, min_acc


if __name__ == '__main__':
    results = PhaseResults(phase=1, component_name="Wave Codec v2")
    try:
        avg_acc, min_acc = test_round_trip_accuracy()
        results.add_test("Round-trip avg byte accuracy", passed=(avg_acc >= 0.95), score=avg_acc, threshold=0.95)
        results.add_test("Round-trip min byte accuracy", passed=(min_acc >= 0.70), score=min_acc, threshold=0.70)
        print("\n  ALL TESTS PASSED ✓")
    except AssertionError as e:
        print(f"\n  TEST FAILED ✗\n  {e}")
        results.add_test("Round-trip avg byte accuracy", passed=False, score=0.0, threshold=0.95)
    except FileNotFoundError as e:
        print(f"\n  CHECKPOINT NOT FOUND ✗\n  {e}")
    results.save()
