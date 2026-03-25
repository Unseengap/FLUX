"""
test_phase2_test3.py — Wave → Field → Wave → TEXT Byte Accuracy > 90%

The most important field test in the project.
Proves the field bridge doesn't destroy decodability:
    text → CSE → WaveChunker → WaveToField → FieldToWave → WaveToText → text

This is the Phase 2 decode gate as a formal test. If the original roadmap
had run this test at Phase 2, the Phase 9 mode collapse would have been
caught 7 phases earlier.

Run: python test_phase2_test3.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from flux_utils import PhaseResults
from decode_gate import byte_accuracy


GATE_TEXTS = [
    "The future of artificial intelligence",
    "Energy equals mass times the speed of light squared",
    "Photosynthesis converts sunlight into chemical energy",
    "Water freezes at zero degrees Celsius",
    "The cat sat on the mat",
    "café naïve résumé",
    "def hello(): return 'world'",
    "∫₀^∞ e^(-x²) dx = √π/2",
]


@torch.no_grad()
def test_field_decode_accuracy():
    """Test: wave → field bridge → WTT → text achieves > 90% byte accuracy."""
    print("=" * 60)
    print("Test 3: Wave → Field → Wave → TEXT (Byte Accuracy > 90%)")
    print("=" * 60)
    print("  This is the Phase 2 decode gate as a formal test.")
    print()

    # Load checkpoints
    try:
        from train_codec import load_phase1_checkpoint
        from train_field import load_phase2_checkpoint
        codec        = load_phase1_checkpoint(device='cpu')
        _, bridge, _ = load_phase2_checkpoint(device='cpu')
        cse      = codec.cse
        chunker  = codec.chunker
        wtt      = codec.wtt
        wtf      = bridge.wtf
        ftw      = bridge.ftw
        print("  ✓ Phase 1 + Phase 2 v2 checkpoints loaded")
    except FileNotFoundError as e:
        print(f"  CHECKPOINT NOT FOUND: {e}")
        raise

    cse.eval()
    chunker.eval()
    wtt.eval()
    wtf.eval()
    ftw.eval()

    results = []

    print("  Full pipeline: text → CSE → Chunker → WTF → FTW → WTT → text")
    print()

    for text in GATE_TEXTS:
        wave          = cse.encode(text)
        chunk_waves, _ = chunker(wave.full)       # [N, 432]
        field_vecs    = wtf(chunk_waves)             # [N, 512]
        reconstructed = ftw(field_vecs)              # [N, 432]
        decoded_text  = wtt.decode_to_text(reconstructed, temperature=0.3)
        acc           = byte_accuracy(text, decoded_text)

        results.append(acc)
        status = '✓' if acc >= 0.70 else '✗'
        print(f"  {status} [{acc:.1%}] '{text[:45]}'")
        if acc < 0.50:
            print(f"          ⟶ decoded: '{decoded_text[:45]}'")

    avg_acc = sum(results) / len(results)
    min_acc = min(results)

    print(f"\n  Avg byte accuracy : {avg_acc:.1%}  (threshold: 90%)")
    print(f"  Min byte accuracy : {min_acc:.1%}  (threshold: 70%)")

    assert avg_acc >= 0.90, (
        f"FAIL: avg byte accuracy {avg_acc:.1%} < 90%\n"
        f"The field bridge is destroying decodability.\n"
        f"Increase decode_loss_weight and retrain: python train_field.py --decode-loss-weight 1.0"
    )
    assert min_acc >= 0.70, (
        f"FAIL: min byte accuracy {min_acc:.1%} < 70%\n"
        f"Some inputs fail entirely through the field bridge."
    )

    print(f"\n  ✓ PASS: Phase 2 through-field decode gate")
    print(f"  ✓ The field bridge preserves decodability")
    return avg_acc, min_acc


if __name__ == '__main__':
    results = PhaseResults(phase=2, component_name="Resonance Field v2")
    try:
        avg_acc, min_acc = test_field_decode_accuracy()
        results.add_test("Field decode gate avg", passed=(avg_acc >= 0.90), score=avg_acc, threshold=0.90)
        results.add_test("Field decode gate min", passed=(min_acc >= 0.70), score=min_acc, threshold=0.70)
        print("\n  ALL TESTS PASSED ✓")
    except AssertionError as e:
        print(f"\n  TEST FAILED ✗\n  {e}")
    except FileNotFoundError as e:
        print(f"\n  CHECKPOINT NOT FOUND ✗\n  {e}")
    results.save()
