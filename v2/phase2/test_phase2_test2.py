"""
test_phase2_test2.py — Wave → Field → Wave Cosine Similarity > 0.85

Verifies that the field bridge preserves wave structure:
    WaveToField → FieldToWave round-trip cosine similarity > 0.85

This is the core geometric invariant. If the bridge doesn't preserve
the wave direction, it can't be used for generation or decoding.

Run: python test_phase2_test2.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F
from flux_utils import PhaseResults

TEST_TEXTS = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning transforms raw data into knowledge",
    "Physics is the study of matter and energy",
    "café naïve résumé",
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "Water freezes at zero degrees Celsius",
    "∫₀^∞ e^(-x²) dx = √π/2",
    "The cat sat on the mat",
]


@torch.no_grad()
def test_wave_field_wave_cosine():
    """Test: wave → field → wave cosine similarity > 0.85."""
    print("=" * 60)
    print("Test 2: Wave → Field → Wave Cosine Similarity > 0.85")
    print("=" * 60)

    # Load checkpoints
    try:
        from train_codec import load_phase1_checkpoint
        from train_field import load_phase2_checkpoint
        codec           = load_phase1_checkpoint(device='cpu')
        _, bridge, _    = load_phase2_checkpoint(device='cpu')
        cse     = codec.cse
        chunker = codec.chunker
        wtf     = bridge.wtf
        ftw     = bridge.ftw
        print("  ✓ Checkpoints loaded")
    except FileNotFoundError as e:
        print(f"  ⚠ Checkpoint missing: {e}")
        print("  Running with UNTRAINED bridge (cosine will be low — expected)")
        from cse import ContinuousSemanticEncoder
        from wave_chunker import WaveChunker
        from wave_to_field import WaveToField
        from field_to_wave import FieldToWave
        cse     = ContinuousSemanticEncoder()
        chunker = WaveChunker()
        wtf     = WaveToField()
        ftw     = FieldToWave()

    cse.eval()
    chunker.eval()
    wtf.eval()
    ftw.eval()

    cosine_scores = []

    print("\n  Round-trip cosine similarities:")
    for text in TEST_TEXTS:
        wave = cse.encode(text)
        chunks, _ = chunker(wave.full)  # [N, 432]

        # Round-trip through field bridge
        field_vecs    = wtf(chunks)         # [N, 512]
        reconstructed = ftw(field_vecs)     # [N, 432]

        # Per-chunk cosine
        cos = F.cosine_similarity(chunks, reconstructed, dim=-1)  # [N]
        avg_cos = cos.mean().item()
        cosine_scores.append(avg_cos)

        status = '✓' if avg_cos >= 0.85 else ('⚠' if avg_cos >= 0.70 else '✗')
        print(f"  {status} cos={avg_cos:.3f}  '{text[:45]}'")

    overall_avg = sum(cosine_scores) / len(cosine_scores)
    overall_min = min(cosine_scores)
    pass_rate   = sum(1 for c in cosine_scores if c >= 0.85) / len(cosine_scores)

    print(f"\n  Overall avg cosine : {overall_avg:.3f}")
    print(f"  Overall min cosine : {overall_min:.3f}")
    print(f"  Pass rate (≥0.85)  : {pass_rate:.0%}")

    assert overall_avg >= 0.85, (
        f"FAIL: avg cosine {overall_avg:.3f} < 0.85\n"
        f"The field bridge is not preserving wave structure.\n"
        f"Retrain with more reconstruction loss weight: --recon-loss-weight 2.0"
    )

    print(f"\n  ✓ PASS: avg cos={overall_avg:.3f} ≥ 0.85")
    return overall_avg, overall_min


if __name__ == '__main__':
    results = PhaseResults(phase=2, component_name="Resonance Field v2")
    try:
        avg_cos, min_cos = test_wave_field_wave_cosine()
        results.add_test("Round-trip avg cosine", passed=(avg_cos >= 0.85), score=avg_cos, threshold=0.85)
        results.add_test("Round-trip min cosine", passed=(min_cos >= 0.70), score=min_cos, threshold=0.70)
        print("\n  ALL TESTS PASSED ✓")
    except AssertionError as e:
        print(f"\n  TEST FAILED ✗\n  {e}")
    except FileNotFoundError as e:
        print(f"\n  CHECKPOINT NOT FOUND ✗\n  {e}")
    results.save()
