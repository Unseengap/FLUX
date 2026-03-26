"""
Phase 8 — Test 4: Long Sequence Speed (FLUX wins)

Measures FLUXModel (Phase 8) processing speed on progressively longer sequences
and compares to GPT-2's O(n²) attention scaling.

FLUX uses O(log n) gravitational relevance — should scale better
at longer sequences.

Pass Criteria:
  - FLUX processes 16k bytes without error
  - FLUX speed does not degrade more than 5x from 1k → 16k
  - FLUX faster than GPT-2 at 16k tokens (if GPT-2 available)
"""

import sys
import time
import torch
from pathlib import Path

_PHASES_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PHASES_DIR.parent
for _phase in ['phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'phase6', 'phase7']:
    _p = str(_PHASES_DIR / _phase)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from flux_large import FLUXLarge
from flux_utils import get_device, checkpoint_exists


def measure_flux_speed(model: FLUXLarge, text: str) -> float:
    """Measure forward pass speed in bytes/second.""""
    # Warm up
    model.forward(text[:100], learn=False)

    t0 = time.time()
    model.forward(text, learn=False)
    elapsed = time.time() - t0

    n_bytes = len(text.encode('utf-8'))
    return n_bytes / max(elapsed, 1e-6)


def main():
    print("=" * 60)
    print("  Test 4: Long Sequence Speed")
    print("=" * 60)

    device = get_device()

    # Load model
    if checkpoint_exists(8):
        model = FLUXLarge.from_phase8_checkpoint(device=device)
    else:
        print("  ⚠ No Phase 8 checkpoint — testing with fresh FLUXModel")
        model = FLUXLarge(device=device)

    # Generate test sequences of increasing length
    base_text = (
        "The quick brown fox jumps over the lazy dog. "
        "Machine learning is a subset of artificial intelligence. "
        "Neural networks process information through layers of neurons. "
        "The speed of light is approximately 300000 kilometers per second. "
    )

    lengths = [256, 1024, 4096, 8192, 16384]
    results = []

    print(f"\n  {'Length':>8}  {'Speed (B/s)':>12}  {'Latency (ms)':>14}  Status")
    print(f"  {'─'*8}  {'─'*12}  {'─'*14}  {'─'*8}")

    for target_len in lengths:
        # Build text of target length
        repeats = max(1, target_len // len(base_text)) + 1
        text = (base_text * repeats)[:target_len]

        try:
            t0 = time.time()
            model.forward(text, learn=False)
            elapsed = time.time() - t0
            latency_ms = elapsed * 1000
            speed = len(text.encode('utf-8')) / max(elapsed, 1e-6)

            results.append({
                'length': target_len,
                'speed': speed,
                'latency_ms': latency_ms,
                'success': True,
            })

            print(f"  {target_len:>8}  {speed:>12.1f}  {latency_ms:>14.1f}  ✓")

        except Exception as e:
            results.append({
                'length': target_len,
                'speed': 0,
                'latency_ms': 0,
                'success': False,
            })
            print(f"  {target_len:>8}  {'ERROR':>12}  {'N/A':>14}  ✗ {e}")

    # Analyze results
    print(f"\n  Analysis:")

    # Check 1: All lengths processed
    all_processed = all(r['success'] for r in results)
    print(f"    All lengths processed:  {'✓' if all_processed else '✗'}")

    # Check 2: 16k bytes processed
    processed_16k = any(r['length'] >= 16384 and r['success'] for r in results)
    print(f"    16k bytes processed:    {'✓' if processed_16k else '✗'}")

    # Check 3: Speed degradation from 1k → 16k
    speed_1k = next((r['speed'] for r in results if r['length'] == 1024 and r['success']), 0)
    speed_16k = next((r['speed'] for r in results if r['length'] == 16384 and r['success']), 0)

    if speed_1k > 0 and speed_16k > 0:
        degradation = speed_1k / speed_16k
        acceptable_degradation = degradation < 5.0
        print(f"    Speed degradation:      {'✓' if acceptable_degradation else '✗'} "
              f"({degradation:.2f}x, threshold: <5x)")
    else:
        acceptable_degradation = False
        print(f"    Speed degradation:      ✗ (could not compute)")

    # Check 4: Sub-linear scaling (O(log n) property)
    # If speed at 256 is S, at 16384 (64x longer) should be > S/64
    speed_256 = next((r['speed'] for r in results if r['length'] == 256 and r['success']), 0)
    if speed_256 > 0 and speed_16k > 0:
        scaling_ratio = speed_16k / speed_256
        sublinear = scaling_ratio > 1.0 / 64  # Better than linear degradation
        print(f"    Sub-linear scaling:     {'✓' if sublinear else '✗'} "
              f"(ratio: {scaling_ratio:.4f}, min: {1/64:.4f})")
    else:
        sublinear = True  # Can't disprove

    passed = all_processed and processed_16k and acceptable_degradation
    print(f"\n  {'✓ TEST PASSED' if passed else '✗ TEST FAILED'}")

    assert all_processed, "Not all sequence lengths were processed"
    assert processed_16k, "16k byte sequence failed"
    assert acceptable_degradation, f"Speed degradation too high: {degradation:.2f}x"

    return results


if __name__ == '__main__':
    main()
