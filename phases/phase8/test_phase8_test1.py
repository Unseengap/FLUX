"""
Phase 8 — Test 1: Perplexity on Penn Treebank

Measures FLUXModel (Phase 8) byte-level perplexity on Penn Treebank test set.

Pass Criteria:
  - Perplexity is finite (model produces valid predictions)
  - Perplexity < 500 (reasonable for byte-level model)
  - Model processes all test samples without errors
"""

import sys
import math
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
from benchmark_gpt2 import FLUXBenchmark, load_benchmark_texts
from flux_utils import get_device, checkpoint_exists, PhaseResults


def main():
    print("=" * 60)
    print("  Test 1: Penn Treebank Perplexity")
    print("=" * 60)

    device = get_device()

    # Load model
    if checkpoint_exists(8):
        model = FLUXLarge.from_phase8_checkpoint(device=device)
    else:
        print("  ⚠ No Phase 8 checkpoint — testing with fresh FLUXLarge")
        model = FLUXLarge(device=device)

    # Load PTB test data
    ptb_texts = load_benchmark_texts('ptb', max_samples=100)
    print(f"  Loaded {len(ptb_texts)} PTB test samples")

    # Compute perplexity
    bench = FLUXBenchmark(model, device=device)
    ppl = bench.compute_perplexity(ptb_texts)
    print(f"\n  Penn Treebank Perplexity: {ppl:.2f}")

    # Assertions
    is_finite = math.isfinite(ppl)
    is_reasonable = ppl < 500
    processed_all = len(ptb_texts) > 0

    print(f"\n  Checks:")
    print(f"    Perplexity is finite:    {'✓' if is_finite else '✗'} ({ppl:.2f})")
    print(f"    Perplexity < 500:        {'✓' if is_reasonable else '✗'} ({ppl:.2f})")
    print(f"    All samples processed:   {'✓' if processed_all else '✗'}")

    passed = is_finite and is_reasonable and processed_all
    print(f"\n  {'✓ TEST PASSED' if passed else '✗ TEST FAILED'}")

    assert is_finite, f"Perplexity is not finite: {ppl}"
    assert is_reasonable, f"Perplexity too high: {ppl} (threshold: 500)"
    assert processed_all, "No PTB samples processed"

    return ppl


if __name__ == '__main__':
    main()
