"""
test_phase2_5_test1.py — Test: Predicted waves decode to readable text

Loads Phase 2.5 checkpoint from HuggingFace.
Encodes test texts, uses prefix to predict next wave, decodes it.
Checks that decoded bytes are valid UTF-8 words (not garbage).

Threshold: avg byte accuracy ≥ 60% across gate texts
"""

import sys
import os
from pathlib import Path

# Path setup
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase2'))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from flux_utils import PhaseResults

from train_wru import (
    load_phase1_and_phase2,
    load_phase2_5_checkpoint,
    run_phase2_5_decode_gate,
    PHASE2_5_CONFIG,
)


def main():
    print("\n" + "=" * 60)
    print("  TEST 1: Predicted waves decode to readable text")
    print("=" * 60)

    device = 'cpu'
    hf_token = os.environ.get('HF_TOKEN', '')

    # Load frozen components
    print("\n  Loading Phase 1 + 2 (frozen)...")
    components = load_phase1_and_phase2(device=device, hf_token=hf_token)

    # Load trained WRU
    print("  Loading Phase 2.5 WRU checkpoint...")
    wru = load_phase2_5_checkpoint(device=device, hf_token=hf_token)
    wru.eval()

    n_params = wru.count_parameters()
    print(f"  WRU parameters: {n_params:,}")

    # Run decode gate
    passed, avg_acc, min_acc = run_phase2_5_decode_gate(
        cse=components['cse'],
        chunker=components['chunker'],
        w2f=components['w2f'],
        wru=wru,
        wtt=components['wtt'],
        phase=2.5,
        verbose=True,
        temperature=0.5,
    )

    # Record results
    threshold = PHASE2_5_CONFIG['gate_avg_threshold']
    results = PhaseResults(phase=2, component_name="WRU Decode Gate (Phase 2.5)")
    results.add_test(
        "Decode Gate Avg Byte Accuracy",
        passed=avg_acc >= threshold,
        score=avg_acc,
        threshold=threshold,
    )
    results.add_test(
        "Decode Gate Min Byte Accuracy",
        passed=min_acc >= PHASE2_5_CONFIG['gate_min_threshold'],
        score=min_acc,
        threshold=PHASE2_5_CONFIG['gate_min_threshold'],
    )
    results.save()

    print(f"\n  {'✓' if passed else '✗'} Test 1: avg={avg_acc:.1%} min={min_acc:.1%}")
    assert avg_acc >= threshold, (
        f"FAILED: avg byte accuracy {avg_acc:.1%} < {threshold:.0%}"
    )
    print("  ✓ TEST 1 PASSED\n")


if __name__ == '__main__':
    main()
