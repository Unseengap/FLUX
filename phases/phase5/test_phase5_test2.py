import sys
import torch
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from multi_timescale import MultiTimescaleCoordination

def test_timescale_separation():
    print("Running Phase 5 Test 2: Multi-timescale Separation")
    feature_dim = 512
    mtc = MultiTimescaleCoordination(feature_dim)

    signal = torch.randn(feature_dim)

    # Observe behavior over steps
    out1 = mtc(signal, steps=1)
    out5 = mtc(signal, steps=5)

    # Check if there is a difference due to multiple steps in fast nodes
    diff = torch.norm(out1 - out5)
    print(f"Difference between 1 step and 5 steps: {diff.item():.4f}")

    # We expect some evolution
    assert diff > 0, "Signal did not evolve across steps"

    # Verify slow vs fast node count
    assert len(mtc.fast_nodes) == 16
    assert len(mtc.slow_nodes) == 4

    print("Test 2: PASS")

if __name__ == "__main__":
    test_timescale_separation()
