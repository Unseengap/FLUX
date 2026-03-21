import sys
import torch
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from causal_graph import CausalGraph

def test_causal_trace():
    print("Running Phase 5 Test 1: Causal Trace Accuracy")
    cg = CausalGraph()

    # Building a simple chain: 1 -> 2 -> 3
    cg.add_arrow(1, 2, weight=1.0)
    cg.add_arrow(2, 3, weight=1.0)

    trace = cg.trace_cause(3, depth=2)
    print(f"Trace for 3: {trace}")

    assert trace == [1, 2, 3], f"Expected [1, 2, 3], got {trace}"

    # Invalidation test
    cg.invalidate_cause(1)
    assert cg.evidence[(1, 2)] == -1.0
    print("Causal invalidation verified")

    print("Test 1: PASS")

if __name__ == "__main__":
    test_causal_trace()
