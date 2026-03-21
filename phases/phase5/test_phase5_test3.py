import sys
import torch
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from cgn import CausalGeometryNode

def test_geometry_correctness():
    print("Running Phase 5 Test 3: Geometry Computation Correctness")
    feature_dim = 512
    node = CausalGeometryNode(0, feature_dim)

    # Test signal bending
    signal = torch.randn(feature_dim)
    output = node(signal)

    # Output should be different from signal due to curvature and orientation
    # but same dimension
    assert output.shape == signal.shape
    diff = torch.norm(output - signal)
    print(f"Bending magnitude: {diff.item():.4f}")

    # High mass should increase magnitude
    with torch.no_grad():
        node.mass.copy_(torch.tensor(10.0))
    output_high_mass = node(signal)
    assert torch.norm(output_high_mass) > torch.norm(output)
    print("Mass influence verified")

    print("Test 3: PASS")

if __name__ == "__main__":
    test_geometry_correctness()
