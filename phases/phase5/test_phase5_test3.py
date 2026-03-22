"""
Phase 5 Test 3: Geometry Computation Correctness

Tests that CGN nodes correctly bend signals via manifold curvature,
that mass influences output magnitude, and that the full CGN
forward pass produces valid, non-degenerate outputs.

Acceptance criteria:
  - Geometry computation produces correct signal bending (output ≠ input)
  - Higher mass → larger output magnitude
  - Output dimension matches input dimension
  - No NaN or Inf in outputs
  - Curvature, orientation, mass are learnable parameters
"""

import sys
import torch
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from flux_utils import PhaseResults
from cgn import CausalGeometryNode
from manifold import ManifoldPatch


def test_geometry_correctness():
    print("=" * 60)
    print("Phase 5 Test 3: Geometry Computation Correctness")
    print("=" * 60)

    results = PhaseResults(phase=5, component_name="Causal Geometry Nodes")
    feature_dim = 512

    # ── Test 3a: Signal bending ──
    print("\n  Test 3a: Signal bending via curvature")
    node = CausalGeometryNode(0, feature_dim)
    signal = torch.randn(feature_dim)
    output = node(signal)

    shape_ok = output.shape == signal.shape
    bending = torch.norm(output - signal).item()
    bends = bending > 0

    print(f"    Input shape:  {signal.shape}")
    print(f"    Output shape: {output.shape}")
    print(f"    Bending magnitude: {bending:.4f}")
    print(f"    Shapes match: {shape_ok}")
    print(f"    Signal bent:  {bends}")

    # ── Test 3b: No NaN/Inf ──
    print(f"\n  Test 3b: Numerical stability")
    no_nan = not torch.isnan(output).any().item()
    no_inf = not torch.isinf(output).any().item()
    print(f"    No NaN: {no_nan}")
    print(f"    No Inf: {no_inf}")

    # Edge cases
    zero_signal = torch.zeros(feature_dim)
    zero_out = node(zero_signal)
    zero_ok = not torch.isnan(zero_out).any().item() and not torch.isinf(zero_out).any().item()
    print(f"    Zero input stable: {zero_ok}")

    large_signal = torch.randn(feature_dim) * 100
    large_out = node(large_signal)
    large_ok = not torch.isnan(large_out).any().item() and not torch.isinf(large_out).any().item()
    print(f"    Large input stable: {large_ok}")

    # ── Test 3c: Mass amplification ──
    print(f"\n  Test 3c: Mass amplification")
    node1 = CausalGeometryNode(1, feature_dim)
    with torch.no_grad():
        node1.mass.copy_(torch.tensor(1.0))
    out_m1 = node1(signal)

    node_high = CausalGeometryNode(2, feature_dim)
    # Copy same parameters
    with torch.no_grad():
        node_high.curvature.copy_(node1.curvature)
        node_high.orientation.copy_(node1.orientation)
        node_high.mass.copy_(torch.tensor(10.0))
        # Copy manifold parameters
        node_high.manifold.load_state_dict(node1.manifold.state_dict())

    out_m10 = node_high(signal)

    norm_m1 = out_m1.norm().item()
    norm_m10 = out_m10.norm().item()
    mass_amplifies = norm_m10 > norm_m1

    print(f"    Mass=1.0 output norm: {norm_m1:.4f}")
    print(f"    Mass=10.0 output norm: {norm_m10:.4f}")
    print(f"    Higher mass → larger output: {mass_amplifies}")

    # ── Test 3d: Learnable parameters ──
    print(f"\n  Test 3d: Learnable parameters")
    param_names = [name for name, _ in node.named_parameters()]
    has_curvature = any('curvature' in n for n in param_names)
    has_orientation = any('orientation' in n for n in param_names)
    has_mass = any('mass' in n for n in param_names)
    total_params = sum(p.numel() for p in node.parameters())
    print(f"    Parameters: {param_names[:5]}...")
    print(f"    Has curvature: {has_curvature}")
    print(f"    Has orientation: {has_orientation}")
    print(f"    Has mass: {has_mass}")
    print(f"    Total params per node: {total_params:,}")

    params_ok = has_curvature and has_orientation and has_mass

    # ── Test 3e: Manifold geodesic step ──
    print(f"\n  Test 3e: Manifold geodesic step")
    manifold = ManifoldPatch(feature_dim)
    position = torch.randn(feature_dim)
    velocity = torch.randn(feature_dim) * 0.1
    curvature = torch.randn(1, feature_dim) * 0.01

    new_pos, new_vel = manifold.compute_geodesic_step(position, velocity, curvature)
    moved = torch.norm(new_pos - position).item() > 0
    print(f"    Position moved: {moved}")
    print(f"    Displacement: {torch.norm(new_pos - position).item():.4f}")

    geodesic_ok = moved and not torch.isnan(new_pos).any().item()

    # ── Test 3f: Batch processing ──
    print(f"\n  Test 3f: Batch processing")
    batch_signal = torch.randn(8, feature_dim)
    batch_out = node(batch_signal)
    batch_ok = batch_out.shape == batch_signal.shape
    print(f"    Batch input:  {batch_signal.shape}")
    print(f"    Batch output: {batch_out.shape}")
    print(f"    Shapes match: {batch_ok}")

    # ── Test 3g: Causal trace ──
    print(f"\n  Test 3g: Forward with trace")
    out_traced, trace = node.forward_with_trace(signal)
    trace_ok = (
        trace.node_id == node.node_id
        and trace.bending_magnitude > 0
        and trace.input_signal.shape == signal.shape
        and trace.output_signal.shape == signal.shape
    )
    print(f"    Trace node_id: {trace.node_id}")
    print(f"    Bending: {trace.bending_magnitude:.4f}")
    print(f"    Influence: {trace.influence_strength:.4f}")
    print(f"    Trace valid: {trace_ok}")

    # ── Record results ──
    all_pass = (
        shape_ok and bends and no_nan and no_inf and zero_ok and large_ok
        and mass_amplifies and params_ok and geodesic_ok and batch_ok and trace_ok
    )

    results.add_test(
        "Signal bending", passed=bends,
        score=f"bending={bending:.4f}", threshold="> 0",
    )
    results.add_test(
        "Numerical stability", passed=no_nan and no_inf and zero_ok and large_ok,
        score="no NaN/Inf", threshold="all stable",
    )
    results.add_test(
        "Mass amplification", passed=mass_amplifies,
        score=f"m1={norm_m1:.4f} m10={norm_m10:.4f}", threshold="m10 > m1",
    )
    results.add_test(
        "Learnable parameters", passed=params_ok,
        score=f"{total_params} params", threshold="curvature+orientation+mass",
    )
    results.add_test(
        "Geodesic step", passed=geodesic_ok,
        score=f"moved={moved}", threshold="position changes",
    )
    results.add_test(
        "Batch processing", passed=batch_ok,
        score=f"{batch_out.shape}", threshold="same shape",
    )
    results.add_test(
        "Causal trace", passed=trace_ok,
        score=f"bending={trace.bending_magnitude:.4f}", threshold="valid trace",
    )

    results.save(str(Path(__file__).parent / 'RESULTS_PHASE_5.md'))

    print(f"\n{'='*60}")
    status = "PASS" if all_pass else "FAIL"
    print(f"Test 3 Overall: {status}")
    print(f"{'='*60}")

    assert all_pass, "Test 3 FAILED — see details above"


if __name__ == "__main__":
    test_geometry_correctness()

