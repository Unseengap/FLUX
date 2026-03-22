"""Temporary smoke test for Phase 5 fixes."""
import sys
sys.path.insert(0, 'phases/phase5')
import torch
from cgn import CausalGeometryNode
from causal_graph import CausalGraph
from multi_timescale import MultiTimescaleCoordinator

# Test 1: CGN forward produces non-zero output
node = CausalGeometryNode(0, 512)
signal = torch.randn(512) * 0.5
out = node(signal)
print(f"CGN output norm: {out.norm().item():.4f} (should be > 0)")

# Test 2: Mass amplification
out1 = node(signal)
with torch.no_grad():
    node.mass.copy_(torch.tensor(5.0))
out5 = node(signal)
ratio = out5.norm().item() / max(out1.norm().item(), 1e-8)
print(f"Mass=5 amplification: {ratio:.2f}x (should be ~5.0)")

# Test 3: Multi-timescale separation
mtc = MultiTimescaleCoordinator(512, 16, 8, 4)
sig = torch.randn(512) * 0.5
sep = mtc.measure_timescale_separation(sig, max_steps=100)
print(f"Fast activates at step:   {sep['fast_steps_to_activate']} (should be < 5)")
print(f"Medium activates at step: {sep['medium_steps_to_activate']}")
print(f"Slow activates at step:   {sep['slow_steps_to_activate']} (should be > 10)")
print(f"Order correct: {sep['fast_steps_to_activate'] < sep['medium_steps_to_activate'] <= sep['slow_steps_to_activate']}")

# Test 4: Conflict detection
cg = CausalGraph()
cg.add_arrow(0, 2, weight=0.8, reason="bird -> can_fly")
cg.add_arrow(1, 0, weight=1.0, reason="penguin -> bird")
cg.add_arrow(1, 3, weight=0.9, reason="penguin -> cannot_fly")
cg.add_contradiction(2, 3)
conflicts = cg.detect_entity_conflicts(1, depth=3)
print(f"Penguin conflicts: {conflicts} (should be [(2,3)] or [(3,2)])")

# Test 5: Pipeline non-zero
mtc2 = MultiTimescaleCoordinator(512, 16, 8, 4)
out = mtc2(torch.randn(512) * 0.5, steps=5)
print(f"Pipeline output norm: {out.norm().item():.4f} (should be > 0)")

print("\n--- ALL SMOKE TESTS COMPLETE ---")
