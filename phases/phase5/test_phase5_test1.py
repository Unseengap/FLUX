"""
Phase 5 Test 1: Causal Trace Accuracy

Tests that every output has a traceable causal chain and that
causal invalidation correctly propagates.

Acceptance criteria:
  - Every output has a traceable causal chain
  - Causal invalidation works (disprove cause → invalidate conclusion)
  - Chain length ≥ 2 for multi-hop reasoning
"""

import sys
import torch
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from flux_utils import load_checkpoint, PhaseResults, checkpoint_exists
from causal_graph import CausalGraph


def test_causal_trace():
    print("=" * 60)
    print("Phase 5 Test 1: Causal Trace Accuracy")
    print("=" * 60)

    results = PhaseResults(phase=5, component_name="Causal Geometry Nodes")

    # ── Build causal graph ──
    cg = CausalGraph()

    # Simple chain: A→B→C
    cg.add_arrow(1, 2, weight=1.0, reason="A causes B")
    cg.add_arrow(2, 3, weight=1.0, reason="B causes C")

    # Branch: D→C (alternative cause)
    cg.add_arrow(4, 3, weight=0.5, reason="D also causes C")

    # ── Test 1a: Simple chain trace ──
    trace = cg.trace_cause(3, depth=3)
    chain_correct = len(trace.chain) >= 3  # Should be [1, 2, 3]
    print(f"\n  Trace for node 3: {trace.chain}")
    print(f"  Chain length: {len(trace.chain)} (expected ≥ 3)")
    print(f"  Total weight: {trace.total_weight:.2f}")
    assert chain_correct, f"Expected chain length ≥ 3, got {len(trace.chain)}"
    print("  ✓ Simple chain trace: PASS")

    # ── Test 1b: Conflict detection ──
    # Add opposing evidence
    cg.add_arrow(5, 3, weight=-0.8, reason="E contradicts C")
    trace_conflict = cg.trace_cause(3, depth=3)
    summary = cg.get_conflict_summary(3)
    print(f"\n  Conflict summary for node 3:")
    print(f"    Supporting: {len(summary['supporting'])} sources")
    print(f"    Opposing:   {len(summary['opposing'])} sources")
    print(f"    Conclusion: {summary['conclusion']}")
    print(f"    Net weight: {summary['net_weight']:.2f}")

    has_conflict_detection = len(summary['opposing']) > 0
    print(f"  ✓ Conflict detection: {'PASS' if has_conflict_detection else 'FAIL'}")

    # ── Test 1c: Causal invalidation ──
    cg2 = CausalGraph()
    cg2.add_arrow(10, 11, weight=1.0, reason="premise")
    cg2.add_arrow(11, 12, weight=0.8, reason="conclusion from premise")

    # Check before invalidation
    edge_before = cg2.get_edge(10, 11)
    weight_before = edge_before.weight if edge_before else 0

    # Invalidate the premise
    affected = cg2.invalidate_cause(10)
    edge_after = cg2.get_edge(10, 11)
    weight_after = edge_after.weight if edge_after else 0

    invalidation_works = weight_after < 0 and len(affected) >= 1
    print(f"\n  Before invalidation: weight={weight_before:.2f}")
    print(f"  After invalidation:  weight={weight_after:.2f}")
    print(f"  Affected downstream: {affected}")
    print(f"  ✓ Causal invalidation: {'PASS' if invalidation_works else 'FAIL'}")

    # ── Test 1d: Multi-hop trace ──
    cg3 = CausalGraph()
    for i in range(5):
        cg3.add_arrow(i, i + 1, weight=0.9, reason=f"step {i}→{i+1}")
    trace_deep = cg3.trace_cause(5, depth=10)
    deep_trace_correct = len(trace_deep.chain) >= 5
    print(f"\n  Deep trace (5 hops): {trace_deep.chain}")
    print(f"  ✓ Multi-hop trace: {'PASS' if deep_trace_correct else 'FAIL'}")

    # ── Record results ──
    all_pass = chain_correct and has_conflict_detection and invalidation_works and deep_trace_correct

    results.add_test(
        "Simple chain trace", passed=chain_correct,
        score=f"chain_len={len(trace.chain)}", threshold="≥ 3",
    )
    results.add_test(
        "Conflict detection", passed=has_conflict_detection,
        score=f"opposing={len(summary['opposing'])}", threshold="> 0",
    )
    results.add_test(
        "Causal invalidation", passed=invalidation_works,
        score=f"affected={len(affected)}", threshold="> 0",
    )
    results.add_test(
        "Multi-hop trace", passed=deep_trace_correct,
        score=f"depth={len(trace_deep.chain)}", threshold="≥ 5",
    )

    results.save(str(Path(__file__).parent / 'RESULTS_PHASE_5.md'))

    print(f"\n{'='*60}")
    status = "PASS" if all_pass else "FAIL"
    print(f"Test 1 Overall: {status}")
    print(f"{'='*60}")

    assert all_pass, "Test 1 FAILED — see details above"


if __name__ == "__main__":
    test_causal_trace()

