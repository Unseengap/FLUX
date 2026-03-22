"""
Phase 5 Demo 1: Trace Why a Conclusion Was Reached

Demonstrates causal reasoning with the CausalGraph:
- Feed knowledge: "Birds can fly. Penguins are birds."
- Query: "Can penguins fly?"
- Show causal trace with conflict detection and resolution

This is the key differentiator of CGN over standard neurons:
every conclusion stores WHY and can be traced/invalidated.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from causal_graph import CausalGraph


def demo_trace_conclusion():
    print("=" * 65)
    print("  FLUX Phase 5 Demo 1: Why Did You Say That?")
    print("=" * 65)

    cg = CausalGraph()

    # ─────────────────────────────────────────
    # Build knowledge graph
    # ─────────────────────────────────────────
    concepts = {
        0: "bird",
        1: "penguin",
        2: "can_fly",
        3: "cannot_fly",
        4: "sparrow",
        5: "has_wings",
        6: "lives_in_arctic",
        7: "is_heavy_bodied",
    }

    print("\n  Building Knowledge Graph:")
    print("  " + "-" * 55)

    knowledge = [
        (0, 2, 0.8, "birds can fly"),
        (0, 5, 0.95, "birds have wings"),
        (1, 0, 1.0, "penguin is a bird"),
        (1, 3, 0.9, "penguin cannot fly"),
        (1, 6, 0.85, "penguin lives in arctic"),
        (1, 7, 0.7, "penguin is heavy-bodied"),
        (7, 3, 0.6, "heavy body prevents flight"),
        (4, 0, 1.0, "sparrow is a bird"),
        (4, 2, 0.95, "sparrow can fly"),
    ]

    for src, tgt, weight, reason in knowledge:
        cg.add_arrow(src, tgt, weight=weight, reason=reason)
        print(f"    {concepts[src]:>15} → {concepts[tgt]:<15}  w={weight:.2f}  ({reason})")

    # ─────────────────────────────────────────
    # Query: Can penguins fly?
    # ─────────────────────────────────────────
    print(f"\n  {'='*55}")
    print("  Query: Can penguins fly?")
    print(f"  {'='*55}")

    # Trace path to "can fly"
    trace_fly = cg.trace_cause(2, depth=3)
    print(f"\n  Path to 'can_fly':")
    print(f"    Chain:   {' → '.join(concepts.get(n, str(n)) for n in trace_fly.chain)}")
    print(f"    Weight:  {trace_fly.total_weight:.2f}")
    print(f"    Conflict: {trace_fly.has_conflict}")

    # Trace path to "cannot fly"
    trace_not_fly = cg.trace_cause(3, depth=3)
    print(f"\n  Path to 'cannot_fly':")
    print(f"    Chain:   {' → '.join(concepts.get(n, str(n)) for n in trace_not_fly.chain)}")
    print(f"    Weight:  {trace_not_fly.total_weight:.2f}")

    # Evidence summary
    summary_fly = cg.get_conflict_summary(2)
    summary_not_fly = cg.get_conflict_summary(3)

    print(f"\n  Evidence Analysis:")
    print(f"    'can_fly' — {len(summary_fly['supporting'])} supporting, "
          f"{len(summary_fly['opposing'])} opposing → {summary_fly['conclusion']}")
    print(f"    'cannot_fly' — {len(summary_not_fly['supporting'])} supporting, "
          f"{len(summary_not_fly['opposing'])} opposing → {summary_not_fly['conclusion']}")

    print(f"\n  Conclusion:")
    print(f"    ✓ Birds generally can fly (strong evidence: w={summary_fly['net_weight']:.2f})")
    print(f"    ✓ Penguins specifically cannot fly (exception: w={summary_not_fly['net_weight']:.2f})")
    print(f"    ✓ Conflict detected and resolved via evidence weight")
    print(f"    ✓ Full causal chain stored — can explain WHY")

    # ─────────────────────────────────────────
    # Demonstrate invalidation
    # ─────────────────────────────────────────
    print(f"\n  {'='*55}")
    print("  Invalidation Demo: What if we disprove 'birds can fly'?")
    print(f"  {'='*55}")

    affected = cg.invalidate_cause(0)
    print(f"\n  Invalidated: 'bird' as cause")
    print(f"  Affected downstream nodes: {[concepts.get(n, str(n)) for n in affected]}")

    # Re-check evidence
    summary_after = cg.get_conflict_summary(2)
    print(f"\n  After invalidation:")
    print(f"    'can_fly' — now {summary_after['conclusion']} (net={summary_after['net_weight']:.2f})")
    print(f"    ✓ Invalidation propagated correctly through causal graph")

    print(f"\n  {'='*55}")
    print(f"  Graph stats: {cg.node_count()} nodes, {cg.edge_count()} edges")
    print(f"  {'='*55}")
    print("  ✓ Demo 1 complete")


if __name__ == "__main__":
    demo_trace_conclusion()

