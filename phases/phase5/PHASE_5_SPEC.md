# Phase 5 Specification: Causal Geometry Nodes (CGN)
## Replacing Neurons with Geometry-Aware Causal Nodes

> Prerequisites: Phases 1–4 checkpoints loaded ✓
> Copilot: Open SPECIFICATION.md + this file while building.

---

## Goal

Build the CGN layer that replaces neurons with geometry-aware nodes that store causal relationships, not just mappings.

---

## What Gets Built

```
phases/phase5/
├── PHASE_5_SPEC.md          ← This file
├── cgn.py                   ← CausalGeometryNode class
├── manifold.py              ← Manifold patch operations
├── causal_graph.py          ← Causal arrow storage and tracing
├── multi_timescale.py       ← Fast/slow node coordination
├── demo_phase5_demo1.py     ← Demo: Trace why a conclusion was reached
├── demo_phase5_demo2.py     ← Demo: Fast vs slow node activation
├── test_phase5_test1.py     ← Test: Causal trace accuracy
├── test_phase5_test2.py     ← Test: Multi-timescale separation
├── test_phase5_test3.py     ← Test: Geometry computation correctness
└── RESULTS_PHASE_5.md       ← Auto-generated results
```

---

## Core Logic

### CausalGeometryNode Structure

Each CGN is not a scalar but a geometric patch:
- **Curvature:** How strongly it bends incoming signals.
- **Orientation:** What directions it is sensitive to.
- **Radius:** Breadth of its influence.
- **Causal Why:** Pointer to what caused this node to form.
- **Time Constant:** How fast this node reacts (fast=surface, slow=deep).
- **Mass:** Evidence weight.

### Causal Reasoning

Standard neurons store: "input X maps to output Y"
CGN stores: "input X maps to output Y **because** of causal relationship Z"

### Multi-Timescale Operation

- **Fast nodes (time_const ≈ 0.01):** Surface syntax patterns.
- **Medium nodes (time_const ≈ 0.1):** Semantic relationships.
- **Slow nodes (time_const ≈ 1.0):** Deep conceptual abstractions.

---

## Acceptance Criteria

- [ ] Every output has a traceable causal chain
- [ ] Fast nodes react in < 5 steps, slow nodes in > 50 steps
- [ ] Geometry computation produces correct signal bending
- [ ] Causal invalidation works (disprove cause → invalidate conclusion)
- [ ] All tests pass
