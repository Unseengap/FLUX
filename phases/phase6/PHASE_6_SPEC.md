# Phase 6 Specification: Three-Tier Memory System
## Working, Episodic, and Semantic Memory Integration

> Prerequisites: Phases 1–5 checkpoints loaded ✓
> Copilot: Open SPECIFICATION.md + this file while building.

---

## Goal

Integrate working memory, episodic memory, and semantic memory into a unified system with no catastrophic forgetting at any tier.

---

## What Gets Built

```
phases/phase6/
├── PHASE_6_SPEC.md          ← This file
├── working_memory.py        ← Rolling field window
├── episodic_memory.py       ← FAISS vector store + metadata
├── semantic_memory.py       ← Protected field core (semantic tier)
├── memory_router.py         ← Routes between tiers
├── consolidation.py         ← Episodic → Semantic distillation
├── demo_phase6_demo1.py     ← Demo: Cross-session memory
├── demo_phase6_demo2.py     ← Demo: Consolidation process live
├── demo_phase6_demo3.py     ← Demo: Zero forgetting over 1000 tasks
├── test_phase6_test1.py     ← Test: One-shot episodic write/read
├── test_phase6_test2.py     ← Test: Semantic memory protection
├── test_phase6_test3.py     ← Test: Forgetting score = 0.0
└── RESULTS_PHASE_6.md       ← Auto-generated results
```

---

## Memory Tiers

### Tier 1: Working Memory
- **Capacity:** Current field state window (rolling).
- **Persistence:** Session only.
- **Tech:** Active field region + attention over recent perturbations.

### Tier 2: Episodic Memory
- **Capacity:** Unlimited (external vector store).
- **Persistence:** Permanent.
- **Tech:** FAISS / similar for fast semantic lookup.
- **Write:** One-shot encoding and storage.

### Tier 3: Semantic Memory
- **Capacity:** Entire field state.
- **Persistence:** Permanent.
- **Update:** Only during offline consolidation.
- **Protection:** Energy barriers around mature attractors.

---

## Acceptance Criteria

- [ ] Write fact to episodic memory → retrieve correctly 100 steps later
- [ ] Semantic memory unchanged after 1000 episodic writes
- [ ] Forgetting score = 0.0 across 10 sequential task pairs
- [ ] Consolidation promotes high-frequency episodic → semantic
- [ ] Memory persists across save/load cycle
- [ ] All tests pass
