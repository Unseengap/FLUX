# Phase 4 Specification: Thermodynamic Learning (TL)
## Replacing Backpropagation with Local Energy Settling

> Prerequisites: Phases 1–3 checkpoints loaded ✓
> Copilot: Open SPECIFICATION.md + this file while building.

---

## Goal

Replace backpropagation with a local energy-settling process that learns in real-time without a separate training phase. The system should learn from single examples in real time without a training loop.

---

## What Gets Built

```
phases/phase4/
├── PHASE_4_SPEC.md         ← This file
├── thermodynamic.py        ← ThermodynamicLearner class
├── temperature.py          ← Temperature schedule and dynamics
├── energy_functions.py     ← Local energy computations
├── online_learner.py       ← Real-time single-sample learning
├── train_tl.py             ← Integration and training script
├── demo_phase4_demo1.py    ← Demo: Learn a fact in one shot
├── demo_phase4_demo2.py    ← Demo: Compare to SGD convergence
├── test_phase4_test1.py    ← Test: Single-shot learning retention
├── test_phase4_test2.py    ← Test: No global gradient required
├── test_phase4_test3.py    ← Test: Temperature annealing behavior
└── RESULTS_PHASE_4.md      ← Auto-generated results
```

---

## Core Principle

The system is always trying to reach minimum energy. When input arrives, it creates a perturbation. The system settling into a new minimum IS both the forward pass (producing output) and learning (updating the field) simultaneously.

```
Input → Perturbation → Field Settles → Output extracted from settled state
                     ↑
                     This settling = learning
                     No separate backward pass needed
```

---

## Acceptance Criteria

- [ ] Model learns from single example (no batch, no epochs)
- [ ] Learned fact retrievable after 100 subsequent updates
- [ ] No global gradient computation at any point (verified)
- [ ] Temperature decreases as error decreases
- [ ] Learning is faster than equivalent SGD on simple tasks
- [ ] All tests pass
