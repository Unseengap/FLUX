# FLUX — Field-based Latent Understanding eXperience
## A Novel AI Architecture Beyond Transformers

> **Mission:** Build a fundamentally new AI architecture that replaces weights with resonance fields, tokens with continuous semantic waves, and neurons with causal geometry — achieving real-time learning, zero catastrophic forgetting, and transformer-beating performance across all domains.

---

## Project Structure

```
flux/
├── README.md                    ← You are here
├── SPECIFICATION.md             ← Full technical specification
├── ROADMAP.md                   ← Phase-by-phase build plan
│
├── phases/
│   ├── phase1/                  ← Continuous Semantic Encoder
│   ├── phase2/                  ← Field State Core
│   ├── phase3/                  ← Gravitational Relevance
│   ├── phase4/                  ← Thermodynamic Learning
│   ├── phase5/                  ← Causal Geometry Nodes
│   ├── phase6/                  ← Three-Tier Memory System
│   ├── phase7/                  ← Full FLUX Integration
│   └── phase8/                  ← Scale & Benchmark vs GPT-2
│
├── shared/
│   ├── utils/                   ← Shared utilities
│   ├── data/                    ← Dataset loaders
│   └── eval/                    ← Evaluation harness
│
├── checkpoints/                 ← Saved field snapshots per phase
├── results/                     ← All RESULTS.md files per phase
└── demos/                       ← All demo scripts
```

---

## Core Philosophy

Every piece of this architecture is a deliberate break from existing standards:

| Old Concept | FLUX Replacement | Why |
|---|---|---|
| Weights/Parameters | Resonance Field Tensors | Dynamic, local, self-organizing |
| Tokens | Continuous Semantic Waves (CSW) | Natural to language, no vocab limit |
| Neurons | Causal Geometry Nodes (CGN) | Stores WHY not just WHAT |
| Attention (O(n²)) | Gravitational Relevance (O(log n)) | Physics-based, evidence-weighted |
| Backpropagation | Thermodynamic Settling | Local, real-time, no global pass |
| Static model file | Living Field Snapshot (.flx) | Picks up mid-thought on reload |

---

## Quick Start Per Phase

Each phase is self-contained and builds on the previous saved checkpoint:

```bash
# Run a phase
cd phases/phaseN
python train.py

# Run phase demo
python demo_phaseN_demo1.py

# Run phase tests
python test_phaseN_test1.py

# View results
cat RESULTS_PHASE_N.md
```

---

## Hardware Requirements

| Phase | Minimum | Recommended |
|---|---|---|
| 1–3 | CPU or any GPU | RTX 3060 |
| 4–5 | RTX 3060 | RTX 3090 / 4090 |
| 6–7 | RTX 3090 | A100 40GB |
| 8 (GPT-2 scale) | A100 40GB | 4x A100 / Kaggle TPU |

---

## Benchmark Target

**Goal:** Match or exceed GPT-2 small (117M equivalent) on:
- Perplexity (Penn Treebank, WikiText-2)
- Text generation quality (human eval)
- Few-shot task performance
- **Continual learning retention (FLUX should win decisively here)**
- **Real-time adaptation speed (FLUX should win decisively here)**

---

## Key Invariants (Never Break These)

1. **Every phase saves a checkpoint** before ending
2. **Every phase loads** the previous phase checkpoint
3. **Every phase has** at least one `demo_phaseN_demoX.py`
4. **Every phase has** at least one `test_phaseN_testX.py`
5. **Every phase produces** a `RESULTS_PHASE_N.md`
6. **No phase assumes** a clean state — always load and verify

---

## Contributing With Copilot

When using GitHub Copilot to build phases:
1. Always open `SPECIFICATION.md` first — Copilot uses open files as context
2. Open the current phase `PHASE_N_SPEC.md` alongside your code file
3. Write descriptive function signatures before letting Copilot fill in bodies
4. After each file, run the phase test before moving on
5. Commit after every passing test — never lose working state
