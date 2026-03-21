# Phase 7 Specification: Full FLUX Integration
## Unified Model Integration and Text Generation

> Prerequisites: All phase checkpoints 1–6 loaded ✓
> Copilot: Open SPECIFICATION.md + this file while building.

---

## Goal

Combine all components into a single unified FLUX model. Run end-to-end text generation. Compare quality to a small LSTM baseline.

---

## What Gets Built

```
phases/phase7/
├── PHASE_7_SPEC.md          ← This file
├── flux_model.py            ← FLUXModel — full integration class
├── flux_generate.py         ← Text generation pipeline
├── flux_trainer.py          ← Unified training on text data
├── baseline_lstm.py         ← Small LSTM for comparison
├── demo_phase7_demo1.py     ← Demo: End-to-end text generation
├── demo_phase7_demo2.py     ← Demo: Real-time learning during chat
├── demo_phase7_demo3.py     ← Demo: FLUX vs LSTM quality comparison
├── test_phase7_test1.py     ← Test: Full pipeline integration
├── test_phase7_test2.py     ← Test: Generation coherence
├── test_phase7_test3.py     ← Test: All components loaded correctly
└── RESULTS_PHASE_7.md       ← Auto-generated results
```

---

## Full Model Stack

1. **Continuous Semantic Encoder (Phase 1):** Raw text → Semantic waves.
2. **Resonance Field Core (Phase 2):** Knowledge storage in field state.
3. **Gravitational Relevance (Phase 3):** O(log n) spatial retrieval.
4. **Thermodynamic Learning (Phase 4):** Backprop-free local settling.
5. **Causal Geometry Nodes (Phase 5):** Causal tracing and geometry.
6. **Three-Tier Memory (Phase 6):** Working, Episodic, Semantic.

---

## Acceptance Criteria

- [ ] Full model loads all phase checkpoints correctly
- [ ] End-to-end text generation works
- [ ] Generation quality >= small LSTM baseline (perplexity)
- [ ] Real-time learning demonstrated (new fact → immediately usable)
- [ ] .flx model file saves and loads for full inference
- [ ] All tests pass
