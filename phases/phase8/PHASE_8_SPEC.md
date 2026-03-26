# Phase 8 Specification: Scale & GPT-2 Benchmark
## Scaled FLUX vs GPT-2 Head-to-Head

> Prerequisites: Full FLUX model from Phase 7 working ✓
> Copilot: Open SPECIFICATION.md + ROADMAP.md + this file while building.

---

## Goal

Scale FLUX to GPT-2 equivalent size (~117M parameters), train on OpenWebText,
and benchmark head-to-head on standard NLP tasks AND FLUX-specific advantages.

---

## What Gets Built

```
phases/phase8/
├── PHASE_8_SPEC.md              ← This file
├── flux_large.py                ← FLUXModel + WaveDecoder (Phase 7 compatible)
├── train_openwebtext.py         ← Training on OpenWebText dataset
├── benchmark_gpt2.py            ← Full benchmark suite vs GPT-2
├── kaggle_train.py              ← Kaggle-optimized training script
├── demo_phase8_demo1.py         ← Demo: FLUX vs GPT-2 generation quality
├── demo_phase8_demo2.py         ← Demo: FLUX continual learning advantage
├── demo_phase8_demo3.py         ← Demo: FLUX speed at long sequences
├── test_phase8_test1.py         ← Test: Perplexity on Penn Treebank
├── test_phase8_test2.py         ← Test: Perplexity on WikiText-2
├── test_phase8_test3.py         ← Test: Continual learning (FLUX wins)
├── test_phase8_test4.py         ← Test: Long sequence speed (FLUX wins)
└── RESULTS_PHASE_8.md           ← Auto-generated results
```

---

## Scaled Architecture: FLUXModel (Phase 8)

Phase 8 uses the SAME FLUXModel from Phase 7 with compatible field_features=512,
so all Phase 1-7 trained weights transfer directly. Capacity scales via larger
spatial field, more CGN nodes, and the new WaveDecoder for byte generation.

| Component | Phase 7 (Base) | Phase 8 (Scaled) | Notes |
|-----------|---------------|------------------|-------|
| CSE wave_dim | 432 | 432 | Frozen Phase 1 — always 432 |
| Field dims | 64³ | 96³ | Larger field = more capacity |
| Field features | 512 | 512 | **SAME** — all weights transfer |
| wave_to_field bridge | 432→512 | 432→512 | **SAME** — all weights transfer |
| GR k_neighbors | 32 | 64 | Deeper relevance search |
| CGN fast nodes | 16 | 32 | More surface pattern capacity |
| CGN medium nodes | 8 | 16 | More semantic capacity |
| CGN slow nodes | 4 | 8 | More conceptual capacity |
| Working memory | 512 | 2048 | Longer context |
| Episodic dim | 256 | 512 | Richer fact storage |
| Output vocab | 256 | 256 | Still byte-level |
| WaveDecoder | — | ~5M params | NEW: autoregressive byte generation |

### Dimension Strategy

The CSE (Phase 1) always outputs 432-dim waves — this is a frozen component.
Phase 8 keeps field_features=512 to preserve Phase 7 compatibility.
Capacity scales via spatial field size, CGN nodes, and WaveDecoder.

```python
# CSE output (fixed):
CSE_WAVE_DIM = 432    # Frozen Phase 1 output, never changes

# Phase 8 scaling (Phase 7 compatible):
FIELD_FEATURES = 512  # SAME as Phase 7 — all trained weights transfer
FIELD_DIMS = (96, 96, 96)  # Larger field = more knowledge storage
```

---

## Training Plan

### Dataset
- **OpenWebText** (HuggingFace `openwebtext`) — ~8M documents
- Kaggle-compatible subset: first 100k documents for initial training
- Full training: stream all documents, single-pass (no epochs)

### Training Schedule
```
Phase A: Resume from Phase 7 checkpoint + scale up
Phase B: Stream OpenWebText through scaled model
         - Output head + bridges: gradient-based (lr=5e-4, cosine decay)
         - Field: thermodynamic settling (no backprop)
         - Memory: one-shot episodic writes
Phase C: Evaluation on held-out validation set
Phase D: Benchmark vs GPT-2 small
```

### Kaggle Optimization
```python
# kaggle_train.py — optimized for Kaggle P100/T4 GPUs
# - Gradient accumulation (effective batch = 32)
# - Mixed precision (fp16) where available
# - Checkpoint every 5000 steps to Kaggle output
# - Resume-safe: loads latest checkpoint if interrupted
# - Memory-efficient: stream data, don't load full dataset
# Estimated training time: 48-72 hours on dual T4
```

---

## Benchmark Suite

### Standard NLP Benchmarks
| Benchmark | Metric | Target |
|-----------|--------|--------|
| Penn Treebank | Perplexity | ≤ GPT-2 small (117M) |
| WikiText-2 | Perplexity | ≤ GPT-2 small |
| Text generation | Human eval score | ≥ GPT-2 small quality |

### FLUX-Specific Benchmarks (Where FLUX Should Win)
| Benchmark | Metric | Target |
|-----------|--------|--------|
| Continual Learning Retention | Forgetting score | FLUX > GPT-2 by 50%+ |
| One-Shot Fact Learning | Immediate recall | FLUX succeeds, GPT-2 fails |
| Long Sequence Speed (16k tokens) | Tokens/second | FLUX > GPT-2 by 5x |
| Real-Time Adaptation | Accuracy after 1 sample | FLUX adapts, GPT-2 cannot |
| Cross-Session Memory | Recall after reload | FLUX retains, GPT-2 cannot |

---

## Acceptance Criteria

- [ ] FLUXModel (Phase 8) builds successfully with Phase 7 weight transfer
- [ ] Model trains on OpenWebText subset without errors
- [ ] Penn Treebank perplexity measurable (even if not beating GPT-2)
- [ ] WikiText-2 perplexity measurable
- [ ] Continual learning retention: forgetting score < 0.05
- [ ] Long sequence speed: FLUX processes 16k tokens faster than GPT-2
- [ ] All 4 tests pass
- [ ] All 3 demos produce visual output
- [ ] Checkpoint saved as phase8.phase.pt
- [ ] Results documented in RESULTS_PHASE_8.md

---

## Definition of Success

**Minimum:** FLUX beats GPT-2 on continual learning retention (forgetting < 0.05 vs 0.3+)

**Target:** FLUX matches GPT-2 on perplexity AND wins all FLUX-specific benchmarks

**Stretch:** FLUX beats GPT-2 on perplexity while being faster at long sequences
