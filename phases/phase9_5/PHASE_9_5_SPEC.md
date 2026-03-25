# Phase 9.5 Specification: Fix WaveGenerator Training

> Prerequisites: Phase 9 checkpoint must exist (`checkpoints/phase9.phase.pt`).
> This is a **patch phase** — it retrains ONLY the WaveGenerator from scratch
> while keeping WaveToText, WaveChunker, and all Phase 1–7 components frozen.

---

## Motivation

Phase 9 trained three generation modules. Stage 1 (WaveToText) succeeded with
loss=0.13 and word_acc=79.8%. Stage 2 (WaveGenerator) mode-collapsed: cross-context
Wave 0 cosine = 1.000 (identical outputs for every input), loss plateaued at step 400.
Stage 3 (Joint FT) skipped 100% of 2000 steps due to bare `except Exception: continue`.

Phase 9.5 loads the working components from `phase9.phase.pt`, discards the broken
WaveGenerator state, and retrains a fresh WaveGenerator with six architectural and
training fixes.

---

## Six Root Causes and Fixes

### 1. Context Collapse (merged vectors avg cosine 0.980)
**Root cause**: CSE → wave_to_field → GR → CGN → field.query pipeline produces
near-identical 512-dim context vectors for different inputs.
**Fix**: L2-normalize merged vectors. Add Gaussian noise augmentation (σ=0.1)
during training. Use 5× contrastive loss weight with margin 0.3.

### 2. Fixed 40 Chunks from Position 0
**Root cause**: `_start = 0` hardcoded, `_MAX_CHUNKS = 40` fixed — only first
80 bytes of every document seen.
**Fix**: Random start: `_start = random.randint(0, max(0, total - max_len))`.
Variable length: `max_chunks = random.randint(5, 40)` per sample.

### 3. Batch Size = 1, 17% GPU Utilization
**Root cause**: GRU called one step at a time with `[1, 1, 864]` tensors in
a Python for-loop. Only 3.9GB/22.5GB GPU used.
**Fix**: Batch GRU forward with `[batch, seq_len, 864]`. Use `DataLoader`
with `batch_size=128`. Pad sequences and mask. Target >200 steps/s.

### 4. Scheduled Sampling Started at 0%
**Root cause**: `ss_p=0.00` for first 1600 steps. Model learned to copy
`prev_wave` and ignore context before SS kicked in.
**Fix**: Start at 50% from step 1. Interleave teacher-forced batches with
fully free-running batches.

### 5. Training-Inference Mismatch
**Root cause**: Training uses static precomputed context. Inference dynamically
re-queries the field via `query_field_attractors()`.
**Fix**: Add noise/jitter (σ=0.1) to precomputed contexts. In joint FT,
use live field queries.

### 6. Silent Error Swallowing
**Root cause**: `except Exception: _skipped += 1; continue` — Stage 3 skipped
100% of steps, reported "PASS".
**Fix**: Log first 5 exceptions with full traceback. Abort if skip rate > 10%.
Never print success markers for zero-work stages.

---

## What Gets Loaded from phase9.phase.pt (KEEP frozen)

| Component | Key in Checkpoint | Status |
|-----------|-------------------|--------|
| CSE | `cse_state_dict` | Phase 1 trained, freeze |
| ResonanceField | `field_state_dict` | 75,561 attractors, freeze |
| GR | `gr_state` | Phase 3, freeze |
| TL | `tl_state` | Phase 4, freeze |
| CGN | `cgn_state` | Phase 5, freeze |
| Memory | `working_memory_state`, etc. | Phase 6, freeze |
| Bridges | `wave_to_field_state`, `field_to_wave_state` | Patched from field, freeze |
| OutputHead | `output_head_state` | Phase 7, freeze |
| WaveChunker | `wave_chunker_state_dict` | Phase 9 trained, freeze |
| WaveToText | `wave_to_text_state_dict` | Phase 9 trained, freeze |

## What Gets Discarded and Retrained

| Component | Key in Checkpoint | Action |
|-----------|-------------------|--------|
| WaveGenerator | `wave_generator_state_dict` | DISCARD, reinit fresh |

---

## Architecture: WaveGeneratorV3

Same GRU-based architecture as Phase 9, with these additions:

```
context [512] → LayerNorm → context_projection (512→512) → context_to_hidden → GRU h0
                                                          → context_to_wave → context_wave [432]

input per GRU step: [prev_wave (432) + context_wave (432)] = [864]
GRU: input=864, hidden=512, layers=1, batch_first=True
GRU output → Dropout(0.15) → gru_to_wave → next_wave [432]
```

Key changes from v2:
- `batch_first=True` for efficient batched processing
- `context_projection`: LayerNorm + Linear(512→512) + GELU decorrelation
- Dropout on GRU output (0.15) — prevents memorization
- `forward_batch()`: processes [batch, seq, 864] in one GRU call
- `init_hidden_batch()`: batched hidden state initialization

---

## Training: train_wave_gen_v2.py

### Stage 1: WaveGenerator Training (main)
- Load phase9.phase.pt, restore all, freeze everything except fresh WaveGenerator
- Precompute with **random windows** and **variable lengths**
- L2-normalize all merged contexts + add noise (σ=0.1)
- `torch.utils.data.DataLoader` with `batch_size=128`
- Pad sequences with `torch.nn.utils.rnn.pad_sequence`
- Scheduled sampling 50% from step 1, ramp to 90%
- Contrastive loss weight = 5.0 (was 1.0)
- Train 15,000+ steps (now fast: >200 steps/s)

### Stage 2: Joint Fine-Tuning (WG + WTT)
- Unfreeze WTT alongside WG
- LR: 1e-4 for WG, 5e-5 for WTT (separate param groups)
- Proper error handling: log tracebacks, abort on >10% skip rate
- 3,000 steps

---

## Acceptance Criteria

| Metric | Phase 9 Value | Target |
|--------|---------------|--------|
| Cross-context Wave 0 cosine | 1.000 | < 0.85 |
| Hidden init cross-context cosine | 0.996 | < 0.90 |
| Free generation quality | gibberish | Recognizable English |
| GPU utilization | 17% | > 60% |
| Training speed | 17.4 step/s | > 200 step/s |
| Silently skipped steps | 100% | 0% |
| Loss trend | Plateau at step 400 | Decreasing at end |

---

## File Structure

```
phases/phase9_5/
├── PHASE_9_5_SPEC.md              ← This file
├── wave_generator_v3.py           ← Batched WaveGenerator
├── train_wave_gen_v2.py           ← Fixed training script
├── demo_phase9_5_demo1.py         ← Free generation demo
├── demo_phase9_5_demo2.py         ← Context diversity demo
├── test_phase9_5_test1.py         ← Context collapse test
├── test_phase9_5_test2.py         ← Training mechanics test
├── test_phase9_5_test3.py         ← Full pipeline generation test
├── RESULTS_PHASE_9_5.md           ← Auto-generated via PhaseResults
```

---

## Key Dimensions Reference

```python
wave_dim        = 432   # phonetic:64 + syntactic:64 + semantic:256 + temporal:32 + intensity:16
field_features  = 512
gru_hidden      = 512
gru_input       = 864   # prev_wave:432 + context:432
```
