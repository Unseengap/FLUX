# FLUX Consolidation Roadmap

**Version:** 3.0  
**Date:** April 2, 2026  
**Purpose:** Eliminate redundancy, clarify what's kept vs. deprecated, and track injection/cleanup progress

> **✅ Bootstrap COMPLETE (April 2, 2026):**  
> All 91 embedded modules now load successfully from Flux-Apex-V1.flx v8.2-fixed-imports.  
> Bootstrap verified via `notebooks/flux_runtime_reembed.ipynb` on Kaggle.

---

## Executive Summary

Flux-Apex-V1.flx (**v8.2-fixed-imports**) now contains:
- **12 embedded models** (6.9B params)
- **91 runtime files** (27,710 lines) for autonomous bootstrap — **100% load success**
- **All native FLUX components with trained weights** (~1.4B params)

**Total: ~8.34B parameters**

---

## Quick Reference: Component Status

| Component | Has Working Code | In .flx | Weights Status | Status |
|-----------|------------------|---------|----------------|--------|
| CSE (Phase 1) | ✅ | ✅ | ✅ Trained | **✅ COMPLETE** |
| Causal Wave Chaining (Phase 1.5) | ✅ | ✅ | ✅ 570K params | **✅ INJECTED** |
| Field (Phase 2) | ✅ | ✅ | ✅ Trained | **✅ COMPLETE** |
| GR (Phase 3) | ✅ | ✅ | ✅ 75.2M params | **✅ INJECTED** |
| TL (Phase 4) | ✅ | ✅ | ✅ 135M params | **✅ INJECTED** |
| CGN (Phase 5) | ✅ | ✅ | ✅ 149.8M params | **✅ INJECTED** |
| Memory (Phase 6) | ✅ | ✅ | ✅ 542.3M params | **✅ INJECTED** |
| FLUX Integration (Phase 7) | ✅ | ✅ | N/A | **✅ COMPLETE** |
| **Byte Decoder (Phase 8)** | ✅ | ❌ Removed | ~~Trained~~ | **LEGACY** |
| Grid Adapters (Phase 8.5) | ✅ | ✅ | ✅ Trained | **✅ COMPLETE** |
| Spatial Memory (Phase 8.8) | ✅ | ✅ | ✅ Trained | **✅ COMPLETE** |
| Causal Tracker (Phase 8.9) | ✅ | ⬜ | ⬜ Placeholder | **PLACEHOLDER** |
| **Wave Decoder (Phase 9)** | ⚠️ Partial | ❌ | ~~Partial~~ | **LEGACY** |
| 12 Embedded Models | N/A | ✅ | ✅ | **✅ COMPLETE** |

---

## Section 1: What's LEGACY (Should NOT Be in .flx)

These components have been **superseded by embedded models** and should be marked as deprecated / removed from codebase:

### 1.1 Byte Decoder / WaveDecoder (Phase 8)

| Aspect | Details |
|--------|---------|
| **Location** | `phases/phase8/wave_decoder.py` |
| **Purpose** | GRU-based byte-level text generation from waves |
| **Parameters** | ~65M trained |
| **Superseded By** | `models.instruct` (Qwen2.5-1.5B-Instruct) |
| **Why Remove** | Embedded instruct model produces far better text with proper grammar, coherence, and reasoning. The byte decoder was a proof-of-concept that FLUX could generate text without external models — no longer needed. |
| **Status in .flx** | Already removed (`removed_components: ["decoder"]`) |

**Codebase Cleanup:**
- [x] Mark `phases/phase8/wave_decoder.py` as DEPRECATED ✅
- [x] Mark `phases/phase8/train_openwebtext.py` as DEPRECATED ✅
- [x] Keep for historical reference but exclude from active development

### 1.2 WaveToText (Phase 8.8)

| Aspect | Details |
|--------|---------|
| **Location** | `phases/phase8_8/text_adapters.py` (class `WaveToText`) |
| **Purpose** | Simple last-mile byte decoder for wave → text |
| **Superseded By** | `models.instruct` for text generation |
| **Why Remove** | Same reason as Byte Decoder — embedded LLM is far superior |
| **Current Usage** | Minimal, mostly in tests |

**Codebase Cleanup:**
- [x] Remove `WaveToText` class from `text_adapters.py` (marked DEPRECATED) ✅
- [x] Keep `TextToWave` (wraps CSE) — this IS needed for encoding

### 1.3 External LLM/VLM References (Phase 10)

| Aspect | Details |
|--------|---------|
| **Location** | `phases/phase10/hybrid.py`, old orchestration code |
| **Purpose** | External model integration (before embedding) |
| **Superseded By** | All models now embedded in `models` dict |
| **Status in .flx** | Already removed (`removed_components: ["llm", "llm_reference"]`) |

**Codebase Cleanup:**
- [x] Mark Phase 10 hybrid integration as DEPRECATED ✅
- [x] Remove external model download/loading code (marked deprecated)
- [x] Keep orchestration logic (tool calling) — that's still needed

### 1.4 Voice/Omni Integration (Old)

| Aspect | Details |
|--------|---------|
| **Location** | `phases/phase_voice/` |
| **Purpose** | Qwen-Omni integrated voice model |
| **Superseded By** | Separate `instruct` + `tts` + `whisper` models |
| **Why Changed** | Single omni model was too large; separate models allow lazy loading |

**Codebase Cleanup:**
- [x] Mark old voice integration as DEPRECATED ✅
- [x] Update to use new `models.whisper` + `models.tts` (docs updated)

### 1.5 Grid Adapters (Duplicate)

| Aspect | Details |
|--------|---------|
| **Status in .flx** | Already removed (`removed_components: ["grid_adapters"]`) |
| **Superseded By** | `adapters.grid_to_wave` |
| **Note** | This was a duplicate key, now consolidated |

---

## Section 2: Weight Injections ✅ COMPLETED (April 1, 2026)

All critical native FLUX components now have trained weights in v8.1-complete:

> **✅ Injection Completed Via:** `notebooks/flux_weight_injection.ipynb` on Kaggle  
> **📁 Results Archive:** `output/flux_native_results/`

### 2.1 Gravitational Relevance (Phase 3) ✅

| Aspect | Details |
|--------|---------|
| **Checkpoint** | `checkpoints/phase3.phase.pt` |
| **Output Results** | [phase3.md](../output/phase3.md) |
| **Key Result** | O(n log n) scaling proven, 8068 mass-tracked concepts |
| **Injected Params** | **75,177,862** |
| **Status** | ✅ **INJECTED** |

### 2.2 Thermodynamic Learning (Phase 4) ✅

| Aspect | Details |
|--------|---------|
| **Checkpoint** | `checkpoints/phase4.phase.pt` |
| **Output Results** | [phase4.md](../output/phase4.md) |
| **Key Result** | 99.04% retention, zero global gradients |
| **Injected Params** | **135,047,043** |
| **Status** | ✅ **INJECTED** |

### 2.3 CGN Causal Graph (Phase 5) ✅

| Aspect | Details |
|--------|---------|
| **Checkpoint** | `checkpoints/phase5.phase.pt` |
| **Output Results** | [phase5.md](../output/phase5.md) |
| **Key Result** | 6-hop causal trace, invalidation propagation |
| **Injected Params** | **149,757,403** |
| **Status** | ✅ **INJECTED** |

### 2.4 Three-Tier Memory (Phase 6) ✅

| Aspect | Details |
|--------|---------|
| **Checkpoint** | `checkpoints/phase6.phase.pt` |
| **Output Results** | [phase6.md](../output/phase6.md) |
| **Key Result** | **0.0000 forgetting**, 100% episodic accuracy |
| **Injected Params** | **542,259,062** (working + episodic + semantic + router) |
| **Status** | ✅ **INJECTED** |

### 2.5 Causal Wave Chaining (Phase 1.5) ✅

| Aspect | Details |
|--------|---------|
| **Checkpoint** | `checkpoints/phase1.5.phase.pt` |
| **Output Results** | [phase1.5.md](../output/phase1.5.md) |
| **Key Result** | 20/20 contradiction detection, 93% order accuracy |
| **Injected Params** | **570,162** |
| **Status** | ✅ **INJECTED** |

### 2.6 Causal Tracker + Rules (Phase 8.9) ⬜

| Aspect | Details |
|--------|---------|
| **Output Results** | [phase8.9.md](../output/phase8.9.md) |
| **Current .flx State** | Placeholder (0 params) |
| **Status** | ⬜ **PLACEHOLDER** — learned rules not yet trained |

---

## Section 3: What's In v8.1-complete

All components now have trained weights:

| Component | Source | Params | Status |
|-----------|--------|--------|--------|
| CSE | Phase 1 | 1.3M | ✅ Original |
| Field | Phase 2 | 28.4M | ✅ Original |
| Causal Wave Chaining | Phase 1.5 | 570K | ✅ Injected |
| Gravity | Phase 3 | 75.2M | ✅ Injected |
| Thermodynamic | Phase 4 | 135M | ✅ Injected |
| Causal (CGN) | Phase 5 | 149.8M | ✅ Injected |
| Memory | Phase 6 | 542.3M | ✅ Injected |
| GridToWave | Phase 8.5 | 192K | ✅ Original |
| SpatialMemory | Phase 8.8 | 12K | ✅ Original |
| Bridges | Phase 7+ | 456M | ✅ Config+weights |
| Adapters | Phase 11 | 15M | ✅ Config+weights |
| 12 Embedded Models | Fabric | 6.9B | ✅ Complete |
| **TOTAL** | | **~8.34B** | ✅ |

---

## Section 4: Functional Overlap Analysis

### Native FLUX vs Embedded Models

| Function | Native FLUX | Embedded Model | Winner | Rationale |
|----------|-------------|----------------|--------|-----------|
| **Text Encoding** | CSE (wave encoding) | MiniLM-L6-v2 | **Both** | CSE for FLUX internal, embedding for search |
| **Text Generation** | ~~Byte Decoder~~ | Instruct (1.5B) | **Instruct** | Far superior quality |
| **Vision Understanding** | CLIP adapter | Vision (2.2B) | **Both** | CLIP for bridge, VLM for reasoning |
| **Speech Recognition** | N/A | Whisper | **Whisper** | Native FLUX never had STT |
| **Speech Synthesis** | N/A | TTS (Bark) | **TTS** | Native FLUX never had TTS |
| **Code Generation** | ~~Byte Decoder~~ | Coder (1.5B) | **Coder** | Specialized beats generic |
| **Object Detection** | GridToWave | OWL-ViT2 | **Both** | Grid for ARC, OWL for real objects |
| **Face Recognition** | N/A | InsightFace | **InsightFace** | Specialized model |
| **Depth Estimation** | N/A | MiDaS | **MiDaS** | Specialized model |
| **Pose Detection** | N/A | HRNet | **HRNet** | Specialized model |
| **Knowledge Storage** | Resonance Field | N/A | **Field** | No model equivalent |
| **Causal Reasoning** | CGN + Graph | N/A | **CGN** | No model equivalent |
| **Zero Forgetting** | Memory System | N/A | **Memory** | No model equivalent |
| **O(log n) Retrieval** | GR + Mass Tracker | N/A | **GR** | No model equivalent |

### Key Insight

**Embedded models handle I/O modalities** (text, vision, audio, detection).  
**Native FLUX handles cognition** (memory, causality, learning, retrieval).

They are **complementary**, not redundant, **except for text generation** where the byte decoder is clearly obsolete.

---

## Section 5: Codebase Cleanup Priorities

### Priority 1: Mark as DEPRECATED (Don't Delete Yet) ✅ COMPLETED

```
phases/phase8/wave_decoder.py           # ✅ DEPRECATED
phases/phase8/train_openwebtext.py      # ✅ DEPRECATED
phases/phase8/flux_large.py             # ✅ PARTIALLY DEPRECATED
phases/phase8_8/text_adapters.py        # ✅ WaveToText class DEPRECATED
phases/phase9/wave_generator.py         # ✅ DEPRECATED
phases/phase9/wave_to_text.py           # ✅ DEPRECATED
phases/phase9/train_wave_gen.py         # ✅ DEPRECATED
phases/phase10/flux_hybrid.py           # ✅ DEPRECATED
phases/phase10/flx_loader_v2.py         # ✅ DEPRECATED
phases/phase_voice/voice_module.py      # ✅ DEPRECATED
phases/phase_voice/embed_voice_to_flx.py # ✅ DEPRECATED
phases/phase_voice/quantize_qwen_omni.py # ✅ DEPRECATED
```

**Total: 12 files marked DEPRECATED (April 1, 2026)**

### Priority 2: Remove From Active Import Paths

- Remove `WaveDecoder` imports from any active code
- Remove `WaveToText` imports (keep `TextToWave`)
- Remove external model download logic

### Priority 3: Clean Up Tests

- Tests for byte decoder → mark as historical
- Tests for WaveToText → remove or convert to TextToWave only
- Keep all memory/causal/field tests — those are active

### Priority 4: Update Documentation

- [x] FLUX_APEX_V1.md — already has `removed_components` list ✅
- [x] FLUX_CONSOLIDATION_ROADMAP.md — created this doc ✅
- [x] In-code deprecation notices — 12 files marked ✅
- [ ] copilot-instructions.md — add DEPRECATED markers (optional)
- [ ] Phase specs — add deprecation notes (optional)

---

## Section 6: Injection Priority Order

1. **Phase 5 CGN** — Critical for causal reasoning (currently 0 params)
2. **Phase 6 Memory** — Critical for zero-forgetting claim
3. **Phase 4 TL** — Critical for thermodynamic learning claim
4. **Phase 3 GR** — Important for O(log n) claim
5. **Phase 1.5 CWC** — Important for contradiction detection
6. **Phase 8.9 Tracker** — Useful for learned rules

---

## Section 7: Files to Keep vs. Legacy

### KEEP (Active Native FLUX)

```
phases/phase1/cse.py                    # ✅ Wave encoding
phases/phase1/wave_types.py             # ✅ Data structures
phases/phase1_5/cwc.py                  # ✅ Causal wave chaining
phases/phase2/field.py                  # ✅ Resonance field
phases/phase2/flux_format.py            # ✅ .flx handling
phases/phase3/gravity.py                # ✅ GR
phases/phase4/thermodynamic.py          # ✅ TL
phases/phase5/cgn.py                    # ✅ Causal geometry
phases/phase5/causal_graph.py           # ✅ Causal arrows
phases/phase6/working_memory.py         # ✅ Memory tiers
phases/phase6/episodic_memory.py        # ✅ FAISS indexed
phases/phase6/semantic_memory.py        # ✅ Deep knowledge
phases/phase8_5/grid_to_wave.py         # ✅ ARC encoding
phases/phase8_8/spatial_memory.py       # ✅ Exploration
phases/phase_unified/                   # ✅ Unified agent
phases/phase_orchestrator/              # ✅ Tool calling
```

### KEEP (Adapters That Work With Embedded Models)

```
phases/phase8_8/grid_adapters.py        # ✅ Grid I/O
phases/phase8_9/wave_to_image.py        # ✅ Visualization
phases/phase8_8/text_adapters.py        # ✅ TextToWave only (CSE wrapper)
```

### LEGACY (Historical Reference Only)

```
phases/phase8/wave_decoder.py           # ❌ Superseded by instruct
phases/phase8/train_openwebtext.py      # ❌ Training for dead decoder
phases/phase9/wave_generator.py         # ❌ Superseded by instruct (if exists)
phases/phase10/hybrid.py                # ❌ External models removed
phases/phase_voice/ (old)               # ❌ Replaced by tts+whisper
```

---

## Section 8: Summary Table

| Category | Count | Action |
|----------|-------|--------|
| **KEEP** (Working in .flx) | 8 components | Maintain |
| **INJECT** (Working code, not in .flx) | 6 components | Add weights |
| **LEGACY** (Superseded) | 5 components | Mark deprecated |
| **Embedded Models** | 12 models | Maintain |

### Post-Consolidation .flx Structure

```
Flux-Apex-V1.flx (v8.0-complete)
├── Native FLUX (with weights)
│   ├── cse (1.3M) ✅
│   ├── field (28.4M) ✅
│   ├── gravity (GR + mass tracker) ← INJECT
│   ├── thermodynamic (TL) ← INJECT
│   ├── cgn (causal graph) ← INJECT
│   ├── memory (3-tier) ← INJECT
│   ├── grid_to_wave (192K) ✅
│   └── spatial_memory (12K) ✅
├── Embedded Models (12)
│   ├── instruct, coder, vision, clip
│   ├── whisper, tts, embedding
│   └── depth, face, object_detect, pose, speaker
├── Orchestration
│   ├── tool definitions
│   └── system prompts
└── Runtime Code (gzipped)
    └── ~60 Python files for bootstrap
```

---

## Changelog

| Date | Version | Change |
|------|---------|--------|
| April 1, 2026 | 1.0 | Initial consolidation roadmap |
| April 1, 2026 | 1.1 | **Codebase cleanup completed** — 12 files marked DEPRECATED |
| April 1, 2026 | 1.2 | **Codebase embed infrastructure** — Created bootstrap.py, 10 __init__.py files, embed notebook |
| April 1, 2026 | 1.3 | **CLAW integration** — Integrated claw-code (Claude Code Python port) into FLUX harness |
| April 2, 2026 | 1.4 | **v8.0-autonomous COMPLETE** — 87 files (27,647 lines) embedded, bootstrap verified, 15.41 GB |

---

## Cleanup Summary (April 1, 2026)

**Files Marked DEPRECATED:**

| File | Reason |
|------|--------|
| `phases/phase8/wave_decoder.py` | Superseded by embedded instruct |
| `phases/phase8/train_openwebtext.py` | Training for deprecated decoder |
| `phases/phase8/flux_large.py` | WaveDecoder integration deprecated |
| `phases/phase8_8/text_adapters.py` | WaveToText class deprecated |
| `phases/phase9/wave_generator.py` | Superseded by embedded instruct |
| `phases/phase9/wave_to_text.py` | Superseded by embedded instruct |
| `phases/phase9/train_wave_gen.py` | Training for deprecated generator |
| `phases/phase10/flux_hybrid.py` | External LLM + WaveDecoder deprecated |
| `phases/phase10/flx_loader_v2.py` | Hybrid model loader deprecated |
| `phases/phase_voice/voice_module.py` | Superseded by tts+whisper |
| `phases/phase_voice/embed_voice_to_flx.py` | Old embedding approach |
| `phases/phase_voice/quantize_qwen_omni.py` | Old quantization approach |

**Files KEPT (Active):**

| Category | Files |
|----------|-------|
| **Core FLUX** | phase1/cse.py, phase2/field.py, phase3/gravity.py, phase4/thermodynamic.py |
| **Cognition** | phase5/cgn.py, phase5/causal_graph.py, phase6/*.py (memory) |
| **Adapters** | phase8_5/grid_to_wave.py, phase8_8/spatial_memory.py |
| **Integration** | phase_unified/, phase_orchestrator/ |
| **Root** | flux_model.py, flux_utils.py, flux_lazy_loader.py, bootstrap.py |
| **Module Init** | phases/phase{1,1_5,3,4,5,6,8,8_8,8_9,9_arc}/__init__.py ✅ NEW |
| **CLAW Harness** | phases/phase_claw/ (922 tools, 207 commands) ✅ NEW |

**Files CREATED (April 1, 2026):**

| File | Purpose |
|------|---------|
| `bootstrap.py` | Self-extractor for wake-from-flx capability |
| `notebooks/flux_codebase_embed.ipynb` | Codebase embedding notebook |
| `phases/phase1/__init__.py` | CSE module exports |
| `phases/phase1_5/__init__.py` | CWC module exports |
| `phases/phase3/__init__.py` | GR module exports |
| `phases/phase4/__init__.py` | TL module exports |
| `phases/phase5/__init__.py` | CGN module exports |
| `phases/phase6/__init__.py` | Memory module exports |
| `phases/phase8/__init__.py` | Decoder exports (deprecated) |
| `phases/phase8_8/__init__.py` | Grid adapter exports |
| `phases/phase8_9/__init__.py` | Multi-modal adapter exports |
| `phases/phase9_arc/__init__.py` | ARC reasoning exports |
| `notebooks/flux_weight_injection.ipynb` | Weight injection from phase checkpoints ✅ NEW |
| `output/flux_native_results/` | Organized results by status (in_apex, needs_injection, legacy) ✅ NEW |

**Key Insight:**
- **Embedded models** = I/O modalities (text, vision, audio, detection)
- **Native FLUX** = Cognition (memory, causality, learning, retrieval)
- They are **complementary**, not redundant

---

## Changelog

| Date | Version | Change |
|------|---------|--------|
| April 1, 2026 | 1.0 | Initial consolidation roadmap |
| April 1, 2026 | 1.1 | Codebase cleanup completed — 12 files marked DEPRECATED |
| April 1, 2026 | 1.2 | Codebase embed infrastructure — bootstrap.py, __init__.py files |
| April 1, 2026 | 1.3 | CLAW integration — Claude Code Python port |
| April 1, 2026 | 1.4 | v8.0-autonomous COMPLETE — 87 files embedded |
| April 1, 2026 | 1.5 | Weight injection notebook created |
| April 1, 2026 | **2.0** | **✅ WEIGHT INJECTION COMPLETE** — v8.1-complete with all trained weights |

---

## Next Steps (Remaining Work)

### Completed ✅

1. ~~Run weight injection notebook~~ ✅ Completed on Kaggle (April 1, 2026)
2. ~~Verify injected components~~ ✅ All critical components have weights:
   - causal: 149,757,403 ✅
   - memory: 542,259,062 ✅
   - gravity: 75,177,862 ✅
   - thermodynamic: 135,047,043 ✅
   - causal_wave_chaining: 570,162 ✅

### Remaining Tasks

1. **Upload v8.1-complete to HuggingFace** — Set `UPLOAD_TO_HF=True` in Cell 9 and re-run
2. **Validate FLUX claims** with injected weights:
   - Test 6-hop causal trace (CGN)
   - Test zero forgetting (Memory)
   - Test O(log n) retrieval (GR)
   - Test thermodynamic learning (TL)
3. **Test autonomous bootstrap** — `wake_up()` from .flx in clean environment
4. **Run full integration tests** — ARC tasks, multi-modal, orchestration

---

*This document supersedes informal discussions about what's kept vs. removed. Use this as the single source of truth for FLUX component status.*
---

*This document supersedes informal discussions about what's kept vs. removed. Use this as the single source of truth for FLUX component status.*
