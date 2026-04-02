# FLUX Consolidation Roadmap

**Version:** 1.5  
**Date:** April 1, 2026  
**Purpose:** Eliminate redundancy, clarify what's kept vs. deprecated, and define injection/cleanup tasks

> **NEW: Weight Injection Notebook Available!**  
> Run `notebooks/flux_weight_injection.ipynb` to inject trained weights from phases 1.5, 3, 4, 5, 6 into Flux-Apex-V1.flx  
> Results organized in `output/flux_native_results/`

---

## Executive Summary

Flux-Apex-V1.flx (v8.0-autonomous) now contains **12 embedded models** and **87 runtime files** (27,647 lines) for fully autonomous operation. This document identifies:

1. **What's proven working** (from output/ results)
2. **What's in the .flx file** (current state)
3. **What needs injection** (trained weights not yet in .flx)
4. **What's LEGACY** (superseded by embedded models)
5. **Codebase cleanup recommendations**

---

## Quick Reference: Component Status

| Component | Has Working Code | In .flx | Weights Status | Status |
|-----------|------------------|---------|----------------|--------|
| CSE (Phase 1) | ✅ | ✅ | ✅ Trained | **KEEP** |
| Causal Wave Chaining (Phase 1.5) | ✅ | ❌ | ✅ Trained | **INJECT** |
| Field (Phase 2) | ✅ | ✅ | ✅ Trained | **KEEP** |
| GR (Phase 3) | ✅ | ⚠️ Config | ✅ Trained | **INJECT** |
| TL (Phase 4) | ✅ | ⚠️ Config | ✅ Trained | **INJECT** |
| CGN (Phase 5) | ✅ | ⚠️ Empty | ✅ Trained | **INJECT** |
| Memory (Phase 6) | ✅ | ⚠️ Metadata | ✅ Trained | **INJECT** |
| FLUX Integration (Phase 7) | ✅ | ✅ | N/A | **KEEP** |
| **Byte Decoder (Phase 8)** | ✅ | ❌ Removed | ✅ Trained | **LEGACY** |
| Grid Adapters (Phase 8.5) | ✅ | ✅ | ✅ Trained | **KEEP** |
| Spatial Memory (Phase 8.8) | ✅ | ✅ | ✅ Trained | **KEEP** |
| Causal Tracker (Phase 8.9) | ✅ | ⚠️ Empty | ✅ Trained | **INJECT** |
| **Wave Decoder (Phase 9)** | ⚠️ Partial | ❌ | ⚠️ Partial | **LEGACY** |
| 12 Embedded Models | N/A | ✅ | ✅ | **KEEP** |

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

## Section 2: What MUST Be Injected

These components have **working code with trained weights** but those weights are NOT in Flux-Apex-V1.flx:

> **🔧 Injection Notebook:** `notebooks/flux_weight_injection.ipynb`  
> **📁 Results Archive:** `output/flux_native_results/needs_injection/`

### 2.1 Gravitational Relevance (Phase 3)

| Aspect | Details |
|--------|---------|
| **Checkpoint** | `checkpoints/phase3.phase.pt` |
| **Output Results** | [phase3.md](../output/phase3.md) |
| **Key Result** | O(n log n) scaling proven, 8068 mass-tracked concepts |
| **Current .flx State** | Config only, no trained weights |
| **Action** | Inject mass_tracker state, spatial_index |

### 2.2 Thermodynamic Learning (Phase 4)

| Aspect | Details |
|--------|---------|
| **Checkpoint** | `checkpoints/phase4.phase.pt` |
| **Output Results** | [phase4.md](../output/phase4.md) |
| **Key Result** | 99.04% retention, zero global gradients |
| **Current .flx State** | Config only |
| **Action** | Inject temperature state, energy functions |

### 2.3 CGN Causal Graph (Phase 5)

| Aspect | Details |
|--------|---------|
| **Checkpoint** | `checkpoints/phase5.phase.pt` |
| **Output Results** | [phase5.md](../output/phase5.md) |
| **Key Result** | 6-hop causal trace, invalidation propagation |
| **Current .flx State** | Empty (0 params!) |
| **Action** | Inject CGN nodes, causal arrows, manifolds |

### 2.4 Three-Tier Memory (Phase 6)

| Aspect | Details |
|--------|---------|
| **Checkpoint** | `checkpoints/phase6.phase.pt` |
| **Output Results** | [phase6.md](../output/phase6.md) |
| **Key Result** | **0.0000 forgetting**, 100% episodic accuracy |
| **Current .flx State** | Metadata only, no FAISS index |
| **Action** | Inject memory router, episodic index, consolidation state |

### 2.5 Causal Tracker + Rules (Phase 8.9)

| Aspect | Details |
|--------|---------|
| **Output Results** | [phase8.9.md](../output/phase8.9.md) |
| **Key Result** | All adapters working, cross-modal proven |
| **Current .flx State** | Empty (0 params) |
| **Action** | Inject learned rules from grid training |

### 2.6 Causal Wave Chaining (Phase 1.5)

| Aspect | Details |
|--------|---------|
| **Checkpoint** | `checkpoints/phase1_5.phase.pt` |
| **Output Results** | [phase1.5.md](../output/phase1.5.md) |
| **Key Result** | 20/20 contradiction detection, 93% order accuracy |
| **Current .flx State** | Not present |
| **Action** | Inject CWC weights into causal system |

---

## Section 3: What's Already in .flx and WORKING

These components are correctly embedded and should be preserved:

| Component | Source | Params | Purpose |
|-----------|--------|--------|---------|
| CSE | Phase 1 | 1.3M | Wave encoding ✅ |
| Field | Phase 2 | 28.4M | Knowledge storage ✅ |
| GridToWave | Phase 8.5 | 192K | ARC grid encoding ✅ |
| SpatialMemory | Phase 8.8 | 12K | Curiosity/exploration ✅ |
| Bridges | Phase 7+ | 456M (config) | Wave↔Field projections ✅ |
| Adapters | Phase 11 | 15M (config) | Multi-modal I/O ✅ |
| 12 Embedded Models | Phase Fabric | 6.4B+ | All AI capabilities ✅ |

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
| April 2, 2026 | 1.4 | v8.0-autonomous COMPLETE — 87 files embedded |
| April 1, 2026 | 1.5 | **Weight injection notebook created** — `flux_weight_injection.ipynb`, organized results |

---

## Next Steps (Action Required)

1. **Run weight injection notebook** on Kaggle/Colab with GPU
2. **Verify** injected components have proper parameter counts
3. **Upload** v8.1-complete to HuggingFace
4. **Validate** all FLUX claims work with injected weights

---

*This document supersedes informal discussions about what's kept vs. removed. Use this as the single source of truth for FLUX component status.*
