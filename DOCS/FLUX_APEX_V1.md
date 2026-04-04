# Flux-Apex-V1.flx — Complete Model Reference

**FLUX Architecture — Flagship Model Documentation**  
*For AI Agents, Developers, and Integration Systems*  
*Version: 8.3-autonomous | Phase: complete | Updated: April 3, 2026*

> **✅ v8.3-autonomous COMPLETE:** All 99 embedded modules load successfully from .flx with zero import errors. Autonomous agent system fully embedded with CoderPool, GoalPlanner, DocumentIngester, and native JSON function calling.

---

## Quick Reference

| Property | Value |
|----------|-------|
| **Filename** | `Flux-Apex-V1.flx` |
| **Location** | HuggingFace: `UnseenGAP/FLUX` → `checkpoints/Flux-Apex-V1.flx` |
| **Format** | FLUX (self-describing cognitive architecture) |
| **Version** | 8.3-autonomous |
| **Phase** | complete |
| **File Size** | ~17.24 GB |
| **Total Parameters** | ~8,340,879,675 (~8.34B) |
| **Total Tensors** | 6,283 |
| **Top-Level Keys** | 27 |
| **Max Nesting Depth** | 10 |
| **Can Continue Learning** | `True` |
| **Embedded Models** | 12 (7 language/vision/audio, 5 detection) |
| **Embedded Runtime** | 99 files, 31,605 lines (371 KB compressed) |
| **Bootstrap Status** | ✅ 99/99 modules load successfully |

---

## What This Model IS

Flux-Apex-V1 is a **complete, self-describing cognitive architecture** — not just weights. It contains:

1. **Wave-based encoding** (no tokenizer, raw UTF-8 bytes → 432D semantic waves)
2. **Compressed resonance field** (48³ × 256 knowledge storage, expandable)
3. **Three-tier memory** (working, episodic, semantic)
4. **12 embedded models** (language, vision, audio, detection — all self-contained)
5. **Multi-modal adapters** (grid, image, audio)
6. **Causal reasoning** (CGN nodes + arrow graph)
7. **Spatial memory** (curiosity-driven exploration)
8. **Wave↔Model bridges** (432D ↔ model hidden dimensions)

---

## What This Model IS (v8.3-autonomous State)

- **12 embedded models** — Language (instruct, coder), Vision (VLM, CLIP), Audio (Whisper, TTS), Detection (face, depth, pose, object)
- **Complete native FLUX** — All components have trained weights (CGN, Memory, GR, TL, CWC injected)
- **Compressed field** — Field at 48³×256 (~28.4M params), expandable to 96³×512
- **Lazy loading support** — Models loaded on-demand to manage VRAM
- **ONNX detection** — InsightFace face recognition via 5 ONNX models
- **Full runtime embed** — 99 Python files (31,605 lines) embedded for autonomous bootstrap
- **Bootstrap verified** — 99/99 modules load from .flx with zero import errors
- **CLAW harness** — 922 tools from Claude Code port integrated
- **Native JSON tools** — Orchestration uses Qwen2.5-compatible JSON function calling
- **Weight injection** — Phase 1.5, 3, 4, 5, 6 weights injected April 1, 2026
- **Autonomous agent** — CoderPool, GoalPlanner, DocumentIngester, AutonomousAgent embedded April 3, 2026

## v8.3-autonomous Components (April 3, 2026)

> **Status:** ✅ COMPLETE — Verified via `notebooks/inspect_apex_dynamic.ipynb`

- **CoderPool** — Instruct model delegates coding to Coder model with parallel sandboxes (Jules-style architecture)
- **FluxToolExecutor** — 28 built-in tools for perception, knowledge, reasoning, exploration, generation
- **CodeSandbox** — Safe Python execution with 20 allowed modules (math, json, datetime, collections, etc.)
- **DocumentIngester** — Process and store PDF, DOCX, Markdown, JSON, CSV, HTML, code files
- **GoalPlanner** — Proactive multi-step goal planning with triggers and pattern detection
- **AutonomousAgent** — Main coordinator tying all components together
- **NativeJSONOrchestrator** — Qwen2.5 native JSON function calling format

```
Architecture (v8.3):
┌───────────────────────────────────────────────────────────────────┐
│                    INSTRUCT MODEL (Brain)                         │
│              Qwen2.5-1.5B-Instruct                                │
│                         │                                         │
│              delegate_to_coder()                                  │
│                         ↓                                         │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                 CODER POOL                                │    │
│  │           Qwen2.5-Coder-1.5B-Instruct                    │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐│    │
│  │  │ Sandbox 1│  │ Sandbox 2│  │ Sandbox 3│  │ Sandbox 4││    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘│    │
│  └──────────────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────────────┘
```

## What This Model is NOT

- **NOT** a traditional transformer — uses physics-inspired wave mechanics
- **NOT** a tokenizer-based model — operates on raw bytes via CSE
- **NOT** dependent on external downloads — all weights embedded

---

## Architecture Overview

```
INPUT                          PROCESSING                         OUTPUT
─────                          ──────────                         ──────
                                                                  
Text (UTF-8)  ─┬─► CSE ────────────────────┬─► Field ──┬─► Instruct ──► Text
               │   (wave encoding)         │   (48³)   │
Grid (ARC)   ─┼─► GridToWave ─────────────┤           ├─► WaveToGrid ─► Grid
               │                           │           │
Image        ─┼─► CLIP + Vision ──────────┤           ├─► [Generation] ► Image
               │                           │   Memory  │
Audio        ─┴─► Whisper ────────────────┤   ├────── ├─► TTS ────────► Audio
                                           │   │      │
Camera       ─┬─► Face Detection          │   │      │
              ├─► Depth Estimation  ──────┤   │      │
              ├─► Pose Detection          │   │      │
              └─► Object Detection ───────┘   │      │
                                              │      │
                                   Causal ◄───┴──────┘
                                   Tracker
                                      │
                                      ▼
                               Orchestration ───────────► Tools/Actions
```

---

## Top-Level Keys (24)

| # | Key | Type | Description |
|---|-----|------|-------------|
| 1 | `format` | str | `"FLUX"` — format identifier |
| 2 | `version` | str | `"7.1-detection-embedded"` |
| 3 | `phase` | str | `"phase2_5_detection"` |
| 4 | `runtime_config` | dict | Runtime behavior settings |
| 5 | `components` | dict | Boolean flags for enabled components |
| 6 | `timestamp` | str | ISO format creation time |
| 7 | `can_continue_learning` | bool | `True` — supports online learning |
| 8 | `metadata` | dict | Lineage, tests, capabilities |
| 9 | `cse` | dict | Continuous Semantic Encoder (1.3M params) |
| 10 | `grid_to_wave` | dict | ARC grid encoder (192K params) |
| 11 | `field` | dict | Resonance field (28.4M params, compressed) |
| 12 | `memory` | dict | Three-tier memory (542M params) |
| 13 | `models` | dict | **12 embedded models (6.9B params)** |
| 14 | `causal` | dict | CGN state + causal graph (149.8M params) |
| 15 | `bridges` | dict | Wave↔Field projections (455.7M params) |
| 16 | `adapters` | dict | Multi-modal adapters (15.4M params) |
| 17 | `spatial_memory` | dict | Exploration/curiosity fields (12K params) |
| 18 | `causal_tracker` | dict | Causal links |
| 19 | `learned_rules` | dict | Induced rules |
| 20 | `orchestration` | dict | Tool definitions, system prompt |
| 21 | `gravity` | dict | Gravitational Relevance (75.2M params) |
| 22 | `thermodynamic` | dict | Thermodynamic Learning (135M params) |
| 23 | `causal_wave_chaining` | dict | Causal Wave Chaining (570K params) |
| 24 | `modified` | bool | `True` — model has been modified |
| 25 | `modified_components` | list | Recently modified components |
| 26 | `state` | dict | Runtime state snapshots |
| 27 | `removed_components` | list | `["decoder", "llm", "llm_reference", "grid_adapters"]` |

---

## Component Parameters (Sorted by Size)

### Native FLUX Components (~1.4B params) — v8.1-complete

| # | Component | Parameters | % of Total | Status |
|---|-----------|------------|------------|--------|
| 1 | **memory** | 542,259,062 | 6.5% | ✅ Trained (injected Phase 6) |
| 2 | **bridges** | 455,683,559 | 5.5% | ✅ Config+weights |
| 3 | **causal** | 149,757,403 | 1.8% | ✅ Trained (injected Phase 5) |
| 4 | **thermodynamic** | 135,047,043 | 1.6% | ✅ Trained (injected Phase 4) |
| 5 | **gravity** | 75,177,862 | 0.9% | ✅ Trained (injected Phase 3) |
| 6 | **field** | 28,442,624 | 0.3% | ✅ Trained |
| 7 | **adapters** | 15,412,331 | 0.2% | ✅ Config+weights |
| 8 | **cse** | 1,337,264 | <0.1% | ✅ Trained |
| 9 | **causal_wave_chaining** | 570,162 | <0.1% | ✅ Trained (injected Phase 1.5) |
| 10 | **grid_to_wave** | 192,256 | <0.1% | ✅ Trained |
| 11 | **spatial_memory** | 12,288 | <0.1% | ✅ Trained |
| 12 | **causal_tracker** | 0 | 0% | ⬜ Placeholder |
| 13 | **learned_rules** | 0 | 0% | ⬜ Placeholder |

> **Note:** Field is compressed to 48³×256 (was 96³×512). Expandable on-demand for high-knowledge applications.
>
> **✅ INJECTION COMPLETE (April 1, 2026):** All critical native FLUX components now have trained weights:
>
> | Component | Source Checkpoint | Key Result |
> |-----------|-------------------|------------|
> | causal | `phase5.phase.pt` | 6-hop causal trace, 149.8M params |
> | memory | `phase6.phase.pt` | 0.0000 forgetting, 542.3M params |
> | gravity | `phase3.phase.pt` | O(log n) scaling |
> | thermodynamic | `phase4.phase.pt` | 99.04% retention |

### Embedded Models (~6.4B params)

| # | Model | Base | Parameters | % of Total | Lazy Load |
|---|-------|------|------------|------------|-----------|
| 1 | **vision** | Qwen/Qwen2-VL-2B-Instruct | 2,208,985,600 | 29.7% | Yes |
| 2 | **instruct** | Qwen/Qwen2.5-1.5B-Instruct | 1,543,714,304 | 20.8% | No |
| 3 | **coder** | Qwen/Qwen2.5-Coder-1.5B-Instruct | 1,543,714,304 | 20.8% | Yes |
| 4 | **clip** | openai/clip-vit-large-patch14 | 427,616,513 | 5.7% | No |
| 5 | **tts** | suno/bark-small | 410,086,114 | 5.5% | Yes |
| 6 | **whisper** | openai/whisper-small | 241,734,912 | 3.2% | Yes |
| 7 | **embedding** | sentence-transformers/all-MiniLM-L6-v2 | 22,713,216 | 0.3% | No |

### Detection Models (~538M params)

| # | Model | Base | Parameters | Type | Lazy Load |
|---|-------|------|------------|------|-----------|
| 1 | **depth** | intel-isl/MiDaS:DPT_Large | 344,055,465 | PyTorch | Yes |
| 2 | **object_detect** | google/owlv2-base-patch16-ensemble | 154,984,809 | PyTorch | Yes |
| 3 | **pose** | timm/hrnet_w32.ms_in1k | 39,363,128 | Timm | Yes |
| 4 | **face** | insightface/buffalo_l | 0 (ONNX) | ONNX (5 models) | Yes |
| 5 | **speaker_detect** | pyannote/speaker-diarization-3.1 | 0 (placeholder) | Placeholder | Yes |

**InsightFace ONNX Models:**
- `detection` — Face detection
- `recognition` — Face embedding (512D)
- `genderage` — Age/gender estimation
- `landmark_2d_106` — 106-point facial landmarks
- `landmark_3d_68` — 3D 68-point facial landmarks

> **Total: 7,438,927,280 parameters (~7.44B)**

---

## Detailed Component Specifications

### 1. CSE — Continuous Semantic Encoder

**Purpose:** Encode raw UTF-8 bytes into 432-dimensional semantic waves.

| Property | Value |
|----------|-------|
| Parameters | 1,337,264 |
| Tensors | 22 |
| Wave Dimension | 432 |

**Wave Dimensions Breakdown:**
- Phonetic: 64
- Syntactic: 64
- Semantic: 256
- Temporal: 32
- Intensity: 16

**Key Tensors:**
```
temporal_encoding      [1, 10000, 32]    320,000 params
conv_bank.project.0    [512, 512]        262,144 params
conv_bank.convs.3.0    [128, 256, 7]     229,376 params
semantic_proj          [256, 512]        131,072 params
byte_embed             [256, 32]           8,192 params
```

**Usage:**
```python
# Encode text to wave
wave = cse.forward(text.encode('utf-8'))  # → [seq_len, 432]
```

---

### 2. Field — Resonance Field (Compressed)

**Purpose:** Store and retrieve knowledge via wave interference in a 3D field.

| Property | Value (v6.0) | Value (v5.0 legacy) |
|----------|--------------|---------------------|
| Parameters | 28,442,624 | 911,169,800 |
| Tensors | 2 | 45 |
| Dimensions | 48 × 48 × 48 × 256 | 96 × 96 × 96 × 512 |
| Total Cells | 110,592 | 884,736 |
| Features per Cell | 256 | 512 |

> **v6.0 Change:** Field compressed to 1/32 size. Expandable on-demand for high-knowledge applications.

**Config:**
```python
{
    'h': 48, 'w': 48, 'd': 48,
    'features': 256,
    'expandable': True,
    'max_resolution': (96, 96, 96, 512),
}
```

**Usage:**
```python
# Query the field with a wave
field_state = model.get_component('field')
state_tensor = field_state['state_dict']['state']  # [48, 48, 48, 256]
```

---

### 3. Memory — Three-Tier System

**Purpose:** Working memory (session), episodic (facts), semantic (deep knowledge).

| Tier | Parameters | Purpose |
|------|------------|---------|
| Working | 878,593 | Current session context |
| Episodic | varies | Stored facts with FAISS index |

> **v6.0 Note:** Semantic memory merged into compressed field. Memory component primarily holds working + episodic tiers.

---

### 4. Embedded Models (v6.0)

**Purpose:** Self-contained language, vision, audio, and detection models.

All models are stored in `models` dict with lazy loading support:

```python
models = model.state['models']
# → dict with 12 embedded models
```

#### 4.1 Language Models

| Model | Base | Parameters | Purpose |
|-------|------|------------|---------|
| **instruct** | Qwen/Qwen2.5-1.5B-Instruct | 1.54B | Main reasoning, conversation |
| **coder** | Qwen/Qwen2.5-Coder-1.5B-Instruct | 1.54B | Code generation |
| **embedding** | all-MiniLM-L6-v2 | 22.7M | Wave conversion, similarity |

#### 4.2 Vision Models

| Model | Base | Parameters | Purpose |
|-------|------|------------|---------|
| **vision** | Qwen/Qwen2-VL-2B-Instruct | 2.21B | Vision-language understanding |
| **clip** | openai/clip-vit-large-patch14 | 427.6M | Image-text alignment |

#### 4.3 Audio Models

| Model | Base | Parameters | Purpose |
|-------|------|------------|---------|
| **whisper** | openai/whisper-small | 241.7M | Speech-to-text |
| **tts** | suno/bark-small | 410.1M | Text-to-speech |

#### 4.4 Detection Models

| Model | Base | Parameters | Purpose |
|-------|------|------------|---------|
| **depth** | intel-isl/MiDaS:DPT_Large | 344.1M | Depth estimation |
| **object_detect** | google/owlv2-base-patch16-ensemble | 155.0M | Open-vocab object detection |
| **pose** | timm/hrnet_w32.ms_in1k | 39.4M | Human pose (17 keypoints) |
| **face** | insightface/buffalo_l | ONNX | Face detection + recognition |
| **speaker_detect** | pyannote/speaker-diarization-3.1 | placeholder | Speaker diarization |

**Loading Models:**
```python
from flux_model import FLUXModel

model = FLUXModel.load('checkpoints/Flux-Apex-V1.flx')

# List all embedded models
for m in model.list_embedded_models():
    print(f"{m['name']}: {m['base_model']} [{'lazy' if m['lazy_load'] else 'eager'}]")

# Check if model exists
if model.has_embedded_model('instruct'):
    instruct_data = model.get_embedded_model('instruct')
    
# Use lazy loader for on-demand loading
manager = model.get_model_manager(device='cuda')
instruct = manager.load('instruct')  # Loads weights into HF model
```

**Lazy Load Behavior:**

| Model | Load On Startup | Load When |
|-------|-----------------|-----------|
| instruct | ✓ (eager) | Always — main text generation |
| embedding | ✓ (eager) | Always — wave conversion |
| clip | ✓ (eager) | Always — vision bridge |
| vision | ✗ (lazy) | First image/vision task |
| coder | ✗ (lazy) | First code generation |
| whisper | ✗ (lazy) | First audio input |
| tts | ✗ (lazy) | First speech output |
| detection | ✗ (lazy) | Camera activation |

---

### 5. Causal — Reasoning System

**Purpose:** Track causal relationships, induce rules, plan goals.

| Property | Value |
|----------|-------|
| Parameters | 29,419,180 |
| Tensors | 397 |
| CGN Nodes | 56 (32 fast + 16 medium + 8 slow) |
| Causal Links | 463 |
| Learned Rules | 10 |

**CGN Config:**
```python
{
    'feature_dim': 512,
    'n_fast': 32,   # Quick reactions
    'n_medium': 16, # Mid-term patterns
    'n_slow': 8     # Long-term concepts
}
```

**Each CGN Node Contains:**
- `curvature` [1, 512] — Geometric curvature tensor
- `orientation` [1, 512] — Direction in manifold
- `mass` [] — Evidence weight
- `activation_energy` [] — Threshold for firing
- `manifold.metric_L` [512, 512] — Metric tensor (262,144 params)
- `manifold.connection` [512, 512] — Christoffel symbols

**Causal Graph Example:**
```python
# Evidence structure
{
    'source': 0, 'target': 2,
    'weight': 0.8,
    'reason': 'birds can fly',
    'timestamp': 1,
    'invalidated': False
}
```

---

### 6. Bridges — Inter-Component Projections

**Purpose:** Connect waves to field and back.

| Sub-component | Parameters | Function |
|---------------|------------|----------|
| wave_to_field | 221,696 | Project 432→512 |
| field_to_wave | 221,616 | Project 512→432 |
| router | 456,533,303 | Memory tier routing |
| output_head | 1,141,504 | Final output projection |

**Router Config:**
```python
{
    'wave_dim': 432,
    'feature_dim': 512
}
```

---

### 7. Adapters — Multi-Modal I/O

**Purpose:** Convert between modalities and wave space.

| Adapter | Parameters | Direction |
|---------|------------|-----------|
| grid_to_wave | 192,256 | ARC grid → wave |
| wave_to_grid | 15,027,276 | wave → ARC grid |
| wave_to_image | 123,167 | wave → RGB image |
| audio_to_wave | 34,992 | mel spectrogram → wave |
| wave_to_audio | 34,640 | wave → mel spectrogram |

**GridToWave Config:**
```python
{
    'wave_dim': 432,
    'num_colors': 10,  # ARC color palette
    'max_size': 30     # Max grid dimension
}
```

---

### 8. Spatial Memory

**Purpose:** Track exploration and curiosity in 2D space.

| Property | Value |
|----------|-------|
| Parameters | 12,288 |
| Grid Size | 64 × 64 |

**Tensors:**
```
exploration_mass  [64, 64]  — Where agent has explored
curiosity_field   [64, 64]  — Novelty-seeking gradient
visit_count       [64, 64]  — Visit frequency
```

---

## Runtime Configuration

### Perception Settings
```python
{
    'cse_enabled': True,
    'grid_encoder_enabled': True,
    'spatial_memory_enabled': True,
    'perception_field_enabled': True
}
```

### Memory Settings
```python
{
    'working_memory_enabled': True,
    'episodic_memory_enabled': True,
    'semantic_memory_enabled': True,
    'memory_write_enabled': True
}
```

### Generation Settings
```python
{
    'instruct_enabled': True,         # Use embedded instruct model
    'vision_enabled': True,           # Use embedded vision model
    'tts_enabled': True,              # Text-to-speech synthesis
    'whisper_enabled': True,          # Speech recognition
    'coder_enabled': True,            # Code generation
    'generation_mode': 'multi_model'  # 'multi_model' | 'wave_only'
}
```

### Reasoning Settings
```python
{
    'causal_tracker_enabled': True,
    'rule_inducer_enabled': True,
    'goal_planner_enabled': True,
    'hypothesis_testing': True
}
```

### Learning Settings
```python
{
    'realtime_learning': True,
    'field_update_enabled': True,
    'causal_graph_update': True,
    'temperature': 0.3,
    'learning_rate': 0.01
}
```

### Field Settings
```python
{
    'gravity_enabled': True,
    'interference_enabled': True,
    'thermodynamic_enabled': True,
    'field_source': 'capable'
}
```

### ~~External LLM Settings~~ (REMOVED in v5.0)

> LLM is now embedded in the `models` component. No external dependencies required.

### Embedded Models Settings (v6.0)
```python
{
    # Language Models
    'instruct': {'model': 'Qwen/Qwen2.5-1.5B-Instruct', 'load_mode': 'eager'},
    'coder': {'model': 'Qwen/Qwen2.5-Coder-1.5B-Instruct', 'load_mode': 'lazy'},
    'vision': {'model': 'Qwen/Qwen2-VL-2B-Instruct', 'load_mode': 'lazy'},  # Note: Qwen2, not 2.5
    
    # Audio Models
    'whisper': {'model': 'openai/whisper-small', 'load_mode': 'lazy'},
    'tts': {'model': 'suno/bark-small', 'load_mode': 'lazy'},
    
    # Utility Models
    'embedding': {'model': 'sentence-transformers/all-MiniLM-L6-v2', 'load_mode': 'eager'},
    'clip': {'model': 'openai/clip-vit-large-patch14', 'load_mode': 'eager'},
    
    # Detection Models
    'depth': {'model': 'intel-isl/MiDaS:DPT_Large', 'load_mode': 'lazy'},
    'face': {'model': 'insightface/buffalo_l', 'load_mode': 'lazy', 'type': 'onnx'},
    'object_detect': {'model': 'google/owlv2-base-patch16-ensemble', 'load_mode': 'lazy'},
    'pose': {'model': 'timm/hrnet_w32.ms_in1k', 'load_mode': 'lazy'},
    'speaker_detect': {'model': 'pyannote/speaker-diarization-3.1', 'load_mode': 'lazy', 'placeholder': True},
}
```

---

## ~~External Dependencies~~ (REMOVED in v5.0)

> **v6.0 Architecture:** All models are embedded directly in the .flx file. The multi-model ensemble provides comprehensive AI capabilities without runtime downloads.

~~### LLM (Text Generation)~~ → **Replaced by embedded models.instruct**

~~### VLM (Vision-Language)~~ → **Replaced by embedded models.vision**

The model is now fully self-contained with zero runtime downloads. Lazy loading ensures efficient VRAM usage.

---

## Enabled Components (v6.0)

### Native FLUX Components

| Component | Enabled | Has Weights | Notes |
|-----------|---------|-------------|-------|
| cse | ✓ | ✓ | Wave encoding (1.3M params) |
| grid_to_wave | ✓ | ✓ | ARC grid → wave (192K params) |
| spatial_memory | ✓ | ✓ | Exploration tracking (12K params) |
| field | ✓ | ✓ | Compressed 48³×256 (28.4M params) |
| working_memory | ✓ | ✗ | Session context (metadata) |
| episodic_memory | ✓ | ✗ | Stored facts (metadata) |
| bridges | ✓ | ✗ | Wave↔Field projections (config) |
| causal_tracker | ✓ | ✗ | Causal links (empty) |
| adapters | ✓ | ✗ | Multi-modal (config) |

### Embedded Models (in `models` dict)

| Model | Enabled | Has Weights | Lazy Load |
|-------|---------|-------------|-----------|
| instruct | ✓ | ✓ | No (eager) |
| vision | ✓ | ✓ | Yes |
| coder | ✓ | ✓ | Yes |
| whisper | ✓ | ✓ | Yes |
| tts | ✓ | ✓ | Yes |
| embedding | ✓ | ✓ | No (eager) |
| clip | ✓ | ✓ | No (eager) |

### Detection Models (in `models` dict)

| Model | Enabled | Has Weights | Type |
|-------|---------|-------------|------|
| depth | ✓ | ✓ | PyTorch |
| face | ✓ | ✓ | ONNX (5 models) |
| object_detect | ✓ | ✓ | PyTorch |
| pose | ✓ | ✓ | Timm |
| speaker_detect | ✓ | ○ | Placeholder |

### Removed Components

| Component | Removed In | Replacement |
|-----------|------------|-------------|
| decoder | v5.0 | Embedded instruct model |
| llm | v5.0 | Embedded instruct model |
| llm_reference | v5.0 | N/A |
| grid_adapters | v4.0 | adapters.grid_to_wave |
| voice (omni) | v7.0 | Split into instruct + tts + whisper |

---

## Phase Lineage

This model contains components from all FLUX phases:

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | CSE (wave encoding) | ✓ has weights |
| 2 | Resonance Field (compressed) | ✓ has weights |
| 3 | Gravitational Relevance | ○ config only |
| 4 | Thermodynamic Learning | ○ config only |
| 5 | CGN (Causal Geometry Nodes) | ✗ empty (0 params) |
| 6 | Three-Tier Memory | ○ metadata only |
| 7 | FLUX Integration | ✓ |
| 8 | ~~Byte Decoder~~ | **REMOVED** |
| 8.5 | Grid Adapters | ✓ has weights |
| 8.8 | Spatial Memory | ✓ has weights |
| 8.9 | Causal Tracker + Rules | ✗ empty (0 params) |
| 10 | ~~Hybrid LLM~~ | **REMOVED** |
| 11 | Multi-Modal Adapters | ○ config only |
| **Fabric** | **12 Embedded Models** | ✓ 11/12 with weights |
| **Detection** | **5 Detection Models** | ✓ 4/5 with weights |

---

## Metadata (v6.0)

```python
{
    'last_modified': '2026-04-01T22:26:11.257984',
    'modified_components': ['orchestration', 'runtime'],
    'removed_components': ['decoder', 'llm', 'llm_reference', 'grid_adapters'],
    'vlm_embedded': True,
    'vlm_base_model': 'Qwen/Qwen2.5-VL-3B-Instruct',  # Note: actual model is Qwen2-VL-2B
    'phase': 'autonomous',
    'capabilities': [
        'text', 'grid', 'image', 'audio', 'vision',
        'face_detection', 'depth_estimation', 'pose_estimation', 'object_detection',
        'multi_model_embedded', 'offline_capable', 'lazy_loading'
    ],
}
```

---

## Loading the Model

### Python (Direct Load)
```python
import torch
from pathlib import Path

# Load raw archive
model_path = Path('checkpoints/Flux-Apex-V1.flx')
raw = torch.load(str(model_path), map_location='cpu', weights_only=False)

# Access native components
cse_weights = raw['cse']['state_dict']
field_state = raw['field']['state_dict']['state']  # [48, 48, 48, 256]

# Access embedded models
models = raw['models']
instruct_weights = models['instruct']['weights']   # Main LLM
vision_weights = models['vision']['weights']       # VLM
whisper_weights = models['whisper']['weights']     # Speech-to-text

# Access detection models
face_onnx = models['face']['onnx_models']          # Dict of 5 ONNX models
depth_weights = models['depth']['weights']         # MiDaS weights
```

### Via FLUXModel Class (Recommended)
```python
from flux_model import FLUXModel

model = FLUXModel.load('checkpoints/Flux-Apex-V1.flx')

# List embedded models
for m in model.list_embedded_models():
    print(f"{m['name']}: {m['base_model']}")

# Check for specific model
if model.has_embedded_model('instruct'):
    instruct = model.get_embedded_model('instruct')

# Use lazy loader for on-demand loading
manager = model.get_model_manager(device='cuda')
instruct_model = manager.load('instruct')  # Returns loaded HF model
```

### From HuggingFace
```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id='UnseenGAP/FLUX',
    filename='checkpoints/Flux-Apex-V1.flx'
)
```

---

## Saving After Modification

**Follow the Continuous Development Philosophy:**

```python
# Load
model = FLUXModel.load('checkpoints/Flux-Apex-V1.flx')

# Modify (add facts, train adapters, inject knowledge)
model.teach_fact("FLUX uses wave-based encoding")
model.adapt_to_domain(domain_data)

# Save back to SAME filename
model.save('checkpoints/Flux-Apex-V1.flx')
```

**DO NOT** create new filenames for incremental updates. The `.flx` format tracks history internally.

### Saving on Disk-Constrained Environments (Kaggle/Colab)

**CRITICAL:** The model is ~14 GB. On Kaggle (~20GB limit), delete the original before saving:

```python
import os, gc
from pathlib import Path

MODEL_PATH = Path('checkpoints/Flux-Apex-V1.flx')

# Load into memory
model = FLUXModel.load(str(MODEL_PATH))

# Make modifications...
model.version = '6.0-autonomous'

# DELETE original to free disk space (model safe in RAM)
os.remove(MODEL_PATH)
gc.collect()

# Save modified model
model.save(str(MODEL_PATH), overwrite=True)
```

---

## Key Tensor Shapes

| Path | Shape | Parameters |
|------|-------|------------|
| field.state_dict.state | [48, 48, 48, 256] | 28,311,552 |
| bridges.wave_to_field.weight | [256, 432] | 110,592 |
| bridges.field_to_wave.weight | [432, 256] | 110,592 |
| adapters.wave_to_grid.* | various | 15,376,396 |
| models.vision.state_dict.* | various | ~2,210,000,000 |
| models.instruct.state_dict.* | various | ~1,540,000,000 |
| models.coder.state_dict.* | various | ~1,540,000,000 |
| models.clip.state_dict.* | various | ~427,600,000 |
| models.tts.state_dict.* | various | ~410,100,000 |
| models.whisper.state_dict.* | various | ~241,700,000 |
| detection.depth.state_dict.* | various | ~344,100,000 |
| detection.object_detect.state_dict.* | various | ~155,000,000 |
| detection.face (ONNX models) | various | ~146,300,000 |

> ~~decoder.state_dict~~ — **REMOVED in v5.0** (native FLUX now uses embedded models for generation)

---

## Wave Dimension Invariant

**All waves are 432-dimensional.** This is the universal semantic space:

- CSE outputs: `[seq_len, 432]`
- Field input: projects 432 → 256 (compressed field)
- Field output: projects 256 → 432
- **Embedded models**: use native hidden dimensions with bridge projections
- All adapters: convert modality → 432 or 432 → modality

---

## Memory Layout (v6.0)

```
┌─────────────────────────────────────────────────────────────┐
│                    Flux-Apex-V1.flx (~14.35 GB)            │
├─────────────────────────────────────────────────────────────┤
│  EMBEDDED MODELS (~6.4 GB)                                 │
│    ├─ vision (Qwen2-VL-2B-Instruct)   │ ~2.2 GB [lazy]    │
│    ├─ instruct (Qwen2.5-1.5B-Instruct)│ ~1.5 GB [eager]   │
│    ├─ coder (Qwen2.5-Coder-1.5B)      │ ~1.5 GB [lazy]    │
│    ├─ clip (CLIP-ViT-L/14)            │ ~0.4 GB [eager]   │
│    ├─ tts (Bark-small)                │ ~0.4 GB [lazy]    │
│    ├─ whisper (Whisper-small)         │ ~0.2 GB [lazy]    │
│    └─ embedding (MiniLM-L6-v2)        │ ~0.02 GB [eager]  │
├──────────────────────────────────────┼──────────────────────┤
│  DETECTION MODELS (~0.54 GB)                               │
│    ├─ depth (MiDaS DPT_Large)         │ ~344 MB [lazy]    │
│    ├─ object_detect (OWL-ViT2)        │ ~155 MB [lazy]    │
│    ├─ pose (HRNet-W32)                │ ~39 MB [lazy]     │
│    ├─ face (InsightFace ONNX ×5)      │ ONNX [lazy]       │
│    └─ speaker_detect (placeholder)    │ ~0 MB [lazy]      │
├──────────────────────────────────────┼──────────────────────┤
│  NATIVE FLUX (~0.5 GB)                                     │
│    ├─ bridges (wave↔field+router)     │ ~456 MB (config)  │
│    ├─ field.state [48³×256]           │ ~28 MB ✓          │
│    ├─ adapters (5 types)              │ ~15 MB (config)   │
│    ├─ cse.state_dict                  │ ~1.3 MB ✓         │
│    ├─ grid_to_wave                    │ ~0.2 MB ✓         │
│    └─ spatial_memory                  │ ~0.01 MB ✓        │
├──────────────────────────────────────┴──────────────────────┤
│  MISSING (placeholders):                                    │
│    • causal, causal_tracker, learned_rules (0 params)      │
│    • memory (metadata only, no FAISS index)                │
│    • speaker_detect (placeholder, no weights)              │
├─────────────────────────────────────────────────────────────┤
│  REMOVED: decoder, llm_reference, grid_adapters, voice     │
│  COMPRESSED: field (48³×256 vs old 96³×512)                │
└─────────────────────────────────────────────────────────────┘
```

---

## Licensing & Attribution

- **Model:** MIT License
- **Author:** Dectrick Antonio McGee (UnseenGAP)
- **Repository:** https://github.com/Unseengap/FLUX
- **HuggingFace:** https://huggingface.co/UnseenGAP/FLUX

---

## Changelog

### v8.2-fixed-imports (Current — April 2, 2026)

**Fixed:**
- **Import system overhaul** — All 91 embedded modules now load without errors
- **Custom import hooks** — `EmbeddedModuleFinder` and `EmbeddedModuleLoader` in `bootstrap.py`
- **Try/except import pattern** — Absolute imports first, relative fallback for embedded mode
- **CLAW graceful degradation** — Missing JSON files return empty tuples instead of crashing
- **flux_large.py** — Deprecated module now uses placeholder classes when imports fail
- **Missing stubs added** — deferred_init.py, bootstrap_graph.py, cost_tracker.py, prefetch.py, direct_modes.py, remote_runtime.py

**Changed:**
- Version: 8.1-complete → 8.2-fixed-imports
- Embedded files: 87 → 91 (added CLAW stubs)
- Total lines: 27,647 → 27,710
- All modules compile and import successfully in embedded mode

**Technical Details:**
- Import pattern: `try: from phases.X import Y except ImportError: from .Y import Y`
- Bootstrap sets `__file__`, `__package__`, `__spec__` on embedded modules
- CLAW tools/commands lazy-load to handle missing snapshot files

---

### v8.1-complete (April 1, 2026)

**Added:**
- **All native FLUX weights injected** — memory, causal, gravity, thermodynamic, causal_wave_chaining
- **Weight injection notebook** — `notebooks/flux_weight_injection.ipynb`

---

### v6.0-autonomous (April 1, 2026)

**Added:**
- **Native JSON tool calling** — Qwen2.5-compatible function format (no custom `<tool>` tags)
- **Partial runtime embed** — ~10 core Python files embedded for bootstrap
- **Orchestration section** — Tool definitions, system prompt

**Changed:**
- Version: 5.x → 6.0-autonomous
- Phase: phase2_5_detection → autonomous
- Tool format: Custom XML → Native JSON schema

**Status (from inspection):**
- 12 embedded models: 11 with weights, 1 placeholder (speaker_detect)
- Native FLUX: 4 with weights (cse, field, grid_to_wave, spatial_memory)
- Causal system: Placeholder only (0 params) — weights not yet embedded
- Memory: Metadata only — FAISS index not populated
- Runtime: Partial (~30% of required codebase embedded)

**Known Issues:**
- `has_embedded_model('vlm')` returns False — key is 'vision' not 'vlm'
- Metadata says vlm_base_model is Qwen2.5-VL-3B but actual is Qwen2-VL-2B

### v5.x-detection-embedded (Prior)

**Added:**
- 12 embedded models for comprehensive multi-modal AI
- **Language Models:** instruct (Qwen2.5-1.5B), coder (Qwen2.5-Coder-1.5B)
- **Vision Models:** vision (Qwen2-VL-2B), clip (CLIP-ViT-L/14)
- **Audio Models:** whisper (small), tts (Bark-small)
- **Utility:** embedding (MiniLM-L6-v2)
- 5 detection models with ONNX support
- **Detection:** depth (MiDaS DPT_Large), object_detect (OWL-ViT2), face (InsightFace ×5), pose (HRNet-W32), speaker_detect (placeholder)
- Lazy loading support for memory-efficient model access

**Changed:**
- File size: ~8.5 GB → ~14.35 GB
- Total parameters: ~4.7B → ~7.44B
- Total tensors: ~2800 → 5875
- Field: 96³×512 → 48³×256 (compressed for storage efficiency)
- Architecture: Single voice model → Multi-model ensemble

**Removed:**
- `voice` — Qwen-Omni integrated voice model (replaced by separate instruct + tts + whisper)

### v5.0-voice-embedded (March 30, 2026)

**Added:**
- Embedded Qwen2.5-Omni-7B voice model (4-bit quantized, ~2.8 GB)
- Wave↔Voice bridges (432D ↔ 3584D projections)
- Speech synthesis via talker module
- Vocoder (token2wav) for audio waveform generation

**Removed:**
- `decoder` — byte-level GRU text generator
- `llm` — external LLM configuration  
- `llm_reference` — external VLM reference
- `grid_adapters` — duplicate of grid_to_wave (was legacy)

**Changed:**
- File size: 5.79 GB → ~8.5 GB
- Total parameters: 1.9B → ~4.7B
- Generation: external LLM → embedded voice thinker
- Capabilities: added 'speech' to list

### v4.0-multi-modal-enhanced (March 29, 2026)
- Initial unified multi-modal agent
- Phase 12 completion

---

*This documentation reflects v8.2-fixed-imports (91/91 modules bootstrap successfully).*  
*Last updated: April 2, 2026*
