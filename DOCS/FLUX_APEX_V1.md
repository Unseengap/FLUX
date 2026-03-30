# Flux-Apex-V1.flx — Complete Model Reference

**FLUX Architecture — Flagship Model Documentation**  
*For AI Agents, Developers, and Integration Systems*  
*Version: 5.0-voice-embedded | Phase: Voice | Created: March 30, 2026*

---

## Quick Reference

| Property | Value |
|----------|-------|
| **Filename** | `Flux-Apex-V1.flx` |
| **Location** | `checkpoints/Flux-Apex-V1.flx` |
| **HuggingFace** | `UnseenGAP/FLUX` → `checkpoints/Flux-Apex-V1.flx` |
| **Format** | FLUX (self-describing cognitive architecture) |
| **Version** | 5.0-voice-embedded |
| **Phase** | phase_voice |
| **File Size** | ~8,500 MB (~8.5 GB) |
| **Total Parameters** | ~4,700,000,000 (~4.7B) |
| **Total Tensors** | ~3,100 |
| **Total Configs** | 18 |
| **Top-Level Keys** | 23 |
| **Max Nesting Depth** | 5 |
| **Can Continue Learning** | `True` |
| **Voice Model** | Qwen2.5-Omni-7B (4-bit quantized) |

---

## What This Model IS

Flux-Apex-V1 is a **complete, self-describing cognitive architecture** — not just weights. It contains:

1. **Wave-based encoding** (no tokenizer, raw UTF-8 bytes → 432D semantic waves)
2. **Resonance field** (96³ × 512 knowledge storage)
3. **Three-tier memory** (working, episodic, semantic)
4. **Embedded voice model** (Qwen2.5-Omni-7B — text, vision, speech in one)
5. **Multi-modal adapters** (grid, image, audio, voice)
6. **Causal reasoning** (CGN nodes + arrow graph)
7. **Spatial memory** (curiosity-driven exploration)
8. **Wave↔Voice bridges** (432D ↔ 3584D projections)

---

## What This Model IS (v5.0 Changes)

- **NOW** an integrated voice LLM — Qwen2.5-Omni-7B weights embedded (4-bit quantized)
- **REMOVED** byte decoder — replaced by voice thinker module
- **REMOVED** external LLM references — all generation is internal
- **NEW** wave↔voice bridges — bidirectional 432D ↔ 3584D projections

## What This Model is NOT

- **NOT** a traditional transformer — uses physics-inspired wave mechanics
- **NOT** a tokenizer-based model — operates on raw bytes via CSE
- **NOT** dependent on external models — fully self-contained

---

## Architecture Overview

```
INPUT                           PROCESSING                        OUTPUT
─────                           ──────────                        ──────
                                                                  
Text (UTF-8)  ─┬─► CSE ─────────────────────┬─► Field ──┬─► Decoder ──► Text
               │   (wave encoding)          │   (96³)   │
Grid (ARC)   ─┼─► GridToWave ──────────────┤           ├─► WaveToGrid ─► Grid
               │                            │           │
Image        ─┼─► [External VLM] ──────────┤           ├─► WaveToImage ► Image
               │                            │   Memory  │
Audio        ─┴─► AudioToWave ─────────────┤   ├────── ├─► WaveToAudio ► Audio
                                            │   │      │
                                            │   │      │
                                   Causal ◄─┴───┘      │
                                   Tracker             │
                                      │                │
                                      ▼                │
                               Rule Inducer ───────────┘
                               Goal Planner
```

---

## Top-Level Keys (23)

| # | Key | Type | Description |
|---|-----|------|-------------|
| 1 | `format` | str | `"FLUX"` — format identifier |
| 2 | `version` | str | `"5.0-voice-embedded"` |
| 3 | `phase` | str | `"phase_voice"` |
| 4 | `runtime_config` | dict | 8 subsections for runtime behavior |
| 5 | `components` | dict | 18 boolean flags for enabled components |
| 6 | `timestamp` | str | ISO format creation time |
| 7 | `can_continue_learning` | bool | `True` — supports online learning |
| 8 | `metadata` | dict | 18 entries (lineage, tests, capabilities) |
| 9 | `cse` | dict | Continuous Semantic Encoder weights |
| 10 | `grid_to_wave` | dict | ARC grid encoder |
| 11 | `field` | dict | Resonance field + gravity + thermodynamic state |
| 12 | `memory` | dict | Working, episodic, semantic memories |
| 13 | `voice` | dict | **Qwen2.5-Omni-7B (4-bit) — thinker, talker, token2wav** |
| 14 | `causal` | dict | CGN state + causal graph |
| 15 | `bridges` | dict | Wave↔Field + Wave↔Voice projections |
| 16 | `adapters` | dict | Multi-modal adapters (6 types) |
| 17 | `spatial_memory` | dict | Exploration/curiosity fields |
| 18 | `causal_tracker` | dict | 463 causal links |
| 19 | `learned_rules` | dict | 10 induced rules |
| 20 | `modified` | bool | `True` — model has been modified |
| 21 | `modified_components` | list | Recently modified components |
| 22 | `state` | dict | Runtime state snapshots |
| 23 | `removed_components` | list | `["decoder", "llm", "llm_reference", "grid_adapters"]` |

> **v5.0 Changes:** Removed `decoder`, `llm`, `llm_reference`, `grid_adapters`. Added `voice` with embedded Qwen-Omni weights.

---

## Component Parameters (Sorted by Size)

| # | Component | Parameters | % of Total | Tensors |
|---|-----------|------------|------------|---------|
| 1 | **voice** | ~2,800,000,000 | 59.6% | ~2,448 |
| 2 | **field** | 1,366,229,131 | 29.1% | 45 |
| 3 | **memory** | 910,997,255 | 19.4% | 27 |
| 4 | **bridges** | 460,000,000 | 9.8% | 50 |
| 5 | **causal** | 58,838,360 | 1.3% | 397 |
| 6 | **adapters** | 15,412,331 | 0.3% | 53 |
| 7 | **cse** | 2,674,528 | <0.1% | 22 |
| 8 | **grid_to_wave** | 384,512 | <0.1% | 13 |
| 9 | **spatial_memory** | 24,576 | <0.1% | 3 |

**Voice Module Breakdown:**
| Sub-component | Tensors | Purpose |
|---------------|---------|---------|
| thinker | 1,346 | Text + Vision + Audio understanding |
| talker | 293 | Speech synthesis |
| token2wav | 809 | Vocoder (speech waveform) |

> **Note:** Percentages based on ~4.7B total params. `decoder`, `llm`, `llm_reference`, `grid_adapters` removed in v5.0.

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

### 2. Field — Resonance Field

**Purpose:** Store and retrieve knowledge via wave interference in a 3D field.

| Property | Value |
|----------|-------|
| Parameters | 911,169,800 (unique) |
| Tensors | 45 |
| Dimensions | 96 × 96 × 96 × 512 |
| Total Cells | 884,736 |
| Features per Cell | 512 |

**Sub-components:**
- `state_dict` — Field state tensor [96, 96, 96, 512]
- `gravity_state` — Gravitational relevance model
- `thermodynamic_state` — Temperature, energy, settling

**Config:**
```python
{
    'h': 96, 'w': 96, 'd': 96,
    'features': 512,
    'settle_iterations': 15,
    'settle_dt': 0.1,
    'perturbation_radius': 4
}
```

**Gravity State:**
```python
{
    'feature_dim': 512,
    'k_neighbors': 64,
    'base_mass': 1.0,
    'growth_rate': 0.005,
    'negative_rate': 0.05
}
```

**Thermodynamic State:**
```python
{
    'current_temp': 0.637,
    'min_temp': 0.005,
    'max_temp': 2.0,
    'decay': 0.9999,
    'step_count': 4500,
    'total_samples': 4500,
    'total_settles': 67500
}
```

---

### 3. Memory — Three-Tier System

**Purpose:** Working memory (session), episodic (facts), semantic (deep knowledge).

| Tier | Parameters | Tensors | Purpose |
|------|------------|---------|---------|
| Working | 878,593 | 10 | Current session context (2048 window) |
| Episodic | — | — | 74 stored facts (FAISS index) |
| Semantic | 455,059,331 | 17 | Deep field-based knowledge |

**Working Memory Config:**
```python
{
    'window_size': 2048,
    'wave_dim': 432,
    'feature_dim': 512
}
```

**Episodic Memory:**
```python
{
    'vectors': numpy.ndarray,  # 74 × 512
    'metadata': [74 items],    # timestamps, content, importance
    'feature_dim': 512,
    'next_id': 74
}
```

---

### 4. Voice — Embedded Qwen2.5-Omni-7B (4-bit)

**Purpose:** Unified text, vision, and speech understanding + generation.

| Property | Value |
|----------|-------|
| Base Model | Qwen/Qwen2.5-Omni-7B |
| Quantization | 4-bit (bitsandbytes nf4) |
| Parameters | ~2,800,000,000 (quantized) |
| Tensors | ~2,448 |
| Hidden Dim | 3584 |
| Vocabulary | 151,936 tokens |

**Sub-modules:**

| Module | Tensors | Purpose |
|--------|---------|---------|
| thinker | 1,346 | Text + Vision + Audio understanding (main reasoning) |
| talker | 293 | Speech synthesis (TTS) |
| token2wav | 809 | DiT-based vocoder for waveform generation |

**Architecture:**
```
                    ┌─────────────────────────────────────┐
                    │        VOICE (Qwen-Omni)            │
                    ├─────────────────────────────────────┤
Text Input  ───────►│                                     │
                    │                                     │
Vision Input ──────►│    THINKER (1346 tensors)          │───► Text Output
                    │    - Unified understanding          │
Audio Input ───────►│    - Multi-modal reasoning          │
                    │                                     │
                    ├─────────────────────────────────────┤
                    │                                     │
                    │    TALKER (293 tensors)             │───► Speech Tokens
                    │    - Text-to-speech synthesis       │
                    │                                     │
                    ├─────────────────────────────────────┤
                    │                                     │
                    │    TOKEN2WAV (809 tensors)          │───► Audio Waveform
                    │    - DiT vocoder                    │
                    │    - Mel → Waveform                 │
                    └─────────────────────────────────────┘
```

**Wave ↔ Voice Bridges:**
```python
# Bridge projections (in bridges component)
wave_to_voice: Linear(432, 3584)   # FLUX waves → voice hidden
voice_to_wave: Linear(3584, 432)   # voice hidden → FLUX waves
```

**Config:**
```python
{
    'model_name': 'Qwen/Qwen2.5-Omni-7B',
    'quantization': '4bit',
    'wave_dim': 432,
    'hidden_dim': 3584,
    'vocab_size': 151936,
    'modules': ['thinker', 'talker', 'token2wav'],
    'capabilities': ['text', 'vision', 'audio', 'speech']
}
```

---

### ~~4. Decoder — REMOVED in v5.0~~

> **Note:** The byte-level GRU decoder has been removed in v5.0. Text generation is now handled by the embedded voice thinker module (Qwen-Omni).

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
    'voice_primary': True,            # Use embedded voice model for generation
    'voice_thinker_enabled': True,    # Text/vision/audio understanding
    'voice_talker_enabled': True,     # Speech synthesis
    'voice_token2wav_enabled': True,  # Vocoder for waveforms
    'generation_mode': 'voice'        # 'voice' | 'wave_only'
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

> LLM is now embedded as the voice module. No external dependencies required.

### Voice Settings (NEW in v5.0)
```python
{
    'model_name': 'Qwen/Qwen2.5-Omni-7B',
    'quantization': '4bit',
    'hidden_dim': 3584,
    'max_tokens': 2048,
    'temperature': 0.7,
    'use_flux_context': True,
    'flux_context_limit': 20,
    'speech_enabled': True,
    'vision_enabled': True
}
```

---

## ~~External Dependencies~~ (REMOVED in v5.0)

> **v5.0 Change:** All external dependencies have been removed. The voice model (Qwen2.5-Omni-7B) is now embedded directly in the .flx file.

~~### LLM (Text Generation)~~ → **Replaced by embedded voice.thinker**

~~### VLM (Vision-Language)~~ → **Replaced by embedded voice.thinker**

The model is now fully self-contained with zero runtime downloads.

---

## Enabled Components

| Component | Enabled | Has Weights | Notes |
|-----------|---------|-------------|-------|
| cse | ✓ | ✓ | Wave encoding |
| grid_to_wave | ✓ | ✓ | ARC grid → wave |
| spatial_memory | ✓ | ✓ | Exploration tracking |
| perception_field | ✗ | ✗ | |
| field | ✓ | ✓ | Resonance field |
| working_memory | ✓ | ✓ | Session context |
| episodic_memory | ✓ | ✓ | Stored facts |
| semantic_memory | ✓ | ✓ | Deep knowledge |
| **voice** | ✓ | ✓ | **Qwen-Omni (NEW)** |
| voice_thinker | ✓ | ✓ | Text+Vision+Audio |
| voice_talker | ✓ | ✓ | Speech synthesis |
| voice_token2wav | ✓ | ✓ | Vocoder |
| causal_tracker | ✓ | ✓ | Causal links |
| rule_inducer | ✗ | ✗ | |
| goal_planner | ✗ | ✗ | |
| causal_graph | ✓ | ✓ | Arrow graph |
| bridges | ✓ | ✓ | Wave↔Field + Wave↔Voice |
| learned_rules | ✓ | ✓ | Induced rules |
| ~~decoder~~ | ✗ | ✗ | **REMOVED** |
| ~~llm~~ | ✗ | ✗ | **REMOVED** |

---

## Phase Lineage

This model contains components from all FLUX phases:

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | CSE (wave encoding) | ✓ |
| 2 | Resonance Field | ✓ |
| 3 | Gravitational Relevance | ✓ |
| 4 | Thermodynamic Learning | ✓ |
| 5 | CGN (Causal Geometry Nodes) | ✓ |
| 6 | Three-Tier Memory | ✓ |
| 7 | FLUX Integration | ✓ |
| 8 | ~~Byte Decoder~~ | **REMOVED** |
| 8.5 | Grid Adapters | ✓ |
| 8.8 | Spatial Memory | ✓ |
| 8.9 | Causal Tracker + Rules | ✓ |
| 10 | ~~Hybrid LLM Integration~~ | **REMOVED** |
| 11 | Multi-Modal Adapters | ✓ |
| 12 | Unified Agent | ✓ |
| **Voice** | **Embedded Qwen-Omni** | **NEW** |

---

## Metadata

```python
{
    'created': '2026-03-29T06:36:42.571848',
    'base': 'Flux-X-complete.flx',
    'field_source': 'Flux-capable.flx',
    'grid_encoder': 'gridtowave_contrastive.pt',
    'voice_source': 'Qwen/Qwen2.5-Omni-7B (4-bit)',  # NEW in v5.0
    'cse_test': 'PASS',
    'gtw_test': 'PASS',
    'field_test': 'PASS',
    'memory_test': 'PASS',
    'voice_test': 'PENDING',                         # NEW in v5.0
    'phase': 'voice',
    'description': 'FLUX Cognitive Architecture with Embedded Voice',
    'saved': '2026-03-30T...',
    'capabilities': ['text', 'grid', 'image', 'audio', 'vision', 'speech'],
    'last_modified': '2026-03-30T...',
    'modified_components': ['voice', 'bridges', 'runtime_config'],
    'removed_components': ['decoder', 'llm', 'llm_reference', 'grid_adapters']
}
```

---

## Loading the Model

### Python (Recommended)
```python
import torch
from pathlib import Path

# Load raw archive
model_path = Path('checkpoints/Flux-Apex-V1.flx')
raw = torch.load(str(model_path), map_location='cpu', weights_only=False)

# Access components
cse_weights = raw['cse']['state_dict']
field_state = raw['field']['state_dict']['state']  # [96, 96, 96, 512]
voice_thinker = raw['voice']['thinker']            # Qwen-Omni thinker
voice_talker = raw['voice']['talker']              # Speech synthesis
voice_token2wav = raw['voice']['token2wav']        # Vocoder
```

### Via FLUXModel Class
```python
from flux_model import FLUXModel

model = FLUXModel.load('checkpoints/Flux-Apex-V1.flx', device='cuda')
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

---

## Key Tensor Shapes

| Path | Shape | Parameters |
|------|-------|------------|
| field.state_dict.state | [96, 96, 96, 512] | 452,984,832 |
| memory.semantic.field_state_dict.state | [96, 96, 96, 512] | 452,984,832 |
| bridges.router.module_state.semantic.field.state | [96, 96, 96, 512] | 452,984,832 |
| bridges.wave_to_voice.weight | [3584, 432] | 1,548,288 |
| bridges.voice_to_wave.weight | [432, 3584] | 1,548,288 |
| voice.thinker.model.* | various | ~2,000,000,000 |
| voice.talker.* | various | ~500,000,000 |
| voice.token2wav.* | various | ~300,000,000 |
| adapters.wave_to_grid.spatial_expand.weight | [57600, 256] | 14,745,600 |
| causal.cgn_state.state_dict.*.manifold.metric_L | [512, 512] | 262,144 each |

> ~~decoder.state_dict~~ — **REMOVED in v5.0**

---

## Wave Dimension Invariant

**All waves are 432-dimensional.** This is the universal semantic space:

- CSE outputs: `[seq_len, 432]`
- Field input: projects 432 → 512
- Field output: projects 512 → 432
- **Voice input: projects 432 → 3584** (wave_to_voice bridge)
- **Voice output: projects 3584 → 432** (voice_to_wave bridge)
- All adapters: convert modality → 432 or 432 → modality

---

## Memory Layout (v5.0)

```
┌─────────────────────────────────────────────────────────────┐
│                    Flux-Apex-V1.flx (~8.5 GB)              │
├─────────────────────────────────────────────────────────────┤
│  voice (Qwen-Omni 4-bit)         │ ~2.8 GB                  │
│    ├─ thinker (1346 tensors)    │   ~2.0 GB                │
│    ├─ talker (293 tensors)      │   ~0.5 GB                │
│    └─ token2wav (809 tensors)   │   ~0.3 GB                │
├──────────────────────────────────┼──────────────────────────┤
│  field.state [96³×512]           │ 1.7 GB (×2 copies)       │
│  memory.semantic.state           │ (shared with field)      │
│  bridges.router.semantic.state   │ (shared with field)      │
├──────────────────────────────────┼──────────────────────────┤
│  causal.cgn_state (56 nodes)     │ 112 MB                   │
│  bridges.wave_to_voice           │ 6 MB                     │
│  bridges.voice_to_wave           │ 6 MB                     │
│  adapters (5 types)              │ 59 MB                    │
│  cse.state_dict                  │ 5 MB                     │
│  bridges.projections             │ 2 MB                     │
│  configs + metadata              │ <1 MB                    │
├──────────────────────────────────┴──────────────────────────┤
│  REMOVED: decoder, llm, llm_reference, grid_adapters       │
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

*This documentation reflects v5.0-voice-embedded.*  
*Last updated: March 30, 2026*
