# Flux-Apex-V1.flx — Complete Model Reference

**FLUX Architecture — Flagship Model Documentation**  
*For AI Agents, Developers, and Integration Systems*  
*Version: 4.0-multi-modal-enhanced | Phase: 12 | Created: March 30, 2026*

---

## Quick Reference

| Property | Value |
|----------|-------|
| **Filename** | `Flux-Apex-V1.flx` |
| **Location** | `checkpoints/Flux-Apex-V1.flx` |
| **HuggingFace** | `UnseenGAP/FLUX` → `checkpoints/Flux-Apex-V1.flx` |
| **Format** | FLUX (self-describing cognitive architecture) |
| **Version** | 4.0-multi-modal-enhanced |
| **Phase** | phase12 |
| **File Size** | 5,793.9 MB (5.79 GB) |
| **Total Parameters** | 1,904,320,314 (1.9B) |
| **Total Tensors** | 652 |
| **Total Configs** | 14 |
| **Top-Level Keys** | 25 |
| **Max Nesting Depth** | 4 |
| **Can Continue Learning** | `True` |

---

## What This Model IS

Flux-Apex-V1 is a **complete, self-describing cognitive architecture** — not just weights. It contains:

1. **Wave-based encoding** (no tokenizer, raw UTF-8 bytes)
2. **Resonance field** (96³ × 512 knowledge storage)
3. **Three-tier memory** (working, episodic, semantic)
4. **Byte-level decoder** (GRU-based text generation)
5. **Multi-modal adapters** (grid, image, audio)
6. **Causal reasoning** (CGN nodes + arrow graph)
7. **Spatial memory** (curiosity-driven exploration)
8. **LLM/VLM references** (external HuggingFace models)

---

## What This Model is NOT

- **NOT** an integrated LLM — external LLM loaded separately at runtime
- **NOT** an integrated VLM — external VLM (Qwen2.5-VL) loaded separately
- **NOT** a traditional transformer — uses physics-inspired wave mechanics
- **NOT** a tokenizer-based model — operates on raw bytes

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

## Top-Level Keys (25)

| # | Key | Type | Description |
|---|-----|------|-------------|
| 1 | `format` | str | `"FLUX"` — format identifier |
| 2 | `version` | str | `"4.0-multi-modal-enhanced"` |
| 3 | `phase` | str | `"phase12"` |
| 4 | `runtime_config` | dict | 7 subsections for runtime behavior |
| 5 | `components` | dict | 16 boolean flags for enabled components |
| 6 | `timestamp` | str | ISO format creation time |
| 7 | `can_continue_learning` | bool | `True` — supports online learning |
| 8 | `metadata` | dict | 15 entries (lineage, tests, capabilities) |
| 9 | `cse` | dict | Continuous Semantic Encoder weights |
| 10 | `grid_to_wave` | dict | ARC grid encoder |
| 11 | `field` | dict | Resonance field + gravity + thermodynamic state |
| 12 | `memory` | dict | Working, episodic, semantic memories |
| 13 | `decoder` | dict | Byte-level text decoder |
| 14 | `llm` | dict | External LLM configuration (no weights) |
| 15 | `causal` | dict | CGN state + causal graph |
| 16 | `bridges` | dict | Wave↔Field projections + router |
| 17 | `adapters` | dict | Multi-modal adapters (6 types) |
| 18 | `spatial_memory` | dict | Exploration/curiosity fields |
| 19 | `causal_tracker` | dict | 463 causal links |
| 20 | `learned_rules` | dict | 10 induced rules |
| 21 | `llm_reference` | dict | VLM reference config |
| 22 | `grid_adapters` | dict | Duplicate encoder (legacy) |
| 23 | `modified` | bool | `True` — model has been modified |
| 24 | `modified_components` | list | 3 recently modified components |
| 25 | `state` | dict | Runtime state snapshots |

---

## Component Parameters (Sorted by Size)

| # | Component | Parameters | % of Total | Tensors |
|---|-----------|------------|------------|---------|
| 1 | **field** | 1,366,229,131 | 71.7% | 45 |
| 2 | **memory** | 910,997,255 | 47.8% | 27 |
| 3 | **bridges** | 458,118,119 | 24.1% | 46 |
| 4 | **decoder** | 65,057,792 | 3.4% | 33 |
| 5 | **causal** | 58,838,360 | 3.1% | 397 |
| 6 | **adapters** | 15,412,331 | 0.8% | 53 |
| 7 | **cse** | 2,674,528 | 0.1% | 22 |
| 8 | **grid_to_wave** | 384,512 | <0.1% | 13 |
| 9 | **grid_adapters** | 192,256 | <0.1% | 13 |
| 10 | **spatial_memory** | 24,576 | <0.1% | 3 |

> **Note:** Percentages sum to >100% because `memory.semantic` and `bridges.router` contain copies of field state.

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

### 4. Decoder — Byte-Level Text Generation

**Purpose:** Generate text byte-by-byte from field context.

| Property | Value |
|----------|-------|
| Parameters | 32,528,896 |
| Tensors | 33 |
| Hidden Dim | 1024 |
| Layers | 4 (GRU) |
| Vocab Size | 257 (256 bytes + EOS) |

**Architecture:**
```
Input Byte → byte_embed (257×256)
           ↓
Context → context_proj (512→1024→4096)
           ↓
         GRU (4 layers, 1024 hidden)
           ↓
         cross_attn (wave context)
           ↓
         output_proj (1024→256)
           ↓
Output Byte Distribution
```

**Key Tensors:**
```
context_proj.2         [4096, 1024]     4,194,304 params
gru.weight_hh_l0       [3072, 1024]     3,145,728 params (× 4 layers)
cross_attn.in_proj     [3072, 1024]     3,145,728 params
```

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
    'llm_primary': True,              # Use external LLM for generation
    'byte_decoder_enabled': True,     # FLUX decoder available
    'byte_decoder_learns_from_llm': True,
    'generation_mode': 'llm'          # 'llm' | 'byte' | 'hybrid'
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

### External LLM Settings
```python
{
    'model_name': 'Qwen/Qwen2.5-3B-Instruct',
    'quantization': '4bit',
    'max_tokens': 512,
    'temperature': 0.7,
    'use_flux_context': True,
    'flux_context_limit': 10
}
```

---

## External Dependencies

### LLM (Text Generation)
```python
{
    'name': 'Qwen/Qwen2.5-3B-Instruct',
    'load_in_4bit': True
}
```
- **NOT stored in .flx** — downloaded from HuggingFace at runtime
- Parameters: ~3B (quantized to ~1.5GB)

### VLM (Vision-Language)
```python
{
    'name': 'Qwen/Qwen2.5-VL-3B-Instruct',
    'load_in_4bit': True,
    'enable_vision': True,
    'enable_audio': False
}
```
- **NOT stored in .flx** — downloaded from HuggingFace at runtime
- Parameters: ~3B (quantized to ~1.5GB)

---

## Enabled Components

| Component | Enabled | Has Weights |
|-----------|---------|-------------|
| cse | ✓ | ✓ |
| grid_to_wave | ✓ | ✓ |
| spatial_memory | ✓ | ✓ |
| perception_field | ✗ | ✗ |
| field | ✓ | ✓ |
| working_memory | ✓ | ✓ |
| episodic_memory | ✓ | ✓ |
| semantic_memory | ✓ | ✓ |
| decoder | ✓ | ✓ |
| llm | ✓ | ✗ (external) |
| causal_tracker | ✓ | ✓ |
| rule_inducer | ✗ | ✗ |
| goal_planner | ✗ | ✗ |
| causal_graph | ✓ | ✓ |
| bridges | ✓ | ✓ |
| learned_rules | ✓ | ✓ |

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
| 8 | Byte Decoder | ✓ |
| 8.5 | Grid Adapters | ✓ |
| 8.8 | Spatial Memory | ✓ |
| 8.9 | Causal Tracker + Rules | ✓ |
| 10 | Hybrid LLM Integration | ✓ |
| 11 | Multi-Modal Adapters | ✓ |
| 12 | Unified Agent | ✓ |

---

## Metadata

```python
{
    'created': '2026-03-29T06:36:42.571848',
    'base': 'Flux-X-complete.flx',
    'field_source': 'Flux-capable.flx',
    'grid_encoder': 'gridtowave_contrastive.pt',
    'llm_config': 'Flux-augmented.flx',
    'cse_test': 'PASS',
    'gtw_test': 'PASS',
    'field_test': 'PASS',
    'memory_test': 'PASS',
    'phase': 12,
    'description': 'FLUX Multi-Modal Agent with Visual Reasoning',
    'saved': '2026-03-29T19:30:02.828863',
    'capabilities': ['text', 'grid', 'image', 'audio', 'vision'],
    'last_modified': '2026-03-30T00:42:59.212371',
    'modified_components': ['causal_tracker', 'learned_rules', 'spatial_memory']
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
decoder_weights = raw['decoder']['state_dict']
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
| adapters.wave_to_grid.spatial_expand.weight | [57600, 256] | 14,745,600 |
| decoder.state_dict.context_proj.2.weight | [4096, 1024] | 4,194,304 |
| decoder.state_dict.gru.weight_hh_l* | [3072, 1024] | 3,145,728 each |
| causal.cgn_state.state_dict.*.manifold.metric_L | [512, 512] | 262,144 each |

---

## Wave Dimension Invariant

**All waves are 432-dimensional.** This is the universal semantic space:

- CSE outputs: `[seq_len, 432]`
- Field input: projects 432 → 512
- Field output: projects 512 → 432
- Decoder input: projects 432 → 1024
- All adapters: convert modality → 432 or 432 → modality

---

## Memory Layout

```
┌─────────────────────────────────────────────────────────────┐
│                    Flux-Apex-V1.flx (5.79 GB)               │
├─────────────────────────────────────────────────────────────┤
│  field.state [96³×512]           │ 1.7 GB (×2 copies)       │
│  memory.semantic.state           │ (shared with field)      │
│  bridges.router.semantic.state   │ (shared with field)      │
├──────────────────────────────────┼──────────────────────────┤
│  causal.cgn_state (56 nodes)     │ 112 MB                   │
│  decoder.state_dict (GRU)        │ 124 MB                   │
│  adapters (5 types)              │ 59 MB                    │
│  cse.state_dict                  │ 5 MB                     │
│  bridges.projections             │ 2 MB                     │
│  configs + metadata              │ <1 MB                    │
└──────────────────────────────────┴──────────────────────────┘
```

---

## Licensing & Attribution

- **Model:** MIT License
- **Author:** Dectrick Antonio McGee (UnseenGAP)
- **Repository:** https://github.com/Unseengap/FLUX
- **HuggingFace:** https://huggingface.co/UnseenGAP/FLUX

---

*This documentation is auto-generated from model introspection.*  
*Last updated: March 30, 2026*
