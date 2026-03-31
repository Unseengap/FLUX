# FLUX Project — Copilot Instructions

## Project Overview

FLUX (Field-based Latent Understanding eXperience) is a novel AI architecture that replaces traditional neural network primitives with physics-inspired components: resonance fields instead of weights, continuous semantic waves instead of tokens, gravitational relevance instead of attention (O(log n)), thermodynamic settling instead of backpropagation, and causal geometry nodes instead of neurons.

**Current Flagship Model:** `Flux-Apex-V1.flx` (v5.1-orchestrated, Phase Orchestrator, ~5.7B params)

**Source of truth:**
- `DOCS/FLUX_APEX_V1.md` — Complete Flux-Apex model reference
- `DOCS/FLUX_FILE_FORMAT.md` — .flx format specification
- `DOCS/FLUX_7B_SPEC.md` — Large-scale architecture spec
- `DOCS/PHASE_ORCHESTRATOR_SPEC.md` — VLM orchestration specification
- `flux_model.py` — FLUXModel class for loading/saving
- `flux_utils.py` — Core utilities (checkpoints, logging, HF Hub)

---

## Flux-Apex Model Architecture

**Flux-Apex-V1.flx** is the complete, self-describing cognitive architecture:

| Property | Value |
|----------|-------|
| **Location** | `checkpoints/Flux-Apex-V1.flx` |
| **HuggingFace** | `UnseenGAP/FLUX` → `checkpoints/Flux-Apex-V1.flx` |
| **File Size** | 5,793.9 MB (5.79 GB) |
| **Total Parameters** | 1,904,320,314 (1.9B) |
| **Wave Dimension** | 432 (universal semantic space) |
| **Field Dimensions** | 96 × 96 × 96 × 512 |
| **Version** | 5.1-orchestrated |

### Top-Level Components (26 keys)

| Component | Parameters | Purpose |
|-----------|------------|---------|
| `cse` | 2.7M | Continuous Semantic Encoder (bytes → 432D waves) |
| `field` | 1.36B | Resonance Field (96³ × 512 knowledge storage) |
| `memory` | 911M | Three-tier memory (working, episodic, semantic) |
| `bridges` | 458M | Wave↔Field projections + router |
| `decoder` | 65M | Byte-level text decoder (GRU-based) |
| `causal` | 59M | CGN nodes + causal arrow graph |
| `adapters` | 15M | Multi-modal (grid, image, audio) |
| `grid_to_wave` | 384K | ARC grid encoder |
| `spatial_memory` | 25K | Curiosity-driven exploration |
| `vlm` | 3.75B | Embedded Qwen2.5-VL-3B (text + vision) |
| `orchestration` | — | Self-describing tool definitions |

### Component Status Flags

Every component has an enabled/disabled flag in `components`:
```python
{
    'cse': True,
    'grid_to_wave': True,
    'field': True,
    'memory': True,
    'vlm': True,             # Embedded Qwen2.5-VL-3B
    'vlm_text': True,        # Text generation
    'vlm_vision': True,      # Vision understanding
    'orchestration': True,   # Self-describing tools
    'tool_use': True,        # VLM can call FLUX tools
    'causal_tracker': True,
    ...
}
```

---

## Legacy Component System

### Marking Components as Legacy

When a component is deprecated but not yet removed, mark it as legacy:

```python
# In the .flx state, add legacy flag
model.state['grid_adapters'] = {
    'state_dict': ...,
    'config': ...,
    'legacy': True,                    # ← REQUIRED: Mark as legacy
    'legacy_reason': 'Replaced by adapters.grid_to_wave in v4.0',
    'legacy_since': '2026-03-30',
    'removal_target': 'v5.0',          # When it will be removed
}
```

### Legacy Component Rules

1. **Always set `legacy: True`** when deprecating a component
2. **Include `legacy_reason`** explaining what replaced it
3. **Include `legacy_since`** with ISO date
4. **Include `removal_target`** for planned removal version
5. **AI agents must check for `legacy` flag** before using components
6. **Legacy components should NOT be used in new code**

### Checking for Legacy Components

```python
def is_legacy(model: FLUXModel, component_name: str) -> bool:
    """Check if a component is marked as legacy."""
    comp = model.state.get(component_name, {})
    return comp.get('legacy', False)

def get_legacy_components(model: FLUXModel) -> List[str]:
    """Get all legacy components."""
    return [
        name for name, data in model.state.items()
        if isinstance(data, dict) and data.get('legacy', False)
    ]
```

### Current Legacy Components

| Component | Legacy Since | Replacement | Removal Target |
|-----------|--------------|-------------|----------------|
| `grid_adapters` | 2026-03-30 | `adapters.grid_to_wave` | v5.0 |

---

## Loading Flux-Apex Model

### Method 1: FLUXModel Class (Recommended)

```python
from flux_model import FLUXModel

# Load from local path
model = FLUXModel.load('checkpoints/Flux-Apex-V1.flx')

# Load with config override
model = FLUXModel.load(
    'checkpoints/Flux-Apex-V1.flx',
    config_override={'generation': {'llm_primary': False}}
)

# Access components
cse_state = model.get_component('cse')
field_state = model.get_component('field')
```

### Method 2: Direct torch.load

```python
import torch
from pathlib import Path

# Load raw archive
path = Path('checkpoints/Flux-Apex-V1.flx')
raw = torch.load(str(path), map_location='cpu', weights_only=False)

# Verify format
assert raw['format'] == 'FLUX'
print(f"Version: {raw['version']}")  # '4.0-multi-modal-enhanced'

# Access components
cse_weights = raw['cse']['state_dict']
field_state = raw['field']['state_dict']['state']  # [96, 96, 96, 512]
decoder_weights = raw['decoder']['state_dict']
```

### Method 3: HuggingFace Hub

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id='UnseenGAP/FLUX',
    filename='checkpoints/Flux-Apex-V1.flx'
)
model = FLUXModel.load(path)
```

---

## Saving & Updating Flux-Apex Model

### Continuous Development Philosophy (CRITICAL)

**Always save back to the SAME filename after modifications:**

```python
# Load
model = FLUXModel.load('checkpoints/Flux-Apex-V1.flx')

# Modify (add facts, train adapters, inject knowledge)
model.add_component('new_adapter', new_state_dict)
model.upgrade_component('cse', improved_cse_weights)

# Save back to SAME filename
model.save('checkpoints/Flux-Apex-V1.flx', overwrite=True)
```

**DO NOT** create new filenames for incremental updates. The `.flx` format tracks history internally via `metadata.modified_components` and timestamps.

### When To Create NEW Filename

- Major architecture version bump (V1 → V2)
- Incompatible format changes
- Domain-specific variant (e.g., `Flux-Apex-V1-Medical.flx`)

### Complete Save Workflow

```python
from flux_model import FLUXModel
from datetime import datetime

# 1. Load current model
model = FLUXModel.load('checkpoints/Flux-Apex-V1.flx')

# 2. Make modifications
model.upgrade_component('decoder', new_decoder_state)
model.add_component('vision_adapter', vision_state, config={'input_size': 224})

# 3. Mark old components as legacy (if replacing)
if 'old_decoder' in model.state:
    model.state['old_decoder']['legacy'] = True
    model.state['old_decoder']['legacy_reason'] = 'Replaced by new decoder v2'
    model.state['old_decoder']['legacy_since'] = datetime.now().isoformat()
    model.state['old_decoder']['removal_target'] = 'v5.0'

# 4. Update metadata
model.metadata['last_modified'] = datetime.now().isoformat()
model.metadata['modified_components'] = ['decoder', 'vision_adapter']

# 5. Save (ALWAYS as last step after all modifications)
model.save('checkpoints/Flux-Apex-V1.flx', overwrite=True)

# 6. Upload to HuggingFace Hub
from flux_utils import upload_flx_to_hf
upload_flx_to_hf('checkpoints/Flux-Apex-V1.flx')
```

---

## Stripping Components

To remove a component from the model:

```python
model = FLUXModel.load('checkpoints/Flux-Apex-V1.flx')

# Remove component entirely
model.remove_component('old_adapter')

# Or disable without removing (preserves state for rollback)
model.components['old_adapter'] = False

model.save('checkpoints/Flux-Apex-V1.flx', overwrite=True)
```

---

## Adding New Components

```python
model = FLUXModel.load('checkpoints/Flux-Apex-V1.flx')

# Add new component with config
model.add_component(
    name='audio_encoder',
    state_dict=trained_audio_encoder.state_dict(),
    config={
        'input_dim': 80,      # mel bins
        'output_dim': 432,    # wave_dim
        'sample_rate': 16000,
    }
)

# Enable in runtime config
model.enable_component('audio_encoder')

model.save('checkpoints/Flux-Apex-V1.flx', overwrite=True)
```

---

## Injecting from External Checkpoints

```python
model = FLUXModel.load('checkpoints/Flux-Apex-V1.flx')

# Inject trained component from .pt file
model.inject_from_checkpoint(
    component_name='grid_to_wave',
    checkpoint_path='checkpoints/gridtowave_contrastive.pt',
)

# Inject from another .flx file
model.inject_from_checkpoint(
    component_name='field',
    checkpoint_path='checkpoints/Flux-capable.flx',
    key_in_checkpoint='field',
)

model.save('checkpoints/Flux-Apex-V1.flx', overwrite=True)
```

---

## .flx File Format Structure

```
Flux-Apex-V1.flx
├── format: "FLUX"                    # Format identifier (required)
├── version: "4.0-multi-modal-enhanced"
├── phase: "phase12"
├── timestamp: "2026-03-30T..."
├── can_continue_learning: True
│
├── runtime_config/                    # How to run the model
│   ├── perception: {cse_enabled, grid_encoder_enabled, ...}
│   ├── memory: {working_enabled, episodic_enabled, ...}
│   ├── generation: {llm_primary, byte_decoder_enabled, ...}
│   ├── reasoning: {causal_tracker_enabled, ...}
│   └── learning: {realtime_learning, temperature, ...}
│
├── components: {cse: true, field: true, ...}  # Enable flags
│
├── metadata/
│   ├── created: "2026-03-29T..."
│   ├── last_modified: "2026-03-30T..."
│   ├── modified_components: ["cse", "field"]
│   └── capabilities: ["text", "grid", "image", "audio"]
│
├── cse/                               # Component: Continuous Semantic Encoder
│   ├── state_dict: {22 tensors}
│   ├── config: {wave_dims: {...}, byte_window: 8}
│   └── legacy: false                  # Legacy flag (add when deprecating)
│
├── field/                             # Component: Resonance Field
│   ├── state_dict: {state: [96,96,96,512]}
│   ├── config: {h: 96, w: 96, d: 96, features: 512}
│   ├── gravity_state: {...}
│   └── thermodynamic_state: {...}
│
├── memory/                            # Component: Three-Tier Memory
│   ├── working: {window_size: 2048, ...}
│   ├── episodic: {vectors, metadata, 74 entries}
│   └── semantic: {field_state_dict, ...}
│
├── decoder/                           # Component: Byte Decoder
│   ├── state_dict: {33 tensors}
│   └── config: {hidden_dim: 1024, layers: 4}
│
├── causal/                            # Component: Causal System
│   ├── cgn_state: {56 nodes, manifolds}
│   └── graph_state: {463 links}
│
├── bridges/                           # Component: Inter-component bridges
│   ├── wave_to_field: {432 → 512}
│   ├── field_to_wave: {512 → 432}
│   └── router: {...}
│
├── adapters/                          # Component: Multi-modal adapters
│   ├── grid_to_wave: {...}
│   ├── wave_to_grid: {...}
│   ├── wave_to_image: {...}
│   └── audio_to_wave: {...}
│
└── llm_reference/                     # External LLM (not stored in file)
    ├── model_name: "Qwen/Qwen2.5-3B-Instruct"
    └── quantization: "4bit"
```

---

## Wave Dimension Invariant

**All waves are 432-dimensional.** This is the universal semantic space:

| Path | Dimension |
|------|-----------|
| CSE output | `[seq_len, 432]` |
| Field input | projects 432 → 512 |
| Field output | projects 512 → 432 |
| Decoder input | projects 432 → 1024 |
| All adapters | modality ↔ 432 |

---

## Architecture & Phase Structure

The project follows a **checkpoint chain** pattern across 12+ phases:

| Phase | Component | Key Class |
|-------|-----------|-----------|
| 1 | Continuous Semantic Encoder (CSE) | `ContinuousSemanticEncoder` |
| 2 | Resonance Field Core (RFC) | `ResonanceField` |
| 3 | Gravitational Relevance (GR) | `GravitationalRelevance` |
| 4 | Thermodynamic Learning (TL) | `ThermodynamicLearner` |
| 5 | Causal Geometry Nodes (CGN) | `CausalGeometryNode` |
| 6 | Three-Tier Memory System | `WorkingMemory`, `EpisodicMemory`, `SemanticMemory` |
| 7 | Full FLUX Integration | `FLUXModel` |
| 8 | Byte Decoder | `WaveDecoder` |
| 8.5 | Grid Adapters | `GridToWave`, `WaveToGrid` |
| 8.8 | Spatial Memory | `SpatialMemory` |
| 8.9 | Causal Tracker + Rules | `CausalTracker`, `RuleInducer` |
| 10 | Hybrid LLM Integration | `HybridModel` |
| 11 | Multi-Modal Adapters | `ImageAdapter`, `AudioAdapter` |
| 12 | Unified Agent | `FLUXAgent` |

### Key Invariants (Never Break)

1. **Flux-Apex is the flagship** — all updates go to `Flux-Apex-V1.flx`
2. **Save after EVERY modification** — call `model.save()` as the final step
3. **Mark deprecated components as legacy** — set `legacy: True` before removal
4. **Wave dimension is 432** — never change this across any component
5. **Field dimension is 96³ × 512** — standard Apex configuration
6. **Use FLUXModel class** for all load/save operations, not raw torch.save
7. **Upload to HuggingFace** after significant changes
8. **Track modified_components** in metadata

---

## Directory Layout

```
flux/
├── phases/phaseN/           # Self-contained phase code
│   ├── PHASE_N_SPEC.md      # Detailed spec for this phase
│   ├── *.py                 # Component modules
│   ├── train_*.py           # Training script
│   ├── demo_phaseN_demoM.py # Demo scripts (runnable)
│   ├── test_phaseN_testM.py # Test scripts (standalone, no pytest)
│   └── RESULTS_PHASE_N.md   # Auto-generated results
├── notebooks/
│   └── phaseN_kaggle.ipynb  # Kaggle notebook for each phase
├── shared/
│   ├── utils/               # Shared utilities
│   ├── data/                # Dataset loaders
│   └── eval/                # Evaluation harness
├── checkpoints/             # Saved .phase.pt files (gitignored)
├── logs/                    # Phase logs (phase1.log, phase2.log, ...)
├── results/                 # Copies of all RESULTS_PHASE_N.md
├── demos/                   # Standalone demo scripts
├── flux_utils.py            # Core utilities (checkpoints, logging, HF Hub)
├── SPECIFICATION.md         # Full technical specification (source of truth)
├── ROADMAP.md               # Phase-by-phase build plan
└── requirements.txt         # Python dependencies
```

---

## Code Conventions

### Naming

| Element | Convention | Example |
|---------|-----------|---------|
| Files | `snake_case.py` | `wave_types.py`, `field_ops.py` |
| Test files | `test_phaseN_testM.py` | `test_phase1_test1.py` |
| Demo files | `demo_phaseN_demoM.py` | `demo_phase1_demo1.py` |
| Train scripts | `train_<component>.py` | `train_cse.py` |
| Classes | `PascalCase` | `ContinuousSemanticEncoder`, `ResonanceField` |
| Functions | `snake_case` | `save_checkpoint()`, `wave_interference()` |
| Constants | `UPPER_SNAKE_CASE` | `CHECKPOINT_DIR`, `TOTAL_WAVE_DIM` |
| Private methods | `_leading_underscore` | `_split_wave()`, `_compute_distances()` |
| Checkpoints | `phaseN.phase.pt` | `phase1.phase.pt` |
| Results files | `RESULTS_PHASE_N.md` | `RESULTS_PHASE_1.md` |

### Type Hints

Always use type hints on function signatures:

```python
from typing import Dict, Any, List, Optional, Tuple

def save_checkpoint(phase: int, state: Dict[str, Any]) -> Path:
    ...
```

### Docstrings

Use Google-style docstrings. Always document tensor shapes in brackets:

```python
def wave_interference(w1: Tensor, w2: Tensor, distance: int) -> Tensor:
    """
    Compute interference of w2 on w1 given their distance.

    Constructive: same phase → waves amplify each other.
    Destructive: opposite phase → waves cancel each other.

    Args:
        w1: [dim] primary wave (being affected)
        w2: [dim] secondary wave (affecting w1)
        distance: positions apart in sequence

    Returns:
        [dim] w1 after w2 interference applied
    """
```

Simple functions use one-liner docstrings:

```python
def checkpoint_exists(phase: int) -> bool:
    """Check if a phase checkpoint exists."""
```

### Imports

Follow this order:
1. Standard library (`pathlib`, `datetime`, `sys`)
2. Third-party (`torch`, `numpy`, `faiss`, `datasets`)
3. Local/project modules (`flux_utils`, phase-specific modules)

Cross-phase imports use `sys.path.append`:

```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'phase1'))
from cse import ContinuousSemanticEncoder
```

Shared utilities imported from project root:

```python
sys.path.append(str(Path(__file__).parent.parent.parent))
from flux_utils import save_checkpoint, load_checkpoint, PhaseResults
```

### Path Handling

Always use `pathlib.Path`, never `os.path`:

```python
from pathlib import Path
phase_dir = Path(__file__).parent
checkpoint_path = phase_dir.parent.parent / 'checkpoints' / f'phase{N}.phase.pt'
path.mkdir(parents=True, exist_ok=True)
```

### Print & Status Output

Use Unicode markers for status feedback:

```python
print(f"  ✓ Phase {phase} checkpoint loaded")
print(f"  ✗ Missing checkpoint for Phase {phase}")
print(f"  ⚠ Warning: < 8GB VRAM")
```

### String Formatting

Use f-strings exclusively:

```python
print(f"Phase {phase} checkpoint saved: {path} ({size_mb:.1f} MB)")
```

### Section Dividers

Use decorated comment bars to separate logical sections in files:

```python
# ─────────────────────────────────────────────
# Section Name
# ─────────────────────────────────────────────
```

---

## PyTorch Patterns

### Device Handling

Use `get_device()` from `flux_utils.py` for auto-detection (cuda > mps > cpu). Pass device as a `str` parameter:

```python
from flux_utils import get_device
device = get_device()
tensor = tensor.to(device)
```

### nn.Module Subclasses

All trainable components subclass `nn.Module` with a `forward()` method:

```python
class ContinuousSemanticEncoder(nn.Module):
    def __init__(self, wave_dim: int = 432, ...):
        super().__init__()
        ...

    def forward(self, x: Tensor) -> SemanticWave:
        ...
```

### Dataclasses for Data Containers

Use `@dataclass` for non-trainable data structures:

```python
from dataclasses import dataclass

@dataclass
class SemanticWave:
    full: Tensor          # [seq_len, 432]
    phonetic: Tensor      # [seq_len, 64]
    semantic: Tensor      # [seq_len, 256]
    ...
```

### Checkpoint Format

Every checkpoint must include:

```python
{
    'phase': int,              # Phase number
    'timestamp': str,          # ISO format
    'config': dict,            # Enough to reconstruct the model
    'state_dict': OrderedDict, # nn.Module state dict
    'metrics': dict,           # Training metrics
}
```

Always use `map_location='cpu'` when loading:

```python
state = torch.load(path, map_location='cpu')
```

### Learnable Fields

Use `nn.Parameter` for learnable tensors that are not standard weight matrices:

```python
self.field = nn.Parameter(torch.randn(H, W, D, features))
```

---

## Constants & Configuration

### Wave Dimensions (Phase 1)

```python
WAVE_DIMS = {
    'phonetic':  64,
    'syntactic': 64,
    'semantic':  256,
    'temporal':  32,
    'intensity': 16,
}
TOTAL_WAVE_DIM = 432  # Sum of all above
```

### Field Dimensions (Phase 2)

```python
FIELD_H, FIELD_W, FIELD_D = 64, 64, 64
FIELD_FEATURES = 512
```

### Master Config Reference

See `SPECIFICATION.md § 5. Configuration` for the full `FLUX_CONFIG` dict covering all phases. Always store config in checkpoints so models can be reconstructed.

---

## Testing Conventions

- **No test framework** — each test is a standalone Python script using `assert`
- Tests define explicit thresholds (e.g., reconstruction loss < 0.1, cosine similarity > 0.7)
- Tests load the checkpoint and validate the trained model, not the training process itself
- Tests report results through `PhaseResults`:

```python
results = PhaseResults(phase=1, component_name="Continuous Semantic Encoder")
results.add_test("Reconstruction Loss", passed=loss < 0.1, score=loss, threshold=0.1)
results.save()
```

- Run tests with: `python test_phaseN_testM.py`
- Each phase needs all tests passing before moving to the next phase

---

## Demo Conventions

- Demos are standalone scripts: `python demo_phaseN_demoM.py`
- Each demo should produce visual or textual output showing the component working
- Use `matplotlib` for visualizations
- Use `rich` for formatted terminal output
- Demos should run in < 60 seconds on consumer hardware

---

## Error Handling

- Raise `FileNotFoundError` with actionable messages when checkpoints are missing
- Use `assert` with descriptive strings for invariant checks
- Include remediation steps in error messages:

```python
raise FileNotFoundError(
    f"Phase {phase} checkpoint not found at {path}\n"
    f"Run Phase {phase} training first."
)
```

---

## Loading Embedded VLM from .flx (CRITICAL)

The Flux-Apex model embeds Qwen2.5-VL-3B weights directly in the .flx file. Loading requires a specific pattern due to `trust_remote_code` requirements.

### Why This Pattern is Required

Qwen2.5-VL and similar models use `trust_remote_code=True`, which means:
1. The model architecture code is downloaded from HuggingFace (not bundled in transformers)
2. `from_config()` does NOT work — it can't load the custom model classes
3. You MUST use `from_pretrained()` to get the architecture, then `load_state_dict()` for weights

### Standard VLM Loading Pattern (ALWAYS USE THIS)

```python
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Load .flx file
model = torch.load('checkpoints/Flux-Apex-V1.flx', map_location='cpu')

if 'vlm' in model:
    vlm_state = model['vlm']
    embedded_weights = vlm_state.get('weights', {})
    
    # Step 1: Load processor (small download - tokenizer config only)
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        trust_remote_code=True,
    )
    
    # Step 2: Load model ARCHITECTURE from HuggingFace (CACHED after first run)
    # CRITICAL: Use Qwen2_5_VLForConditionalGeneration for Qwen2.5-VL models!
    # Qwen2VLForConditionalGeneration is for Qwen2-VL (different MLP architecture)
    vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    # Step 3: REPLACE HuggingFace weights with embedded .flx weights
    missing, unexpected = vlm_model.load_state_dict(embedded_weights, strict=False)
    # missing/unexpected keys are OK for tied weights
    
    vlm_model.eval()
    # Now vlm_model uses YOUR embedded weights, not HuggingFace weights!
```

### Using the Utility Function

```python
from phases.phase2.flux_format import load_embedded_vlm

model = torch.load('checkpoints/Flux-Apex-V1.flx', map_location='cpu')
vlm_model, processor = load_embedded_vlm(model)

# Use for inference
messages = [{"role": "user", "content": "Hello!"}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], return_tensors="pt").to(vlm_model.device)
outputs = vlm_model.generate(**inputs, max_new_tokens=100)
response = processor.decode(outputs[0], skip_special_tokens=True)
```

### Anti-Patterns (DO NOT USE)

```python
# ❌ WRONG: AutoModel gives base model WITHOUT generate() method
vlm_model = AutoModel.from_pretrained(...)  # No generate()!

# ❌ WRONG: from_config() doesn't work with trust_remote_code
config = AutoConfig.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
vlm_model = AutoModel.from_config(config)  # FAILS!

# ❌ WRONG: AutoModelForCausalLM doesn't support VL models  
vlm_model = AutoModelForCausalLM.from_pretrained(...)  # FAILS!

# ❌ WRONG: AutoModelForVision2Seq doesn't exist
vlm_model = AutoModelForVision2Seq.from_pretrained(...)  # FAILS!

# ❌ WRONG: Qwen2VLForConditionalGeneration is for Qwen2-VL, not Qwen2.5-VL!
from transformers import Qwen2VLForConditionalGeneration  # Wrong for 2.5!

# ✅ CORRECT: Use Qwen2_5_VLForConditionalGeneration for Qwen2.5-VL
from transformers import Qwen2_5_VLForConditionalGeneration  # Use this!
```

### VLM Storage Structure in .flx

```python
model['vlm'] = {
    'base_model': 'Qwen/Qwen2.5-VL-3B-Instruct',
    'quantization': 'fp16',
    'total_params': 3_756_000_000,
    'weights': {                    # 824 tensors
        'model.embed_tokens.weight': tensor(...),
        'model.layers.0.self_attn.q_proj.weight': tensor(...),
        # ... all model weights
    },
    'config': {                     # Original HF config
        'hidden_size': 2048,
        'num_hidden_layers': 36,
        'vocab_size': 151936,
        ...
    },
    'bridges': {                    # Wave ↔ VLM projections
        'wave_to_vlm': {'in': 432, 'out': 2048},
        'vlm_to_wave': {'in': 2048, 'out': 432},
    },
}
```

---

## VLM Orchestration (v5.1+)

Starting with v5.1-orchestrated, the embedded VLM can **call FLUX components as tools**.

### Tool Categories

| Category | Tools |
|----------|-------|
| **Perception** | `encode_text`, `encode_grid`, `encode_image` |
| **Knowledge** | `query_field`, `recall_memory`, `store_memory` |
| **Reasoning** | `predict_effect`, `get_applicable_rules`, `trace_causality` |
| **Exploration** | `get_curiosity_map`, `mark_explored` |
| **CGN** | `query_cgn`, `fire_cgn`, `add_causal_arrow` |
| **Generation** | `decode_grid`, `generate_text` |

### Tool Call Format

```xml
<tool>encode_grid</tool>
<params>{"grid": [[0,1],[1,0]], "mode": "holistic"}</params>
```

### Self-Describing Model

The `.flx` file contains the orchestration section:

```python
model = torch.load('Flux-Apex-V1.flx', map_location='cpu')

# Discover capabilities
if 'orchestration' in model:
    tools = model['orchestration']['tools']
    prompt = model['orchestration']['system_prompt']
    caps = model['orchestration']['capabilities']
```

### Orchestration Workflow

```
User Input → VLM (decides which tools) → Tool Calls → Execute → VLM (synthesize) → Output
```

The VLM is the **brain** that orchestrates FLUX components as cognitive tools.

---

## Key Technical Concepts

When implementing any component, remember these physics-inspired principles:

1. **No vocabulary / no tokenization** — CSE works on raw UTF-8 bytes with a sliding window
2. **Local updates only** — field updates affect only the neighborhood, never the whole field
3. **Energy minimization** — settling to minimum energy IS both inference and learning
4. **Mass = evidence** — concepts grow heavier with more evidence, attracting related queries
5. **Negative mass = contradiction** — disproven concepts repel related queries
6. **Causal arrows** — every conclusion stores WHY it was reached, enabling invalidation
7. **No epochs** — learning is a continuous stream, not batched repetitions
8. **Three memory tiers** — working (session), episodic (permanent facts), semantic (deep field)
9. **Zero catastrophic forgetting** — new attractors form without destroying old ones (by design)
10. **O(log n) relevance** — gravitational relevance uses spatial trees, not all-pairs attention

---

## Dependencies

Core stack: `torch>=2.0.0`, `numpy`, `scipy`, `faiss-gpu`, `datasets`, `evaluate`, `matplotlib`, `tensorboard`, `tqdm`, `rich`, `transformers`, `huggingface_hub`

See `requirements.txt` for full pinned versions.

---

## HuggingFace Hub Integration

### Constants
```python
HF_REPO_ID = "UnseenGAP/FLUX"          # HuggingFace model repo
GITHUB_REPO_URL = "https://github.com/Unseengap/FLUX.git"
```

### Token Resolution
Always use `_resolve_hf_token()` — never hardcode tokens:
```python
from flux_utils import _resolve_hf_token
token = _resolve_hf_token()
# Checks: 1. Kaggle secrets  2. HF_TOKEN env var  3. .env file
```

### Checkpoint Upload
After training, upload to HuggingFace Hub:
```python
from flux_utils import upload_checkpoint_to_hf
upload_checkpoint_to_hf(phase=1, hf_token=token)
```

### Checkpoint Loading with Fallback
`load_checkpoint()` automatically falls back to HuggingFace if local file missing:
```python
checkpoint = load_checkpoint(1)  # Tries local, then HF Hub
```

---

## Logging Conventions

### PhaseLogger
Every phase uses `PhaseLogger` from `flux_utils.py`:

```python
from flux_utils import PhaseLogger
log = PhaseLogger(phase=1)

log.separator("Phase 1: Continuous Semantic Encoder")
log.cell_start("Cell 3 — Hardware & Secrets")
log.info("Device: cuda")
log.success("Checkpoint saved")
log.warning("Low VRAM")
log.error("Test failed")
log.metric("loss", "0.0123")
log.cell_end("Cell 3 — Hardware & Secrets", "PASS")
```

### In Kaggle Notebooks
Every code cell must call:
```python
log.cell_start("Cell N — Description")
# ... cell code ...
log.cell_end("Cell N — Description", "PASS/FAIL")  # status optional
```

### Log Upload
Logs are uploaded to both HuggingFace Hub and GitHub:
```python
from flux_utils import upload_logs_to_hf, git_commit_and_push
upload_logs_to_hf(phase=1, hf_token=token)
git_commit_and_push(files=['logs/phase1.log'], message='Phase 1 logs')
```

---

## Kaggle Notebook Conventions

### Standard Cell Structure
Every phase notebook follows this template:
1. Clone/pull repo from GitHub
2. Install deps + setup.py
3. Init PhaseLogger + detect hardware + load HF_TOKEN from Kaggle secrets
4. Smoke test
5. Training
6. Upload checkpoint to HuggingFace Hub
7–9. Run 3 tests
10–11. Run 2 demos
12. Interactive exploration
13. View results
14. View full log
15. Final upload (logs → HF + GitHub)
16. Save artifacts to Kaggle output

### Kaggle Secrets
Add `HF_TOKEN` via Kaggle → Add-ons → Secrets. Accessed via:
```python
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("HF_TOKEN")
```

---

## Workflow Reminders

- Always read `PHASE_N_SPEC.md` before implementing a phase
- Use `flux_utils.py` utilities — never reimplement checkpoint management
- Run `verify_checkpoint_chain(up_to_phase=N)` at the start of each phase
- Save checkpoints via `save_checkpoint()`, load via `load_checkpoint()`
- Upload checkpoints via `upload_checkpoint_to_hf()` after training
- Use `PhaseLogger` for all logging in notebooks and scripts
- Generate results via `PhaseResults` — never write `RESULTS_PHASE_N.md` manually
- Check acceptance criteria in `ROADMAP.md` before declaring a phase complete
- Mark TODO items as `# TODO: Copilot — <description>` for unimplemented methods
---

## Flux-Apex Model Workflow (CRITICAL)

When modifying the Flux-Apex model, follow this exact workflow:

1. **Load the model**
   ```python
   from flux_model import FLUXModel
   model = FLUXModel.load('checkpoints/Flux-Apex-V1.flx')
   ```

2. **Make modifications**
   - Add components via `model.add_component()`
   - Upgrade components via `model.upgrade_component()`
   - Inject from checkpoints via `model.inject_from_checkpoint()`
   - Remove components via `model.remove_component()`

3. **Mark deprecated components as legacy**
   ```python
   model.state['old_component']['legacy'] = True
   model.state['old_component']['legacy_reason'] = 'Replaced by X'
   model.state['old_component']['legacy_since'] = datetime.now().isoformat()
   model.state['old_component']['removal_target'] = 'v5.0'
   ```

4. **Update metadata**
   ```python
   model.metadata['last_modified'] = datetime.now().isoformat()
   model.metadata['modified_components'] = ['component1', 'component2']
   ```

5. **Save as LAST STEP** (ALWAYS)
   ```python
   model.save('checkpoints/Flux-Apex-V1.flx', overwrite=True)
   ```

6. **Upload to HuggingFace Hub** (after significant changes)
   ```python
   from flux_utils import upload_flx_to_hf
   upload_flx_to_hf('checkpoints/Flux-Apex-V1.flx')
   ```

### Legacy Flag Checklist

Before removing any component:
- [ ] Set `legacy: True` on the component
- [ ] Include `legacy_reason` explaining what replaced it
- [ ] Include `legacy_since` with ISO date
- [ ] Include `removal_target` version
- [ ] Save the model
- [ ] Wait at least one version cycle before actual removal