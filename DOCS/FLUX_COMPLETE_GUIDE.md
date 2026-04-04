# FLUX — Complete Architecture Guide

**Field-based Latent Understanding eXperience**  
*A Physics-Inspired Cognitive AI System*  
*Version 8.3-autonomous | April 2026*

---

## Table of Contents

1. [What is FLUX?](#what-is-flux)
2. [Architecture Overview](#architecture-overview)
3. [Core Components](#core-components)
4. [Embedded Models](#embedded-models)
5. [The .flx File Format](#the-flx-file-format)
6. [Bootstrap System](#bootstrap-system)
7. [Autonomous Agent](#autonomous-agent)
8. [Memory Fabric Hardware](#memory-fabric-hardware)
9. [Quick Start](#quick-start)
10. [API Reference](#api-reference)

---

## What is FLUX?

FLUX (Field-based Latent Understanding eXperience) is a novel AI architecture that replaces traditional neural network primitives with physics-inspired components:

| Traditional AI | FLUX Equivalent |
|----------------|-----------------|
| Weights | **Resonance Fields** — knowledge stored as interference patterns |
| Tokens | **Semantic Waves** — continuous 432D representations from raw bytes |
| Attention (O(n²)) | **Gravitational Relevance** — O(log n) spatial tree lookup |
| Backpropagation | **Thermodynamic Settling** — energy minimization as learning |
| Neurons | **Causal Geometry Nodes** — interconnected causal manifolds |

### Key Properties

- **No Tokenizer** — Operates directly on UTF-8 bytes via Continuous Semantic Encoder
- **Self-Contained** — All 12 models embedded in single .flx file (no external downloads)
- **Continuous Learning** — Can learn new facts without catastrophic forgetting
- **Multi-Modal** — Text, vision, audio, grid (ARC), face detection all integrated
- **Self-Describing** — The .flx file contains its own architecture specification

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FLUX COGNITIVE ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  INPUT LAYER                                                             │
│  ───────────                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │  Text    │  │  Image   │  │  Audio   │  │  Grid    │                │
│  │ (UTF-8)  │  │ (pixels) │  │  (wav)   │  │  (ARC)   │                │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘                │
│       │             │             │             │                        │
│       ▼             ▼             ▼             ▼                        │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐                 │
│  │   CSE   │   │  CLIP   │   │ Whisper │   │GridToWav│                 │
│  │ 1.3M p  │   │  428M p │   │  242M p │   │  192K p │                 │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘                │
│       │             │             │             │                        │
│       └──────────┬──┴─────────────┴─────────────┘                        │
│                  │                                                        │
│                  ▼  [432-dimensional semantic wave]                      │
│                                                                          │
│  KNOWLEDGE LAYER                                                         │
│  ───────────────                                                         │
│  ┌────────────────────────────────────────────────────────────┐         │
│  │                    RESONANCE FIELD                          │         │
│  │                    48 × 48 × 48 × 256                       │         │
│  │                      (28.4M params)                         │         │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │         │
│  │  │   Gravity    │  │ Thermodynamic│  │    Causal    │     │         │
│  │  │  Relevance   │  │   Learning   │  │   Geometry   │     │         │
│  │  │   75.2M p    │  │   135M p     │  │   149.8M p   │     │         │
│  │  └──────────────┘  └──────────────┘  └──────────────┘     │         │
│  └────────────────────────────────────────────────────────────┘         │
│                  │                                                        │
│                  ▼                                                        │
│  MEMORY LAYER                                                            │
│  ────────────                                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │   Working    │  │   Episodic   │  │   Semantic   │                  │
│  │   Memory     │  │   Memory     │  │   Memory     │                  │
│  │  (session)   │  │  (facts)     │  │  (concepts)  │                  │
│  └──────────────┘  └──────────────┘  └──────────────┘                  │
│        └───────────────────┬───────────────────┘                         │
│                            │  (542.3M params total)                      │
│                            ▼                                              │
│  REASONING LAYER                                                         │
│  ───────────────                                                         │
│  ┌────────────────────────────────────────────────────────────┐         │
│  │                    BRIDGES (455.7M params)                  │         │
│  │        Wave ↔ Field projections + Output Head               │         │
│  └────────────────────────────────────────────────────────────┘         │
│                            │                                              │
│                            ▼                                              │
│  GENERATION LAYER                                                        │
│  ────────────────                                                        │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                     EMBEDDED MODELS (6.4B params)                  │  │
│  │ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐│  │
│  │ │ Instruct │ │  Coder   │ │  Vision  │ │    TTS   │ │ Detection││  │
│  │ │  1.54B   │ │  1.54B   │ │  2.21B   │ │   410M   │ │   538M   ││  │
│  │ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘│  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Continuous Semantic Encoder (CSE)
**Parameters:** 1,337,264 | **Output:** 432D wave

The CSE encodes raw UTF-8 bytes into semantic waves without any tokenizer:

```python
# Wave dimensions breakdown
WAVE_DIMS = {
    'phonetic':  64,   # Sound patterns
    'syntactic': 64,   # Grammar structure
    'semantic':  256,  # Meaning representation
    'temporal':  32,   # Sequence position
    'intensity': 16,   # Emphasis/importance
}
# Total: 432 dimensions
```

**Key insight:** By operating on bytes, FLUX handles any language, emoji, code, or binary data uniformly.

### 2. Resonance Field
**Parameters:** 28,442,624 | **Dimensions:** 48³ × 256

The field stores knowledge as interference patterns in a 3D semantic space:

- **Writing:** Wave patterns create attractors at specific field coordinates
- **Reading:** Queries excite resonances that return stored knowledge
- **Learning:** New facts modify local field regions without global disruption

### 3. Gravitational Relevance
**Parameters:** 75,181,958 | **Complexity:** O(log n)

Replaces attention mechanism with physics-inspired relevance:

- **Mass = Evidence:** Concepts with more supporting facts are "heavier"
- **Attraction:** Queries are pulled toward relevant knowledge
- **Negative Mass:** Contradicted facts repel queries (graceful forgetting)

### 4. Thermodynamic Learning
**Parameters:** 135,047,043

Learning as energy minimization:

- **Temperature:** Controls exploration vs. exploitation
- **Energy:** Represents confidence in knowledge
- **Settling:** System naturally finds stable states (learning = cooling)

### 5. Causal Geometry Nodes (CGN)
**Parameters:** 149,757,403

Every conclusion stores WHY it was reached:

- **Causal Arrows:** A → B with confidence and evidence
- **Multi-timescale:** Fast (immediate), Medium (session), Slow (permanent)
- **Invalidation:** If A becomes false, conclusions depending on A are marked uncertain

### 6. Three-Tier Memory
**Total Parameters:** 542,274,422

| Tier | Purpose | Persistence |
|------|---------|-------------|
| **Working** | Current context | Session only |
| **Episodic** | Specific facts with FAISS index | Permanent |
| **Semantic** | Deep conceptual knowledge | Permanent |

---

## Embedded Models

All models are self-contained in the .flx file with lazy loading support:

### Language Models (3.1B params)
| Model | Base | Params | Loading | Purpose |
|-------|------|--------|---------|---------|
| **instruct** | Qwen2.5-1.5B-Instruct | 1.54B | Eager | Primary reasoning |
| **coder** | Qwen2.5-Coder-1.5B-Instruct | 1.54B | Lazy | Code generation |

### Vision Models (2.64B params)
| Model | Base | Params | Loading | Purpose |
|-------|------|--------|---------|---------|
| **vision** | Qwen2-VL-2B-Instruct | 2.21B | Lazy | Visual understanding |
| **clip** | clip-vit-large-patch14 | 428M | Eager | Image embeddings |

### Audio Models (652M params)
| Model | Base | Params | Loading | Purpose |
|-------|------|--------|---------|---------|
| **whisper** | whisper-small | 242M | Lazy | Speech-to-text |
| **tts** | bark-small | 410M | Lazy | Text-to-speech |

### Detection Models (538M params)
| Model | Base | Params | Format | Purpose |
|-------|------|--------|--------|---------|
| **depth** | MiDaS DPT_Large | 344M | PyTorch | Depth estimation |
| **object_detect** | OWLv2 | 155M | PyTorch | Object detection |
| **pose** | HRNet-W32 | 39M | Timm | Pose estimation |
| **face** | InsightFace buffalo_l | — | ONNX (5) | Face recognition |

### Utility Models
| Model | Base | Params | Purpose |
|-------|------|--------|---------|
| **embedding** | all-MiniLM-L6-v2 | 23M | Sentence embeddings |

---

## The .flx File Format

The `.flx` format is a self-describing cognitive architecture container:

```
Flux-Apex-V1.flx (17.24 GB)
├── format: "FLUX"
├── version: "8.3-autonomous"
├── phase: "complete"
├── can_continue_learning: True
│
├── runtime_config/           # Runtime behavior settings
│   ├── perception/
│   ├── memory/
│   ├── generation/
│   ├── reasoning/
│   └── learning/
│
├── components/               # Enable/disable flags
│   ├── cse: True
│   ├── field: True
│   ├── memory: True
│   ├── instruct: True
│   └── ... (44 total)
│
├── cse/                      # Continuous Semantic Encoder
│   ├── config: {wave_dim: 432}
│   └── state_dict: {...}
│
├── field/                    # Resonance Field
│   ├── config: {48³ × 256}
│   └── state_dict: {...}
│
├── memory/                   # Three-tier memory
│   ├── working/
│   ├── episodic/
│   └── semantic/
│
├── models/                   # 12 embedded models
│   ├── instruct/
│   ├── vision/
│   ├── coder/
│   └── ...
│
├── runtime/                  # Embedded Python code
│   ├── version: "8.3-autonomous"
│   ├── bootstrap: "<code>"
│   ├── code: {99 files}
│   └── metadata: {...}
│
└── orchestration/           # Tool definitions
    ├── tools: [...]
    └── system_prompt: "..."
```

---

## Bootstrap System

FLUX can "wake up" from a .flx file with zero external dependencies:

```python
from bootstrap import wake_up

# Wake from .flx file
result = wake_up('Flux-Apex-V1.flx', device='cuda', verbose=True)

# Result contains:
# - flx: Full model dictionary
# - modules: 99 loaded Python modules
# - version: "8.3-autonomous"
# - components: Enabled component flags
```

### How Bootstrap Works

1. **Load Archive:** `torch.load()` the .flx file
2. **Install Import Hook:** Register `EmbeddedModuleFinder` in `sys.meta_path`
3. **Decompress Code:** gzip+base64 → Python source
4. **Execute Modules:** Load in dependency order
5. **Return Control:** All FLUX modules now importable

```python
# After wake_up(), these imports work:
from phases.phase_autonomous import AutonomousAgent
from phases.phase_orchestrator import NativeJSONOrchestrator
from phases.phase_claw import FluxBridge
```

---

## Autonomous Agent

The v8.3-autonomous release adds a complete autonomous agent system:

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INSTRUCT MODEL (Brain)                        │
│              Qwen2.5-1.5B-Instruct — Reasoning                  │
│                          │                                       │
│              ┌───────────┼───────────┐                          │
│              │           │           │                           │
│         delegate_    use_tools    goal_plan                      │
│         to_coder()                                               │
│              │                                                   │
│              ▼                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    CODER POOL                             │  │
│  │            Qwen2.5-Coder-1.5B-Instruct                   │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐│  │
│  │  │ Sandbox 1│  │ Sandbox 2│  │ Sandbox 3│  │ Sandbox 4││  │
│  │  │  Python  │  │  Python  │  │  Python  │  │  Python  ││  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘│  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Purpose |
|-----------|---------|
| **FluxToolExecutor** | Dispatches 28 built-in tools |
| **CodeSandbox** | Safe Python execution (20 allowed modules) |
| **CoderPool** | Parallel sandbox execution |
| **DocumentIngester** | PDF, DOCX, MD, JSON, CSV, HTML processing |
| **GoalPlanner** | Proactive multi-step planning |
| **AutonomousAgent** | Main coordinator |

### Tool Categories

| Category | Tools |
|----------|-------|
| **Perception** | encode_text, encode_grid, encode_image, encode_audio |
| **Knowledge** | query_field, recall_memory, store_memory |
| **Reasoning** | predict_effect, get_rules, trace_causality |
| **Exploration** | get_curiosity_map, mark_explored |
| **Generation** | decode_grid, generate_text, delegate_to_coder |

---

## Memory Fabric Hardware

FLUX is designed to run on dedicated hardware — the Memory Fabric Hub:

```
┌─────────────────────────────────────────────────────────────────┐
│                     MEMORY FABRIC HUB                            │
│                Your Personal AI Ecosystem                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  COMPUTE          STORAGE           CONNECTIVITY                │
│  ───────          ───────           ────────────                │
│  GPU (RTX 4090    NVMe SSD          HDMI Out    ───► TV/Monitor│
│  or equivalent)   2TB+              MIDI In/Out ───► Synths    │
│                                     USB-A/C     ───► Devices   │
│  RAM 32GB+        Model .flx        Audio I/O   ───► Speakers  │
│                   (local)           Ethernet    ───► Network   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

See [FLUX_HUB_BOOTLOADER_SPEC.md](FLUX_HUB_BOOTLOADER_SPEC.md) for UI specifications.

---

## Quick Start

### Option 1: Kaggle/Colab (Recommended for Testing)

```python
# Cell 1: Clone and setup
!git clone https://github.com/Unseengap/FLUX.git
%cd FLUX

# Cell 2: Download model
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='UnseenGAP/FLUX',
    filename='checkpoints/Flux-Apex-V1.flx',
    local_dir='.'
)

# Cell 3: Bootstrap and run
from bootstrap import wake_up
result = wake_up('checkpoints/Flux-Apex-V1.flx', device='cuda')

# Cell 4: Use the model
from phases.phase_autonomous import AutonomousAgent
agent = AutonomousAgent(result['flx'])
response = agent.process("Hello, FLUX!")
```

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/Unseengap/FLUX.git
cd FLUX

# Create venv and install deps
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download model (requires HF_TOKEN for gated access)
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('UnseenGAP/FLUX', 'checkpoints/Flux-Apex-V1.flx', local_dir='.')
"

# Run
python run.py
```

### Option 3: Memory Fabric Hub

The model runs automatically on dedicated FLUX hardware.
See [FLUX_HUB_BOOTLOADER_SPEC.md](FLUX_HUB_BOOTLOADER_SPEC.md).

---

## API Reference

### FLUXModel Class

```python
from flux_model import FLUXModel

# Load model
model = FLUXModel.load('checkpoints/Flux-Apex-V1.flx')

# Get component
cse = model.get_component('cse')
field = model.get_component('field')

# List embedded models
models = model.list_embedded_models()

# Check for model
has_vision = model.has_embedded_model('vision')

# Print summary
model.summary()

# Save (after modifications)
model.save('checkpoints/Flux-Apex-V1.flx', overwrite=True)
```

### Bootstrap API

```python
from bootstrap import wake_up

result = wake_up(
    flx_path='Flux-Apex-V1.flx',
    device='cuda',        # 'cpu', 'cuda', 'mps'
    extract_code=True,    # Load embedded runtime
    verbose=True          # Print loading progress
)

# Returns:
# {
#     'flx': <full model dict>,
#     'modules': {99 loaded modules},
#     'version': '8.3-autonomous',
#     'components': {44 boolean flags}
# }
```

### Autonomous Agent API

```python
from phases.phase_autonomous import (
    AutonomousAgent,
    FluxToolExecutor,
    CodeSandbox,
    CoderPool,
)

# Create agent
agent = AutonomousAgent(flx_dict)

# Process input
response = agent.process("Write a Python function to sort a list")

# Execute code safely
sandbox = CodeSandbox(timeout=10)
result = sandbox.execute("print(sum(range(10)))")
# result.output == '45\n'

# Parallel code execution
pool = CoderPool(models_dict, max_sandboxes=4)
results = pool.execute_parallel([
    "print(1+1)",
    "print(2+2)",
])
pool.shutdown()
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 8.3-autonomous | 2026-04-03 | CoderPool, GoalPlanner, DocumentIngester, 99 files |
| 8.2-fixed-imports | 2026-04-02 | Bootstrap fix, 91/91 modules load |
| 8.1-complete | 2026-04-01 | All phases injected, detection models |
| 7.1-detection | 2026-03-30 | Face, depth, pose, object detection |
| 6.0-autonomous | 2026-03-29 | Full model compression |
| 5.0-voice | 2026-03-28 | TTS + Whisper embedded |
| 4.0-multi-modal | 2026-03-27 | Grid adapters, initial .flx format |

---

## Resources

- **HuggingFace:** https://huggingface.co/UnseenGAP/FLUX
- **GitHub:** https://github.com/Unseengap/FLUX
- **Model File:** `checkpoints/Flux-Apex-V1.flx` (17.24 GB)

---

*FLUX — Your local, private, physics-inspired AI that learns and grows with you.*
