# FLUX Project — Copilot Instructions

## Project Overview

FLUX (Field-based Latent Understanding eXperience) is a novel AI architecture that replaces traditional neural network primitives with physics-inspired components: resonance fields instead of weights, continuous semantic waves instead of tokens, gravitational relevance instead of attention (O(log n)), thermodynamic settling instead of backpropagation, and causal geometry nodes instead of neurons. The project is structured as 8 sequential phases, each building on the previous checkpoint.

**Source of truth:** Always consult `SPECIFICATION.md` for technical details and `ROADMAP.md` for phase plans before implementing anything.

---

## Architecture & Phase Structure

The project follows a strict **checkpoint chain** pattern across 8 phases:

| Phase | Component | Key Class |
|-------|-----------|-----------|
| 1 | Continuous Semantic Encoder (CSE) | `ContinuousSemanticEncoder` |
| 2 | Resonance Field Core (RFC) | `ResonanceField` |
| 3 | Gravitational Relevance (GR) | `GravitationalRelevance` |
| 4 | Thermodynamic Learning (TL) | `ThermodynamicLearner` |
| 5 | Causal Geometry Nodes (CGN) | `CausalGeometryNode` |
| 6 | Three-Tier Memory System | `WorkingMemory`, `EpisodicMemory`, `SemanticMemory` |
| 7 | Full FLUX Integration | `FLUXModel` |
| 8 | Scale & GPT-2 Benchmark | `FLUXLarge` |

### Key Invariants (Never Break)

1. **Every phase saves a checkpoint** to `checkpoints/phaseN.phase.pt`
2. **Every phase loads and verifies** the previous phase's checkpoint before starting
3. **Every phase has** at least one `demo_phaseN_demoX.py` and three `test_phaseN_testX.py` files
4. **Every phase generates** `RESULTS_PHASE_N.md` via the `PhaseResults` utility
5. **Checkpoints accumulate** — phase N contains all components from phases 1 through N
6. **Every phase has** a Kaggle notebook (`notebooks/phaseN_kaggle.ipynb`)
7. **Every phase uploads** checkpoint to HuggingFace Hub (`UnseenGAP/FLUX`)
8. **Every phase logs** to `logs/phaseN.log` via `PhaseLogger`

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
