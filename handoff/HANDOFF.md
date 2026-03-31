# FLUX Custom VLM Handoff Documentation

**Generated:** 2026-03-30
**Model Version:** 5.2-custom-vlm
**Purpose:** Enable next AI agent to continue FLUX VLM development

---

## Quick Start

```python
import torch
from pathlib import Path

# Load model
model = torch.load('checkpoints/Flux-Apex-V1.flx', map_location='cpu', weights_only=False)

# Check version
assert model['version'] == '5.2-custom-vlm'

# Access VLM
vlm = model['vlm']
weights = vlm['weights']  # 824 tensors, ~3.75B params
bridges = vlm['bridges']  # Wave↔VLM projections

# Access custom VLM wrapper info
custom_vlm = vlm['custom_vlm']
print(custom_vlm['features'])
```

---

## Architecture Overview

### FluxVLM Wrapper

The `FluxVLM` class wraps Qwen2.5-VL-3B-Instruct with:

1. **Pre/Post Generation Hooks**
   - `register_pre_generate_hook(fn)` - Called before generation starts
   - `register_post_token_hook(fn)` - Called after each token
   - `register_layer_hook(layer_idx, fn)` - Hook specific layers

2. **Wave Injection**
   - `inject_wave_context(wave, layer)` - Inject 432D wave into generation

3. **Tool Call Detection**
   - Automatic detection of `<tool>` tags mid-generation
   - Parser: `parse_tool_calls(text)` returns list of tool calls

4. **Layer Access**
   - `get_layer(name)` - Direct layer access
   - `introspect()` - Model info dict

### Bridge Layers

```python
# Wave → VLM (432 → 2048)
wave_to_vlm = WaveToVLMBridge(wave_dim=432, hidden_size=2048)
vlm_hidden = wave_to_vlm(wave)  # [batch, seq, 2048]

# VLM → Wave (2048 → 432)
vlm_to_wave = VLMToWaveBridge(hidden_size=2048, wave_dim=432)
wave_out = vlm_to_wave(vlm_hidden)  # [batch, seq, 432]
```

---

## Key Classes

### FluxVLM
```python
class FluxVLM(nn.Module):
    def __init__(self, config, weights, device='cuda')
    def load_backend(self, use_hf_model=True)
    def generate(self, prompt, max_new_tokens, temperature, ...)
    def get_layer(self, layer_name) -> nn.Module
    def inject_wave_context(self, wave, layer)
    def register_pre_generate_hook(self, hook)
    def introspect(self) -> Dict
```

### FluxGenerator
```python
class FluxGenerator:
    def __init__(self, flux_vlm)
    def generate_with_hooks(self, prompt, system_prompt, max_new_tokens, ...)
    def pre_generate_hook(self, input_ids, attention_mask, wave_context)
    def post_token_hook(self, token_id, token_str, position)
```

### ToolInjector
```python
class ToolInjector:
    def __init__(self, model_state)
    def parse_tool_call(self, text) -> Optional[Dict]
    def execute_tool(self, tool_call) -> Dict
    def format_result_for_injection(self, result) -> str
    def set_wave_variable(self, name, value)
```

### ModelSurgeon
```python
class ModelSurgeon:
    def __init__(self, model)
    def get_layer(self, name) -> nn.Module
    def set_layer(self, name, module) -> bool
    def freeze_layers(self, pattern) -> int
    def unfreeze_layers(self, pattern) -> int
    def get_trainable_params(self) -> Dict
    def save_checkpoint(self, name)
    def rollback(self, name) -> bool
```

### WeightSurgeon
```python
class WeightSurgeon:
    def __init__(self, model)
    def scale_layer_weights(self, layer_name, factor) -> int
    def prune_layer(self, layer_name, threshold) -> Dict
    def add_noise(self, layer_name, std) -> int
    def quantize_layer(self, layer_name, bits) -> Dict
```

---

## Model State Structure

```python
model = {
    'format': 'FLUX',
    'version': '5.2-custom-vlm',
    'phase': 'phase_vlm_native',
    
    'vlm': {
        'base_model': 'Qwen/Qwen2.5-VL-3B-Instruct',
        'weights': {...},  # 824 tensors
        'config': {
            'hidden_size': 2048,
            'num_hidden_layers': 36,
            'vocab_size': 151936,
        },
        'bridges': {
            'wave_to_vlm': {'state_dict': ...},
            'vlm_to_wave': {'state_dict': ...},
        },
        'custom_vlm': {
            'enabled': True,
            'wrapper_class': 'FluxVLM',
            'features': [...],
        },
    },
    
    'orchestration': {
        'tools': {...},  # 17 tools
        'system_prompt': '...',
    },
    
    # Other components...
    'cse': {...},
    'field': {...},
    'memory': {...},
}
```

---

## Known Issues / Limitations

1. **HuggingFace Architecture Dependency**
   - Still loads model architecture from HF (cached)
   - Weights are embedded, but class needs `trust_remote_code`
   - Future: Implement native transformer without HF dependency

2. **Bridge Layers Not Trained**
   - WaveToVLMBridge and VLMToWaveBridge have random weights
   - Need contrastive training on wave-text pairs
   - Low cosine similarity expected until trained

3. **Tool Execution is Simulated**
   - ToolInjector returns mock results
   - Actual integration with FLUX components needed

4. **No Vision Testing Done**
   - VLM supports vision but not tested in this notebook
   - Need to verify image encoding through bridges

---

## Recommended Next Steps

1. **Train Bridge Layers**
   - Create contrastive dataset: (wave, text) pairs
   - Train bridges to align wave and VLM spaces
   - Target: cos_sim > 0.9 after training

2. **Implement Native VLM**
   - Use `phases/phase_vlm_native/vlm_architecture.py`
   - Remove HuggingFace dependency entirely
   - Load weights directly from .flx

3. **Integrate Tool Execution**
   - Connect ToolInjector to actual FLUX components
   - Test query_field, encode_grid, etc.

4. **Test on ARC Puzzles**
   - Use orchestrated VLM for puzzle solving
   - Evaluate tool call accuracy

5. **Fine-tune for Tool Use**
   - Create tool-use training data
   - Fine-tune VLM on tool calling patterns

---

## Test Results Summary

| Test | Status | Notes |
|------|--------|-------|
| FluxVLM wrapper | ✓ | Created successfully |
| Bridge layers | ✓ | Random init, needs training |
| Generation (T=0.7) | ✓ | Works with embedded weights |
| Tool detection | ✓ | Parses <tool> tags correctly |
| Layer inspection | ✓ | Can access all 36 layers |
| Weight surgery | ✓ | Scale/prune/noise work |
| Rollback | ✓ | Checkpoint/restore works |

---

## File Locations

- **Model:** `checkpoints/Flux-Apex-V1.flx` (v5.2-custom-vlm)
- **Notebook:** `notebooks/flux_vlm_native_embed.ipynb`
- **Native VLM:** `phases/phase_vlm_native/vlm_architecture.py`
- **SVD Utils:** `phases/phase_vlm_native/vlm_svd.py`
- **Handoff Dir:** `handoff/`

---

## Architecture Diagram

```
FLUX Custom VLM Architecture
═════════════════════════════

┌─────────────────────────────────────────────────────────────┐
│                      User Input                              │
│                    (text or image)                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      FluxProcessor                          │
│   - Apply chat template                                      │
│   - Handle $WAVE_VARIABLES                                   │
│   - Extract tool calls                                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                       FluxVLM                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Pre-Generate Hooks                       │   │
│  │   - Wave context injection                            │   │
│  │   - System prompt prepending                          │   │
│  └─────────────────────────┬────────────────────────────┘   │
│                             │                                │
│  ┌──────────────────────────▼────────────────────────────┐   │
│  │           Qwen2.5-VL-3B Backbone                      │   │
│  │  ┌────────────────────────────────────────────────┐   │   │
│  │  │ Embed Tokens (151936 vocab)                    │   │   │
│  │  ├────────────────────────────────────────────────┤   │   │
│  │  │ 36 Transformer Layers                          │   │   │
│  │  │   - Self-Attention (GQA: 16 heads, 2 KV)       │   │   │
│  │  │   - MLP (2048 → 11008 → 2048)                  │   │   │
│  │  │   - Layer hooks available                      │   │   │
│  │  ├────────────────────────────────────────────────┤   │   │
│  │  │ LM Head (2048 → 151936)                        │   │   │
│  │  └────────────────────────────────────────────────┘   │   │
│  └─────────────────────────┬────────────────────────────┘   │
│                             │                                │
│  ┌──────────────────────────▼────────────────────────────┐   │
│  │              Post-Token Hooks                         │   │
│  │   - Tool call detection (<tool>...</tool>)            │   │
│  │   - Early stopping                                    │   │
│  └─────────────────────────┬────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    ToolInjector                             │
│   - Parse tool calls                                         │
│   - Execute via FLUX components                              │
│   - Inject results back                                      │
└──────────────────────────────────────────────────────────────┘

Wave Integration Points:
────────────────────────

   CSE Output                          VLM Hidden
  [seq, 432]   ──► WaveToVLMBridge ──► [seq, 2048]
                                             │
                                             ▼
                                    Inject at layer N
                                             │
                                             ▼
                              VLM generates with context
                                             │
  [seq, 432]   ◄── VLMToWaveBridge ◄── [seq, 2048]
   Wave Out                            Hidden States
```

---

## Instructions for Next AI Agent

Hello, future AI! Here's how to continue this work:

### 1. START HERE
Read this file (handoff/HANDOFF.md) for complete context.

### 2. LOAD THE MODEL
```python
import torch
model = torch.load('checkpoints/Flux-Apex-V1.flx', map_location='cpu', weights_only=False)
assert model['version'] == '5.2-custom-vlm'
```

### 3. KEY GOAL
Remove HuggingFace dependency by using native VLM implementation.
The code is in `phases/phase_vlm_native/vlm_architecture.py`

### 4. WHAT WORKS
- FluxVLM wrapper with embedded weights
- Generation with hooks
- Tool detection
- Model surgery

### 5. WHAT NEEDS WORK
- Train bridge layers (currently random)
- Use native VLM instead of HF
- Connect tools to actual FLUX components
- Test on ARC puzzles

### 6. QUICK TEST
Run `notebooks/flux_vlm_native_embed.ipynb` to verify everything works.

---

## API Reference

### FluxVLM Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `__init__` | `config: Dict, weights: Dict, device: str` | - | Initialize wrapper |
| `load_backend` | `use_hf_model: bool` | - | Load underlying model |
| `generate` | `prompt: str, max_new_tokens: int, temperature: float` | `str` | Generate text |
| `get_layer` | `layer_name: str` | `nn.Module` | Access layer by name |
| `inject_wave_context` | `wave: Tensor, layer: int` | - | Set wave for injection |
| `register_pre_generate_hook` | `hook: callable` | - | Add pre-gen hook |
| `register_post_token_hook` | `hook: callable` | - | Add post-token hook |
| `introspect` | - | `Dict` | Get model info |

### WeightSurgeon Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `scale_layer_weights` | `layer_name: str, factor: float` | `int` | Scale weights |
| `prune_layer` | `layer_name: str, threshold: float` | `Dict` | Zero small weights |
| `add_noise` | `layer_name: str, std: float` | `int` | Add Gaussian noise |
| `quantize_layer` | `layer_name: str, bits: int` | `Dict` | Simulate quantization |

---

## Contact / Context

This handoff was generated by an AI agent working on FLUX VLM integration.
The goal is to enable fully self-contained VLM generation without HuggingFace
runtime dependencies, with full control over inference via hooks and surgery.

**Key Insight:** The embedded weights work, but the architecture class still
comes from HuggingFace. The next step is using `vlm_architecture.py` to
implement the transformer natively.

---

*FLUX v5.2-custom-vlm — Physics-inspired AI with embedded VLM and full control*
