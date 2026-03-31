# FLUX Custom VLM Handoff — v5.2-custom-vlm

**Generated:** 2026-03-31
**Model Version:** 5.2-custom-vlm  
**File Size:** 13.18 GB  
**Status:** ✓ Foundation Complete, Needs Training  

---

## TL;DR — What You Need to Know

The FLUX model now has an embedded Qwen2.5-VL-3B VLM with **working generation** but **no tool calling** and **untrained bridges**. The VLM generates coherent text but doesn't know how to use FLUX's cognitive tools (`encode_grid`, `query_field`, etc.).

**Your Goal:** Make the VLM actually use FLUX tools for ARC puzzle solving.

---

## Quick Start

```python
import torch

# Load model
model = torch.load('checkpoints/Flux-Apex-V1.flx', map_location='cpu', weights_only=False)

# Check version
print(f"Version: {model['version']}")  # 5.2-custom-vlm
print(f"VLM weights: {len(model['vlm']['weights'])}")  # 824 tensors

# VLM is embedded but needs HF architecture to run
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)
vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# Load embedded weights with key remapping
weights = model['vlm']['weights']
remapped = {}
for k, v in weights.items():
    if k.startswith('model.'):
        remapped['model.language_model.' + k[6:]] = v
    elif k.startswith('visual.'):
        remapped['model.' + k] = v
    else:
        remapped[k] = v

vlm.load_state_dict(remapped, strict=False)  # 823/824 load
```

---

## Current Capabilities — What Works ✓

| Capability | Status | Evidence |
|------------|--------|----------|
| **VLM Weight Loading** | ✓ Working | 823/824 weights, 5/5 verified |
| **Text Generation** | ✓ Working | Coherent responses, 1.8-4s latency |
| **Temperature Control** | ✓ Working | T=0, 0.7, 1.0 all produce valid output |
| **FluxVLM Wrapper** | ✓ Working | Hooks, introspection, layer access |
| **Model Surgery** | ✓ Working | Scale, prune, noise, rollback all work |
| **Wave Bridges** | ✓ Created | 890K + 886K params, random init |
| **Save/Load** | ✓ Working | Verified round-trip |

### Test Results (Cell 9)

```
Temperature: 0.0  → 175 chars, 3.93s
Temperature: 0.7  → 213 chars, 2.63s  
Temperature: 1.0  → 171 chars, 1.83s
```

Response quality: Coherent paragraphs about FLUX (though not factually accurate about FLUX specifically).

---

## Critical Gaps — What's Broken ✗

### 1. Tool Calling Does NOT Work

**Problem:** When asked to use tools like `encode_grid`, the VLM writes *about* the tool instead of calling it.

**Test Prompt:**
```
Analyze this ARC grid pattern...
Use the encode_grid tool to analyze it, then describe the pattern.
```

**Actual Output:** Long explanation of what ARC grids are (no `<tool>` tags)

**Expected Output:**
```xml
<tool>encode_grid</tool>
<params>{"grid": [[0,1,0],[1,2,1],[0,1,0]], "mode": "holistic"}</params>
```

**Root Cause:** VLM was never fine-tuned on tool-use format.

### 2. Orchestration Not Loaded

**Problem:** `model['orchestration']` exists but has 0 tools.

```
Available tools: 0
```

This means even if VLM emitted `<tool>` tags, there's nothing to execute.

### 3. Bridge Layers Untrained

**Problem:** Wave↔VLM bridges have random weights.

```
Cosine similarity: 0.0026 (random init)
MSE: 2.12
```

**Impact:** Cannot inject FLUX wave context into VLM generations.

### 4. No Vision Testing

**Problem:** VLM supports images but vision pathway untested.

---

## The Goal — ARC Puzzle Solving

FLUX should solve ARC-AGI puzzles using this workflow:

```
User: "Solve this ARC puzzle: [input grid]"
      
VLM: <tool>encode_grid</tool>
     <params>{"grid": [...], "mode": "holistic"}</params>

System: [executes, returns wave]

VLM: <tool>query_field</tool>
     <params>{"wave": "$LAST_WAVE", "top_k": 5}</params>

System: [returns related patterns from memory]

VLM: "I see a rotation pattern. The output should be:
     [[1,0,1],
      [0,2,0],
      [1,0,1]]"
```

**Current Reality:** VLM just writes prose about grids instead of calling tools.

---

## Priority Task List

### P0: Enable Tool Calling (CRITICAL)

1. **Add orchestration tools to model**
   - Load from `DOCS/PHASE_ORCHESTRATOR_SPEC.md`
   - Include: `encode_grid`, `query_field`, `recall_memory`, `decode_grid`
   - Save to `model['orchestration']['tools']`

2. **Fine-tune VLM on tool-use format**
   - Create dataset of (prompt, tool_call) pairs
   - Format: User asks → VLM emits `<tool>name</tool><params>{...}</params>`
   - Fine-tune on 1000+ examples
   - Or use few-shot prompting in system prompt

3. **Connect ToolInjector to real components**
   - Replace mock `execute_tool()` with actual CSE, Field, Memory calls
   - Wire `encode_grid` → `model['adapters']['grid_to_wave']`
   - Wire `query_field` → `model['field']['state_dict']`

### P1: Train Bridge Layers

1. **Create wave-text pairs dataset**
   - Encode text with CSE → wave
   - Store (wave, text) pairs

2. **Contrastive training**
   - wave → WaveToVLMBridge → VLM hidden
   - text → VLM embedding → target hidden  
   - Loss: cosine similarity + reconstruction

3. **Target:** cos_sim > 0.9 after training

### P2: Vision Integration

1. Test image input through VLM
2. Verify `visual.*` weights work
3. Connect to ARC puzzle images

### P3: Remove HuggingFace Dependency

1. Implement native transformer in `phases/phase_vlm_native/vlm_architecture.py`
2. Load weights directly without `from_pretrained()`
3. No more network dependency at runtime

---

## Model Structure

```python
model = {
    'format': 'FLUX',
    'version': '5.2-custom-vlm',
    'phase': 'phase_vlm_native',
    
    'vlm': {
        'base_model': 'Qwen/Qwen2.5-VL-3B-Instruct',
        'quantization': 'fp16',
        'total_params': 3_754_622_976,
        'weights': {...},  # 824 tensors
        'config': {
            'hidden_size': 2048,
            'num_hidden_layers': 36,
            'num_attention_heads': 16,
            'num_key_value_heads': 2,
            'intermediate_size': 11008,
            'vocab_size': 151936,
        },
        'bridges': {
            'wave_to_vlm': {'wave_dim': 432, 'hidden_size': 2048, 'state_dict': ...},
            'vlm_to_wave': {'hidden_size': 2048, 'wave_dim': 432, 'state_dict': ...},
        },
        'custom_vlm': {
            'enabled': True,
            'wrapper_class': 'FluxVLM',
            'features': ['hooks', 'wave_injection', 'tool_detection'],
        },
    },
    
    # OTHER FLUX COMPONENTS
    'cse': {...},           # Continuous Semantic Encoder (bytes → 432D waves)
    'field': {...},         # Resonance Field (96³ × 512 knowledge storage)
    'memory': {...},        # Three-tier memory (working, episodic, semantic)
    'decoder': {...},       # Byte decoder (waves → text)
    'causal': {...},        # CGN nodes + causal graph
    'adapters': {
        'grid_to_wave': {...},  # ARC grid encoder
    },
    
    # NEEDS POPULATION
    'orchestration': {
        'tools': {},        # EMPTY — needs tool definitions
        'system_prompt': '',
    },
}
```

---

## Key Files

| File | Purpose |
|------|---------|
| `checkpoints/Flux-Apex-V1.flx` | The model (13.18 GB) |
| `notebooks/flux_vlm_native_embed.ipynb` | VLM integration notebook |
| `flux_utils.py` | Checkpoint management, HF upload |
| `.github/copilot-instructions.md` | AI agent instructions |
| `DOCS/PHASE_ORCHESTRATOR_SPEC.md` | Tool definitions |
| `DOCS/FLUX_APEX_V1.md` | Model reference |
| `phases/phase2/flux_format.py` | VLM loading utility |

---

## Critical Code Patterns

### Loading VLM (MUST use Qwen2.5, not Qwen2)

```python
# CORRECT
from transformers import Qwen2_5_VLForConditionalGeneration

# WRONG — architecture mismatch
from transformers import Qwen2VLForConditionalGeneration
```

### Weight Key Remapping

```python
# Embedded keys use: model.X, visual.X
# HuggingFace expects: model.language_model.X, model.visual.X

for emb_key, tensor in weights.items():
    if emb_key.startswith('model.'):
        hf_key = 'model.language_model.' + emb_key[6:]
    elif emb_key.startswith('visual.'):
        hf_key = 'model.' + emb_key
    else:
        hf_key = emb_key
    remapped[hf_key] = tensor
```

### Tool Call Format

```xml
<tool>tool_name</tool>
<params>{"key": "value"}</params>
```

Parse with:
```python
import re
pattern = r'<tool>\s*([^<]+?)\s*</tool>\s*<params>\s*([^<]+?)\s*</params>'
matches = re.findall(pattern, text, re.DOTALL)
```

---

## Bugs Already Fixed

| Issue | Symptom | Fix |
|-------|---------|-----|
| Wrong transformer class | MLP key mismatch | Use `Qwen2_5_VLForConditionalGeneration` |
| Key prefix mismatch | 0/5 verification | Remap `model.X` → `model.language_model.X` |
| Verification wrong | Pass load, fail verify | Use same remapping in verify |

---

## Run the Notebook

On Kaggle (Tesla T4 16GB):

```bash
# 1. Clone repo
git clone https://github.com/Unseengap/FLUX.git

# 2. Open notebook
notebooks/flux_vlm_native_embed.ipynb

# 3. Run all cells
# Expected: 18 cells, ~5 min total, all PASS
```

---

## Success Criteria

You've achieved the goal when:

- [ ] VLM emits `<tool>encode_grid</tool>` when asked to analyze grids
- [ ] ToolInjector executes the call and returns actual wave
- [ ] VLM uses tool results to produce correct ARC output
- [ ] Bridge similarity > 0.9 (wave ↔ VLM aligned)
- [ ] At least 1 ARC puzzle solved end-to-end

---

## Contact / Context

This handoff was generated after running `flux_vlm_native_embed.ipynb` on Kaggle. The VLM infrastructure is complete but needs training/fine-tuning to actually use FLUX tools.

**Key Insight:** The model can generate text. It just doesn't know it should call tools.

---

*FLUX v5.2-custom-vlm — Embedded VLM, needs tool training*
