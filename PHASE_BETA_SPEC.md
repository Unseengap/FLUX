# FLUX Beta: Unified Architecture & .flx Specification

## Overview

The `FLUX-Beta` phase marks the transition from fragmented sub-component development (Phases 1-8) to a stable, unified "Motherboard" architecture. In this phase, we ensure that every physical component of the FLUX system across all prior phases interoperates seamlessly and is packaged into the highly durable `.flx` format (`Flux-beta.flx`).

**Goal:** Export the unified FLUXLarge (with its Phase 8 WaveDecoder, Phase 6 Memory, Phase 5 Causal Geometry, Phase 3 Gravity, etc.) into a cohesive `.flx` artifact that can be loaded anywhere, maintaining all facts, attractors, and trained parameters without the risk of "dimension mismatch" or state wiping.

---

## 1. the `.flx` Modular Architecture

A `Flux-beta.flx` file is no longer a blind, melted-together PyTorch state dictionary. It is a strictly structured archive (a specialized dictionary) acting as a component registry.

### Core Structure:
```python
{
    "format": "FLUX",
    "version": "1.0-beta",
    "metadata": { ... },
    
    # ── Component 1: Continuous Semantic Encoder (Phase 1)
    "cse": {
        "config": {"wave_dim": 432, "byte_window": 8},
        "state_dict": { ... }
    },
    
    # ── Component 2: Field & Gravity (Phases 2 & 3)
    "field": {
        "config": {"h": 96, "w": 96, "d": 96, "features": 512},
        "state_dict": { ... },
        "gravity_state": { ... }
    },
    
    # ── Component 3: Memory Systems (Phase 6)
    "memory": {
        "episodic": { "index_data": [... numpy bytes ...], "metadata": { ... } },
        "semantic": { "graph_nodes": [...], "edges": [...] }
    },
    
    # ── Component 4: Generation (Phase 8)
    "decoder": {
        "config": {"hidden_dim": 1024, "layers": 4},
        "state_dict": { ... }
    }
}
```

### Component Invariants:
- **Language Invariant:** All components *must* communicate via the `[432]` Semantic Wave or the `[512]` Field Feature. 
- **Upgradability:** The loader must be able to swap out the `decoder` sector of a `.flx` file without touching the `memory` sector. If the decoder changes shape, memory remains perfectly intact.

---

## 2. Beta Evaluation & Demo Notebook

A new master notebook (`flux_beta_unified.ipynb`) will act as the ultimate smoke test and demo suite. Because FLUX combines physical thermodynamics with memory and continuous learning, the demos must push the system across *all* its unique capabilities.

### Required Notebook Workflow:

1. **Environment Setup & Logging:** Initialize `PhaseLogger`.
2. **Component Assembly:** Load the standard `phase8.phase.pt` from HuggingFace to get the latest trained weights.
3. **The `Flux-beta.flx` Export:** Repackage the PyTorch checkpoint into the strict `.flx` modular format. Save this file locally to Google Drive.
4. **Smoke Test (Reload):** Destroy the model in memory. Load it back entirely from `Flux-beta.flx`. Verify that memory count == expected count, parameters == expected params.
5. **Full Capabilities Demos (All logs/graphs saved between cells):**
   - **Demo A: Byte-level Encoding (No Tokenizer):** Feed it emojis, Arabic, and misspellings to prove the CSE never drops a token (Phase 1).
   - **Demo B: Thermodynamic Settling (Physics):** Visualize the field energy dropping as the model processes a complex sentence (Phases 2-4).
   - **Demo C: Zero-Forgetting Continual Learning:** Teach it 10 new fictional facts, test recall, teach 10 more, test recall of original 10 to prove exactly 0.0% forgetting (Phase 6).
   - **Demo D: Generative Autoregression:** Generate text using the Phase 8 WaveDecoder responding to standard prompts.
   - **Demo E: The "Motherboard" Stress Test:** Provide a prompt that requires recalling a learned fact from memory, settling the field to understand it causally, and autoregressively generating the answer.

---

## 3. Success Criteria

- [ ] `Flux-beta.flx` is generated, structured cleanly by component.
- [ ] Model successfully reconstructs from `.flx` without errors or warnings about `_orig_mod.` or dim mismatches.
- [ ] Demos successfully output their generated charts (`.png` / `.svg`) and logs to a dedicated `/output/flux_beta/` folder before moving to the next cell.
- [ ] All 5 capabilities (CSE, Settling, Memory, Causal nodes, Generation) are actively invoked and logged in the demos.