# Phase 12 Specification: FLUX Multi-Modal Agent
## The Final Assembly — Connect Spatial Vision to LLM Reasoning

> Prerequisites:
> - `Flux-Base.flx` (Contains ALL FLUX phases 1-11 components)
> - Qwen2.5-Omni (Unified vision + audio + text model)
> - Spatial Memory (Phase 9 ARC — Ice/Water field navigation)
> - Grid Adapters (Phase 8.8 — GridToWave, WaveToGrid)
> - Cognitive Layer (phase_unified — CausalTracker, RuleInducer, GoalPlanner)
>
> **KEY INSIGHT:** Use Qwen2.5-Omni (7B) instead of separate models.
> One model handles: Vision + Audio + Voice + Text = saves RAM!

---

## The Breakthrough: Spatial Memory Already SEES the Game

The ice field PERFECTLY maps the game puzzle structure! 
The problem: The current agent has this perfect visual map, but uses basic heuristics.
The solution: Give this map to the LLM (Qwen) and let it REASON about what to do!

---

## What Gets Built

```
phases/phase12/
├── PHASE_12_SPEC.md              ← This file
├── flux_multi_agent.py           ← FLUXMultiAgent — unified reasoning agent
├── visual_reasoner.py            ← VisualReasoner — Qwen-Omni + grid images
├── grid_renderer.py              ← Render grids as images for vision input
├── action_parser.py              ← Parse LLM responses to game actions
├── game_memory.py                ← Per-game episodic memory
├── qwen_omni_bridge.py           ← Bridge for Qwen-Omni vision+text
├── demo_phase12_demo1.py         ← Demo: Watch agent SEE and reason through ls20
├── demo_phase12_demo2.py         ← Demo: Compare heuristic vs vision-LLM agent
├── test_phase12_test1.py         ← Test: Grid rendering + vision input
├── test_phase12_test2.py         ← Test: LLM action parsing
├── test_phase12_test3.py         ← Test: End-to-end game play
├── RESULTS_PHASE_12.md           ← Auto-generated results
├── __init__.py                  ← Module init
```

---

## Checkpoints

| Checkpoint | Contents | Size |
|------------|----------|------|
| Flux-Base.flx | ALL Phases 1-11 (main checkpoint) | ~500 MB |
| Flux-multi-model.flx | Phase 12 additions + learned rules | ~550 MB |

---

## LLM Choice: Qwen2.5-Omni (Unified Multi-Modal)

- **Vision**: Can see images/grids directly (no ASCII conversion needed!)
- **Audio**: Can process game sounds if available  
- **Text**: Full language reasoning
- **Voice**: Can generate audio responses
- **All in one 7B model** that runs on T4 GPU with 4-bit quantization (~8GB VRAM)

---

## Acceptance Criteria

| # | Criterion | Target |
|---|-----------|--------|
| 1 | Loads from Flux-Base.flx | ✓ All Phase 1-11 components |
| 2 | Qwen-Omni initializes | ✓ Vision + text working |
| 3 | Grid renders to image | ✓ Colors, ice overlay, agent marker |
| 4 | LLM sees image + reasons | ✓ Describes game structure correctly |
| 5 | Action parsing reliable | >90% correct parse rate |
| 6 | Causal learning records effects | ✓ Tracks action → change |
| 7 | ls20 shows improvement vs heuristic | Score > 0 (was 0 before) |
| 8 | Saves to Flux-multi-model.flx | ✓ All components serialized |
| 9 | Cross-session learning preserved | Rules learned stay in memory |
| 10 | VRAM usage ≤ 10GB | ✓ Fits T4 GPU |

---

*FLUX: Field-based Latent Understanding eXperience*  
*Phase 12: The Final Assembly*  
*github.com/Unseengap/FLUX — huggingface.co/UnseenGAP/FLUX*
