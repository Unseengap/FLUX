# Phase 12 Specification: FLUX Multi-Modal Agent
## The Final Assembly — Connect Spatial Vision to LLM Reasoning

> Prerequisites:
> - `Flux-Base.flx` (Contains ALL FLUX phases 1-11 components)
> - Qwen2.5-Omni (Unified vision + audio + text model)
> - Spatial Memory (Phase 9 ARC — Ice/Water field navigation)
> - Grid Adapters (Phase 8.8 — GridToWave, WaveToGrid)
> - Cognitive Layer (phase_unified — CausalTracker, RuleInducer, GoalPlanner)
>
> **Copilot:** Open this file. Flux-Base.flx already has everything.
>
> **KEY INSIGHT:** Use Qwen2.5-Omni (7B) instead of separate models.
> One model handles: Vision + Audio + Voice + Text = saves RAM!

---

## The Breakthrough: Spatial Memory Already SEES the Game

```
ls20 Curiosity Ice Field (actual output):
┌─────────────────────────────────────────────────────────┐
│   🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊│  ← The ice field
│   🧊          ██████        🧊                  🧊│     PERFECTLY maps
│   🧊          ██    ██      🧊                  🧊│     the game puzzle
│   🧊          ██    ██      🧊                  🧊│     structure!
│   🧊          ██████████████🧊                  🧊│
│   🧊          ██    ██      🧊                  🧊│  ← Human can see
│   🧊██████████              🧊                  🧊│     this is a maze
│   🧊                        🧊                  🧊│     with arrows
│   🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊🧊│
└─────────────────────────────────────────────────────────┘
```

**The problem:** The current agent has this perfect visual map, but uses basic heuristics.
**The solution:** Give this map to the LLM (Qwen) and let it REASON about what to do!

---

## Architecture: Connect Vision to Reasoning

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     FLUX Multi-Modal Agent (Phase 12)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │                         GAME FRAME                                  │   │
│   │                      (64×64 grid from API)                          │   │
│   └────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │                      SPATIAL MEMORY                                 │   │
│   │   ┌─────────────────────┐    ┌─────────────────────┐               │   │
│   │   │  EXPLORATION MASS   │    │   CURIOSITY ICE     │               │   │
│   │   │    (Water Field)    │    │    (Ice Field)      │               │   │
│   │   │   Where we've been  │    │  What's interesting │               │   │
│   │   └─────────────────────┘    └─────────────────────┘               │   │
│   │                │                        │                           │   │
│   │                └────────┬───────────────┘                           │   │
│   │                         ▼                                           │   │
│   │              ┌─────────────────────┐                                │   │
│   │              │   ASCII MAP VIEW    │  ← NEW: Convert fields to     │   │
│   │              │   for LLM to read   │    human-readable text!       │   │
│   │              └─────────────────────┘                                │   │
│   └────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │                     LLM REASONING (Qwen)                            │   │
│   │                                                                     │   │
│   │   PROMPT:                                                           │   │
│   │   "You are playing ls20, a puzzle game.                             │   │
│   │    Current position: (32, 20)                                       │   │
│   │    Available actions: UP, DOWN, LEFT, RIGHT                         │   │
│   │                                                                     │   │
│   │    ASCII MAP (🧊 = interesting, ● = explored, + = target):          │   │
│   │    [the spatial memory visualization]                               │   │
│   │                                                                     │   │
│   │    OBSERVATION: I see a maze with arrows. The goal seems to be      │   │
│   │    stepping on + symbols. The ice shows interesting areas I         │   │
│   │    haven't fully explored yet.                                      │   │
│   │                                                                     │   │
│   │    What action should I take? Think step by step."                  │   │
│   │                                                                     │   │
│   │   OUTPUT → Reasoned action with explanation                         │   │
│   └────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │                     CAUSAL TRACKER                                  │   │
│   │   action (LEFT) → effect (arrow rotated)                            │   │
│   │   Learn: stepping on + rotates nearby arrows                        │   │
│   └────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │                     EXECUTE ACTION                                  │   │
│   │                  Send to ARC-AGI-3 API                              │   │
│   └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## What Gets Built

```
phases/phase12/
├── PHASE_12_SPEC.md              ← This file (copy to phases/)
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
```

### New Checkpoints

```
checkpoints/
├── Flux-Base.flx                 ← MAIN: Contains ALL phases 1-11
│   ├── CSE, Field, GR, TL, CGN, Memory (Phase 1-7)
│   ├── ByteDecoder (Phase 8)
│   ├── GridToWave, WaveToGrid, Image, Audio adapters (Phase 8.8-8.9)
│   ├── Wave+Byte hybrid, TaskRouter (Phase 10)
│   └── LLM bridge config (Phase 11)
│
├── Flux-multi-model.flx          ← OUTPUT: Phase 12 additions
│   ├── Inherits all from Flux-Base.flx
│   ├── Qwen-Omni reference (vision+audio+text unified)
│   ├── VisualReasoner state
│   ├── Learned game rules
│   └── Cross-session memory
```

### LLM Choice: Qwen2.5-Omni (Unified Multi-Modal)

```
┌─────────────────────────────────────────────────────────────────┐
│                  WHY QWEN-OMNI?                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  OLD APPROACH (Phase 11):           NEW APPROACH (Phase 12):    │
│  ┌─────────────┐                    ┌─────────────────────┐     │
│  │ Qwen-3B     │ ← Text only        │   Qwen2.5-Omni-7B   │     │
│  └─────────────┘                    │                     │     │
│  ┌─────────────┐                    │  ┌───┐ ┌───┐ ┌───┐ │     │
│  │ CLIP        │ ← Vision           │  │ 👁 │ │ 🔊 │ │ 📝 │ │     │
│  └─────────────┘                    │  │vis│ │aud│ │txt│ │     │
│  ┌─────────────┐                    │  └───┘ └───┘ └───┘ │     │
│  │ Whisper     │ ← Audio            │                     │     │
│  └─────────────┘                    │  ONE MODEL DOES ALL │     │
│                                     └─────────────────────┘     │
│  ~12GB VRAM total                   ~8GB VRAM (4-bit)           │
│  Complex bridging                   Single inference call       │
│  Separate embeddings                Unified representation      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Qwen2.5-Omni capabilities:**
- **Vision**: Can see images/grids directly (no ASCII conversion needed!)
- **Audio**: Can process game sounds if available  
- **Text**: Full language reasoning
- **Voice**: Can generate audio responses
- **All in one 7B model** that runs on T4 GPU with 4-bit quantization

---

## Component 1: VisualReasoner — Give the LLM Eyes

**Two Vision Modes (Qwen-Omni enables both!):**

```
MODE 1: ASCII Text (fallback)         MODE 2: Direct Image (preferred)
┌─────────────────────────────┐       ┌─────────────────────────────┐
│ @ = my position             │       │ [Actual rendered image of   │
│ ! = high curiosity          │       │  the game grid + overlays   │
│ + = target                  │       │  showing Ice/Water fields]  │
│ ● = explored                │       │                             │
│                             │       │  Qwen-Omni can DIRECTLY     │
│ ●●●●●!!!                    │       │  see colors, shapes, and    │
│ ●●●@●!!!                    │       │  spatial relationships!     │
│ ●●●●●!!!                    │       │                             │
└─────────────────────────────┘       └─────────────────────────────┘
~300 tokens                            Image embedding (fast)
```

```python
class VisualReasoner:
    """
    Bridge between spatial memory visualization and LLM reasoning.
    
    With Qwen-Omni, we have TWO options:
    1. ASCII text representation (works with any LLM)
    2. Direct image input (Qwen-Omni's native vision!)
    
    The direct image approach is BETTER because the model can see:
    - Actual colors (not just symbols)
    - Spatial relationships naturally
    - Fine details that ASCII misses
    """
    
    def __init__(
        self,
        llm_model: FLUXAugmentedLLM,
        spatial_memory: SpatialMemory,
        verbose: bool = True,
    ):
        self.llm = llm_model
        self.spatial = spatial_memory
        self.verbose = verbose
    
    def render_spatial_to_text(
        self,
        grid: List[List[int]],
        position: Tuple[int, int],
        show_size: Tuple[int, int] = (20, 40),
    ) -> str:
        """
        Convert spatial memory fields to ASCII text for LLM.
        
        Creates a visual representation showing:
        - Current position (@)
        - Explored areas (●)
        - High curiosity/ice areas (!)
        - Unexplored areas (.)
        - Obstacles/walls (#)
        - Targets/goals (+)
        
        Returns:
            ASCII string the LLM can reason about
        """
        h, w = show_size
        grid_h, grid_w = len(grid), len(grid[0])
        
        # Get fields
        mass = self.spatial.exploration_mass[:grid_h, :grid_w]
        ice = self.spatial.curiosity_field[:grid_h, :grid_w]
        
        lines = []
        for r in range(min(h, grid_h)):
            row = ""
            for c in range(min(w, grid_w)):
                cell_val = grid[r][c] if r < grid_h and c < len(grid[r]) else 0
                m = mass[r, c].item()
                i = ice[r, c].item()
                
                # Position marker
                if (r, c) == position:
                    row += "@"
                # High curiosity (ice) = interesting!
                elif i > 10:
                    row += "!"
                # Target patterns (different colors usually mean targets)
                elif cell_val not in [0, self._get_background(grid)] and m < 5:
                    row += "+"
                # Walls/obstacles (background color)
                elif cell_val == 0:
                    row += "#"
                # Explored
                elif m > 5:
                    row += "●"
                # Lightly explored
                elif m > 0:
                    row += "·"
                # Unexplored
                else:
                    row += " "
            lines.append(row)
        
        return "\n".join(lines)
    
    def render_grid_to_image(
        self,
        grid: List[List[int]],
        position: Tuple[int, int],
        ice_overlay: bool = True,
    ) -> "PIL.Image":
        """
        Render grid as actual image for Qwen-Omni's vision.
        
        This is BETTER than ASCII because the model sees:
        - True colors (not symbols)
        - Spatial relationships naturally
        - Ice field as semi-transparent overlay
        
        Returns:
            PIL Image ready for Qwen-Omni vision input
        """
        import numpy as np
        from PIL import Image, ImageDraw
        
        # ARC color palette (standard 10 colors)
        COLORS = [
            (0, 0, 0),        # 0: black
            (0, 116, 217),    # 1: blue
            (255, 65, 54),    # 2: red
            (46, 204, 64),    # 3: green
            (255, 220, 0),    # 4: yellow
            (128, 128, 128),  # 5: grey
            (240, 18, 190),   # 6: magenta
            (255, 133, 27),   # 7: orange
            (127, 219, 255),  # 8: cyan
            (135, 12, 37),    # 9: brown
        ]
        
        cell_size = 10  # Pixels per cell
        h, w = len(grid), len(grid[0])
        
        # Create base image from grid
        img = Image.new('RGB', (w * cell_size, h * cell_size))
        draw = ImageDraw.Draw(img)
        
        for r in range(h):
            for c in range(w):
                color = COLORS[min(grid[r][c], 9)]
                x0, y0 = c * cell_size, r * cell_size
                x1, y1 = x0 + cell_size - 1, y0 + cell_size - 1
                draw.rectangle([x0, y0, x1, y1], fill=color)
        
        # Overlay ice field (semi-transparent cyan)
        if ice_overlay:
            ice = self.spatial.curiosity_field[:h, :w]
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            draw_overlay = ImageDraw.Draw(overlay)
            
            for r in range(h):
                for c in range(w):
                    ice_val = ice[r, c].item()
                    if ice_val > 5:
                        alpha = min(int(ice_val * 10), 180)
                        x0, y0 = c * cell_size, r * cell_size
                        x1, y1 = x0 + cell_size - 1, y0 + cell_size - 1
                        draw_overlay.rectangle([x0, y0, x1, y1], fill=(0, 255, 255, alpha))
            
            img = Image.alpha_composite(img.convert('RGBA'), overlay)
        
        # Mark agent position
        pr, pc = position
        draw = ImageDraw.Draw(img)
        cx, cy = pc * cell_size + cell_size // 2, pr * cell_size + cell_size // 2
        draw.ellipse([cx-4, cy-4, cx+4, cy+4], fill=(255, 255, 255), outline=(0, 0, 0))
        
        return img.convert('RGB')
    
    def reason_with_vision(
        self,
        game_id: str,
        grid: List[List[int]],
        position: Tuple[int, int],
        available_actions: List[int],
        history: List[Dict] = None,
    ) -> Tuple[int, str]:
        """
        Use Qwen-Omni's NATIVE VISION to reason about the game.
        
        Instead of converting to ASCII, we pass:
        1. The actual grid rendered as an image
        2. Ice field overlay showing curiosity
        3. Agent position marked
        
        This is more natural for visual reasoning!
        """
        # Render grid as image
        game_image = self.render_grid_to_image(grid, position, ice_overlay=True)
        
        # Build action descriptions
        action_names = {
            1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
            5: "INTERACT", 6: "CLICK", 7: "UNDO"
        }
        available = [action_names.get(a, f"ACTION{a}") for a in available_actions]
        
        # Prompt for Qwen-Omni with image
        prompt = f"""You are playing {game_id}, a puzzle game.

Look at this game image. The white dot is my current position.
Cyan/teal overlays show areas of high curiosity (interesting spots).

Available actions: {', '.join(available)}

Based on what you SEE in the image:
1. What patterns or structures do you notice?
2. What seems to be the goal?
3. Which direction should I move?

Think step by step, then end with: ACTION: [your choice]"""
        
        # Call Qwen-Omni with image + text
        response = self.llm.chat_with_image(
            image=game_image,
            prompt=prompt,
            max_new_tokens=300,
            temperature=0.3,
        )
        
        action = self._parse_action(response, available_actions)
        return action, response
    
    def reason_about_game(
        self,
        game_id: str,
        grid: List[List[int]],
        position: Tuple[int, int],
        available_actions: List[int],
        history: List[Dict] = None,
    ) -> Tuple[int, str]:
        """
        Use LLM to reason about the game and decide action.
        
        Args:
            game_id: Game identifier (e.g., "ls20")
            grid: Current game frame
            position: Agent position
            available_actions: List of available action IDs
            history: Recent action history
            
        Returns:
            (action_id, reasoning_text)
        """
        # Convert spatial memory to visual text
        visual_map = self.render_spatial_to_text(grid, position)
        
        # Build action descriptions
        action_names = {
            1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
            5: "INTERACT", 6: "CLICK", 7: "UNDO"
        }
        available = [action_names.get(a, f"ACTION{a}") for a in available_actions]
        
        # Build prompt
        prompt = f"""You are playing {game_id}, a puzzle game from ARC-AGI-3.

CURRENT STATE:
- Position: row {position[0]}, column {position[1]}
- Available actions: {', '.join(available)}

WHAT I SEE (ASCII map):
@ = my position
! = interesting area (high curiosity)
+ = potential target/goal
● = explored area
· = lightly visited
# = wall/obstacle
(space) = unexplored

```
{visual_map}
```

RECENT HISTORY:
{self._format_history(history) if history else "Just started"}

TASK: Analyze this puzzle visually. What patterns do you see? What should I do next?

Think step by step, then end with: ACTION: [your chosen action]"""
        
        # Get LLM response
        response = self.llm.chat(prompt, max_new_tokens=300, temperature=0.3)
        
        # Parse action from response
        action = self._parse_action(response, available_actions)
        
        return action, response
    
    def _parse_action(self, response: str, available: List[int]) -> int:
        """Extract action from LLM response."""
        response_upper = response.upper()
        
        # Try to find "ACTION: X" pattern
        if "ACTION:" in response_upper:
            after = response_upper.split("ACTION:")[-1].strip()
            for action_id, name in {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 
                                     5: "INTERACT", 6: "CLICK"}.items():
                if after.startswith(name) and action_id in available:
                    return action_id
        
        # Fallback: look for action word anywhere
        for action_id, name in {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}.items():
            if name in response_upper and action_id in available:
                return action_id
        
        # Last resort: random available
        import random
        return random.choice(available) if available else 1
```

---

## Component 2: FLUXMultiAgent — The Complete Agent

```python
"""
flux_multi_agent.py — Phase 12 Multi-Modal Agent

Uses flux_model.py and flux_utils.py for proper .flx handling.
Loads from Flux-Base.flx, saves to Flux-multi-model.flx with FULL state.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# Add project root
FLUX_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(FLUX_ROOT))

# Use the OFFICIAL flux utilities
from flux_model import FLUXModel, load_flux, COMPONENT_CATEGORIES
from flux_utils import (
    PhaseLogger, 
    PhaseResults, 
    get_device, 
    save_and_upload_checkpoint,
    _resolve_hf_token,
)


class FLUXMultiAgent:
    """
    The complete FLUX agent that combines:
    - Spatial Memory (visual mapping)
    - LLM Reasoning (Qwen-Omni — unified vision+audio+text)
    - Causal Tracking (action → effect learning)
    - Game Memory (per-game episodic storage)
    
    This is the final assembly of all FLUX components.
    
    IMPORTANT: Uses FLUXModel from flux_model.py for proper .flx handling!
    - Loads FULL base from Flux-Base.flx
    - Saves FULL updated state to Flux-multi-model.flx
    - LLM referenced by name (not stored in .flx)
    
    KEY: Uses Qwen2.5-Omni instead of separate models!
    - ONE model for vision + audio + text
    - Can directly "see" grid images (no ASCII conversion needed)
    - Saves ~4GB VRAM vs loading separate CLIP/Whisper
    """
    
    def __init__(
        self,
        flx_path: str = "checkpoints/Flux-Base.flx",  # Main checkpoint!
        device: str = None,
        log: Optional[PhaseLogger] = None,
    ):
        self.device = device or get_device()
        self.log = log or PhaseLogger(phase=12)
        self.flx_path = Path(flx_path)
        
        self.log.separator("Initializing FLUXMultiAgent")
        
        # ═══════════════════════════════════════════════════════════
        # Step 1: Load base model using FLUXModel (flux_model.py)
        # ═══════════════════════════════════════════════════════════
        self.log.info(f"Loading base from {flx_path}...")
        self.flux_model = FLUXModel.load(flx_path)
        self.flux_model.summary()
        
        # ═══════════════════════════════════════════════════════════
        # Step 2: Initialize LLM with Qwen-Omni (vision+audio+text)
        # ═══════════════════════════════════════════════════════════
        self.log.info("Initializing Qwen-Omni LLM bridge...")
        
        # Update LLM reference in the model
        self.flux_model.config.generation.llm_name = "Qwen/Qwen2.5-Omni-7B"
        self.flux_model.config.generation.llm_primary = True
        self.flux_model.config.generation.load_in_4bit = True
        
        # Additional Omni-specific config
        self.flux_model.config.generation.enable_vision = True
        self.flux_model.config.generation.enable_audio = False
        
        # Load the actual LLM
        from flux_augmented_llm import FLUXAugmentedLLM, FLUXAugmentedConfig
        
        llm_config = FLUXAugmentedConfig(
            llm_name="Qwen/Qwen2.5-Omni-7B",
            load_in_4bit=True,
            flx_path=str(flx_path),
            load_full_flux=True,
            load_adapters=True,
        )
        self.llm_model = FLUXAugmentedLLM(llm_config, device=self.device)
        self.log.success("Qwen-Omni LLM ready")
        
        # ═══════════════════════════════════════════════════════════
        # Step 3: Initialize Spatial Memory
        # ═══════════════════════════════════════════════════════════
        sys.path.insert(0, str(FLUX_ROOT / 'phases' / 'phase9_arc'))
        from spatial_memory import SpatialMemory
        
        self.spatial_memory = SpatialMemory(max_size=64, device=self.device)
        self.log.success("SpatialMemory (Ice/Water) ready")
        
        # ═══════════════════════════════════════════════════════════
        # Step 4: Initialize Visual Reasoner
        # ═══════════════════════════════════════════════════════════
        self.reasoner = VisualReasoner(
            llm_model=self.llm_model,
            spatial_memory=self.spatial_memory,
        )
        self.log.success("VisualReasoner ready")
        
        # ═══════════════════════════════════════════════════════════
        # Step 5: Initialize Cognitive Components
        # ═══════════════════════════════════════════════════════════
        sys.path.insert(0, str(FLUX_ROOT / 'phases' / 'phase_unified'))
        from causal_tracker import CausalTracker
        from rule_inducer import RuleInducer
        
        self.causal_tracker = CausalTracker(max_history=1000, device=self.device)
        self.rule_inducer = RuleInducer(causal_tracker=self.causal_tracker)
        self.log.success("CausalTracker + RuleInducer ready")
        
        # ═══════════════════════════════════════════════════════════
        # Step 6: Game State
        # ═══════════════════════════════════════════════════════════
        self.current_game = None
        self.action_history = []
        self.observation_count = 0
        self.learned_rules = []  # Persist across saves
        
        # Mark Phase 12 components as active
        self.flux_model.components['spatial_memory'] = True
        self.flux_model.components['visual_reasoner'] = True
        self.flux_model.components['causal_tracker'] = True
        self.flux_model.components['rule_inducer'] = True
        
        self.log.separator("FLUXMultiAgent Initialized")
        self.log.info(f"Active components: {self.flux_model.active_component_count}")
    
    def reset(self, game_id: str):
        """Reset for new game."""
        self.current_game = game_id
        self.spatial_memory.reset()
        self.action_history = []
        self.observation_count = 0
        
        # Teach the LLM about this game type
        self._teach_game_context(game_id)
    
    def _teach_game_context(self, game_id: str):
        """Teach LLM context about the game type."""
        game_contexts = {
            'ls20': "ls20 is a maze puzzle where stepping on + symbols rotates arrows. "
                    "The goal is to make all arrows point in the same direction as the indicator.",
            'ft09': "ft09 is a logic puzzle where you click cells to transform patterns. "
                    "Look for repeating patterns and click to apply transformations.",
            'vc33': "vc33 is an orchestration puzzle. Click cells in the correct sequence "
                    "to achieve the target pattern.",
        }
        
        if game_id in game_contexts:
            self.llm_model.teach(game_contexts[game_id], verify=False)
    
    def observe(self, frame: List[List[int]], position: Tuple[int, int]):
        """Process observation from game."""
        self.observation_count += 1
        
        # Update spatial memory
        self.spatial_memory.observe(
            position=position,
            local_view=frame,
            global_grid=frame,
        )
        
        # Store in episodic memory
        wave = self.llm_model.encode(str(frame[:5]))
        self.llm_model.store(
            f"Game {self.current_game}, step {self.observation_count}: "
            f"At position {position}, grid size {len(frame)}x{len(frame[0])}",
            wave=wave,
        )
    
    def decide_action(
        self,
        frame: List[List[int]],
        position: Tuple[int, int],
        available_actions: List[int],
    ) -> Tuple[int, str]:
        """Use LLM reasoning to decide action."""
        action, reasoning = self.reasoner.reason_about_game(
            game_id=self.current_game,
            grid=frame,
            position=position,
            available_actions=available_actions,
            history=self.action_history[-5:],
        )
        
        self.action_history.append({
            'action': action,
            'position': position,
            'reasoning': reasoning[:100],
        })
        
        return action, reasoning
    
    def record_effect(self, old_grid, new_grid, action: int):
        """Record action → effect for causal learning."""
        import numpy as np
        self.causal_tracker.record(
            position=self.action_history[-1]['position'] if self.action_history else (0, 0),
            action=action,
            grid_before=np.array(old_grid),
            grid_after=np.array(new_grid),
        )
        
        new_rules = self.rule_inducer.try_induce()
        if new_rules:
            for rule in new_rules:
                self.learned_rules.append({
                    'condition': rule.condition,
                    'effect': rule.effect,
                    'confidence': rule.confidence,
                })
                self.llm_model.teach(
                    f"Learned rule: {rule.condition} → {rule.effect}",
                    verify=False,
                )
    
    # ═══════════════════════════════════════════════════════════════
    # SAVE: Using FLUXModel properly (flux_model.py)
    # ═══════════════════════════════════════════════════════════════
    
    def save_flx(
        self, 
        path: str = "checkpoints/Flux-MULTI.flx",
        upload_to_hf: bool = True,
    ):
        """
        Save the COMPLETE multi-modal agent to .flx format.
        
        Uses FLUXModel from flux_model.py to ensure:
        - Full base model is preserved (all Phase 1-11 components)
        - Phase 12 additions are properly added
        - LLM is referenced by name (not stored)
        - Format is valid and can be reloaded
        
        Args:
            path: Output path for .flx file
            upload_to_hf: Whether to upload to HuggingFace Hub
        """
        import torch
        
        self.log.separator("Saving Flux-multi-model.flx")
        
        # ─────────────────────────────────────────────
        # Update version and metadata
        # ─────────────────────────────────────────────
        self.flux_model.version = '4.0-multi-modal'
        self.flux_model.phase = 'phase12'
        self.flux_model.metadata['phase'] = 12
        self.flux_model.metadata['description'] = 'FLUX Multi-Modal Agent with Visual Reasoning'
        self.flux_model.metadata['saved'] = datetime.now().isoformat()
        self.flux_model.metadata['capabilities'] = [
            'spatial_vision',
            'llm_reasoning',
            'causal_learning',
            'game_memory',
            'qwen_omni_vision',
        ]
        
        # ─────────────────────────────────────────────
        # Add Phase 12 component states
        # ─────────────────────────────────────────────
        
        # Spatial Memory state
        self.flux_model.add_component('spatial_memory', {
            'exploration_mass': self.spatial_memory.exploration_mass.cpu(),
            'curiosity_field': self.spatial_memory.curiosity_field.cpu(),
            'visit_count': self.spatial_memory.visit_count.cpu(),
            'last_observation': self.spatial_memory.last_observation.cpu(),
            'config': {
                'max_size': self.spatial_memory.max_size,
                'feature_dim': self.spatial_memory.feature_dim,
            },
        })
        self.log.success("Added spatial_memory state")
        
        # Causal Tracker state
        causal_links = []
        for link in self.causal_tracker.causal_links[-100:]:
            causal_links.append({
                'action': link.action,
                'position': link.position,
                'changes': link.changes if hasattr(link, 'changes') else [],
            })
        
        self.flux_model.add_component('causal_tracker', {
            'links': causal_links,
            'total_observations': len(self.causal_tracker.causal_links),
            'config': {'max_history': self.causal_tracker.max_history},
        })
        self.log.success("Added causal_tracker state")
        
        # Learned rules
        self.flux_model.add_component('rule_inducer', {
            'rules': self.learned_rules,
            'total_rules': len(self.learned_rules),
        })
        self.log.success("Added rule_inducer state")
        
        # Visual Reasoner config
        self.flux_model.add_component('visual_reasoner', {
            'config': {
                'vision_enabled': True,
                'ascii_fallback': True,
            },
        })
        self.log.success("Added visual_reasoner config")
        
        # ─────────────────────────────────────────────
        # Update LLM reference (NOT weights!)
        # ─────────────────────────────────────────────
        self.flux_model.state['llm_reference'] = {
            'name': "Qwen/Qwen2.5-Omni-7B",
            'load_in_4bit': True,
            'enable_vision': True,
            'enable_audio': False,
            'adapter': None,
        }
        self.log.info("LLM reference: Qwen/Qwen2.5-Omni-7B")
        
        # ─────────────────────────────────────────────
        # Save using FLUXModel.save() (proper format!)
        # ─────────────────────────────────────────────
        self.flux_model.save(path, overwrite=True)
        
        size_mb = Path(path).stat().st_size / 1e6
        self.log.success(f"Saved: {path} ({size_mb:.1f} MB)")
        self.log.info(f"Components: {self.flux_model.active_component_count}")
        
        # ─────────────────────────────────────────────
        # Upload to HuggingFace Hub
        # ─────────────────────────────────────────────
        if upload_to_hf:
            token = _resolve_hf_token()
            if token:
                try:
                    from huggingface_hub import HfApi
                    api = HfApi(token=token)
                    api.upload_file(
                        path_or_fileobj=str(path),
                        path_in_repo="checkpoints/Flux-multi-model.flx",
                        repo_id="UnseenGAP/FLUX",
                        commit_message=f"Phase 12: Flux-multi-model.flx — {datetime.now().isoformat()}",
                    )
                    self.log.success("Uploaded to HuggingFace Hub")
                except Exception as e:
                    self.log.warning(f"HF upload failed: {e}")
            else:
                self.log.warning("No HF_TOKEN — skipping upload")
        
        return path
    
    # ═══════════════════════════════════════════════════════════════
    # LOAD: Restore from saved .flx
    # ═══════════════════════════════════════════════════════════════
    
    @classmethod
    def from_flx(
        cls, 
        path: str = "checkpoints/Flux-multi-model.flx",
        device: str = None,
    ) -> 'FLUXMultiAgent':
        """
        Load a FLUXMultiAgent from saved .flx file.
        
        Uses FLUXModel.load() to properly restore all components.
        """
        device = device or get_device()
        log = PhaseLogger(phase=12)
        
        log.info(f"Loading FLUXMultiAgent from {path}...")
        
        # Create agent (loads base)
        agent = cls(flx_path=path, device=device, log=log)
        
        # Restore Phase 12 specific state
        flux = agent.flux_model
        
        # Restore spatial memory state
        if flux.has_component('spatial_memory'):
            sm_state = flux.get_component('spatial_memory')
            if isinstance(sm_state, dict) and 'exploration_mass' in sm_state:
                agent.spatial_memory.exploration_mass.copy_(
                    sm_state['exploration_mass'].to(device)
                )
                agent.spatial_memory.curiosity_field.copy_(
                    sm_state['curiosity_field'].to(device)
                )
                log.success("Restored spatial_memory state")
        
        # Restore learned rules
        if flux.has_component('rule_inducer'):
            ri_state = flux.get_component('rule_inducer')
            if isinstance(ri_state, dict) and 'rules' in ri_state:
                agent.learned_rules = ri_state['rules']
                log.success(f"Restored {len(agent.learned_rules)} learned rules")
        
        log.success("FLUXMultiAgent restored from .flx")
        return agent
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            'flux_version': self.flux_model.version,
            'active_components': self.flux_model.active_component_count,
            'episodic_entries': len(self.llm_model.memory) if hasattr(self.llm_model, 'memory') else 0,
            'learned_rules': len(self.learned_rules),
            'causal_links': len(self.causal_tracker.causal_links),
            'current_game': self.current_game,
            'observations': self.observation_count,
        }
```
```

---

## Component 3: SpatialToText — Visual Field Converter

```python
class SpatialToText:
    """
    Convert spatial memory fields to text formats for LLM consumption.
    
    Multiple output modes:
    1. ASCII art (simple visual)
    2. Coordinate list (structured)
    3. Natural language description (for complex reasoning)
    """
    
    @staticmethod
    def to_ascii(
        mass: Tensor,
        ice: Tensor,
        grid: List[List[int]],
        position: Tuple[int, int],
        width: int = 40,
        height: int = 20,
    ) -> str:
        """Convert to ASCII art."""
        # Implementation above in VisualReasoner
        pass
    
    @staticmethod
    def to_coordinates(
        ice: Tensor,
        threshold: float = 5.0,
    ) -> str:
        """
        Convert ice field to coordinate list.
        
        Returns something like:
        "High curiosity areas (ice > 5):
         - Row 10-15, Col 20-30: Large interesting region
         - Row 25, Col 52-54: Potential target cluster"
        """
        regions = []
        # Find contiguous high-ice regions
        high_ice = (ice > threshold).nonzero(as_tuple=False)
        
        if len(high_ice) == 0:
            return "No high-curiosity areas detected."
        
        # Cluster nearby points
        # (simplified - could use proper clustering)
        current_region = []
        for point in high_ice:
            r, c = point[0].item(), point[1].item()
            if not current_region:
                current_region = [(r, c)]
            elif abs(r - current_region[-1][0]) <= 2 and abs(c - current_region[-1][1]) <= 2:
                current_region.append((r, c))
            else:
                if len(current_region) >= 3:
                    regions.append(current_region)
                current_region = [(r, c)]
        
        if current_region and len(current_region) >= 3:
            regions.append(current_region)
        
        # Format as text
        lines = ["Interesting regions detected:"]
        for i, region in enumerate(regions[:5]):  # Top 5
            min_r = min(p[0] for p in region)
            max_r = max(p[0] for p in region)
            min_c = min(p[1] for p in region)
            max_c = max(p[1] for p in region)
            lines.append(f"  Region {i+1}: rows {min_r}-{max_r}, cols {min_c}-{max_c} ({len(region)} cells)")
        
        return "\n".join(lines)
    
    @staticmethod  
    def to_description(
        grid: List[List[int]],
        mass: Tensor,
        ice: Tensor,
        position: Tuple[int, int],
    ) -> str:
        """
        Generate natural language description for complex reasoning.
        """
        grid_h, grid_w = len(grid), len(grid[0])
        
        # Analyze grid composition
        colors = {}
        for row in grid:
            for val in row:
                colors[val] = colors.get(val, 0) + 1
        
        bg_color = max(colors, key=colors.get)
        non_bg = {k: v for k, v in colors.items() if k != bg_color}
        
        # Analyze exploration
        explored_pct = (mass > 0).sum().item() / (grid_h * grid_w) * 100
        ice_total = ice.sum().item()
        
        # Generate description
        desc = f"""Grid Analysis:
- Size: {grid_h} × {grid_w}
- Background color: {bg_color} ({colors[bg_color]} cells)
- Notable colors: {', '.join(f'{k}:{v}' for k, v in sorted(non_bg.items(), key=lambda x: -x[1])[:5])}
- My position: row {position[0]}, col {position[1]}

Exploration Status:
- {explored_pct:.1f}% of grid explored
- Total curiosity (ice): {ice_total:.1f}
- {'High curiosity in unexplored areas' if ice_total > 100 else 'Mostly explored interesting areas'}"""
        
        return desc
```

---

## Integration: Updated Play Loop

```python
def play_game_with_multi_agent(
    game_id: str,
    session,
    agent: FLUXMultiAgent,
    max_actions: int = 100,
):
    """
    Play a game using the full multi-modal agent.
    
    The LLM reasons about the spatial visualization to decide actions.
    """
    agent.reset(game_id)
    
    # Reset game
    response = session.post(f"{API}/cmd/RESET", json={"game_id": game_id, ...})
    frame = response["frame"]
    available = response["available_actions"]
    
    for step in range(max_actions):
        # Normalize frame
        grid = normalize_frame(frame)
        position = agent.find_position(grid)
        
        # Update spatial memory
        agent.observe(grid, position)
        
        # LLM REASONING — The key new part!
        action, reasoning = agent.decide_action(grid, position, available)
        
        print(f"\n[Step {step}]")
        print(f"LLM Reasoning: {reasoning[:200]}...")
        print(f"Decided: ACTION {action}")
        
        # Track previous state for causal learning
        old_grid = [row[:] for row in grid]
        
        # Execute action
        response = session.post(f"{API}/cmd/ACTION{action}", ...)
        frame = response["frame"]
        available = response["available_actions"]
        
        # Learn from effect
        new_grid = normalize_frame(frame)
        agent.record_effect(old_grid, new_grid, action)
        
        if response["state"] != "NOT_FINISHED":
            break
    
    # Save learned model
    agent.save_flx("checkpoints/Flux-multi-model.flx")
```

---

## Training/Fine-tuning Schedule

### Phase 12a: Integration (2 hours)

```
1. Load Flux-Base.flx (ALL Phase 1-11 components)
2. Initialize Qwen2.5-Omni with 4-bit quantization
3. Implement grid-to-image renderer
4. Test vision input works (Qwen sees grid correctly)
```

### Phase 12b: Vision Reasoning (4 hours)

```
1. Test Qwen-Omni describes game grids correctly
2. Compare: ASCII text vs direct image (which works better?)
3. Tune reasoning temperature (lower = more consistent)
4. Add few-shot examples in prompts if needed
```

### Phase 12c: Live Testing (4 hours)

```
1. Test on ARC-AGI-3 API (ls20, ft09, vc33)
2. Compare scores: heuristic agent vs vision-LLM agent
3. Record learned rules across games
4. Save final Flux-multi-model.flx
```

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

## Expected File Sizes

| Checkpoint | Contents | Size |
|------------|----------|------|
| Flux-Base.flx | ALL Phases 1-11 (main checkpoint) | ~500 MB |
| Flux-multi-model.flx | Phase 12 additions + learned rules | ~550 MB |

**LLM Loading Strategy:**
- LLM weights NOT stored in .flx (too large)
- Flux-Base.flx stores: `llm_name = "Qwen/Qwen2.5-Omni-7B"`
- On load: Download Qwen-Omni from HuggingFace (~4GB with 4-bit)
- Total VRAM: ~8GB (fits T4 GPU with room to spare)

**RAM Savings vs Separate Models:**
```
OLD (Phase 11):                 NEW (Phase 12):
├── Qwen-3B       ~3GB          ├── Qwen-Omni-7B  ~4GB (4-bit)
├── CLIP          ~1GB          └── (vision built-in)
├── Whisper       ~1GB          └── (audio built-in)
└── TOTAL         ~5GB          └── TOTAL          ~4GB

PLUS: Unified embeddings = better cross-modal reasoning!
```

---

## Summary: The Final Assembly

```
Phase 12 connects everything:

┌─────────────────────────────────────────────────────────────┐
│                    Flux-multi-model.flx                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  LOAD FROM: Flux-Base.flx (contains everything below)       │
│                                                             │
│  Phase 1-8: FLUX Core                                       │
│    CSE, Field, GR, TL, CGN, Memory, ByteDecoder             │
│                                                             │
│  Phase 8.8-8.9: Adapters (in Flux-Base.flx)                 │
│    GridToWave, WaveToGrid, Image, Audio                     │
│                                                             │
│  Phase 9 ARC: Spatial Memory                                │
│    Exploration Mass (Water) + Curiosity Ice                 │
│                                                             │
│  Phase 10: Hybrid Generation                                │
│    Wave-mode + Byte-mode + TaskRouter                       │
│                                                             │
│  Phase 11: LLM Bridge (upgraded to Qwen-Omni)               │
│    Qwen2.5-Omni-7B (vision+audio+text unified)              │
│                                                             │
│  Phase 12: Visual Reasoning (NEW)                           │
│    - Direct image input ← Qwen-Omni SEES the grid!          │
│    - VisualReasoner ← LLM reasons from actual image         │
│    - Causal learning ← Discover game rules                  │
│    - Game memory ← Remember across sessions                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘

The result: An AI that can SEE game structure (real vision, not ASCII!),
REASON about it, LEARN rules from experience, and REMEMBER across sessions.

KEY ADVANTAGES OF QWEN-OMNI:
✓ Native vision — sees colors, shapes, spatial relationships
✓ One model — no separate CLIP/Whisper needed
✓ Saves ~1GB RAM vs multiple models
✓ Better cross-modal reasoning (unified embeddings)
✓ Can handle audio games too (built-in)
```

---

*FLUX: Field-based Latent Understanding eXperience*  
*Phase 12: The Final Assembly*  
*github.com/Unseengap/FLUX — huggingface.co/UnseenGAP/FLUX*
