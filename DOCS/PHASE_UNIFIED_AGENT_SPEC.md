# FLUX Unified Multi-Modal ARC Agent Specification

> **Version:** 1.0
> **Date:** 2026-03-29
> **Status:** Active Development
> **API Key:** `da685361-745a-4c31-8116-0f16ec5c342f`

---

## 1. Executive Summary

This specification defines the **FLUX Unified Multi-Modal ARC Agent** — a vision-language agent that combines:

| Component | Purpose |
|-----------|---------|
| **Qwen2.5-VL-3B** | Visual reasoning about game state |
| **SpatialMemory** | Ice (curiosity) + Water (exploration) tracking |
| **GameControlAdapter** | Per-game control scheme adaptation |
| **StepAwareLoop** | Shows agent what changed after each action |
| **CausalTracker** | Action → effect learning |

### Key Innovation
The agent **SEES what happened** at each step via visual diff between frames, enabling it to:
1. Understand the effect of its last action
2. Adapt strategy based on observed changes
3. Handle different game control types automatically

---

## 2. Base Model Results — Problems to Solve

The original FLUX heuristic agent achieved **Score: 0** on all tested games:

### Test Results (Reference: `flux_arc_agi_3_live_test.ipynb`)

| Game | Control Type | API Tags | Result | Actions | Issue Identified |
|------|--------------|----------|--------|---------|------------------|
| `ls20` | MOVEMENT_ONLY [1,2,3,4] | `keyboard` | Score: 0 | 50 | Position stuck at (32,20), target at (25,52) — agent kept sending RIGHT but position never moved |
| `ft09` | CLICK_ONLY [6] | `click` | Score: 0 | 50 | Systematic clicks at row y=2 then y=3 — no understanding of game logic |
| `vc33` | CLICK_ONLY [6] | `click` | Score: 0 | 50 | Sequential clicks (0,0)→(1,0)→(2,0)... — no pattern recognition |

### Verified API Game Tags (25 games available)

| Tag | Control Scheme | Example Games |
|-----|---------------|---------------|
| `keyboard` | MOVEMENT_ONLY | ls20, wa30, tr87 |
| `click` | CLICK_ONLY | ft09, tn36, lp85 |
| `keyboard_click` | MOVE_AND_CLICK | sc25, tu93, re86, bp35, ka59 |

### Spatial Memory Visualizations

The base agent correctly **mapped** game structure but didn't **understand** it:

```
Game: ls20 | Final Position: (32, 20)
● ● ● ● ● ● ● ● ● ● ● ● ● ● (explored)
● ● ● ● ● ● ● ● ● ● ● ● 🧊🧊 (ice = unexplored targets at 52+ cols)

Game: ft09 | Final Position: (10, 12)
● ● ● ● ● ● ●     🧊🧊🧊🧊🧊 (clear puzzle structure in ice)
● ● ● @ ● ● ●     🧊🧊🧊🧊🧊 (ice shows clickable targets!)

Game: vc33 | Final Position: (1, 0)
● ● ● ● ● ● ● ● 🧊🧊🧊🧊🧊🧊🧊 (left half explored, right half = ice)
```

**Key Insight:** The spatial memory correctly identifies interesting areas (ice = curiosity) but the heuristic agent doesn't reason about WHY those areas matter.

---

## 3. ARC API Game Control Discovery

### 3.1 Control Types

Each ARC game has a specific control scheme visible from `available_actions`:

| Actions | Control Scheme | Examples |
|---------|---------------|----------|
| `[1,2,3,4]` | **Movement only** | Arrow keys — up/down/left/right grid movement |
| `[1,2,3,4,5]` | **Movement + Interact** | Move + SPACE to interact with objects |
| `[6]` | **Click only** | Mouse clicks at (x,y) coordinates |
| `[1,2,3,4,6]` | **Movement + Click** | Hybrid: move AND click |
| `[5]` | **Space only** | Single action toggle/step games |
| `[7]` | **Undo** | Can undo previous action |

### 3.2 Pre-Game Control Query

```python
import arc_agi

# Initialize arcade
arc = arc_agi.Arcade(arc_api_key="da685361-745a-4c31-8116-0f16ec5c342f")

# Get ALL available games and their info
all_games = arc.get_environments()

# Create a control registry BEFORE playing
GAME_CONTROLS = {}
for env_info in all_games:
    game_id = env_info.game_id[:4]  # e.g., "ls20"
    GAME_CONTROLS[game_id] = {
        'title': env_info.title,
        'tags': env_info.tags,
        'default_fps': env_info.default_fps,
    }

# When creating an environment, actions are in first frame
env = arc.make("ft09")
first_frame = env.observation_space
available = first_frame.available_actions  # e.g., [6] = click only

# Map to human-readable control scheme
def get_control_scheme(actions: list) -> str:
    """Determine game control type from available actions."""
    has_movement = any(a in [1,2,3,4] for a in actions)
    has_click = 6 in actions
    has_interact = 5 in actions
    has_undo = 7 in actions
    
    if has_movement and has_click:
        return "MOVE_AND_CLICK"
    elif has_movement and has_interact:
        return "MOVE_AND_INTERACT"
    elif has_movement:
        return "MOVEMENT_ONLY"
    elif has_click:
        return "CLICK_ONLY"
    elif has_interact:
        return "SPACE_ONLY"
    else:
        return "UNKNOWN"

# Example output
# ls20: MOVEMENT_ONLY [1,2,3,4]
# ft09: CLICK_ONLY [6]
# vc33: CLICK_ONLY [6]
```

### 3.3 Action Reference

```python
from arcengine import GameAction

ACTION_MAP = {
    1: GameAction.ACTION1,  # UP
    2: GameAction.ACTION2,  # DOWN
    3: GameAction.ACTION3,  # LEFT
    4: GameAction.ACTION4,  # RIGHT
    5: GameAction.ACTION5,  # INTERACT/SPACE
    6: GameAction.ACTION6,  # CLICK (requires {"x": int, "y": int})
    7: GameAction.ACTION7,  # UNDO
}

ACTION_NAMES = {
    1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
    5: "SPACE/INTERACT", 6: "CLICK", 7: "UNDO"
}

# Complex action example (click)
if action == 6:
    result = env.step(
        GameAction.ACTION6,
        data={"x": click_col, "y": click_row},
        reasoning={"thought": "clicking on cyan area"}
    )
```

---

## 4. Step-Aware Cognitive Loop

### 4.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    FLUX UNIFIED AGENT                            │
├─────────────────────────────────────────────────────────────────┤
│  1. PRE-GAME PHASE                                               │
│     ├── Query ARC API for game info (tags, title)                │
│     ├── Get available_actions from first frame                   │
│     └── Determine control_scheme (MOVEMENT/CLICK/HYBRID)         │
├─────────────────────────────────────────────────────────────────┤
│  2. EACH STEP (loop until WIN/GAME_OVER/max_actions)             │
│     ┌──────────────────────────────────────────────────────┐     │
│     │ A. OBSERVE: Get current frame                        │     │
│     │    ├── Render grid → image (with ice overlay)        │     │
│     │    └── Compare with previous frame (visual diff)     │     │
│     ├──────────────────────────────────────────────────────┤     │
│     │ B. REFLECT: Show agent what happened                 │     │
│     │    ├── "Last action: UP"                             │     │
│     │    ├── "Effect: Position moved (32,20)→(31,20)"      │     │
│     │    └── "Changes: 2 cells changed color"              │     │
│     ├──────────────────────────────────────────────────────┤     │
│     │ C. REASON (VLM): Qwen2.5-VL analyzes image           │     │
│     │    ├── Input: image + ice overlay + position marker  │     │
│     │    ├── Context: control_scheme + history + changes   │     │
│     │    └── Output: ACTION: [UP/DOWN/LEFT/RIGHT/CLICK]    │     │
│     ├──────────────────────────────────────────────────────┤     │
│     │ D. EXECUTE: Send action to ARC API                   │     │
│     │    ├── Movement: env.step(GameAction.ACTIONx)        │     │
│     │    └── Click: env.step(ACTION6, data={"x","y"})      │     │
│     ├──────────────────────────────────────────────────────┤     │
│     │ E. UPDATE: Learn from result                         │     │
│     │    ├── SpatialMemory.observe(position, grid)         │     │
│     │    ├── CausalTracker.record(action, changes)         │     │
│     │    └── Store frame for next step's diff              │     │
│     └──────────────────────────────────────────────────────┘     │
├─────────────────────────────────────────────────────────────────┤
│  3. POST-GAME: Save causal links, rules, spatial snapshots       │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Step-Aware Prompt Template

```python
STEP_AWARE_PROMPT = """You are playing {game_id} ({control_scheme} controls).

GAME STATE:
- Grid: {rows} rows × {cols} cols
- My position: row {pos_r}, col {pos_c} (marked with white circle)
- Cyan areas = unexplored interesting spots
- Available actions: {available_actions}

LAST ACTION RESULT:
- Action taken: {last_action}
- Effect: {effect_description}
- Cells changed: {num_changes}
- Position change: {pos_change}

HISTORY (last 3):
{history_summary}

YOUR TASK:
1. Look at the grid image
2. Notice what changed from your last action
3. Identify patterns (matching colors, symmetry, paths)
4. Decide the BEST next action to make progress

For {control_scheme} games:
{control_instructions}

End with exactly: ACTION: {action_format}
"""

# Control-specific instructions
CONTROL_INSTRUCTIONS = {
    "MOVEMENT_ONLY": "Move toward unexplored cyan areas. Find paths through obstacles.",
    "CLICK_ONLY": "Click on cells to change their state. Look for matching patterns.",
    "MOVE_AND_INTERACT": "Navigate to objects and press SPACE to interact.",
    "MOVE_AND_CLICK": "Move to position, then click specific targets.",
    "SPACE_ONLY": "Press SPACE to advance. Watch for timing patterns.",
}

# Action format by control type
ACTION_FORMAT = {
    "MOVEMENT_ONLY": "[UP/DOWN/LEFT/RIGHT]",
    "CLICK_ONLY": "CLICK(row, col)  # e.g., CLICK(3, 5)",
    "MOVE_AND_INTERACT": "[UP/DOWN/LEFT/RIGHT/SPACE]",
    "MOVE_AND_CLICK": "[UP/DOWN/LEFT/RIGHT] or CLICK(row, col)",
    "SPACE_ONLY": "SPACE",
}
```

---

## 5. Visual Diff System

The agent must **see what happened** after each action:

### 5.1 Frame Comparison

```python
class FrameDiffer:
    """Compute and describe changes between frames."""
    
    def __init__(self):
        self.prev_frame = None
        self.prev_position = None
    
    def diff(
        self, 
        new_frame: List[List[int]], 
        new_position: Tuple[int, int],
        last_action: int,
    ) -> Dict:
        """Compare new frame to previous, return change summary."""
        
        if self.prev_frame is None:
            self.prev_frame = new_frame
            self.prev_position = new_position
            return {"first_frame": True}
        
        # Position change
        pos_change = "none"
        if new_position != self.prev_position:
            dr = new_position[0] - self.prev_position[0]
            dc = new_position[1] - self.prev_position[1]
            pos_change = f"({self.prev_position[0]},{self.prev_position[1]}) → ({new_position[0]},{new_position[1]})"
        
        # Cell changes
        changes = []
        for r, (old_row, new_row) in enumerate(zip(self.prev_frame, new_frame)):
            for c, (old_val, new_val) in enumerate(zip(old_row, new_row)):
                if old_val != new_val:
                    changes.append({
                        "row": r, "col": c,
                        "old": old_val, "new": new_val,
                    })
        
        # Effect description
        if len(changes) == 0 and pos_change == "none":
            effect = "NO EFFECT - action had no visible result"
        elif len(changes) > 0 and pos_change != "none":
            effect = f"Position moved AND {len(changes)} cells changed"
        elif len(changes) > 0:
            effect = f"{len(changes)} cells changed colors"
        else:
            effect = f"Position moved to {new_position}"
        
        # Store for next comparison
        self.prev_frame = [row[:] for row in new_frame]  # Deep copy
        self.prev_position = new_position
        
        return {
            "effect": effect,
            "pos_change": pos_change,
            "num_changes": len(changes),
            "changes": changes[:10],  # First 10 for prompt
            "action_worked": len(changes) > 0 or pos_change != "none",
        }
```

### 5.2 Visual Diff Rendering

```python
def render_with_diff(
    grid: List[List[int]],
    position: Tuple[int, int],
    changes: List[Dict],
    ice_field: torch.Tensor,
    cell_size: int = 20,
) -> PIL.Image:
    """
    Render grid with:
    - Ice overlay (cyan for curiosity)
    - Position marker (white circle)
    - Change markers (yellow border on changed cells)
    """
    img = render_grid_to_image(grid, cell_size, position, ice_field)
    draw = ImageDraw.Draw(img)
    
    # Highlight changed cells with yellow border
    for change in changes:
        r, c = change["row"], change["col"]
        x0, y0 = c * cell_size, r * cell_size
        x1, y1 = x0 + cell_size, y0 + cell_size
        draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 0), width=2)
    
    return img
```

---

## 6. Control-Specific Strategy Modules

### 6.1 Movement Games (ls20 style)

```python
class MovementStrategy:
    """Strategy for UP/DOWN/LEFT/RIGHT only games."""
    
    def __init__(self, spatial_memory: SpatialMemory):
        self.memory = spatial_memory
    
    def suggest_action(
        self,
        grid: List[List[int]],
        position: Tuple[int, int],
        available: List[int],
    ) -> Tuple[int, str]:
        """Suggest movement toward highest curiosity."""
        
        r, c = position
        ice = self.memory.curiosity_field
        
        # Check curiosity in each direction
        directions = {
            1: (r-1, c, "UP"),    # UP
            2: (r+1, c, "DOWN"),  # DOWN
            3: (r, c-1, "LEFT"),  # LEFT
            4: (r, c+1, "RIGHT"), # RIGHT
        }
        
        best_action = None
        best_ice = -1
        
        for action, (nr, nc, name) in directions.items():
            if action not in available:
                continue
            if 0 <= nr < ice.shape[0] and 0 <= nc < ice.shape[1]:
                curiosity = ice[nr, nc].item()
                if curiosity > best_ice:
                    best_ice = curiosity
                    best_action = action
        
        if best_action:
            return best_action, f"Moving toward curiosity (ice={best_ice:.1f})"
        
        return available[0], "Random fallback"
```

### 6.2 Click Games (ft09, vc33 style)

```python
class ClickStrategy:
    """Strategy for CLICK only games."""
    
    def __init__(self, spatial_memory: SpatialMemory):
        self.memory = spatial_memory
        self.clicked = set()  # (row, col) already clicked
        self.click_effects = {}  # (row, col) → effect count
    
    def suggest_click(
        self,
        grid: List[List[int]],
        available: List[int],
    ) -> Tuple[int, int, str]:
        """Suggest click position based on curiosity and history."""
        
        if 6 not in available:
            return None, None, "Click not available"
        
        ice = self.memory.curiosity_field
        h, w = len(grid), len(grid[0])
        
        # Find highest curiosity unclicked cell
        best_pos = None
        best_ice = -1
        
        for r in range(min(h, ice.shape[0])):
            for c in range(min(w, ice.shape[1])):
                if (r, c) in self.clicked:
                    continue
                if grid[r][c] == 0:  # Skip background
                    continue
                curiosity = ice[r, c].item()
                if curiosity > best_ice:
                    best_ice = curiosity
                    best_pos = (r, c)
        
        if best_pos:
            self.clicked.add(best_pos)
            return best_pos[0], best_pos[1], f"Clicking curious cell (ice={best_ice:.1f})"
        
        # Fallback: click first non-background unclicked cell
        for r in range(h):
            for c in range(w):
                if (r, c) not in self.clicked and grid[r][c] != 0:
                    self.clicked.add((r, c))
                    return r, c, "Fallback: first non-background"
        
        return 0, 0, "No good click target"
    
    def record_effect(self, row: int, col: int, had_effect: bool):
        """Record whether a click had effect."""
        key = (row, col)
        if had_effect:
            self.click_effects[key] = self.click_effects.get(key, 0) + 1
```

---

## 7. Unified Agent Implementation

### 7.1 FLUXUnifiedAgent Class

```python
class FLUXUnifiedAgent:
    """
    Unified multi-modal ARC agent combining:
    - Qwen2.5-VL vision reasoning
    - SpatialMemory (ice/water)
    - Control-aware strategy
    - Step-aware cognitive loop
    """
    
    def __init__(
        self,
        model,           # Qwen2.5-VL model
        processor,       # Qwen2.5-VL processor
        device: str = "cuda",
        verbose: bool = True,
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.verbose = verbose
        
        # Core components
        self.spatial_memory = SpatialMemory(max_size=64, device=device)
        self.frame_differ = FrameDiffer()
        self.causal_tracker = CausalTracker()
        
        # Strategy modules (selected per-game)
        self.movement_strategy = MovementStrategy(self.spatial_memory)
        self.click_strategy = ClickStrategy(self.spatial_memory)
        
        # State
        self.game_id = None
        self.control_scheme = None
        self.step_count = 0
        self.history = []
    
    def start_game(self, game_id: str, first_frame, available_actions: List[int]):
        """Initialize for new game."""
        self.game_id = game_id
        self.control_scheme = self._detect_control_scheme(available_actions)
        self.step_count = 0
        self.history = []
        
        # Reset components
        self.spatial_memory.reset()
        self.frame_differ = FrameDiffer()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Starting: {game_id}")
            print(f"Control scheme: {self.control_scheme}")
            print(f"Available actions: {available_actions}")
            print(f"{'='*60}\n")
    
    def _detect_control_scheme(self, actions: List[int]) -> str:
        has_movement = any(a in [1,2,3,4] for a in actions)
        has_click = 6 in actions
        has_interact = 5 in actions
        
        if has_movement and has_click:
            return "MOVE_AND_CLICK"
        elif has_movement and has_interact:
            return "MOVE_AND_INTERACT"
        elif has_movement:
            return "MOVEMENT_ONLY"
        elif has_click:
            return "CLICK_ONLY"
        elif has_interact:
            return "SPACE_ONLY"
        return "UNKNOWN"
    
    def step(
        self,
        frame: List[List[int]],
        available_actions: List[int],
        last_action: int = None,
    ) -> Tuple[int, Dict, str]:
        """
        Execute one step of the cognitive loop.
        
        Returns:
            action: int - action ID to take
            action_data: dict - {"x": col, "y": row} for clicks, {} otherwise
            reasoning: str - explanation of decision
        """
        self.step_count += 1
        
        # A. OBSERVE: Get position and grid
        grid = self._normalize_frame(frame)
        position = self._find_position(grid)
        
        # B. REFLECT: Compare to previous frame
        diff = self.frame_differ.diff(grid, position, last_action)
        
        # C. UPDATE SPATIAL MEMORY
        self.spatial_memory.observe(position, grid)
        ice = self.spatial_memory.curiosity_field
        
        # D. RENDER IMAGE WITH OVERLAYS
        changes = diff.get("changes", [])
        game_image = render_with_diff(grid, position, changes, ice)
        
        # E. REASON WITH VLM
        action, reasoning = self._vlm_reason(
            grid=grid,
            position=position,
            image=game_image,
            available=available_actions,
            diff=diff,
        )
        
        # F. PREPARE ACTION DATA
        action_data = {}
        if action == 6:  # Click
            # Get click coordinates from reasoning or strategy
            click_r, click_c = self._get_click_target(grid, reasoning)
            action_data = {"x": click_c, "y": click_r}
        
        # G. RECORD HISTORY
        self.history.append({
            "step": self.step_count,
            "action": action,
            "position": position,
            "effect": diff.get("effect", "unknown"),
        })
        
        # H. LOG
        if self.verbose:
            print(f"Step {self.step_count}:")
            print(f"  Position: {position}")
            print(f"  Last effect: {diff.get('effect', 'first')}")
            print(f"  Action: {ACTION_NAMES.get(action, action)}")
            if action_data:
                print(f"  Click at: ({action_data.get('y')}, {action_data.get('x')})")
        
        return action, action_data, reasoning
    
    def _vlm_reason(
        self,
        grid: List[List[int]],
        position: Tuple[int, int],
        image: PIL.Image,
        available: List[int],
        diff: Dict,
    ) -> Tuple[int, str]:
        """Use Qwen2.5-VL to reason about action."""
        
        # Build step-aware prompt
        prompt = self._build_prompt(grid, position, available, diff)
        
        if self.model is None:
            # Fallback to strategy-based decision
            return self._strategy_fallback(grid, position, available)
        
        try:
            # Qwen2.5-VL inference
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.3,
                    do_sample=True,
                )
            
            generated_ids = outputs[:, inputs.input_ids.shape[1]:]
            response = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            
            action = self._parse_response(response, available)
            return action, response
            
        except Exception as e:
            print(f"  VLM error: {e}")
            return self._strategy_fallback(grid, position, available)
    
    def _build_prompt(
        self,
        grid: List[List[int]],
        position: Tuple[int, int],
        available: List[int],
        diff: Dict,
    ) -> str:
        """Build step-aware prompt."""
        
        available_names = [ACTION_NAMES.get(a, f"ACTION{a}") for a in available]
        
        # History summary
        history_text = ""
        if self.history:
            recent = self.history[-3:]
            history_text = "\n".join(
                f"  Step {h['step']}: {ACTION_NAMES.get(h['action'])} → {h['effect']}"
                for h in recent
            )
        
        return f"""You are playing {self.game_id} ({self.control_scheme} controls).

GAME STATE:
- Grid: {len(grid)} rows × {len(grid[0])} cols
- My position: row {position[0]}, col {position[1]} (white circle)
- Cyan areas = high curiosity (unexplored interesting spots)
- Available: {', '.join(available_names)}

LAST ACTION RESULT:
- Effect: {diff.get('effect', 'first frame')}
- Position change: {diff.get('pos_change', 'none')}
- Cells changed: {diff.get('num_changes', 0)}

RECENT HISTORY:
{history_text if history_text else '  (first move)'}

{CONTROL_INSTRUCTIONS.get(self.control_scheme, '')}

Analyze the image, notice patterns, and choose the BEST action.
End with: ACTION: {ACTION_FORMAT.get(self.control_scheme, '[action]')}"""
    
    def _strategy_fallback(
        self,
        grid: List[List[int]],
        position: Tuple[int, int],
        available: List[int],
    ) -> Tuple[int, str]:
        """Use strategy module when VLM unavailable."""
        
        if self.control_scheme == "MOVEMENT_ONLY":
            return self.movement_strategy.suggest_action(grid, position, available)
        elif self.control_scheme == "CLICK_ONLY":
            r, c, reason = self.click_strategy.suggest_click(grid, available)
            return 6, f"CLICK({r},{c}): {reason}"
        else:
            return available[0] if available else 1, "Random fallback"
```

---

## 8. Integration with ARC API

### 8.1 Game Loop

```python
import arc_agi
from arcengine import GameAction, GameState

def play_game_unified(
    agent: FLUXUnifiedAgent,
    game_id: str,
    api_key: str,
    max_actions: int = 100,
) -> Dict:
    """Play an ARC game with the unified agent."""
    
    # Initialize
    arc = arc_agi.Arcade(arc_api_key=api_key)
    env = arc.make(game_id)
    
    if env is None:
        return {"error": f"Failed to create {game_id}"}
    
    # Get first frame
    first_frame = env.observation_space
    frame = first_frame.frame
    available = [a.value for a in first_frame.available_actions]
    
    # Start agent
    agent.start_game(game_id, frame, available)
    
    last_action = None
    
    for step in range(max_actions):
        # Get current state
        obs = env.observation_space
        frame = obs.frame
        state = obs.state
        available = [a.value for a in obs.available_actions]
        
        # Check win/loss
        if state == GameState.WIN:
            print(f"✓ WON at step {step}!")
            break
        elif state == GameState.GAME_OVER:
            print(f"✗ Game over at step {step}")
            env.reset()
            continue
        
        # Agent decides action
        action, action_data, reasoning = agent.step(
            frame=frame,
            available_actions=available,
            last_action=last_action,
        )
        
        # Execute action
        game_action = ACTION_MAP.get(action)
        if game_action is None:
            continue
        
        result = env.step(
            game_action,
            data=action_data if action_data else None,
            reasoning={"thought": reasoning[:100]},
        )
        
        last_action = action
    
    # Return results
    scorecard = arc.get_scorecard()
    return {
        "game_id": game_id,
        "steps": step + 1,
        "score": scorecard.score if scorecard else 0,
        "history": agent.history,
    }
```

---

## 9. Files Reference

| File | Purpose |
|------|---------|
| `notebooks/flux_arc_agi_3_live_test.ipynb` | Original heuristic agent (Score 0 results) |
| `notebooks/phase12_kaggle.ipynb` | Multi-modal agent with Qwen2.5-VL |
| `phases/phase9_arc/spatial_memory.py` | SpatialMemory (ice/water) implementation |
| `phases/phase_unified/cognitive_layer.py` | CausalTracker, RuleInducer, GoalPlanner |
| `arc-agi-training/arc_agi/` | ARC API client library |

---

## 10. Success Criteria

| Metric | Base Agent | Target |
|--------|------------|--------|
| ls20 Score | 0 | >0.3 |
| ft09 Score | 0 | >0.3 |
| vc33 Score | 0 | >0.3 |
| Average Score | 0 | >0.25 |
| Steps to first win | ∞ | <30 |

---

## 11. Next Steps

1. **Run unified agent on ls20, ft09, vc33** — compare to base results
2. **Tune VLM prompts** — make reasoning more action-focused
3. **Add causal learning** — remember what actions have effects
4. **Test on more games** — expand to full ARC game set
5. **Kaggle submission** — run in competition mode

---

*Specification authored for FLUX Project — 2026-03-29*
