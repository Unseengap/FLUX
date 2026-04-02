"""
FLUX Unified Agent — Multi-Modal ARC Agent with Step-Aware Cognitive Loop

Combines:
- Qwen2.5-VL (or similar VLM) for visual reasoning
- SpatialMemory (ice/water fields) for navigation
- FrameDiffer for seeing what changed
- Control-specific strategies (movement/click)
- CausalTracker for action→effect learning

The agent SEES what happened at each step via visual diff between frames.

Physics Analogy:
    The agent is a conscious observer of the field, updating its beliefs
    based on observed effects and navigating toward curiosity gradients.
"""

from __future__ import annotations
import torch
from torch import Tensor
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import sys
from pathlib import Path

# Local imports
try:
    from phases.phase_unified.frame_differ import FrameDiffer, FrameDiff, ACTION_NAMES
    from phases.phase_unified.strategies import (
        MovementStrategy,
        ClickStrategy, 
        HybridStrategy,
        get_control_scheme,
        CONTROL_INSTRUCTIONS,
        ACTION_FORMAT,
    )
    from phases.phase_unified.rendering import render_with_diff, render_grid_ascii
except ImportError:
    try:
        from .frame_differ import FrameDiffer, FrameDiff, ACTION_NAMES
        from .strategies import (
            MovementStrategy,
            ClickStrategy, 
            HybridStrategy,
            get_control_scheme,
            CONTROL_INSTRUCTIONS,
            ACTION_FORMAT,
        )
        from .rendering import render_with_diff, render_grid_ascii
    except ImportError:
        from frame_differ import FrameDiffer, FrameDiff, ACTION_NAMES
        from strategies import (
            MovementStrategy,
            ClickStrategy,
            HybridStrategy,
            get_control_scheme,
            CONTROL_INSTRUCTIONS,
            ACTION_FORMAT,
        )
        from rendering import render_with_diff, render_grid_ascii


# ─────────────────────────────────────────────
# History Entry
# ─────────────────────────────────────────────

@dataclass
class StepHistory:
    """Record of one step in the game."""
    step: int
    action: int
    action_name: str
    position: Tuple[int, int]
    effect: str
    action_worked: bool
    click_pos: Optional[Tuple[int, int]] = None


# ─────────────────────────────────────────────
# Prompt Builder
# ─────────────────────────────────────────────

class PromptBuilder:
    """Builds step-aware prompts for the VLM."""
    
    @staticmethod
    def build_step_prompt(
        game_id: str,
        control_scheme: str,
        grid: List[List[int]],
        position: Tuple[int, int],
        available: List[int],
        diff: FrameDiff,
        history: List[StepHistory],
    ) -> str:
        """Build a step-aware prompt for VLM reasoning."""
        grid_h = len(grid)
        grid_w = len(grid[0]) if grid else 0
        
        # Available action names
        available_names = [ACTION_NAMES.get(a, f"ACTION{a}") for a in available]
        
        # History summary (last 3)
        history_text = ""
        if history:
            recent = history[-3:]
            history_lines = []
            for h in recent:
                status = "✓" if h.action_worked else "✗"
                history_lines.append(f"  Step {h.step}: {h.action_name} → {h.effect} [{status}]")
            history_text = "\n".join(history_lines)
        else:
            history_text = "  (first move)"
        
        # Control instructions
        ctrl_inst = CONTROL_INSTRUCTIONS.get(control_scheme, "")
        action_fmt = ACTION_FORMAT.get(control_scheme, "[action]")
        
        prompt = f"""You are playing {game_id} ({control_scheme} controls).

GAME STATE:
- Grid: {grid_h} rows × {grid_w} cols
- My position: row {position[0]}, col {position[1]} (white circle in image)
- Cyan areas = high curiosity (unexplored interesting spots)
- Available: {', '.join(available_names)}

LAST ACTION RESULT:
- Effect: {diff.effect if not diff.first_frame else 'first frame'}
- Position change: {diff.position_delta if diff.position_changed else 'none'}
- Cells changed: {diff.num_changes}

RECENT HISTORY:
{history_text}

{ctrl_inst}

Analyze the image, notice patterns: matching colors, symmetry, paths to explore.
Choose the BEST action to make progress.

End with exactly: ACTION: {action_fmt}"""
        
        return prompt


# ─────────────────────────────────────────────
# FLUX Unified Agent
# ─────────────────────────────────────────────

class FLUXUnifiedAgent:
    """
    Unified multi-modal ARC agent combining:
    - VLM vision reasoning (Qwen2.5-VL or similar)
    - SpatialMemory (ice/water navigation)
    - Control-aware strategy
    - Step-aware cognitive loop
    """
    
    def __init__(
        self,
        model=None,
        processor=None,
        spatial_memory=None,
        causal_tracker=None,
        device: str = "cpu",
        verbose: bool = True,
    ):
        """
        Initialize unified agent.
        
        Args:
            model: VLM model (Qwen2.5-VL or similar) - optional
            processor: VLM processor - optional
            spatial_memory: SpatialMemory instance - optional
            causal_tracker: CausalTracker instance - optional
            device: Device for tensors
            verbose: Print debug info
        """
        self.model = model
        self.processor = processor
        self.spatial_memory = spatial_memory
        self.causal_tracker = causal_tracker
        self.device = device
        self.verbose = verbose
        
        # Core components
        self.frame_differ = FrameDiffer()
        
        # Strategy modules (created per-game based on control scheme)
        self.movement_strategy: Optional[MovementStrategy] = None
        self.click_strategy: Optional[ClickStrategy] = None
        self.hybrid_strategy: Optional[HybridStrategy] = None
        
        # State
        self.game_id: Optional[str] = None
        self.control_scheme: Optional[str] = None
        self.step_count: int = 0
        self.history: List[StepHistory] = []
        self.last_grid: Optional[List[List[int]]] = None
        self.current_position: Tuple[int, int] = (0, 0)
        self._detected_agent_pos: Optional[Tuple[int, int]] = None  # Movement-tracked position
        
        # VLM state
        self.vlm_available = model is not None and processor is not None
        
        # Tracking
        self.wins = 0
        self.total_actions = 0
    
    def log(self, msg: str):
        """Log message if verbose."""
        if self.verbose:
            print(f"  [Unified] {msg}")
    
    # ─────────────────────────────────────────────
    # Game Lifecycle
    # ─────────────────────────────────────────────
    
    def start_game(
        self,
        game_id: str,
        first_frame: List[List[int]],
        available_actions: List[int],
    ):
        """
        Initialize for a new game.
        
        Args:
            game_id: Game identifier (e.g., "ls20")
            first_frame: Initial grid state
            available_actions: List of available action IDs
        """
        self.game_id = game_id
        self.control_scheme = get_control_scheme(available_actions)
        self.step_count = 0
        self.history = []
        self.last_grid = None
        self._detected_agent_pos = None  # Reset movement tracking
        
        # Reset components
        self.frame_differ.reset()
        
        if self.spatial_memory is not None:
            self.spatial_memory.reset()
        
        # Create appropriate strategy
        self._setup_strategy()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Starting: {game_id}")
            print(f"Control scheme: {self.control_scheme}")
            print(f"Available actions: {available_actions}")
            print(f"VLM available: {self.vlm_available}")
            print(f"{'='*60}\n")
    
    def _setup_strategy(self):
        """Setup appropriate strategy for control scheme."""
        sm = self.spatial_memory
        
        if self.control_scheme == "MOVEMENT_ONLY":
            self.movement_strategy = MovementStrategy(sm)
        elif self.control_scheme == "CLICK_ONLY":
            self.click_strategy = ClickStrategy(sm)
        elif self.control_scheme in ["MOVE_AND_CLICK", "MOVE_AND_INTERACT"]:
            self.hybrid_strategy = HybridStrategy(sm)
            self.movement_strategy = self.hybrid_strategy.movement
            self.click_strategy = self.hybrid_strategy.click
        else:
            # Fallback
            self.movement_strategy = MovementStrategy(sm)
            self.click_strategy = ClickStrategy(sm)
    
    # ─────────────────────────────────────────────
    # Core Step Loop
    # ─────────────────────────────────────────────
    
    def step(
        self,
        frame: List[List[int]],
        available_actions: List[int],
        last_action: Optional[int] = None,
    ) -> Tuple[int, Dict[str, Any], str]:
        """
        Execute one step of the cognitive loop.
        
        Args:
            frame: Current grid state
            available_actions: Available action IDs
            last_action: Action taken to get here
            
        Returns:
            (action_id, action_data, reasoning)
            action_data is {"x": col, "y": row} for clicks, {} otherwise
        """
        self.step_count += 1
        
        # A. OBSERVE: Normalize grid and find position
        grid = self._normalize_frame(frame)
        position = self._find_position(grid)
        self.current_position = position
        
        grid_h = len(grid)
        grid_w = len(grid[0]) if grid else 0
        
        # B. REFLECT: Compare to previous frame
        diff = self.frame_differ.diff(grid, position, last_action)
        
        # DEBUG: Log if we detected changes
        if self.verbose and diff.num_changes > 0:
            self.log(f"Detected {diff.num_changes} changes: {[(c.row, c.col, c.old_value, c.new_value) for c in diff.changes[:5]]}")
        
        # B2. DETECT AGENT MOVEMENT: Track where agent actually moved
        if last_action is not None and last_action in [1, 2, 3, 4] and self.last_grid is not None:
            self._detect_agent_by_movement(self.last_grid, grid, last_action)
            # Re-find position using updated movement tracking
            position = self._find_position(grid)
            self.current_position = position
        
        # C. RECORD EFFECT OF LAST ACTION
        if last_action is not None and self.history:
            last_entry = self.history[-1]
            # Update causal tracker
            if self.causal_tracker is not None and self.last_grid is not None:
                try:
                    import numpy as np
                    self.causal_tracker.record(
                        position=last_entry.position,
                        action=last_action,
                        grid_before=np.array(self.last_grid),
                        grid_after=np.array(grid),
                    )
                except Exception as e:
                    self.log(f"Causal tracking error: {e}")
            
            # Update click strategy if was a click
            if last_action == 6 and self.click_strategy is not None:
                if last_entry.click_pos is not None:
                    r, c = last_entry.click_pos
                    self.click_strategy.record_effect(r, c, diff.action_worked)
        
        # D. UPDATE SPATIAL MEMORY
        ice_field = None
        if self.spatial_memory is not None:
            try:
                clamped = self._clamp_grid(grid, max_val=9)
                self.spatial_memory.observe(
                    position=position,
                    local_view=clamped,
                    global_grid=clamped,
                )
                
                # Add curiosity to change locations
                for change in diff.changes:
                    r, c = change.row, change.col
                    if r < self.spatial_memory.max_size and c < self.spatial_memory.max_size:
                        self.spatial_memory.curiosity_field[r, c] += 5.0
                
                ice_field = self.spatial_memory.curiosity_field
                self.spatial_memory.step_decay()
            except Exception as e:
                self.log(f"Spatial memory error: {e}")
        
        # E. DECIDE ACTION
        action, action_data, reasoning = self._decide_action(
            grid=grid,
            position=position,
            available=available_actions,
            diff=diff,
            ice_field=ice_field,
        )
        
        # F. RECORD HISTORY
        click_pos = None
        if action == 6 and action_data:
            click_pos = (action_data.get("y", 0), action_data.get("x", 0))
        
        self.history.append(StepHistory(
            step=self.step_count,
            action=action,
            action_name=ACTION_NAMES.get(action, f"ACTION{action}"),
            position=position,
            effect=diff.effect,
            action_worked=diff.action_worked,
            click_pos=click_pos,
        ))
        
        # G. STORE FOR NEXT STEP
        self.last_grid = [row[:] for row in grid]
        self.total_actions += 1
        
        # H. LOG
        if self.verbose:
            print(f"Step {self.step_count}:")
            print(f"  Position: {position}")
            print(f"  Last effect: {diff.effect if not diff.first_frame else 'first'}")
            print(f"  Action: {ACTION_NAMES.get(action, action)}")
            if action_data:
                print(f"  Click at: ({action_data.get('y')}, {action_data.get('x')})")
        
        return action, action_data, reasoning
    
    # ─────────────────────────────────────────────
    # Decision Making
    # ─────────────────────────────────────────────
    
    def _decide_action(
        self,
        grid: List[List[int]],
        position: Tuple[int, int],
        available: List[int],
        diff: FrameDiff,
        ice_field: Optional[Tensor],
    ) -> Tuple[int, Dict[str, Any], str]:
        """
        Decide next action.
        
        Uses VLM if available, otherwise falls back to strategy.
        """
        # Try VLM reasoning first
        if self.vlm_available:
            try:
                action, data, reason = self._vlm_reason(
                    grid, position, available, diff, ice_field
                )
                if action is not None:
                    return action, data, reason
            except Exception as e:
                self.log(f"VLM error: {e}")
        
        # Fallback to strategy-based decision
        return self._strategy_decide(grid, position, available, ice_field)
    
    def _strategy_decide(
        self,
        grid: List[List[int]],
        position: Tuple[int, int],
        available: List[int],
        ice_field: Optional[Tensor],
    ) -> Tuple[int, Dict[str, Any], str]:
        """Strategy-based action decision."""
        
        # Use appropriate strategy based on control scheme
        if self.control_scheme == "MOVEMENT_ONLY" and self.movement_strategy:
            action, reason = self.movement_strategy.suggest_action(
                grid, position, available, ice_field
            )
            return action, {}, reason
        
        elif self.control_scheme == "CLICK_ONLY" and self.click_strategy:
            r, c, reason = self.click_strategy.suggest_click(grid, available, ice_field)
            return 6, {"x": c, "y": r}, reason
        
        elif self.hybrid_strategy:
            action, data, reason = self.hybrid_strategy.suggest_action(
                grid, position, available, ice_field
            )
            return action, data, reason
        
        # Ultimate fallback
        if available:
            return available[0], {}, "Random fallback"
        return 1, {}, "No actions available"
    
    def _vlm_reason(
        self,
        grid: List[List[int]],
        position: Tuple[int, int],
        available: List[int],
        diff: FrameDiff,
        ice_field: Optional[Tensor],
    ) -> Tuple[Optional[int], Dict[str, Any], str]:
        """Use VLM to reason about action."""
        
        if not self.vlm_available:
            return None, {}, "VLM not available"
        
        try:
            # Render image for VLM
            from .rendering import render_with_diff
        except ImportError:
            from rendering import render_with_diff
        
        try:
            # Build image
            changes = [{"row": c.row, "col": c.col} for c in diff.changes]
            image = render_with_diff(grid, position, changes, ice_field)
            
            # Build prompt
            prompt = PromptBuilder.build_step_prompt(
                game_id=self.game_id or "unknown",
                control_scheme=self.control_scheme or "UNKNOWN",
                grid=grid,
                position=position,
                available=available,
                diff=diff,
                history=self.history,
            )
            
            # VLM inference
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
            
            # Parse response
            action, data = self._parse_vlm_response(response, available)
            return action, data, response
            
        except Exception as e:
            self.log(f"VLM inference error: {e}")
            return None, {}, f"VLM error: {e}"
    
    def _parse_vlm_response(
        self,
        response: str,
        available: List[int],
    ) -> Tuple[Optional[int], Dict[str, Any]]:
        """Parse VLM response to extract action."""
        response_upper = response.upper()
        
        # Look for "ACTION: ..." pattern
        if "ACTION:" in response_upper:
            action_part = response_upper.split("ACTION:")[-1].strip()
            
            # Check for movement
            if "UP" in action_part and 1 in available:
                return 1, {}
            if "DOWN" in action_part and 2 in available:
                return 2, {}
            if "LEFT" in action_part and 3 in available:
                return 3, {}
            if "RIGHT" in action_part and 4 in available:
                return 4, {}
            if "SPACE" in action_part and 5 in available:
                return 5, {}
            
            # Check for click CLICK(row, col)
            if "CLICK" in action_part and 6 in available:
                import re
                match = re.search(r'CLICK\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', action_part)
                if match:
                    row, col = int(match.group(1)), int(match.group(2))
                    return 6, {"x": col, "y": row}
                # Fallback click at center
                return 6, {"x": 5, "y": 5}
        
        # Fallback to any mentioned direction
        if "UP" in response_upper and 1 in available:
            return 1, {}
        if "DOWN" in response_upper and 2 in available:
            return 2, {}
        if "LEFT" in response_upper and 3 in available:
            return 3, {}
        if "RIGHT" in response_upper and 4 in available:
            return 4, {}
        
        return None, {}
    
    # ─────────────────────────────────────────────
    # Grid Utilities
    # ─────────────────────────────────────────────
    
    def _normalize_frame(self, frame) -> List[List[int]]:
        """Normalize ARC API frame to standard 2D grid."""
        if isinstance(frame, dict):
            if 'grid' in frame:
                frame = frame['grid']
            elif 'frame' in frame:
                frame = frame['frame']
        
        if not isinstance(frame, list):
            return [[0] * 10 for _ in range(10)]
        
        if len(frame) == 0:
            return [[0] * 10 for _ in range(10)]
        
        # Check for triple nesting
        if isinstance(frame[0], list) and len(frame[0]) > 0 and isinstance(frame[0][0], list):
            frame = frame[0]
        
        if len(frame) == 0:
            return [[0] * 10 for _ in range(10)]
        
        if isinstance(frame[0], list):
            if len(frame[0]) > 0 and isinstance(frame[0][0], int):
                return frame
            elif len(frame[0]) > 0 and isinstance(frame[0][0], list):
                return frame[0]
        
        if isinstance(frame[0], int):
            return [frame]
        
        return [[0] * 10 for _ in range(10)]
    
    def _clamp_grid(self, grid: List[List[int]], max_val: int = 9) -> List[List[int]]:
        """Clamp grid values to [0, max_val]."""
        return [
            [max(0, min(max_val, val)) if isinstance(val, int) else 0 for val in row]
            for row in grid
        ]
    
    def _find_position(self, grid: List[List[int]]) -> Tuple[int, int]:
        """
        Find agent position in grid.
        
        Uses multiple strategies:
        1. Movement-based detection (if we have previous grid)
        2. Look for common agent colors
        3. Find rarest non-zero color
        4. Default to grid center
        """
        if not grid or not grid[0]:
            return (0, 0)
        
        grid_h, grid_w = len(grid), len(grid[0])
        
        # Strategy 1: Track via movement (if we moved and grid changed)
        if hasattr(self, '_detected_agent_pos') and self._detected_agent_pos is not None:
            return self._detected_agent_pos
        
        # Strategy 2: Look for common agent colors (often color 1, 2, or 3)
        agent_colors = [1, 2, 3, 8, 9]
        for color in agent_colors:
            positions = []
            for r, row in enumerate(grid):
                for c, val in enumerate(row):
                    if isinstance(val, int) and val == color:
                        positions.append((r, c))
            # If there's exactly one cell of this color, it's likely the agent
            if len(positions) == 1:
                return positions[0]
        
        # Strategy 3: Find rarest non-zero color (likely the agent marker)
        color_counts: Dict[int, int] = {}
        color_positions: Dict[int, List[Tuple[int, int]]] = {}
        for r, row in enumerate(grid):
            for c, val in enumerate(row):
                if isinstance(val, int) and val > 0:
                    color_counts[val] = color_counts.get(val, 0) + 1
                    if val not in color_positions:
                        color_positions[val] = []
                    color_positions[val].append((r, c))
        
        if color_counts:
            # Find rarest color with only a few cells (likely agent)
            rarest = min(
                (c for c in color_counts if color_counts[c] <= 5),
                key=lambda x: color_counts[x],
                default=None,
            )
            if rarest is not None and color_positions.get(rarest):
                # Return the first position of the rarest color
                return color_positions[rarest][0]
        
        # Default: center
        return (grid_h // 2, grid_w // 2)
    
    def _detect_agent_by_movement(
        self,
        old_grid: List[List[int]],
        new_grid: List[List[int]],
        action: int,
    ):
        """
        Detect agent position by observing what changed after a movement action.
        
        When UP/DOWN/LEFT/RIGHT is pressed, the agent moves. Look for:
        - A cell that disappeared (old position)
        - A cell that appeared nearby (new position)
        """
        if action not in [1, 2, 3, 4]:  # Only movement actions
            return
        
        if old_grid is None or new_grid is None:
            return
        
        if not old_grid or not new_grid:
            return
        
        # Find cells that changed
        appeared = []  # Cells that are now non-zero but were zero
        disappeared = []  # Cells that are now zero but were non-zero
        
        old_h, old_w = len(old_grid), len(old_grid[0]) if old_grid else 0
        new_h, new_w = len(new_grid), len(new_grid[0]) if new_grid else 0
        
        for r in range(min(old_h, new_h)):
            old_row = old_grid[r] if r < old_h else []
            new_row = new_grid[r] if r < new_h else []
            
            for c in range(min(len(old_row), len(new_row))):
                old_val = old_row[c] if isinstance(old_row[c], int) else 0
                new_val = new_row[c] if isinstance(new_row[c], int) else 0
                
                if old_val == 0 and new_val > 0:
                    appeared.append((r, c, new_val))
                elif old_val > 0 and new_val == 0:
                    disappeared.append((r, c, old_val))
        
        # If we have both appeared and disappeared cells, track the movement
        if appeared and disappeared:
            # Find cells that match the movement direction
            dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(action, (0, 0))
            
            for old_r, old_c, old_color in disappeared:
                expected_new = (old_r + dr, old_c + dc)
                for new_r, new_c, new_color in appeared:
                    # Check if the new cell is where we expected based on movement
                    if (new_r, new_c) == expected_new or (new_r == old_r + dr and new_c == old_c + dc):
                        self._detected_agent_pos = (new_r, new_c)
                        if self.verbose:
                            self.log(f"Detected agent moved: ({old_r},{old_c}) -> ({new_r},{new_c})")
                        return
            
            # Fallback: just use the first appeared cell
            if appeared:
                self._detected_agent_pos = (appeared[0][0], appeared[0][1])
                if self.verbose:
                    self.log(f"Agent moved to: {self._detected_agent_pos}")
    
    # ─────────────────────────────────────────────
    # Statistics
    # ─────────────────────────────────────────────
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        effective_actions = sum(1 for h in self.history if h.action_worked)
        
        return {
            'game_id': self.game_id,
            'control_scheme': self.control_scheme,
            'steps': self.step_count,
            'total_actions': self.total_actions,
            'effective_actions': effective_actions,
            'effectiveness_rate': effective_actions / max(1, len(self.history)),
            'wins': self.wins,
            'vlm_available': self.vlm_available,
            'current_position': self.current_position,
        }
    
    def reset(self):
        """Reset agent for new session."""
        self.frame_differ.reset()
        if self.movement_strategy:
            self.movement_strategy.reset()
        if self.click_strategy:
            self.click_strategy.reset()
        if self.hybrid_strategy:
            self.hybrid_strategy.reset()
        if self.spatial_memory:
            self.spatial_memory.reset()
        
        self.game_id = None
        self.control_scheme = None
        self.step_count = 0
        self.history = []
        self.last_grid = None
        self.current_position = (0, 0)
        self._detected_agent_pos = None  # Reset movement tracking


# ─────────────────────────────────────────────
# Factory Function
# ─────────────────────────────────────────────

def create_unified_agent(
    spatial_memory=None,
    causal_tracker=None,
    model=None,
    processor=None,
    device: str = "cpu",
    verbose: bool = True,
) -> FLUXUnifiedAgent:
    """
    Create a FLUXUnifiedAgent with optional components.
    
    Args:
        spatial_memory: SpatialMemory instance
        causal_tracker: CausalTracker instance
        model: VLM model (Qwen2.5-VL)
        processor: VLM processor
        device: Device for tensors
        verbose: Print debug info
        
    Returns:
        Configured FLUXUnifiedAgent
    """
    return FLUXUnifiedAgent(
        model=model,
        processor=processor,
        spatial_memory=spatial_memory,
        causal_tracker=causal_tracker,
        device=device,
        verbose=verbose,
    )
