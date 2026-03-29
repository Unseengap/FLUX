"""
Strategies — Control-Specific Action Strategies for FLUX Unified Agent

Provides specialized strategies for different ARC game control schemes:
- MovementStrategy: UP/DOWN/LEFT/RIGHT navigation (e.g., ls20)
- ClickStrategy: Mouse click at (x,y) coordinates (e.g., ft09, vc33)
- HybridStrategy: Movement + Click combined

Physics Analogy:
    Each strategy is a different force field that guides the agent.
    Movement follows curiosity gradients; clicks target high-energy cells.
"""

from __future__ import annotations
import torch
from torch import Tensor
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import random


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

# Movement directions
DIRECTION_DELTAS = {
    1: (-1, 0),  # UP
    2: (1, 0),   # DOWN
    3: (0, -1),  # LEFT
    4: (0, 1),   # RIGHT
}

ACTION_NAMES = {
    1: "UP",
    2: "DOWN",
    3: "LEFT", 
    4: "RIGHT",
    5: "SPACE",
    6: "CLICK",
    7: "UNDO",
}


# ─────────────────────────────────────────────
# Click Target Data
# ─────────────────────────────────────────────

@dataclass
class ClickTarget:
    """A potential click target with reasoning."""
    row: int
    col: int
    score: float  # Higher = more likely to click
    reason: str


# ─────────────────────────────────────────────
# Movement Strategy (ls20 style)
# ─────────────────────────────────────────────

class MovementStrategy:
    """
    Strategy for UP/DOWN/LEFT/RIGHT only games.
    
    Uses SpatialMemory's curiosity field (ice) to navigate toward
    unexplored interesting areas.
    """
    
    def __init__(self, spatial_memory=None):
        """
        Initialize movement strategy.
        
        Args:
            spatial_memory: SpatialMemory instance with curiosity_field
        """
        self.memory = spatial_memory
        self.movement_history: List[int] = []
        self.stuck_count = 0
        self.last_position: Optional[Tuple[int, int]] = None
    
    def reset(self):
        """Reset for new game."""
        self.movement_history = []
        self.stuck_count = 0
        self.last_position = None
    
    def suggest_action(
        self,
        grid: List[List[int]],
        position: Tuple[int, int],
        available: List[int],
        ice_field: Optional[Tensor] = None,
    ) -> Tuple[int, str]:
        """
        Suggest movement toward highest curiosity.
        
        Args:
            grid: Current game grid
            position: Current (row, col) position
            available: List of available action IDs (subset of [1,2,3,4])
            ice_field: Optional curiosity field tensor
            
        Returns:
            (action_id, reason_string)
        """
        r, c = position
        grid_h = len(grid)
        grid_w = len(grid[0]) if grid else 0
        
        # Check if stuck
        if self.last_position == position:
            self.stuck_count += 1
        else:
            self.stuck_count = 0
        self.last_position = position
        
        # Get curiosity field
        if ice_field is None and self.memory is not None:
            ice_field = self.memory.curiosity_field
        
        # Filter to movement actions only
        move_actions = [a for a in available if a in [1, 2, 3, 4]]
        if not move_actions:
            return available[0] if available else 1, "No movement available"
        
        # If stuck, try a different direction
        if self.stuck_count >= 3:
            # Avoid last direction
            last_dir = self.movement_history[-1] if self.movement_history else None
            other_dirs = [a for a in move_actions if a != last_dir]
            if other_dirs:
                action = random.choice(other_dirs)
                self.movement_history.append(action)
                return action, f"Unsticking: trying {ACTION_NAMES.get(action)}"
        
        # Score each direction by curiosity
        best_action = None
        best_score = -float('inf')
        best_reason = ""
        
        for action in move_actions:
            dr, dc = DIRECTION_DELTAS.get(action, (0, 0))
            nr, nc = r + dr, c + dc
            
            # Check bounds
            if nr < 0 or nr >= grid_h or nc < 0 or nc >= grid_w:
                continue
            
            # Score by curiosity ice
            score = 0.0
            if ice_field is not None:
                if 0 <= nr < ice_field.shape[0] and 0 <= nc < ice_field.shape[1]:
                    score = ice_field[nr, nc].item()
            
            # Bonus for unexplored cells
            if ice_field is not None and score > 5.0:
                score += 10.0  # High curiosity bonus
            
            # Penalty for recent directions (encourage exploration)
            recent_count = self.movement_history[-5:].count(action)
            score -= recent_count * 2.0
            
            if score > best_score:
                best_score = score
                best_action = action
                best_reason = f"Moving {ACTION_NAMES.get(action)} toward curiosity (ice={score:.1f})"
        
        if best_action is not None:
            self.movement_history.append(best_action)
            return best_action, best_reason
        
        # Random fallback
        action = random.choice(move_actions)
        self.movement_history.append(action)
        return action, "Random exploration"
    
    def get_movement_toward(
        self,
        position: Tuple[int, int],
        target: Tuple[int, int],
        available: List[int],
    ) -> Tuple[int, str]:
        """
        Get movement action toward a specific target.
        
        Uses simple manhattan distance reduction.
        """
        r, c = position
        tr, tc = target
        
        move_actions = [a for a in available if a in [1, 2, 3, 4]]
        if not move_actions:
            return available[0] if available else 1, "No movement"
        
        # Prioritize direction with larger delta
        dr = tr - r  # Positive = need to go DOWN
        dc = tc - c  # Positive = need to go RIGHT
        
        candidates = []
        
        if dr < 0 and 1 in move_actions:  # Need UP
            candidates.append((1, abs(dr), "UP"))
        if dr > 0 and 2 in move_actions:  # Need DOWN
            candidates.append((2, abs(dr), "DOWN"))
        if dc < 0 and 3 in move_actions:  # Need LEFT
            candidates.append((3, abs(dc), "LEFT"))
        if dc > 0 and 4 in move_actions:  # Need RIGHT
            candidates.append((4, abs(dc), "RIGHT"))
        
        if candidates:
            # Pick direction with largest delta
            action, delta, name = max(candidates, key=lambda x: x[1])
            return action, f"Moving {name} toward target ({delta} cells)"
        
        # Already at target or blocked
        return random.choice(move_actions), "Exploring"


# ─────────────────────────────────────────────
# Click Strategy (ft09, vc33 style)
# ─────────────────────────────────────────────

class ClickStrategy:
    """
    Strategy for CLICK only games.
    
    Uses curiosity field and click history to find the best
    cells to click, avoiding repeated ineffective clicks.
    """
    
    def __init__(self, spatial_memory=None):
        """
        Initialize click strategy.
        
        Args:
            spatial_memory: SpatialMemory instance with curiosity_field
        """
        self.memory = spatial_memory
        self.clicked: Set[Tuple[int, int]] = set()  # (row, col) already clicked
        self.click_effects: Dict[Tuple[int, int], int] = {}  # (row, col) → effect count
        self.no_effect_count: Dict[Tuple[int, int], int] = {}  # (row, col) → no-effect count
        self.click_sequence: List[Tuple[int, int]] = []  # Order of clicks
    
    def reset(self):
        """Reset for new game."""
        self.clicked = set()
        self.click_effects = {}
        self.no_effect_count = {}
        self.click_sequence = []
    
    def suggest_click(
        self,
        grid: List[List[int]],
        available: List[int],
        ice_field: Optional[Tensor] = None,
    ) -> Tuple[int, int, str]:
        """
        Suggest click position based on curiosity and history.
        
        Args:
            grid: Current game grid
            available: List of available action IDs (should include 6)
            ice_field: Optional curiosity field tensor
            
        Returns:
            (row, col, reason_string)
        """
        if 6 not in available:
            return 0, 0, "Click not available"
        
        grid_h = len(grid)
        grid_w = len(grid[0]) if grid else 0
        
        if grid_h == 0 or grid_w == 0:
            return 0, 0, "Empty grid"
        
        # Get curiosity field
        if ice_field is None and self.memory is not None:
            ice_field = self.memory.curiosity_field
        
        # Find click targets
        targets = self._find_click_targets(grid, ice_field)
        
        # Score and filter targets
        scored_targets = []
        for target in targets:
            r, c = target.row, target.col
            
            # Skip if clicked too many times without effect
            if self.no_effect_count.get((r, c), 0) >= 2:
                continue
            
            # Bonus for unclicked cells
            bonus = 10.0 if (r, c) not in self.clicked else 0.0
            
            # Bonus for cells that had effects before
            effect_bonus = self.click_effects.get((r, c), 0) * 5.0
            
            score = target.score + bonus + effect_bonus
            scored_targets.append(ClickTarget(r, c, score, target.reason))
        
        # Sort by score
        scored_targets.sort(key=lambda t: t.score, reverse=True)
        
        if scored_targets:
            best = scored_targets[0]
            self.clicked.add((best.row, best.col))
            self.click_sequence.append((best.row, best.col))
            return best.row, best.col, best.reason
        
        # Fallback: systematic exploration
        return self._systematic_explore(grid_h, grid_w)
    
    def _find_click_targets(
        self,
        grid: List[List[int]],
        ice_field: Optional[Tensor],
    ) -> List[ClickTarget]:
        """Find potential click targets in the grid."""
        targets = []
        grid_h = len(grid)
        grid_w = len(grid[0]) if grid else 0
        
        # Count colors to identify background
        color_counts: Dict[int, int] = {}
        for r, row in enumerate(grid):
            for c, val in enumerate(row):
                if isinstance(val, int):
                    color_counts[val] = color_counts.get(val, 0) + 1
        
        # Background is most common color (usually 0)
        bg_color = max(color_counts, key=color_counts.get) if color_counts else 0
        
        # Find non-background cells
        for r, row in enumerate(grid):
            for c, val in enumerate(row):
                if not isinstance(val, int):
                    continue
                if val == bg_color:
                    continue
                
                # Base score from curiosity
                score = 0.0
                if ice_field is not None:
                    if 0 <= r < ice_field.shape[0] and 0 <= c < ice_field.shape[1]:
                        score = ice_field[r, c].item()
                
                # Rare colors are more interesting
                rarity = 1.0 / max(1, color_counts.get(val, 1))
                score += rarity * 10.0
                
                targets.append(ClickTarget(
                    row=r,
                    col=c,
                    score=score,
                    reason=f"Clicking cell ({r},{c}) color={val} ice={score:.1f}",
                ))
        
        return targets
    
    def _systematic_explore(
        self,
        grid_h: int,
        grid_w: int,
    ) -> Tuple[int, int, str]:
        """Systematically explore the grid when no good targets."""
        # Divide grid into regions
        region_size = max(1, min(grid_h, grid_w) // 4)
        
        for start_r in range(0, grid_h, region_size):
            for start_c in range(0, grid_w, region_size):
                r = min(start_r + region_size // 2, grid_h - 1)
                c = min(start_c + region_size // 2, grid_w - 1)
                
                if (r, c) not in self.clicked:
                    self.clicked.add((r, c))
                    self.click_sequence.append((r, c))
                    return r, c, f"Systematic exploration of region ({r},{c})"
        
        # All regions explored, try random
        for _ in range(10):
            r = random.randint(0, grid_h - 1)
            c = random.randint(0, grid_w - 1)
            if (r, c) not in self.clicked:
                self.clicked.add((r, c))
                self.click_sequence.append((r, c))
                return r, c, f"Random unexplored ({r},{c})"
        
        # Everything clicked, reset and try again
        r, c = grid_h // 2, grid_w // 2
        return r, c, "Center fallback"
    
    def record_effect(self, row: int, col: int, had_effect: bool):
        """Record whether a click had effect."""
        key = (row, col)
        if had_effect:
            self.click_effects[key] = self.click_effects.get(key, 0) + 1
            # Reset no-effect count on success
            self.no_effect_count[key] = 0
        else:
            self.no_effect_count[key] = self.no_effect_count.get(key, 0) + 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get click strategy statistics."""
        total_clicked = len(self.clicked)
        effective = sum(1 for v in self.click_effects.values() if v > 0)
        
        return {
            'total_clicked': total_clicked,
            'effective_clicks': effective,
            'effectiveness_rate': effective / max(1, total_clicked),
            'sequence_length': len(self.click_sequence),
        }


# ─────────────────────────────────────────────
# Hybrid Strategy (Movement + Click)
# ─────────────────────────────────────────────

class HybridStrategy:
    """
    Strategy for games with both movement and click.
    
    Coordinates between MovementStrategy and ClickStrategy based on
    the current game state and what seems most effective.
    """
    
    def __init__(self, spatial_memory=None):
        """Initialize hybrid strategy."""
        self.memory = spatial_memory
        self.movement = MovementStrategy(spatial_memory)
        self.click = ClickStrategy(spatial_memory)
        self.action_history: List[int] = []
        self.mode: str = "explore"  # "explore" | "click" | "navigate"
    
    def reset(self):
        """Reset for new game."""
        self.movement.reset()
        self.click.reset()
        self.action_history = []
        self.mode = "explore"
    
    def suggest_action(
        self,
        grid: List[List[int]],
        position: Tuple[int, int],
        available: List[int],
        ice_field: Optional[Tensor] = None,
    ) -> Tuple[int, Dict, str]:
        """
        Suggest next action (movement or click).
        
        Returns:
            (action_id, action_data, reason_string)
            action_data is {"x": col, "y": row} for clicks, {} for movement
        """
        has_movement = any(a in [1, 2, 3, 4] for a in available)
        has_click = 6 in available
        
        # Only click available
        if has_click and not has_movement:
            r, c, reason = self.click.suggest_click(grid, available, ice_field)
            return 6, {"x": c, "y": r}, reason
        
        # Only movement available
        if has_movement and not has_click:
            action, reason = self.movement.suggest_action(grid, position, available, ice_field)
            return action, {}, reason
        
        # Both available - decide based on mode and state
        if self.mode == "click" or (len(self.action_history) % 10 == 0):
            # Occasionally try clicking
            r, c, reason = self.click.suggest_click(grid, available, ice_field)
            self.action_history.append(6)
            return 6, {"x": c, "y": r}, reason
        else:
            # Default to movement
            action, reason = self.movement.suggest_action(grid, position, available, ice_field)
            self.action_history.append(action)
            return action, {}, reason
    
    def record_effect(self, action: int, had_effect: bool, click_pos: Optional[Tuple[int, int]] = None):
        """Record action effect for learning."""
        if action == 6 and click_pos is not None:
            r, c = click_pos
            self.click.record_effect(r, c, had_effect)
            
            # Switch to click mode if clicking works
            if had_effect:
                self.mode = "click"
        elif action in [1, 2, 3, 4] and had_effect:
            self.mode = "explore"


# ─────────────────────────────────────────────
# Control Scheme Detection
# ─────────────────────────────────────────────

def get_control_scheme(available_actions: List[int]) -> str:
    """
    Determine game control type from available actions.
    
    Args:
        available_actions: List of action IDs available in the game
        
    Returns:
        Control scheme string
    """
    has_movement = any(a in [1, 2, 3, 4] for a in available_actions)
    has_click = 6 in available_actions
    has_interact = 5 in available_actions
    has_undo = 7 in available_actions
    
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


CONTROL_INSTRUCTIONS = {
    "MOVEMENT_ONLY": "Move toward unexplored cyan areas. Find paths through obstacles.",
    "CLICK_ONLY": "Click on cells to change their state. Look for matching patterns.",
    "MOVE_AND_INTERACT": "Navigate to objects and press SPACE to interact.",
    "MOVE_AND_CLICK": "Move to position, then click specific targets.",
    "SPACE_ONLY": "Press SPACE to advance. Watch for timing patterns.",
}

ACTION_FORMAT = {
    "MOVEMENT_ONLY": "[UP/DOWN/LEFT/RIGHT]",
    "CLICK_ONLY": "CLICK(row, col)  # e.g., CLICK(3, 5)", 
    "MOVE_AND_INTERACT": "[UP/DOWN/LEFT/RIGHT/SPACE]",
    "MOVE_AND_CLICK": "[UP/DOWN/LEFT/RIGHT] or CLICK(row, col)",
    "SPACE_ONLY": "SPACE",
}
