"""
Frame Differ — Visual Diff System for FLUX Unified Agent

Computes and describes changes between game frames, enabling the agent
to SEE what happened after each action.

Physics Analogy:
    Every action creates ripples in the field. FrameDiffer detects these
    ripples by comparing before/after states, giving the agent feedback
    about action effects.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import copy


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class CellChange:
    """A single cell change in the grid."""
    row: int
    col: int
    old_value: int
    new_value: int
    
    def __hash__(self):
        return hash((self.row, self.col, self.old_value, self.new_value))
    
    @property
    def is_appearance(self) -> bool:
        """Cell went from background to foreground."""
        return self.old_value == 0 and self.new_value > 0
    
    @property
    def is_disappearance(self) -> bool:
        """Cell went from foreground to background."""
        return self.old_value > 0 and self.new_value == 0
    
    @property
    def is_color_change(self) -> bool:
        """Cell changed color (both non-zero)."""
        return self.old_value > 0 and self.new_value > 0 and self.old_value != self.new_value


@dataclass
class FrameDiff:
    """Result of comparing two frames."""
    # Core info
    first_frame: bool = False
    effect: str = ""
    
    # Position tracking
    old_position: Optional[Tuple[int, int]] = None
    new_position: Optional[Tuple[int, int]] = None
    position_changed: bool = False
    position_delta: Tuple[int, int] = (0, 0)
    
    # Cell changes
    changes: List[CellChange] = field(default_factory=list)
    num_changes: int = 0
    
    # Derived info
    action_worked: bool = False
    appeared_cells: List[Tuple[int, int]] = field(default_factory=list)
    disappeared_cells: List[Tuple[int, int]] = field(default_factory=list)
    color_changed_cells: List[Tuple[int, int]] = field(default_factory=list)
    
    def summary(self) -> str:
        """Human-readable summary of the diff."""
        if self.first_frame:
            return "First frame - no comparison available"
        
        parts = []
        if self.position_changed:
            parts.append(f"Position: {self.old_position} → {self.new_position}")
        if self.num_changes > 0:
            parts.append(f"{self.num_changes} cells changed")
        if not parts:
            return "No visible changes"
        return " | ".join(parts)


# ─────────────────────────────────────────────
# Action Names
# ─────────────────────────────────────────────

ACTION_NAMES = {
    0: "RESET",
    1: "UP",
    2: "DOWN", 
    3: "LEFT",
    4: "RIGHT",
    5: "SPACE/INTERACT",
    6: "CLICK",
    7: "UNDO",
}


# ─────────────────────────────────────────────
# Core: Frame Differ
# ─────────────────────────────────────────────

class FrameDiffer:
    """
    Computes and describes changes between consecutive game frames.
    
    Used to show the agent what happened after each action, enabling
    it to learn action → effect relationships.
    """
    
    def __init__(self):
        """Initialize frame differ."""
        self.prev_frame: Optional[List[List[int]]] = None
        self.prev_position: Optional[Tuple[int, int]] = None
        self.prev_action: Optional[int] = None
        self.diff_history: List[FrameDiff] = []
    
    def reset(self):
        """Reset for new game."""
        self.prev_frame = None
        self.prev_position = None
        self.prev_action = None
        self.diff_history = []
    
    def diff(
        self,
        new_frame: List[List[int]],
        new_position: Tuple[int, int],
        last_action: Optional[int] = None,
    ) -> FrameDiff:
        """
        Compare new frame to previous, return change summary.
        
        Args:
            new_frame: Current grid state [rows x cols] of ints
            new_position: Current agent position (row, col)
            last_action: Action taken to get here (1-7)
            
        Returns:
            FrameDiff with all change information
        """
        # Handle first frame
        if self.prev_frame is None:
            self.prev_frame = self._deep_copy_grid(new_frame)
            self.prev_position = new_position
            self.prev_action = last_action
            
            result = FrameDiff(
                first_frame=True,
                new_position=new_position,
                effect="First frame",
            )
            self.diff_history.append(result)
            return result
        
        # Compute position change
        pos_changed = new_position != self.prev_position
        pos_delta = (0, 0)
        if pos_changed and self.prev_position is not None:
            pos_delta = (
                new_position[0] - self.prev_position[0],
                new_position[1] - self.prev_position[1],
            )
        
        # Compute cell changes
        changes = self._compute_cell_changes(self.prev_frame, new_frame)
        
        # Categorize changes
        appeared = [(c.row, c.col) for c in changes if c.is_appearance]
        disappeared = [(c.row, c.col) for c in changes if c.is_disappearance]
        color_changed = [(c.row, c.col) for c in changes if c.is_color_change]
        
        # Build effect description
        effect = self._describe_effect(
            pos_changed=pos_changed,
            pos_delta=pos_delta,
            num_changes=len(changes),
            last_action=last_action,
        )
        
        # Action worked if anything changed
        action_worked = pos_changed or len(changes) > 0
        
        # Build result
        result = FrameDiff(
            first_frame=False,
            effect=effect,
            old_position=self.prev_position,
            new_position=new_position,
            position_changed=pos_changed,
            position_delta=pos_delta,
            changes=changes,
            num_changes=len(changes),
            action_worked=action_worked,
            appeared_cells=appeared,
            disappeared_cells=disappeared,
            color_changed_cells=color_changed,
        )
        
        # Store for next comparison
        self.prev_frame = self._deep_copy_grid(new_frame)
        self.prev_position = new_position
        self.prev_action = last_action
        self.diff_history.append(result)
        
        return result
    
    def _compute_cell_changes(
        self,
        old_frame: List[List[int]],
        new_frame: List[List[int]],
    ) -> List[CellChange]:
        """Find all cells that changed between frames."""
        changes = []
        
        if not old_frame or not new_frame:
            return changes
        
        old_h = len(old_frame)
        new_h = len(new_frame)
        old_w = len(old_frame[0]) if old_frame else 0
        new_w = len(new_frame[0]) if new_frame else 0
        
        # Compare overlapping region
        for r in range(min(old_h, new_h)):
            old_row = old_frame[r] if r < old_h else []
            new_row = new_frame[r] if r < new_h else []
            
            for c in range(min(len(old_row), len(new_row))):
                old_val = old_row[c] if c < len(old_row) else 0
                new_val = new_row[c] if c < len(new_row) else 0
                
                # Ensure int values
                old_val = old_val if isinstance(old_val, int) else 0
                new_val = new_val if isinstance(new_val, int) else 0
                
                if old_val != new_val:
                    changes.append(CellChange(
                        row=r,
                        col=c,
                        old_value=old_val,
                        new_value=new_val,
                    ))
        
        return changes
    
    def _describe_effect(
        self,
        pos_changed: bool,
        pos_delta: Tuple[int, int],
        num_changes: int,
        last_action: Optional[int],
    ) -> str:
        """Generate human-readable effect description."""
        action_name = ACTION_NAMES.get(last_action, f"ACTION{last_action}") if last_action else "unknown"
        
        if num_changes == 0 and not pos_changed:
            return f"NO EFFECT - {action_name} had no visible result"
        
        parts = []
        
        if pos_changed:
            dr, dc = pos_delta
            direction = []
            if dr < 0:
                direction.append("up")
            elif dr > 0:
                direction.append("down")
            if dc < 0:
                direction.append("left")
            elif dc > 0:
                direction.append("right")
            dir_str = "-".join(direction) if direction else "moved"
            parts.append(f"Moved {dir_str} ({abs(dr) + abs(dc)} cells)")
        
        if num_changes > 0:
            parts.append(f"{num_changes} cells changed")
        
        return " AND ".join(parts)
    
    def _deep_copy_grid(self, grid: List[List[int]]) -> List[List[int]]:
        """Create a deep copy of the grid."""
        return [row[:] for row in grid]
    
    def get_recent_effects(self, n: int = 3) -> List[str]:
        """Get descriptions of last N effects."""
        recent = self.diff_history[-n:] if self.diff_history else []
        return [d.effect for d in recent]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get diff statistics."""
        total_changes = sum(d.num_changes for d in self.diff_history)
        effective_actions = sum(1 for d in self.diff_history if d.action_worked)
        
        return {
            'total_diffs': len(self.diff_history),
            'total_changes': total_changes,
            'effective_actions': effective_actions,
            'effectiveness_rate': effective_actions / max(1, len(self.diff_history)),
        }


# ─────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────

def compute_grid_diff(
    old_grid: List[List[int]],
    new_grid: List[List[int]],
) -> List[CellChange]:
    """
    Compute cell changes between two grids.
    
    Standalone function for quick comparisons.
    """
    differ = FrameDiffer()
    differ.prev_frame = old_grid
    return differ._compute_cell_changes(old_grid, new_grid)


def describe_changes(changes: List[CellChange]) -> str:
    """Generate description of cell changes."""
    if not changes:
        return "No changes"
    
    appeared = sum(1 for c in changes if c.is_appearance)
    disappeared = sum(1 for c in changes if c.is_disappearance)
    color_changed = sum(1 for c in changes if c.is_color_change)
    
    parts = []
    if appeared:
        parts.append(f"{appeared} appeared")
    if disappeared:
        parts.append(f"{disappeared} disappeared")
    if color_changed:
        parts.append(f"{color_changed} changed color")
    
    return ", ".join(parts) if parts else f"{len(changes)} changes"
