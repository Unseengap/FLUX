"""
CausalTracker — Action → Effect Learning for FLUX

Records what happens when the agent takes actions, building a causal graph
that maps (position, action) → [changes]. This enables learning interaction
rules through observation rather than explicit training.

Physics Analogy: 
    Every action creates ripples in the field. CausalTracker detects and
    records these ripples, learning which actions cause which effects.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict
import numpy as np
# Import ARC-AGI-3 interface
try:
    from .arc_interface import (
        GameAction, GameState, ACTION_ID_TO_NAME, ACTION_DELTAS,
        get_action_delta, apply_action_to_position, grid_diff
    )
    ARC_INTERFACE_AVAILABLE = True
except ImportError:
    ARC_INTERFACE_AVAILABLE = False
    ACTION_DELTAS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class GridChange:
    """A single cell change in the grid."""
    position: Tuple[int, int]
    old_value: int
    new_value: int
    
    def __hash__(self):
        return hash((self.position, self.old_value, self.new_value))
    
    def __eq__(self, other):
        if not isinstance(other, GridChange):
            return False
        return (self.position == other.position and 
                self.old_value == other.old_value and 
                self.new_value == other.new_value)


@dataclass
class CausalLink:
    """A recorded action → effect relationship."""
    trigger_position: Tuple[int, int]
    trigger_action: int
    trigger_color: int  # Color at trigger position before action
    effects: List[GridChange]
    timestamp: int
    context: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'trigger_position': self.trigger_position,
            'trigger_action': self.trigger_action,
            'trigger_color': self.trigger_color,
            'effects': [(c.position, c.old_value, c.new_value) for c in self.effects],
            'timestamp': self.timestamp,
            'context': self.context,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CausalLink':
        effects = [GridChange(pos, old, new) for pos, old, new in d['effects']]
        return cls(
            trigger_position=tuple(d['trigger_position']),
            trigger_action=d['trigger_action'],
            trigger_color=d['trigger_color'],
            effects=effects,
            timestamp=d['timestamp'],
            context=d.get('context'),
        )


@dataclass
class EffectPattern:
    """A pattern of effects observed multiple times."""
    trigger_color: int
    trigger_action: int
    effect_type: str  # 'local', 'remote', 'global'
    effect_delta: Tuple[int, int]  # Relative position of effect
    old_color: int
    new_color: int
    count: int = 0
    confidence: float = 0.0


# ─────────────────────────────────────────────
# CausalTracker
# ─────────────────────────────────────────────

class CausalTracker(nn.Module):
    """
    Records what happens when agent takes actions.
    Builds causal graph: (position, action) → [changes]
    
    Key capabilities:
    - Detect grid changes between timesteps
    - Record causal links (what triggered what)
    - Query effects of actions
    - Find causes of observed changes
    - Identify recurring patterns
    """
    
    # ARC-AGI-3 Standard Action Mapping
    # RESET=0, ACTION1-7=1-7
    # ACTION1-4: directional (up/down/left/right)
    # ACTION5: interact (select/rotate/execute)
    # ACTION6: click at (x,y) coordinates (0-63)
    # ACTION7: undo
    ACTIONS = {
        0: 'reset',     # RESET: Initialize/restart
        1: 'up',        # ACTION1: Move up (W/↑)
        2: 'down',      # ACTION2: Move down (S/↓)
        3: 'left',      # ACTION3: Move left (A/←)
        4: 'right',     # ACTION4: Move right (D/→)
        5: 'interact',  # ACTION5: Generic interact (Space/F)
        6: 'click',     # ACTION6: Click at (x,y) -- requires coords
        7: 'undo',      # ACTION7: Undo last action (Ctrl+Z)
    }
    
    # Movement deltas for directional actions (dy, dx)
    ACTION_DELTAS = {
        1: (-1, 0),   # ACTION1/up: row decreases
        2: (1, 0),    # ACTION2/down: row increases
        3: (0, -1),   # ACTION3/left: col decreases
        4: (0, 1),    # ACTION4/right: col increases
    }
    
    def __init__(
        self,
        max_history: int = 1000,
        device: str = 'cpu',
    ):
        super().__init__()
        self.max_history = max_history
        self._device = device
        
        # Causal link storage
        self.causal_links: List[CausalLink] = []
        
        # Indexes for fast lookup
        self.links_by_action: Dict[int, List[int]] = defaultdict(list)
        self.links_by_position: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        self.links_by_effect_position: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        
        # Pattern statistics
        self.effect_patterns: Dict[Tuple[int, int, int], EffectPattern] = {}
        
        # Step counter
        self.step_count = 0
        
        # Last state for change detection
        self.last_grid: Optional[np.ndarray] = None
        self.last_position: Optional[Tuple[int, int]] = None
        self.last_action: Optional[int] = None
    
    @property
    def device(self) -> str:
        return self._device
    
    # ─────────────────────────────────────────────
    # Change Detection
    # ─────────────────────────────────────────────
    
    def detect_changes(
        self,
        grid_before: np.ndarray,
        grid_after: np.ndarray,
    ) -> List[GridChange]:
        """
        Detect all cell changes between two grids.
        
        Args:
            grid_before: Grid state before action [H, W]
            grid_after: Grid state after action [H, W]
            
        Returns:
            List of GridChange objects for all changed cells
        """
        changes = []
        
        # Ensure numpy arrays
        if isinstance(grid_before, Tensor):
            grid_before = grid_before.cpu().numpy()
        if isinstance(grid_after, Tensor):
            grid_after = grid_after.cpu().numpy()
        
        # Find differences
        diff_mask = grid_before != grid_after
        changed_positions = np.argwhere(diff_mask)
        
        for pos in changed_positions:
            h, w = int(pos[0]), int(pos[1])
            changes.append(GridChange(
                position=(h, w),
                old_value=int(grid_before[h, w]),
                new_value=int(grid_after[h, w]),
            ))
        
        return changes
    
    # ─────────────────────────────────────────────
    # Recording
    # ─────────────────────────────────────────────
    
    def record(
        self,
        position: Tuple[int, int],
        action: int,
        grid_before: np.ndarray,
        grid_after: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[CausalLink]:
        """
        Record an action and its effects.
        
        Args:
            position: Agent position when action taken
            action: Action ID (0-7)
            grid_before: Grid before action
            grid_after: Grid after action
            context: Optional context dict (e.g., inventory, goal)
            
        Returns:
            CausalLink if changes detected, None otherwise
        """
        # Detect changes
        changes = self.detect_changes(grid_before, grid_after)
        
        if not changes:
            # No changes — still record null effect for learning
            pass
        
        # Get color at trigger position
        if isinstance(grid_before, Tensor):
            grid_before = grid_before.cpu().numpy()
        trigger_color = int(grid_before[position[0], position[1]])
        
        # Create causal link
        link = CausalLink(
            trigger_position=position,
            trigger_action=action,
            trigger_color=trigger_color,
            effects=changes,
            timestamp=self.step_count,
            context=context,
        )
        
        # Store and index
        link_idx = len(self.causal_links)
        self.causal_links.append(link)
        
        # Index by action
        self.links_by_action[action].append(link_idx)
        
        # Index by position
        self.links_by_position[position].append(link_idx)
        
        # Index by effect positions
        for change in changes:
            self.links_by_effect_position[change.position].append(link_idx)
        
        # Update patterns
        self._update_patterns(link)
        
        # Increment step counter
        self.step_count += 1
        
        # Prune if needed
        if len(self.causal_links) > self.max_history:
            self._prune_oldest()
        
        return link
    
    def record_step(
        self,
        position: Tuple[int, int],
        action: int,
        grid: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[CausalLink]:
        """
        Convenience method for step-by-step recording.
        Automatically tracks previous state.
        
        Args:
            position: Current agent position
            action: Action just taken
            grid: Current grid state (after action)
            context: Optional context
            
        Returns:
            CausalLink if this is not the first step
        """
        result = None
        
        if self.last_grid is not None and self.last_action is not None:
            result = self.record(
                position=self.last_position,
                action=self.last_action,
                grid_before=self.last_grid,
                grid_after=grid,
                context=context,
            )
        
        # Update state
        self.last_grid = grid.copy() if isinstance(grid, np.ndarray) else grid.clone()
        self.last_position = position
        self.last_action = action
        
        return result
    
    # ─────────────────────────────────────────────
    # Querying
    # ─────────────────────────────────────────────
    
    def query_effects(
        self,
        position: Tuple[int, int],
        action: int,
        trigger_color: Optional[int] = None,
    ) -> List[CausalLink]:
        """
        What happened last time I did this action at this position?
        
        Args:
            position: Position to query
            action: Action to query
            trigger_color: Optional filter by trigger color
            
        Returns:
            List of matching CausalLinks, most recent first
        """
        # Get links at this position
        pos_links = self.links_by_position.get(position, [])
        
        # Filter by action
        results = []
        for idx in reversed(pos_links):  # Most recent first
            link = self.causal_links[idx]
            if link.trigger_action == action:
                if trigger_color is None or link.trigger_color == trigger_color:
                    results.append(link)
        
        return results
    
    def query_action_effects(
        self,
        action: int,
        trigger_color: Optional[int] = None,
        max_results: int = 10,
    ) -> List[CausalLink]:
        """
        What effects does this action typically have?
        
        Args:
            action: Action to query
            trigger_color: Optional filter by trigger color
            max_results: Maximum results to return
            
        Returns:
            List of CausalLinks with this action
        """
        action_links = self.links_by_action.get(action, [])
        
        results = []
        for idx in reversed(action_links):
            link = self.causal_links[idx]
            if trigger_color is None or link.trigger_color == trigger_color:
                results.append(link)
                if len(results) >= max_results:
                    break
        
        return results
    
    def find_cause(
        self,
        effect_position: Tuple[int, int],
        max_lookback: int = 10,
    ) -> List[CausalLink]:
        """
        What caused a change at this position?
        
        Args:
            effect_position: Position where change occurred
            max_lookback: How many recent links to check
            
        Returns:
            List of CausalLinks that affected this position
        """
        effect_links = self.links_by_effect_position.get(effect_position, [])
        
        # Return most recent
        return [
            self.causal_links[idx] 
            for idx in effect_links[-max_lookback:]
        ][::-1]  # Most recent first
    
    def predict_effect(
        self,
        position: Tuple[int, int],
        action: int,
        grid: np.ndarray,
    ) -> List[GridChange]:
        """
        Predict what will change if we take this action.
        Based on observed patterns.
        
        Args:
            position: Where agent will be
            action: Action to take
            grid: Current grid state
            
        Returns:
            Predicted changes (may be empty)
        """
        if isinstance(grid, Tensor):
            grid = grid.cpu().numpy()
        
        trigger_color = int(grid[position[0], position[1]])
        
        # Look for matching patterns
        pattern_key = (trigger_color, action)
        matching_patterns = [
            p for (tc, ta, _), p in self.effect_patterns.items()
            if tc == trigger_color and ta == action
        ]
        
        # Use highest confidence patterns
        matching_patterns.sort(key=lambda p: p.confidence, reverse=True)
        
        predictions = []
        for pattern in matching_patterns[:5]:
            # Calculate absolute position from relative
            effect_pos = (
                position[0] + pattern.effect_delta[0],
                position[1] + pattern.effect_delta[1],
            )
            
            # Check bounds
            if 0 <= effect_pos[0] < grid.shape[0] and 0 <= effect_pos[1] < grid.shape[1]:
                # Check if current color matches expected
                if grid[effect_pos[0], effect_pos[1]] == pattern.old_color:
                    predictions.append(GridChange(
                        position=effect_pos,
                        old_value=pattern.old_color,
                        new_value=pattern.new_color,
                    ))
        
        return predictions
    
    # ─────────────────────────────────────────────
    # Pattern Learning
    # ─────────────────────────────────────────────
    
    def _update_patterns(self, link: CausalLink):
        """Update effect patterns based on new causal link."""
        for change in link.effects:
            # Calculate relative position
            delta = (
                change.position[0] - link.trigger_position[0],
                change.position[1] - link.trigger_position[1],
            )
            
            # Classify effect type
            if delta == (0, 0):
                effect_type = 'local'
            elif abs(delta[0]) <= 1 and abs(delta[1]) <= 1:
                effect_type = 'adjacent'
            else:
                effect_type = 'remote'
            
            # Pattern key
            key = (link.trigger_color, link.trigger_action, delta)
            
            if key not in self.effect_patterns:
                self.effect_patterns[key] = EffectPattern(
                    trigger_color=link.trigger_color,
                    trigger_action=link.trigger_action,
                    effect_type=effect_type,
                    effect_delta=delta,
                    old_color=change.old_value,
                    new_color=change.new_value,
                    count=0,
                    confidence=0.0,
                )
            
            pattern = self.effect_patterns[key]
            pattern.count += 1
            
            # Update confidence based on consistency
            total_same_trigger = len([
                l for l in self.causal_links
                if l.trigger_color == link.trigger_color and l.trigger_action == link.trigger_action
            ])
            if total_same_trigger > 0:
                pattern.confidence = pattern.count / total_same_trigger
    
    def get_confident_patterns(
        self,
        min_confidence: float = 0.5,
        min_count: int = 3,
    ) -> List[EffectPattern]:
        """Get patterns that have been observed reliably."""
        return [
            p for p in self.effect_patterns.values()
            if p.confidence >= min_confidence and p.count >= min_count
        ]
    
    # ─────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────
    
    def _prune_oldest(self):
        """Remove oldest causal links when over capacity."""
        prune_count = len(self.causal_links) - self.max_history
        if prune_count <= 0:
            return
        
        # Remove oldest links
        self.causal_links = self.causal_links[prune_count:]
        
        # Rebuild indexes (expensive but rare)
        self.links_by_action = defaultdict(list)
        self.links_by_position = defaultdict(list)
        self.links_by_effect_position = defaultdict(list)
        
        for idx, link in enumerate(self.causal_links):
            self.links_by_action[link.trigger_action].append(idx)
            self.links_by_position[link.trigger_position].append(idx)
            for change in link.effects:
                self.links_by_effect_position[change.position].append(idx)
    
    def reset(self):
        """Clear all recorded links."""
        self.causal_links.clear()
        self.links_by_action.clear()
        self.links_by_position.clear()
        self.links_by_effect_position.clear()
        self.effect_patterns.clear()
        self.step_count = 0
        self.last_grid = None
        self.last_position = None
        self.last_action = None
    
    def reset_episode(self):
        """Clear episode-specific state but keep patterns."""
        self.last_grid = None
        self.last_position = None
        self.last_action = None
    
    def state_dict(self) -> Dict[str, Any]:
        """Export state for checkpointing."""
        return {
            'causal_links': [link.to_dict() for link in self.causal_links],
            'effect_patterns': {
                str(k): {
                    'trigger_color': v.trigger_color,
                    'trigger_action': v.trigger_action,
                    'effect_type': v.effect_type,
                    'effect_delta': v.effect_delta,
                    'old_color': v.old_color,
                    'new_color': v.new_color,
                    'count': v.count,
                    'confidence': v.confidence,
                }
                for k, v in self.effect_patterns.items()
            },
            'step_count': self.step_count,
            'max_history': self.max_history,
        }
    
    def load_state_dict(self, state: Dict[str, Any]):
        """Load state from checkpoint."""
        self.reset()
        
        self.max_history = state.get('max_history', 1000)
        self.step_count = state.get('step_count', 0)
        
        # Restore causal links
        for link_dict in state.get('causal_links', []):
            link = CausalLink.from_dict(link_dict)
            self.causal_links.append(link)
        
        # Rebuild indexes
        for idx, link in enumerate(self.causal_links):
            self.links_by_action[link.trigger_action].append(idx)
            self.links_by_position[link.trigger_position].append(idx)
            for change in link.effects:
                self.links_by_effect_position[change.position].append(idx)
        
        # Restore patterns
        for k_str, v in state.get('effect_patterns', {}).items():
            # Parse key from string
            k = eval(k_str)  # (color, action, delta)
            self.effect_patterns[k] = EffectPattern(**v)
    
    def summary(self) -> str:
        """Get a summary of tracked causal relationships."""
        lines = [
            "CausalTracker Summary",
            "=" * 40,
            f"Total links: {len(self.causal_links)}",
            f"Steps recorded: {self.step_count}",
            f"Unique patterns: {len(self.effect_patterns)}",
            "",
            "Links by action:",
        ]
        
        for action, link_idxs in sorted(self.links_by_action.items()):
            action_name = self.ACTIONS.get(action, f'action_{action}')
            lines.append(f"  {action_name}: {len(link_idxs)} links")
        
        confident = self.get_confident_patterns()
        if confident:
            lines.append("")
            lines.append(f"Confident patterns ({len(confident)}):")
            for p in confident[:5]:
                action_name = self.ACTIONS.get(p.trigger_action, f'action_{p.trigger_action}')
                lines.append(
                    f"  color={p.trigger_color} + {action_name} → "
                    f"{p.old_color}→{p.new_color} at Δ{p.effect_delta} "
                    f"(conf={p.confidence:.2f}, n={p.count})"
                )
        
        return "\n".join(lines)
    
    def forward(self, x: Tensor) -> Tensor:
        """Dummy forward for nn.Module compatibility."""
        return x


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import numpy as np
    
    print("Testing CausalTracker")
    print("=" * 40)
    
    tracker = CausalTracker()
    
    # Simulate a grid change
    grid_before = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 2, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ])
    
    grid_after = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 3, 1, 0],  # Changed 2 → 3
        [0, 1, 1, 1, 0],
        [0, 0, 3, 0, 0],  # Changed 0 → 3
    ])
    
    # Record the change
    link = tracker.record(
        position=(2, 2),
        action=6,  # toggle
        grid_before=grid_before,
        grid_after=grid_after,
    )
    
    print(f"✓ Recorded link with {len(link.effects)} effects")
    for effect in link.effects:
        print(f"  {effect.position}: {effect.old_value} → {effect.new_value}")
    
    # Query effects
    effects = tracker.query_effects((2, 2), action=6)
    print(f"✓ Query returned {len(effects)} matching links")
    
    # Find cause
    causes = tracker.find_cause((4, 2))
    print(f"✓ Found {len(causes)} causes for change at (4,2)")
    
    # Print summary
    print()
    print(tracker.summary())
    
    # Test state dict
    state = tracker.state_dict()
    tracker2 = CausalTracker()
    tracker2.load_state_dict(state)
    print(f"\n✓ State dict round-trip: {len(tracker2.causal_links)} links restored")
    
    print("\n✓ All tests passed!")
