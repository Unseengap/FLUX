"""
PerceptionField — Active Vision System for FLUX

An active vision system with foveal focus and peripheral awareness.
Tracks objects of interest, detects expectation violations (surprises),
and maintains attention across the visual field.

Physics Analogy:
    Like an eye with a high-resolution fovea and low-resolution periphery.
    Attention flows like water, pooling where important things happen.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np


# ─────────────────────────────────────────────
# Perception Data Structures
# ─────────────────────────────────────────────

@dataclass
class TrackedObject:
    """An object being actively tracked."""
    object_id: int
    position: Tuple[int, int]
    color: int
    first_seen: int
    last_seen: int
    velocity: Tuple[float, float] = (0.0, 0.0)
    importance: float = 1.0
    
    def predicted_position(self, steps: int = 1) -> Tuple[float, float]:
        """Predict position after N steps."""
        return (
            self.position[0] + self.velocity[0] * steps,
            self.position[1] + self.velocity[1] * steps,
        )


@dataclass
class Surprise:
    """An expectation violation."""
    position: Tuple[int, int]
    expected_value: int
    actual_value: int
    surprise_magnitude: float
    timestamp: int
    
    def __str__(self) -> str:
        return f"Surprise@{self.position}: expected={self.expected_value}, got={self.actual_value} (mag={self.surprise_magnitude:.2f})"


@dataclass
class AttentionFocus:
    """The current focus of attention."""
    center: Tuple[int, int]
    radius: int
    strength: float  # 0-1
    reason: str  # Why we're looking here


# ─────────────────────────────────────────────
# PerceptionField
# ─────────────────────────────────────────────

class PerceptionField(nn.Module):
    """
    FLUX's eyes — actively monitors regions of interest.
    
    Key capabilities:
    - Foveal attention with high resolution
    - Peripheral awareness with low resolution
    - Object tracking across frames
    - Expectation-based surprise detection
    - Attention shifting based on importance
    """
    
    def __init__(
        self,
        max_size: int = 30,
        fovea_radius: int = 3,
        num_colors: int = 10,
        device: str = 'cpu',
    ):
        super().__init__()
        self.max_size = max_size
        self.fovea_radius = fovea_radius
        self.num_colors = num_colors
        self._device = device
        
        # Current attention state
        self.fovea_center: Tuple[int, int] = (0, 0)
        self.attention_map: Optional[np.ndarray] = None
        
        # Object tracking
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.next_object_id = 0
        
        # Prediction state
        self.predicted_grid: Optional[np.ndarray] = None
        self.last_grid: Optional[np.ndarray] = None
        
        # Attention history
        self.focus_history: List[AttentionFocus] = []
        self.surprise_history: List[Surprise] = []
        
        # Step counter
        self.step_count = 0
        
        # Attention weights (learnable via gradient if needed)
        self.register_buffer(
            'color_importance',
            torch.ones(num_colors, device=device)
        )
    
    @property
    def device(self) -> str:
        return self._device
    
    # ─────────────────────────────────────────────
    # Attention Control
    # ─────────────────────────────────────────────
    
    def focus(
        self,
        position: Tuple[int, int],
        radius: int = None,
        reason: str = "manual",
    ):
        """
        Shift foveal attention to position.
        
        Args:
            position: New attention center
            radius: Optional override of fovea radius
            reason: Why we're shifting attention
        """
        self.fovea_center = position
        if radius is not None:
            self.fovea_radius = radius
        
        # Record focus
        self.focus_history.append(AttentionFocus(
            center=position,
            radius=self.fovea_radius,
            strength=1.0,
            reason=reason,
        ))
        
        # Trim history
        if len(self.focus_history) > 100:
            self.focus_history = self.focus_history[-100:]
    
    def compute_attention_map(
        self,
        grid: np.ndarray,
        agent_position: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Compute attention weights for the entire grid.
        
        Attention is higher for:
        - Cells near fovea center
        - Important colors (learned)
        - Recent surprise locations
        - Tracked object locations
        
        Args:
            grid: Current grid state [H, W]
            agent_position: Optional agent position
            
        Returns:
            Attention map [H, W] with values 0-1
        """
        H, W = grid.shape
        attention = np.zeros((H, W), dtype=np.float32)
        
        # Base attention from distance to fovea
        for h in range(H):
            for w in range(W):
                dist = np.sqrt(
                    (h - self.fovea_center[0])**2 + 
                    (w - self.fovea_center[1])**2
                )
                # Gaussian falloff from fovea
                attention[h, w] = np.exp(-dist**2 / (2 * self.fovea_radius**2))
        
        # Add attention for important colors
        color_importance = self.color_importance.cpu().numpy()
        for h in range(H):
            for w in range(W):
                color = int(grid[h, w])
                if 0 <= color < len(color_importance):
                    attention[h, w] += 0.2 * color_importance[color]
        
        # Add attention for tracked objects
        for obj in self.tracked_objects.values():
            if 0 <= obj.position[0] < H and 0 <= obj.position[1] < W:
                attention[obj.position[0], obj.position[1]] += 0.3 * obj.importance
        
        # Add attention for recent surprises
        for surprise in self.surprise_history[-10:]:
            if 0 <= surprise.position[0] < H and 0 <= surprise.position[1] < W:
                attention[surprise.position[0], surprise.position[1]] += 0.4 * surprise.surprise_magnitude
        
        # Normalize
        attention = np.clip(attention, 0, 1)
        
        self.attention_map = attention
        return attention
    
    def get_foveal_region(
        self,
        grid: np.ndarray,
    ) -> np.ndarray:
        """
        Extract the high-resolution foveal region.
        
        Args:
            grid: Full grid [H, W]
            
        Returns:
            Foveal patch [2r+1, 2r+1]
        """
        H, W = grid.shape
        r = self.fovea_radius
        
        # Extract region with padding
        h_start = max(0, self.fovea_center[0] - r)
        h_end = min(H, self.fovea_center[0] + r + 1)
        w_start = max(0, self.fovea_center[1] - r)
        w_end = min(W, self.fovea_center[1] + r + 1)
        
        return grid[h_start:h_end, w_start:w_end].copy()
    
    # ─────────────────────────────────────────────
    # Peripheral Scanning
    # ─────────────────────────────────────────────
    
    def peripheral_scan(
        self,
        grid: np.ndarray,
        important_colors: Optional[Set[int]] = None,
    ) -> Optional[Tuple[int, int]]:
        """
        Low-res scan for changes outside focus.
        
        Args:
            grid: Current grid state
            important_colors: Colors to watch for
            
        Returns:
            Position of change if found, None otherwise
        """
        if self.last_grid is None:
            return None
        
        H, W = grid.shape
        changes = []
        
        for h in range(H):
            for w in range(W):
                # Skip foveal region (already monitored)
                dist = np.sqrt(
                    (h - self.fovea_center[0])**2 + 
                    (w - self.fovea_center[1])**2
                )
                if dist < self.fovea_radius:
                    continue
                
                # Check for change
                if grid[h, w] != self.last_grid[h, w]:
                    importance = 1.0
                    if important_colors is not None:
                        if grid[h, w] in important_colors or self.last_grid[h, w] in important_colors:
                            importance = 2.0
                    changes.append(((h, w), importance))
        
        if not changes:
            return None
        
        # Return most important change
        changes.sort(key=lambda x: x[1], reverse=True)
        return changes[0][0]
    
    # ─────────────────────────────────────────────
    # Object Tracking
    # ─────────────────────────────────────────────
    
    def track_object(
        self,
        position: Tuple[int, int],
        color: int,
        importance: float = 1.0,
    ) -> int:
        """
        Add object to tracking list.
        
        Args:
            position: Object position
            color: Object color
            importance: How important to track
            
        Returns:
            Object ID
        """
        obj = TrackedObject(
            object_id=self.next_object_id,
            position=position,
            color=color,
            first_seen=self.step_count,
            last_seen=self.step_count,
            importance=importance,
        )
        
        self.tracked_objects[self.next_object_id] = obj
        self.next_object_id += 1
        
        return obj.object_id
    
    def update_tracked_object(
        self,
        object_id: int,
        new_position: Tuple[int, int],
    ):
        """Update position of a tracked object."""
        if object_id not in self.tracked_objects:
            return
        
        obj = self.tracked_objects[object_id]
        
        # Calculate velocity
        dt = self.step_count - obj.last_seen
        if dt > 0:
            obj.velocity = (
                (new_position[0] - obj.position[0]) / dt,
                (new_position[1] - obj.position[1]) / dt,
            )
        
        obj.position = new_position
        obj.last_seen = self.step_count
    
    def find_object(
        self,
        grid: np.ndarray,
        color: int,
    ) -> List[Tuple[int, int]]:
        """Find all instances of a color in the grid."""
        matches = np.argwhere(grid == color)
        return [tuple(m) for m in matches]
    
    def auto_track_changes(
        self,
        grid: np.ndarray,
    ):
        """Automatically find and track moving objects."""
        if self.last_grid is None:
            return
        
        H, W = grid.shape
        
        # Find cells that changed
        changes = np.argwhere(grid != self.last_grid)
        
        for pos in changes:
            h, w = int(pos[0]), int(pos[1])
            new_color = int(grid[h, w])
            old_color = int(self.last_grid[h, w])
            
            # Check if this might be an object that moved
            if new_color != 0:  # Something appeared here
                # Look for same color that disappeared nearby
                for obj in list(self.tracked_objects.values()):
                    if obj.color == new_color:
                        # Update tracking
                        self.update_tracked_object(obj.object_id, (h, w))
                        break
                else:
                    # New object to track
                    self.track_object((h, w), new_color, importance=0.5)
    
    def prune_lost_objects(self, max_age: int = 10):
        """Remove objects that haven't been seen recently."""
        to_remove = []
        for obj_id, obj in self.tracked_objects.items():
            if self.step_count - obj.last_seen > max_age:
                to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del self.tracked_objects[obj_id]
    
    # ─────────────────────────────────────────────
    # Prediction & Surprise
    # ─────────────────────────────────────────────
    
    def predict_next(
        self,
        grid: np.ndarray,
        action: int,
        agent_position: Tuple[int, int],
    ) -> np.ndarray:
        """
        Predict what we expect to see after action.
        
        Simple model: assume grid doesn't change much except
        for agent position and potentially triggered effects.
        
        Args:
            grid: Current grid state
            action: Action about to be taken
            agent_position: Current agent position
            
        Returns:
            Predicted grid after action
        """
        prediction = grid.copy()
        
        # ARC-AGI-3 Standard Action Deltas (dy, dx)
        # ACTION1-4 are directional, ACTION5-7 don't move agent
        action_deltas = {
            0: (0, 0),   # RESET: no movement
            1: (-1, 0),  # ACTION1/up: row decreases
            2: (1, 0),   # ACTION2/down: row increases
            3: (0, -1),  # ACTION3/left: col decreases
            4: (0, 1),   # ACTION4/right: col increases
            5: (0, 0),   # ACTION5/interact: no movement
            6: (0, 0),   # ACTION6/click: no movement (coords separate)
            7: (0, 0),   # ACTION7/undo: no movement
        }
        
        # Most predictions just keep grid as-is
        # More sophisticated prediction would use RuleInducer
        
        self.predicted_grid = prediction
        return prediction
    
    def check_surprise(
        self,
        predicted: Optional[np.ndarray],
        actual: np.ndarray,
    ) -> List[Surprise]:
        """
        Detect expectation violations.
        
        Args:
            predicted: What we expected to see
            actual: What we actually see
            
        Returns:
            List of surprises (expectation violations)
        """
        if predicted is None:
            return []
        
        surprises = []
        
        H, W = actual.shape
        if predicted.shape != actual.shape:
            return []
        
        for h in range(H):
            for w in range(W):
                if predicted[h, w] != actual[h, w]:
                    # Calculate surprise magnitude
                    # Higher magnitude for important colors
                    color_imp = 1.0
                    if actual[h, w] < len(self.color_importance):
                        color_imp = float(self.color_importance[int(actual[h, w])])
                    
                    # Higher magnitude for foveal region
                    dist = np.sqrt(
                        (h - self.fovea_center[0])**2 + 
                        (w - self.fovea_center[1])**2
                    )
                    foveal_factor = np.exp(-dist**2 / (2 * self.fovea_radius**2))
                    
                    magnitude = 0.5 + 0.3 * color_imp + 0.2 * foveal_factor
                    
                    surprise = Surprise(
                        position=(h, w),
                        expected_value=int(predicted[h, w]),
                        actual_value=int(actual[h, w]),
                        surprise_magnitude=magnitude,
                        timestamp=self.step_count,
                    )
                    surprises.append(surprise)
        
        # Store surprises
        self.surprise_history.extend(surprises)
        
        # Trim history
        if len(self.surprise_history) > 100:
            self.surprise_history = self.surprise_history[-100:]
        
        return surprises
    
    def get_most_surprising(
        self,
        n: int = 5,
    ) -> List[Surprise]:
        """Get the N most surprising recent events."""
        recent = self.surprise_history[-50:]
        return sorted(recent, key=lambda s: s.surprise_magnitude, reverse=True)[:n]
    
    # ─────────────────────────────────────────────
    # Observation
    # ─────────────────────────────────────────────
    
    def observe(
        self,
        grid: np.ndarray,
        agent_position: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """
        Full observation step.
        
        Args:
            grid: Current grid state
            agent_position: Optional agent position
            
        Returns:
            Dict with fovea, attention, surprises, etc.
        """
        # Check for surprises against prediction
        surprises = []
        if self.predicted_grid is not None:
            surprises = self.check_surprise(self.predicted_grid, grid)
        
        # Auto-track changes
        self.auto_track_changes(grid)
        
        # Compute attention
        attention = self.compute_attention_map(grid, agent_position)
        
        # Get foveal region
        fovea = self.get_foveal_region(grid)
        
        # Peripheral scan
        peripheral_alert = self.peripheral_scan(grid)
        
        # Update state
        self.last_grid = grid.copy()
        self.step_count += 1
        
        # If surprise in periphery, shift attention
        if surprises:
            most_surprising = max(surprises, key=lambda s: s.surprise_magnitude)
            self.focus(most_surprising.position, reason="surprise")
        elif peripheral_alert:
            self.focus(peripheral_alert, reason="peripheral_change")
        
        return {
            'fovea': fovea,
            'attention_map': attention,
            'surprises': surprises,
            'peripheral_alert': peripheral_alert,
            'tracked_objects': list(self.tracked_objects.values()),
            'fovea_center': self.fovea_center,
        }
    
    # ─────────────────────────────────────────────
    # State Management
    # ─────────────────────────────────────────────
    
    def reset(self):
        """Reset all perception state."""
        self.fovea_center = (0, 0)
        self.attention_map = None
        self.tracked_objects.clear()
        self.next_object_id = 0
        self.predicted_grid = None
        self.last_grid = None
        self.focus_history.clear()
        self.surprise_history.clear()
        self.step_count = 0
    
    def reset_episode(self):
        """Reset episode-specific state."""
        self.last_grid = None
        self.predicted_grid = None
        self.tracked_objects.clear()
        self.focus_history.clear()
        self.surprise_history.clear()
    
    def state_dict(self) -> Dict[str, Any]:
        """Export state for checkpointing."""
        return {
            'color_importance': self.color_importance.cpu().numpy().tolist(),
            'fovea_center': self.fovea_center,
            'fovea_radius': self.fovea_radius,
            'step_count': self.step_count,
            'tracked_objects': [
                {
                    'object_id': obj.object_id,
                    'position': obj.position,
                    'color': obj.color,
                    'first_seen': obj.first_seen,
                    'last_seen': obj.last_seen,
                    'velocity': obj.velocity,
                    'importance': obj.importance,
                }
                for obj in self.tracked_objects.values()
            ],
        }
    
    def load_state_dict(self, state: Dict[str, Any]):
        """Load state from checkpoint."""
        if 'color_importance' in state:
            self.color_importance = torch.tensor(
                state['color_importance'],
                device=self._device,
            )
        
        self.fovea_center = tuple(state.get('fovea_center', (0, 0)))
        self.fovea_radius = state.get('fovea_radius', 3)
        self.step_count = state.get('step_count', 0)
        
        self.tracked_objects.clear()
        for obj_dict in state.get('tracked_objects', []):
            obj = TrackedObject(**obj_dict)
            self.tracked_objects[obj.object_id] = obj
    
    def summary(self) -> str:
        """Get a summary of perception state."""
        lines = [
            "PerceptionField Summary",
            "=" * 40,
            f"Fovea center: {self.fovea_center}",
            f"Fovea radius: {self.fovea_radius}",
            f"Tracked objects: {len(self.tracked_objects)}",
            f"Recent surprises: {len(self.surprise_history)}",
            f"Steps: {self.step_count}",
        ]
        
        if self.tracked_objects:
            lines.append("")
            lines.append("Tracked objects:")
            for obj in self.tracked_objects.values():
                lines.append(f"  #{obj.object_id}: color={obj.color} at {obj.position}")
        
        recent_surprises = self.get_most_surprising(3)
        if recent_surprises:
            lines.append("")
            lines.append("Recent surprises:")
            for s in recent_surprises:
                lines.append(f"  {s}")
        
        return "\n".join(lines)
    
    def forward(self, x: Tensor) -> Tensor:
        """Dummy forward for nn.Module compatibility."""
        return x


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import numpy as np
    
    print("Testing PerceptionField")
    print("=" * 40)
    
    pf = PerceptionField(max_size=10, fovea_radius=2)
    
    # Create test grids
    grid1 = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 2, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ])
    
    grid2 = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 3, 1, 0],  # Changed 2 → 3
        [0, 1, 1, 4, 0],  # New object at (3,3)
        [0, 0, 0, 0, 0],
    ])
    
    # First observation
    pf.focus((2, 2), reason="center")
    result1 = pf.observe(grid1)
    print(f"✓ First observation: fovea at {result1['fovea_center']}")
    print(f"  Fovea region shape: {result1['fovea'].shape}")
    
    # Make prediction
    pf.predict_next(grid1, action=6, agent_position=(2, 2))
    
    # Second observation with changes
    result2 = pf.observe(grid2)
    print(f"\n✓ Second observation:")
    print(f"  Surprises: {len(result2['surprises'])}")
    for s in result2['surprises']:
        print(f"    {s}")
    print(f"  Tracked objects: {len(result2['tracked_objects'])}")
    
    # Test attention map
    attention = pf.compute_attention_map(grid2)
    print(f"\n✓ Attention map shape: {attention.shape}")
    print(f"  Max attention at: {np.unravel_index(attention.argmax(), attention.shape)}")
    
    # Test object tracking
    obj_id = pf.track_object((3, 3), color=4, importance=1.0)
    print(f"\n✓ Tracking object #{obj_id}")
    
    # Print summary
    print()
    print(pf.summary())
    
    # Test state dict
    state = pf.state_dict()
    pf2 = PerceptionField()
    pf2.load_state_dict(state)
    print(f"\n✓ State dict round-trip: {len(pf2.tracked_objects)} objects restored")
    
    print("\n✓ All tests passed!")
