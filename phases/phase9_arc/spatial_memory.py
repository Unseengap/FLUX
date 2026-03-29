"""
Phase 9 ARC: Spatial Memory — Dual-Field Navigation System

The Ice & Water Model for ARC-AGI-3 spatial awareness:

    EXPLORATION MASS FIELD — "Where have I been, what did I see"
    - Visited locations gain mass (gravitational breadcrumbs)
    - Mass stores WHAT was observed (last observation)
    - Enables efficient pathfinding via mass gradient
    
    CURIOSITY ICE FIELD — "What's interesting right now"
    - Anomalies/contrasts create "ice" (high curiosity mass)
    - Changes to visited areas spawn ice (something's different!)
    - Agent is gravitationally pulled toward ice
    
Combined, this gives FLUX agents:
- Spatial memory (knows where things were)
- Curiosity-driven exploration (pulled toward anomalies)
- Change detection (notices when reality differs from memory)
- Efficient navigation (uses mass trail to plan paths)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import math


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

SPATIAL_MEMORY_CONFIG = {
    'max_grid_size': 30,          # Max ARC grid size
    'feature_dim': 64,            # Features per location
    'curiosity_threshold': 0.1,   # Change amount to trigger ice
    'mass_decay': 0.99,           # Exploration mass decay per step
    'ice_decay': 0.95,            # Curiosity ice decay per step
    'visit_mass_gain': 1.0,       # Mass gained per visit
    'change_ice_multiplier': 5.0, # Ice multiplier for detected changes
}


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class NavigationTarget:
    """A target location for the agent to visit."""
    position: Tuple[int, int]     # (row, col) target
    curiosity_score: float        # Why we want to go there
    reason: str                   # Human-readable reason
    path: Optional[List[Tuple[int, int]]] = None  # Planned path


@dataclass
class SpatialObservation:
    """Observation at a specific spatial location."""
    position: Tuple[int, int]
    features: Tensor              # [feature_dim] encoded observation
    timestamp: int                # When observed
    raw_grid: Optional[List[List[int]]] = None


# ─────────────────────────────────────────────
# Core: Spatial Memory System
# ─────────────────────────────────────────────

class SpatialMemory(nn.Module):
    """
    Dual-Field Spatial Memory for ARC-AGI-3 Agent.
    
    Combines:
    1. Exploration Mass Field — tracks where agent has been
    2. Curiosity Ice Field — highlights interesting locations
    
    Together, enables efficient goal-directed exploration.
    """
    
    def __init__(
        self,
        max_size: int = 30,
        feature_dim: int = 64,
        curiosity_threshold: float = 0.1,
        device: str = 'cpu',
    ):
        super().__init__()
        self.max_size = max_size
        self.feature_dim = feature_dim
        self.curiosity_threshold = curiosity_threshold
        self.device = device
        
        # ── Exploration Mass Field ──
        # How many times each location visited
        self.register_buffer(
            'visit_count',
            torch.zeros(max_size, max_size)
        )
        # Accumulated mass per location (decays over time)
        self.register_buffer(
            'exploration_mass',
            torch.zeros(max_size, max_size)
        )
        # Last observed features at each location
        self.register_buffer(
            'last_observation',
            torch.zeros(max_size, max_size, feature_dim)
        )
        # Timestamp of last visit
        self.register_buffer(
            'last_visit_time',
            torch.zeros(max_size, max_size, dtype=torch.long)
        )
        
        # ── Curiosity Ice Field ──
        # Salience score per location (ice = high curiosity)
        self.register_buffer(
            'curiosity_field',
            torch.zeros(max_size, max_size)
        )
        # Change detection flags
        self.register_buffer(
            'change_detected',
            torch.zeros(max_size, max_size, dtype=torch.bool)
        )
        
        # ── Observation Encoder ──
        # Encode raw grid cells to feature vectors
        self.cell_encoder = nn.Sequential(
            nn.Linear(10, 32),  # 10 colors → 32 dim
            nn.ReLU(),
            nn.Linear(32, feature_dim),
        )
        
        # Global step counter
        self.current_step = 0
        
        # History for causal tracking
        self.observation_history: List[SpatialObservation] = []
        
        self.to(device)
    
    def reset(self):
        """Reset all fields for new episode."""
        self.visit_count.zero_()
        self.exploration_mass.zero_()
        self.last_observation.zero_()
        self.last_visit_time.zero_()
        self.curiosity_field.zero_()
        self.change_detected.zero_()
        self.current_step = 0
        self.observation_history = []
    
    # ─────────────────────────────────────────────
    # Core Operations
    # ─────────────────────────────────────────────
    
    def encode_cell(self, color: int) -> Tensor:
        """Encode single cell color to feature vector."""
        one_hot = F.one_hot(torch.tensor(color), num_classes=10).float()
        one_hot = one_hot.to(self.device)
        return self.cell_encoder(one_hot)
    
    def encode_grid(self, grid: List[List[int]]) -> Tensor:
        """
        Encode full grid to spatial feature map.
        
        Args:
            grid: [H, W] integer grid (colors 0-9)
            
        Returns:
            [H, W, feature_dim] feature tensor
        """
        h, w = len(grid), len(grid[0])
        features = torch.zeros(h, w, self.feature_dim, device=self.device)
        
        for r in range(h):
            for c in range(w):
                features[r, c] = self.encode_cell(grid[r][c])
        
        return features
    
    def observe(
        self,
        position: Tuple[int, int],
        local_view: List[List[int]],
        global_grid: Optional[List[List[int]]] = None,
    ) -> Dict[str, Any]:
        """
        Record observation at current position.
        
        This updates:
        1. Exploration mass (we've been here)
        2. Last observation (what we saw)
        3. Curiosity ice (if something changed)
        
        Args:
            position: (row, col) current agent position
            local_view: Local grid visible to agent
            global_grid: Full grid if available (for ARC-AGI-3)
            
        Returns:
            Dict with change_detected, curiosity_delta, etc.
        """
        r, c = position
        self.current_step += 1
        
        # Encode current observation
        current_features = self.encode_grid(local_view)
        h, w = len(local_view), len(local_view[0])
        
        result = {
            'position': position,
            'step': self.current_step,
            'change_detected': False,
            'curiosity_delta': 0.0,
            'new_ice_positions': [],
        }
        
        # Update for each cell in local view
        for dr in range(h):
            for dc in range(w):
                gr, gc = r + dr - h//2, c + dc - w//2  # Grid coords
                
                # Bounds check
                if 0 <= gr < self.max_size and 0 <= gc < self.max_size:
                    cell_features = current_features[dr, dc]
                    
                    # Check if we've been here before
                    if self.visit_count[gr, gc] > 0:
                        # Compare to last observation
                        old_features = self.last_observation[gr, gc]
                        delta = (cell_features - old_features).norm().item()
                        
                        # Change detection: spawn ice if different
                        if delta > self.curiosity_threshold:
                            ice_amount = delta * SPATIAL_MEMORY_CONFIG['change_ice_multiplier']
                            self.curiosity_field[gr, gc] += ice_amount
                            self.change_detected[gr, gc] = True
                            result['change_detected'] = True
                            result['curiosity_delta'] += ice_amount
                            result['new_ice_positions'].append((gr, gc))
                    
                    # Update exploration mass
                    self.visit_count[gr, gc] += 1
                    self.exploration_mass[gr, gc] += SPATIAL_MEMORY_CONFIG['visit_mass_gain']
                    self.last_observation[gr, gc] = cell_features
                    self.last_visit_time[gr, gc] = self.current_step
        
        # Record in history
        obs = SpatialObservation(
            position=position,
            features=current_features[h//2, w//2].clone(),
            timestamp=self.current_step,
            raw_grid=local_view,
        )
        self.observation_history.append(obs)
        
        return result
    
    def detect_anomalies(
        self,
        grid: List[List[int]],
        top_k: int = 5,
    ) -> List[Tuple[Tuple[int, int], float, str]]:
        """
        Detect anomalies in grid — these become "ice".
        
        Anomalies include:
        - Color contrasts (red cell in sea of blue)
        - Isolated objects
        - Pattern breaks
        - Previously unseen configurations
        
        Args:
            grid: Full grid to analyze
            top_k: Return top K anomalies
            
        Returns:
            List of (position, score, reason)
        """
        h, w = len(grid), len(grid[0])
        anomaly_scores = torch.zeros(h, w, device=self.device)
        reasons = {}
        
        # 1. Color contrast — cells different from neighbors
        for r in range(h):
            for c in range(w):
                color = grid[r][c]
                neighbor_colors = []
                
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        neighbor_colors.append(grid[nr][nc])
                
                if neighbor_colors:
                    contrast = sum(1 for nc in neighbor_colors if nc != color)
                    if contrast >= len(neighbor_colors) * 0.75:
                        anomaly_scores[r, c] += contrast
                        reasons[(r, c)] = reasons.get((r, c), []) + ["color_contrast"]
        
        # 2. Non-background cells (assuming 0 is background)
        for r in range(h):
            for c in range(w):
                if grid[r][c] != 0:
                    anomaly_scores[r, c] += 0.5
                    if "object" not in reasons.get((r, c), []):
                        reasons[(r, c)] = reasons.get((r, c), []) + ["object"]
        
        # 3. Isolated cells (single non-zero cell surrounded by zeros)
        for r in range(h):
            for c in range(w):
                if grid[r][c] != 0:
                    neighbors = []
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            neighbors.append(grid[nr][nc])
                    
                    if all(n == 0 for n in neighbors):
                        anomaly_scores[r, c] += 2.0
                        reasons[(r, c)] = reasons.get((r, c), []) + ["isolated"]
        
        # 4. Update curiosity field with anomalies
        self.curiosity_field[:h, :w] += anomaly_scores
        
        # Get top K
        flat_scores = anomaly_scores.flatten()
        top_k_actual = min(top_k, (flat_scores > 0).sum().item())
        
        if top_k_actual == 0:
            return []
        
        top_indices = torch.topk(flat_scores, top_k_actual).indices
        results = []
        
        for idx in top_indices:
            r, c = idx.item() // w, idx.item() % w
            score = anomaly_scores[r, c].item()
            reason = "_".join(reasons.get((r, c), ["unknown"]))
            results.append(((r, c), score, reason))
        
        return results
    
    def step_decay(self):
        """Apply decay to fields (call once per step)."""
        self.exploration_mass *= SPATIAL_MEMORY_CONFIG['mass_decay']
        self.curiosity_field *= SPATIAL_MEMORY_CONFIG['ice_decay']
    
    # ─────────────────────────────────────────────
    # Navigation
    # ─────────────────────────────────────────────
    
    def get_navigation_target(
        self,
        current_pos: Tuple[int, int],
        grid_size: Tuple[int, int],
    ) -> Optional[NavigationTarget]:
        """
        Find the best navigation target based on curiosity field.
        
        Prefers:
        1. Highest curiosity (ice) locations
        2. Unvisited areas
        3. Reachable within reasonable distance
        
        Returns:
            NavigationTarget or None if nowhere interesting
        """
        h, w = grid_size
        curiosity = self.curiosity_field[:h, :w].clone()
        
        # Boost unvisited areas
        unvisited_boost = (self.visit_count[:h, :w] == 0).float() * 0.5
        curiosity += unvisited_boost
        
        # Find peak
        if curiosity.max() <= 0:
            return None
        
        flat_idx = curiosity.argmax().item()
        target_r, target_c = flat_idx // w, flat_idx % w
        
        # Plan path via mass gradient
        path = self._plan_path_via_mass(
            current_pos,
            (target_r, target_c),
            grid_size,
        )
        
        reason = "high_curiosity"
        if self.change_detected[target_r, target_c]:
            reason = "change_detected"
        elif self.visit_count[target_r, target_c] == 0:
            reason = "unexplored"
        
        return NavigationTarget(
            position=(target_r, target_c),
            curiosity_score=curiosity[target_r, target_c].item(),
            reason=reason,
            path=path,
        )
    
    def _plan_path_via_mass(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        grid_size: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """
        Plan path from start to goal using exploration mass as guidance.
        
        Uses A* with mass as heuristic bonus — prefers paths through
        known (high-mass) territory when possible.
        """
        h, w = grid_size
        
        # Simple A* implementation
        from heapq import heappush, heappop
        
        def heuristic(pos: Tuple[int, int]) -> float:
            # Manhattan distance + bonus for known territory
            dist = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
            mass_bonus = self.exploration_mass[pos[0], pos[1]].item() * 0.1
            return dist - mass_bonus  # Lower is better
        
        open_set = [(heuristic(start), 0, start, [start])]
        visited = set()
        
        while open_set:
            _, cost, current, path = heappop(open_set)
            
            if current == goal:
                return path
            
            if current in visited:
                continue
            visited.add(current)
            
            # Explore neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = current[0] + dr, current[1] + dc
                
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                    new_cost = cost + 1
                    new_path = path + [(nr, nc)]
                    heappush(open_set, (
                        new_cost + heuristic((nr, nc)),
                        new_cost,
                        (nr, nc),
                        new_path,
                    ))
            
            # Limit search
            if len(visited) > 1000:
                break
        
        # No path found — return direct line
        return [start, goal]
    
    def get_next_action(
        self,
        current_pos: Tuple[int, int],
        target: NavigationTarget,
    ) -> int:
        """
        Get action to move toward target.
        
        Actions: 0=up, 1=down, 2=left, 3=right
        """
        if not target.path or len(target.path) < 2:
            # Random exploration
            return torch.randint(0, 4, (1,)).item()
        
        next_pos = target.path[1]  # First step after current
        dr = next_pos[0] - current_pos[0]
        dc = next_pos[1] - current_pos[1]
        
        if dr < 0:
            return 0  # Up
        elif dr > 0:
            return 1  # Down
        elif dc < 0:
            return 2  # Left
        else:
            return 3  # Right
    
    # ─────────────────────────────────────────────
    # Visualization
    # ─────────────────────────────────────────────
    
    def visualize(
        self,
        grid_size: Tuple[int, int],
        current_pos: Optional[Tuple[int, int]] = None,
    ) -> str:
        """
        Create ASCII visualization of the dual fields.
        
        Returns:
            String visualization
        """
        h, w = grid_size
        lines = []
        
        lines.append("┌" + "─" * (w * 2 + 1) + "┐")
        
        for r in range(h):
            row = "│ "
            for c in range(w):
                mass = self.exploration_mass[r, c].item()
                ice = self.curiosity_field[r, c].item()
                
                if current_pos and (r, c) == current_pos:
                    row += "@ "  # Agent
                elif ice > 1.0:
                    row += "🧊"  # High ice
                elif ice > 0.5:
                    row += "❄ "  # Medium ice
                elif self.change_detected[r, c]:
                    row += "! "  # Change detected
                elif mass > 2.0:
                    row += "● "  # High mass (well explored)
                elif mass > 0.5:
                    row += "○ "  # Medium mass
                elif self.visit_count[r, c] > 0:
                    row += "· "  # Visited once
                else:
                    row += "  "  # Unknown
            row += "│"
            lines.append(row)
        
        lines.append("└" + "─" * (w * 2 + 1) + "┘")
        
        lines.append("")
        lines.append("Legend: @ agent, 🧊 high ice, ❄ ice, ! change, ● explored, · visited")
        
        return "\n".join(lines)
    
    def get_stats(self, grid_size: Tuple[int, int]) -> Dict[str, Any]:
        """Get statistics about current memory state."""
        h, w = grid_size
        
        return {
            'total_visits': self.visit_count[:h, :w].sum().item(),
            'unique_visited': (self.visit_count[:h, :w] > 0).sum().item(),
            'total_cells': h * w,
            'coverage': (self.visit_count[:h, :w] > 0).sum().item() / (h * w),
            'total_exploration_mass': self.exploration_mass[:h, :w].sum().item(),
            'total_curiosity_ice': self.curiosity_field[:h, :w].sum().item(),
            'ice_peaks': (self.curiosity_field[:h, :w] > 1.0).sum().item(),
            'changes_detected': self.change_detected[:h, :w].sum().item(),
            'step': self.current_step,
        }


# ─────────────────────────────────────────────
# Enhanced ARC Agent with Spatial Memory
# ─────────────────────────────────────────────

class SpatialARCAgent:
    """
    ARC-AGI-3 Agent with Spatial Memory.
    
    Uses dual-field navigation:
    - Exploration mass for pathfinding
    - Curiosity ice for goal selection
    """
    
    def __init__(
        self,
        max_size: int = 30,
        device: str = 'cpu',
    ):
        self.spatial_memory = SpatialMemory(
            max_size=max_size,
            device=device,
        )
        self.current_pos = (0, 0)
        self.current_target: Optional[NavigationTarget] = None
        self.action_history: List[int] = []
        self.device = device
    
    def reset(self, initial_pos: Tuple[int, int] = (0, 0)):
        """Reset agent for new episode."""
        self.spatial_memory.reset()
        self.current_pos = initial_pos
        self.current_target = None
        self.action_history = []
    
    def observe_and_decide(
        self,
        grid: List[List[int]],
        position: Tuple[int, int],
    ) -> int:
        """
        Full observe → plan → decide cycle.
        
        Returns:
            Action to take (0=up, 1=down, 2=left, 3=right)
        """
        self.current_pos = position
        grid_size = (len(grid), len(grid[0]))
        
        # 1. Record observation (updates mass and checks for changes)
        obs_result = self.spatial_memory.observe(
            position=position,
            local_view=grid,  # Full grid as local view for now
            global_grid=grid,
        )
        
        # 2. Detect anomalies (creates ice)
        anomalies = self.spatial_memory.detect_anomalies(grid)
        
        # 3. Decay fields
        self.spatial_memory.step_decay()
        
        # 4. Get/update navigation target
        if (self.current_target is None or 
            self.current_pos == self.current_target.position or
            obs_result['change_detected']):
            
            self.current_target = self.spatial_memory.get_navigation_target(
                self.current_pos,
                grid_size,
            )
        
        # 5. Decide action
        if self.current_target:
            action = self.spatial_memory.get_next_action(
                self.current_pos,
                self.current_target,
            )
        else:
            # Random exploration if nothing interesting
            action = torch.randint(0, 4, (1,)).item()
        
        self.action_history.append(action)
        return action
    
    def get_visualization(self, grid_size: Tuple[int, int]) -> str:
        """Get current spatial memory visualization."""
        return self.spatial_memory.visualize(grid_size, self.current_pos)
    
    def get_stats(self, grid_size: Tuple[int, int]) -> Dict[str, Any]:
        """Get agent statistics."""
        stats = self.spatial_memory.get_stats(grid_size)
        stats['actions_taken'] = len(self.action_history)
        stats['current_target'] = (
            self.current_target.position if self.current_target else None
        )
        return stats


# ─────────────────────────────────────────────
# Demo / Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Spatial Memory Demo: Ice & Water Navigation")
    print("=" * 60)
    
    # Create agent
    agent = SpatialARCAgent(max_size=10, device='cpu')
    agent.reset()
    
    # Simple test grid with anomalies
    grid = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # Object at (2,2-3)
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 2, 0, 0],  # Isolated cell at (6,7)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 3, 3],  # Corner object
    ]
    
    print("\n  Initial grid (0=empty, 1-9=colors):")
    for row in grid:
        print("  " + " ".join(str(c) for c in row))
    
    print("\n  Running 10 steps of exploration...")
    
    pos = (0, 0)
    for step in range(10):
        action = agent.observe_and_decide(grid, pos)
        
        # Move
        dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
        new_r = max(0, min(9, pos[0] + dr))
        new_c = max(0, min(9, pos[1] + dc))
        pos = (new_r, new_c)
        
        if step % 3 == 0:
            print(f"\n  Step {step + 1}: pos={pos}, action={['up','down','left','right'][action]}")
            print(agent.get_visualization((10, 10)))
    
    print("\n  Final Statistics:")
    stats = agent.get_stats((10, 10))
    for k, v in stats.items():
        print(f"    {k}: {v}")
    
    print("\n  ✓ Spatial Memory demo complete")
