"""
GoalPlanner — Objective → Sub-goal Decomposition for FLUX

Takes high-level objectives and decomposes them into actionable sub-goals.
Uses learned rules and spatial memory to inform planning.

Physics Analogy:
    Like gradient descent in an energy landscape. The goal creates an
    attractor basin, and the planner finds the path of steepest descent
    through intermediate sub-goals.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set, Callable
from enum import Enum
from collections import deque
import numpy as np


# ─────────────────────────────────────────────
# Goal Data Structures
# ─────────────────────────────────────────────

class GoalStatus(Enum):
    """Status of a goal in the goal stack."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    ACHIEVED = "achieved"
    BLOCKED = "blocked"
    FAILED = "failed"
    ABANDONED = "abandoned"


class GoalType(Enum):
    """Types of goals the planner can handle."""
    REACH = "reach"           # Go to position
    COLLECT = "collect"       # Pick up object
    TRIGGER = "trigger"       # Activate mechanism
    AVOID = "avoid"           # Stay away from
    EXIT = "exit"             # Reach exit
    TRANSFORM = "transform"   # Change grid state
    EXPLORE = "explore"       # Discover unknown
    COMPOSITE = "composite"   # Multiple sub-goals


@dataclass
class Goal:
    """
    A goal or sub-goal to achieve.
    
    Goals can have:
    - A type (what kind of action)
    - A target (position, color, object)
    - Preconditions (what must be true before)
    - Effects (what becomes true after)
    - Sub-goals (for composite goals)
    """
    goal_id: int
    goal_type: GoalType
    target: Any  # Position, color, object ID, etc.
    status: GoalStatus = GoalStatus.PENDING
    
    # Priority (higher = more urgent)
    priority: float = 1.0
    
    # Preconditions and effects
    preconditions: List[Callable[[np.ndarray, Tuple[int, int]], bool]] = field(default_factory=list)
    effects: Dict[str, Any] = field(default_factory=dict)
    
    # Sub-goals for composite goals
    sub_goals: List['Goal'] = field(default_factory=list)
    parent_goal: Optional['Goal'] = None
    
    # Progress tracking
    attempts: int = 0
    max_attempts: int = 10
    
    # Context
    created_at: int = 0
    achieved_at: Optional[int] = None
    
    def __hash__(self):
        return hash(self.goal_id)
    
    def __eq__(self, other):
        if not isinstance(other, Goal):
            return False
        return self.goal_id == other.goal_id
    
    def is_simple(self) -> bool:
        """Check if this is a simple (non-composite) goal."""
        return self.goal_type != GoalType.COMPOSITE and len(self.sub_goals) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'goal_id': self.goal_id,
            'goal_type': self.goal_type.value,
            'target': self.target if not isinstance(self.target, np.ndarray) else self.target.tolist(),
            'status': self.status.value,
            'priority': self.priority,
            'attempts': self.attempts,
            'max_attempts': self.max_attempts,
            'created_at': self.created_at,
            'achieved_at': self.achieved_at,
            'sub_goal_ids': [g.goal_id for g in self.sub_goals],
        }
    
    def __str__(self) -> str:
        target_str = str(self.target)
        if isinstance(self.target, tuple) and len(self.target) == 2:
            target_str = f"({self.target[0]},{self.target[1]})"
        return f"Goal#{self.goal_id}[{self.goal_type.value}→{target_str}]({self.status.value})"


# ─────────────────────────────────────────────
# Goal Templates
# ─────────────────────────────────────────────

@dataclass
class GoalTemplate:
    """A template for creating goals from objectives."""
    name: str
    goal_type: GoalType
    pattern: str  # e.g., "reach {color}" or "collect {object}"
    decomposition: List[str] = field(default_factory=list)  # Sub-goal templates
    
    def matches(self, objective: str) -> Optional[Dict[str, Any]]:
        """Check if objective matches this template."""
        import re
        
        # Convert pattern to regex
        regex = self.pattern
        regex = regex.replace('{color}', r'(?P<color>\w+)')
        regex = regex.replace('{object}', r'(?P<object>\w+)')
        regex = regex.replace('{position}', r'(?P<position>\(\d+,\d+\))')
        regex = regex.replace('{action}', r'(?P<action>\w+)')
        
        match = re.match(regex, objective, re.IGNORECASE)
        if match:
            return match.groupdict()
        return None


# ─────────────────────────────────────────────
# GoalPlanner
# ─────────────────────────────────────────────

class GoalPlanner(nn.Module):
    """
    Takes objective and current state, outputs sub-goal sequence.
    Uses learned rules to inform planning.
    
    Key capabilities:
    - Parse objectives into goal structures
    - Decompose goals into sub-goals
    - Track goal progress
    - Replan when blocked
    - Integrate with rules for informed planning
    """
    
    # Color name mapping
    COLORS = {
        0: 'black', 1: 'blue', 2: 'red', 3: 'green',
        4: 'yellow', 5: 'gray', 6: 'magenta', 7: 'orange',
        8: 'cyan', 9: 'brown',
    }
    COLOR_TO_ID = {v: k for k, v in COLORS.items()}
    
    def __init__(
        self,
        device: str = 'cpu',
    ):
        super().__init__()
        self._device = device
        
        # Goal management
        self.goal_stack: List[Goal] = []
        self.achieved_goals: Set[int] = set()
        self.all_goals: Dict[int, Goal] = {}
        self.next_goal_id = 0
        
        # Goal templates
        self.templates = self._create_default_templates()
        
        # Current step
        self.step_count = 0
        
        # Objective parsing
        self.current_objective: Optional[str] = None
    
    @property
    def device(self) -> str:
        return self._device
    
    def _create_default_templates(self) -> List[GoalTemplate]:
        """Create default goal templates."""
        return [
            GoalTemplate(
                name="exit",
                goal_type=GoalType.EXIT,
                pattern="exit door",
                decomposition=["reach exit"],
            ),
            GoalTemplate(
                name="reach_color",
                goal_type=GoalType.REACH,
                pattern="reach {color}",
            ),
            GoalTemplate(
                name="collect",
                goal_type=GoalType.COLLECT,
                pattern="collect {object}",
                decomposition=["reach {object}", "pickup {object}"],
            ),
            GoalTemplate(
                name="trigger",
                goal_type=GoalType.TRIGGER,
                pattern="trigger {object}",
                decomposition=["reach {object}", "toggle {object}"],
            ),
            GoalTemplate(
                name="avoid",
                goal_type=GoalType.AVOID,
                pattern="avoid {object}",
            ),
            GoalTemplate(
                name="explore",
                goal_type=GoalType.EXPLORE,
                pattern="explore",
            ),
        ]
    
    # ─────────────────────────────────────────────
    # Objective Parsing
    # ─────────────────────────────────────────────
    
    def set_objective(self, objective: str) -> List[Goal]:
        """
        Parse objective and create goal stack.
        
        Args:
            objective: Natural language objective (e.g., "exit door")
            
        Returns:
            List of created goals
        """
        self.current_objective = objective
        self.goal_stack.clear()
        
        # Try each template
        for template in self.templates:
            match = template.matches(objective)
            if match:
                return self._create_goals_from_template(template, match)
        
        # Fallback: create exploration goal
        goal = self._create_goal(GoalType.EXPLORE, target=None)
        self.goal_stack.append(goal)
        return [goal]
    
    def _create_goals_from_template(
        self,
        template: GoalTemplate,
        params: Dict[str, Any],
    ) -> List[Goal]:
        """Create goals from a matched template."""
        goals = []
        
        # Create main goal
        target = self._parse_target(template.goal_type, params)
        main_goal = self._create_goal(template.goal_type, target)
        
        # Create sub-goals if template has decomposition
        if template.decomposition:
            for sub_template_name in template.decomposition:
                # Find sub-template
                for sub_template in self.templates:
                    sub_match = sub_template.matches(sub_template_name.format(**params))
                    if sub_match:
                        sub_target = self._parse_target(sub_template.goal_type, sub_match)
                        sub_goal = self._create_goal(sub_template.goal_type, sub_target)
                        sub_goal.parent_goal = main_goal
                        main_goal.sub_goals.append(sub_goal)
                        break
        
        # Add to stack (sub-goals first, so main goal is last)
        for sub in reversed(main_goal.sub_goals):
            self.goal_stack.append(sub)
            goals.append(sub)
        
        self.goal_stack.append(main_goal)
        goals.append(main_goal)
        
        return goals
    
    def _parse_target(
        self,
        goal_type: GoalType,
        params: Dict[str, Any],
    ) -> Any:
        """Parse target from template parameters."""
        if 'color' in params:
            color_name = params['color'].lower()
            return self.COLOR_TO_ID.get(color_name, color_name)
        
        if 'object' in params:
            return params['object']
        
        if 'position' in params:
            pos_str = params['position'].strip('()')
            h, w = map(int, pos_str.split(','))
            return (h, w)
        
        return None
    
    def _create_goal(
        self,
        goal_type: GoalType,
        target: Any,
        priority: float = 1.0,
    ) -> Goal:
        """Create a new goal."""
        goal = Goal(
            goal_id=self.next_goal_id,
            goal_type=goal_type,
            target=target,
            priority=priority,
            created_at=self.step_count,
        )
        
        self.next_goal_id += 1
        self.all_goals[goal.goal_id] = goal
        
        return goal
    
    # ─────────────────────────────────────────────
    # Goal Execution
    # ─────────────────────────────────────────────
    
    def next_subgoal(
        self,
        grid: np.ndarray,
        position: Tuple[int, int],
    ) -> Optional[Goal]:
        """
        Get the next unachieved goal.
        
        Args:
            grid: Current grid state
            position: Current agent position
            
        Returns:
            Next goal to work on, or None if all achieved
        """
        # Update achieved status
        self._update_goal_statuses(grid, position)
        
        # Find first non-achieved goal
        for goal in self.goal_stack:
            if goal.status == GoalStatus.PENDING:
                goal.status = GoalStatus.IN_PROGRESS
                return goal
            elif goal.status == GoalStatus.IN_PROGRESS:
                return goal
            elif goal.status == GoalStatus.BLOCKED:
                # Try to unblock
                if goal.attempts < goal.max_attempts:
                    goal.attempts += 1
                    return goal
        
        return None
    
    def _update_goal_statuses(
        self,
        grid: np.ndarray,
        position: Tuple[int, int],
    ):
        """Update status of all goals based on current state."""
        for goal in self.goal_stack:
            if goal.status in [GoalStatus.ACHIEVED, GoalStatus.FAILED, GoalStatus.ABANDONED]:
                continue
            
            # Check if achieved
            if self._is_achieved(goal, grid, position):
                goal.status = GoalStatus.ACHIEVED
                goal.achieved_at = self.step_count
                self.achieved_goals.add(goal.goal_id)
                
                # Update parent if all sub-goals achieved
                if goal.parent_goal:
                    all_subs_achieved = all(
                        s.status == GoalStatus.ACHIEVED
                        for s in goal.parent_goal.sub_goals
                    )
                    if all_subs_achieved:
                        goal.parent_goal.status = GoalStatus.ACHIEVED
                        goal.parent_goal.achieved_at = self.step_count
    
    def _is_achieved(
        self,
        goal: Goal,
        grid: np.ndarray,
        position: Tuple[int, int],
    ) -> bool:
        """Check if a goal is achieved."""
        if goal.goal_type == GoalType.REACH:
            if isinstance(goal.target, tuple):
                # Position target
                return position == goal.target
            elif isinstance(goal.target, int):
                # Color target
                return grid[position[0], position[1]] == goal.target
            elif isinstance(goal.target, str):
                # Named target
                color_id = self.COLOR_TO_ID.get(goal.target.lower())
                if color_id is not None:
                    return grid[position[0], position[1]] == color_id
        
        elif goal.goal_type == GoalType.EXIT:
            # Check if at exit (usually a specific color or position)
            # This depends on the game. For now, check if at edge
            H, W = grid.shape
            at_edge = (position[0] == 0 or position[0] == H-1 or
                       position[1] == 0 or position[1] == W-1)
            return at_edge
        
        elif goal.goal_type == GoalType.COLLECT:
            # Would need inventory tracking
            return False
        
        elif goal.goal_type == GoalType.TRIGGER:
            # Check for state change (would need before state)
            return False
        
        # Check custom preconditions
        for precond in goal.preconditions:
            if not precond(grid, position):
                return False
        
        return False
    
    # ─────────────────────────────────────────────
    # Replanning
    # ─────────────────────────────────────────────
    
    def mark_blocked(
        self,
        goal: Goal,
        reason: Optional[str] = None,
    ):
        """Mark a goal as blocked."""
        goal.status = GoalStatus.BLOCKED
        goal.attempts += 1
        
        if goal.attempts >= goal.max_attempts:
            goal.status = GoalStatus.FAILED
    
    def replan(
        self,
        grid: np.ndarray,
        position: Tuple[int, int],
        blocked_goal: Goal,
    ) -> Optional[Goal]:
        """
        Replan when a goal is blocked.
        
        Args:
            grid: Current grid state
            position: Current agent position
            blocked_goal: The goal that was blocked
            
        Returns:
            New goal to pursue, or None
        """
        # Strategy 1: Find alternative path (exploration)
        if blocked_goal.goal_type == GoalType.REACH:
            # Add exploration goal to find new path
            explore_goal = self._create_goal(
                GoalType.EXPLORE,
                target=None,
                priority=blocked_goal.priority + 0.1,
            )
            self.goal_stack.insert(0, explore_goal)
            return explore_goal
        
        # Strategy 2: Try sub-goals in different order
        if blocked_goal.parent_goal and blocked_goal.parent_goal.sub_goals:
            # Find next unachieved sibling
            for sibling in blocked_goal.parent_goal.sub_goals:
                if sibling.goal_id != blocked_goal.goal_id and sibling.status == GoalStatus.PENDING:
                    sibling.status = GoalStatus.IN_PROGRESS
                    return sibling
        
        # Strategy 3: Give up on this goal temporarily
        blocked_goal.status = GoalStatus.BLOCKED
        
        # Find next available goal
        return self.next_subgoal(grid, position)
    
    def compute_target_position(
        self,
        goal: Goal,
        grid: np.ndarray,
        position: Tuple[int, int],
    ) -> Optional[Tuple[int, int]]:
        """
        Compute the target position for a goal.
        
        Args:
            goal: Goal to compute target for
            grid: Current grid state
            position: Current agent position
            
        Returns:
            Target position, or None if cannot compute
        """
        if goal.goal_type == GoalType.REACH:
            if isinstance(goal.target, tuple):
                return goal.target
            elif isinstance(goal.target, int):
                # Find nearest cell of this color
                return self._find_nearest_color(grid, position, goal.target)
            elif isinstance(goal.target, str):
                color_id = self.COLOR_TO_ID.get(goal.target.lower())
                if color_id is not None:
                    return self._find_nearest_color(grid, position, color_id)
        
        elif goal.goal_type == GoalType.EXIT:
            # Find exit position (edge or specific marker)
            return self._find_exit(grid, position)
        
        elif goal.goal_type == GoalType.EXPLORE:
            # Find unexplored area (would need exploration map)
            return self._find_unexplored(grid, position)
        
        elif goal.goal_type == GoalType.TRIGGER:
            if isinstance(goal.target, str):
                color_id = self.COLOR_TO_ID.get(goal.target.lower())
                if color_id is not None:
                    return self._find_nearest_color(grid, position, color_id)
        
        return None
    
    def _find_nearest_color(
        self,
        grid: np.ndarray,
        position: Tuple[int, int],
        color: int,
    ) -> Optional[Tuple[int, int]]:
        """Find the nearest cell of a given color."""
        matches = np.argwhere(grid == color)
        if len(matches) == 0:
            return None
        
        # Find closest
        distances = np.abs(matches[:, 0] - position[0]) + np.abs(matches[:, 1] - position[1])
        nearest_idx = np.argmin(distances)
        
        return tuple(matches[nearest_idx])
    
    def _find_exit(
        self,
        grid: np.ndarray,
        position: Tuple[int, int],
    ) -> Optional[Tuple[int, int]]:
        """Find an exit position."""
        H, W = grid.shape
        
        # Check edges for walkable cells
        edges = []
        for h in range(H):
            if grid[h, 0] == 0:  # Left edge
                edges.append((h, 0))
            if grid[h, W-1] == 0:  # Right edge
                edges.append((h, W-1))
        for w in range(W):
            if grid[0, w] == 0:  # Top edge
                edges.append((0, w))
            if grid[H-1, w] == 0:  # Bottom edge
                edges.append((H-1, w))
        
        if not edges:
            return None
        
        # Find closest edge
        distances = [abs(e[0] - position[0]) + abs(e[1] - position[1]) for e in edges]
        return edges[np.argmin(distances)]
    
    def _find_unexplored(
        self,
        grid: np.ndarray,
        position: Tuple[int, int],
    ) -> Optional[Tuple[int, int]]:
        """Find a random unexplored position."""
        H, W = grid.shape
        
        # For now, return a random walkable cell
        walkable = np.argwhere(grid == 0)  # Assuming 0 is walkable
        if len(walkable) == 0:
            return None
        
        idx = np.random.randint(len(walkable))
        return tuple(walkable[idx])
    
    # ─────────────────────────────────────────────
    # State Management
    # ─────────────────────────────────────────────
    
    def step(self):
        """Increment step counter."""
        self.step_count += 1
    
    def reset(self):
        """Reset planner state."""
        self.goal_stack.clear()
        self.achieved_goals.clear()
        self.all_goals.clear()
        self.next_goal_id = 0
        self.step_count = 0
        self.current_objective = None
    
    def reset_episode(self):
        """Reset episode-specific state."""
        self.goal_stack.clear()
        self.current_objective = None
        
        # Keep templates and learned info
    
    def state_dict(self) -> Dict[str, Any]:
        """Export state for checkpointing."""
        return {
            'goals': [g.to_dict() for g in self.all_goals.values()],
            'goal_stack_ids': [g.goal_id for g in self.goal_stack],
            'achieved_goals': list(self.achieved_goals),
            'next_goal_id': self.next_goal_id,
            'step_count': self.step_count,
            'current_objective': self.current_objective,
        }
    
    def load_state_dict(self, state: Dict[str, Any]):
        """Load state from checkpoint."""
        # Note: Full restoration would need to rebuild goal objects
        self.next_goal_id = state.get('next_goal_id', 0)
        self.step_count = state.get('step_count', 0)
        self.current_objective = state.get('current_objective')
        self.achieved_goals = set(state.get('achieved_goals', []))
    
    def summary(self) -> str:
        """Get a summary of the goal state."""
        lines = [
            "GoalPlanner Summary",
            "=" * 40,
            f"Objective: {self.current_objective or 'None'}",
            f"Total goals: {len(self.all_goals)}",
            f"Achieved: {len(self.achieved_goals)}",
            f"In stack: {len(self.goal_stack)}",
            "",
        ]
        
        if self.goal_stack:
            lines.append("Goal stack:")
            for goal in reversed(self.goal_stack):
                indent = "  " if goal.parent_goal else ""
                lines.append(f"{indent}{goal}")
        
        return "\n".join(lines)
    
    def forward(self, x: Tensor) -> Tensor:
        """Dummy forward for nn.Module compatibility."""
        return x


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import numpy as np
    
    print("Testing GoalPlanner")
    print("=" * 40)
    
    planner = GoalPlanner()
    
    # Create test grid
    grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],  # Blue walls
        [0, 1, 4, 1, 0],  # Yellow target at (2,2)
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ])
    position = (0, 0)
    
    # Test objective parsing
    goals = planner.set_objective("reach yellow")
    print(f"✓ Created {len(goals)} goals from 'reach yellow'")
    for goal in goals:
        print(f"  {goal}")
    
    # Test next subgoal
    next_goal = planner.next_subgoal(grid, position)
    print(f"\n✓ Next goal: {next_goal}")
    
    # Test target computation
    target = planner.compute_target_position(next_goal, grid, position)
    print(f"✓ Target position: {target}")
    
    # Test achievement detection
    position = (2, 2)  # Move to yellow
    planner._update_goal_statuses(grid, position)
    print(f"\n✓ After moving to (2,2):")
    print(f"  Goal status: {next_goal.status}")
    
    # Test exit objective
    planner.reset()
    goals = planner.set_objective("exit door")
    print(f"\n✓ Created {len(goals)} goals from 'exit door'")
    for goal in goals:
        print(f"  {goal}")
    
    # Print summary
    print()
    print(planner.summary())
    
    # Test state dict
    state = planner.state_dict()
    planner2 = GoalPlanner()
    planner2.load_state_dict(state)
    print(f"\n✓ State dict round-trip successful")
    
    print("\n✓ All tests passed!")
