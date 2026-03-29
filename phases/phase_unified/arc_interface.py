"""
ARC-AGI-3 Interface — Standard Action & State Definitions for FLUX

This module provides the official ARC-AGI-3 interface definitions:
- GameAction enum: RESET, ACTION1-7 with semantic meanings
- GameState enum: NOT_FINISHED, WIN, GAME_OVER
- Frame parsing and available_actions tracking
- Coordinate handling for ACTION6 (click at x,y)

Based on ARC-AGI-3 documentation: https://docs.arcprize.org

Physics analogy:
    Actions are forces applied to the game state field.
    Each action creates ripples that propagate through the grid.
"""

from enum import IntEnum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np


# ─────────────────────────────────────────────
# Game Actions (ARC-AGI-3 Standard)
# ─────────────────────────────────────────────

class GameAction(IntEnum):
    """
    Standard ARC-AGI-3 action interface.
    
    All games implement these 7 core actions + RESET:
    - RESET: Initialize or restart game/level
    - ACTION1-4: Simple directional (up/down/left/right semantic)
    - ACTION5: Generic interact (select, rotate, execute, etc.)
    - ACTION6: Complex action requiring x,y coordinates (0-63)
    - ACTION7: Undo action
    
    Human keybindings:
    - WASD + Space: W=ACTION1, S=ACTION2, A=ACTION3, D=ACTION4, Space=ACTION5
    - Arrow keys: ↑=ACTION1, ↓=ACTION2, ←=ACTION3, →=ACTION4, F=ACTION5
    - Mouse click: ACTION6
    - Ctrl/Cmd+Z: ACTION7
    """
    RESET = 0
    ACTION1 = 1   # Up (W / ↑)
    ACTION2 = 2   # Down (S / ↓)
    ACTION3 = 3   # Left (A / ←)
    ACTION4 = 4   # Right (D / →)
    ACTION5 = 5   # Interact (Space / F)
    ACTION6 = 6   # Click at (x, y) — requires coordinates
    ACTION7 = 7   # Undo (Ctrl+Z / Cmd+Z)
    
    @property
    def is_directional(self) -> bool:
        """Check if this is a directional action (1-4)."""
        return self in (GameAction.ACTION1, GameAction.ACTION2, 
                       GameAction.ACTION3, GameAction.ACTION4)
    
    @property
    def is_complex(self) -> bool:
        """Check if this action requires coordinates (ACTION6)."""
        return self == GameAction.ACTION6
    
    @property
    def semantic_name(self) -> str:
        """Human-readable semantic meaning."""
        names = {
            GameAction.RESET: 'reset',
            GameAction.ACTION1: 'up',
            GameAction.ACTION2: 'down',
            GameAction.ACTION3: 'left',
            GameAction.ACTION4: 'right',
            GameAction.ACTION5: 'interact',
            GameAction.ACTION6: 'click',
            GameAction.ACTION7: 'undo',
        }
        return names.get(self, 'unknown')
    
    @property
    def delta(self) -> Tuple[int, int]:
        """
        Get (dy, dx) movement delta for directional actions.
        Returns (0, 0) for non-directional actions.
        """
        deltas = {
            GameAction.ACTION1: (-1, 0),  # Up: row decreases
            GameAction.ACTION2: (1, 0),   # Down: row increases
            GameAction.ACTION3: (0, -1),  # Left: col decreases
            GameAction.ACTION4: (0, 1),   # Right: col increases
        }
        return deltas.get(self, (0, 0))
    
    @classmethod
    def from_name(cls, name: str) -> 'GameAction':
        """
        Create action from name (case-insensitive).
        
        Accepts: 'ACTION1', 'action1', 'up', 'UP', 'RESET', etc.
        """
        name_upper = name.upper()
        
        # Direct enum name
        if hasattr(cls, name_upper):
            return getattr(cls, name_upper)
        
        # Semantic name mapping
        semantic_map = {
            'UP': cls.ACTION1,
            'DOWN': cls.ACTION2,
            'LEFT': cls.ACTION3,
            'RIGHT': cls.ACTION4,
            'INTERACT': cls.ACTION5,
            'SELECT': cls.ACTION5,
            'SPACE': cls.ACTION5,
            'CLICK': cls.ACTION6,
            'UNDO': cls.ACTION7,
        }
        
        if name_upper in semantic_map:
            return semantic_map[name_upper]
        
        raise ValueError(f"Unknown action: {name}")
    
    @classmethod
    def directional_actions(cls) -> List['GameAction']:
        """Get list of directional actions."""
        return [cls.ACTION1, cls.ACTION2, cls.ACTION3, cls.ACTION4]
    
    @classmethod
    def simple_actions(cls) -> List['GameAction']:
        """Get list of simple (non-coordinate) actions."""
        return [cls.ACTION1, cls.ACTION2, cls.ACTION3, cls.ACTION4, 
                cls.ACTION5, cls.ACTION7]


# ─────────────────────────────────────────────
# Game State
# ─────────────────────────────────────────────

class GameState(IntEnum):
    """
    ARC-AGI-3 game state enumeration.
    
    States:
    - NOT_FINISHED: Game is active, awaiting next action
    - WIN: Objective completed successfully
    - GAME_OVER: Game terminated (max actions reached or failure)
    """
    NOT_FINISHED = 0
    WIN = 1
    GAME_OVER = 2
    
    @classmethod
    def from_string(cls, s: str) -> 'GameState':
        """Parse state from API response string."""
        mapping = {
            'NOT_FINISHED': cls.NOT_FINISHED,
            'WIN': cls.WIN,
            'GAME_OVER': cls.GAME_OVER,
        }
        return mapping.get(s.upper(), cls.NOT_FINISHED)
    
    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self in (GameState.WIN, GameState.GAME_OVER)
    
    @property
    def is_success(self) -> bool:
        """Check if this is a successful terminal state."""
        return self == GameState.WIN


# ─────────────────────────────────────────────
# Action with Coordinates (for ACTION6)
# ─────────────────────────────────────────────

@dataclass
class ActionCommand:
    """
    A complete action command including optional coordinates.
    
    For ACTION6 (click), x and y must be provided (0-63 range).
    For other actions, coordinates are ignored.
    
    Optional reasoning field for audit/research (≤16KB JSON).
    """
    action: GameAction
    x: Optional[int] = None
    y: Optional[int] = None
    reasoning: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        # Validate coordinates for ACTION6
        if self.action == GameAction.ACTION6:
            if self.x is None or self.y is None:
                raise ValueError("ACTION6 requires x and y coordinates")
            if not (0 <= self.x <= 63 and 0 <= self.y <= 63):
                raise ValueError(f"Coordinates must be 0-63, got ({self.x}, {self.y})")
    
    def to_api_dict(self, game_id: str, guid: str, card_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert to API request dictionary.
        
        Args:
            game_id: Game identifier
            guid: Session identifier from RESET
            card_id: Optional scorecard ID
            
        Returns:
            Dictionary ready for API request body
        """
        data = {
            'game_id': game_id,
            'guid': guid,
        }
        
        if card_id:
            data['card_id'] = card_id
        
        if self.action == GameAction.ACTION6:
            data['x'] = self.x
            data['y'] = self.y
        
        if self.reasoning:
            data['reasoning'] = self.reasoning
        
        return data
    
    @property
    def api_endpoint(self) -> str:
        """Get API endpoint for this action."""
        if self.action == GameAction.RESET:
            return '/api/cmd/RESET'
        return f'/api/cmd/ACTION{self.action.value}'


# ─────────────────────────────────────────────
# Frame Data (API Response)
# ─────────────────────────────────────────────

@dataclass
class GameFrame:
    """
    A single frame response from the ARC-AGI-3 API.
    
    Contains:
    - grid: Current game state as 2D array (max 64x64, values 0-15)
    - state: Current game state (NOT_FINISHED/WIN/GAME_OVER)
    - score: Current score
    - available_actions: Set of valid actions for this state
    - guid: Session identifier
    - levels_completed: Number of levels completed so far
    """
    grid: np.ndarray                      # [H, W] with values 0-15
    state: GameState
    score: int
    available_actions: Set[GameAction]
    guid: str
    levels_completed: int = 0
    action_count: int = 0
    game_id: Optional[str] = None
    
    @classmethod
    def from_api_response(cls, response: Dict[str, Any]) -> 'GameFrame':
        """
        Parse frame from API JSON response.
        
        API response structure:
        {
            "game_id": "ls20-xxxx",
            "frame": [[0,0,1,...], ...],  # 2D grid
            "state": "NOT_FINISHED",
            "score": 5,
            "guid": "abc-123",
            "available_actions": [1, 2, 3, 4, 5],
            "levels_completed": 0
        }
        """
        # Parse grid
        frame_data = response.get('frame', [[]])
        grid = np.array(frame_data, dtype=np.int32)
        
        # Parse state
        state_str = response.get('state', 'NOT_FINISHED')
        state = GameState.from_string(state_str)
        
        # Parse available actions
        action_ids = response.get('available_actions', [1, 2, 3, 4, 5])
        available = {GameAction(a) for a in action_ids if 0 <= a <= 7}
        
        return cls(
            grid=grid,
            state=state,
            score=response.get('score', 0),
            available_actions=available,
            guid=response.get('guid', ''),
            levels_completed=response.get('levels_completed', 0),
            action_count=response.get('action_count', 0),
            game_id=response.get('game_id'),
        )
    
    @property
    def height(self) -> int:
        """Grid height."""
        return self.grid.shape[0]
    
    @property
    def width(self) -> int:
        """Grid width."""
        return self.grid.shape[1]
    
    @property  
    def is_terminal(self) -> bool:
        """Check if game has ended."""
        return self.state.is_terminal
    
    def can_take(self, action: GameAction) -> bool:
        """Check if action is available in current state."""
        return action in self.available_actions


# ─────────────────────────────────────────────
# Action-Effect Mapping (for CausalTracker)
# ─────────────────────────────────────────────

# Map action IDs to legacy names for backward compatibility
ACTION_ID_TO_NAME = {
    0: 'reset',
    1: 'up',
    2: 'down', 
    3: 'left',
    4: 'right',
    5: 'interact',
    6: 'click',
    7: 'undo',
}

# Reverse mapping
ACTION_NAME_TO_ID = {v: k for k, v in ACTION_ID_TO_NAME.items()}

# Movement deltas for directional actions
ACTION_DELTAS = {
    1: (-1, 0),   # ACTION1/up: row decreases
    2: (1, 0),    # ACTION2/down: row increases
    3: (0, -1),   # ACTION3/left: col decreases
    4: (0, 1),    # ACTION4/right: col increases
}


def get_action_delta(action: int) -> Tuple[int, int]:
    """
    Get (dy, dx) movement delta for an action.
    
    Args:
        action: Action ID (1-4 for directional)
        
    Returns:
        (dy, dx) tuple, (0, 0) for non-directional
    """
    return ACTION_DELTAS.get(action, (0, 0))


def apply_action_to_position(
    position: Tuple[int, int],
    action: int,
    grid_shape: Tuple[int, int]
) -> Tuple[int, int]:
    """
    Apply directional action to position, respecting boundaries.
    
    Args:
        position: Current (row, col) position
        action: Action ID
        grid_shape: (height, width) of grid
        
    Returns:
        New (row, col) position after action
    """
    dy, dx = get_action_delta(action)
    new_row = max(0, min(position[0] + dy, grid_shape[0] - 1))
    new_col = max(0, min(position[1] + dx, grid_shape[1] - 1))
    return (new_row, new_col)


# ─────────────────────────────────────────────
# RHAE Scoring (Relative Human Action Efficiency)
# ─────────────────────────────────────────────

@dataclass
class LevelScore:
    """Score for a single level."""
    level_index: int
    ai_actions: int
    human_baseline_actions: int
    completed: bool = False
    
    @property
    def raw_score(self) -> float:
        """
        Calculate raw level score.
        
        Formula: (human_baseline / ai_actions)²
        Capped at 1.0 (can't score higher than human baseline)
        """
        if not self.completed or self.ai_actions == 0:
            return 0.0
        
        ratio = self.human_baseline_actions / self.ai_actions
        return min(1.0, ratio ** 2)
    
    @property
    def weight(self) -> int:
        """Level weight (1-indexed level number)."""
        return self.level_index + 1


@dataclass
class GameScore:
    """Aggregate score for a complete game."""
    game_id: str
    level_scores: List[LevelScore] = field(default_factory=list)
    
    @property
    def weighted_score(self) -> float:
        """
        Calculate weighted average of level scores.
        
        Higher levels are weighted more heavily (tutorial levels = low weight).
        """
        if not self.level_scores:
            return 0.0
        
        total_weighted = sum(ls.raw_score * ls.weight for ls in self.level_scores)
        total_weight = sum(ls.weight for ls in self.level_scores)
        
        return total_weighted / total_weight if total_weight > 0 else 0.0
    
    @property
    def completion_rate(self) -> float:
        """Fraction of levels completed."""
        if not self.level_scores:
            return 0.0
        return sum(1 for ls in self.level_scores if ls.completed) / len(self.level_scores)


# ─────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────

def grid_diff(before: np.ndarray, after: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Find all cells that changed between two grids.
    
    Args:
        before: Grid before action
        after: Grid after action
        
    Returns:
        List of (row, col, old_value, new_value) tuples
    """
    changes = []
    
    # Handle different shapes
    min_h = min(before.shape[0], after.shape[0])
    min_w = min(before.shape[1], after.shape[1])
    
    for r in range(min_h):
        for c in range(min_w):
            if before[r, c] != after[r, c]:
                changes.append((r, c, int(before[r, c]), int(after[r, c])))
    
    return changes


def find_agent_position(grid: np.ndarray, agent_color: int = 1) -> Optional[Tuple[int, int]]:
    """
    Find agent position in grid (assumes single agent cell).
    
    Args:
        grid: Game grid
        agent_color: Color value representing agent (default 1)
        
    Returns:
        (row, col) position or None if not found
    """
    positions = np.where(grid == agent_color)
    if len(positions[0]) > 0:
        return (int(positions[0][0]), int(positions[1][0]))
    return None


def find_goal_position(grid: np.ndarray, goal_color: int = 2) -> Optional[Tuple[int, int]]:
    """
    Find goal position in grid.
    
    Args:
        grid: Game grid
        goal_color: Color value representing goal (default 2)
        
    Returns:
        (row, col) position or None if not found
    """
    positions = np.where(grid == goal_color)
    if len(positions[0]) > 0:
        return (int(positions[0][0]), int(positions[1][0]))
    return None


# ─────────────────────────────────────────────
# Demo / Test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("ARC-AGI-3 Interface Module")
    print("=" * 50)
    
    # Test GameAction
    print("\n1. GameAction enum:")
    for action in GameAction:
        print(f"   {action.name}: value={action.value}, "
              f"semantic={action.semantic_name}, "
              f"directional={action.is_directional}, "
              f"complex={action.is_complex}")
    
    # Test from_name
    print("\n2. Action parsing:")
    for name in ['up', 'ACTION1', 'CLICK', 'undo', 'RESET']:
        action = GameAction.from_name(name)
        print(f"   '{name}' -> {action.name}")
    
    # Test GameState
    print("\n3. GameState enum:")
    for state in GameState:
        print(f"   {state.name}: terminal={state.is_terminal}, success={state.is_success}")
    
    # Test ActionCommand
    print("\n4. ActionCommand:")
    
    # Simple action
    cmd1 = ActionCommand(GameAction.ACTION1)
    print(f"   Simple: {cmd1.action.name}")
    
    # Complex action with coordinates
    cmd2 = ActionCommand(GameAction.ACTION6, x=10, y=20)
    api_dict = cmd2.to_api_dict('ls20-test', 'guid-123', 'card-456')
    print(f"   Complex: {cmd2.action.name} at ({cmd2.x}, {cmd2.y})")
    print(f"   API dict: {api_dict}")
    
    # Test grid_diff
    print("\n5. Grid diff:")
    grid1 = np.array([[0, 1, 0], [0, 0, 0], [2, 0, 0]])
    grid2 = np.array([[0, 0, 0], [0, 1, 0], [2, 0, 0]])
    changes = grid_diff(grid1, grid2)
    print(f"   Changes: {changes}")
    
    # Test position finding
    print("\n6. Position finding:")
    agent_pos = find_agent_position(grid2, agent_color=1)
    goal_pos = find_goal_position(grid2, goal_color=2)
    print(f"   Agent at: {agent_pos}")
    print(f"   Goal at: {goal_pos}")
    
    # Test RHAE scoring
    print("\n7. RHAE Scoring:")
    game_score = GameScore(
        game_id='ls20',
        level_scores=[
            LevelScore(0, ai_actions=10, human_baseline_actions=10, completed=True),
            LevelScore(1, ai_actions=20, human_baseline_actions=15, completed=True),
            LevelScore(2, ai_actions=50, human_baseline_actions=20, completed=True),
        ]
    )
    print(f"   Level scores: {[ls.raw_score for ls in game_score.level_scores]}")
    print(f"   Weighted game score: {game_score.weighted_score:.4f}")
    print(f"   Completion rate: {game_score.completion_rate:.1%}")
    
    print("\n" + "=" * 50)
    print("  ✓ All ARC-AGI-3 interface components ready")
