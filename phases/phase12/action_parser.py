"""
action_parser.py — Phase 12: Parse LLM Responses to Game Actions

Extracts action decisions from LLM text output with multiple fallback strategies.
Handles various response formats from different LLMs.
"""

import re
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass


# ─────────────────────────────────────────────
# ARC-AGI-3 Action Definitions
# ─────────────────────────────────────────────

# Standard action mapping
ACTION_MAP = {
    0: 'RESET',
    1: 'UP',
    2: 'DOWN',
    3: 'LEFT',
    4: 'RIGHT',
    5: 'INTERACT',
    6: 'CLICK',
    7: 'UNDO',
}

# Reverse mapping for parsing
ACTION_NAME_TO_ID = {v: k for k, v in ACTION_MAP.items()}

# Common aliases
ACTION_ALIASES = {
    # Direction aliases
    'UP': 1, 'U': 1, 'NORTH': 1, 'N': 1, 'W': 1, '↑': 1, 'MOVE_UP': 1,
    'DOWN': 2, 'D': 2, 'SOUTH': 2, 'S': 2, '↓': 2, 'MOVE_DOWN': 2,
    'LEFT': 3, 'L': 3, 'WEST': 3, 'A': 3, '←': 3, 'MOVE_LEFT': 3,
    'RIGHT': 4, 'R': 4, 'EAST': 4, 'D': 4, '→': 4, 'MOVE_RIGHT': 4,
    
    # Action aliases
    'INTERACT': 5, 'SELECT': 5, 'USE': 5, 'ACTIVATE': 5, 'SPACE': 5, 'E': 5,
    'CLICK': 6, 'PRESS': 6, 'TOUCH': 6,
    'UNDO': 7, 'BACK': 7, 'CANCEL': 7, 'Z': 7,
    'RESET': 0, 'RESTART': 0,
    
    # Numbered actions
    'ACTION1': 1, 'ACTION2': 2, 'ACTION3': 3, 'ACTION4': 4,
    'ACTION5': 5, 'ACTION6': 6, 'ACTION7': 7,
    'ACTION_1': 1, 'ACTION_2': 2, 'ACTION_3': 3, 'ACTION_4': 4,
    'ACTION_5': 5, 'ACTION_6': 6, 'ACTION_7': 7,
}


# ─────────────────────────────────────────────
# Parse Result
# ─────────────────────────────────────────────

@dataclass
class ParseResult:
    """Result of action parsing."""
    action_id: int
    action_name: str
    confidence: float  # 0-1 confidence in parse
    method: str        # How the action was extracted
    click_coords: Optional[Tuple[int, int]] = None  # For CLICK action
    raw_match: Optional[str] = None  # The matched text
    
    def __str__(self) -> str:
        s = f"Action: {self.action_name} (id={self.action_id}, conf={self.confidence:.2f}, via {self.method})"
        if self.click_coords:
            s += f" @ coords {self.click_coords}"
        return s


# ─────────────────────────────────────────────
# Action Parser
# ─────────────────────────────────────────────

class ActionParser:
    """
    Parse LLM responses to extract game actions.
    
    Uses multiple strategies in order:
    1. Explicit "ACTION: X" format
    2. "I choose X" or "I'll do X" patterns
    3. Action word at end of response
    4. Any action word in response
    5. Fallback to random/default
    """
    
    def __init__(
        self,
        default_action: int = 1,
        verbose: bool = False,
    ):
        """
        Initialize parser.
        
        Args:
            default_action: Action to use if parsing fails
            verbose: Print parsing details
        """
        self.default_action = default_action
        self.verbose = verbose
        
        # Pre-compile regex patterns
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for action extraction."""
        # Pattern 1: Explicit ACTION: format
        self.pattern_explicit = re.compile(
            r'ACTION\s*[:\-=]\s*(\w+)',
            re.IGNORECASE
        )
        
        # Pattern 2: "I choose/select/will X"
        self.pattern_choice = re.compile(
            r"(?:i'?ll?|i\s+will|i\s+choose|i\s+select|my\s+action\s+is|"
            r"i\s+(?:should|would|must))\s+(?:go\s+)?(\w+)",
            re.IGNORECASE
        )
        
        # Pattern 3: "Move X" or "Go X"
        self.pattern_movement = re.compile(
            r"(?:move|go|head|proceed)\s+(\w+)",
            re.IGNORECASE
        )
        
        # Pattern 4: Directional at sentence end
        self.pattern_end = re.compile(
            r"(?:UP|DOWN|LEFT|RIGHT|INTERACT|CLICK|UNDO)\s*[.!?]?\s*$",
            re.IGNORECASE
        )
        
        # Pattern 5: Click with coordinates
        self.pattern_click = re.compile(
            r"CLICK\s*(?:at)?\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?",
            re.IGNORECASE
        )
    
    def parse(
        self,
        response: str,
        available_actions: Optional[List[int]] = None,
    ) -> ParseResult:
        """
        Parse LLM response to extract action.
        
        Args:
            response: LLM text response
            available_actions: List of available action IDs (for validation)
            
        Returns:
            ParseResult with extracted action
        """
        if not response:
            return self._fallback(available_actions, "empty_response")
        
        response_upper = response.upper()
        
        # Strategy 1: Explicit ACTION: format
        result = self._try_explicit(response_upper, available_actions)
        if result:
            return result
        
        # Strategy 2: Click with coordinates
        result = self._try_click_coords(response, available_actions)
        if result:
            return result
        
        # Strategy 3: Choice patterns
        result = self._try_choice(response_upper, available_actions)
        if result:
            return result
        
        # Strategy 4: Movement patterns
        result = self._try_movement(response_upper, available_actions)
        if result:
            return result
        
        # Strategy 5: Action at end of response
        result = self._try_end_action(response_upper, available_actions)
        if result:
            return result
        
        # Strategy 6: Any action word in response (lowest confidence)
        result = self._try_any_action(response_upper, available_actions)
        if result:
            return result
        
        # Fallback
        return self._fallback(available_actions, "no_match")
    
    def _try_explicit(
        self,
        response: str,
        available: Optional[List[int]],
    ) -> Optional[ParseResult]:
        """Try to extract explicit ACTION: format."""
        match = self.pattern_explicit.search(response)
        if match:
            action_text = match.group(1).upper()
            action_id = ACTION_ALIASES.get(action_text)
            
            if action_id is not None:
                if self._is_available(action_id, available):
                    return ParseResult(
                        action_id=action_id,
                        action_name=ACTION_MAP.get(action_id, f'ACTION{action_id}'),
                        confidence=0.95,
                        method='explicit',
                        raw_match=match.group(0),
                    )
        return None
    
    def _try_click_coords(
        self,
        response: str,
        available: Optional[List[int]],
    ) -> Optional[ParseResult]:
        """Try to extract CLICK with coordinates."""
        match = self.pattern_click.search(response)
        if match:
            if self._is_available(6, available):  # CLICK action
                x, y = int(match.group(1)), int(match.group(2))
                return ParseResult(
                    action_id=6,
                    action_name='CLICK',
                    confidence=0.9,
                    method='click_coords',
                    click_coords=(x, y),
                    raw_match=match.group(0),
                )
        return None
    
    def _try_choice(
        self,
        response: str,
        available: Optional[List[int]],
    ) -> Optional[ParseResult]:
        """Try to extract "I choose X" patterns."""
        match = self.pattern_choice.search(response)
        if match:
            action_text = match.group(1).upper()
            action_id = ACTION_ALIASES.get(action_text)
            
            if action_id is not None:
                if self._is_available(action_id, available):
                    return ParseResult(
                        action_id=action_id,
                        action_name=ACTION_MAP.get(action_id, f'ACTION{action_id}'),
                        confidence=0.85,
                        method='choice',
                        raw_match=match.group(0),
                    )
        return None
    
    def _try_movement(
        self,
        response: str,
        available: Optional[List[int]],
    ) -> Optional[ParseResult]:
        """Try to extract "move X" patterns."""
        match = self.pattern_movement.search(response)
        if match:
            action_text = match.group(1).upper()
            action_id = ACTION_ALIASES.get(action_text)
            
            if action_id is not None:
                if self._is_available(action_id, available):
                    return ParseResult(
                        action_id=action_id,
                        action_name=ACTION_MAP.get(action_id, f'ACTION{action_id}'),
                        confidence=0.8,
                        method='movement',
                        raw_match=match.group(0),
                    )
        return None
    
    def _try_end_action(
        self,
        response: str,
        available: Optional[List[int]],
    ) -> Optional[ParseResult]:
        """Try to find action at end of response."""
        match = self.pattern_end.search(response)
        if match:
            action_text = match.group(0).strip(' .!?').upper()
            action_id = ACTION_ALIASES.get(action_text)
            
            if action_id is not None:
                if self._is_available(action_id, available):
                    return ParseResult(
                        action_id=action_id,
                        action_name=ACTION_MAP.get(action_id, f'ACTION{action_id}'),
                        confidence=0.75,
                        method='end_action',
                        raw_match=match.group(0),
                    )
        return None
    
    def _try_any_action(
        self,
        response: str,
        available: Optional[List[int]],
    ) -> Optional[ParseResult]:
        """Try to find any action word in response."""
        # Check for directional words (prioritize based on prominence)
        for direction in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            if direction in response:
                action_id = ACTION_ALIASES[direction]
                if self._is_available(action_id, available):
                    return ParseResult(
                        action_id=action_id,
                        action_name=direction,
                        confidence=0.5,
                        method='any_match',
                        raw_match=direction,
                    )
        
        # Check for other actions
        for action_name in ['INTERACT', 'CLICK', 'UNDO']:
            if action_name in response:
                action_id = ACTION_ALIASES[action_name]
                if self._is_available(action_id, available):
                    return ParseResult(
                        action_id=action_id,
                        action_name=action_name,
                        confidence=0.4,
                        method='any_match',
                        raw_match=action_name,
                    )
        
        return None
    
    def _is_available(
        self,
        action_id: int,
        available: Optional[List[int]],
    ) -> bool:
        """Check if action is in available list."""
        if available is None:
            return True
        return action_id in available
    
    def _fallback(
        self,
        available: Optional[List[int]],
        reason: str,
    ) -> ParseResult:
        """Return fallback action."""
        import random
        
        if available and len(available) > 0:
            # Filter to directional actions if possible
            directional = [a for a in available if a in [1, 2, 3, 4]]
            if directional:
                action_id = random.choice(directional)
            else:
                action_id = random.choice(available)
        else:
            action_id = self.default_action
        
        if self.verbose:
            print(f"  [ActionParser] Fallback due to: {reason}")
        
        return ParseResult(
            action_id=action_id,
            action_name=ACTION_MAP.get(action_id, f'ACTION{action_id}'),
            confidence=0.1,
            method=f'fallback_{reason}',
        )
    
    def parse_batch(
        self,
        responses: List[str],
        available_actions: Optional[List[int]] = None,
    ) -> List[ParseResult]:
        """Parse multiple responses."""
        return [self.parse(r, available_actions) for r in responses]


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────

def format_available_actions(actions: List[int]) -> str:
    """Format available actions for LLM prompt."""
    names = [ACTION_MAP.get(a, f'ACTION{a}') for a in actions]
    return ', '.join(names)


def action_to_delta(action_id: int) -> Tuple[int, int]:
    """
    Convert action ID to position delta.
    
    Args:
        action_id: Action identifier (1-4 for movement)
        
    Returns:
        (delta_row, delta_col) tuple
    """
    deltas = {
        1: (-1, 0),   # UP: row decreases
        2: (1, 0),    # DOWN: row increases
        3: (0, -1),   # LEFT: col decreases
        4: (0, 1),    # RIGHT: col increases
    }
    return deltas.get(action_id, (0, 0))


def get_action_description(action_id: int) -> str:
    """Get human-readable action description."""
    descriptions = {
        0: "Reset the game to initial state",
        1: "Move up (decrease row)",
        2: "Move down (increase row)",
        3: "Move left (decrease column)",
        4: "Move right (increase column)",
        5: "Interact with current cell (select/toggle/execute)",
        6: "Click at specific coordinates",
        7: "Undo the last action",
    }
    return descriptions.get(action_id, f"Unknown action {action_id}")


# ─────────────────────────────────────────────
# Module Exports
# ─────────────────────────────────────────────

__all__ = [
    'ActionParser',
    'ParseResult',
    'ACTION_MAP',
    'ACTION_NAME_TO_ID',
    'ACTION_ALIASES',
    'format_available_actions',
    'action_to_delta',
    'get_action_description',
]
