"""
Game Loop — ARC API Integration for FLUX Unified Agent

Provides the main game loop connecting FLUXUnifiedAgent to the ARC-AGI-3 API.

Physics Analogy:
    The game loop is the time evolution of the system - each step advances
    the field state based on agent actions and environmental response.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import requests
import time

try:
    from .unified_agent import FLUXUnifiedAgent
    from .strategies import get_control_scheme
except ImportError:
    from unified_agent import FLUXUnifiedAgent
    from strategies import get_control_scheme


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

ARC_ROOT_URL = "https://three.arcprize.org"

ACTION_MAP = {
    1: "ACTION1",  # UP
    2: "ACTION2",  # DOWN
    3: "ACTION3",  # LEFT
    4: "ACTION4",  # RIGHT
    5: "ACTION5",  # INTERACT/SPACE
    6: "ACTION6",  # CLICK
    7: "ACTION7",  # UNDO
}


# ─────────────────────────────────────────────
# Game Result
# ─────────────────────────────────────────────

@dataclass
class GameResult:
    """Result of playing a game."""
    game_id: str
    card_id: str
    final_state: str
    final_score: float
    actions_taken: int
    level_wins: int
    agent_stats: Dict[str, Any]
    spatial_snapshot: Optional[Dict[str, Any]] = None
    history: List[Dict] = field(default_factory=list)
    error: Optional[str] = None


# ─────────────────────────────────────────────
# ARC Session
# ─────────────────────────────────────────────

class ARCSession:
    """HTTP session for ARC API."""
    
    def __init__(self, api_key: str):
        """Initialize session with API key."""
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "X-API-Key": api_key,
            "Accept": "application/json",
        })
    
    def get(self, endpoint: str) -> Dict[str, Any]:
        """GET request."""
        url = f"{ARC_ROOT_URL}{endpoint}"
        response = self.session.get(url)
        if response.status_code != 200:
            raise RuntimeError(f"GET {endpoint} failed: {response.text}")
        return response.json()
    
    def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """POST request."""
        url = f"{ARC_ROOT_URL}{endpoint}"
        response = self.session.post(url, json=data)
        if response.status_code != 200:
            raise RuntimeError(f"POST {endpoint} failed: {response.text}")
        return response.json()
    
    def list_games(self) -> List[Dict[str, Any]]:
        """List available games."""
        return self.get("/api/games")
    
    def open_scorecard(self, tags: List[str] = None) -> str:
        """Open a new scorecard."""
        tags = tags or ["flux-unified"]
        result = self.post("/api/scorecard/open", {"tags": tags})
        return result["card_id"]
    
    def close_scorecard(self, card_id: str):
        """Close a scorecard."""
        try:
            self.post("/api/scorecard/close", {"card_id": card_id})
        except Exception:
            pass  # Ignore close errors
    
    def reset_game(self, game_id: str, card_id: str) -> Dict[str, Any]:
        """Reset game to initial state."""
        return self.post("/api/cmd/RESET", {
            "game_id": game_id,
            "card_id": card_id,
        })
    
    def take_action(
        self,
        game_id: str,
        card_id: str,
        guid: str,
        action: int,
        action_data: Dict[str, Any] = None,
        reasoning: str = "",
    ) -> Dict[str, Any]:
        """Take an action in the game."""
        endpoint_name = ACTION_MAP.get(action, f"ACTION{action}")
        
        request_data = {
            "game_id": game_id,
            "card_id": card_id,
            "guid": guid,
        }
        
        # Add click coordinates if provided
        if action_data:
            request_data.update(action_data)
        
        return self.post(f"/api/cmd/{endpoint_name}", request_data)


# ─────────────────────────────────────────────
# Main Game Loop
# ─────────────────────────────────────────────

def play_game_unified(
    agent: FLUXUnifiedAgent,
    game_id: str,
    api_key: str,
    max_actions: int = 100,
    verbose: bool = True,
    tags: List[str] = None,
) -> GameResult:
    """
    Play an ARC game with the unified agent.
    
    Args:
        agent: FLUXUnifiedAgent instance
        game_id: Game identifier (e.g., "ls20")
        api_key: ARC API key
        max_actions: Maximum actions before stopping
        verbose: Print progress
        tags: Tags for scorecard
        
    Returns:
        GameResult with outcome and stats
    """
    session = ARCSession(api_key)
    tags = tags or ["flux-unified", game_id]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Playing: {game_id}")
        print(f"{'='*60}")
    
    # Open scorecard
    try:
        card_id = session.open_scorecard(tags)
        if verbose:
            print(f"Scorecard: {card_id}")
    except Exception as e:
        return GameResult(
            game_id=game_id,
            card_id="",
            final_state="ERROR",
            final_score=0,
            actions_taken=0,
            level_wins=0,
            agent_stats={},
            error=str(e),
        )
    
    # Reset game
    try:
        game_data = session.reset_game(game_id, card_id)
        guid = game_data["guid"]
        state = game_data["state"]
        score = game_data.get("score", 0)
        frame = game_data.get("frame", [[]])
        available_actions = game_data.get("available_actions", [1, 2, 3, 4, 5])
    except Exception as e:
        session.close_scorecard(card_id)
        return GameResult(
            game_id=game_id,
            card_id=card_id,
            final_state="ERROR",
            final_score=0,
            actions_taken=0,
            level_wins=0,
            agent_stats={},
            error=f"Reset failed: {e}",
        )
    
    # Initialize agent
    # Convert available_actions to list of ints if needed
    if available_actions and hasattr(available_actions[0], 'value'):
        available_actions = [a.value for a in available_actions]
    
    agent.start_game(game_id, frame, available_actions)
    
    if verbose:
        print(f"Initial state: {state}, Score: {score}")
        print(f"Control scheme: {agent.control_scheme}")
        print(f"Available: {available_actions}")
    
    # Play loop
    actions_taken = 0
    level_wins = 0
    last_action = None
    history = []
    
    while state == "NOT_FINISHED" and actions_taken < max_actions:
        # Update available actions
        if available_actions and hasattr(available_actions[0], 'value'):
            available_actions = [a.value for a in available_actions]
        
        try:
            # Agent step
            action, action_data, reasoning = agent.step(
                frame=frame,
                available_actions=available_actions,
                last_action=last_action,
            )
            
            if verbose and actions_taken % 10 == 0:
                action_name = ACTION_MAP.get(action, f"ACTION{action}")
                print(f"\nStep {actions_taken}: action={action_name}, pos={agent.current_position}")
            
        except Exception as e:
            if verbose:
                print(f"✗ Agent error at step {actions_taken}: {e}")
                import traceback
                traceback.print_exc()
            break
        
        # Take action
        try:
            game_data = session.take_action(
                game_id=game_id,
                card_id=card_id,
                guid=guid,
                action=action,
                action_data=action_data,
                reasoning=reasoning[:100] if reasoning else "",
            )
            
            state = game_data["state"]
            new_score = game_data.get("score", score)
            frame = game_data.get("frame", frame)
            new_available = game_data.get("available_actions", available_actions)
            
            # Convert if needed
            if new_available and hasattr(new_available[0], 'value'):
                available_actions = [a.value for a in new_available]
            else:
                available_actions = new_available
            
            # Track wins
            if new_score > score:
                level_wins += 1
                if verbose:
                    print(f"  ↑ Level completed! Score: {score} -> {new_score}")
            
            score = new_score
            last_action = action
            actions_taken += 1
            
            # Record history
            history.append({
                "step": actions_taken,
                "action": action,
                "action_data": action_data,
                "state": state,
                "score": score,
            })
            
        except Exception as e:
            if verbose:
                print(f"✗ Action failed: {e}")
            break
    
    # Close scorecard
    session.close_scorecard(card_id)
    
    # Capture spatial memory snapshot
    spatial_snapshot = None
    if agent.spatial_memory is not None:
        try:
            grid_size = 64
            if agent.last_grid:
                grid_size = max(len(agent.last_grid), len(agent.last_grid[0]) if agent.last_grid else 64)
            
            spatial_snapshot = {
                'exploration_mass': agent.spatial_memory.exploration_mass[:grid_size, :grid_size].clone().cpu().numpy(),
                'curiosity_field': agent.spatial_memory.curiosity_field[:grid_size, :grid_size].clone().cpu().numpy(),
                'grid_size': grid_size,
                'final_pos': agent.current_position,
            }
        except Exception as e:
            print(f"Warning: Could not capture spatial snapshot: {e}")
    
    # Build result
    result = GameResult(
        game_id=game_id,
        card_id=card_id,
        final_state=state,
        final_score=score,
        actions_taken=actions_taken,
        level_wins=level_wins,
        agent_stats=agent.get_stats(),
        spatial_snapshot=spatial_snapshot,
        history=history,
    )
    
    if verbose:
        print(f"\n{'-'*40}")
        print(f"Result: {state}")
        print(f"Score: {score}")
        print(f"Actions: {actions_taken}")
        print(f"Levels won: {level_wins}")
        print(f"Scorecard: {ARC_ROOT_URL}/scorecards/{card_id}")
    
    return result


# ─────────────────────────────────────────────
# Multi-Game Runner
# ─────────────────────────────────────────────

def run_games(
    agent: FLUXUnifiedAgent,
    game_ids: List[str],
    api_key: str,
    max_actions: int = 50,
    verbose: bool = True,
) -> List[GameResult]:
    """
    Run agent on multiple games.
    
    Args:
        agent: FLUXUnifiedAgent instance
        game_ids: List of game IDs to play
        api_key: ARC API key
        max_actions: Max actions per game
        verbose: Print progress
        
    Returns:
        List of GameResult
    """
    results = []
    
    for game_id in game_ids:
        # Reset agent between games
        agent.reset()
        
        result = play_game_unified(
            agent=agent,
            game_id=game_id,
            api_key=api_key,
            max_actions=max_actions,
            verbose=verbose,
        )
        results.append(result)
        
        # Small delay between games
        time.sleep(1)
    
    return results


def print_results_summary(results: List[GameResult]):
    """Print summary of game results."""
    print(f"\n{'='*60}")
    print("FLUX UNIFIED AGENT RESULTS")
    print(f"{'='*60}")
    
    print(f"\n{'Game':<15} {'State':<15} {'Score':<10} {'Actions':<10} {'Levels':<10}")
    print("-" * 60)
    
    total_score = 0
    total_actions = 0
    total_wins = 0
    
    for r in results:
        print(f"{r.game_id:<15} {r.final_state:<15} {r.final_score:<10.1f} {r.actions_taken:<10} {r.level_wins:<10}")
        total_score += r.final_score
        total_actions += r.actions_taken
        total_wins += r.level_wins
    
    print("-" * 60)
    print(f"{'TOTAL':<15} {'':<15} {total_score:<10.1f} {total_actions:<10} {total_wins:<10}")
    
    # Effectiveness stats
    print(f"\nAgent Stats:")
    if results:
        last_stats = results[-1].agent_stats
        for key, val in last_stats.items():
            print(f"  {key}: {val}")
