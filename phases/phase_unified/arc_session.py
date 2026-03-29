"""
ARC-AGI-3 Session Manager — API Client for FLUX

Handles all communication with the ARC-AGI-3 API:
- Session lifecycle (guid tracking, cookies)
- Scorecard management (open/close)
- Action execution with proper request formatting
- Local vs Online mode switching

Based on ARC-AGI-3 documentation: https://docs.arcprize.org

Physics analogy:
    The session is the conduit through which FLUX exerts forces
    on the game environment. Each action is a perturbation that
    propagates through the API and returns the resulting field state.
"""

import os
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from enum import Enum
import logging

# Try to import requests for online mode
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Try to import arc_agi toolkit for local mode
try:
    import arc_agi
    from arcengine import GameAction as ArcEngineAction, GameState as ArcEngineState
    ARC_AGI_AVAILABLE = True
except ImportError:
    ARC_AGI_AVAILABLE = False

# Local imports
try:
    from .arc_interface import (
        GameAction, GameState, GameFrame, ActionCommand,
        GameScore, LevelScore, ACTION_ID_TO_NAME
    )
except ImportError:
    from arc_interface import (
        GameAction, GameState, GameFrame, ActionCommand,
        GameScore, LevelScore, ACTION_ID_TO_NAME
    )


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

class OperationMode(Enum):
    """How to interact with ARC-AGI-3 environments."""
    OFFLINE = 'offline'    # Local engine, no API calls
    ONLINE = 'online'      # API calls, scorecards tracked
    COMPETITION = 'competition'  # Competition mode with restrictions


@dataclass
class SessionConfig:
    """Configuration for an ARC session."""
    operation_mode: OperationMode = OperationMode.OFFLINE
    api_key: Optional[str] = None
    api_url: str = 'https://three.arcprize.org'
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    save_recordings: bool = False
    recordings_dir: Path = Path('recordings')
    
    @classmethod
    def from_env(cls) -> 'SessionConfig':
        """Create config from environment variables."""
        mode_str = os.getenv('OPERATION_MODE', 'offline').lower()
        mode = OperationMode(mode_str) if mode_str in ['offline', 'online', 'competition'] else OperationMode.OFFLINE
        
        return cls(
            operation_mode=mode,
            api_key=os.getenv('ARC_API_KEY'),
            api_url=os.getenv('ARC_API_URL', 'https://three.arcprize.org'),
        )


# ─────────────────────────────────────────────
# Recording (for offline replay analysis)
# ─────────────────────────────────────────────

@dataclass
class RecordingEntry:
    """Single entry in a game recording."""
    timestamp: str
    action: int
    action_data: Dict[str, Any]
    state: str
    score: int
    grid: List[List[int]]
    reasoning: Optional[Dict[str, Any]] = None


@dataclass  
class GameRecording:
    """Complete recording of a game session."""
    game_id: str
    guid: str
    entries: List[RecordingEntry] = field(default_factory=list)
    
    def save(self, path: Path):
        """Save recording to JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            for entry in self.entries:
                line = {
                    'timestamp': entry.timestamp,
                    'action': entry.action,
                    'action_data': entry.action_data,
                    'state': entry.state,
                    'score': entry.score,
                    'grid': entry.grid,
                }
                if entry.reasoning:
                    line['reasoning'] = entry.reasoning
                f.write(json.dumps(line) + '\n')
    
    @classmethod
    def load(cls, path: Path) -> 'GameRecording':
        """Load recording from JSONL file."""
        entries = []
        game_id = path.stem.split('.')[0]
        guid = path.stem.split('.')[-2] if '.' in path.stem else ''
        
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line)
                entries.append(RecordingEntry(
                    timestamp=data['timestamp'],
                    action=data['action'],
                    action_data=data.get('action_data', {}),
                    state=data['state'],
                    score=data['score'],
                    grid=data['grid'],
                    reasoning=data.get('reasoning'),
                ))
        
        return cls(game_id=game_id, guid=guid, entries=entries)


# ─────────────────────────────────────────────
# Scorecard
# ─────────────────────────────────────────────

@dataclass
class Scorecard:
    """
    Scorecard for tracking agent performance.
    
    In online mode, this maps to the API scorecard.
    In offline mode, this is computed locally.
    """
    card_id: str
    tags: List[str] = field(default_factory=list)
    source_url: Optional[str] = None
    opaque: Optional[Dict[str, Any]] = None
    
    # Computed fields
    total_score: float = 0.0
    total_actions: int = 0
    games_played: int = 0
    games_completed: int = 0
    game_scores: Dict[str, GameScore] = field(default_factory=dict)
    
    # Timestamps
    opened_at: Optional[str] = None
    closed_at: Optional[str] = None
    
    @property
    def is_open(self) -> bool:
        """Check if scorecard is still open."""
        return self.closed_at is None
    
    def add_game_result(self, game_score: GameScore):
        """Add a completed game's score."""
        self.game_scores[game_score.game_id] = game_score
        self.games_played += 1
        if game_score.completion_rate > 0:
            self.games_completed += 1
        
        # Recompute total score as average of game scores
        if self.game_scores:
            self.total_score = sum(gs.weighted_score for gs in self.game_scores.values()) / len(self.game_scores)


# ─────────────────────────────────────────────  
# Session Manager
# ─────────────────────────────────────────────

class ARCSession:
    """
    Manages interaction with ARC-AGI-3 environments.
    
    Supports both local (offline) and API (online) modes.
    Handles session state, scorecard lifecycle, and action execution.
    
    Usage:
        # Offline mode
        session = ARCSession()
        env = session.make_env('ls20')
        frame = session.reset(env)
        frame = session.step(env, GameAction.ACTION1)
        scorecard = session.get_scorecard()
        
        # Online mode
        session = ARCSession(config=SessionConfig(
            operation_mode=OperationMode.ONLINE,
            api_key='your-api-key'
        ))
        session.open_scorecard(tags=['experiment'])
        # ... play games ...
        session.close_scorecard()
    """
    
    def __init__(
        self,
        config: Optional[SessionConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config or SessionConfig.from_env()
        self.logger = logger or logging.getLogger(__name__)
        
        # Session state
        self._session = None  # requests.Session for online mode
        self._arcade = None   # arc_agi.Arcade for offline mode
        self._environments: Dict[str, Any] = {}  # game_id -> env
        
        # Current session tracking
        self._current_scorecard: Optional[Scorecard] = None
        self._current_guid: Optional[str] = None
        self._current_game_id: Optional[str] = None
        
        # Recordings
        self._current_recording: Optional[GameRecording] = None
        
        # Initialize based on mode
        self._initialize()
    
    def _initialize(self):
        """Initialize session based on operation mode."""
        if self.config.operation_mode == OperationMode.OFFLINE:
            if not ARC_AGI_AVAILABLE:
                self.logger.warning("arc_agi not available, some features disabled")
            else:
                self._arcade = arc_agi.Arcade()
        else:
            if not REQUESTS_AVAILABLE:
                raise RuntimeError("requests library required for online mode")
            
            self._session = requests.Session()
            self._session.headers.update({
                'X-API-Key': self.config.api_key or '',
                'Accept': 'application/json',
                'Content-Type': 'application/json',
            })
    
    # ─────────────────────────────────────────
    # Environment Management
    # ─────────────────────────────────────────
    
    def list_games(self) -> List[str]:
        """Get list of available game IDs."""
        if self.config.operation_mode == OperationMode.OFFLINE:
            if self._arcade:
                return [g.game_id for g in self._arcade.list_games()]
            return []
        else:
            response = self._api_get('/api/games')
            return [g['game_id'] for g in response]
    
    def make_env(self, game_id: str, render_mode: Optional[str] = None) -> Any:
        """
        Create or get environment for a game.
        
        Args:
            game_id: Game identifier (e.g., 'ls20')
            render_mode: Optional render mode ('terminal', None)
            
        Returns:
            Environment handle
        """
        if game_id in self._environments:
            return self._environments[game_id]
        
        if self.config.operation_mode == OperationMode.OFFLINE:
            if self._arcade:
                env = self._arcade.make(game_id, render_mode=render_mode)
                self._environments[game_id] = env
                return env
            else:
                # Return a mock env for testing
                return {'game_id': game_id, 'mock': True}
        else:
            # Online mode doesn't pre-create envs
            self._environments[game_id] = {'game_id': game_id}
            return self._environments[game_id]
    
    # ─────────────────────────────────────────
    # Scorecard Management
    # ─────────────────────────────────────────
    
    def open_scorecard(
        self,
        tags: Optional[List[str]] = None,
        source_url: Optional[str] = None,
        opaque: Optional[Dict[str, Any]] = None,
    ) -> Scorecard:
        """
        Open a new scorecard for tracking performance.
        
        Args:
            tags: Optional tags for categorization
            source_url: Optional URL to code/notebook
            opaque: Optional arbitrary metadata (≤16KB)
            
        Returns:
            Created Scorecard
        """
        if self.config.operation_mode == OperationMode.OFFLINE:
            # Create local scorecard
            import uuid
            card_id = str(uuid.uuid4())
            self._current_scorecard = Scorecard(
                card_id=card_id,
                tags=tags or [],
                source_url=source_url,
                opaque=opaque,
                opened_at=time.strftime('%Y-%m-%dT%H:%M:%S'),
            )
        else:
            # API call to open scorecard
            payload = {}
            if tags:
                payload['tags'] = tags
            if source_url:
                payload['source_url'] = source_url
            if opaque:
                payload['opaque'] = opaque
            
            response = self._api_post('/api/scorecard/open', payload)
            self._current_scorecard = Scorecard(
                card_id=response['card_id'],
                tags=tags or [],
                source_url=source_url,
                opaque=opaque,
                opened_at=time.strftime('%Y-%m-%dT%H:%M:%S'),
            )
        
        return self._current_scorecard
    
    def close_scorecard(self) -> Optional[Scorecard]:
        """
        Close the current scorecard and finalize results.
        
        Returns:
            Finalized Scorecard with results
        """
        if not self._current_scorecard:
            return None
        
        self._current_scorecard.closed_at = time.strftime('%Y-%m-%dT%H:%M:%S')
        
        if self.config.operation_mode != OperationMode.OFFLINE:
            # API call to close scorecard
            response = self._api_post('/api/scorecard/close', {
                'card_id': self._current_scorecard.card_id
            })
            
            # Update scorecard with API response
            if 'total_score' in response:
                self._current_scorecard.total_score = response['total_score']
            if 'total_actions' in response:
                self._current_scorecard.total_actions = response['total_actions']
        
        scorecard = self._current_scorecard
        self._current_scorecard = None
        return scorecard
    
    def get_scorecard(self, card_id: Optional[str] = None) -> Optional[Scorecard]:
        """Get current or specified scorecard."""
        if card_id is None:
            return self._current_scorecard
        
        if self.config.operation_mode == OperationMode.OFFLINE:
            return self._current_scorecard if self._current_scorecard and self._current_scorecard.card_id == card_id else None
        else:
            response = self._api_get(f'/api/scorecard/{card_id}')
            return Scorecard(
                card_id=card_id,
                total_score=response.get('total_score', 0),
                total_actions=response.get('total_actions', 0),
            )
    
    # ─────────────────────────────────────────
    # Game Actions
    # ─────────────────────────────────────────
    
    def reset(self, env: Any, full_reset: bool = False) -> GameFrame:
        """
        Reset game to initial state.
        
        Args:
            env: Environment handle
            full_reset: If True, reset entire game. If False, reset current level only.
            
        Note: Two consecutive resets guarantee a full game reset.
        
        Returns:
            Initial game frame
        """
        game_id = env.get('game_id') if isinstance(env, dict) else getattr(env, 'game_id', None)
        self._current_game_id = game_id
        
        if self.config.operation_mode == OperationMode.OFFLINE:
            if self._arcade and not env.get('mock', False):
                obs = env.reset() if hasattr(env, 'reset') else None
                if obs:
                    return self._obs_to_frame(obs, game_id)
            
            # Mock frame for testing
            return GameFrame(
                grid=np.zeros((8, 8), dtype=np.int32),
                state=GameState.NOT_FINISHED,
                score=0,
                available_actions={GameAction.ACTION1, GameAction.ACTION2, 
                                  GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5},
                guid='mock-guid',
                game_id=game_id,
            )
        else:
            # API RESET
            payload = {
                'game_id': game_id,
            }
            if self._current_scorecard:
                payload['card_id'] = self._current_scorecard.card_id
            if self._current_guid and not full_reset:
                payload['guid'] = self._current_guid
            
            response = self._api_post('/api/cmd/RESET', payload)
            self._current_guid = response.get('guid')
            
            # Start recording
            if self.config.save_recordings:
                self._current_recording = GameRecording(
                    game_id=game_id,
                    guid=self._current_guid,
                )
            
            return GameFrame.from_api_response(response)
    
    def step(
        self,
        env: Any,
        action: GameAction,
        x: Optional[int] = None,
        y: Optional[int] = None,
        reasoning: Optional[Dict[str, Any]] = None,
    ) -> GameFrame:
        """
        Take an action in the game.
        
        Args:
            env: Environment handle
            action: GameAction to take
            x: X coordinate for ACTION6 (0-63)
            y: Y coordinate for ACTION6 (0-63)
            reasoning: Optional reasoning data for audit (≤16KB)
            
        Returns:
            Next game frame
        """
        game_id = env.get('game_id') if isinstance(env, dict) else getattr(env, 'game_id', None)
        
        # Validate ACTION6 coordinates
        if action == GameAction.ACTION6:
            if x is None or y is None:
                raise ValueError("ACTION6 requires x and y coordinates")
            if not (0 <= x <= 63 and 0 <= y <= 63):
                raise ValueError(f"Coordinates must be 0-63, got ({x}, {y})")
        
        if self.config.operation_mode == OperationMode.OFFLINE:
            if self._arcade and hasattr(env, 'step'):
                # Convert to arcengine action
                arc_action = ArcEngineAction(action.value) if ARC_AGI_AVAILABLE else None
                action_data = {'x': x, 'y': y} if action.is_complex else {}
                
                obs = env.step(arc_action, data=action_data) if arc_action else None
                if obs:
                    return self._obs_to_frame(obs, game_id)
            
            # Mock frame
            return GameFrame(
                grid=np.zeros((8, 8), dtype=np.int32),
                state=GameState.NOT_FINISHED,
                score=0,
                available_actions={GameAction.ACTION1, GameAction.ACTION2,
                                  GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5},
                guid='mock-guid',
                game_id=game_id,
            )
        else:
            # Build command
            cmd = ActionCommand(action, x=x, y=y, reasoning=reasoning)
            payload = cmd.to_api_dict(
                game_id=game_id,
                guid=self._current_guid,
                card_id=self._current_scorecard.card_id if self._current_scorecard else None,
            )
            
            response = self._api_post(cmd.api_endpoint, payload)
            frame = GameFrame.from_api_response(response)
            
            # Record
            if self._current_recording:
                self._current_recording.entries.append(RecordingEntry(
                    timestamp=time.strftime('%Y-%m-%dT%H:%M:%S.%f'),
                    action=action.value,
                    action_data={'x': x, 'y': y} if action.is_complex else {},
                    state=frame.state.name,
                    score=frame.score,
                    grid=frame.grid.tolist(),
                    reasoning=reasoning,
                ))
                
                # Save on terminal state
                if frame.is_terminal and self.config.save_recordings:
                    filename = f"{game_id}.{self._current_guid}.recording.jsonl"
                    self._current_recording.save(self.config.recordings_dir / filename)
            
            return frame
    
    # ─────────────────────────────────────────
    # Convenience Methods
    # ─────────────────────────────────────────
    
    def play_action(
        self,
        env: Any,
        action: str,
        x: Optional[int] = None,
        y: Optional[int] = None,
    ) -> GameFrame:
        """
        Play an action by name (convenience method).
        
        Args:
            env: Environment
            action: Action name ('up', 'down', 'click', etc.)
            x, y: Coordinates for click
            
        Returns:
            Next frame
        """
        game_action = GameAction.from_name(action)
        return self.step(env, game_action, x=x, y=y)
    
    def play_sequence(
        self,
        env: Any,
        actions: List[str],
    ) -> List[GameFrame]:
        """
        Play a sequence of actions.
        
        Args:
            env: Environment
            actions: List of action names
            
        Returns:
            List of resulting frames
        """
        frames = []
        for action in actions:
            frame = self.play_action(env, action)
            frames.append(frame)
            if frame.is_terminal:
                break
        return frames
    
    # ─────────────────────────────────────────
    # API Helpers
    # ─────────────────────────────────────────
    
    def _api_get(self, endpoint: str) -> Dict[str, Any]:
        """Make GET request to API."""
        url = f"{self.config.api_url}{endpoint}"
        
        for attempt in range(self.config.max_retries):
            try:
                response = self._session.get(url, timeout=self.config.timeout)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise
                time.sleep(self.config.retry_delay * (attempt + 1))
        
        return {}
    
    def _api_post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make POST request to API."""
        url = f"{self.config.api_url}{endpoint}"
        
        for attempt in range(self.config.max_retries):
            try:
                response = self._session.post(url, json=payload, timeout=self.config.timeout)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise
                time.sleep(self.config.retry_delay * (attempt + 1))
        
        return {}
    
    def _obs_to_frame(self, obs: Any, game_id: str) -> GameFrame:
        """Convert arcengine observation to GameFrame."""
        import numpy as np
        
        # Extract grid
        grid = np.array(obs.grid if hasattr(obs, 'grid') else obs.state, dtype=np.int32)
        
        # Extract state
        if hasattr(obs, 'state') and isinstance(obs.state, ArcEngineState):
            state = GameState(obs.state.value)
        else:
            state = GameState.NOT_FINISHED
        
        # Extract available actions
        available = set()
        if hasattr(obs, 'available_actions'):
            for a in obs.available_actions:
                if isinstance(a, ArcEngineAction):
                    available.add(GameAction(a.value))
                else:
                    available.add(GameAction(int(a)))
        else:
            available = {GameAction.ACTION1, GameAction.ACTION2,
                        GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5}
        
        return GameFrame(
            grid=grid,
            state=state,
            score=getattr(obs, 'score', 0),
            available_actions=available,
            guid=self._current_guid or '',
            game_id=game_id,
        )
    
    # ─────────────────────────────────────────
    # Context Manager
    # ─────────────────────────────────────────
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._current_scorecard:
            self.close_scorecard()
        return False


# ─────────────────────────────────────────────
# Convenience Factory
# ─────────────────────────────────────────────

def create_session(
    mode: str = 'offline',
    api_key: Optional[str] = None,
) -> ARCSession:
    """
    Create an ARC session with specified mode.
    
    Args:
        mode: 'offline', 'online', or 'competition'
        api_key: API key for online modes
        
    Returns:
        Configured ARCSession
    """
    config = SessionConfig(
        operation_mode=OperationMode(mode),
        api_key=api_key or os.getenv('ARC_API_KEY'),
    )
    return ARCSession(config=config)


# ─────────────────────────────────────────────
# Module-level imports for convenience
# ─────────────────────────────────────────────

import numpy as np


# ─────────────────────────────────────────────
# Demo / Test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("ARC-AGI-3 Session Manager")
    print("=" * 50)
    
    # Test offline session
    print("\n1. Creating offline session...")
    session = create_session(mode='offline')
    print(f"   Mode: {session.config.operation_mode.value}")
    
    # Test scorecard
    print("\n2. Testing scorecard...")
    scorecard = session.open_scorecard(tags=['test'])
    print(f"   Opened: {scorecard.card_id}")
    print(f"   Is open: {scorecard.is_open}")
    
    # Test mock environment
    print("\n3. Testing mock environment...")
    env = session.make_env('test-game')
    frame = session.reset(env)
    print(f"   Grid shape: {frame.grid.shape}")
    print(f"   State: {frame.state.name}")
    print(f"   Available: {[a.semantic_name for a in frame.available_actions]}")
    
    # Test action
    print("\n4. Testing action...")
    frame = session.step(env, GameAction.ACTION1)
    print(f"   After ACTION1: state={frame.state.name}")
    
    # Test close
    print("\n5. Closing scorecard...")
    result = session.close_scorecard()
    print(f"   Closed at: {result.closed_at}")
    
    print("\n" + "=" * 50)
    print("  ✓ Session manager ready")
