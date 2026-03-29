"""
game_memory.py — Phase 12: Per-Game Episodic Memory

Stores game-specific memories for cross-session learning.
Tracks observations, actions, effects, and learned rules per game.
"""

import torch
from torch import Tensor
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import numpy as np


# ─────────────────────────────────────────────
# Memory Entry Types
# ─────────────────────────────────────────────

@dataclass
class ActionMemory:
    """Memory of an action taken."""
    step: int
    position: Tuple[int, int]
    action_id: int
    action_name: str
    reasoning: str
    confidence: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            'step': self.step,
            'position': self.position,
            'action_id': self.action_id,
            'action_name': self.action_name,
            'reasoning': self.reasoning[:200],  # Truncate long reasoning
            'confidence': self.confidence,
            'timestamp': self.timestamp,
        }


@dataclass
class EffectMemory:
    """Memory of observed effect (action → change)."""
    step: int
    action_id: int
    position: Tuple[int, int]
    changes: List[Dict]  # List of {position, old, new}
    summary: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            'step': self.step,
            'action_id': self.action_id,
            'position': self.position,
            'changes': self.changes[:10],  # Limit changes
            'summary': self.summary,
            'timestamp': self.timestamp,
        }


@dataclass
class RuleMemory:
    """Memory of a learned rule."""
    rule_id: int
    trigger_action: int
    trigger_color: int
    effect_description: str
    confidence: float
    observations: int
    discovered_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            'rule_id': self.rule_id,
            'trigger_action': self.trigger_action,
            'trigger_color': self.trigger_color,
            'effect_description': self.effect_description,
            'confidence': self.confidence,
            'observations': self.observations,
            'discovered_at': self.discovered_at,
        }


@dataclass
class SessionSummary:
    """Summary of a game session."""
    session_id: str
    game_id: str
    start_time: str
    end_time: str
    total_steps: int
    final_score: Optional[float]
    final_state: str  # 'WON', 'LOST', 'TIMEOUT', 'ABANDONED'
    key_discoveries: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'session_id': self.session_id,
            'game_id': self.game_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'total_steps': self.total_steps,
            'final_score': self.final_score,
            'final_state': self.final_state,
            'key_discoveries': self.key_discoveries,
        }


# ─────────────────────────────────────────────
# Game Memory System
# ─────────────────────────────────────────────

class GameMemory:
    """
    Per-game episodic memory for FLUX Multi-Modal Agent.
    
    Features:
    - Session-based memory (current game)
    - Cross-session learning (persisted discoveries)
    - Rule accumulation (learned patterns)
    - Navigation history
    - Goal tracking
    """
    
    def __init__(
        self,
        game_id: str,
        max_session_memories: int = 1000,
        max_cross_session_rules: int = 100,
    ):
        """
        Initialize game memory.
        
        Args:
            game_id: Game identifier (e.g., "ls20")
            max_session_memories: Max memories per session
            max_cross_session_rules: Max rules to keep across sessions
        """
        self.game_id = game_id
        self.max_session_memories = max_session_memories
        self.max_cross_session_rules = max_cross_session_rules
        
        # ── Current Session ──
        self.session_id = self._generate_session_id()
        self.session_start = datetime.now().isoformat()
        self.current_step = 0
        
        # Session memories
        self.action_memories: List[ActionMemory] = []
        self.effect_memories: List[EffectMemory] = []
        
        # ── Cross-Session Learning ──
        self.learned_rules: List[RuleMemory] = []
        self.session_summaries: List[SessionSummary] = []
        
        # ── Navigation State ──
        self.visited_positions: Dict[Tuple[int, int], int] = {}
        self.position_sequence: List[Tuple[int, int]] = []
        
        # ── Goal Tracking ──
        self.goals_completed: List[str] = []
        self.goals_pending: List[str] = []
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        import uuid
        return f"{self.game_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    
    # ─────────────────────────────────────────────
    # Session Memory Operations
    # ─────────────────────────────────────────────
    
    def record_action(
        self,
        position: Tuple[int, int],
        action_id: int,
        action_name: str,
        reasoning: str,
        confidence: float,
    ):
        """Record an action taken."""
        self.current_step += 1
        
        memory = ActionMemory(
            step=self.current_step,
            position=position,
            action_id=action_id,
            action_name=action_name,
            reasoning=reasoning,
            confidence=confidence,
        )
        
        self.action_memories.append(memory)
        
        # Track position visit
        self.visited_positions[position] = self.visited_positions.get(position, 0) + 1
        self.position_sequence.append(position)
        
        # Trim if too long
        if len(self.action_memories) > self.max_session_memories:
            self.action_memories = self.action_memories[-self.max_session_memories:]
    
    def record_effect(
        self,
        action_id: int,
        position: Tuple[int, int],
        changes: List[Dict],
        summary: str,
    ):
        """Record observed effect of action."""
        memory = EffectMemory(
            step=self.current_step,
            action_id=action_id,
            position=position,
            changes=changes,
            summary=summary,
        )
        
        self.effect_memories.append(memory)
        
        # Trim if too long
        if len(self.effect_memories) > self.max_session_memories:
            self.effect_memories = self.effect_memories[-self.max_session_memories:]
    
    def record_rule(
        self,
        trigger_action: int,
        trigger_color: int,
        effect_description: str,
        confidence: float,
        observations: int,
    ):
        """Record a discovered rule."""
        # Check if rule already exists
        for existing in self.learned_rules:
            if (existing.trigger_action == trigger_action and 
                existing.trigger_color == trigger_color):
                # Update existing rule
                existing.confidence = max(existing.confidence, confidence)
                existing.observations += observations
                return
        
        # Add new rule
        rule = RuleMemory(
            rule_id=len(self.learned_rules),
            trigger_action=trigger_action,
            trigger_color=trigger_color,
            effect_description=effect_description,
            confidence=confidence,
            observations=observations,
        )
        
        self.learned_rules.append(rule)
        
        # Trim to max
        if len(self.learned_rules) > self.max_cross_session_rules:
            # Keep highest confidence rules
            self.learned_rules.sort(key=lambda r: -r.confidence)
            self.learned_rules = self.learned_rules[:self.max_cross_session_rules]
    
    # ─────────────────────────────────────────────
    # Query Operations
    # ─────────────────────────────────────────────
    
    def get_recent_actions(self, n: int = 5) -> List[ActionMemory]:
        """Get n most recent actions."""
        return self.action_memories[-n:]
    
    def get_recent_effects(self, n: int = 5) -> List[EffectMemory]:
        """Get n most recent effects."""
        return self.effect_memories[-n:]
    
    def get_position_visits(self, position: Tuple[int, int]) -> int:
        """Get number of times position was visited."""
        return self.visited_positions.get(position, 0)
    
    def get_unvisited_neighbors(
        self,
        position: Tuple[int, int],
        grid_size: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Get unvisited neighboring positions."""
        r, c = position
        h, w = grid_size
        neighbors = []
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                if (nr, nc) not in self.visited_positions:
                    neighbors.append((nr, nc))
        
        return neighbors
    
    def format_action_history(self, n: int = 5) -> str:
        """Format recent action history for LLM context."""
        recent = self.get_recent_actions(n)
        if not recent:
            return "No previous actions."
        
        lines = []
        for mem in recent:
            lines.append(
                f"Step {mem.step}: {mem.action_name} at {mem.position} "
                f"(conf={mem.confidence:.2f})"
            )
        
        return "\n".join(lines)
    
    def format_learned_rules(self) -> str:
        """Format learned rules for LLM context."""
        if not self.learned_rules:
            return "No rules learned yet."
        
        lines = ["Learned rules:"]
        for rule in sorted(self.learned_rules, key=lambda r: -r.confidence)[:10]:
            lines.append(
                f"- Action {rule.trigger_action} on color {rule.trigger_color}: "
                f"{rule.effect_description} (conf={rule.confidence:.2f})"
            )
        
        return "\n".join(lines)
    
    # ─────────────────────────────────────────────
    # Session Management
    # ─────────────────────────────────────────────
    
    def end_session(
        self,
        final_score: Optional[float],
        final_state: str,
        key_discoveries: Optional[List[str]] = None,
    ):
        """End current session and save summary."""
        summary = SessionSummary(
            session_id=self.session_id,
            game_id=self.game_id,
            start_time=self.session_start,
            end_time=datetime.now().isoformat(),
            total_steps=self.current_step,
            final_score=final_score,
            final_state=final_state,
            key_discoveries=key_discoveries or [],
        )
        
        self.session_summaries.append(summary)
        
        # Keep only recent summaries
        if len(self.session_summaries) > 50:
            self.session_summaries = self.session_summaries[-50:]
    
    def new_session(self):
        """Start a new session (preserves cross-session learning)."""
        self.session_id = self._generate_session_id()
        self.session_start = datetime.now().isoformat()
        self.current_step = 0
        
        # Clear session-specific memories
        self.action_memories = []
        self.effect_memories = []
        self.visited_positions = {}
        self.position_sequence = []
        
        # Keep cross-session learning (rules, summaries)
    
    # ─────────────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────────────
    
    def get_state(self) -> Dict[str, Any]:
        """Get serializable state."""
        return {
            'game_id': self.game_id,
            'session_id': self.session_id,
            'session_start': self.session_start,
            'current_step': self.current_step,
            'action_memories': [m.to_dict() for m in self.action_memories[-100:]],
            'effect_memories': [m.to_dict() for m in self.effect_memories[-100:]],
            'learned_rules': [r.to_dict() for r in self.learned_rules],
            'session_summaries': [s.to_dict() for s in self.session_summaries[-20:]],
            'goals_completed': self.goals_completed,
            'goals_pending': self.goals_pending,
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load from state dict."""
        self.game_id = state.get('game_id', self.game_id)
        
        # Load cross-session data (rules, summaries)
        if 'learned_rules' in state:
            self.learned_rules = [
                RuleMemory(
                    rule_id=r.get('rule_id', i),
                    trigger_action=r['trigger_action'],
                    trigger_color=r['trigger_color'],
                    effect_description=r['effect_description'],
                    confidence=r['confidence'],
                    observations=r['observations'],
                    discovered_at=r.get('discovered_at', datetime.now().isoformat()),
                )
                for i, r in enumerate(state['learned_rules'])
            ]
        
        if 'session_summaries' in state:
            self.session_summaries = [
                SessionSummary(**s) for s in state['session_summaries']
            ]
        
        self.goals_completed = state.get('goals_completed', [])
        self.goals_pending = state.get('goals_pending', [])
    
    def save(self, path: Union[str, Path]):
        """Save to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.get_state(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'GameMemory':
        """Load from file."""
        with open(path, 'r') as f:
            state = json.load(f)
        
        memory = cls(game_id=state.get('game_id', 'unknown'))
        memory.load_state(state)
        return memory
    
    # ─────────────────────────────────────────────
    # Statistics
    # ─────────────────────────────────────────────
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'game_id': self.game_id,
            'session_id': self.session_id,
            'current_step': self.current_step,
            'action_count': len(self.action_memories),
            'effect_count': len(self.effect_memories),
            'rules_learned': len(self.learned_rules),
            'sessions_total': len(self.session_summaries),
            'unique_positions_visited': len(self.visited_positions),
            'most_visited': max(self.visited_positions.values()) if self.visited_positions else 0,
        }


# ─────────────────────────────────────────────
# Game Memory Manager
# ─────────────────────────────────────────────

class GameMemoryManager:
    """
    Manages memories for multiple games.
    Persists cross-session learning to disk.
    """
    
    def __init__(self, save_dir: Optional[Union[str, Path]] = None):
        """
        Initialize manager.
        
        Args:
            save_dir: Directory to save game memories
        """
        self.memories: Dict[str, GameMemory] = {}
        self.save_dir = Path(save_dir) if save_dir else None
    
    def get_memory(self, game_id: str) -> GameMemory:
        """Get or create memory for a game."""
        if game_id not in self.memories:
            # Try to load from disk
            if self.save_dir:
                path = self.save_dir / f"{game_id}_memory.json"
                if path.exists():
                    self.memories[game_id] = GameMemory.load(path)
                else:
                    self.memories[game_id] = GameMemory(game_id)
            else:
                self.memories[game_id] = GameMemory(game_id)
        
        return self.memories[game_id]
    
    def save_all(self):
        """Save all memories to disk."""
        if not self.save_dir:
            return
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        for game_id, memory in self.memories.items():
            path = self.save_dir / f"{game_id}_memory.json"
            memory.save(path)
    
    def get_all_games(self) -> List[str]:
        """Get list of all games with memories."""
        return list(self.memories.keys())
    
    def get_total_rules(self) -> int:
        """Get total rules learned across all games."""
        return sum(len(m.learned_rules) for m in self.memories.values())


# ─────────────────────────────────────────────
# Module Exports
# ─────────────────────────────────────────────

__all__ = [
    'GameMemory',
    'GameMemoryManager',
    'ActionMemory',
    'EffectMemory',
    'RuleMemory',
    'SessionSummary',
]
