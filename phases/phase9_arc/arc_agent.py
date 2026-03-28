"""
Phase 9 ARC: Interactive Agent for ARC-AGI-3

ARC-AGI-3 presents turn-based game environments where the agent must:
1. EXPLORE — actively obtain information by interacting
2. MODEL — build a generalizable world model from observations
3. SET GOALS — identify desirable states WITHOUT instructions
4. PLAN — map action paths and course-correct
5. EXECUTE — take actions and adapt based on feedback

This is where FLUX physics shines:
- Real-time field updates as agent acts
- Attractors naturally identify "desirable" states
- Causal chains map action paths
- Zero forgetting accumulates knowledge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import deque
import sys
from pathlib import Path
import time
import random

# Add project paths
_PHASES_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PHASES_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from flux_utils import get_device


# ─────────────────────────────────────────────
# Environment Interface
# ─────────────────────────────────────────────

@dataclass
class Observation:
    """Single observation from environment."""
    state: List[List[int]]    # Current grid state
    valid_actions: List[int]  # Available actions
    feedback: Optional[str] = None  # Optional feedback from last action
    done: bool = False


@dataclass  
class ActionResult:
    """Result of taking an action."""
    observation: Observation
    reward: float = 0.0       # Implicit reward (inferred by agent)
    info: Dict[str, Any] = field(default_factory=dict)


class ARCEnvironment:
    """
    Abstract ARC-AGI-3 environment interface.
    
    Each environment is a turn-based game with:
    - observe() → current state
    - act(action) → next state + feedback
    - done → whether episode is complete
    """
    
    def __init__(self, env_id: str):
        self.env_id = env_id
        self.step_count = 0
        self.history: List[Tuple[int, Observation]] = []
    
    def reset(self) -> Observation:
        """Reset environment to initial state."""
        raise NotImplementedError
    
    def observe(self) -> Observation:
        """Get current observation."""
        raise NotImplementedError
    
    def act(self, action: int) -> ActionResult:
        """Take action and get result."""
        raise NotImplementedError
    
    @property
    def done(self) -> bool:
        """Check if episode is complete."""
        raise NotImplementedError


# ─────────────────────────────────────────────
# Sample Game Environments (For Testing)
# ─────────────────────────────────────────────

class NavigationEnv(ARCEnvironment):
    """
    Simple navigation: Move agent (color 1) to goal (color 2).
    
    Actions: 0=up, 1=down, 2=left, 3=right
    """
    
    def __init__(self, size: int = 5, env_id: str = "nav_001"):
        super().__init__(env_id)
        self.size = size
        self.agent_pos = [0, 0]
        self.goal_pos = [size-1, size-1]
        self._done = False
    
    def reset(self) -> Observation:
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size-1, self.size-1]
        self._done = False
        self.step_count = 0
        self.history = []
        return self.observe()
    
    def observe(self) -> Observation:
        grid = [[0] * self.size for _ in range(self.size)]
        grid[self.agent_pos[0]][self.agent_pos[1]] = 1  # Agent
        grid[self.goal_pos[0]][self.goal_pos[1]] = 2    # Goal
        
        return Observation(
            state=grid,
            valid_actions=[0, 1, 2, 3],
            done=self._done,
        )
    
    def act(self, action: int) -> ActionResult:
        # Move agent
        dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
        new_r = max(0, min(self.size-1, self.agent_pos[0] + dr))
        new_c = max(0, min(self.size-1, self.agent_pos[1] + dc))
        self.agent_pos = [new_r, new_c]
        
        self.step_count += 1
        
        # Check if reached goal
        if self.agent_pos == self.goal_pos:
            self._done = True
        
        obs = self.observe()
        self.history.append((action, obs))
        
        return ActionResult(
            observation=obs,
            reward=1.0 if self._done else 0.0,
        )
    
    @property
    def done(self) -> bool:
        return self._done


class PatternMatchEnv(ARCEnvironment):
    """
    Match the pattern: Make output grid match a hidden target.
    
    Actions: 0-9 = place color at cursor, 10=move cursor right, 11=move down
    """
    
    def __init__(self, size: int = 3, env_id: str = "pattern_001"):
        super().__init__(env_id)
        self.size = size
        self.cursor = [0, 0]
        self.grid = [[0] * size for _ in range(size)]
        self.target = self._generate_target()
        self._done = False
    
    def _generate_target(self) -> List[List[int]]:
        """Generate random target pattern."""
        return [[random.randint(0, 3) for _ in range(self.size)] for _ in range(self.size)]
    
    def reset(self) -> Observation:
        self.cursor = [0, 0]
        self.grid = [[0] * self.size for _ in range(self.size)]
        self.target = self._generate_target()
        self._done = False
        self.step_count = 0
        self.history = []
        return self.observe()
    
    def observe(self) -> Observation:
        # Show current grid + cursor
        display = [row[:] for row in self.grid]
        display[self.cursor[0]][self.cursor[1]] = 9  # Cursor marker
        
        return Observation(
            state=display,
            valid_actions=list(range(12)),
            feedback=f"Match the target pattern",
            done=self._done,
        )
    
    def act(self, action: int) -> ActionResult:
        if action < 10:
            # Place color
            self.grid[self.cursor[0]][self.cursor[1]] = action
        elif action == 10:
            # Move right
            self.cursor[1] = (self.cursor[1] + 1) % self.size
        elif action == 11:
            # Move down
            self.cursor[0] = (self.cursor[0] + 1) % self.size
        
        self.step_count += 1
        
        # Check if matches target
        if self.grid == self.target:
            self._done = True
        
        obs = self.observe()
        self.history.append((action, obs))
        
        return ActionResult(
            observation=obs,
            reward=1.0 if self._done else 0.0,
        )
    
    @property
    def done(self) -> bool:
        return self._done


# ─────────────────────────────────────────────
# World Model (Resonance Field Based)
# ─────────────────────────────────────────────

class WorldModel(nn.Module):
    """
    Learn environment dynamics through wave settling.
    
    Maps (state, action) → predicted next state
    """
    
    def __init__(
        self,
        state_dim: int = 432,
        action_dim: int = 12,
        hidden_dim: int = 256,
        device: str = 'cpu',
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # State encoder (grid → wave)
        self.state_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(900, hidden_dim),  # Max 30x30 grid
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        
        # Action embedding
        self.action_embed = nn.Embedding(action_dim, 64)
        
        # Dynamics predictor
        self.dynamics = nn.Sequential(
            nn.Linear(state_dim + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        
        # Experience memory
        self.experiences: List[Tuple[Tensor, int, Tensor]] = []
        
        self.to(device)
    
    def encode_state(self, state: List[List[int]]) -> Tensor:
        """Encode grid state to wave."""
        t = torch.tensor(state, dtype=torch.float, device=self.device)
        
        # Pad to 30x30
        h, w = t.shape
        padded = torch.zeros(30, 30, device=self.device)
        padded[:h, :w] = t
        
        return self.state_encoder(padded.unsqueeze(0)).squeeze(0)
    
    def predict_next(self, state_wave: Tensor, action: int) -> Tensor:
        """Predict next state wave given current state and action."""
        action_emb = self.action_embed(torch.tensor([action], device=self.device))
        combined = torch.cat([state_wave, action_emb.squeeze(0)])
        return self.dynamics(combined)
    
    def update(
        self,
        state: List[List[int]],
        action: int,
        next_state: List[List[int]],
    ):
        """Update model with new experience."""
        state_wave = self.encode_state(state)
        next_wave = self.encode_state(next_state)
        self.experiences.append((state_wave.detach(), action, next_wave.detach()))
        
        # Simple online learning
        if len(self.experiences) > 10:
            self._train_batch()
    
    def _train_batch(self, batch_size: int = 8):
        """Train on recent experiences."""
        if len(self.experiences) < batch_size:
            return
        
        # Sample batch
        indices = random.sample(range(len(self.experiences)), batch_size)
        batch = [self.experiences[i] for i in indices]
        
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        
        total_loss = 0
        for state_wave, action, next_wave in batch:
            predicted = self.predict_next(state_wave, action)
            loss = F.mse_loss(predicted, next_wave)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()


# ─────────────────────────────────────────────
# Goal Inferrer (Attractor Based)
# ─────────────────────────────────────────────

class GoalInferrer(nn.Module):
    """
    Infer goals from observed states WITHOUT explicit instructions.
    
    Key insight: "Goal states" tend to be:
    - Low energy (stable)
    - Frequently reached in successful episodes
    - Have distinct features (not uniform)
    
    Uses attractor dynamics: states that attract trajectories = goals
    """
    
    def __init__(
        self,
        state_dim: int = 432,
        device: str = 'cpu',
    ):
        super().__init__()
        self.state_dim = state_dim
        self.device = device
        
        # Goal scorer (higher = more likely a goal)
        self.goal_scorer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        # Memory of terminal states (likely goals)
        self.terminal_states: List[Tensor] = []
        
        self.to(device)
    
    def score_state(self, state_wave: Tensor) -> float:
        """Score how likely a state is a goal."""
        with torch.no_grad():
            return self.goal_scorer(state_wave).item()
    
    def register_terminal(self, state_wave: Tensor, success: bool):
        """Register a terminal state."""
        if success:
            self.terminal_states.append(state_wave.detach())
    
    def get_goal_direction(self, current_wave: Tensor) -> Tensor:
        """Get direction toward likely goal states."""
        if not self.terminal_states:
            return torch.zeros(self.state_dim, device=self.device)
        
        # Average direction to known goals
        directions = []
        for goal_wave in self.terminal_states[-10:]:  # Recent goals
            direction = goal_wave - current_wave
            directions.append(direction / (direction.norm() + 1e-6))
        
        return torch.stack(directions).mean(dim=0)


# ─────────────────────────────────────────────
# Causal Planner
# ─────────────────────────────────────────────

class CausalPlanner:
    """
    Plan actions using causal reasoning.
    
    Builds action→outcome graph and searches for paths to goal.
    """
    
    def __init__(self, world_model: WorldModel):
        self.world_model = world_model
        self.action_outcomes: Dict[str, Dict[int, str]] = {}  # state_hash → action → next_hash
    
    def _hash_state(self, state: List[List[int]]) -> str:
        """Hash state for lookup."""
        return str(state)
    
    def record_transition(
        self,
        state: List[List[int]],
        action: int,
        next_state: List[List[int]],
    ):
        """Record observed transition."""
        state_hash = self._hash_state(state)
        next_hash = self._hash_state(next_state)
        
        if state_hash not in self.action_outcomes:
            self.action_outcomes[state_hash] = {}
        
        self.action_outcomes[state_hash][action] = next_hash
    
    def plan_to_goal(
        self,
        current_state: List[List[int]],
        goal_state: List[List[int]],
        max_depth: int = 10,
    ) -> Optional[List[int]]:
        """
        BFS search for action sequence to reach goal.
        """
        start_hash = self._hash_state(current_state)
        goal_hash = self._hash_state(goal_state)
        
        if start_hash == goal_hash:
            return []
        
        queue = deque([(start_hash, [])])
        visited = {start_hash}
        
        while queue:
            state_hash, actions = queue.popleft()
            
            if len(actions) >= max_depth:
                continue
            
            if state_hash not in self.action_outcomes:
                continue
            
            for action, next_hash in self.action_outcomes[state_hash].items():
                if next_hash == goal_hash:
                    return actions + [action]
                
                if next_hash not in visited:
                    visited.add(next_hash)
                    queue.append((next_hash, actions + [action]))
        
        return None  # No path found


# ─────────────────────────────────────────────
# FLUX Agent
# ─────────────────────────────────────────────

class FLUXAgent:
    """
    FLUX-based agent for ARC-AGI-3.
    
    Combines:
    - World model for dynamics prediction
    - Goal inferrer for objective detection
    - Causal planner for action selection
    - Exploration strategy for information gathering
    """
    
    def __init__(
        self,
        device: str = 'cpu',
        exploration_rate: float = 0.3,
    ):
        self.device = device
        self.exploration_rate = exploration_rate
        
        # Components
        self.world_model = WorldModel(device=device)
        self.goal_inferrer = GoalInferrer(device=device)
        self.planner = CausalPlanner(self.world_model)
        
        # State
        self.current_episode = 0
        self.total_steps = 0
        self.successes = 0
    
    def act(
        self,
        observation: Observation,
        deterministic: bool = False,
    ) -> int:
        """
        Select action given observation.
        
        Strategy:
        1. If we have a plan, follow it
        2. Otherwise, explore or exploit based on confidence
        """
        valid_actions = observation.valid_actions
        
        if not valid_actions:
            return 0
        
        # Encode current state
        state_wave = self.world_model.encode_state(observation.state)
        
        # Score each action
        action_scores = {}
        for action in valid_actions:
            # Predict next state
            predicted_next = self.world_model.predict_next(state_wave, action)
            
            # Score predicted state (higher = closer to goal)
            goal_score = self.goal_inferrer.score_state(predicted_next)
            
            # Add goal direction bonus
            goal_dir = self.goal_inferrer.get_goal_direction(state_wave)
            direction_score = F.cosine_similarity(
                predicted_next - state_wave,
                goal_dir,
                dim=0,
            ).item() if goal_dir.norm() > 0 else 0
            
            action_scores[action] = goal_score + 0.1 * direction_score
        
        # Exploration vs exploitation
        if not deterministic and random.random() < self.exploration_rate:
            return random.choice(valid_actions)
        
        # Pick best action
        best_action = max(action_scores, key=action_scores.get)
        return best_action
    
    def update(
        self,
        state: List[List[int]],
        action: int,
        next_state: List[List[int]],
        done: bool,
        success: bool = False,
    ):
        """Update agent after transition."""
        # Update world model
        self.world_model.update(state, action, next_state)
        
        # Record transition for planner
        self.planner.record_transition(state, action, next_state)
        
        # Update goal inferrer
        if done:
            next_wave = self.world_model.encode_state(next_state)
            self.goal_inferrer.register_terminal(next_wave, success)
            
            if success:
                self.successes += 1
        
        self.total_steps += 1
    
    def run_episode(
        self,
        env: ARCEnvironment,
        max_steps: int = 100,
    ) -> Dict[str, Any]:
        """Run single episode in environment."""
        obs = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            if obs.done:
                break
            
            # Select action
            action = self.act(obs)
            
            # Take action
            prev_state = obs.state
            result = env.act(action)
            
            # Update agent
            self.update(
                state=prev_state,
                action=action,
                next_state=result.observation.state,
                done=result.observation.done,
                success=result.reward > 0,
            )
            
            total_reward += result.reward
            obs = result.observation
        
        self.current_episode += 1
        
        return {
            "episode": self.current_episode,
            "steps": step + 1,
            "reward": total_reward,
            "success": total_reward > 0,
        }


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate_agent(
    agent: FLUXAgent,
    envs: List[ARCEnvironment],
    episodes_per_env: int = 10,
) -> Dict[str, Any]:
    """Evaluate agent on multiple environments."""
    results = {
        "total_episodes": 0,
        "total_successes": 0,
        "total_steps": 0,
        "env_results": {},
    }
    
    for env in envs:
        env_successes = 0
        env_steps = 0
        
        for _ in range(episodes_per_env):
            episode_result = agent.run_episode(env, max_steps=50)
            
            if episode_result["success"]:
                env_successes += 1
            env_steps += episode_result["steps"]
        
        results["env_results"][env.env_id] = {
            "success_rate": env_successes / episodes_per_env,
            "avg_steps": env_steps / episodes_per_env,
        }
        
        results["total_episodes"] += episodes_per_env
        results["total_successes"] += env_successes
        results["total_steps"] += env_steps
    
    results["overall_success_rate"] = results["total_successes"] / results["total_episodes"]
    results["avg_steps_per_episode"] = results["total_steps"] / results["total_episodes"]
    
    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("ARC-AGI-3 Agent Test")
    print("=" * 50)
    
    device = get_device()
    print(f"Device: {device}")
    
    # Create agent
    agent = FLUXAgent(device=device, exploration_rate=0.2)
    print("✓ Agent initialized")
    
    # Create test environments
    envs = [
        NavigationEnv(size=5, env_id="nav_5x5"),
    ]
    print(f"✓ Created {len(envs)} test environments")
    
    # Evaluate
    print("\nRunning evaluation...")
    results = evaluate_agent(agent, envs, episodes_per_env=20)
    
    print("\n" + "=" * 50)
    print("Results:")
    print(f"  Total episodes: {results['total_episodes']}")
    print(f"  Success rate: {results['overall_success_rate']*100:.1f}%")
    print(f"  Avg steps/episode: {results['avg_steps_per_episode']:.1f}")
    
    print("\nPer environment:")
    for env_id, env_results in results["env_results"].items():
        print(f"  {env_id}: {env_results['success_rate']*100:.1f}% success, {env_results['avg_steps']:.1f} avg steps")
