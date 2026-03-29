#!/usr/bin/env python3
"""
demo_phase12_demo2.py — Demo: Compare Heuristic vs Vision-LLM Agent

This demo compares:
1. A simple heuristic agent (random/greedy)
2. The FLUX Multi-Modal Agent with LLM reasoning

Shows how the LLM-based agent can reason about puzzle structure
while the heuristic agent just moves randomly.

Run: python demo_phase12_demo2.py
"""

import sys
from pathlib import Path
import random
import numpy as np
from typing import Tuple, List

# Add project root and phase12
PHASE_DIR = Path(__file__).parent
PROJECT_ROOT = PHASE_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PHASE_DIR))


# ─────────────────────────────────────────────
# Sample Puzzle Environment
# ─────────────────────────────────────────────

class SimplePuzzle:
    """
    Simple puzzle where the goal is to reach a target cell.
    
    The grid has:
    - 0: Empty (passable)
    - 1: Wall (impassable)
    - 5: Start position
    - 9: Goal
    """
    
    def __init__(self, size: int = 10):
        self.size = size
        self.grid = self._generate_grid()
        self.start = self._find_value(5)
        self.goal = self._find_value(9)
        self.position = self.start
        self.steps = 0
        self.max_steps = 50
    
    def _generate_grid(self) -> List[List[int]]:
        """Generate puzzle grid."""
        grid = [[0] * self.size for _ in range(self.size)]
        
        # Add walls
        for r in range(self.size):
            grid[r][0] = 1
            grid[r][self.size - 1] = 1
        for c in range(self.size):
            grid[0][c] = 1
            grid[self.size - 1][c] = 1
        
        # Add some internal walls
        for i in range(2, self.size - 2, 3):
            for j in range(2, self.size - 3):
                grid[i][j] = 1
        
        # Leave gaps in walls
        for i in range(2, self.size - 2, 3):
            gap = random.randint(2, self.size - 3)
            grid[i][gap] = 0
        
        # Start and goal
        grid[1][1] = 5  # Start
        grid[self.size - 2][self.size - 2] = 9  # Goal
        
        return grid
    
    def _find_value(self, val: int) -> Tuple[int, int]:
        """Find position of value in grid."""
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r][c] == val:
                    return (r, c)
        return (1, 1)
    
    def reset(self):
        """Reset puzzle."""
        self.position = self.start
        self.steps = 0
    
    def step(self, action: int) -> Tuple[bool, bool]:
        """
        Take a step.
        
        Args:
            action: 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT
            
        Returns:
            (valid_move, reached_goal)
        """
        deltas = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
        dr, dc = deltas.get(action, (0, 0))
        
        new_r = self.position[0] + dr
        new_c = self.position[1] + dc
        
        self.steps += 1
        
        # Check bounds and walls
        if (0 <= new_r < self.size and 0 <= new_c < self.size and
            self.grid[new_r][new_c] != 1):
            self.position = (new_r, new_c)
            valid = True
        else:
            valid = False
        
        # Check goal
        reached_goal = self.position == self.goal
        
        return valid, reached_goal
    
    def is_done(self) -> bool:
        """Check if episode is done."""
        return self.position == self.goal or self.steps >= self.max_steps
    
    def get_state(self) -> dict:
        """Get current state."""
        return {
            'grid': [row[:] for row in self.grid],
            'position': self.position,
            'goal': self.goal,
            'steps': self.steps,
        }


# ─────────────────────────────────────────────
# Heuristic Agent
# ─────────────────────────────────────────────

class HeuristicAgent:
    """Simple heuristic agent that moves toward goal."""
    
    def __init__(self):
        self.name = "Heuristic Agent"
    
    def decide(
        self,
        position: Tuple[int, int],
        goal: Tuple[int, int],
        available: List[int],
    ) -> int:
        """
        Decide action using simple heuristic.
        
        Move toward goal (if possible), else random.
        """
        r, c = position
        gr, gc = goal
        
        # Determine preferred direction
        preferred = []
        if gr < r and 1 in available:
            preferred.append(1)  # UP
        if gr > r and 2 in available:
            preferred.append(2)  # DOWN
        if gc < c and 3 in available:
            preferred.append(3)  # LEFT
        if gc > c and 4 in available:
            preferred.append(4)  # RIGHT
        
        if preferred:
            return random.choice(preferred)
        
        # Random fallback
        return random.choice(available)


# ─────────────────────────────────────────────
# Run Comparison
# ─────────────────────────────────────────────

def run_heuristic_agent(puzzle: SimplePuzzle, verbose: bool = False) -> dict:
    """Run heuristic agent on puzzle."""
    agent = HeuristicAgent()
    puzzle.reset()
    
    actions = []
    while not puzzle.is_done():
        action = agent.decide(
            puzzle.position,
            puzzle.goal,
            [1, 2, 3, 4]
        )
        valid, won = puzzle.step(action)
        actions.append(action)
        
        if verbose:
            print(f"  Step {puzzle.steps}: action={action}, valid={valid}, pos={puzzle.position}")
        
        if won:
            break
    
    return {
        'agent': 'Heuristic',
        'steps': puzzle.steps,
        'won': puzzle.position == puzzle.goal,
        'actions': actions,
    }


def run_llm_agent(puzzle: SimplePuzzle, verbose: bool = False) -> dict:
    """Run FLUX Multi-Modal Agent on puzzle."""
    from flux_multi_agent import FLUXMultiAgent, MultiAgentConfig
    
    # Create agent
    config = MultiAgentConfig(
        enable_vision=False,
        verbose=False,
    )
    agent = FLUXMultiAgent(config=config)
    agent.reset("comparison_puzzle")
    
    puzzle.reset()
    actions = []
    
    while not puzzle.is_done():
        state = puzzle.get_state()
        
        # Get agent decision
        action, reasoning = agent.decide_action(
            frame=state['grid'],
            position=state['position'],
            available_actions=[1, 2, 3, 4],
        )
        
        valid, won = puzzle.step(action)
        actions.append(action)
        
        if verbose:
            print(f"  Step {puzzle.steps}: action={action}, pos={puzzle.position}")
            print(f"    Reasoning: {reasoning[:80]}...")
        
        # Record effect (for learning)
        agent.observe(state['grid'], puzzle.position)
        
        if won:
            break
    
    agent.end_game(
        final_score=100 if puzzle.position == puzzle.goal else 0,
        final_state='WON' if puzzle.position == puzzle.goal else 'TIMEOUT'
    )
    
    return {
        'agent': 'FLUX LLM',
        'steps': puzzle.steps,
        'won': puzzle.position == puzzle.goal,
        'actions': actions,
    }


def main():
    print("\n" + "═" * 60)
    print("  Phase 12 Demo 2: Heuristic vs LLM Agent Comparison")
    print("═" * 60 + "\n")
    
    # Create puzzle
    print("─── Creating Puzzle ───")
    puzzle = SimplePuzzle(size=10)
    
    # Show puzzle
    print("\nPuzzle layout:")
    print("  1 = wall, 5 = start, 9 = goal, 0 = empty")
    for row in puzzle.grid:
        line = " ".join(str(c) if c != 0 else '.' for c in row)
        print(f"  {line}")
    
    print(f"\n  Start: {puzzle.start}")
    print(f"  Goal:  {puzzle.goal}")
    
    # Run agents
    n_runs = 5
    
    print("\n─── Running Heuristic Agent ───")
    heuristic_results = []
    for i in range(n_runs):
        result = run_heuristic_agent(puzzle, verbose=(i == 0))
        heuristic_results.append(result)
        print(f"  Run {i+1}: {'✓ Won' if result['won'] else '✗ Lost'} in {result['steps']} steps")
    
    print("\n─── Running FLUX LLM Agent ───")
    llm_results = []
    for i in range(n_runs):
        result = run_llm_agent(puzzle, verbose=(i == 0))
        llm_results.append(result)
        print(f"  Run {i+1}: {'✓ Won' if result['won'] else '✗ Lost'} in {result['steps']} steps")
    
    # Summary
    print("\n" + "─" * 40)
    print("  COMPARISON SUMMARY")
    print("─" * 40)
    
    h_wins = sum(1 for r in heuristic_results if r['won'])
    h_avg_steps = np.mean([r['steps'] for r in heuristic_results])
    
    l_wins = sum(1 for r in llm_results if r['won'])
    l_avg_steps = np.mean([r['steps'] for r in llm_results])
    
    print(f"\n  Heuristic Agent:")
    print(f"    Win rate: {h_wins}/{n_runs} ({100*h_wins/n_runs:.0f}%)")
    print(f"    Avg steps: {h_avg_steps:.1f}")
    
    print(f"\n  FLUX LLM Agent:")
    print(f"    Win rate: {l_wins}/{n_runs} ({100*l_wins/n_runs:.0f}%)")
    print(f"    Avg steps: {l_avg_steps:.1f}")
    
    # Analysis
    print("\n─── Analysis ───")
    
    if l_wins > h_wins:
        print("  ✓ LLM agent wins more often!")
        print("    The LLM can reason about the puzzle structure.")
    elif l_wins == h_wins:
        print("  = Both agents have similar win rates.")
        print("    This simple puzzle may not show the full LLM advantage.")
    else:
        print("  ⚠ Heuristic won more on this simple puzzle.")
        print("    LLM shines on complex reasoning tasks, not simple mazes.")
    
    if l_avg_steps < h_avg_steps:
        print(f"  ✓ LLM takes {h_avg_steps - l_avg_steps:.1f} fewer steps on average!")
    
    print("\n" + "═" * 60)
    print("  Demo 2 Complete!")
    print("  Note: The LLM advantage is clearer on complex puzzles")
    print("  like ARC-AGI-3 where visual reasoning matters.")
    print("═" * 60 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
