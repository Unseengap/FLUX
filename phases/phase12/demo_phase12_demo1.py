#!/usr/bin/env python3
"""
demo_phase12_demo1.py — Demo: Watch Agent SEE and Reason Through a Puzzle

This demo shows the FLUX Multi-Modal Agent:
1. Seeing a game grid (rendered as image or ASCII)
2. Using LLM reasoning to decide actions
3. Learning from observed effects
4. Building spatial memory

Run: python demo_phase12_demo1.py
"""

import sys
from pathlib import Path
import time

# Add project root and phase12
PHASE_DIR = Path(__file__).parent
PROJECT_ROOT = PHASE_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PHASE_DIR))


def create_sample_game():
    """Create a sample puzzle grid."""
    # A simple maze-like grid
    grid = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 2, 2, 2, 0, 0],
        [0, 1, 0, 1, 0, 2, 0, 2, 0, 0],
        [0, 1, 1, 1, 0, 2, 2, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 3, 3, 3, 0, 4, 4, 4, 0, 0],
        [0, 3, 0, 3, 0, 4, 0, 4, 0, 0],
        [0, 3, 3, 3, 0, 4, 4, 4, 0, 0],
        [0, 0, 0, 0, 5, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 5, 0, 0, 0, 0, 0],
    ]
    return grid


def main():
    print("\n" + "═" * 60)
    print("  Phase 12 Demo 1: Visual Reasoning Agent")
    print("═" * 60 + "\n")
    
    # Check for PIL
    try:
        from PIL import Image
        print("✓ PIL available — can render images")
        has_pil = True
    except ImportError:
        print("⚠ PIL not available — using ASCII mode")
        has_pil = False
    
    # Import components
    from flux_multi_agent import FLUXMultiAgent, MultiAgentConfig
    from grid_renderer import GridRenderer, ASCIIRenderer, PIL_AVAILABLE
    
    # Create agent with minimal config for demo
    print("\n─── Initializing Agent ───")
    config = MultiAgentConfig(
        enable_vision=False,  # Use ASCII for this demo
        verbose=True,
    )
    
    agent = FLUXMultiAgent(config=config)
    agent.summary()
    
    # Create sample game
    grid = create_sample_game()
    print("\n─── Sample Puzzle Grid ───")
    print(f"Grid size: {len(grid)} x {len(grid[0])}")
    
    # Show ASCII representation
    ascii_renderer = ASCIIRenderer(show_colors=True)
    print("\nASCII view:")
    print(ascii_renderer.render(grid, position=(4, 4)))
    
    # Save image if PIL available
    if has_pil:
        renderer = GridRenderer(cell_size=20)
        img = renderer.render(grid, position=(4, 4))
        output_path = PHASE_DIR / "demo_grid.png"
        img.save(str(output_path))
        print(f"\n✓ Grid image saved: {output_path}")
    
    # Start game session
    print("\n─── Game Session ───")
    agent.reset("demo_puzzle")
    print(f"Game: {agent.current_game}")
    
    # Simulate game loop
    position = (4, 4)
    available_actions = [1, 2, 3, 4]  # UP, DOWN, LEFT, RIGHT
    
    print("\n─── Agent Reasoning Steps ───\n")
    
    for step in range(5):
        print(f"Step {step + 1}:")
        print(f"  Position: {position}")
        
        # Observe
        obs = agent.observe(grid, position)
        
        # Decide action
        action, reasoning = agent.decide_action(grid, position, available_actions)
        
        print(f"  Action decided: {action} ({['?', 'UP', 'DOWN', 'LEFT', 'RIGHT'][action]})")
        print(f"  Reasoning: {reasoning[:150]}...")
        
        # Simulate movement
        deltas = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
        dr, dc = deltas.get(action, (0, 0))
        new_pos = (position[0] + dr, position[1] + dc)
        
        # Check bounds
        if 0 <= new_pos[0] < len(grid) and 0 <= new_pos[1] < len(grid[0]):
            position = new_pos
        
        print(f"  New position: {position}")
        print()
        
        time.sleep(0.5)  # Brief pause for readability
    
    # Show final stats
    print("─── Final Statistics ───")
    stats = agent.get_stats()
    print(f"  Actions taken: {stats['actions_taken']}")
    print(f"  Observations: {stats['observations']}")
    print(f"  Rules learned: {stats['learned_rules']}")
    
    # Save agent state
    print("\n─── Saving Agent State ───")
    output_path = PHASE_DIR / "demo_agent.flx"
    try:
        agent.save_flx(str(output_path), upload_to_hf=False)
        print(f"✓ Agent saved: {output_path}")
    except Exception as e:
        print(f"⚠ Save failed: {e}")
    
    # End game
    agent.end_game(final_score=100, final_state='DEMO_COMPLETE')
    
    print("\n" + "═" * 60)
    print("  Demo 1 Complete!")
    print("═" * 60 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
