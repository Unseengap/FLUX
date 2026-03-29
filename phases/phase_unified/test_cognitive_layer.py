#!/usr/bin/env python3
"""
Test script for FLUX Unified Cognitive Layer

Tests all four cognitive components:
1. CausalTracker — action → effect learning
2. RuleInducer — pattern → rule abstraction  
3. GoalPlanner — objective decomposition
4. PerceptionField — active vision system

ARC-AGI-3 Action Mapping:
- RESET=0, ACTION1-7=1-7
- ACTION1-4: directional (up/down/left/right)
- ACTION5: interact (previously toggle)
- ACTION6: click at (x,y)
- ACTION7: undo
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch

from causal_tracker import CausalTracker, GridChange, CausalLink
from rule_inducer import RuleInducer, Rule
from goal_planner import GoalPlanner, Goal, GoalType, GoalStatus
from perception_field import PerceptionField, Surprise

# ARC-AGI-3 action constants
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_INTERACT = 5  # Generic interact (was toggle)
ACTION_CLICK = 6     # Click at (x,y)
ACTION_UNDO = 7


def test_causal_tracker():
    """Test CausalTracker."""
    print("\n" + "=" * 60)
    print("Testing CausalTracker")
    print("=" * 60)
    
    tracker = CausalTracker()
    
    # Simulate grid changes
    grid_before = np.array([
        [0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    
    grid_after = np.array([
        [0, 0, 0, 0, 0],
        [0, 3, 0, 0, 0],  # 2 → 3
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    
    # Record action (using ACTION_INTERACT=5 for toggle-like behavior)
    link = tracker.record(
        position=(1, 1),
        action=ACTION_INTERACT,  # Action 5: interact
        grid_before=grid_before,
        grid_after=grid_after,
    )
    
    assert link is not None, "Should return a causal link"
    assert len(link.effects) == 1, f"Should have 1 effect, got {len(link.effects)}"
    print(f"  ✓ Recorded action with {len(link.effects)} effects")
    
    # Query effects
    effects = tracker.query_effects((1, 1), action=ACTION_INTERACT)
    assert len(effects) >= 1, "Should find at least one matching link"
    print(f"  ✓ Query found {len(effects)} matching links")
    
    # Test prediction
    predictions = tracker.predict_effect((1, 1), action=ACTION_INTERACT, grid=grid_before)
    print(f"  ✓ Prediction returned {len(predictions)} predicted changes")
    
    # Test state dict
    state = tracker.state_dict()
    tracker2 = CausalTracker()
    tracker2.load_state_dict(state)
    assert len(tracker2.causal_links) == len(tracker.causal_links)
    print(f"  ✓ State dict round-trip successful")
    
    return tracker


def test_rule_inducer(tracker: CausalTracker):
    """Test RuleInducer."""
    print("\n" + "=" * 60)
    print("Testing RuleInducer")
    print("=" * 60)
    
    inducer = RuleInducer(tracker, min_observations=2, min_confidence=0.4)
    
    # Add more observations to create patterns
    grid_before = np.array([
        [0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    
    grid_after = np.array([
        [0, 0, 0, 0, 0],
        [0, 3, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    
    # Record same pattern multiple times
    for _ in range(3):
        tracker.record((1, 1), action=ACTION_INTERACT, grid_before=grid_before, grid_after=grid_after)
    
    # Analyze and induce rules
    new_rules = inducer.analyze_causal_links(force=True)
    print(f"  ✓ Induced {len(new_rules)} rules from patterns")
    
    for rule in new_rules[:3]:
        print(f"    {rule}")
    
    # Test prediction
    predictions = inducer.predict_effects(
        position=(1, 1),
        action=ACTION_INTERACT,
        grid=grid_before,
    )
    print(f"  ✓ Predictions: {len(predictions)} predicted effects")
    
    # Test hypothesis testing
    if new_rules:
        result = inducer.test_rule(
            new_rules[0],
            trigger_position=(1, 1),
            grid_before=grid_before,
            grid_after=grid_after,
        )
        print(f"  ✓ Hypothesis test: {'passed' if result else 'failed'}")
    
    # Test state dict
    state = inducer.state_dict()
    inducer2 = RuleInducer()
    inducer2.load_state_dict(state)
    assert len(inducer2.rules) == len(inducer.rules)
    print(f"  ✓ State dict round-trip successful")
    
    return inducer


def test_goal_planner():
    """Test GoalPlanner."""
    print("\n" + "=" * 60)
    print("Testing GoalPlanner")
    print("=" * 60)
    
    planner = GoalPlanner()
    
    grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 4, 1, 0],  # Yellow at (2,2)
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ])
    position = (0, 0)
    
    # Test objective parsing
    goals = planner.set_objective("reach yellow")
    assert len(goals) >= 1, "Should create at least one goal"
    print(f"  ✓ Created {len(goals)} goals from 'reach yellow'")
    
    for goal in goals:
        print(f"    {goal}")
    
    # Test next subgoal
    next_goal = planner.next_subgoal(grid, position)
    assert next_goal is not None, "Should return a goal"
    print(f"  ✓ Next goal: {next_goal}")
    
    # Test target computation
    target = planner.compute_target_position(next_goal, grid, position)
    assert target is not None, "Should compute a target"
    print(f"  ✓ Target position: {target}")
    
    # Test achievement
    position = (2, 2)  # Move to yellow
    planner._update_goal_statuses(grid, position)
    print(f"  ✓ Goal status after moving: {next_goal.status}")
    
    # Test exit objective
    planner.reset()
    goals = planner.set_objective("exit door")
    print(f"  ✓ Created {len(goals)} goals from 'exit door'")
    
    # Test state dict
    state = planner.state_dict()
    planner2 = GoalPlanner()
    planner2.load_state_dict(state)
    print(f"  ✓ State dict round-trip successful")
    
    return planner


def test_perception_field():
    """Test PerceptionField."""
    print("\n" + "=" * 60)
    print("Testing PerceptionField")
    print("=" * 60)
    
    pf = PerceptionField(max_size=10, fovea_radius=2)
    
    grid1 = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 2, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ])
    
    grid2 = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 3, 1, 0],  # Changed 2 → 3
        [0, 1, 1, 4, 0],  # New at (3,3)
        [0, 0, 0, 0, 0],
    ])
    
    # First observation
    pf.focus((2, 2), reason="center")
    result1 = pf.observe(grid1)
    print(f"  ✓ First observation: fovea at {result1['fovea_center']}")
    
    # Make prediction (using ACTION_INTERACT=5 for grid manipulation)
    pf.predict_next(grid1, action=ACTION_INTERACT, agent_position=(2, 2))
    
    # Second observation
    result2 = pf.observe(grid2)
    print(f"  ✓ Second observation: {len(result2['surprises'])} surprises")
    
    for s in result2['surprises'][:3]:
        print(f"    {s}")
    
    # Test attention map
    attention = pf.compute_attention_map(grid2)
    print(f"  ✓ Attention map computed: shape={attention.shape}")
    
    # Test object tracking
    obj_id = pf.track_object((3, 3), color=4, importance=1.0)
    print(f"  ✓ Tracking object #{obj_id}")
    
    # Test state dict
    state = pf.state_dict()
    pf2 = PerceptionField()
    pf2.load_state_dict(state)
    assert len(pf2.tracked_objects) == len(pf.tracked_objects)
    print(f"  ✓ State dict round-trip successful")
    
    return pf


def test_integration():
    """Test integrated cognitive system."""
    print("\n" + "=" * 60)
    print("Testing Integrated Cognitive System")
    print("=" * 60)
    
    # Create all components
    tracker = CausalTracker()
    inducer = RuleInducer(tracker, min_observations=2)
    planner = GoalPlanner()
    perception = PerceptionField()
    
    # Simulate a simple episode
    grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0],  # Red trigger at (1,1)
        [0, 0, 0, 0, 0],
        [0, 0, 0, 4, 0],  # Yellow goal at (3,3)
        [0, 0, 0, 0, 0],
    ])
    position = (0, 0)
    
    # 1. Set objective
    goals = planner.set_objective("reach yellow")
    print(f"  1. Set objective: {len(goals)} goals")
    
    # 2. Perceive initial state
    perception.focus(position, reason="start")
    obs = perception.observe(grid)
    print(f"  2. Initial perception: fovea at {obs['fovea_center']}")
    
    # 3. Get next goal
    goal = planner.next_subgoal(grid, position)
    target = planner.compute_target_position(goal, grid, position)
    print(f"  3. Goal: {goal}, target: {target}")
    
    # 4. Simulate action and its effect
    grid_after = grid.copy()
    grid_after[1, 1] = 3  # Red changed to green
    
    tracker.record_step(position, action=ACTION_INTERACT, grid=grid)
    link = tracker.record_step((1, 1), action=ACTION_INTERACT, grid=grid_after)
    print(f"  4. Recorded action: {len(link.effects) if link else 0} effects")
    
    # 5. Induce rules
    rules = inducer.analyze_causal_links(force=True)
    print(f"  5. Induced {len(rules)} rules")
    
    # 6. Check for surprises (using ACTION_UP=1 for movement)
    perception.predict_next(grid, action=ACTION_UP, agent_position=position)
    obs2 = perception.observe(grid_after)
    print(f"  6. Surprises: {len(obs2['surprises'])}")
    
    # 7. Verify all components can be saved
    states = {
        'tracker': tracker.state_dict(),
        'inducer': inducer.state_dict(),
        'planner': planner.state_dict(),
        'perception': perception.state_dict(),
    }
    print(f"  7. All components serializable: {len(states)} states")
    
    print("\n  ✓ Integration test passed!")


def main():
    """Run all tests."""
    print("FLUX Unified Cognitive Layer Tests")
    print("=" * 60)
    
    try:
        tracker = test_causal_tracker()
        inducer = test_rule_inducer(tracker)
        planner = test_goal_planner()
        perception = test_perception_field()
        test_integration()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        
        # Print summaries
        print("\n--- Component Summaries ---\n")
        print(tracker.summary())
        print()
        print(inducer.summary())
        print()
        print(planner.summary())
        print()
        print(perception.summary())
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
