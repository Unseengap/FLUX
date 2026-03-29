#!/usr/bin/env python3
"""
Test script for FLUX ARC-AGI-3 Interface

Tests the complete ARC-AGI-3 interface:
1. GameAction enum — action parsing and delta computation
2. GameState enum — state transitions and terminal detection
3. ActionCommand — coordinate validation and API formatting
4. GameFrame — frame parsing from API responses
5. Scoring — RHAE calculation
6. Session — offline session management
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np


def test_game_action():
    """Test GameAction enum."""
    from arc_interface import GameAction
    
    print("\n" + "=" * 60)
    print("Testing GameAction")
    print("=" * 60)
    
    # Test all actions exist
    assert GameAction.RESET.value == 0
    assert GameAction.ACTION1.value == 1
    assert GameAction.ACTION2.value == 2
    assert GameAction.ACTION3.value == 3
    assert GameAction.ACTION4.value == 4
    assert GameAction.ACTION5.value == 5
    assert GameAction.ACTION6.value == 6
    assert GameAction.ACTION7.value == 7
    print("  ✓ All 8 actions defined correctly")
    
    # Test directional detection
    directional = GameAction.directional_actions()
    assert len(directional) == 4
    assert all(a.is_directional for a in directional)
    assert not GameAction.ACTION5.is_directional
    assert not GameAction.ACTION6.is_directional
    print("  ✓ Directional actions detected correctly")
    
    # Test complex detection
    assert GameAction.ACTION6.is_complex
    assert not GameAction.ACTION1.is_complex
    print("  ✓ Complex action (ACTION6) detected correctly")
    
    # Test deltas
    assert GameAction.ACTION1.delta == (-1, 0)  # up
    assert GameAction.ACTION2.delta == (1, 0)   # down
    assert GameAction.ACTION3.delta == (0, -1)  # left
    assert GameAction.ACTION4.delta == (0, 1)   # right
    print("  ✓ Movement deltas correct")
    
    # Test from_name parsing
    assert GameAction.from_name('up') == GameAction.ACTION1
    assert GameAction.from_name('ACTION1') == GameAction.ACTION1
    assert GameAction.from_name('click') == GameAction.ACTION6
    assert GameAction.from_name('undo') == GameAction.ACTION7
    assert GameAction.from_name('INTERACT') == GameAction.ACTION5
    print("  ✓ Action parsing from names works")
    
    # Test semantic names
    assert GameAction.ACTION1.semantic_name == 'up'
    assert GameAction.ACTION5.semantic_name == 'interact'
    assert GameAction.ACTION6.semantic_name == 'click'
    print("  ✓ Semantic names correct")
    
    print("\n  All GameAction tests passed!")


def test_game_state():
    """Test GameState enum."""
    from arc_interface import GameState
    
    print("\n" + "=" * 60)
    print("Testing GameState")
    print("=" * 60)
    
    # Test states exist
    assert GameState.NOT_FINISHED.value == 0
    assert GameState.WIN.value == 1
    assert GameState.GAME_OVER.value == 2
    print("  ✓ All 3 states defined")
    
    # Test terminal detection
    assert not GameState.NOT_FINISHED.is_terminal
    assert GameState.WIN.is_terminal
    assert GameState.GAME_OVER.is_terminal
    print("  ✓ Terminal state detection works")
    
    # Test success detection
    assert GameState.WIN.is_success
    assert not GameState.GAME_OVER.is_success
    assert not GameState.NOT_FINISHED.is_success
    print("  ✓ Success detection works")
    
    # Test from_string
    assert GameState.from_string('NOT_FINISHED') == GameState.NOT_FINISHED
    assert GameState.from_string('WIN') == GameState.WIN
    assert GameState.from_string('GAME_OVER') == GameState.GAME_OVER
    print("  ✓ String parsing works")
    
    print("\n  All GameState tests passed!")


def test_action_command():
    """Test ActionCommand."""
    from arc_interface import GameAction, ActionCommand
    
    print("\n" + "=" * 60)
    print("Testing ActionCommand")
    print("=" * 60)
    
    # Test simple action
    cmd1 = ActionCommand(GameAction.ACTION1)
    assert cmd1.action == GameAction.ACTION1
    assert cmd1.x is None
    assert cmd1.y is None
    print("  ✓ Simple action command created")
    
    # Test complex action with coordinates
    cmd2 = ActionCommand(GameAction.ACTION6, x=10, y=20)
    assert cmd2.x == 10
    assert cmd2.y == 20
    print("  ✓ Complex action with coordinates created")
    
    # Test coordinate validation
    try:
        ActionCommand(GameAction.ACTION6)  # Missing coordinates
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    print("  ✓ Coordinate validation works")
    
    try:
        ActionCommand(GameAction.ACTION6, x=100, y=20)  # x out of range
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    print("  ✓ Coordinate range validation works")
    
    # Test API dict generation
    api_dict = cmd2.to_api_dict('ls20-test', 'guid-123', 'card-456')
    assert api_dict['game_id'] == 'ls20-test'
    assert api_dict['guid'] == 'guid-123'
    assert api_dict['card_id'] == 'card-456'
    assert api_dict['x'] == 10
    assert api_dict['y'] == 20
    print("  ✓ API dict generation works")
    
    # Test API endpoint
    assert cmd1.api_endpoint == '/api/cmd/ACTION1'
    assert cmd2.api_endpoint == '/api/cmd/ACTION6'
    assert ActionCommand(GameAction.RESET).api_endpoint == '/api/cmd/RESET'
    print("  ✓ API endpoint generation works")
    
    print("\n  All ActionCommand tests passed!")


def test_game_frame():
    """Test GameFrame parsing."""
    from arc_interface import GameAction, GameState, GameFrame
    
    print("\n" + "=" * 60)
    print("Testing GameFrame")
    print("=" * 60)
    
    # Mock API response
    response = {
        'game_id': 'ls20-test',
        'frame': [[0, 0, 1], [0, 2, 0], [3, 0, 0]],
        'state': 'NOT_FINISHED',
        'score': 5,
        'guid': 'abc-123',
        'available_actions': [1, 2, 3, 4, 5],
        'levels_completed': 1
    }
    
    frame = GameFrame.from_api_response(response)
    
    assert frame.game_id == 'ls20-test'
    assert frame.grid.shape == (3, 3)
    assert frame.state == GameState.NOT_FINISHED
    assert frame.score == 5
    assert frame.guid == 'abc-123'
    assert frame.levels_completed == 1
    print("  ✓ Frame parsing works")
    
    # Test available actions
    assert GameAction.ACTION1 in frame.available_actions
    assert GameAction.ACTION5 in frame.available_actions
    assert GameAction.ACTION6 not in frame.available_actions
    assert frame.can_take(GameAction.ACTION1)
    assert not frame.can_take(GameAction.ACTION6)
    print("  ✓ Available actions tracked correctly")
    
    # Test terminal state
    assert not frame.is_terminal
    
    win_response = {**response, 'state': 'WIN'}
    win_frame = GameFrame.from_api_response(win_response)
    assert win_frame.is_terminal
    assert win_frame.state.is_success
    print("  ✓ Terminal state detection works")
    
    print("\n  All GameFrame tests passed!")


def test_scoring():
    """Test RHAE scoring."""
    from arc_interface import LevelScore, GameScore
    
    print("\n" + "=" * 60)
    print("Testing RHAE Scoring")
    print("=" * 60)
    
    # Test level scoring: (human_baseline / ai_actions)²
    
    # Exact match: score = 1.0
    level1 = LevelScore(0, ai_actions=10, human_baseline_actions=10, completed=True)
    assert abs(level1.raw_score - 1.0) < 0.001
    print(f"  ✓ Exact match score: {level1.raw_score:.4f}")
    
    # 2x slower: score = (10/20)² = 0.25
    level2 = LevelScore(1, ai_actions=20, human_baseline_actions=10, completed=True)
    assert abs(level2.raw_score - 0.25) < 0.001
    print(f"  ✓ 2x slower score: {level2.raw_score:.4f}")
    
    # Faster than human: capped at 1.0
    level3 = LevelScore(2, ai_actions=5, human_baseline_actions=10, completed=True)
    assert abs(level3.raw_score - 1.0) < 0.001
    print(f"  ✓ Faster-than-human score (capped): {level3.raw_score:.4f}")
    
    # Not completed: score = 0
    level4 = LevelScore(3, ai_actions=10, human_baseline_actions=10, completed=False)
    assert level4.raw_score == 0.0
    print(f"  ✓ Incomplete level score: {level4.raw_score:.4f}")
    
    # Test game scoring with weighted average
    game = GameScore(
        game_id='test',
        level_scores=[level1, level2]
    )
    # Weighted: (1.0 * 1 + 0.25 * 2) / (1 + 2) = 1.5 / 3 = 0.5
    expected = (1.0 * 1 + 0.25 * 2) / (1 + 2)
    assert abs(game.weighted_score - expected) < 0.001
    print(f"  ✓ Weighted game score: {game.weighted_score:.4f}")
    
    # Test completion rate
    game2 = GameScore(
        game_id='test',
        level_scores=[level1, level2, level4]  # 2 completed, 1 not
    )
    assert abs(game2.completion_rate - (2/3)) < 0.001
    print(f"  ✓ Completion rate: {game2.completion_rate:.2%}")
    
    print("\n  All scoring tests passed!")


def test_utility_functions():
    """Test utility functions."""
    from arc_interface import (
        grid_diff, find_agent_position, find_goal_position,
        get_action_delta, apply_action_to_position
    )
    
    print("\n" + "=" * 60)
    print("Testing Utility Functions")
    print("=" * 60)
    
    # Test grid_diff
    grid1 = np.array([[0, 1, 0], [0, 0, 0], [2, 0, 0]])
    grid2 = np.array([[0, 0, 0], [0, 1, 0], [2, 0, 0]])
    changes = grid_diff(grid1, grid2)
    assert len(changes) == 2  # (0,1) and (1,1) changed
    print(f"  ✓ Grid diff detected {len(changes)} changes")
    
    # Test find_agent_position
    agent_pos = find_agent_position(grid2, agent_color=1)
    assert agent_pos == (1, 1)
    print(f"  ✓ Found agent at {agent_pos}")
    
    # Test find_goal_position
    goal_pos = find_goal_position(grid2, goal_color=2)
    assert goal_pos == (2, 0)
    print(f"  ✓ Found goal at {goal_pos}")
    
    # Test get_action_delta
    assert get_action_delta(1) == (-1, 0)  # up
    assert get_action_delta(4) == (0, 1)   # right
    assert get_action_delta(5) == (0, 0)   # interact - no movement
    print("  ✓ Action deltas correct")
    
    # Test apply_action_to_position
    new_pos = apply_action_to_position((1, 1), action=1, grid_shape=(8, 8))
    assert new_pos == (0, 1)  # moved up
    print(f"  ✓ Applied action: (1,1) + up = {new_pos}")
    
    # Test boundary clamping
    edge_pos = apply_action_to_position((0, 0), action=1, grid_shape=(8, 8))
    assert edge_pos == (0, 0)  # can't move up from row 0
    print(f"  ✓ Boundary clamping works")
    
    print("\n  All utility tests passed!")


def test_session_offline():
    """Test offline session management."""
    from arc_session import create_session, OperationMode
    
    print("\n" + "=" * 60)
    print("Testing Offline Session")
    print("=" * 60)
    
    # Create offline session
    session = create_session(mode='offline')
    assert session.config.operation_mode == OperationMode.OFFLINE
    print("  ✓ Offline session created")
    
    # Open scorecard
    scorecard = session.open_scorecard(tags=['test'])
    assert scorecard is not None
    assert scorecard.card_id is not None
    assert scorecard.is_open
    print(f"  ✓ Scorecard opened: {scorecard.card_id[:8]}...")
    
    # Create mock environment
    env = session.make_env('test-game')
    assert env is not None
    print("  ✓ Mock environment created")
    
    # Reset
    frame = session.reset(env)
    assert frame is not None
    assert frame.grid is not None
    print(f"  ✓ Reset returned frame with shape {frame.grid.shape}")
    
    # Close scorecard
    result = session.close_scorecard()
    assert result is not None
    assert not result.is_open
    print(f"  ✓ Scorecard closed at {result.closed_at}")
    
    print("\n  All session tests passed!")


def run_all_tests():
    """Run all ARC interface tests."""
    print("\n" + "=" * 60)
    print("FLUX ARC-AGI-3 Interface Tests")
    print("=" * 60)
    
    tests = [
        ("GameAction", test_game_action),
        ("GameState", test_game_state),
        ("ActionCommand", test_action_command),
        ("GameFrame", test_game_frame),
        ("RHAE Scoring", test_scoring),
        ("Utility Functions", test_utility_functions),
        ("Offline Session", test_session_offline),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n  ✗ {name} FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n  ✓ ALL ARC-AGI-3 INTERFACE TESTS PASSED")
    else:
        print(f"\n  ✗ {failed} tests failed")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
