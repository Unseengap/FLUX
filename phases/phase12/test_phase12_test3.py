#!/usr/bin/env python3
"""
test_phase12_test3.py — Test: End-to-End Game Play

Tests:
1. Agent initialization
2. Game session management
3. Observation processing
4. Action decision making
5. Effect recording
6. Game memory persistence
7. Model serialization
"""

import sys
from pathlib import Path
import tempfile
import numpy as np

# Add project root and phase12
PHASE_DIR = Path(__file__).parent
PROJECT_ROOT = PHASE_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PHASE_DIR))

from flux_utils import PhaseResults, get_device


def test_agent_initialization():
    """Test FLUXMultiAgent initialization."""
    from flux_multi_agent import FLUXMultiAgent, MultiAgentConfig
    
    passed = 0
    total = 0
    
    # Test 1: Basic initialization
    total += 1
    try:
        config = MultiAgentConfig(
            enable_vision=False,  # Skip vision for faster test
            load_in_4bit=True,
            verbose=False,
        )
        agent = FLUXMultiAgent(config=config)
        assert agent is not None
        print(f"  ✓ Agent initialized on {agent.device}")
        passed += 1
    except Exception as e:
        print(f"  ✗ Agent initialization failed: {e}")
        return passed, total
    
    # Test 2: Components loaded
    total += 1
    try:
        stats = agent.get_stats()
        components = stats['components']
        active = sum(1 for v in components.values() if v)
        assert active >= 2, "At least 2 components should be active"
        print(f"  ✓ Components active: {active}/{len(components)}")
        passed += 1
    except Exception as e:
        print(f"  ✗ Component check failed: {e}")
    
    # Test 3: LLM bridge ready
    total += 1
    try:
        llm_info = agent.llm_bridge.get_info()
        assert llm_info['loaded'], "LLM should be loaded"
        print(f"  ✓ LLM ready: {llm_info['model_name']}")
        passed += 1
    except Exception as e:
        print(f"  ✗ LLM check failed: {e}")
    
    return passed, total


def test_game_session():
    """Test game session lifecycle."""
    from flux_multi_agent import FLUXMultiAgent, MultiAgentConfig
    
    config = MultiAgentConfig(enable_vision=False, verbose=False)
    agent = FLUXMultiAgent(config=config)
    
    passed = 0
    total = 0
    
    # Test 1: Reset for game
    total += 1
    try:
        agent.reset("ls20")
        assert agent.current_game == "ls20"
        assert agent.observation_count == 0
        assert len(agent.action_history) == 0
        print(f"  ✓ Game reset: {agent.current_game}")
        passed += 1
    except Exception as e:
        print(f"  ✗ Reset failed: {e}")
    
    # Test 2: Game memory created
    total += 1
    try:
        memory = agent.current_game_memory
        assert memory is not None
        assert memory.game_id == "ls20"
        print(f"  ✓ Game memory: session {memory.session_id[:20]}...")
        passed += 1
    except Exception as e:
        print(f"  ✗ Game memory failed: {e}")
    
    # Test 3: End game
    total += 1
    try:
        agent.end_game(final_score=50.0, final_state='WON')
        print(f"  ✓ Game ended successfully")
        passed += 1
    except Exception as e:
        print(f"  ✗ End game failed: {e}")
    
    return passed, total


def test_observation_processing():
    """Test observation processing."""
    from flux_multi_agent import FLUXMultiAgent, MultiAgentConfig
    
    config = MultiAgentConfig(enable_vision=False, verbose=False)
    agent = FLUXMultiAgent(config=config)
    agent.reset("test_game")
    
    passed = 0
    total = 0
    
    # Test 1: Process observation
    total += 1
    try:
        grid = [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [0, 0, 1, 0, 0],
            [2, 0, 0, 0, 3],
            [0, 4, 0, 5, 0],
        ]
        result = agent.observe(grid, position=(2, 2))
        
        assert result['position'] == (2, 2)
        assert agent.observation_count == 1
        print(f"  ✓ Observation processed: step {agent.observation_count}")
        passed += 1
    except Exception as e:
        print(f"  ✗ Observation failed: {e}")
    
    # Test 2: Multiple observations
    total += 1
    try:
        for i in range(5):
            agent.observe(grid, position=(2, 2 + i % 3))
        assert agent.observation_count == 6
        print(f"  ✓ Multiple observations: {agent.observation_count} total")
        passed += 1
    except Exception as e:
        print(f"  ✗ Multiple observations failed: {e}")
    
    # Test 3: Position finding
    total += 1
    try:
        # Grid with unique cell
        unique_grid = [[0, 0, 0], [0, 5, 0], [0, 0, 0]]
        pos = agent.find_position(unique_grid)
        assert pos == (1, 1), f"Expected (1,1), got {pos}"
        print(f"  ✓ Position detection: {pos}")
        passed += 1
    except Exception as e:
        print(f"  ✗ Position detection failed: {e}")
    
    return passed, total


def test_action_decision():
    """Test action decision making."""
    from flux_multi_agent import FLUXMultiAgent, MultiAgentConfig
    
    config = MultiAgentConfig(enable_vision=False, verbose=False)
    agent = FLUXMultiAgent(config=config)
    agent.reset("test_game")
    
    passed = 0
    total = 0
    
    grid = [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [0, 0, 1, 0, 0],
        [2, 0, 0, 0, 3],
        [0, 4, 0, 5, 0],
    ]
    
    # Test 1: Get action decision
    total += 1
    try:
        action, reasoning = agent.decide_action(
            frame=grid,
            position=(2, 2),
            available_actions=[1, 2, 3, 4],
        )
        
        assert action in [1, 2, 3, 4], f"Action should be valid: {action}"
        assert len(reasoning) > 0, "Should have reasoning"
        print(f"  ✓ Decision: action={action}, reasoning length={len(reasoning)}")
        passed += 1
    except Exception as e:
        print(f"  ✗ Action decision failed: {e}")
    
    # Test 2: Action recorded in history
    total += 1
    try:
        assert len(agent.action_history) == 1
        assert agent.action_history[0]['action'] == action
        print(f"  ✓ Action recorded in history")
        passed += 1
    except Exception as e:
        print(f"  ✗ History recording failed: {e}")
    
    # Test 3: Multiple decisions
    total += 1
    try:
        for i in range(3):
            agent.decide_action(grid, (2, 2), [1, 2, 3, 4])
        assert len(agent.action_history) == 4
        print(f"  ✓ Multiple decisions: {len(agent.action_history)} actions")
        passed += 1
    except Exception as e:
        print(f"  ✗ Multiple decisions failed: {e}")
    
    return passed, total


def test_effect_recording():
    """Test action effect recording."""
    from flux_multi_agent import FLUXMultiAgent, MultiAgentConfig
    
    config = MultiAgentConfig(
        enable_vision=False,
        enable_causal_learning=True,
        verbose=False,
    )
    agent = FLUXMultiAgent(config=config)
    agent.reset("test_game")
    
    passed = 0
    total = 0
    
    # Test 1: Record simple change
    total += 1
    try:
        old_grid = [[0, 1], [2, 3]]
        new_grid = [[0, 1], [2, 4]]  # Cell (1,1) changed
        
        changes = agent.record_effect(
            old_grid=old_grid,
            new_grid=new_grid,
            action=1,
            position=(0, 0),
        )
        
        assert len(changes) == 1
        assert changes[0]['position'] == (1, 1)
        assert changes[0]['old'] == 3
        assert changes[0]['new'] == 4
        print(f"  ✓ Change detected: {changes[0]}")
        passed += 1
    except Exception as e:
        print(f"  ✗ Effect recording failed: {e}")
    
    # Test 2: Multiple changes
    total += 1
    try:
        old_grid = [[0, 1, 2], [3, 4, 5]]
        new_grid = [[0, 9, 2], [3, 4, 9]]  # Two changes
        
        changes = agent.record_effect(old_grid, new_grid, 2, (0, 1))
        assert len(changes) == 2
        print(f"  ✓ Multiple changes detected: {len(changes)}")
        passed += 1
    except Exception as e:
        print(f"  ✗ Multiple changes failed: {e}")
    
    # Test 3: No changes
    total += 1
    try:
        grid = [[1, 2], [3, 4]]
        changes = agent.record_effect(grid, grid, 3, (0, 0))
        assert len(changes) == 0
        print(f"  ✓ No changes correctly detected")
        passed += 1
    except Exception as e:
        print(f"  ✗ No changes detection failed: {e}")
    
    return passed, total


def test_game_memory():
    """Test game memory persistence."""
    from game_memory import GameMemory, GameMemoryManager
    
    passed = 0
    total = 0
    
    # Test 1: Create and use memory
    total += 1
    try:
        memory = GameMemory("test_game")
        memory.record_action(
            position=(1, 1),
            action_id=1,
            action_name="UP",
            reasoning="Testing",
            confidence=0.9,
        )
        assert len(memory.action_memories) == 1
        print(f"  ✓ Memory recording works")
        passed += 1
    except Exception as e:
        print(f"  ✗ Memory recording failed: {e}")
    
    # Test 2: Save and load
    total += 1
    try:
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        memory.save(temp_path)
        loaded = GameMemory.load(temp_path)
        
        assert loaded.game_id == "test_game"
        Path(temp_path).unlink()  # Cleanup
        print(f"  ✓ Save/load cycle works")
        passed += 1
    except Exception as e:
        print(f"  ✗ Save/load failed: {e}")
    
    # Test 3: Memory manager
    total += 1
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GameMemoryManager(save_dir=tmpdir)
            
            mem1 = manager.get_memory("game1")
            mem2 = manager.get_memory("game2")
            
            mem1.record_action((0, 0), 1, "UP", "test", 0.5)
            mem2.record_action((0, 0), 2, "DOWN", "test", 0.5)
            
            manager.save_all()
            
            assert len(manager.get_all_games()) == 2
        print(f"  ✓ Memory manager works")
        passed += 1
    except Exception as e:
        print(f"  ✗ Memory manager failed: {e}")
    
    return passed, total


def test_serialization():
    """Test model serialization."""
    from flux_multi_agent import FLUXMultiAgent, MultiAgentConfig
    
    config = MultiAgentConfig(enable_vision=False, verbose=False)
    agent = FLUXMultiAgent(config=config)
    
    passed = 0
    total = 0
    
    # Test 1: Save model
    total += 1
    try:
        with tempfile.NamedTemporaryFile(suffix='.flx', delete=False) as f:
            temp_path = f.name
        
        # Play a brief game
        agent.reset("test_save")
        grid = [[0, 1], [2, 3]]
        agent.observe(grid, (0, 0))
        agent.decide_action(grid, (0, 0), [1, 2, 3, 4])
        
        # Save
        saved_path = agent.save_flx(temp_path, upload_to_hf=False)
        assert Path(saved_path).exists()
        
        size_kb = Path(saved_path).stat().st_size / 1024
        print(f"  ✓ Model saved: {size_kb:.1f} KB")
        passed += 1
    except Exception as e:
        print(f"  ✗ Save failed: {e}")
        return passed, total
    
    # Test 2: Load model
    total += 1
    try:
        loaded = FLUXMultiAgent.load(temp_path)
        assert loaded is not None
        print(f"  ✓ Model loaded successfully")
        passed += 1
    except Exception as e:
        print(f"  ✗ Load failed: {e}")
    
    # Cleanup
    try:
        Path(temp_path).unlink()
    except:
        pass
    
    return passed, total


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  Phase 12 Test 3: End-to-End Game Play")
    print("=" * 60 + "\n")
    
    results = PhaseResults(phase=12, component_name="Multi-Modal Agent")
    
    total_passed = 0
    total_tests = 0
    
    # Run test suites
    print("Testing agent initialization...")
    passed, total = test_agent_initialization()
    total_passed += passed
    total_tests += total
    results.add_test("Initialization", passed=passed == total,
                     score=passed/total if total > 0 else 0, threshold=0.8)
    
    print("\nTesting game session...")
    passed, total = test_game_session()
    total_passed += passed
    total_tests += total
    results.add_test("Game Session", passed=passed == total,
                     score=passed/total if total > 0 else 0, threshold=0.8)
    
    print("\nTesting observation processing...")
    passed, total = test_observation_processing()
    total_passed += passed
    total_tests += total
    results.add_test("Observation", passed=passed == total,
                     score=passed/total if total > 0 else 0, threshold=0.8)
    
    print("\nTesting action decision...")
    passed, total = test_action_decision()
    total_passed += passed
    total_tests += total
    results.add_test("Action Decision", passed=passed == total,
                     score=passed/total if total > 0 else 0, threshold=0.8)
    
    print("\nTesting effect recording...")
    passed, total = test_effect_recording()
    total_passed += passed
    total_tests += total
    results.add_test("Effect Recording", passed=passed == total,
                     score=passed/total if total > 0 else 0, threshold=0.8)
    
    print("\nTesting game memory...")
    passed, total = test_game_memory()
    total_passed += passed
    total_tests += total
    results.add_test("Game Memory", passed=passed == total,
                     score=passed/total if total > 0 else 0, threshold=0.8)
    
    print("\nTesting serialization...")
    passed, total = test_serialization()
    total_passed += passed
    total_tests += total
    results.add_test("Serialization", passed=passed == total,
                     score=passed/total if total > 0 else 0, threshold=0.8)
    
    # Summary
    print("\n" + "-" * 40)
    print(f"Total: {total_passed}/{total_tests} tests passed")
    
    overall = total_passed / total_tests if total_tests > 0 else 0
    results.add_test("Overall", passed=overall >= 0.7, score=overall, threshold=0.7)
    
    if overall >= 0.7:
        print("✓ Test 3 PASSED")
    else:
        print("✗ Test 3 FAILED")
    
    # Save results
    results.save()
    print(f"\nResults saved")
    
    return 0 if overall >= 0.7 else 1


if __name__ == "__main__":
    sys.exit(main())
