"""
Test Phase Autonomous — Self-contained FLUX architecture

Tests FluxToolExecutor, CodeSandbox, DocumentIngester, and GoalPlanner.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


def test_tool_executor():
    """Test FluxToolExecutor."""
    print("Testing FluxToolExecutor...")
    
    from phases.phase_autonomous import FluxToolExecutor
    
    flx_state = {
        'version': '8.2-test',
        'cse': {},
        'field': {},
        'memory': {'state_dict': {'episodic': {'metadata': []}}},
        'user_tools': {},
    }
    
    executor = FluxToolExecutor(flx_state, device='cpu')
    print(f"  ✓ Created executor")
    
    # Test encode_text
    result = executor.execute('encode_text', {'text': 'Hello FLUX'})
    assert result.success, f"encode_text failed: {result.error}"
    print(f"  ✓ encode_text: {result.result}")
    
    # Test store_memory
    result = executor.execute('store_memory', {'content': 'Test fact', 'importance': 0.9})
    assert result.success, f"store_memory failed: {result.error}"
    print(f"  ✓ store_memory: {result.result}")
    
    # Test query_memory
    result = executor.execute('query_memory', {'query': 'Test', 'limit': 5})
    assert result.success, f"query_memory failed: {result.error}"
    assert len(result.result) > 0, "Should find stored memory"
    print(f"  ✓ query_memory: found {len(result.result)} results")
    
    # Test get_curiosity_map
    result = executor.execute('get_curiosity_map', {'grid_size': [5, 5]})
    assert result.success, f"get_curiosity_map failed: {result.error}"
    print(f"  ✓ get_curiosity_map: {result.result['hotspots']}")
    
    # Test list_tools
    result = executor.execute('list_tools', {})
    assert result.success, f"list_tools failed: {result.error}"
    print(f"  ✓ list_tools: {len(result.result['builtin'])} builtin tools")
    
    return True


def test_code_sandbox():
    """Test CodeSandbox."""
    print("\nTesting CodeSandbox...")
    
    from phases.phase_autonomous import CodeSandbox
    
    sandbox = CodeSandbox(timeout=10)
    print(f"  ✓ Created sandbox")
    
    # Test simple calculation
    result = sandbox.execute("print(2 + 2)")
    assert result.success, f"Simple calc failed: {result.error}"
    assert "4" in result.output, f"Expected 4 in output: {result.output}"
    print(f"  ✓ Simple calc: {result.output.strip()}")
    
    # Test math module
    result = sandbox.execute("import math\nprint(math.sqrt(16))")
    assert result.success, f"Math import failed: {result.error}"
    assert "4" in result.output, f"Expected 4 in output: {result.output}"
    print(f"  ✓ Math module: {result.output.strip()}")
    
    # Test forbidden import
    result = sandbox.execute("import os")
    assert not result.success, "OS import should be blocked"
    print(f"  ✓ Blocked forbidden import: {result.error}")
    
    # Test list comprehension
    result = sandbox.execute("result = [x**2 for x in range(5)]\nprint(result)")
    assert result.success, f"List comp failed: {result.error}"
    print(f"  ✓ List comprehension: {result.output.strip()}")
    
    # Test expression result
    result = sandbox.execute("42")
    assert result.success and result.result == 42, f"Expression result failed: {result.error}"
    print(f"  ✓ Expression result: {result.result}")
    
    return True


def test_document_ingester():
    """Test DocumentIngester."""
    print("\nTesting DocumentIngester...")
    
    from phases.phase_autonomous import DocumentIngester
    
    flx_state = {
        'version': '8.2-test',
        'cse': {},
        'memory': {'state_dict': {'episodic': {'metadata': []}}},
        'documents': {},
    }
    
    ingester = DocumentIngester(flx_state, wave_dim=432)
    print(f"  ✓ Created ingester")
    
    # Ingest a text document
    content = "This is a test document about ARC puzzles. Grid transformations include rotation, reflection, and color mapping."
    result = ingester.ingest(content, "test_doc.txt")
    
    assert result.success, f"Ingest failed: {result.error}"
    print(f"  ✓ Ingested: {result.filename}, {result.total_chunks} chunks")
    
    # List documents
    docs = ingester.list_documents()
    assert len(docs) == 1, f"Expected 1 document, got {len(docs)}"
    print(f"  ✓ Listed documents: {len(docs)}")
    
    # Search
    results = ingester.search("ARC puzzles")
    assert len(results) > 0, "Should find search results"
    print(f"  ✓ Search found {len(results)} results")
    
    # Ingest JSON
    json_content = '{"name": "test", "values": [1, 2, 3]}'
    result = ingester.ingest(json_content, "data.json")
    assert result.success, f"JSON ingest failed: {result.error}"
    print(f"  ✓ Ingested JSON: {result.total_chunks} chunks")
    
    # Delete document
    deleted = ingester.delete_document("test_doc.txt")
    assert deleted, "Delete should succeed"
    docs = ingester.list_documents()
    assert len(docs) == 1, "Should have 1 document left"
    print(f"  ✓ Deleted document")
    
    return True


def test_goal_planner():
    """Test GoalPlanner."""
    print("\nTesting GoalPlanner...")
    
    from phases.phase_autonomous import GoalPlanner, Goal, Step, FluxToolExecutor
    
    flx_state = {
        'version': '8.2-test',
        'goals': {},
        'goal_patterns': {},
        'memory': {'state_dict': {'episodic': {'metadata': []}}},
    }
    
    executor = FluxToolExecutor(flx_state)
    planner = GoalPlanner(flx_state, executor)
    print(f"  ✓ Created planner")
    
    # Create a goal
    goal = planner.create_goal(
        description="Test goal with memory operations",
        steps=[
            Step("Store a fact", "store_memory", {"content": "Goal test fact", "importance": 0.7}),
            Step("Query memory", "query_memory", {"query": "Goal test", "limit": 3}),
        ],
        priority=0.8,
        triggers=["test:true"],
    )
    
    print(f"  ✓ Created goal: {goal.id}")
    assert goal.status.value == "pending"
    
    # List goals
    goals = planner.list_goals()
    assert len(goals) == 1, f"Expected 1 goal, got {len(goals)}"
    print(f"  ✓ Listed goals: {len(goals)}")
    
    # Execute goal
    result = planner.execute_goal(goal.id)
    print(f"  ✓ Executed goal: {result.status.value}, {result.completed_steps}/{result.total_steps} steps")
    assert result.success, f"Goal execution failed: {result.error}"
    assert result.completed_steps == 2, "Should complete both steps"
    
    # Check triggers
    triggered = planner.check_triggers({"test": "true"})
    # Goal is now completed, so shouldn't trigger
    print(f"  ✓ Trigger check: {len(triggered)} triggered")
    
    # Get stats
    stats = planner.get_stats()
    print(f"  ✓ Stats: {stats}")
    assert stats['total_goals'] == 1
    
    return True


def test_autonomous_agent():
    """Test AutonomousAgent."""
    print("\nTesting AutonomousAgent...")
    
    from phases.phase_autonomous import AutonomousAgent, create_agent
    
    # Create agent with empty state
    agent = create_agent(device='cpu')
    print(f"  ✓ Created agent")
    
    # Test chat
    response = agent.chat("Hello! Remember that I like ARC puzzles.")
    print(f"  ✓ Chat response: {response[:80]}...")
    
    # Test recall
    response = agent.chat("What do you know about ARC?")
    print(f"  ✓ Recall response: {response[:80]}...")
    
    # Test calculation
    response = agent.chat("calculate 2 + 2")
    assert "4" in response, f"Should calculate 4: {response}"
    print(f"  ✓ Calculate: {response}")
    
    # Test help
    response = agent.chat("help")
    assert "Memory" in response or "memory" in response, "Help should mention memory"
    print(f"  ✓ Help works")
    
    # Get stats
    stats = agent.get_stats()
    print(f"  ✓ Stats: turn_count={stats['turn_count']}")
    
    return True


def test_coder_pool():
    """Test CoderPool (parallel sandbox execution via coder model)."""
    print("\nTesting CoderPool...")
    
    from phases.phase_autonomous import CoderPool, CodingTask, TaskStatus
    
    flx_state = {'version': 'test'}
    pool = CoderPool(flx_state, max_sandboxes=2)
    print(f"  ✓ Created coder pool with {pool.max_sandboxes} sandboxes")
    
    # Test single task - compound interest
    task = CodingTask(
        description="Calculate compound interest",
        context={'principal': 1000, 'rate': 0.05, 'years': 10}
    )
    result = pool.execute_task(task)
    assert result.success, f"Compound interest failed: {result.error}"
    # Output contains "Final Amount" or result was computed
    assert 'amount' in result.output.lower() or result.result_value, f"Expected result: {result.output}"
    print(f"  ✓ Compound interest: {result.output.strip()[:60]}...")
    
    # Test Fibonacci
    task2 = CodingTask(
        description="Generate the first 8 Fibonacci numbers",
        context={'n': 8}
    )
    result2 = pool.execute_task(task2)
    assert result2.success, f"Fibonacci failed: {result2.error}"
    print(f"  ✓ Fibonacci: {result2.output.strip()[:50]}...")
    
    # Test parallel execution
    parallel_tasks = [
        CodingTask("Calculate factorial of 10", context={'n': 10}),
        CodingTask("Find primes up to 30", context={'n': 30}),
    ]
    results = pool.execute_parallel(parallel_tasks)
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    successful = sum(1 for r in results if r.success)
    print(f"  ✓ Parallel execution: {successful}/2 tasks succeeded")
    
    # Test delegate API
    delegate_result = pool.delegate(
        description="Sum a list of numbers",
        context={'data': [1, 2, 3, 4, 5]}
    )
    assert delegate_result['success'] or 'output' in delegate_result, f"Delegate failed: {delegate_result}"
    print(f"  ✓ Delegate API works")
    
    # Test stats
    stats = pool.get_stats()
    assert stats['total_tasks'] >= 4, f"Expected >= 4 tasks, got {stats['total_tasks']}"
    print(f"  ✓ Stats: {stats['total_tasks']} tasks, {stats['successful_tasks']} successful")
    
    pool.shutdown()
    print(f"  ✓ Pool shutdown cleanly")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Phase Autonomous Tests")
    print("=" * 60)
    
    tests = [
        test_tool_executor,
        test_code_sandbox,
        test_document_ingester,
        test_goal_planner,
        test_autonomous_agent,
        test_coder_pool,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            import traceback
            print(f"  ✗ {test.__name__} failed: {e}")
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
