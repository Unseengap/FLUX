"""
Test Phase Orchestrator — Native JSON Function Calling

Tests the NativeJSONOrchestrator and native tool execution.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


def test_native_json_orchestrator():
    """Test NativeJSONOrchestrator initialization and tool loading."""
    print("Testing NativeJSONOrchestrator...")
    
    from phases.phase_orchestrator import (
        NativeJSONOrchestrator,
        FunctionCall,
        FunctionResult,
        load_native_tools,
    )
    
    # Load tools
    tools = load_native_tools()
    print(f"  ✓ Loaded {len(tools)} native tools")
    assert len(tools) > 0, "Should have at least one tool"
    
    # Create orchestrator with empty state
    flx_state = {
        'version': '8.2-test',
        'cse': {},
        'field': {},
        'memory': {'state_dict': {'episodic': {'metadata': []}}},
    }
    
    orchestrator = NativeJSONOrchestrator(flx_state, device='cpu')
    print(f"  ✓ Created orchestrator")
    
    # Get system prompt
    prompt = orchestrator.get_system_prompt()
    assert len(prompt) > 100, "System prompt should be substantial"
    print(f"  ✓ System prompt: {len(prompt)} chars")
    
    # Get tools for model
    model_tools = orchestrator.get_tools_for_model()
    assert len(model_tools) == len(tools), "Should return all tools"
    print(f"  ✓ Model tools: {len(model_tools)}")
    
    return True


def test_function_execution():
    """Test executing functions through the orchestrator."""
    print("\nTesting function execution...")
    
    from phases.phase_orchestrator import (
        NativeJSONOrchestrator,
        FunctionCall,
    )
    
    flx_state = {
        'version': '8.2-test',
        'cse': {'config': {'wave_dim': 432}},
        'memory': {'state_dict': {'episodic': {'metadata': []}}},
    }
    
    orchestrator = NativeJSONOrchestrator(flx_state, device='cpu')
    
    # Test encode_text
    call = FunctionCall(
        name='encode_text',
        arguments={'text': 'Hello, FLUX!'},
        raw_arguments='{"text": "Hello, FLUX!"}'
    )
    
    result = orchestrator.execute_function(call)
    print(f"  ✓ encode_text: success={result.success}")
    assert result.success, f"encode_text should succeed: {result.error}"
    assert 'wave_id' in result.result, "Should return wave_id"
    
    # Test query_memory
    call = FunctionCall(
        name='recall_memory',
        arguments={'query': 'test', 'limit': 3},
        raw_arguments='{"query": "test", "limit": 3}'
    )
    
    result = orchestrator.execute_function(call)
    print(f"  ✓ recall_memory: success={result.success}")
    assert result.success, f"recall_memory should succeed: {result.error}"
    
    # Test store_memory
    call = FunctionCall(
        name='store_memory',
        arguments={'content': 'Test memory', 'importance': 0.8},
        raw_arguments='{"content": "Test memory", "importance": 0.8}'
    )
    
    result = orchestrator.execute_function(call)
    print(f"  ✓ store_memory: success={result.success}")
    assert result.success, f"store_memory should succeed: {result.error}"
    
    # Verify memory stored
    call = FunctionCall(
        name='recall_memory',
        arguments={'query': 'Test', 'limit': 5},
        raw_arguments='{"query": "Test", "limit": 5}'
    )
    
    result = orchestrator.execute_function(call)
    assert result.success and len(result.result) > 0, "Should find stored memory"
    print(f"  ✓ Memory recall found stored content")
    
    return True


def test_legacy_xml_orchestrator():
    """Test the legacy XML-based FLUXOrchestrator."""
    print("\nTesting legacy XML orchestrator...")
    
    from phases.phase_orchestrator import (
        FLUXOrchestrator,
        ToolCall,
        FLUX_TOOLS,
    )
    
    # Check tool registry
    print(f"  ✓ FLUX_TOOLS has {len(FLUX_TOOLS)} tools")
    assert len(FLUX_TOOLS) > 0
    
    # Check specific tools exist
    required_tools = ['encode_text', 'encode_grid', 'query_field', 'recall_memory']
    for tool in required_tools:
        assert tool in FLUX_TOOLS, f"Missing required tool: {tool}"
    print(f"  ✓ All required tools present")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Phase Orchestrator Tests")
    print("=" * 60)
    
    tests = [
        test_native_json_orchestrator,
        test_function_execution,
        test_legacy_xml_orchestrator,
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
            print(f"  ✗ {test.__name__} failed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
