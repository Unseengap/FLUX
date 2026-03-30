"""
Demo: FLUX Orchestrator — VLM calling FLUX tools.

This demo shows how the VLM can orchestrate FLUX components
as tools for multi-step reasoning.

Run:
    python demo_orchestrator_demo1.py
"""

import sys
from pathlib import Path

# Add paths
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import torch
from flux_orchestrator import FLUXOrchestrator, ToolCall, ToolResult
from tool_registry import FLUX_TOOLS, get_tool, generate_full_tool_prompt
from system_prompt import FLUX_SYSTEM_PROMPT


def demo_tool_parsing():
    """Demo: Parse tool calls from VLM output."""
    print("\n" + "="*60)
    print("DEMO 1: Tool Call Parsing")
    print("="*60)
    
    # Simulated VLM output with tool calls
    vlm_output = """
    I see an input grid. Let me analyze it.
    
    <tool>encode_grid</tool>
    <params>{"grid": [[0,1,0],[1,2,1],[0,1,0]], "mode": "holistic"}</params>
    
    Now let me check for similar patterns.
    
    <tool>query_field</tool>
    <params>{"wave": "$LAST_WAVE", "top_k": 3}</params>
    
    Based on my analysis, this is a cross pattern centered on red.
    """
    
    # Create minimal orchestrator (without full model)
    class MockModel:
        state = {}
    
    orchestrator = FLUXOrchestrator(MockModel(), device='cpu')
    
    # Parse tool calls
    calls = orchestrator.parse_tool_calls(vlm_output)
    
    print(f"\nFound {len(calls)} tool calls:\n")
    for i, call in enumerate(calls, 1):
        print(f"  {i}. {call.name}")
        print(f"     Params: {call.params}")
        print()
    
    return calls


def demo_tool_registry():
    """Demo: Show available tools."""
    print("\n" + "="*60)
    print("DEMO 2: Tool Registry")
    print("="*60)
    
    print(f"\nTotal tools available: {len(FLUX_TOOLS)}")
    print("\nTools by category:\n")
    
    from tool_registry import ToolCategory, get_tools_by_category
    
    for category in ToolCategory:
        tools = get_tools_by_category(category)
        if tools:
            print(f"  {category.value.upper()} ({len(tools)} tools):")
            for tool in tools:
                print(f"    - {tool.name}: {tool.description[:50]}...")
            print()


def demo_system_prompt():
    """Demo: Show system prompt for VLM."""
    print("\n" + "="*60)
    print("DEMO 3: System Prompt Preview")
    print("="*60)
    
    # Show first 80 lines
    lines = FLUX_SYSTEM_PROMPT.split('\n')
    print(f"\nSystem prompt ({len(lines)} lines):\n")
    print('\n'.join(lines[:40]))
    print(f"\n... ({len(lines) - 40} more lines)")


def demo_mock_execution():
    """Demo: Mock tool execution flow."""
    print("\n" + "="*60)
    print("DEMO 4: Mock Tool Execution")
    print("="*60)
    
    # Simulated execution without real model
    print("\nSimulating ARC puzzle solving flow:\n")
    
    steps = [
        ("encode_grid", {"grid": [[0,1,0],[1,2,1],[0,1,0]], "mode": "holistic"}, 
         "Tensor shape [432], dtype=float32"),
        
        ("query_field", {"wave": "$LAST_WAVE", "top_k": 3},
         "3 matches: rotation_90 (0.89), mirror_h (0.67), identity (0.45)"),
        
        ("get_applicable_rules", {"trigger_color": 2, "trigger_action": 0},
         "Rule #7: 'Red center → 90° rotation' (confidence: 0.91)"),
        
        ("decode_grid", {"wave": "$LAST_WAVE", "grid_size": (3, 3)},
         "Grid [[0,1,0],[1,2,1],[0,1,0]]"),
    ]
    
    for i, (tool, params, result) in enumerate(steps, 1):
        print(f"  Step {i}: <tool>{tool}</tool>")
        print(f"          params: {params}")
        print(f"          → {result}")
        print()
    
    print("  Final answer: The grid has 90° rotational symmetry,")
    print("                so the output equals the input.")


def demo_integration_sketch():
    """Demo: Show how full integration would work."""
    print("\n" + "="*60)
    print("DEMO 5: Integration Sketch")
    print("="*60)
    
    print("""
    Full integration (requires Flux-Apex-V1.flx v5.0+):
    
    ```python
    from flux_model import FLUXModel
    from phases.phase_orchestrator import (
        FLUXOrchestrator,
        orchestrated_inference,
    )
    
    # Load model with embedded VLM
    model = FLUXModel.load('checkpoints/Flux-Apex-V1.flx')
    
    # Create orchestrator
    orchestrator = FLUXOrchestrator(model, device='cuda')
    
    # Run with tool orchestration
    response = orchestrated_inference(
        orchestrator=orchestrator,
        vlm=model.vlm,
        user_input="Solve this ARC puzzle",
        grid=[[0,1,0],[1,2,1],[0,1,0]],
        max_iterations=5,
        verbose=True,
    )
    
    print(response)
    ```
    
    The VLM will:
    1. Receive the system prompt with tool descriptions
    2. Generate response with <tool>...</tool> tags
    3. Orchestrator parses and executes tool calls
    4. Results injected back into VLM context
    5. VLM continues until no more tools needed
    6. Return final synthesized answer
    """)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("FLUX ORCHESTRATOR DEMO")
    print("VLM as Brain — FLUX Components as Tools")
    print("="*60)
    
    demo_tool_parsing()
    demo_tool_registry()
    demo_system_prompt()
    demo_mock_execution()
    demo_integration_sketch()
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("  1. Load Flux-Apex-V1.flx v5.0-vlm-embedded")
    print("  2. Initialize FLUXOrchestrator with loaded model")
    print("  3. Create VLM wrapper with generate() method")
    print("  4. Run orchestrated_inference() on real tasks")
    print()
