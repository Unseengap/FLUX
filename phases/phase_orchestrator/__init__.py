"""
Phase Orchestrator — VLM as FLUX Brain

This module enables the embedded VLM to orchestrate FLUX components
as callable cognitive tools, enabling adaptive multi-step reasoning.

Usage:
    from phases.phase_orchestrator import (
        FLUXOrchestrator,
        orchestrated_inference,
        FLUX_TOOLS,
    )
    
    from flux_model import FLUXModel
    
    # Load model
    model = FLUXModel.load('checkpoints/Flux-Apex-V1.flx')
    
    # Create orchestrator
    orchestrator = FLUXOrchestrator(model, device='cuda')
    
    # Run orchestrated inference
    response = orchestrated_inference(
        orchestrator=orchestrator,
        vlm=model.vlm,
        user_input="Solve this ARC puzzle",
        grid=[[0,1,0],[1,2,1],[0,1,0]],
    )
"""

from .flux_orchestrator import (
    FLUXOrchestrator,
    ComponentWrapper,
    ToolCall,
    ToolResult,
    orchestrated_inference,
)

from .tool_registry import (
    FLUX_TOOLS,
    ToolDefinition,
    ToolCategory,
    get_tool,
    get_tools_by_category,
    get_tool_names,
    generate_full_tool_prompt,
)

from .system_prompt import (
    FLUX_SYSTEM_PROMPT,
    FLUX_SYSTEM_PROMPT_SHORT,
    TOOL_USE_EXAMPLES,
)

from .embed_orchestration import (
    embed_orchestration,
    serialize_tool_definitions,
    build_orchestration_config,
    show_embedded_tools,
)


__all__ = [
    # Core classes
    'FLUXOrchestrator',
    'ComponentWrapper',
    'ToolCall',
    'ToolResult',
    
    # Main function
    'orchestrated_inference',
    
    # Embedding
    'embed_orchestration',
    'serialize_tool_definitions',
    'build_orchestration_config',
    'show_embedded_tools',
    
    # Tool registry
    'FLUX_TOOLS',
    'ToolDefinition',
    'ToolCategory',
    'get_tool',
    'get_tools_by_category',
    'get_tool_names',
    'generate_full_tool_prompt',
    
    # Prompts
    'FLUX_SYSTEM_PROMPT',
    'FLUX_SYSTEM_PROMPT_SHORT',
    'TOOL_USE_EXAMPLES',
]
