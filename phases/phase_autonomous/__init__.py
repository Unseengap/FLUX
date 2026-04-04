"""
Phase Autonomous — Self-contained, self-modifying FLUX architecture.

This module implements the autonomous capabilities:
- FluxToolExecutor: Execute tools from native JSON format
- CodeSandbox: Safe Python execution
- DocumentIngester: Ingest documents into memory
- GoalPlanner: Proactive goal and planning system

Usage:
    from phases.phase_autonomous import (
        FluxToolExecutor,
        CodeSandbox,
        DocumentIngester,
        GoalPlanner,
        autonomous_inference,
    )
    
    # Load model
    import torch
    flx = torch.load('Flux-Apex-V1.flx', map_location='cpu', weights_only=False)
    
    # Create executor
    executor = FluxToolExecutor(flx)
    
    # Execute a tool
    result = executor.execute('query_memory', {'query': 'ARC rules'})
"""

from .tool_executor import (
    FluxToolExecutor,
    ToolExecution,
)

from .code_sandbox import (
    CodeSandbox,
    SandboxResult,
)

from .document_ingester import (
    DocumentIngester,
    DocumentChunk,
    IngestResult,
)

from .goal_planner import (
    GoalPlanner,
    Goal,
    Step,
    PlanResult,
)

from .coder_pool import (
    CoderPool,
    CoderPoolExecutor,
    CodingTask,
    TaskResult,
    TaskStatus,
)

from .autonomous_agent import (
    AutonomousAgent,
    autonomous_inference,
    create_agent,
    AgentResponse,
)


__all__ = [
    # Tool Execution
    'FluxToolExecutor',
    'ToolExecution',
    
    # Code Sandbox
    'CodeSandbox',
    'SandboxResult',
    
    # Document Ingestion
    'DocumentIngester',
    'DocumentChunk',
    'IngestResult',
    
    # Goal Planning
    'GoalPlanner',
    'Goal',
    'Step',
    'PlanResult',
    
    # Coder Pool (parallel sandbox execution)
    'CoderPool',
    'CoderPoolExecutor',
    'CodingTask',
    'TaskResult',
    'TaskStatus',
    
    # Main Agent
    'AutonomousAgent',
    'autonomous_inference',
    'create_agent',
    'AgentResponse',
]
