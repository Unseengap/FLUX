"""
Phase CLAW: Claude Code Harness Integration

This module integrates the claw-code harness (clean-room Python port of Claude Code)
directly into FLUX, enabling the same agentic tool-calling capabilities.

The claw harness provides:
- 922+ tool definitions (BashTool, FileEditTool, AgentTool, etc.)
- 1000+ command definitions (/branch, /add-dir, /agents, etc.)
- Session management and context tracking
- Query engine with budget and turn management
- Permission system for tool access control

Integration with FLUX:
- FLUX's native CSE → wave encoding for tool descriptions
- FLUX's field → knowledge storage for tool capabilities
- FLUX's orchestrator → executes claw tools via VLM
- FLUX's memory → persists claw sessions

History:
- March 31, 2026: Claude Code source leaked
- March 31, 2026: Clean-room Python port created in 24 hours
- April 1, 2026: Integrated into FLUX harness

Usage:
    from phases.phase_claw import (
        ClawHarness,
        get_claw_harness,
        execute_claw_tool,
        search_claw_tools,
    )
    
    # Initialize harness
    harness = get_claw_harness()
    
    # Get available tools
    tools = harness.list_tools()
    
    # Execute a tool
    result = harness.execute_tool('BashTool', command='ls -la')
    
    # Register with FLUX orchestrator
    from phases.phase_orchestrator import FLUXOrchestrator
    harness.register_with_orchestrator(orchestrator)
"""

# Core claw-code modules
from .commands import PORTED_COMMANDS, build_command_backlog
from .tools import PORTED_TOOLS, build_tool_backlog
from .parity_audit import ParityAuditResult, run_parity_audit
from .port_manifest import PortManifest, build_port_manifest
from .query_engine import QueryEnginePort, TurnResult
from .runtime import PortRuntime, RuntimeSession
from .session_store import StoredSession, load_session, save_session
from .system_init import build_system_init_message

# FLUX bridge (integration layer)
from .flux_bridge import (
    ClawHarness,
    FluxToolDefinition,
    claw_to_flux_tool,
    flux_to_claw_tool,
    get_claw_harness,
    register_claw_tools_with_orchestrator,
    execute_claw_tool,
    search_claw_tools,
)

__all__ = [
    # Core claw modules
    'ParityAuditResult',
    'PortManifest',
    'PortRuntime',
    'QueryEnginePort',
    'RuntimeSession',
    'StoredSession',
    'TurnResult',
    'PORTED_COMMANDS',
    'PORTED_TOOLS',
    'build_command_backlog',
    'build_port_manifest',
    'build_system_init_message',
    'build_tool_backlog',
    'load_session',
    'run_parity_audit',
    'save_session',
    
    # FLUX bridge
    'ClawHarness',
    'FluxToolDefinition',
    'claw_to_flux_tool',
    'flux_to_claw_tool',
    'get_claw_harness',
    'register_claw_tools_with_orchestrator',
    'execute_claw_tool',
    'search_claw_tools',
]

# Version info
__version__ = '1.0.0'
__claw_source__ = 'claw-code (clean-room Python port)'
__integration_date__ = '2026-04-01'
