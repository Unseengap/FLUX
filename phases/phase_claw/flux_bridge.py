"""
FLUX Bridge — Connect Claw Harness to FLUX Orchestration

This module bridges the claw-code harness into FLUX's orchestration system,
enabling FLUX to use Claude Code's 922+ tools and 1000+ commands.

Architecture:
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                          FLUX Orchestrator                       │
    │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
    │  │  VLM Brain  │ ←→ │ Tool Router │ ←→ │ Native FLUX Tools   │  │
    │  │ (Qwen2.5)   │    │             │    │ (encode, query,     │  │
    │  └─────────────┘    │             │    │  memory, causal)    │  │
    │                     │             │    └─────────────────────┘  │
    │                     │             │              ↑               │
    │                     │             │    ┌─────────────────────┐  │
    │                     │             │ ←→ │ Claw Harness Bridge │  │
    │                     └─────────────┘    │ (BashTool, FileEdit,│  │
    │                                        │  AgentTool, etc.)   │  │
    │                                        └─────────────────────┘  │
    └─────────────────────────────────────────────────────────────────┘

Usage:
    from phases.phase_claw.flux_bridge import ClawHarness
    
    harness = ClawHarness()
    
    # Register with FLUX orchestrator
    harness.register_with_orchestrator(orchestrator)
    
    # Or use standalone
    result = harness.execute('BashTool', command='ls -la')
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple

# Import claw components
try:
    from phases.phase_claw.tools import (
        PORTED_TOOLS,
        get_tool,
        get_tools,
        execute_tool as claw_execute_tool,
        ToolExecution,
    )
    from phases.phase_claw.commands import (
        PORTED_COMMANDS,
        get_command,
        get_commands,
        execute_command as claw_execute_command,
        CommandExecution,
    )
    from phases.phase_claw.query_engine import QueryEnginePort, QueryEngineConfig, TurnResult
    from phases.phase_claw.models import PortingModule, PermissionDenial, UsageSummary
    from phases.phase_claw.permissions import ToolPermissionContext
    from phases.phase_claw.tool_pool import ToolPool, assemble_tool_pool
    from phases.phase_claw.runtime import PortRuntime, RuntimeSession
except ImportError:
    from .tools import (
        PORTED_TOOLS,
        get_tool,
        get_tools,
        execute_tool as claw_execute_tool,
        ToolExecution,
    )
    from .commands import (
        PORTED_COMMANDS,
        get_command,
        get_commands,
        execute_command as claw_execute_command,
        CommandExecution,
    )
    from .query_engine import QueryEnginePort, QueryEngineConfig, TurnResult
    from .models import PortingModule, PermissionDenial, UsageSummary
    from .permissions import ToolPermissionContext
    from .tool_pool import ToolPool, assemble_tool_pool
    from .runtime import PortRuntime, RuntimeSession


# ─────────────────────────────────────────────
# Tool Definition Conversion
# ─────────────────────────────────────────────

@dataclass
class FluxToolDefinition:
    """FLUX-compatible tool definition converted from Claw."""
    name: str
    description: str
    category: str
    parameters: Dict[str, Any]
    source: str  # 'claw' or 'native'
    claw_module: Optional[PortingModule] = None
    
    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON schema for VLM function calling."""
        return {
            'type': 'function',
            'function': {
                'name': self.name,
                'description': self.description,
                'parameters': self.parameters,
            }
        }


def claw_to_flux_tool(claw_module: PortingModule) -> FluxToolDefinition:
    """Convert a Claw tool module to FLUX tool definition."""
    # Infer category from source hint
    source_hint = claw_module.source_hint.lower()
    if 'bash' in source_hint:
        category = 'execution'
    elif 'file' in source_hint:
        category = 'filesystem'
    elif 'agent' in source_hint:
        category = 'agents'
    elif 'mcp' in source_hint:
        category = 'mcp'
    elif 'git' in source_hint or 'github' in source_hint:
        category = 'vcs'
    else:
        category = 'general'
    
    # Build parameter schema based on tool name
    parameters = _infer_parameters(claw_module.name)
    
    return FluxToolDefinition(
        name=claw_module.name,
        description=claw_module.responsibility,
        category=category,
        parameters=parameters,
        source='claw',
        claw_module=claw_module,
    )


def flux_to_claw_tool(flux_tool: Dict[str, Any]) -> PortingModule:
    """Convert a FLUX tool definition to Claw module format."""
    return PortingModule(
        name=flux_tool.get('name', 'unknown'),
        responsibility=flux_tool.get('description', ''),
        source_hint=f"flux/{flux_tool.get('category', 'native')}",
        status='active',
    )


def _infer_parameters(tool_name: str) -> Dict[str, Any]:
    """Infer parameter schema from tool name."""
    # Common parameter patterns
    if 'Bash' in tool_name:
        return {
            'type': 'object',
            'properties': {
                'command': {'type': 'string', 'description': 'Shell command to execute'},
                'timeout': {'type': 'integer', 'description': 'Timeout in seconds', 'default': 120},
            },
            'required': ['command'],
        }
    elif 'FileRead' in tool_name or 'Read' in tool_name:
        return {
            'type': 'object',
            'properties': {
                'path': {'type': 'string', 'description': 'File path to read'},
                'start_line': {'type': 'integer', 'description': 'Start line (1-indexed)'},
                'end_line': {'type': 'integer', 'description': 'End line (inclusive)'},
            },
            'required': ['path'],
        }
    elif 'FileEdit' in tool_name or 'Edit' in tool_name:
        return {
            'type': 'object',
            'properties': {
                'path': {'type': 'string', 'description': 'File path to edit'},
                'old_text': {'type': 'string', 'description': 'Text to replace'},
                'new_text': {'type': 'string', 'description': 'Replacement text'},
            },
            'required': ['path', 'old_text', 'new_text'],
        }
    elif 'FileWrite' in tool_name or 'Create' in tool_name:
        return {
            'type': 'object',
            'properties': {
                'path': {'type': 'string', 'description': 'File path to create'},
                'content': {'type': 'string', 'description': 'File content'},
            },
            'required': ['path', 'content'],
        }
    elif 'Agent' in tool_name:
        return {
            'type': 'object',
            'properties': {
                'prompt': {'type': 'string', 'description': 'Task for the agent'},
                'agent_type': {'type': 'string', 'description': 'Agent type to use'},
            },
            'required': ['prompt'],
        }
    elif 'Search' in tool_name or 'Grep' in tool_name:
        return {
            'type': 'object',
            'properties': {
                'query': {'type': 'string', 'description': 'Search query or pattern'},
                'path': {'type': 'string', 'description': 'Path to search in'},
                'regex': {'type': 'boolean', 'description': 'Use regex matching'},
            },
            'required': ['query'],
        }
    else:
        # Generic parameters
        return {
            'type': 'object',
            'properties': {
                'input': {'type': 'string', 'description': 'Tool input'},
            },
            'required': [],
        }


# ─────────────────────────────────────────────
# Claw Harness Class
# ─────────────────────────────────────────────

@dataclass
class ClawHarness:
    """
    FLUX-integrated Claw harness for agent tool execution.
    
    This class wraps the claw-code harness and provides:
    - Tool discovery and registration
    - Command routing
    - Session management
    - Integration with FLUX orchestrator
    """
    
    permission_context: Optional[ToolPermissionContext] = None
    query_engine: Optional[QueryEnginePort] = None
    runtime: Optional[PortRuntime] = None
    _flux_tools_cache: Dict[str, FluxToolDefinition] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize harness components."""
        self.runtime = PortRuntime()
        self.query_engine = QueryEnginePort.from_workspace()
        self._build_tool_cache()
    
    def _build_tool_cache(self):
        """Build cache of FLUX tool definitions from Claw modules."""
        for module in PORTED_TOOLS:
            flux_tool = claw_to_flux_tool(module)
            self._flux_tools_cache[module.name] = flux_tool
    
    # ─────────────────────────────────────────
    # Tool Operations
    # ─────────────────────────────────────────
    
    def list_tools(self, category: Optional[str] = None, limit: int = 50) -> List[FluxToolDefinition]:
        """List available tools, optionally filtered by category."""
        tools = list(self._flux_tools_cache.values())
        if category:
            tools = [t for t in tools if t.category == category]
        return tools[:limit]
    
    def get_tool(self, name: str) -> Optional[FluxToolDefinition]:
        """Get a specific tool by name."""
        return self._flux_tools_cache.get(name)
    
    def get_tool_categories(self) -> List[str]:
        """Get list of unique tool categories."""
        return list(set(t.category for t in self._flux_tools_cache.values()))
    
    def search_tools(self, query: str, limit: int = 20) -> List[FluxToolDefinition]:
        """Search tools by name or description."""
        query_lower = query.lower()
        matches = []
        for tool in self._flux_tools_cache.values():
            if query_lower in tool.name.lower() or query_lower in tool.description.lower():
                matches.append(tool)
        return matches[:limit]
    
    def execute_tool(self, name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a Claw tool.
        
        Args:
            name: Tool name
            **kwargs: Tool parameters
            
        Returns:
            Execution result dict
        """
        tool = self.get_tool(name)
        if not tool:
            return {
                'success': False,
                'error': f'Unknown tool: {name}',
                'output': None,
            }
        
        # Check permissions
        if self.permission_context and self.permission_context.blocks(name):
            return {
                'success': False,
                'error': f'Permission denied for tool: {name}',
                'output': None,
            }
        
        # Execute via claw runtime
        payload = json.dumps(kwargs)
        result = claw_execute_tool(name, payload)
        
        return {
            'success': result.handled,
            'tool': result.name,
            'source': result.source_hint,
            'output': result.message,
            'payload': result.payload,
        }
    
    # ─────────────────────────────────────────
    # Command Operations
    # ─────────────────────────────────────────
    
    def list_commands(self, limit: int = 50) -> List[PortingModule]:
        """List available commands."""
        return list(PORTED_COMMANDS)[:limit]
    
    def get_command(self, name: str) -> Optional[PortingModule]:
        """Get a specific command by name."""
        return get_command(name)
    
    def execute_command(self, name: str, prompt: str = '') -> Dict[str, Any]:
        """Execute a Claw command."""
        result = claw_execute_command(name, prompt)
        return {
            'success': result.handled,
            'command': result.name,
            'source': result.source_hint,
            'output': result.message,
        }
    
    # ─────────────────────────────────────────
    # Session Management
    # ─────────────────────────────────────────
    
    def submit_prompt(self, prompt: str) -> TurnResult:
        """Submit a prompt to the query engine."""
        if not self.query_engine:
            self.query_engine = QueryEnginePort.from_workspace()
        
        # Route the prompt
        matches = self.runtime.route_prompt(prompt) if self.runtime else []
        matched_commands = tuple(m.name for m in matches if m.kind == 'command')
        matched_tools = tuple(m.name for m in matches if m.kind == 'tool')
        
        return self.query_engine.submit_message(
            prompt=prompt,
            matched_commands=matched_commands,
            matched_tools=matched_tools,
        )
    
    def persist_session(self) -> str:
        """Persist the current session and return path."""
        if self.query_engine:
            return self.query_engine.persist_session()
        return ''
    
    # ─────────────────────────────────────────
    # FLUX Orchestrator Integration
    # ─────────────────────────────────────────
    
    def get_tools_for_orchestrator(self) -> List[Dict[str, Any]]:
        """Get tool definitions in FLUX orchestrator format."""
        return [tool.to_json_schema() for tool in self._flux_tools_cache.values()]
    
    def register_with_orchestrator(self, orchestrator: Any) -> None:
        """
        Register Claw tools with FLUX orchestrator.
        
        Args:
            orchestrator: FLUXOrchestrator instance
        """
        # Get FLUX tool format
        tool_defs = self.get_tools_for_orchestrator()
        
        # Register each tool
        for tool_def in tool_defs:
            tool_name = tool_def['function']['name']
            
            # Create executor function
            def make_executor(name):
                def executor(**kwargs):
                    return self.execute_tool(name, **kwargs)
                return executor
            
            # Register with orchestrator
            if hasattr(orchestrator, 'register_tool'):
                orchestrator.register_tool(
                    name=tool_name,
                    definition=tool_def,
                    executor=make_executor(tool_name),
                    source='claw',
                )
    
    # ─────────────────────────────────────────
    # Statistics
    # ─────────────────────────────────────────
    
    def get_stats(self) -> Dict[str, Any]:
        """Get harness statistics."""
        categories = self.get_tool_categories()
        return {
            'total_tools': len(self._flux_tools_cache),
            'total_commands': len(PORTED_COMMANDS),
            'tool_categories': categories,
            'tools_by_category': {
                cat: len([t for t in self._flux_tools_cache.values() if t.category == cat])
                for cat in categories
            },
            'session_active': self.query_engine is not None,
        }


# ─────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────

_default_harness: Optional[ClawHarness] = None


def get_claw_harness() -> ClawHarness:
    """Get or create the default Claw harness instance."""
    global _default_harness
    if _default_harness is None:
        _default_harness = ClawHarness()
    return _default_harness


def register_claw_tools_with_orchestrator(orchestrator: Any) -> None:
    """Register all Claw tools with a FLUX orchestrator."""
    harness = get_claw_harness()
    harness.register_with_orchestrator(orchestrator)


def execute_claw_tool(name: str, **kwargs) -> Dict[str, Any]:
    """Execute a Claw tool using the default harness."""
    harness = get_claw_harness()
    return harness.execute_tool(name, **kwargs)


def search_claw_tools(query: str, limit: int = 20) -> List[FluxToolDefinition]:
    """Search Claw tools by name or description."""
    harness = get_claw_harness()
    return harness.search_tools(query, limit)
