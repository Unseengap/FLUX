from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:
    from phases.phase_claw.models import PortingBacklog, PortingModule
    from phases.phase_claw.permissions import ToolPermissionContext
except ImportError:
    from .models import PortingBacklog, PortingModule
    from .permissions import ToolPermissionContext

SNAPSHOT_PATH = Path(__file__).resolve().parent / 'reference_data' / 'tools_snapshot.json'


@dataclass(frozen=True)
class ToolExecution:
    name: str
    source_hint: str
    payload: str
    handled: bool
    message: str


@lru_cache(maxsize=1)
def load_tool_snapshot() -> tuple[PortingModule, ...]:
    if not SNAPSHOT_PATH.exists():
        return ()  # Graceful degradation when embedded
    raw_entries = json.loads(SNAPSHOT_PATH.read_text())
    return tuple(
        PortingModule(
            name=entry['name'],
            responsibility=entry['responsibility'],
            source_hint=entry['source_hint'],
            status='mirrored',
        )
        for entry in raw_entries
    )


# Lazy load to avoid crash when embedded
def get_ported_tools() -> tuple[PortingModule, ...]:
    return load_tool_snapshot()


# Empty tuple at module level for backwards compat, use get_ported_tools() instead
PORTED_TOOLS = ()


def build_tool_backlog() -> PortingBacklog:
    return PortingBacklog(title='Tool surface', modules=list(get_ported_tools()))


def tool_names() -> list[str]:
    return [module.name for module in get_ported_tools()]


def get_tool(name: str) -> PortingModule | None:
    needle = name.lower()
    for module in get_ported_tools():
        if module.name.lower() == needle:
            return module
    return None


def filter_tools_by_permission_context(tools: tuple[PortingModule, ...], permission_context: ToolPermissionContext | None = None) -> tuple[PortingModule, ...]:
    if permission_context is None:
        return tools
    return tuple(module for module in tools if not permission_context.blocks(module.name))


def get_tools(
    simple_mode: bool = False,
    include_mcp: bool = True,
    permission_context: ToolPermissionContext | None = None,
) -> tuple[PortingModule, ...]:
    tools = list(get_ported_tools())
    if simple_mode:
        tools = [module for module in tools if module.name in {'BashTool', 'FileReadTool', 'FileEditTool'}]
    if not include_mcp:
        tools = [module for module in tools if 'mcp' not in module.name.lower() and 'mcp' not in module.source_hint.lower()]
    return filter_tools_by_permission_context(tuple(tools), permission_context)


def find_tools(query: str, limit: int = 20) -> list[PortingModule]:
    needle = query.lower()
    matches = [module for module in get_ported_tools() if needle in module.name.lower() or needle in module.source_hint.lower()]
    return matches[:limit]


def execute_tool(name: str, payload: str = '') -> ToolExecution:
    module = get_tool(name)
    if module is None:
        return ToolExecution(name=name, source_hint='', payload=payload, handled=False, message=f'Unknown mirrored tool: {name}')
    action = f"Mirrored tool '{module.name}' from {module.source_hint} would handle payload {payload!r}."
    return ToolExecution(name=module.name, source_hint=module.source_hint, payload=payload, handled=True, message=action)


def render_tool_index(limit: int = 20, query: str | None = None) -> str:
    ported = get_ported_tools()
    modules = find_tools(query, limit) if query else list(ported[:limit])
    lines = [f'Tool entries: {len(ported)}', '']
    if query:
        lines.append(f'Filtered by: {query}')
        lines.append('')
    lines.extend(f'- {module.name} — {module.source_hint}' for module in modules)
    return '\n'.join(lines)
