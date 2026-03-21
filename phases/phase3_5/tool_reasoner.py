"""
ToolReasoner: Deciding WHEN and HOW to use tools.
"""
from enum import Enum

class ToolDecision(Enum):
    NO_TOOL_NEEDED   = 'no_tool'
    TOOL_REQUIRED    = 'required'

class ToolReasoner:
    def __init__(self, fabric, registry, gr):
        self.fabric, self.registry, self.gr = fabric, registry, gr
    def should_use_tool(self, query):
        return ToolDecision.NO_TOOL_NEEDED
