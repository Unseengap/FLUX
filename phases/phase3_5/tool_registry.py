"""
ToolRegistry: Structural awareness of external capabilities.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

class ToolType(Enum):
    REST_API   = 'rest_api'
    SDK        = 'sdk'
    MCP        = 'mcp'
    SHELL      = 'shell'

class ToolTrust(Enum):
    VERIFIED     = 'verified'
    COMMUNITY    = 'community'

@dataclass
class ToolDefinition:
    tool_id:      str
    name:         str
    description:  str
    tool_type:    ToolType
    trust:        ToolTrust

class ToolRegistry:
    def __init__(self, cse, cwc, field, device='cpu'):
        self.cse, self.cwc, self.field, self.device = cse, cwc, field, device
        self.tools = {}
    def register_tool(self, tool, strength=0.9):
        self.tools[tool.tool_id] = tool
