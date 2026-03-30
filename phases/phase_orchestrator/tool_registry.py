"""
Tool Registry — Defines all FLUX tools available to the VLM orchestrator.

Each tool maps to a FLUX component method, enabling the VLM to
call FLUX's physics-inspired modules as cognitive tools.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from enum import Enum


class ToolCategory(Enum):
    """Categories of FLUX tools."""
    PERCEPTION = "perception"
    KNOWLEDGE = "knowledge"
    REASONING = "reasoning"
    EXPLORATION = "exploration"
    GENERATION = "generation"
    CGN = "cgn"


@dataclass
class ToolDefinition:
    """Definition of a single FLUX tool."""
    name: str
    description: str
    category: ToolCategory
    component_path: str  # e.g., 'adapters.grid_to_wave'
    method_name: str     # e.g., 'encode'
    params: Dict[str, str]  # param_name → type description
    returns: str
    example: str = ""
    requires_wave_dim: bool = False  # If True, input must be 432D


# ─────────────────────────────────────────────
# Tool Definitions
# ─────────────────────────────────────────────

FLUX_TOOLS: Dict[str, ToolDefinition] = {
    
    # ═══ PERCEPTION TOOLS ═══
    
    "encode_text": ToolDefinition(
        name="encode_text",
        description="Encode text into 432-dimensional semantic waves using the Continuous Semantic Encoder",
        category=ToolCategory.PERCEPTION,
        component_path="cse",
        method_name="forward",
        params={
            "text": "str — Raw text to encode (UTF-8)"
        },
        returns="wave tensor [seq_len, 432]",
        example='<tool>encode_text</tool>\n<params>{"text": "The cat sat on the mat"}</params>',
    ),
    
    "encode_grid": ToolDefinition(
        name="encode_grid",
        description="Encode ARC-style grid (0-9 colors, up to 30x30) into wave representation",
        category=ToolCategory.PERCEPTION,
        component_path="adapters.grid_to_wave",
        method_name="encode",
        params={
            "grid": "List[List[int]] — 2D grid with values 0-9",
            "mode": "str — 'holistic' (single vector) or 'spatial' (per-cell)"
        },
        returns="wave tensor [432] (holistic) or [H*W, 432] (spatial)",
        example='<tool>encode_grid</tool>\n<params>{"grid": [[0,1,0],[1,2,1]], "mode": "holistic"}</params>',
    ),
    
    "encode_image": ToolDefinition(
        name="encode_image",
        description="Encode image into wave representation using VLM vision encoder",
        category=ToolCategory.PERCEPTION,
        component_path="vlm",
        method_name="encode_image",
        params={
            "image": "Tensor [C, H, W] — RGB image tensor"
        },
        returns="wave tensor [patches, 432]",
    ),
    
    # ═══ KNOWLEDGE TOOLS ═══
    
    "query_field": ToolDefinition(
        name="query_field",
        description="Find relevant knowledge patterns in the 96³ resonance field using gravitational relevance",
        category=ToolCategory.KNOWLEDGE,
        component_path="field",
        method_name="query",
        params={
            "wave": "Tensor [432] or [seq, 432] — Query wave",
            "top_k": "int — Number of results (default 5)"
        },
        returns="List of (wave, relevance_score, position) tuples",
        example='<tool>query_field</tool>\n<params>{"wave": "$LAST_WAVE", "top_k": 3}</params>',
        requires_wave_dim=True,
    ),
    
    "recall_memory": ToolDefinition(
        name="recall_memory",
        description="Search episodic memory for relevant facts and experiences",
        category=ToolCategory.KNOWLEDGE,
        component_path="memory.episodic",
        method_name="search",
        params={
            "query": "str — Natural language query",
            "limit": "int — Maximum results (default 5)"
        },
        returns="List of (content, timestamp, importance) tuples",
        example='<tool>recall_memory</tool>\n<params>{"query": "grid transformation rules", "limit": 5}</params>',
    ),
    
    "store_memory": ToolDefinition(
        name="store_memory",
        description="Store a new fact or experience in episodic memory",
        category=ToolCategory.KNOWLEDGE,
        component_path="memory.episodic",
        method_name="store",
        params={
            "content": "str — What to remember",
            "importance": "float — Priority 0.0-1.0",
            "tags": "List[str] — Optional categories"
        },
        returns="memory_id: int",
        example='<tool>store_memory</tool>\n<params>{"content": "Blue cells rotate clockwise", "importance": 0.9}</params>',
    ),
    
    # ═══ REASONING TOOLS ═══
    
    "predict_effect": ToolDefinition(
        name="predict_effect",
        description="Predict what will change if an action is taken at a position",
        category=ToolCategory.REASONING,
        component_path="causal_tracker",
        method_name="predict",
        params={
            "action": "int — Action ID (1-7: up/down/left/right/interact/click/undo)",
            "position": "Tuple[int, int] — (row, col) location",
            "grid": "List[List[int]] — Current grid state"
        },
        returns="List of predicted GridChange objects",
        example='<tool>predict_effect</tool>\n<params>{"action": 5, "position": [2, 3], "grid": [[0,1],[1,0]]}</params>',
    ),
    
    "get_applicable_rules": ToolDefinition(
        name="get_applicable_rules",
        description="Find learned rules that apply to a trigger situation",
        category=ToolCategory.REASONING,
        component_path="rule_inducer",
        method_name="match_rules",
        params={
            "trigger_color": "int — Color at trigger position (0-9)",
            "trigger_action": "int — Action being performed",
            "context": "Optional[Dict] — Additional context"
        },
        returns="List of matching Rule objects with confidence scores",
        example='<tool>get_applicable_rules</tool>\n<params>{"trigger_color": 2, "trigger_action": 5}</params>',
    ),
    
    "trace_causality": ToolDefinition(
        name="trace_causality",
        description="Find the causal chain that led to an observed effect",
        category=ToolCategory.REASONING,
        component_path="causal_tracker",
        method_name="trace_back",
        params={
            "effect_position": "Tuple[int, int] — Where the effect occurred",
            "effect_type": "str — What changed (e.g., 'color_change')"
        },
        returns="Causal chain: List[CausalLink]",
    ),
    
    # ═══ EXPLORATION TOOLS ═══
    
    "get_curiosity_map": ToolDefinition(
        name="get_curiosity_map",
        description="Get the spatial curiosity field showing where to explore next",
        category=ToolCategory.EXPLORATION,
        component_path="spatial_memory",
        method_name="get_curiosity",
        params={
            "grid_size": "Tuple[int, int] — (height, width) of current grid"
        },
        returns="Curiosity heatmap [H, W] — high values = unexplored",
        example='<tool>get_curiosity_map</tool>\n<params>{"grid_size": [10, 10]}</params>',
    ),
    
    "mark_explored": ToolDefinition(
        name="mark_explored",
        description="Mark a position as explored (reduces curiosity, increases exploration mass)",
        category=ToolCategory.EXPLORATION,
        component_path="spatial_memory",
        method_name="update",
        params={
            "position": "Tuple[int, int] — Position that was visited",
            "novelty": "float — How interesting was it (0-1)"
        },
        returns="Updated exploration_mass at position",
    ),
    
    "get_exploration_summary": ToolDefinition(
        name="get_exploration_summary",
        description="Get summary of what's been explored vs. unknown",
        category=ToolCategory.EXPLORATION,
        component_path="spatial_memory",
        method_name="get_summary",
        params={},
        returns="Dict with explored_ratio, hot_spots, cold_spots",
    ),
    
    # ═══ CGN TOOLS ═══
    
    "query_cgn": ToolDefinition(
        name="query_cgn",
        description="Find Causal Geometry Nodes relevant to a concept",
        category=ToolCategory.CGN,
        component_path="causal.cgn",
        method_name="query",
        params={
            "wave": "Tensor [432] — Query wave",
            "node_tier": "str — 'fast' (32), 'medium' (16), 'slow' (8), or 'all'"
        },
        returns="List of (node_id, activation, curvature) tuples",
        requires_wave_dim=True,
    ),
    
    "fire_cgn": ToolDefinition(
        name="fire_cgn",
        description="Manually trigger a CGN node to propagate activation through the causal graph",
        category=ToolCategory.CGN,
        component_path="causal.cgn",
        method_name="fire",
        params={
            "node_id": "int — Which node to fire",
            "strength": "float — Activation strength"
        },
        returns="Downstream activations",
    ),
    
    "add_causal_arrow": ToolDefinition(
        name="add_causal_arrow",
        description="Record a new causal relationship in the causal graph",
        category=ToolCategory.CGN,
        component_path="causal.graph",
        method_name="add_link",
        params={
            "source": "int — Source node ID",
            "target": "int — Target node ID",
            "reason": "str — Why this link exists",
            "weight": "float — Link strength"
        },
        returns="arrow_id: int",
    ),
    
    # ═══ GENERATION TOOLS ═══
    
    "decode_grid": ToolDefinition(
        name="decode_grid",
        description="Convert wave representation back to ARC grid",
        category=ToolCategory.GENERATION,
        component_path="adapters.wave_to_grid",
        method_name="decode",
        params={
            "wave": "Tensor [432] — Wave to decode",
            "grid_size": "Tuple[int, int] — Target size (H, W)"
        },
        returns="Grid as List[List[int]]",
        example='<tool>decode_grid</tool>\n<params>{"wave": "$LAST_WAVE", "grid_size": [3, 3]}</params>',
        requires_wave_dim=True,
    ),
    
    "generate_text": ToolDefinition(
        name="generate_text",
        description="Generate text response using VLM (called implicitly if no other output tool used)",
        category=ToolCategory.GENERATION,
        component_path="vlm",
        method_name="generate",
        params={
            "prompt": "str — What to generate",
            "context_waves": "Optional[List[Tensor]] — FLUX context to inject",
            "max_tokens": "int — Generation limit"
        },
        returns="Generated text string",
    ),
}


def get_tool(name: str) -> Optional[ToolDefinition]:
    """Get tool definition by name."""
    return FLUX_TOOLS.get(name)


def get_tools_by_category(category: ToolCategory) -> List[ToolDefinition]:
    """Get all tools in a category."""
    return [t for t in FLUX_TOOLS.values() if t.category == category]


def get_tool_names() -> List[str]:
    """Get all tool names."""
    return list(FLUX_TOOLS.keys())


def format_tool_for_prompt(tool: ToolDefinition) -> str:
    """Format a tool definition for inclusion in system prompt."""
    params_str = "\n".join(f"    - {k}: {v}" for k, v in tool.params.items())
    return f"""- **{tool.name}**: {tool.description}
  Parameters:
{params_str}
  Returns: {tool.returns}"""


def generate_tool_section(category: ToolCategory) -> str:
    """Generate prompt section for a tool category."""
    tools = get_tools_by_category(category)
    header = f"### {category.value.title()} Tools\n"
    body = "\n\n".join(format_tool_for_prompt(t) for t in tools)
    return header + body


def generate_full_tool_prompt() -> str:
    """Generate the complete tool documentation for system prompt."""
    sections = []
    for category in ToolCategory:
        tools = get_tools_by_category(category)
        if tools:
            sections.append(generate_tool_section(category))
    return "\n\n".join(sections)
