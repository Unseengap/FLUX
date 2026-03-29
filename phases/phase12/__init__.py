"""
Phase 12: FLUX Multi-Modal Agent
================================

The final assembly — connects Spatial Vision to LLM Reasoning.

Components:
- FLUXMultiAgent: The complete multi-modal reasoning agent
- VisualReasoner: Bridge between spatial memory and LLM
- GridRenderer: Renders ARC grids as images for vision input
- ActionParser: Parses LLM responses to game actions
- GameMemory: Per-game episodic memory for cross-session learning
- QwenOmniBridge: Interface for Qwen2.5-Omni unified model

Usage:
    from phases.phase12 import FLUXMultiAgent, create_agent
    
    # Create agent
    agent = create_agent(enable_vision=True)
    
    # Play a game
    agent.reset("ls20")
    action, reasoning = agent.decide_action(grid, position, [1, 2, 3, 4])
    agent.record_effect(old_grid, new_grid, action, position)
    agent.end_game(score=100)
    
    # Save
    agent.save_flx("checkpoint.flx")
"""

# Core agent
from .flux_multi_agent import (
    FLUXMultiAgent,
    MultiAgentConfig,
    create_agent,
)

# Visual reasoning
from .visual_reasoner import (
    VisualReasoner,
    SpatialToText,
)

# Grid rendering
from .grid_renderer import (
    GridRenderer,
    ASCIIRenderer,
    ARC_COLORS,
    normalize_grid,
    PIL_AVAILABLE,
)

# Action parsing
from .action_parser import (
    ActionParser,
    ParseResult,
    ACTION_MAP,
    format_available_actions,
)

# Game memory
from .game_memory import (
    GameMemory,
    GameMemoryManager,
    ActionMemory,
    EffectMemory,
    RuleMemory,
)

# LLM bridge
from .qwen_omni_bridge import (
    QwenOmniBridge,
    QwenOmniConfig,
    create_qwen_bridge,
)


__all__ = [
    # Main agent
    'FLUXMultiAgent',
    'MultiAgentConfig',
    'create_agent',
    
    # Reasoning
    'VisualReasoner',
    'SpatialToText',
    
    # Rendering
    'GridRenderer',
    'ASCIIRenderer',
    'ARC_COLORS',
    'normalize_grid',
    'PIL_AVAILABLE',
    
    # Parsing
    'ActionParser',
    'ParseResult',
    'ACTION_MAP',
    'format_available_actions',
    
    # Memory
    'GameMemory',
    'GameMemoryManager',
    'ActionMemory',
    'EffectMemory',
    'RuleMemory',
    
    # LLM
    'QwenOmniBridge',
    'QwenOmniConfig',
    'create_qwen_bridge',
]

__version__ = '4.0-multi-modal'
__phase__ = 12
