"""
visual_reasoner.py — Phase 12: Visual Reasoning for Game Puzzles

Bridge between spatial memory visualization and LLM reasoning.
Supports both ASCII text and direct image input (Qwen-Omni's native vision).

The direct image approach is BETTER because the model can see:
- Actual colors (not just symbols)
- Spatial relationships naturally
- Fine details that ASCII misses
"""

import torch
from torch import Tensor
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
import sys

# Local imports
from grid_renderer import (
    GridRenderer, 
    ASCIIRenderer, 
    normalize_grid, 
    PIL_AVAILABLE,
)
from action_parser import (
    ActionParser, 
    ParseResult, 
    ACTION_MAP, 
    format_available_actions,
)
from game_memory import GameMemory

# Try PIL import
try:
    from PIL import Image
except ImportError:
    Image = None


# ─────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI agent playing a puzzle game. Your task is to analyze the game state and decide the best action.

Key principles:
1. Observe patterns in the grid (colors, shapes, arrangements)
2. Remember what actions caused what effects
3. Plan moves toward apparent goals
4. Explore unknown areas to discover game mechanics

Always end your response with: ACTION: [your choice]
Available actions are provided in each query."""

VISION_PROMPT_TEMPLATE = """You are playing {game_id}, a puzzle game.

Look at this game image:
- The white dot marks my current position at ({row}, {col})
- Cyan/teal overlays show areas of high curiosity (interesting spots to explore)
- Grid colors follow a specific meaning in each game

Available actions: {actions}

{history}

Based on what you SEE in the image:
1. What structures or patterns do you notice?
2. What seems to be the goal?
3. Which direction should I move?

Think step by step, then end with: ACTION: [your choice]"""

ASCII_PROMPT_TEMPLATE = """You are playing {game_id}, a puzzle game from ARC-AGI-3.

CURRENT STATE:
- Position: row {row}, column {col}
- Available actions: {actions}

WHAT I SEE (ASCII map):
@ = my position
! = high curiosity (explore here!)
+ = potential target/goal
● = explored area
· = lightly visited
# = wall/obstacle
(space) = unexplored

```
{ascii_map}
```

{history}

TASK: Analyze this puzzle visually. What patterns do you see? What should I do next?

Think step by step, then end with: ACTION: [your chosen action]"""


# ─────────────────────────────────────────────
# Visual Reasoner
# ─────────────────────────────────────────────

class VisualReasoner:
    """
    Bridge between spatial memory visualization and LLM reasoning.
    
    With Qwen-Omni, we have TWO options:
    1. ASCII text representation (works with any LLM)
    2. Direct image input (Qwen-Omni's native vision!)
    
    The direct image approach is BETTER because the model can see:
    - Actual colors (not symbols)
    - Spatial relationships naturally
    - Ice field as visual overlay
    """
    
    def __init__(
        self,
        llm_bridge,
        spatial_memory=None,
        prefer_vision: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize visual reasoner.
        
        Args:
            llm_bridge: QwenOmniBridge or similar LLM interface
            spatial_memory: SpatialMemory instance (optional)
            prefer_vision: Prefer image input over ASCII when possible
            verbose: Print reasoning details
        """
        self.llm = llm_bridge
        self.spatial = spatial_memory
        self.prefer_vision = prefer_vision
        self.verbose = verbose
        
        # Initialize renderers
        self.grid_renderer = GridRenderer() if PIL_AVAILABLE else None
        self.ascii_renderer = ASCIIRenderer()
        
        # Action parser
        self.action_parser = ActionParser(verbose=verbose)
    
    def set_spatial_memory(self, spatial_memory):
        """Set spatial memory instance."""
        self.spatial = spatial_memory
    
    # ─────────────────────────────────────────────
    # Main Reasoning API
    # ─────────────────────────────────────────────
    
    def reason_about_game(
        self,
        game_id: str,
        grid: Union[List[List[int]], Tensor],
        position: Tuple[int, int],
        available_actions: List[int],
        history: Optional[List[Dict]] = None,
        use_vision: Optional[bool] = None,
    ) -> Tuple[int, str]:
        """
        Use LLM to reason about the game and decide action.
        
        Automatically chooses between vision and ASCII based on:
        - prefer_vision setting
        - LLM capabilities
        - PIL availability
        
        Args:
            game_id: Game identifier (e.g., "ls20")
            grid: Current game frame [H, W] colors
            position: Agent position (row, col)
            available_actions: List of available action IDs
            history: Recent action history
            use_vision: Force vision mode (None = auto)
            
        Returns:
            (action_id, reasoning_text)
        """
        # Normalize grid
        grid = normalize_grid(grid)
        
        # Decide mode
        if use_vision is None:
            use_vision = (
                self.prefer_vision and 
                self.grid_renderer is not None and
                getattr(self.llm, 'has_vision', False)
            )
        
        if use_vision:
            return self.reason_with_vision(
                game_id, grid, position, available_actions, history
            )
        else:
            return self.reason_with_ascii(
                game_id, grid, position, available_actions, history
            )
    
    def reason_with_vision(
        self,
        game_id: str,
        grid,
        position: Tuple[int, int],
        available_actions: List[int],
        history: Optional[List[Dict]] = None,
    ) -> Tuple[int, str]:
        """
        Use Qwen-Omni's NATIVE VISION to reason about the game.
        
        Instead of converting to ASCII, we pass:
        1. The actual grid rendered as an image
        2. Ice field overlay showing curiosity
        3. Agent position marked
        """
        # Get spatial memory fields if available
        ice_field = None
        exploration_mass = None
        if self.spatial is not None:
            ice_field = self.spatial.curiosity_field
            exploration_mass = self.spatial.exploration_mass
        
        # Render grid as image
        game_image = self.grid_renderer.render(
            grid=grid,
            position=position,
            ice_field=ice_field,
            exploration_mass=exploration_mass,
        )
        
        # Format history
        history_text = self._format_history(history)
        
        # Build prompt
        prompt = VISION_PROMPT_TEMPLATE.format(
            game_id=game_id,
            row=position[0],
            col=position[1],
            actions=format_available_actions(available_actions),
            history=history_text,
        )
        
        if self.verbose:
            print(f"  [VisualReasoner] Vision mode @ position {position}")
        
        # Call LLM with image
        response = self.llm.chat_with_image(
            image=game_image,
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
        )
        
        # Parse action
        result = self.action_parser.parse(response, available_actions)
        
        if self.verbose:
            print(f"  [VisualReasoner] → {result}")
        
        return result.action_id, response
    
    def reason_with_ascii(
        self,
        game_id: str,
        grid,
        position: Tuple[int, int],
        available_actions: List[int],
        history: Optional[List[Dict]] = None,
    ) -> Tuple[int, str]:
        """
        Use ASCII text representation for LLM reasoning.
        
        Works with any text LLM, no vision required.
        """
        # Get spatial memory fields if available
        ice_field = None
        exploration_mass = None
        if self.spatial is not None:
            ice_field = self.spatial.curiosity_field
            exploration_mass = self.spatial.exploration_mass
        
        # Render ASCII map
        ascii_map = self.ascii_renderer.render(
            grid=grid,
            position=position,
            ice_field=ice_field,
            exploration_mass=exploration_mass,
        )
        
        # Format history
        history_text = self._format_history(history)
        
        # Build prompt
        prompt = ASCII_PROMPT_TEMPLATE.format(
            game_id=game_id,
            row=position[0],
            col=position[1],
            actions=format_available_actions(available_actions),
            ascii_map=ascii_map,
            history=history_text,
        )
        
        if self.verbose:
            print(f"  [VisualReasoner] ASCII mode @ position {position}")
        
        # Call LLM
        response = self.llm.chat(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
        )
        
        # Parse action
        result = self.action_parser.parse(response, available_actions)
        
        if self.verbose:
            print(f"  [VisualReasoner] → {result}")
        
        return result.action_id, response
    
    # ─────────────────────────────────────────────
    # Analysis Methods
    # ─────────────────────────────────────────────
    
    def describe_grid(
        self,
        grid,
        position: Optional[Tuple[int, int]] = None,
        use_vision: bool = True,
    ) -> str:
        """
        Ask LLM to describe what it sees in the grid.
        
        Useful for debugging and understanding LLM perception.
        """
        grid = normalize_grid(grid)
        
        prompt = """Describe what you see in this game grid:
1. What colors/patterns are present?
2. Are there any obvious structures (walls, paths, objects)?
3. What might be interactive elements?
4. What do you think the goal might be?

Be specific about positions and shapes."""
        
        if use_vision and self.grid_renderer and getattr(self.llm, 'has_vision', False):
            image = self.grid_renderer.render(grid=grid, position=position)
            return self.llm.chat_with_image(image=image, prompt=prompt)
        else:
            ascii_map = self.ascii_renderer.render(grid=grid, position=position)
            full_prompt = f"ASCII Grid:\n```\n{ascii_map}\n```\n\n{prompt}"
            return self.llm.chat(prompt=full_prompt)
    
    def analyze_change(
        self,
        grid_before,
        grid_after,
        action_taken: int,
        position: Tuple[int, int],
    ) -> str:
        """
        Ask LLM to explain what changed after an action.
        
        Helps build causal understanding.
        """
        grid_before = normalize_grid(grid_before)
        grid_after = normalize_grid(grid_after)
        
        action_name = ACTION_MAP.get(action_taken, f'ACTION{action_taken}')
        
        prompt = f"""I took action {action_name} at position {position}.

What changed? Please identify:
1. Which cells changed color/state
2. The pattern of change (did something rotate? shift? toggle?)
3. Any rules you can infer from this change

Be specific about positions and colors."""
        
        if self.grid_renderer and getattr(self.llm, 'has_vision', False):
            # Render comparison image
            image = self.grid_renderer.render_comparison(
                grid_before=grid_before,
                grid_after=grid_after,
                position_before=position,
                position_after=position,
            )
            return self.llm.chat_with_image(image=image, prompt=prompt)
        else:
            before_ascii = self.ascii_renderer.render(grid_before, position)
            after_ascii = self.ascii_renderer.render(grid_after, position)
            full_prompt = f"BEFORE:\n```\n{before_ascii}\n```\n\nAFTER:\n```\n{after_ascii}\n```\n\n{prompt}"
            return self.llm.chat(prompt=full_prompt)
    
    # ─────────────────────────────────────────────
    # Helper Methods
    # ─────────────────────────────────────────────
    
    def _format_history(self, history: Optional[List[Dict]]) -> str:
        """Format action history for prompts."""
        if not history:
            return "HISTORY: Just started, no previous actions."
        
        lines = ["RECENT HISTORY:"]
        for item in history[-5:]:
            action = item.get('action', '?')
            pos = item.get('position', (0, 0))
            reasoning = item.get('reasoning', '')[:50]
            
            action_name = ACTION_MAP.get(action, f'ACTION{action}')
            lines.append(f"  - {action_name} at {pos}: {reasoning}...")
        
        return '\n'.join(lines)
    
    def _get_background(self, grid) -> int:
        """Detect background color (most common)."""
        import numpy as np
        grid_np = normalize_grid(grid)
        unique, counts = np.unique(grid_np, return_counts=True)
        return unique[counts.argmax()]


# ─────────────────────────────────────────────
# Spatial Field to Text Converter
# ─────────────────────────────────────────────

class SpatialToText:
    """
    Convert spatial memory fields to text formats for LLM consumption.
    
    Multiple output modes:
    1. ASCII art (simple visual)
    2. Coordinate list (structured)
    3. Natural language description (for complex reasoning)
    """
    
    @staticmethod
    def to_coordinates(
        ice: Union[Tensor, 'numpy.ndarray'],
        threshold: float = 5.0,
    ) -> str:
        """
        Convert ice field to coordinate list.
        
        Returns something like:
        "High curiosity areas (ice > 5):
         - Row 10-15, Col 20-30: Large interesting region
         - Row 25, Col 52-54: Potential target cluster"
        """
        import numpy as np
        
        if isinstance(ice, Tensor):
            ice = ice.cpu().numpy()
        
        # Find high-ice positions
        high_ice = np.argwhere(ice > threshold)
        
        if len(high_ice) == 0:
            return "No high-curiosity areas detected."
        
        # Simple clustering by proximity
        regions = []
        current_region = []
        
        for i, point in enumerate(high_ice):
            r, c = point[0], point[1]
            
            if not current_region:
                current_region = [(r, c)]
            elif abs(r - current_region[-1][0]) <= 2 and abs(c - current_region[-1][1]) <= 2:
                current_region.append((r, c))
            else:
                if len(current_region) >= 3:
                    regions.append(current_region)
                current_region = [(r, c)]
        
        if current_region and len(current_region) >= 3:
            regions.append(current_region)
        
        # Format as text
        lines = ["Interesting regions detected:"]
        for i, region in enumerate(regions[:5]):  # Top 5
            min_r = min(p[0] for p in region)
            max_r = max(p[0] for p in region)
            min_c = min(p[1] for p in region)
            max_c = max(p[1] for p in region)
            lines.append(
                f"  Region {i+1}: rows {min_r}-{max_r}, "
                f"cols {min_c}-{max_c} ({len(region)} cells)"
            )
        
        return '\n'.join(lines)
    
    @staticmethod
    def to_description(
        grid,
        mass: Optional[Union[Tensor, 'numpy.ndarray']] = None,
        ice: Optional[Union[Tensor, 'numpy.ndarray']] = None,
        position: Optional[Tuple[int, int]] = None,
    ) -> str:
        """
        Generate natural language description for complex reasoning.
        """
        import numpy as np
        
        grid = normalize_grid(grid)
        grid_h, grid_w = grid.shape
        
        # Analyze grid composition
        unique, counts = np.unique(grid, return_counts=True)
        color_counts = dict(zip(unique.tolist(), counts.tolist()))
        
        bg_color = max(color_counts, key=color_counts.get)
        non_bg = {k: v for k, v in color_counts.items() if k != bg_color}
        
        # Analyze exploration
        explored_pct = 0.0
        ice_total = 0.0
        
        if mass is not None:
            if isinstance(mass, Tensor):
                mass = mass.cpu().numpy()
            explored_pct = (mass[:grid_h, :grid_w] > 0).sum() / (grid_h * grid_w) * 100
        
        if ice is not None:
            if isinstance(ice, Tensor):
                ice = ice.cpu().numpy()
            ice_total = ice[:grid_h, :grid_w].sum()
        
        # Build description
        pos_str = f"row {position[0]}, col {position[1]}" if position else "unknown"
        
        color_summary = ', '.join(
            f'{k}:{v}' for k, v in sorted(non_bg.items(), key=lambda x: -x[1])[:5]
        )
        
        desc = f"""Grid Analysis:
- Size: {grid_h} × {grid_w}
- Background color: {bg_color} ({color_counts[bg_color]} cells)
- Notable colors: {color_summary}
- My position: {pos_str}

Exploration Status:
- {explored_pct:.1f}% of grid explored
- Total curiosity (ice): {ice_total:.1f}
- {'High curiosity in unexplored areas' if ice_total > 100 else 'Mostly explored interesting areas'}"""
        
        return desc


# ─────────────────────────────────────────────
# Module Exports
# ─────────────────────────────────────────────

__all__ = [
    'VisualReasoner',
    'SpatialToText',
    'SYSTEM_PROMPT',
    'VISION_PROMPT_TEMPLATE',
    'ASCII_PROMPT_TEMPLATE',
]
