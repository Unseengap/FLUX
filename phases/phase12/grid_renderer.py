"""
grid_renderer.py — Phase 12: Grid to Image Rendering

Renders ARC-AGI-3 grids as images for Qwen-Omni's vision input.
Includes overlays for spatial memory (ice field, exploration mass).

The direct image approach is BETTER than ASCII because the model can see:
- Actual colors (not symbols)
- Spatial relationships naturally
- Ice field as semi-transparent overlay
"""

import torch
from torch import Tensor
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path
import numpy as np

# Optional PIL import (for image rendering)
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None
    ImageDraw = None


# ─────────────────────────────────────────────
# ARC Color Palette (Standard 10 Colors)
# ─────────────────────────────────────────────

ARC_COLORS = [
    (0, 0, 0),        # 0: black
    (0, 116, 217),    # 1: blue
    (255, 65, 54),    # 2: red
    (46, 204, 64),    # 3: green
    (255, 220, 0),    # 4: yellow
    (128, 128, 128),  # 5: grey
    (240, 18, 190),   # 6: magenta
    (255, 133, 27),   # 7: orange
    (127, 219, 255),  # 8: cyan
    (135, 12, 37),    # 9: brown
]

# Extended colors for games with >10 colors
EXTENDED_COLORS = ARC_COLORS + [
    (255, 255, 255),  # 10: white
    (128, 0, 128),    # 11: purple
    (0, 128, 128),    # 12: teal
    (128, 128, 0),    # 13: olive
    (192, 192, 192),  # 14: silver
    (0, 0, 128),      # 15: navy
]


# ─────────────────────────────────────────────
# Grid Renderer
# ─────────────────────────────────────────────

class GridRenderer:
    """
    Renders ARC-AGI-3 grids as images for vision model input.
    
    Supports:
    - Basic grid rendering with ARC color palette
    - Ice field overlay (curiosity visualization)
    - Exploration mass overlay
    - Agent position marker
    - Grid lines for clarity
    """
    
    def __init__(
        self,
        cell_size: int = 10,
        show_grid_lines: bool = True,
        grid_line_color: Tuple[int, int, int] = (60, 60, 60),
        agent_color: Tuple[int, int, int] = (255, 255, 255),
        ice_color: Tuple[int, int, int, int] = (0, 255, 255, 150),
        mass_color: Tuple[int, int, int, int] = (100, 100, 255, 80),
    ):
        """
        Initialize the grid renderer.
        
        Args:
            cell_size: Pixels per grid cell
            show_grid_lines: Whether to draw grid lines
            grid_line_color: Color for grid lines (RGB)
            agent_color: Color for agent marker (RGB)
            ice_color: Color for ice overlay (RGBA)
            mass_color: Color for mass overlay (RGBA)
        """
        if not PIL_AVAILABLE:
            raise ImportError(
                "PIL/Pillow required for GridRenderer. "
                "Install with: pip install Pillow"
            )
        
        self.cell_size = cell_size
        self.show_grid_lines = show_grid_lines
        self.grid_line_color = grid_line_color
        self.agent_color = agent_color
        self.ice_color = ice_color
        self.mass_color = mass_color
    
    def render(
        self,
        grid: Union[List[List[int]], np.ndarray, Tensor],
        position: Optional[Tuple[int, int]] = None,
        ice_field: Optional[Union[np.ndarray, Tensor]] = None,
        exploration_mass: Optional[Union[np.ndarray, Tensor]] = None,
        highlight_cells: Optional[List[Tuple[int, int]]] = None,
        target_cells: Optional[List[Tuple[int, int]]] = None,
    ) -> Image.Image:
        """
        Render grid to PIL Image.
        
        Args:
            grid: [H, W] integer grid (colors 0-9+)
            position: Agent position (row, col) to mark
            ice_field: [H, W] curiosity ice values (optional)
            exploration_mass: [H, W] exploration mass values (optional)
            highlight_cells: List of (row, col) to highlight
            target_cells: List of (row, col) potential targets
            
        Returns:
            PIL Image ready for Qwen-Omni vision input
        """
        # Convert to numpy if needed
        if isinstance(grid, Tensor):
            grid = grid.cpu().numpy()
        if isinstance(grid, list):
            grid = np.array(grid)
        
        h, w = grid.shape
        img_h = h * self.cell_size
        img_w = w * self.cell_size
        
        # Create base image
        img = Image.new('RGBA', (img_w, img_h), (0, 0, 0, 255))
        draw = ImageDraw.Draw(img)
        
        # ─────────────────────────────────────────────
        # Draw base grid cells
        # ─────────────────────────────────────────────
        for r in range(h):
            for c in range(w):
                color_idx = int(grid[r, c])
                if color_idx < len(EXTENDED_COLORS):
                    color = EXTENDED_COLORS[color_idx]
                else:
                    # Fallback for unknown colors
                    color = ((color_idx * 37) % 256, 
                             (color_idx * 91) % 256, 
                             (color_idx * 157) % 256)
                
                x0, y0 = c * self.cell_size, r * self.cell_size
                x1, y1 = x0 + self.cell_size - 1, y0 + self.cell_size - 1
                draw.rectangle([x0, y0, x1, y1], fill=color + (255,))
        
        # ─────────────────────────────────────────────
        # Draw exploration mass overlay (blue tint)
        # ─────────────────────────────────────────────
        if exploration_mass is not None:
            mass = self._to_numpy(exploration_mass)
            if mass.shape[0] >= h and mass.shape[1] >= w:
                mass_overlay = Image.new('RGBA', (img_w, img_h), (0, 0, 0, 0))
                mass_draw = ImageDraw.Draw(mass_overlay)
                
                max_mass = mass.max() + 1e-6
                for r in range(h):
                    for c in range(w):
                        m = mass[r, c] if r < mass.shape[0] and c < mass.shape[1] else 0
                        if m > 0:
                            alpha = min(int((m / max_mass) * self.mass_color[3]), 200)
                            x0, y0 = c * self.cell_size, r * self.cell_size
                            x1, y1 = x0 + self.cell_size - 1, y0 + self.cell_size - 1
                            color = self.mass_color[:3] + (alpha,)
                            mass_draw.rectangle([x0, y0, x1, y1], fill=color)
                
                img = Image.alpha_composite(img, mass_overlay)
                draw = ImageDraw.Draw(img)  # Refresh draw object
        
        # ─────────────────────────────────────────────
        # Draw ice field overlay (cyan highlights)
        # ─────────────────────────────────────────────
        if ice_field is not None:
            ice = self._to_numpy(ice_field)
            if ice.shape[0] >= h and ice.shape[1] >= w:
                ice_overlay = Image.new('RGBA', (img_w, img_h), (0, 0, 0, 0))
                ice_draw = ImageDraw.Draw(ice_overlay)
                
                for r in range(h):
                    for c in range(w):
                        ice_val = ice[r, c] if r < ice.shape[0] and c < ice.shape[1] else 0
                        if ice_val > 1.0:  # Threshold for showing ice
                            alpha = min(int(ice_val * 15), self.ice_color[3])
                            x0, y0 = c * self.cell_size, r * self.cell_size
                            x1, y1 = x0 + self.cell_size - 1, y0 + self.cell_size - 1
                            color = self.ice_color[:3] + (alpha,)
                            ice_draw.rectangle([x0, y0, x1, y1], fill=color)
                
                img = Image.alpha_composite(img, ice_overlay)
                draw = ImageDraw.Draw(img)  # Refresh draw object
        
        # ─────────────────────────────────────────────
        # Draw target cells (green border)
        # ─────────────────────────────────────────────
        if target_cells:
            for r, c in target_cells:
                if 0 <= r < h and 0 <= c < w:
                    x0, y0 = c * self.cell_size, r * self.cell_size
                    x1, y1 = x0 + self.cell_size - 1, y0 + self.cell_size - 1
                    draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0, 255), width=2)
        
        # ─────────────────────────────────────────────
        # Draw highlight cells (yellow border)
        # ─────────────────────────────────────────────
        if highlight_cells:
            for r, c in highlight_cells:
                if 0 <= r < h and 0 <= c < w:
                    x0, y0 = c * self.cell_size, r * self.cell_size
                    x1, y1 = x0 + self.cell_size - 1, y0 + self.cell_size - 1
                    draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 0, 255), width=2)
        
        # ─────────────────────────────────────────────
        # Draw grid lines
        # ─────────────────────────────────────────────
        if self.show_grid_lines:
            grid_color = self.grid_line_color + (100,)
            for r in range(h + 1):
                y = r * self.cell_size
                draw.line([(0, y), (img_w, y)], fill=grid_color, width=1)
            for c in range(w + 1):
                x = c * self.cell_size
                draw.line([(x, 0), (x, img_h)], fill=grid_color, width=1)
        
        # ─────────────────────────────────────────────
        # Draw agent position marker
        # ─────────────────────────────────────────────
        if position is not None:
            r, c = position
            if 0 <= r < h and 0 <= c < w:
                cx = c * self.cell_size + self.cell_size // 2
                cy = r * self.cell_size + self.cell_size // 2
                radius = max(self.cell_size // 3, 3)
                draw.ellipse(
                    [cx - radius, cy - radius, cx + radius, cy + radius],
                    fill=self.agent_color + (255,),
                    outline=(0, 0, 0, 255),
                    width=1
                )
        
        # Convert to RGB for compatibility
        return img.convert('RGB')
    
    def _to_numpy(self, arr: Union[np.ndarray, Tensor]) -> np.ndarray:
        """Convert tensor to numpy array."""
        if isinstance(arr, Tensor):
            return arr.cpu().numpy()
        return arr
    
    def render_comparison(
        self,
        grid_before: Union[List[List[int]], np.ndarray],
        grid_after: Union[List[List[int]], np.ndarray],
        position_before: Optional[Tuple[int, int]] = None,
        position_after: Optional[Tuple[int, int]] = None,
        changes: Optional[List[Tuple[int, int]]] = None,
    ) -> Image.Image:
        """
        Render before/after comparison side by side.
        
        Args:
            grid_before: Grid before action
            grid_after: Grid after action
            position_before: Agent position before
            position_after: Agent position after
            changes: List of changed cells to highlight
            
        Returns:
            Combined image showing both states
        """
        # Render both grids
        img_before = self.render(
            grid_before, 
            position=position_before,
            highlight_cells=changes,
        )
        img_after = self.render(
            grid_after,
            position=position_after,
            highlight_cells=changes,
        )
        
        # Combine horizontally with small gap
        gap = 10
        combined_w = img_before.width + gap + img_after.width
        combined_h = max(img_before.height, img_after.height)
        
        combined = Image.new('RGB', (combined_w, combined_h), (30, 30, 30))
        combined.paste(img_before, (0, 0))
        combined.paste(img_after, (img_before.width + gap, 0))
        
        return combined
    
    def save(
        self,
        img: Image.Image,
        path: Union[str, Path],
        format: str = 'PNG',
    ):
        """Save image to file."""
        img.save(str(path), format=format)
    
    def to_bytes(self, img: Image.Image, format: str = 'PNG') -> bytes:
        """Convert image to bytes for API calls."""
        import io
        buffer = io.BytesIO()
        img.save(buffer, format=format)
        return buffer.getvalue()


# ─────────────────────────────────────────────
# ASCII Renderer (Fallback)
# ─────────────────────────────────────────────

class ASCIIRenderer:
    """
    ASCII text renderer for grids when PIL is not available
    or for text-only LLMs.
    """
    
    # Character mappings for grid visualization
    CHARS = {
        'position': '@',
        'ice_high': '!',
        'ice_medium': '*',
        'target': '+',
        'explored': '●',
        'light_explored': '·',
        'wall': '#',
        'empty': ' ',
        'unknown': '?',
    }
    
    # Color to character mapping
    COLOR_CHARS = '0123456789ABCDEF'
    
    def __init__(self, show_colors: bool = False):
        """
        Initialize ASCII renderer.
        
        Args:
            show_colors: If True, show color values. If False, use abstract symbols.
        """
        self.show_colors = show_colors
    
    def render(
        self,
        grid: Union[List[List[int]], np.ndarray, Tensor],
        position: Optional[Tuple[int, int]] = None,
        ice_field: Optional[Union[np.ndarray, Tensor]] = None,
        exploration_mass: Optional[Union[np.ndarray, Tensor]] = None,
        max_width: int = 60,
        max_height: int = 30,
    ) -> str:
        """
        Render grid to ASCII text.
        
        Args:
            grid: [H, W] integer grid
            position: Agent position (row, col)
            ice_field: [H, W] curiosity ice values
            exploration_mass: [H, W] exploration mass values
            max_width: Max output width in characters
            max_height: Max output height in lines
            
        Returns:
            ASCII string representation of grid
        """
        # Convert to numpy
        if isinstance(grid, Tensor):
            grid = grid.cpu().numpy()
        if isinstance(grid, list):
            grid = np.array(grid)
        
        h, w = grid.shape
        
        # Subsample if too large
        step_h = max(1, h // max_height)
        step_w = max(1, w // max_width)
        
        # Convert fields to numpy
        ice = None
        mass = None
        if ice_field is not None:
            ice = self._to_numpy(ice_field)
        if exploration_mass is not None:
            mass = self._to_numpy(exploration_mass)
        
        # Detect background color (most common)
        unique, counts = np.unique(grid, return_counts=True)
        bg_color = unique[counts.argmax()]
        
        lines = []
        for r in range(0, h, step_h):
            row_chars = []
            for c in range(0, w, step_w):
                char = self._get_char(
                    grid, r, c, position, ice, mass, bg_color
                )
                row_chars.append(char)
            lines.append(''.join(row_chars))
        
        return '\n'.join(lines)
    
    def _get_char(
        self,
        grid: np.ndarray,
        r: int,
        c: int,
        position: Optional[Tuple[int, int]],
        ice: Optional[np.ndarray],
        mass: Optional[np.ndarray],
        bg_color: int,
    ) -> str:
        """Get character for a cell."""
        # Position marker takes priority
        if position is not None and (r, c) == position:
            return self.CHARS['position']
        
        cell_val = grid[r, c]
        
        # Get ice and mass values
        ice_val = 0
        if ice is not None and r < ice.shape[0] and c < ice.shape[1]:
            ice_val = ice[r, c]
        
        mass_val = 0
        if mass is not None and r < mass.shape[0] and c < mass.shape[1]:
            mass_val = mass[r, c]
        
        # High curiosity (ice) takes priority
        if ice_val > 10:
            return self.CHARS['ice_high']
        elif ice_val > 5:
            return self.CHARS['ice_medium']
        
        # Non-background cells that haven't been explored
        if cell_val != bg_color and mass_val < 3:
            return self.CHARS['target']
        
        # Background/wall
        if cell_val == 0:
            return self.CHARS['wall']
        
        # Explored areas
        if mass_val > 5:
            return self.CHARS['explored']
        elif mass_val > 0:
            return self.CHARS['light_explored']
        
        # Show color value
        if self.show_colors:
            if cell_val < len(self.COLOR_CHARS):
                return self.COLOR_CHARS[cell_val]
            return '?'
        
        # Default: empty
        return self.CHARS['empty']
    
    def _to_numpy(self, arr: Union[np.ndarray, Tensor]) -> np.ndarray:
        """Convert tensor to numpy array."""
        if isinstance(arr, Tensor):
            return arr.cpu().numpy()
        return arr
    
    def render_legend(self) -> str:
        """Return legend for ASCII symbols."""
        return """Legend:
@ = Your position
! = High curiosity (explore here!)
* = Medium curiosity
+ = Potential target/goal
● = Explored area
· = Lightly visited
# = Wall/obstacle
(space) = Unexplored"""


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────

def create_renderer(
    prefer_image: bool = True,
    cell_size: int = 10,
) -> Union[GridRenderer, ASCIIRenderer]:
    """
    Create appropriate renderer based on availability.
    
    Args:
        prefer_image: Prefer image rendering if PIL available
        cell_size: Size of cells for image rendering
        
    Returns:
        GridRenderer if PIL available and preferred, else ASCIIRenderer
    """
    if prefer_image and PIL_AVAILABLE:
        return GridRenderer(cell_size=cell_size)
    return ASCIIRenderer()


def normalize_grid(
    frame: Union[List[List[int]], np.ndarray, Dict],
) -> np.ndarray:
    """
    Normalize different grid formats to numpy array.
    
    Args:
        frame: Grid in various formats (list, array, or dict with 'frame' key)
        
    Returns:
        [H, W] numpy array
    """
    if isinstance(frame, dict):
        frame = frame.get('frame', frame.get('grid', []))
    
    if isinstance(frame, Tensor):
        return frame.cpu().numpy()
    
    return np.array(frame)


# ─────────────────────────────────────────────
# Module Exports
# ─────────────────────────────────────────────

__all__ = [
    'GridRenderer',
    'ASCIIRenderer',
    'ARC_COLORS',
    'EXTENDED_COLORS',
    'create_renderer',
    'normalize_grid',
    'PIL_AVAILABLE',
]
