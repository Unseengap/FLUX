"""
Rendering — Grid Visualization with Overlays for FLUX Unified Agent

Renders game grids as images with:
- ARC color palette
- Ice (curiosity) overlay in cyan
- Position marker (white circle)
- Change markers (yellow border on changed cells)

Physics Analogy:
    The rendering shows the agent's "vision" of the field - with
    curiosity highlighted as frozen areas awaiting exploration.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import torch
from torch import Tensor

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available, image rendering disabled")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# ─────────────────────────────────────────────
# ARC Color Palette
# ─────────────────────────────────────────────

# Standard ARC-AGI color palette (0-9)
ARC_COLORS = {
    0: (0, 0, 0),        # Black (background)
    1: (0, 116, 217),    # Blue
    2: (255, 65, 54),    # Red
    3: (46, 204, 64),    # Green
    4: (255, 220, 0),    # Yellow
    5: (170, 170, 170),  # Gray
    6: (240, 18, 190),   # Magenta/Pink
    7: (255, 133, 27),   # Orange
    8: (127, 219, 255),  # Cyan/Light Blue
    9: (135, 12, 37),    # Maroon/Dark Red
}

# Extended colors for games using 10-15
EXTENDED_COLORS = {
    10: (255, 255, 255),  # White
    11: (128, 0, 128),    # Purple
    12: (0, 128, 128),    # Teal
    13: (128, 128, 0),    # Olive
    14: (255, 192, 203),  # Pink
    15: (165, 42, 42),    # Brown
}

def get_color(value: int) -> Tuple[int, int, int]:
    """Get RGB color for a cell value."""
    if value in ARC_COLORS:
        return ARC_COLORS[value]
    if value in EXTENDED_COLORS:
        return EXTENDED_COLORS[value]
    # Fallback for unknown colors
    return (128, 128, 128)


# ─────────────────────────────────────────────
# Grid Rendering
# ─────────────────────────────────────────────

def render_grid_to_image(
    grid: List[List[int]],
    cell_size: int = 20,
    position: Optional[Tuple[int, int]] = None,
    ice_field: Optional[Tensor] = None,
    changes: Optional[List[Dict]] = None,
    border_width: int = 1,
) -> "Image.Image":
    """
    Render a grid to a PIL Image with overlays.
    
    Args:
        grid: 2D list of cell values (ints)
        cell_size: Size of each cell in pixels
        position: Agent position (row, col) to mark with circle
        ice_field: Curiosity field tensor for cyan overlay
        changes: List of {"row": r, "col": c} dicts for change markers
        border_width: Width of cell borders
        
    Returns:
        PIL Image
    """
    if not PIL_AVAILABLE:
        raise RuntimeError("PIL not available for image rendering")
    
    if not grid or not grid[0]:
        # Empty grid - return small placeholder
        return Image.new('RGB', (100, 100), (50, 50, 50))
    
    grid_h = len(grid)
    grid_w = len(grid[0])
    
    # Create image
    img_w = grid_w * cell_size
    img_h = grid_h * cell_size
    img = Image.new('RGB', (img_w, img_h), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    
    # Draw cells
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            x0 = c * cell_size
            y0 = r * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            
            # Get cell color
            val = val if isinstance(val, int) else 0
            color = get_color(val)
            
            # Apply ice overlay if high curiosity
            if ice_field is not None:
                if 0 <= r < ice_field.shape[0] and 0 <= c < ice_field.shape[1]:
                    ice_val = ice_field[r, c].item()
                    if ice_val > 1.0:
                        # Blend with cyan based on ice intensity
                        alpha = min(0.7, ice_val / 20.0)
                        cyan = (0, 255, 255)
                        color = tuple(int(c * (1 - alpha) + cyan[i] * alpha) for i, c in enumerate(color))
            
            # Draw cell
            draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=color)
            
            # Draw border
            if border_width > 0:
                draw.rectangle(
                    [x0, y0, x1 - 1, y1 - 1],
                    outline=(50, 50, 50),
                    width=border_width,
                )
    
    # Highlight changed cells with yellow border
    if changes:
        for change in changes:
            r = change.get("row", change.get("r", 0))
            c = change.get("col", change.get("c", 0))
            x0 = c * cell_size
            y0 = r * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(255, 255, 0), width=2)
    
    # Draw position marker (white circle)
    if position is not None:
        r, c = position
        cx = c * cell_size + cell_size // 2
        cy = r * cell_size + cell_size // 2
        radius = cell_size // 3
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=(255, 255, 255),
            outline=(0, 0, 0),
            width=2,
        )
    
    return img


def render_with_diff(
    grid: List[List[int]],
    position: Tuple[int, int],
    changes: List[Dict],
    ice_field: Optional[Tensor] = None,
    cell_size: int = 20,
) -> "Image.Image":
    """
    Render grid with diff visualization.
    
    Convenience wrapper for render_grid_to_image with change markers.
    
    Args:
        grid: 2D list of cell values
        position: Agent position (row, col)
        changes: List of CellChange or dict with row/col
        ice_field: Curiosity field tensor
        cell_size: Pixels per cell
        
    Returns:
        PIL Image with all overlays
    """
    # Convert CellChange objects to dicts if needed
    change_dicts = []
    for c in changes:
        if hasattr(c, 'row') and hasattr(c, 'col'):
            change_dicts.append({"row": c.row, "col": c.col})
        elif isinstance(c, dict):
            change_dicts.append(c)
    
    return render_grid_to_image(
        grid=grid,
        cell_size=cell_size,
        position=position,
        ice_field=ice_field,
        changes=change_dicts,
    )


# ─────────────────────────────────────────────
# ASCII Rendering (for terminals)
# ─────────────────────────────────────────────

def render_grid_ascii(
    grid: List[List[int]],
    position: Optional[Tuple[int, int]] = None,
    ice_field: Optional[Tensor] = None,
    max_width: int = 80,
    max_height: int = 40,
) -> str:
    """
    Render grid as ASCII art.
    
    Args:
        grid: 2D list of cell values
        position: Agent position (row, col)
        ice_field: Curiosity field for ice markers
        max_width: Maximum characters wide
        max_height: Maximum lines tall
        
    Returns:
        ASCII string representation
    """
    if not grid or not grid[0]:
        return "(empty grid)"
    
    grid_h = len(grid)
    grid_w = len(grid[0])
    
    # Determine display size
    chars_per_cell = 2
    display_w = min(grid_w, max_width // chars_per_cell)
    display_h = min(grid_h, max_height)
    
    # ASCII symbols for colors
    color_chars = {
        0: "  ",  # Background (space)
        1: "██",  # Blue
        2: "▓▓",  # Red
        3: "▒▒",  # Green
        4: "░░",  # Yellow
        5: "##",  # Gray
        6: "**",  # Magenta
        7: "++",  # Orange
        8: "~~",  # Cyan
        9: "@@",  # Maroon
    }
    
    lines = []
    for r in range(display_h):
        row_str = ""
        for c in range(display_w):
            val = grid[r][c] if isinstance(grid[r][c], int) else 0
            
            # Check for position marker
            if position is not None and (r, c) == position:
                row_str += "@ "
                continue
            
            # Check for high curiosity (ice)
            if ice_field is not None:
                if 0 <= r < ice_field.shape[0] and 0 <= c < ice_field.shape[1]:
                    ice_val = ice_field[r, c].item()
                    if ice_val > 10.0:
                        row_str += "🧊"
                        continue
                    elif ice_val > 2.0:
                        row_str += "❄ "
                        continue
            
            # Normal cell
            row_str += color_chars.get(val % 10, "??")
        
        lines.append(row_str)
    
    # Add truncation notice
    if display_h < grid_h or display_w < grid_w:
        lines.append(f"  (showing {display_h}x{display_w} of {grid_h}x{grid_w})")
    
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Spatial Memory Visualization
# ─────────────────────────────────────────────

def render_spatial_memory(
    exploration_mass: Tensor,
    curiosity_field: Tensor,
    position: Optional[Tuple[int, int]] = None,
    cell_size: int = 10,
) -> "Image.Image":
    """
    Render spatial memory as a visualization image.
    
    Shows exploration (red/orange heat) and curiosity (cyan ice).
    
    Args:
        exploration_mass: [H, W] tensor of exploration values
        curiosity_field: [H, W] tensor of curiosity values
        position: Current position to mark
        cell_size: Pixels per cell
        
    Returns:
        PIL Image
    """
    if not PIL_AVAILABLE:
        raise RuntimeError("PIL not available")
    
    h, w = exploration_mass.shape
    img_w = w * cell_size
    img_h = h * cell_size
    img = Image.new('RGB', (img_w, img_h), (20, 20, 30))
    draw = ImageDraw.Draw(img)
    
    # Normalize values
    max_mass = exploration_mass.max().item() + 1e-6
    max_ice = curiosity_field.max().item() + 1e-6
    
    for r in range(h):
        for c in range(w):
            x0 = c * cell_size
            y0 = r * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            
            mass = exploration_mass[r, c].item()
            ice = curiosity_field[r, c].item()
            
            # Blend exploration (red) and curiosity (cyan)
            mass_norm = mass / max_mass
            ice_norm = ice / max_ice
            
            r_val = int(255 * mass_norm)
            g_val = int(180 * ice_norm)
            b_val = int(255 * ice_norm)
            
            color = (r_val, g_val, b_val)
            draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=color)
    
    # Draw position
    if position is not None:
        pr, pc = position
        cx = pc * cell_size + cell_size // 2
        cy = pr * cell_size + cell_size // 2
        radius = cell_size // 2
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=(255, 255, 255),
            outline=(0, 0, 0),
            width=1,
        )
    
    return img


# ─────────────────────────────────────────────
# Comparison Image
# ─────────────────────────────────────────────

def create_comparison_image(
    before: List[List[int]],
    after: List[List[int]],
    cell_size: int = 20,
) -> "Image.Image":
    """
    Create side-by-side comparison of two grids.
    
    Args:
        before: Grid before action
        after: Grid after action
        cell_size: Pixels per cell
        
    Returns:
        PIL Image with before/after side by side
    """
    if not PIL_AVAILABLE:
        raise RuntimeError("PIL not available")
    
    img_before = render_grid_to_image(before, cell_size)
    img_after = render_grid_to_image(after, cell_size)
    
    # Create combined image
    combined_w = img_before.width + 10 + img_after.width
    combined_h = max(img_before.height, img_after.height)
    combined = Image.new('RGB', (combined_w, combined_h), (50, 50, 50))
    
    combined.paste(img_before, (0, 0))
    combined.paste(img_after, (img_before.width + 10, 0))
    
    return combined
