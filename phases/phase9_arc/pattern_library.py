"""
Phase 9 ARC: Pattern Library

50+ primitive transformations that cover nearly all ARC tasks.

Categories:
    C: Color operations
    G: Geometric operations
    T: Topological operations
    P: Pattern operations
    O: Object operations
    X: Compositional operations

Each pattern is implemented as a callable that takes a grid and returns
a transformed grid. Patterns can be composed via apply_sequence().
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import sys
from pathlib import Path

# Add project paths
_PHASES_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PHASES_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from phases.phase9_arc.object_detector import ObjectDetector, ObjectGraph, ARCObject, reconstruct_grid
except ImportError:
    try:
        from object_detector import ObjectDetector, ObjectGraph, ARCObject, reconstruct_grid
    except ImportError:
        # Stub when not available
        ObjectDetector = None
        ObjectGraph = None
        ARCObject = None
        reconstruct_grid = None


# ─────────────────────────────────────────────
# Pattern Registry
# ─────────────────────────────────────────────

@dataclass
class Pattern:
    """Single transformation pattern."""
    id: str
    name: str
    category: str
    description: str
    fn: Callable[[Tensor], Tensor]
    params: List[str] = None  # Parameter names (e.g., ['color1', 'color2'])
    
    def __call__(self, grid: Tensor, **kwargs) -> Tensor:
        return self.fn(grid, **kwargs)


PATTERNS: Dict[str, Pattern] = {}


def register_pattern(id: str, name: str, category: str, description: str, params: List[str] = None):
    """Decorator to register a pattern."""
    def decorator(fn: Callable):
        PATTERNS[id] = Pattern(
            id=id,
            name=name,
            category=category,
            description=description,
            fn=fn,
            params=params or [],
        )
        return fn
    return decorator


# ─────────────────────────────────────────────
# Color Operations (C)
# ─────────────────────────────────────────────

@register_pattern("C1", "color_swap", "color", "Swap two colors", ["color1", "color2"])
def color_swap(grid: Tensor, color1: int = 1, color2: int = 2) -> Tensor:
    """Swap all occurrences of color1 and color2."""
    result = grid.clone()
    mask1 = grid == color1
    mask2 = grid == color2
    result[mask1] = color2
    result[mask2] = color1
    return result


@register_pattern("C2", "color_fill", "color", "Fill entire grid with color", ["color"])
def color_fill(grid: Tensor, color: int = 0) -> Tensor:
    """Fill entire grid with a single color."""
    return torch.full_like(grid, color)


@register_pattern("C3", "color_replace", "color", "Replace all X with Y", ["from_color", "to_color"])
def color_replace(grid: Tensor, from_color: int = 0, to_color: int = 1) -> Tensor:
    """Replace all occurrences of one color with another."""
    result = grid.clone()
    result[grid == from_color] = to_color
    return result


@register_pattern("C4", "color_dominant", "color", "Fill with most common color", [])
def color_dominant(grid: Tensor) -> Tensor:
    """Fill grid with the most common color."""
    colors, counts = torch.unique(grid, return_counts=True)
    dominant = colors[counts.argmax()].item()
    return torch.full_like(grid, dominant)


@register_pattern("C5", "color_invert", "color", "Invert colors (9-x)", [])
def color_invert(grid: Tensor) -> Tensor:
    """Invert all colors (9 - x)."""
    return 9 - grid


@register_pattern("C6", "color_rarest", "color", "Keep only rarest color", [])
def color_rarest(grid: Tensor) -> Tensor:
    """Keep only the rarest non-background color."""
    colors, counts = torch.unique(grid, return_counts=True)
    # Exclude background (0) if present
    if 0 in colors:
        mask = colors != 0
        colors = colors[mask]
        counts = counts[mask]
    
    if len(colors) == 0:
        return grid.clone()
    
    rarest = colors[counts.argmin()].item()
    result = torch.zeros_like(grid)
    result[grid == rarest] = rarest
    return result


@register_pattern("C7", "color_count_to_value", "color", "Replace colors with their count", [])
def color_count_to_value(grid: Tensor) -> Tensor:
    """Replace each color with the count of that color (mod 10)."""
    result = grid.clone()
    colors, counts = torch.unique(grid, return_counts=True)
    for c, cnt in zip(colors.tolist(), counts.tolist()):
        result[grid == c] = cnt % 10
    return result


# ─────────────────────────────────────────────
# Geometric Operations (G)
# ─────────────────────────────────────────────

@register_pattern("G1", "rotate_90", "geometric", "Rotate 90° clockwise", [])
def rotate_90(grid: Tensor) -> Tensor:
    """Rotate grid 90° clockwise."""
    return torch.rot90(grid, k=-1)


@register_pattern("G2", "rotate_180", "geometric", "Rotate 180°", [])
def rotate_180(grid: Tensor) -> Tensor:
    """Rotate grid 180°."""
    return torch.rot90(grid, k=2)


@register_pattern("G3", "rotate_270", "geometric", "Rotate 270° clockwise (90° counter-clockwise)", [])
def rotate_270(grid: Tensor) -> Tensor:
    """Rotate grid 270° clockwise (90° counter-clockwise)."""
    return torch.rot90(grid, k=1)


@register_pattern("G4", "mirror_h", "geometric", "Mirror horizontally", [])
def mirror_h(grid: Tensor) -> Tensor:
    """Mirror grid horizontally (flip columns)."""
    return torch.flip(grid, dims=[1])


@register_pattern("G5", "mirror_v", "geometric", "Mirror vertically", [])
def mirror_v(grid: Tensor) -> Tensor:
    """Mirror grid vertically (flip rows)."""
    return torch.flip(grid, dims=[0])


@register_pattern("G6", "mirror_diag", "geometric", "Mirror diagonally (transpose)", [])
def mirror_diag(grid: Tensor) -> Tensor:
    """Mirror grid diagonally (transpose)."""
    return grid.T


@register_pattern("G7", "translate", "geometric", "Translate by (dr, dc)", ["dr", "dc"])
def translate(grid: Tensor, dr: int = 0, dc: int = 1) -> Tensor:
    """Translate grid by (dr, dc), wrapping or filling with 0."""
    H, W = grid.shape
    result = torch.zeros_like(grid)
    
    for i in range(H):
        for j in range(W):
            ni, nj = i - dr, j - dc
            if 0 <= ni < H and 0 <= nj < W:
                result[i, j] = grid[ni, nj]
    
    return result


@register_pattern("G8", "scale_up_2x", "geometric", "Scale up 2x", [])
def scale_up_2x(grid: Tensor) -> Tensor:
    """Scale grid up by 2x."""
    return grid.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)


@register_pattern("G9", "scale_up_3x", "geometric", "Scale up 3x", [])
def scale_up_3x(grid: Tensor) -> Tensor:
    """Scale grid up by 3x."""
    return grid.repeat_interleave(3, dim=0).repeat_interleave(3, dim=1)


@register_pattern("G10", "crop_to_content", "geometric", "Crop to bounding box of non-zero", [])
def crop_to_content(grid: Tensor) -> Tensor:
    """Crop grid to bounding box of non-zero cells."""
    nonzero = torch.nonzero(grid)
    if len(nonzero) == 0:
        return grid
    
    r_min, c_min = nonzero.min(dim=0).values.tolist()
    r_max, c_max = nonzero.max(dim=0).values.tolist()
    
    return grid[r_min:r_max+1, c_min:c_max+1]


@register_pattern("G11", "pad_to_square", "geometric", "Pad to square shape", [])
def pad_to_square(grid: Tensor) -> Tensor:
    """Pad grid to make it square."""
    H, W = grid.shape
    max_dim = max(H, W)
    
    result = torch.zeros(max_dim, max_dim, dtype=grid.dtype)
    result[:H, :W] = grid
    return result


# ─────────────────────────────────────────────
# Topological Operations (T)
# ─────────────────────────────────────────────

@register_pattern("T1", "flood_fill", "topological", "Flood fill from point", ["row", "col", "color"])
def flood_fill(grid: Tensor, row: int = 0, col: int = 0, color: int = 1) -> Tensor:
    """Flood fill from (row, col) with given color."""
    from collections import deque
    
    result = grid.clone()
    H, W = grid.shape
    
    if not (0 <= row < H and 0 <= col < W):
        return result
    
    target = grid[row, col].item()
    if target == color:
        return result
    
    queue = deque([(row, col)])
    visited = set()
    
    while queue:
        r, c = queue.popleft()
        if (r, c) in visited:
            continue
        visited.add((r, c))
        
        if not (0 <= r < H and 0 <= c < W):
            continue
        if result[r, c].item() != target:
            continue
        
        result[r, c] = color
        queue.extend([(r-1, c), (r+1, c), (r, c-1), (r, c+1)])
    
    return result


@register_pattern("T2", "fill_enclosed", "topological", "Fill enclosed regions", ["color"])
def fill_enclosed(grid: Tensor, color: int = 1) -> Tensor:
    """Fill enclosed regions (regions not touching border)."""
    H, W = grid.shape
    result = grid.clone()
    
    # Find all cells connected to border (these are NOT enclosed)
    from collections import deque
    border_connected = set()
    queue = deque()
    
    # Add border cells to queue
    for i in range(H):
        if grid[i, 0].item() == 0:
            queue.append((i, 0))
        if grid[i, W-1].item() == 0:
            queue.append((i, W-1))
    for j in range(W):
        if grid[0, j].item() == 0:
            queue.append((0, j))
        if grid[H-1, j].item() == 0:
            queue.append((H-1, j))
    
    while queue:
        r, c = queue.popleft()
        if (r, c) in border_connected:
            continue
        if not (0 <= r < H and 0 <= c < W):
            continue
        if grid[r, c].item() != 0:
            continue
        
        border_connected.add((r, c))
        queue.extend([(r-1, c), (r+1, c), (r, c-1), (r, c+1)])
    
    # Fill cells that are 0 but not border-connected
    for i in range(H):
        for j in range(W):
            if grid[i, j].item() == 0 and (i, j) not in border_connected:
                result[i, j] = color
    
    return result


@register_pattern("T3", "dilate", "topological", "Dilate (expand) by 1 cell", ["color"])
def dilate(grid: Tensor, color: int = None) -> Tensor:
    """Expand non-zero regions by 1 cell in all directions."""
    H, W = grid.shape
    result = grid.clone()
    
    for i in range(H):
        for j in range(W):
            if grid[i, j].item() != 0:
                c = color if color is not None else grid[i, j].item()
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W and result[ni, nj].item() == 0:
                        result[ni, nj] = c
    
    return result


@register_pattern("T4", "erode", "topological", "Erode (shrink) by 1 cell", [])
def erode(grid: Tensor) -> Tensor:
    """Shrink non-zero regions by 1 cell."""
    H, W = grid.shape
    result = grid.clone()
    
    for i in range(H):
        for j in range(W):
            if grid[i, j].item() != 0:
                # Check if any neighbor is 0 or border
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if not (0 <= ni < H and 0 <= nj < W) or grid[ni, nj].item() == 0:
                        result[i, j] = 0
                        break
    
    return result


@register_pattern("T5", "boundary", "topological", "Extract boundaries only", [])
def boundary(grid: Tensor) -> Tensor:
    """Extract boundary cells (non-zero cells adjacent to zero)."""
    H, W = grid.shape
    result = torch.zeros_like(grid)
    
    for i in range(H):
        for j in range(W):
            if grid[i, j].item() != 0:
                # Check if any neighbor is 0 or border
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if not (0 <= ni < H and 0 <= nj < W) or grid[ni, nj].item() == 0:
                        result[i, j] = grid[i, j]
                        break
    
    return result


@register_pattern("T6", "interior", "topological", "Extract interior only", [])
def interior(grid: Tensor) -> Tensor:
    """Extract interior cells (non-zero cells NOT adjacent to zero)."""
    H, W = grid.shape
    result = torch.zeros_like(grid)
    
    for i in range(H):
        for j in range(W):
            if grid[i, j].item() != 0:
                is_interior = True
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if not (0 <= ni < H and 0 <= nj < W) or grid[ni, nj].item() == 0:
                        is_interior = False
                        break
                if is_interior:
                    result[i, j] = grid[i, j]
    
    return result


# ─────────────────────────────────────────────
# Pattern Operations (P)
# ─────────────────────────────────────────────

@register_pattern("P1", "tile_2x2", "pattern", "Tile grid 2x2", [])
def tile_2x2(grid: Tensor) -> Tensor:
    """Tile grid in 2x2 pattern."""
    return torch.cat([
        torch.cat([grid, grid], dim=1),
        torch.cat([grid, grid], dim=1),
    ], dim=0)


@register_pattern("P2", "tile_3x3", "pattern", "Tile grid 3x3", [])
def tile_3x3(grid: Tensor) -> Tensor:
    """Tile grid in 3x3 pattern."""
    row = torch.cat([grid, grid, grid], dim=1)
    return torch.cat([row, row, row], dim=0)


@register_pattern("P3", "checkerboard", "pattern", "Apply checkerboard pattern", ["color1", "color2"])
def checkerboard(grid: Tensor, color1: int = 0, color2: int = 1) -> Tensor:
    """Create checkerboard pattern of given size."""
    H, W = grid.shape
    result = torch.zeros_like(grid)
    for i in range(H):
        for j in range(W):
            result[i, j] = color1 if (i + j) % 2 == 0 else color2
    return result


@register_pattern("P4", "grid_lines", "pattern", "Add grid lines every N cells", ["spacing", "color"])
def grid_lines(grid: Tensor, spacing: int = 3, color: int = 1) -> Tensor:
    """Add grid lines every N cells."""
    result = grid.clone()
    H, W = grid.shape
    
    for i in range(0, H, spacing):
        result[i, :] = color
    for j in range(0, W, spacing):
        result[:, j] = color
    
    return result


@register_pattern("P5", "make_symmetric_h", "pattern", "Make horizontally symmetric", [])
def make_symmetric_h(grid: Tensor) -> Tensor:
    """Copy left half to right half (mirrored)."""
    H, W = grid.shape
    result = grid.clone()
    mid = W // 2
    result[:, mid:] = torch.flip(grid[:, :mid], dims=[1])[:, :W-mid]
    return result


@register_pattern("P6", "make_symmetric_v", "pattern", "Make vertically symmetric", [])
def make_symmetric_v(grid: Tensor) -> Tensor:
    """Copy top half to bottom half (mirrored)."""
    H, W = grid.shape
    result = grid.clone()
    mid = H // 2
    result[mid:, :] = torch.flip(grid[:mid, :], dims=[0])[:H-mid, :]
    return result


# ─────────────────────────────────────────────
# Object Operations (O)
# ─────────────────────────────────────────────

@register_pattern("O1", "keep_largest", "object", "Keep only largest object", [])
def keep_largest(grid: Tensor) -> Tensor:
    """Keep only the largest object, remove all others."""
    detector = ObjectDetector()
    graph = detector.detect(grid)
    
    if not graph.objects:
        return torch.zeros_like(grid)
    
    largest = graph.get_largest()
    result = torch.zeros_like(grid)
    for r, c in largest.cells:
        result[r, c] = largest.color
    
    return result


@register_pattern("O2", "keep_smallest", "object", "Keep only smallest object", [])
def keep_smallest(grid: Tensor) -> Tensor:
    """Keep only the smallest object, remove all others."""
    detector = ObjectDetector()
    graph = detector.detect(grid)
    
    if not graph.objects:
        return torch.zeros_like(grid)
    
    smallest = graph.get_smallest()
    result = torch.zeros_like(grid)
    for r, c in smallest.cells:
        result[r, c] = smallest.color
    
    return result


@register_pattern("O3", "remove_largest", "object", "Remove largest object", [])
def remove_largest(grid: Tensor) -> Tensor:
    """Remove the largest object, keep all others."""
    detector = ObjectDetector()
    graph = detector.detect(grid)
    
    result = grid.clone()
    if not graph.objects:
        return result
    
    largest = graph.get_largest()
    for r, c in largest.cells:
        result[r, c] = 0
    
    return result


@register_pattern("O4", "remove_smallest", "object", "Remove smallest object", [])
def remove_smallest(grid: Tensor) -> Tensor:
    """Remove the smallest object, keep all others."""
    detector = ObjectDetector()
    graph = detector.detect(grid)
    
    result = grid.clone()
    if not graph.objects:
        return result
    
    smallest = graph.get_smallest()
    for r, c in smallest.cells:
        result[r, c] = 0
    
    return result


@register_pattern("O5", "keep_by_color", "object", "Keep only objects of color X", ["color"])
def keep_by_color(grid: Tensor, color: int = 1) -> Tensor:
    """Keep only objects of specified color."""
    result = torch.zeros_like(grid)
    result[grid == color] = color
    return result


@register_pattern("O6", "remove_by_color", "object", "Remove objects of color X", ["color"])
def remove_by_color(grid: Tensor, color: int = 1) -> Tensor:
    """Remove all objects of specified color."""
    result = grid.clone()
    result[grid == color] = 0
    return result


@register_pattern("O7", "count_objects", "object", "Output is count of objects", [])
def count_objects(grid: Tensor) -> Tensor:
    """Create 1x1 grid with count of objects."""
    detector = ObjectDetector()
    graph = detector.detect(grid)
    count = len(graph.objects)
    return torch.tensor([[count % 10]], dtype=grid.dtype)


@register_pattern("O8", "count_colors", "object", "Output is count of unique colors", [])
def count_colors(grid: Tensor) -> Tensor:
    """Create 1x1 grid with count of unique non-zero colors."""
    colors = torch.unique(grid)
    count = len([c for c in colors.tolist() if c != 0])
    return torch.tensor([[count % 10]], dtype=grid.dtype)


# ─────────────────────────────────────────────
# Compositional Operations (X)
# ─────────────────────────────────────────────

def apply_sequence(grid: Tensor, pattern_ids: List[str], **kwargs) -> Tensor:
    """Apply a sequence of patterns."""
    result = grid.clone()
    for pid in pattern_ids:
        if pid in PATTERNS:
            result = PATTERNS[pid](result, **kwargs)
    return result


def apply_if_condition(
    grid: Tensor,
    condition_fn: Callable[[Tensor], bool],
    true_pattern: str,
    false_pattern: str = None,
    **kwargs,
) -> Tensor:
    """Apply pattern based on condition."""
    if condition_fn(grid):
        return PATTERNS[true_pattern](grid, **kwargs)
    elif false_pattern:
        return PATTERNS[false_pattern](grid, **kwargs)
    return grid


def apply_to_each_object(
    grid: Tensor,
    transform_fn: Callable[[ARCObject], ARCObject],
) -> Tensor:
    """Apply transformation to each object independently."""
    detector = ObjectDetector()
    graph = detector.detect(grid)
    
    result = torch.zeros_like(grid)
    for obj in graph.objects:
        transformed = transform_fn(obj)
        for r, c in transformed.cells:
            H, W = grid.shape
            if 0 <= r < H and 0 <= c < W:
                result[r, c] = transformed.color
    
    return result


# ─────────────────────────────────────────────
# Gravity / Physics Operations (V)
# ─────────────────────────────────────────────

@register_pattern("V1", "gravity_down", "physics", "Drop all non-zero cells down", [])
def gravity_down(grid: Tensor) -> Tensor:
    """Apply gravity - all non-zero cells fall down."""
    H, W = grid.shape
    result = torch.zeros_like(grid)
    
    for j in range(W):
        # Collect non-zero cells in column
        column = []
        for i in range(H):
            if grid[i, j].item() != 0:
                column.append(grid[i, j].item())
        
        # Place at bottom
        for idx, val in enumerate(reversed(column)):
            result[H - 1 - idx, j] = val
    
    return result


@register_pattern("V2", "gravity_up", "physics", "Push all non-zero cells up", [])
def gravity_up(grid: Tensor) -> Tensor:
    """Apply upward gravity - all non-zero cells rise up."""
    H, W = grid.shape
    result = torch.zeros_like(grid)
    
    for j in range(W):
        column = []
        for i in range(H):
            if grid[i, j].item() != 0:
                column.append(grid[i, j].item())
        
        for idx, val in enumerate(column):
            result[idx, j] = val
    
    return result


@register_pattern("V3", "gravity_left", "physics", "Push all non-zero cells left", [])
def gravity_left(grid: Tensor) -> Tensor:
    """Apply leftward gravity."""
    H, W = grid.shape
    result = torch.zeros_like(grid)
    
    for i in range(H):
        row = []
        for j in range(W):
            if grid[i, j].item() != 0:
                row.append(grid[i, j].item())
        
        for idx, val in enumerate(row):
            result[i, idx] = val
    
    return result


@register_pattern("V4", "gravity_right", "physics", "Push all non-zero cells right", [])
def gravity_right(grid: Tensor) -> Tensor:
    """Apply rightward gravity."""
    H, W = grid.shape
    result = torch.zeros_like(grid)
    
    for i in range(H):
        row = []
        for j in range(W):
            if grid[i, j].item() != 0:
                row.append(grid[i, j].item())
        
        for idx, val in enumerate(reversed(row)):
            result[i, W - 1 - idx] = val
    
    return result


# ─────────────────────────────────────────────
# Overlay / Masking Operations (M)
# ─────────────────────────────────────────────

@register_pattern("M1", "mask_nonzero", "masking", "Keep only cells where mask is non-zero", ["mask"])
def mask_nonzero(grid: Tensor, mask: Tensor = None) -> Tensor:
    """Apply binary mask - keep grid values only where mask is non-zero."""
    if mask is None:
        return grid.clone()
    
    result = torch.zeros_like(grid)
    result[mask != 0] = grid[mask != 0]
    return result


@register_pattern("M2", "overlay_nonzero", "masking", "Overlay non-zero cells from second grid", ["overlay"])
def overlay_nonzero(grid: Tensor, overlay: Tensor = None) -> Tensor:
    """Overlay non-zero cells from another grid on top."""
    if overlay is None:
        return grid.clone()
    
    result = grid.clone()
    result[overlay != 0] = overlay[overlay != 0]
    return result


@register_pattern("M3", "xor_grids", "masking", "XOR between two grids (keep differences)", ["other"])
def xor_grids(grid: Tensor, other: Tensor = None) -> Tensor:
    """Keep cells that differ between grids."""
    if other is None:
        return grid.clone()
    
    result = torch.zeros_like(grid)
    diff_mask = grid != other
    result[diff_mask] = grid[diff_mask]
    return result


# ─────────────────────────────────────────────
# Centering / Alignment Operations (A)
# ─────────────────────────────────────────────

@register_pattern("A1", "center_content", "alignment", "Center non-zero content in grid", [])
def center_content(grid: Tensor) -> Tensor:
    """Center the non-zero content in the grid."""
    H, W = grid.shape
    
    nonzero = torch.nonzero(grid)
    if len(nonzero) == 0:
        return grid.clone()
    
    r_min, c_min = nonzero.min(dim=0).values.tolist()
    r_max, c_max = nonzero.max(dim=0).values.tolist()
    
    content_h = r_max - r_min + 1
    content_w = c_max - c_min + 1
    
    # Calculate offset to center
    offset_r = (H - content_h) // 2 - r_min
    offset_c = (W - content_w) // 2 - c_min
    
    result = torch.zeros_like(grid)
    for r in range(r_min, r_max + 1):
        for c in range(c_min, c_max + 1):
            new_r, new_c = r + offset_r, c + offset_c
            if 0 <= new_r < H and 0 <= new_c < W:
                result[new_r, new_c] = grid[r, c]
    
    return result


@register_pattern("A2", "align_top_left", "alignment", "Align content to top-left corner", [])
def align_top_left(grid: Tensor) -> Tensor:
    """Align non-zero content to top-left."""
    nonzero = torch.nonzero(grid)
    if len(nonzero) == 0:
        return grid.clone()
    
    r_min, c_min = nonzero.min(dim=0).values.tolist()
    return translate(grid, dr=-r_min, dc=-c_min)


@register_pattern("A3", "align_bottom_right", "alignment", "Align content to bottom-right corner", [])
def align_bottom_right(grid: Tensor) -> Tensor:
    """Align non-zero content to bottom-right."""
    H, W = grid.shape
    
    nonzero = torch.nonzero(grid)
    if len(nonzero) == 0:
        return grid.clone()
    
    r_min, c_min = nonzero.min(dim=0).values.tolist()
    r_max, c_max = nonzero.max(dim=0).values.tolist()
    
    content_h = r_max - r_min + 1
    content_w = c_max - c_min + 1
    
    offset_r = (H - content_h) - r_min
    offset_c = (W - content_w) - c_min
    
    return translate(grid, dr=offset_r, dc=offset_c)


# ─────────────────────────────────────────────
# Completion / Extension Operations (E)
# ─────────────────────────────────────────────

@register_pattern("E1", "extend_lines_h", "extension", "Extend horizontal lines to edges", [])
def extend_lines_h(grid: Tensor) -> Tensor:
    """Extend horizontal lines to grid edges."""
    H, W = grid.shape
    result = grid.clone()
    
    for i in range(H):
        # Find first and last non-zero in row
        row = grid[i]
        nonzero_idx = torch.nonzero(row).flatten()
        
        if len(nonzero_idx) > 0:
            color = row[nonzero_idx[0]].item()
            # Check if it's a horizontal line (same color)
            if all(row[idx].item() == color for idx in nonzero_idx):
                # Extend to full row
                result[i, :] = color
    
    return result


@register_pattern("E2", "extend_lines_v", "extension", "Extend vertical lines to edges", [])
def extend_lines_v(grid: Tensor) -> Tensor:
    """Extend vertical lines to grid edges."""
    H, W = grid.shape
    result = grid.clone()
    
    for j in range(W):
        col = grid[:, j]
        nonzero_idx = torch.nonzero(col).flatten()
        
        if len(nonzero_idx) > 0:
            color = col[nonzero_idx[0]].item()
            if all(col[idx].item() == color for idx in nonzero_idx):
                result[:, j] = color
    
    return result


@register_pattern("E3", "complete_rectangle", "extension", "Complete partial rectangle", [])
def complete_rectangle(grid: Tensor) -> Tensor:
    """Complete any partial rectangle outline."""
    detector = ObjectDetector()
    graph = detector.detect(grid)
    result = grid.clone()
    
    for obj in graph.objects:
        y1, x1, y2, x2 = obj.bbox
        color = obj.color
        
        # Draw complete rectangle outline
        for r in range(y1, y2):
            result[r, x1] = color
            result[r, x2 - 1] = color
        for c in range(x1, x2):
            result[y1, c] = color
            result[y2 - 1, c] = color
    
    return result


@register_pattern("E4", "fill_rectangle", "extension", "Fill bounding box of each object", [])
def fill_rectangle(grid: Tensor) -> Tensor:
    """Fill the bounding box of each object with its color."""
    detector = ObjectDetector()
    graph = detector.detect(grid)
    result = torch.zeros_like(grid)
    
    for obj in graph.objects:
        y1, x1, y2, x2 = obj.bbox
        result[y1:y2, x1:x2] = obj.color
    
    return result


# ─────────────────────────────────────────────
# Sorting / Ordering Operations (S)
# ─────────────────────────────────────────────

@register_pattern("S1", "sort_rows_by_sum", "sorting", "Sort rows by sum of values", [])
def sort_rows_by_sum(grid: Tensor) -> Tensor:
    """Sort rows by their sum (ascending)."""
    H, W = grid.shape
    row_sums = grid.sum(dim=1)
    _, indices = torch.sort(row_sums)
    return grid[indices]


@register_pattern("S2", "sort_cols_by_sum", "sorting", "Sort columns by sum of values", [])
def sort_cols_by_sum(grid: Tensor) -> Tensor:
    """Sort columns by their sum (ascending)."""
    col_sums = grid.sum(dim=0)
    _, indices = torch.sort(col_sums)
    return grid[:, indices]


@register_pattern("S3", "reverse_rows", "sorting", "Reverse order of rows", [])
def reverse_rows(grid: Tensor) -> Tensor:
    """Reverse the order of rows."""
    return torch.flip(grid, dims=[0])


@register_pattern("S4", "reverse_cols", "sorting", "Reverse order of columns", [])
def reverse_cols(grid: Tensor) -> Tensor:
    """Reverse the order of columns."""
    return torch.flip(grid, dims=[1])


# ─────────────────────────────────────────────
# Downscaling Operations (D) 
# ─────────────────────────────────────────────

@register_pattern("D1", "scale_down_2x", "scaling", "Scale down by 2x (majority vote)", [])
def scale_down_2x(grid: Tensor) -> Tensor:
    """Scale grid down by 2x using majority voting in 2x2 blocks."""
    H, W = grid.shape
    new_H, new_W = H // 2, W // 2
    
    if new_H == 0 or new_W == 0:
        return grid.clone()
    
    result = torch.zeros(new_H, new_W, dtype=grid.dtype)
    
    for i in range(new_H):
        for j in range(new_W):
            # Get 2x2 block
            block = grid[i*2:i*2+2, j*2:j*2+2].flatten()
            # Majority vote
            values, counts = torch.unique(block, return_counts=True)
            result[i, j] = values[counts.argmax()]
    
    return result


@register_pattern("D2", "scale_down_3x", "scaling", "Scale down by 3x (majority vote)", [])
def scale_down_3x(grid: Tensor) -> Tensor:
    """Scale grid down by 3x using majority voting in 3x3 blocks."""
    H, W = grid.shape
    new_H, new_W = H // 3, W // 3
    
    if new_H == 0 or new_W == 0:
        return grid.clone()
    
    result = torch.zeros(new_H, new_W, dtype=grid.dtype)
    
    for i in range(new_H):
        for j in range(new_W):
            block = grid[i*3:i*3+3, j*3:j*3+3].flatten()
            values, counts = torch.unique(block, return_counts=True)
            result[i, j] = values[counts.argmax()]
    
    return result


# ─────────────────────────────────────────────
# Filtering Operations (F)
# ─────────────────────────────────────────────

@register_pattern("F1", "filter_by_area_min", "filtering", "Keep only objects >= min area", ["min_area"])
def filter_by_area_min(grid: Tensor, min_area: int = 4) -> Tensor:
    """Keep only objects with area >= min_area."""
    detector = ObjectDetector()
    graph = detector.detect(grid)
    result = torch.zeros_like(grid)
    
    for obj in graph.objects:
        if obj.area >= min_area:
            for r, c in obj.cells:
                result[r, c] = obj.color
    
    return result


@register_pattern("F2", "filter_by_area_max", "filtering", "Keep only objects <= max area", ["max_area"])
def filter_by_area_max(grid: Tensor, max_area: int = 4) -> Tensor:
    """Keep only objects with area <= max_area."""
    detector = ObjectDetector()
    graph = detector.detect(grid)
    result = torch.zeros_like(grid)
    
    for obj in graph.objects:
        if obj.area <= max_area:
            for r, c in obj.cells:
                result[r, c] = obj.color
    
    return result


@register_pattern("F3", "filter_rectangles", "filtering", "Keep only rectangular objects", [])
def filter_rectangles(grid: Tensor) -> Tensor:
    """Keep only objects that are perfect rectangles."""
    detector = ObjectDetector()
    graph = detector.detect(grid)
    result = torch.zeros_like(grid)
    
    for obj in graph.objects:
        if obj.is_rectangular:
            for r, c in obj.cells:
                result[r, c] = obj.color
    
    return result


@register_pattern("F4", "filter_lines", "filtering", "Keep only line-shaped objects", [])
def filter_lines(grid: Tensor) -> Tensor:
    """Keep only objects that are lines (width=1 or height=1)."""
    detector = ObjectDetector()
    graph = detector.detect(grid)
    result = torch.zeros_like(grid)
    
    for obj in graph.objects:
        if obj.is_line:
            for r, c in obj.cells:
                result[r, c] = obj.color
    
    return result


# ─────────────────────────────────────────────
# Pattern Query
# ─────────────────────────────────────────────

def list_patterns() -> List[str]:
    """List all registered pattern IDs."""
    return list(PATTERNS.keys())


def get_pattern(id: str) -> Optional[Pattern]:
    """Get pattern by ID."""
    return PATTERNS.get(id)


def patterns_by_category(category: str) -> List[Pattern]:
    """Get all patterns in a category."""
    return [p for p in PATTERNS.values() if p.category == category]


def describe_patterns() -> str:
    """Get table of all patterns."""
    lines = ["| ID | Name | Category | Description |", "|-----|------|----------|-------------|"]
    for p in PATTERNS.values():
        lines.append(f"| {p.id} | {p.name} | {p.category} | {p.description} |")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("Pattern Library Test")
    print("=" * 50)
    print(f"Total patterns: {len(PATTERNS)}")
    print()
    
    # Show categories
    categories = {}
    for p in PATTERNS.values():
        categories[p.category] = categories.get(p.category, 0) + 1
    
    print("By category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    
    # Test a few patterns
    print("\n" + "=" * 50)
    print("Testing patterns:")
    
    grid = torch.tensor([
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [2, 2, 0, 0],
        [0, 0, 3, 3],
    ])
    
    print("\nOriginal:")
    print(grid)
    
    print("\nAfter rotate_90:")
    print(rotate_90(grid))
    
    print("\nAfter mirror_h:")
    print(mirror_h(grid))
    
    print("\nAfter color_swap(1, 2):")
    print(color_swap(grid, 1, 2))
    
    print("\nAfter keep_largest:")
    print(keep_largest(grid))
    
    print("\nAfter dilate:")
    print(dilate(grid))
