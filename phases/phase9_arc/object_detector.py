"""
Phase 9 ARC: Object Detector

Extract objects (connected components) from ARC grids.

Objects are defined as:
    - Connected components of same color (4-connected by default)
    - Each object has properties: color, mask, bbox, area, centroid

This is CRITICAL for ARC because most tasks operate on objects,
not individual pixels. The challenge categories all reference "objects":
    - "Find the smallest shape and remove it"
    - "Move objects to edges"
    - "Count objects"
    - "Fill enclosed areas"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import sys
from pathlib import Path
from collections import deque

# Add project paths
_PHASES_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PHASES_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class ARCObject:
    """Single object extracted from grid."""
    object_id: int
    color: int
    mask: Tensor                    # [H, W] boolean mask
    cells: List[Tuple[int, int]]    # List of (row, col) coordinates
    
    @property
    def area(self) -> int:
        """Number of cells in object."""
        return len(self.cells)
    
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Bounding box (y1, x1, y2, x2)."""
        rows = [c[0] for c in self.cells]
        cols = [c[1] for c in self.cells]
        return min(rows), min(cols), max(rows) + 1, max(cols) + 1
    
    @property
    def centroid(self) -> Tuple[float, float]:
        """Center of mass (row, col)."""
        rows = [c[0] for c in self.cells]
        cols = [c[1] for c in self.cells]
        return sum(rows) / len(rows), sum(cols) / len(cols)
    
    @property
    def width(self) -> int:
        """Bounding box width."""
        y1, x1, y2, x2 = self.bbox
        return x2 - x1
    
    @property
    def height(self) -> int:
        """Bounding box height."""
        y1, x1, y2, x2 = self.bbox
        return y2 - y1
    
    @property
    def is_rectangular(self) -> bool:
        """Check if object is a filled rectangle."""
        return self.area == self.width * self.height
    
    @property
    def is_line(self) -> bool:
        """Check if object is a line (width=1 or height=1)."""
        return self.width == 1 or self.height == 1
    
    def extract_subgrid(self, grid: Tensor) -> Tensor:
        """Extract bounding box region from grid."""
        y1, x1, y2, x2 = self.bbox
        return grid[y1:y2, x1:x2]
    
    def to_isolated_grid(self) -> Tensor:
        """Create grid with only this object (others = 0)."""
        y1, x1, y2, x2 = self.bbox
        h, w = y2 - y1, x2 - x1
        grid = torch.zeros(h, w, dtype=torch.long)
        for r, c in self.cells:
            grid[r - y1, c - x1] = self.color
        return grid


@dataclass
class ObjectGraph:
    """All objects in a grid with spatial relationships."""
    objects: List[ARCObject]
    grid_shape: Tuple[int, int]
    background_color: int = 0
    
    def __len__(self) -> int:
        return len(self.objects)
    
    def __iter__(self):
        return iter(self.objects)
    
    def __getitem__(self, idx: int) -> ARCObject:
        return self.objects[idx]
    
    @property
    def colors(self) -> List[int]:
        """Unique colors in objects."""
        return list(set(obj.color for obj in self.objects))
    
    @property
    def total_object_area(self) -> int:
        """Total area covered by all objects."""
        return sum(obj.area for obj in self.objects)
    
    def get_by_color(self, color: int) -> List[ARCObject]:
        """Get all objects of a specific color."""
        return [obj for obj in self.objects if obj.color == color]
    
    def get_largest(self) -> Optional[ARCObject]:
        """Get largest object by area."""
        if not self.objects:
            return None
        return max(self.objects, key=lambda o: o.area)
    
    def get_smallest(self) -> Optional[ARCObject]:
        """Get smallest object by area."""
        if not self.objects:
            return None
        return min(self.objects, key=lambda o: o.area)
    
    def sorted_by_area(self, descending: bool = True) -> List[ARCObject]:
        """Get objects sorted by area."""
        return sorted(self.objects, key=lambda o: o.area, reverse=descending)
    
    def sorted_by_position(self, by: str = 'top-left') -> List[ARCObject]:
        """Sort by position in grid."""
        if by == 'top-left':
            return sorted(self.objects, key=lambda o: (o.centroid[0], o.centroid[1]))
        elif by == 'left-right':
            return sorted(self.objects, key=lambda o: o.centroid[1])
        elif by == 'top-bottom':
            return sorted(self.objects, key=lambda o: o.centroid[0])
        else:
            return self.objects
    
    def get_touching(self, obj: ARCObject) -> List[ARCObject]:
        """Get objects that touch (are adjacent to) the given object."""
        touching = []
        obj_cells = set(obj.cells)
        
        for other in self.objects:
            if other.object_id == obj.object_id:
                continue
            
            # Check if any cell in 'other' is adjacent to any cell in 'obj'
            for r, c in other.cells:
                neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
                if any(n in obj_cells for n in neighbors):
                    touching.append(other)
                    break
        
        return touching


# ─────────────────────────────────────────────
# Object Detection
# ─────────────────────────────────────────────

class ObjectDetector:
    """
    Extract objects from ARC grids using connected component analysis.
    
    Methods:
        detect(grid) → ObjectGraph with all objects
        detect_by_color(grid, ignore_colors=[0]) → ObjectGraph ignoring background
    """
    
    def __init__(
        self,
        connectivity: int = 4,
        ignore_colors: List[int] = None,
        min_area: int = 1,
    ):
        """
        Args:
            connectivity: 4 or 8 (4-connected or 8-connected)
            ignore_colors: Colors to treat as background (default: [0])
            min_area: Minimum object size (smaller = noise)
        """
        self.connectivity = connectivity
        self.ignore_colors = ignore_colors if ignore_colors is not None else [0]
        self.min_area = min_area
    
    def detect(
        self,
        grid: Tensor,
        ignore_colors: List[int] = None,
    ) -> ObjectGraph:
        """
        Detect all objects in grid.
        
        Args:
            grid: [H, W] integer tensor (values 0-9)
            ignore_colors: Override default ignored colors
        
        Returns:
            ObjectGraph containing all detected objects
        """
        if isinstance(grid, list):
            grid = torch.tensor(grid, dtype=torch.long)
        
        H, W = grid.shape
        visited = torch.zeros_like(grid, dtype=torch.bool)
        objects = []
        object_id = 0
        
        ignore = ignore_colors if ignore_colors is not None else self.ignore_colors
        
        # Flood fill to find connected components
        for i in range(H):
            for j in range(W):
                if visited[i, j]:
                    continue
                
                color = grid[i, j].item()
                if color in ignore:
                    visited[i, j] = True
                    continue
                
                # BFS to find connected component
                cells = []
                queue = deque([(i, j)])
                visited[i, j] = True
                
                while queue:
                    r, c = queue.popleft()
                    cells.append((r, c))
                    
                    # Get neighbors based on connectivity
                    if self.connectivity == 4:
                        neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
                    else:  # 8-connected
                        neighbors = [
                            (r-1, c), (r+1, c), (r, c-1), (r, c+1),
                            (r-1, c-1), (r-1, c+1), (r+1, c-1), (r+1, c+1),
                        ]
                    
                    for nr, nc in neighbors:
                        if 0 <= nr < H and 0 <= nc < W:
                            if not visited[nr, nc] and grid[nr, nc].item() == color:
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                
                # Create object if large enough
                if len(cells) >= self.min_area:
                    mask = torch.zeros(H, W, dtype=torch.bool)
                    for r, c in cells:
                        mask[r, c] = True
                    
                    objects.append(ARCObject(
                        object_id=object_id,
                        color=color,
                        mask=mask,
                        cells=cells,
                    ))
                    object_id += 1
        
        # Determine background color (most common ignored or most common overall)
        bg_color = 0
        if ignore:
            bg_color = ignore[0]
        
        return ObjectGraph(
            objects=objects,
            grid_shape=(H, W),
            background_color=bg_color,
        )
    
    def detect_all_colors(self, grid: Tensor) -> ObjectGraph:
        """Detect objects treating NO color as background."""
        return self.detect(grid, ignore_colors=[])
    
    def detect_regions(
        self,
        grid: Tensor,
        target_colors: List[int],
    ) -> ObjectGraph:
        """Detect objects only of specific colors."""
        full = self.detect(grid, ignore_colors=[])
        filtered = [obj for obj in full.objects if obj.color in target_colors]
        return ObjectGraph(
            objects=filtered,
            grid_shape=full.grid_shape,
            background_color=full.background_color,
        )


# ─────────────────────────────────────────────
# Spatial Analysis
# ─────────────────────────────────────────────

class SpatialAnalyzer:
    """Analyze spatial relationships between objects."""
    
    @staticmethod
    def distance(obj1: ARCObject, obj2: ARCObject) -> float:
        """Euclidean distance between object centroids."""
        c1 = obj1.centroid
        c2 = obj2.centroid
        return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5
    
    @staticmethod
    def manhattan_distance(obj1: ARCObject, obj2: ARCObject) -> int:
        """Manhattan distance between centroids."""
        c1 = obj1.centroid
        c2 = obj2.centroid
        return int(abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]))
    
    @staticmethod
    def relative_position(obj1: ARCObject, obj2: ARCObject) -> str:
        """Get relative position of obj2 to obj1."""
        c1 = obj1.centroid
        c2 = obj2.centroid
        
        dr = c2[0] - c1[0]  # positive = below
        dc = c2[1] - c1[1]  # positive = right
        
        if abs(dr) < 0.5 and abs(dc) < 0.5:
            return "same"
        elif abs(dr) > abs(dc):
            return "below" if dr > 0 else "above"
        else:
            return "right" if dc > 0 else "left"
    
    @staticmethod
    def are_aligned(obj1: ARCObject, obj2: ARCObject, axis: str = 'any') -> bool:
        """Check if objects are aligned horizontally or vertically."""
        c1 = obj1.centroid
        c2 = obj2.centroid
        
        h_aligned = abs(c1[0] - c2[0]) < 0.5
        v_aligned = abs(c1[1] - c2[1]) < 0.5
        
        if axis == 'horizontal':
            return h_aligned
        elif axis == 'vertical':
            return v_aligned
        else:
            return h_aligned or v_aligned
    
    @staticmethod
    def bboxes_overlap(obj1: ARCObject, obj2: ARCObject) -> bool:
        """Check if bounding boxes overlap."""
        y1_1, x1_1, y2_1, x2_1 = obj1.bbox
        y1_2, x1_2, y2_2, x2_2 = obj2.bbox
        
        return not (y2_1 <= y1_2 or y2_2 <= y1_1 or x2_1 <= x1_2 or x2_2 <= x1_1)
    
    @staticmethod
    def find_symmetry_axis(grid: Tensor) -> Optional[str]:
        """Detect if grid has horizontal or vertical symmetry."""
        H, W = grid.shape
        
        # Check horizontal symmetry (flip rows)
        h_sym = torch.all(grid == grid.flip(0))
        
        # Check vertical symmetry (flip columns)
        v_sym = torch.all(grid == grid.flip(1))
        
        if h_sym and v_sym:
            return "both"
        elif h_sym:
            return "horizontal"
        elif v_sym:
            return "vertical"
        else:
            return None


# ─────────────────────────────────────────────
# Shape Recognition
# ─────────────────────────────────────────────

class ShapeRecognizer:
    """Recognize common shapes in objects."""
    
    SHAPES = [
        'point',
        'line_horizontal',
        'line_vertical',
        'line_diagonal',
        'rectangle',
        'square',
        'l_shape',
        't_shape',
        'cross',
        'plus',
        'irregular',
    ]
    
    @staticmethod
    def classify(obj: ARCObject) -> str:
        """Classify object shape."""
        if obj.area == 1:
            return 'point'
        
        h, w = obj.height, obj.width
        
        # Line detection
        if h == 1:
            return 'line_horizontal'
        if w == 1:
            return 'line_vertical'
        
        # Rectangle/square detection
        if obj.is_rectangular:
            if h == w:
                return 'square'
            return 'rectangle'
        
        # Check fill ratio for more complex shapes
        fill_ratio = obj.area / (h * w)
        
        # L-shape (roughly 50% fill in bounding box)
        if 0.4 < fill_ratio < 0.6:
            return 'l_shape'
        
        # Plus/cross (roughly 33-60% fill, but more structured)
        if 0.3 < fill_ratio < 0.6:
            # Check for cross pattern (center row/col mostly filled)
            return 'plus'
        
        return 'irregular'
    
    @staticmethod
    def get_skeleton(obj: ARCObject) -> List[Tuple[int, int]]:
        """Get the skeleton (medial axis) of the object."""
        # Simple implementation: return cells that have <4 same-color neighbors
        skeleton = []
        cell_set = set(obj.cells)
        
        for r, c in obj.cells:
            neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
            neighbor_count = sum(1 for n in neighbors if n in cell_set)
            
            if neighbor_count < 3:  # Edge or endpoint
                skeleton.append((r, c))
        
        return skeleton if skeleton else obj.cells


# ─────────────────────────────────────────────
# Grid Reconstruction
# ─────────────────────────────────────────────

def reconstruct_grid(
    graph: ObjectGraph,
    background: int = 0,
) -> Tensor:
    """Reconstruct grid from object graph."""
    H, W = graph.grid_shape
    grid = torch.full((H, W), background, dtype=torch.long)
    
    for obj in graph.objects:
        for r, c in obj.cells:
            grid[r, c] = obj.color
    
    return grid


def apply_to_objects(
    graph: ObjectGraph,
    transform_fn,
    background: int = 0,
) -> Tensor:
    """Apply transformation to each object and reconstruct grid."""
    H, W = graph.grid_shape
    grid = torch.full((H, W), background, dtype=torch.long)
    
    for obj in graph.objects:
        transformed = transform_fn(obj)
        for r, c in transformed.cells:
            if 0 <= r < H and 0 <= c < W:
                grid[r, c] = transformed.color
    
    return grid


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("Object Detector Test")
    print("=" * 50)
    
    # Test grid
    grid = torch.tensor([
        [0, 0, 1, 1, 0, 0, 2],
        [0, 0, 1, 1, 0, 0, 2],
        [0, 0, 0, 0, 0, 0, 2],
        [3, 3, 3, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 0, 0],
    ])
    
    print("Input grid:")
    print(grid)
    
    detector = ObjectDetector()
    graph = detector.detect(grid)
    
    print(f"\nDetected {len(graph)} objects:")
    for obj in graph:
        print(f"  Object {obj.object_id}: color={obj.color}, area={obj.area}, "
              f"bbox={obj.bbox}, shape={ShapeRecognizer.classify(obj)}")
    
    print(f"\nLargest: {graph.get_largest().object_id if graph.get_largest() else 'none'}")
    print(f"Smallest: {graph.get_smallest().object_id if graph.get_smallest() else 'none'}")
    
    # Test reconstruction
    reconstructed = reconstruct_grid(graph, background=0)
    print(f"\nReconstruction matches: {torch.all(grid == reconstructed).item()}")
    
    # Test spatial analysis
    analyzer = SpatialAnalyzer()
    if len(graph) >= 2:
        obj1, obj2 = graph[0], graph[1]
        print(f"\nDistance between obj 0 and 1: {analyzer.distance(obj1, obj2):.2f}")
        print(f"Relative position: obj1 is {analyzer.relative_position(obj1, obj2)} of obj0")
