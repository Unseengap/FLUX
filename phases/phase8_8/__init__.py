"""
Phase 8.8: Grid Adapters & Spatial Memory

Adapters for ARC grid encoding/decoding and spatial memory for exploration.

Key components:
- GridToWave: Encode ARC grids to 432D waves
- WaveToGrid: Decode waves back to ARC grids
- SpatialMemory: Curiosity-driven exploration tracking
- TextToWave: CSE wrapper for text encoding

Usage:
    from phases.phase8_8 import GridToWave, WaveToGrid, load_adapters_from_flx
    
    grid_to_wave = GridToWave(wave_dim=432)
    wave = grid_to_wave(grid)  # [H, W] → [432]
    
    wave_to_grid = WaveToGrid(wave_dim=432, max_size=30)
    output_grid = wave_to_grid(wave)  # [432] → [H, W]
"""

try:
    from .grid_adapters import (
        GridToWave,
        WaveToGrid,
        create_grid_adapters,
    )
    from .flx_loader import (
        load_adapters_from_flx,
        save_adapters_to_flx,
        get_adapter_config,
    )
    from .wave_to_x import (
        WaveToX,
        MultiModalDecoder,
    )
    from .text_adapters import (
        TextToWave,  # CSE wrapper — KEEP
        # WaveToText — DEPRECATED, use instruct model
    )
except ImportError:
    # Allow partial imports for standalone execution
    pass

__all__ = [
    # Grid adapters
    'GridToWave',
    'WaveToGrid',
    'create_grid_adapters',
    
    # FLX loading
    'load_adapters_from_flx',
    'save_adapters_to_flx',
    'get_adapter_config',
    
    # Multi-modal
    'WaveToX',
    'MultiModalDecoder',
    
    # Text (encoding only)
    'TextToWave',
]
