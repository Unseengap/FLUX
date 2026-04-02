"""
Phase 1: Continuous Semantic Encoder (CSE)

Converts raw UTF-8 bytes into 432-dimensional semantic waves.
No tokenizer, no vocabulary — pure byte-level encoding.

Key components:
- ContinuousSemanticEncoder: Main encoder class
- SemanticWave: Output dataclass with wave decomposition
- wave_interference: Interference calculations for wave interactions

Usage:
    from phases.phase1 import ContinuousSemanticEncoder, SemanticWave
    
    cse = ContinuousSemanticEncoder(wave_dim=432)
    wave = cse.encode("Hello, world!")  # → SemanticWave[seq_len, 432]
"""

try:
    from .cse import ContinuousSemanticEncoder
    from .wave_types import (
        SemanticWave,
        WAVE_DIMS,
        TOTAL_WAVE_DIM,
        split_wave,
        merge_wave,
    )
    from .interference import (
        wave_interference,
        phase_alignment,
        constructive_interference,
        destructive_interference,
    )
except ImportError:
    # Allow partial imports for standalone execution
    pass

__all__ = [
    # Main encoder
    'ContinuousSemanticEncoder',
    
    # Wave types
    'SemanticWave',
    'WAVE_DIMS',
    'TOTAL_WAVE_DIM',
    'split_wave',
    'merge_wave',
    
    # Interference
    'wave_interference',
    'phase_alignment',
    'constructive_interference',
    'destructive_interference',
]
