"""
Phase 8: Wave Decoder

DEPRECATED: The byte decoder has been superseded by embedded instruct model.
This module is kept for historical reference only.

The original wave decoder was a GRU-based byte-level text generator that
demonstrated FLUX could generate text without external models. The embedded
Qwen2.5-1.5B-Instruct model now provides far superior text generation.

See:
- models.instruct in Flux-Apex-V1.flx for text generation
- FLUX_CONSOLIDATION_ROADMAP.md for deprecation details

Legacy Usage (not recommended):
    from phases.phase8 import WaveDecoder  # DEPRECATED
"""

import warnings

def _deprecated_warning():
    warnings.warn(
        "Phase 8 WaveDecoder is DEPRECATED. Use the embedded instruct model instead. "
        "See FLUX_CONSOLIDATION_ROADMAP.md for details.",
        DeprecationWarning,
        stacklevel=3
    )

try:
    from .wave_decoder import WaveDecoder
    from .flux_large import FLUXLarge  # Also deprecated
    
    # Emit deprecation warning on import
    _deprecated_warning()
except ImportError:
    pass

__all__ = [
    'WaveDecoder',      # DEPRECATED
    'FLUXLarge',        # DEPRECATED
]
