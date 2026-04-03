"""
Field-Based BLM: Parameter-minimal Byte-Level Model.

A novel architecture that replaces traditional weight matrices with:
- Resonance Field: Dynamic knowledge storage (0 parameters)
- Thermodynamic Settler: Energy-based inference (0 parameters)

Only ~100K trainable parameters (byte embedding), compared to 141M in FLUX-LM.
"""

from .byte_wave_encoder import ByteWaveEncoder, ByteWaveConfig
from .resonance_field import ResonanceField, FieldConfig
from .thermodynamic_settler import ThermodynamicSettler, ThermodynamicConfig
from .field_blm import FieldBLM, FieldBLMConfig

__all__ = [
    'ByteWaveEncoder',
    'ByteWaveConfig',
    'ResonanceField', 
    'FieldConfig',
    'ThermodynamicSettler',
    'ThermodynamicConfig',
    'FieldBLM',
    'FieldBLMConfig',
]
