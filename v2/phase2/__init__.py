"""
v2/phase2 — Resonance Field with Decode Gate
Waves live in a field. Round-trip: wave → field → wave → text.
"""
from .field import ResonanceField, FieldLocation, FIELD_H, FIELD_W, FIELD_D, FIELD_FEATURES
from .attractor import AttractorCatalog, Attractor
from .wave_to_field import WaveToField
from .field_to_wave import FieldToWave

__all__ = [
    'ResonanceField', 'FieldLocation',
    'FIELD_H', 'FIELD_W', 'FIELD_D', 'FIELD_FEATURES',
    'AttractorCatalog', 'Attractor',
    'WaveToField', 'FieldToWave',
]
