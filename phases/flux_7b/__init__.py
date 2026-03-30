"""
FLUX-7B: Physics-Native Language Model

7B parameters, but only 200M trainable.

Architecture:
    ResonanceField7B (6.5B)  — Knowledge storage via injection (no gradients)
    PhysicsStack (300M)      — CSE + physics components (frozen)
    EmissionHead (200M)      — Spelling/generation (trainable)

Key Innovation:
    Knowledge is INJECTED into the field, not TRAINED.
    Only the emission head (how to spell) needs gradient training.

Training Time:
    - Knowledge injection: ~12 hours (vs weeks for LLM)
    - Emission training: ~2-3 days (vs weeks for LLM)
    - Add new knowledge: ~seconds (vs fine-tuning for LLM)

Usage:
    from flux_7b import FLUX7B, create_flux_7b_full
    
    # Create system
    system = create_flux_7b_full()
    
    # Inject knowledge (no gradients!)
    system.inject_document("Paris is the capital of France...")
    
    # Generate
    output = system.generate("What is the capital of France?")
"""

from .resonance_field_7b import (
    ResonanceField7B,
    FieldConfig,
    create_field_7b,
)

from .emission_head import (
    EmissionHead,
    EmissionConfig,
    create_emission_head_200m,
    create_emission_head_50m,
)

from .flux_7b_model import (
    FLUX7B,
    FLUX7BConfig,
    PhysicsStack,
    create_flux_7b_full,
    create_flux_7b_test,
)

__all__ = [
    # Field
    'ResonanceField7B',
    'FieldConfig',
    'create_field_7b',
    
    # Emission
    'EmissionHead',
    'EmissionConfig',
    'create_emission_head_200m',
    'create_emission_head_50m',
    
    # Full system
    'FLUX7B',
    'FLUX7BConfig',
    'PhysicsStack',
    'create_flux_7b_full',
    'create_flux_7b_test',
]
