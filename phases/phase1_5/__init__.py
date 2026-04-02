"""
Phase 1.5: Causal Wave Chaining (CWC)

Extends CSE with causal relationships between semantic waves.
Enables contradiction detection, temporal ordering, and implication tracking.

Key components:
- CausalEncoder: Encodes causal relationships into wave modifications
- Contradiction: Detects and handles contradictory information
- Implication: Tracks implication chains

Usage:
    from phases.phase1_5 import CausalEncoder, detect_contradiction
    
    encoder = CausalEncoder(wave_dim=432)
    causal_wave = encoder.encode_with_causality(wave1, wave2, relation='implies')
"""

try:
    from .causal_encoder import (
        CausalEncoder,
        CausalWave,
        encode_causal_relation,
    )
    from .causal_types import (
        CausalRelation,
        IMPLIES,
        CONTRADICTS,
        PRECEDES,
        CAUSES,
    )
    from .contradiction import (
        detect_contradiction,
        ContradictionResult,
        resolve_contradiction,
    )
    from .implication import (
        ImplicationChain,
        trace_implications,
        validate_implication,
    )
except ImportError:
    # Allow partial imports for standalone execution
    pass

__all__ = [
    # Causal encoder
    'CausalEncoder',
    'CausalWave',
    'encode_causal_relation',
    
    # Causal types
    'CausalRelation',
    'IMPLIES',
    'CONTRADICTS',
    'PRECEDES',
    'CAUSES',
    
    # Contradiction handling
    'detect_contradiction',
    'ContradictionResult',
    'resolve_contradiction',
    
    # Implication chains
    'ImplicationChain',
    'trace_implications',
    'validate_implication',
]
