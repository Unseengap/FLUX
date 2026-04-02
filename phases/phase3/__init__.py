"""
Phase 3: Gravitational Relevance (GR)

O(log n) relevance calculation using mass-based attraction.
Replaces O(n²) attention with physics-inspired gravity mechanics.

Key components:
- GravitationalRelevance: Main relevance calculation engine
- MassTracker: Tracks evidence mass for concepts
- SpatialIndex: KD-tree for O(log n) nearest-neighbor queries
- NegativeMass: Handles contradictions via repulsion

Usage:
    from phases.phase3 import GravitationalRelevance, MassTracker
    
    gr = GravitationalRelevance(feature_dim=512)
    relevance = gr.compute_relevance(query_wave, field_state)
"""

try:
    from .gravity import (
        GravitationalRelevance,
        compute_gravitational_pull,
        apply_mass_update,
    )
    from .mass_tracker import (
        MassTracker,
        MassEntry,
        update_mass_with_evidence,
        decay_masses,
    )
    from .spatial_index import (
        SpatialIndex,
        build_spatial_index,
        query_nearest,
        query_radius,
    )
    from .negative_mass import (
        NegativeMass,
        apply_repulsion,
        detect_contradiction_mass,
    )
except ImportError:
    # Allow partial imports for standalone execution
    pass

__all__ = [
    # Gravity engine
    'GravitationalRelevance',
    'compute_gravitational_pull',
    'apply_mass_update',
    
    # Mass tracking
    'MassTracker',
    'MassEntry',
    'update_mass_with_evidence',
    'decay_masses',
    
    # Spatial indexing
    'SpatialIndex',
    'build_spatial_index',
    'query_nearest',
    'query_radius',
    
    # Negative mass (contradictions)
    'NegativeMass',
    'apply_repulsion',
    'detect_contradiction_mass',
]
