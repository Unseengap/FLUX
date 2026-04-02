"""
Phase 5: Causal Geometry Nodes (CGN)

Neurons replaced by curved manifolds that encode causal relationships.
Every conclusion knows WHY it was reached, enabling invalidation.

Key components:
- CausalGeometryNode: Single CGN with curvature, mass, orientation
- CausalGraph: Graph of causal arrows between nodes
- Manifold: Riemannian manifold for geometric reasoning
- MultiTimescale: Fast/medium/slow CGN tiers

Usage:
    from phases.phase5 import CausalGeometryNode, CausalGraph
    
    cgn = CausalGeometryNode(feature_dim=512)
    cgn.fire(input_wave)  # Activate node
    graph = CausalGraph()
    graph.add_arrow(source=0, target=1, evidence=0.9)
"""

try:
    from .cgn import (
        CausalGeometryNode,
        CGNConfig,
        fire_cgn,
        update_curvature,
    )
    from .causal_graph import (
        CausalGraph,
        CausalArrow,
        add_arrow,
        trace_causality,
        invalidate_downstream,
    )
    from .manifold import (
        Manifold,
        compute_geodesic,
        parallel_transport,
        curvature_tensor,
    )
    from .multi_timescale import (
        MultiTimescaleCGN,
        FastCGN,
        MediumCGN,
        SlowCGN,
        synchronize_timescales,
    )
except ImportError:
    # Allow partial imports for standalone execution
    pass

__all__ = [
    # CGN core
    'CausalGeometryNode',
    'CGNConfig',
    'fire_cgn',
    'update_curvature',
    
    # Causal graph
    'CausalGraph',
    'CausalArrow',
    'add_arrow',
    'trace_causality',
    'invalidate_downstream',
    
    # Manifold geometry
    'Manifold',
    'compute_geodesic',
    'parallel_transport',
    'curvature_tensor',
    
    # Multi-timescale
    'MultiTimescaleCGN',
    'FastCGN',
    'MediumCGN',
    'SlowCGN',
    'synchronize_timescales',
]
