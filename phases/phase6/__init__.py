"""
Phase 6: Three-Tier Memory System

Working memory (session), Episodic memory (facts), Semantic memory (deep knowledge).
Zero catastrophic forgetting by design — new attractors don't destroy old ones.

Key components:
- WorkingMemory: Current session context buffer
- EpisodicMemory: FAISS-indexed stored facts
- SemanticMemory: Deep field-based knowledge
- MemoryRouter: Routes queries to appropriate tier
- Consolidation: Sleep-like memory consolidation

Usage:
    from phases.phase6 import WorkingMemory, EpisodicMemory, MemoryRouter
    
    working = WorkingMemory(window_size=2048)
    episodic = EpisodicMemory(dim=432)
    episodic.store("The sky is blue", wave)
    results = episodic.recall(query_wave, k=5)
"""

try:
    from .working_memory import (
        WorkingMemory,
        SessionContext,
        push_to_working,
        get_recent_context,
    )
    from .episodic_memory import (
        EpisodicMemory,
        EpisodeEntry,
        store_episode,
        recall_episodes,
        rebuild_index,
    )
    from .semantic_memory import (
        SemanticMemory,
        store_semantic,
        query_semantic,
        merge_semantic,
    )
    from .memory_router import (
        MemoryRouter,
        route_query,
        MemoryTier,
        WORKING,
        EPISODIC,
        SEMANTIC,
    )
    from .consolidation import (
        Consolidator,
        consolidate_memories,
        prune_weak_memories,
        sleep_cycle,
    )
except ImportError:
    # Allow partial imports for standalone execution
    pass

__all__ = [
    # Working memory
    'WorkingMemory',
    'SessionContext',
    'push_to_working',
    'get_recent_context',
    
    # Episodic memory
    'EpisodicMemory',
    'EpisodeEntry',
    'store_episode',
    'recall_episodes',
    'rebuild_index',
    
    # Semantic memory
    'SemanticMemory',
    'store_semantic',
    'query_semantic',
    'merge_semantic',
    
    # Memory router
    'MemoryRouter',
    'route_query',
    'MemoryTier',
    'WORKING',
    'EPISODIC',
    'SEMANTIC',
    
    # Consolidation
    'Consolidator',
    'consolidate_memories',
    'prune_weak_memories',
    'sleep_cycle',
]
