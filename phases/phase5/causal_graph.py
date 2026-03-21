import torch
from typing import Dict, List, Optional

class CausalGraph:
    """
    Stores and traces causal arrows between nodes.
    Enabled tracing 'why' a conclusion was reached.
    """
    def __init__(self):
        self.arrows: Dict[int, List[int]] = {}  # source_id -> list of target_ids
        self.evidence: Dict[tuple, float] = {}  # (source, target) -> weight

    def add_arrow(self, source: int, target: int, weight: float = 1.0):
        if source not in self.arrows:
            self.arrows[source] = []
        self.arrows[source].append(target)
        self.evidence[(source, target)] = weight

    def trace_cause(self, target: int, depth: int = 5) -> List[int]:
        """
        Traces back the causal chain for a given target.
        """
        chain = [target]
        current = target
        for _ in range(depth):
            # Find sources that point to current
            sources = [s for s, targets in self.arrows.items() if current in targets]
            if not sources:
                break
            # Pick strongest source
            best_source = max(sources, key=lambda s: self.evidence.get((s, current), 0.0))
            chain.insert(0, best_source)
            current = best_source
        return chain

    def invalidate_cause(self, source: int):
        """
        Invalidates a cause, potentially affecting dependent conclusions.
        """
        if source in self.arrows:
            for target in self.arrows[source]:
                self.evidence[(source, target)] = -1.0  # Repulsion / invalidation
