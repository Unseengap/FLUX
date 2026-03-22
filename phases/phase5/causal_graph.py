"""
CausalGraph: Storage and tracing of causal arrows between nodes.

Every conclusion in FLUX stores WHY it was reached. This enables:
- Genuine causal reasoning (not just correlation)
- Tracing why a conclusion was reached
- Invalidating conclusions when their cause is disproven

The graph is a directed acyclic structure where:
- Nodes = CGN node IDs
- Edges = causal arrows with evidence weights
- Invalidation propagates forward (disprove cause → invalidate conclusion)
"""

import torch
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field


@dataclass
class CausalEdge:
    """A directed causal relationship between two nodes."""
    source: int
    target: int
    weight: float = 1.0
    reason: str = ""
    timestamp: int = 0
    invalidated: bool = False


@dataclass
class CausalTrace:
    """Complete trace of why a conclusion was reached."""
    target: int
    chain: List[int]
    edges: List[CausalEdge]
    total_weight: float
    has_conflict: bool
    conflicting_paths: List[List[int]]


class CausalGraph:
    """
    Stores and traces causal arrows between CGN nodes.
    Enables tracing 'why' a conclusion was reached and
    propagating invalidation when causes are disproven.
    """

    def __init__(self):
        self.arrows: Dict[int, List[int]] = {}  # source_id -> list of target_ids
        self.evidence: Dict[Tuple[int, int], CausalEdge] = {}
        self._step_counter: int = 0

    def add_arrow(
        self,
        source: int,
        target: int,
        weight: float = 1.0,
        reason: str = "",
    ):
        """
        Add a causal arrow: source CAUSES target.

        Args:
            source: ID of the cause node
            target: ID of the effect node
            weight: evidence strength (>0 = supports, <0 = contradicts)
            reason: human-readable explanation
        """
        if source not in self.arrows:
            self.arrows[source] = []
        if target not in self.arrows[source]:
            self.arrows[source].append(target)

        self._step_counter += 1
        self.evidence[(source, target)] = CausalEdge(
            source=source,
            target=target,
            weight=weight,
            reason=reason,
            timestamp=self._step_counter,
        )

    def get_edge(self, source: int, target: int) -> Optional[CausalEdge]:
        """Get the causal edge between source and target."""
        return self.evidence.get((source, target))

    def get_sources(self, target: int) -> List[int]:
        """Find all source nodes that cause a given target."""
        sources = []
        for s, targets in self.arrows.items():
            if target in targets:
                sources.append(s)
        return sources

    def get_targets(self, source: int) -> List[int]:
        """Find all target nodes caused by a given source."""
        return self.arrows.get(source, [])

    def trace_cause(self, target: int, depth: int = 5) -> CausalTrace:
        """
        Trace back the causal chain for a given target.
        Follows the strongest evidence path backward.

        Args:
            target: node ID to trace causes of
            depth:  maximum trace depth

        Returns:
            CausalTrace with full chain and conflict detection
        """
        chain = [target]
        edges = []
        current = target

        for _ in range(depth):
            sources = self.get_sources(current)
            if not sources:
                break
            # Pick strongest source
            best_source = max(
                sources,
                key=lambda s: abs(self.evidence.get((s, current), CausalEdge(s, current, 0.0)).weight)
            )
            edge = self.evidence.get((best_source, current))
            if edge:
                edges.insert(0, edge)
            chain.insert(0, best_source)
            current = best_source

        # Detect conflicts: multiple paths to same target with opposing weights
        all_paths = self._find_all_paths_to(target, depth)
        conflicting = []
        for path in all_paths:
            path_weight = self._path_weight(path)
            if path_weight < 0:
                conflicting.append(path)

        total_weight = sum(e.weight for e in edges)

        return CausalTrace(
            target=target,
            chain=[n for n in chain],
            edges=edges,
            total_weight=total_weight,
            has_conflict=len(conflicting) > 0,
            conflicting_paths=conflicting,
        )

    def _find_all_paths_to(
        self, target: int, max_depth: int, _visited: Optional[Set[int]] = None
    ) -> List[List[int]]:
        """Find all paths leading to target (BFS with depth limit)."""
        if _visited is None:
            _visited = set()

        paths = []
        sources = self.get_sources(target)

        for source in sources:
            if source in _visited or max_depth <= 0:
                paths.append([source, target])
                continue
            _visited.add(source)
            sub_paths = self._find_all_paths_to(source, max_depth - 1, _visited)
            _visited.discard(source)
            if sub_paths:
                for sp in sub_paths:
                    paths.append(sp + [target])
            else:
                paths.append([source, target])

        return paths

    def _path_weight(self, path: List[int]) -> float:
        """Compute cumulative evidence weight along a path."""
        total = 0.0
        for i in range(len(path) - 1):
            edge = self.evidence.get((path[i], path[i + 1]))
            if edge:
                total += edge.weight
        return total

    def invalidate_cause(self, source: int) -> List[int]:
        """
        Invalidate a cause node. All conclusions that depend on it
        are marked as invalidated (forward propagation).

        Args:
            source: the disproven cause node ID

        Returns:
            List of affected target node IDs
        """
        affected = []
        if source in self.arrows:
            for target in self.arrows[source]:
                edge = self.evidence.get((source, target))
                if edge:
                    edge.weight = -abs(edge.weight)  # Flip to negative
                    edge.invalidated = True
                affected.append(target)

                # Propagate invalidation to downstream nodes
                downstream = self._propagate_invalidation(target)
                affected.extend(downstream)

        return list(set(affected))

    def _propagate_invalidation(self, node: int) -> List[int]:
        """
        Check if a node has lost all valid causal support.
        If so, propagate invalidation to its targets.
        """
        affected = []
        sources = self.get_sources(node)
        valid_sources = [
            s for s in sources
            if not self.evidence.get((s, node), CausalEdge(s, node)).invalidated
        ]

        # If no valid causes remain, propagate invalidation forward
        if len(valid_sources) == 0 and node in self.arrows:
            for target in self.arrows[node]:
                edge = self.evidence.get((node, target))
                if edge and not edge.invalidated:
                    edge.weight *= 0.5  # Weaken (partial invalidation)
                    affected.append(target)

        return affected

    def get_conflict_summary(self, target: int) -> Dict:
        """
        Summary of all evidence for/against a target node.

        Returns:
            Dict with supporting/opposing sources and net weight
        """
        sources = self.get_sources(target)
        supporting = []
        opposing = []

        for s in sources:
            edge = self.evidence.get((s, target))
            if edge:
                if edge.weight > 0 and not edge.invalidated:
                    supporting.append((s, edge.weight, edge.reason))
                else:
                    opposing.append((s, edge.weight, edge.reason))

        net_weight = sum(w for _, w, _ in supporting) + sum(w for _, w, _ in opposing)

        return {
            'target': target,
            'supporting': supporting,
            'opposing': opposing,
            'net_weight': net_weight,
            'conclusion': 'supported' if net_weight > 0 else 'contested' if net_weight == 0 else 'contradicted',
        }

    def node_count(self) -> int:
        """Number of unique nodes in the graph."""
        nodes = set()
        for s, targets in self.arrows.items():
            nodes.add(s)
            nodes.update(targets)
        return len(nodes)

    def edge_count(self) -> int:
        """Number of causal edges."""
        return len(self.evidence)

    def save_state(self) -> Dict:
        """Save graph state for checkpointing."""
        return {
            'arrows': dict(self.arrows),
            'evidence': {
                f"{k[0]}_{k[1]}": {
                    'source': e.source,
                    'target': e.target,
                    'weight': e.weight,
                    'reason': e.reason,
                    'timestamp': e.timestamp,
                    'invalidated': e.invalidated,
                }
                for k, e in self.evidence.items()
            },
            'step_counter': self._step_counter,
        }

    def load_state(self, state: Dict):
        """Load graph state from checkpoint."""
        self.arrows = {int(k): v for k, v in state['arrows'].items()}
        self.evidence = {}
        for key_str, e_dict in state['evidence'].items():
            edge = CausalEdge(**e_dict)
            self.evidence[(edge.source, edge.target)] = edge
        self._step_counter = state.get('step_counter', 0)
