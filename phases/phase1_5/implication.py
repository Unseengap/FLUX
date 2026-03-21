"""
ImplicationChain: Stores and propagates learned causal arrows.

An implication is: knowing A makes B more likely.
A chain is: A → B → C → D — a sequence of causally linked concepts.

These chains are how FLUX reasons transitively:
    "I know A. A implies B. B implies C. Therefore C is likely."

Unlike classical logic, implications are PROBABILISTIC and WEIGHTED.
Strong evidence makes the arrow stronger.
Contradicting evidence weakens or reverses the arrow.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from causal_types import CausalArrow
from typing import List, Tuple
import numpy as np


class ImplicationChainStore:
    """
    Stores all known causal arrows and enables forward propagation.

    At training time: arrows are added when sequential patterns are seen.
    At inference time: given wave A, retrieve all B that A implies,
                       weighted by arrow strength and evidence count.
    """

    def __init__(
        self,
        causal_dim:   int   = 608,
        max_arrows:   int   = 50000,
        min_strength: float = 0.3,
        device:       str   = 'cuda',
    ):
        self.causal_dim   = causal_dim
        self.max_arrows   = max_arrows
        self.min_strength = min_strength
        self.device       = device
        self.arrows: List[CausalArrow] = []

    def add_arrow(
        self,
        source: Tensor,
        target: Tensor,
        strength: float,
        arrow_type: str = 'temporal',
    ):
        """
        Register a causal arrow: source → target.

        Args:
            source:     [causal_dim] source CausalWave mean vector
            target:     [causal_dim] target CausalWave mean vector
            strength:   how reliable this implication is (0 to 1)
            arrow_type: 'temporal', 'logical', 'semantic', 'contradiction'
        """
        if strength < self.min_strength:
            return

        # Check for existing arrow — reinforce if found
        for arrow in self.arrows:
            src = F.normalize(arrow.source_vector.float(), dim=-1)
            new = F.normalize(source.cpu().float(), dim=-1)
            if F.cosine_similarity(src.unsqueeze(0), new.unsqueeze(0)).item() > 0.9:
                tgt_src = F.normalize(arrow.target_vector.float(), dim=-1)
                tgt_new = F.normalize(target.cpu().float(), dim=-1)
                if F.cosine_similarity(tgt_src.unsqueeze(0), tgt_new.unsqueeze(0)).item() > 0.9:
                    arrow.evidence_count += 1
                    arrow.strength = min(1.0, arrow.strength + 0.01)
                    return

        arrow = CausalArrow(
            source_vector  = source.detach().cpu(),
            target_vector  = target.detach().cpu(),
            strength       = strength,
            evidence_count = 1,
            arrow_type     = arrow_type,
        )
        self.arrows.append(arrow)

        if len(self.arrows) > self.max_arrows:
            self._prune()

    def forward_propagate(
        self,
        query: Tensor,
        k: int = 10,
        min_strength: float = 0.3,
    ) -> List[Tuple[Tensor, float]]:
        """
        Given a query wave, find all waves that this query implies.

        Returns:
            List of (target_vector, effective_strength) sorted by strength desc
        """
        if not self.arrows:
            return []

        q = F.normalize(query.cpu().float(), dim=-1)
        results = []

        for arrow in self.arrows:
            src = F.normalize(arrow.source_vector.float(), dim=-1)
            sim = F.cosine_similarity(q.unsqueeze(0), src.unsqueeze(0)).item()
            if sim > 0.5:
                evidence_weight    = min(1.0, np.log1p(arrow.evidence_count) / 5.0)
                effective_strength = arrow.strength * sim * evidence_weight
                if effective_strength > min_strength:
                    results.append((arrow.target_vector, effective_strength))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def chain_propagate(
        self,
        query: Tensor,
        depth: int = 3,
        k_per_step: int = 5,
    ) -> List[List[Tuple[Tensor, float]]]:
        """
        Multi-step chain propagation: A → B → C → D.

        This is how FLUX reasons transitively — following arrows
        that were never seen together in training.

        Returns:
            List of chains, each chain is list of (vector, strength) tuples
        """
        step_0    = [(query, 1.0)]
        all_chains = [[s] for s in step_0]

        for _ in range(depth):
            new_chains = []
            for chain in all_chains:
                last_vec, last_str = chain[-1]
                next_steps = self.forward_propagate(
                    last_vec, k=k_per_step, min_strength=0.2
                )
                for next_vec, next_str in next_steps:
                    compound_str = last_str * next_str
                    if compound_str > 0.05:
                        new_chains.append(chain + [(next_vec, compound_str)])
            if new_chains:
                all_chains = new_chains
            else:
                break

        return [chain[1:] for chain in all_chains if len(chain) > 1]

    def _prune(self, keep_fraction: float = 0.8):
        """Keep highest-evidence, highest-strength arrows."""
        self.arrows.sort(
            key=lambda a: a.evidence_count * a.strength,
            reverse=True,
        )
        keep_n = int(self.max_arrows * keep_fraction)
        self.arrows = self.arrows[:keep_n]

    def stats(self) -> dict:
        if not self.arrows:
            return {'count': 0}
        by_type = {}
        for a in self.arrows:
            by_type[a.arrow_type] = by_type.get(a.arrow_type, 0) + 1
        return {
            'count':         len(self.arrows),
            'by_type':       by_type,
            'mean_strength': float(np.mean([a.strength for a in self.arrows])),
            'max_strength':  float(max(a.strength for a in self.arrows)),
            'mean_evidence': float(np.mean([a.evidence_count for a in self.arrows])),
        }

    def save(self) -> dict:
        return {
            'arrows': [
                {
                    'source_vector':  a.source_vector,
                    'target_vector':  a.target_vector,
                    'strength':       a.strength,
                    'evidence_count': a.evidence_count,
                    'arrow_type':     a.arrow_type,
                }
                for a in self.arrows
            ],
            'config': {
                'causal_dim':   self.causal_dim,
                'max_arrows':   self.max_arrows,
                'min_strength': self.min_strength,
            }
        }

    def load(self, state: dict):
        self.arrows = [
            CausalArrow(
                source_vector  = a['source_vector'],
                target_vector  = a['target_vector'],
                strength       = a['strength'],
                evidence_count = a['evidence_count'],
                arrow_type     = a['arrow_type'],
            )
            for a in state['arrows']
        ]