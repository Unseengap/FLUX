"""
AnalogicalMapper: The engine of generalization.

"A is to B as C is to D"

Standard models fail at out-of-distribution reasoning.
FLUX solves this by looking at the SHAPE of causal wave chains.

Two modes:
1. find_analogical_target(A, B, C) → D
   delta = B - A
   D = nearest_attractor(C + delta)

2. map_causal_chain(novel_problem) → known_chain
   Compute shape signature of novel problem's causal structure.
   Find the stored chain with highest shape similarity.
   Zero word overlap is fine — geometry is the signal.
"""

import sys
import torch
import torch.nn.functional as F
from torch import Tensor
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1_5'))

from implication import ImplicationChainStore
from causal_types import CausalArrow


class AnalogicalMapper:
    """
    Performs analogical reasoning over the SparseResonanceField.

    Usage:
        mapper = AnalogicalMapper(field, impl_store)
        result = mapper.find_analogical_target(
            vec_king, vec_man, vec_queen
        )
        # Returns vector nearest to "woman"
    """

    def __init__(
        self,
        field:      'SparseResonanceField',
        impl_store: ImplicationChainStore,
        device:     str = 'cuda',
    ):
        self.field      = field
        self.impl_store = impl_store
        self.device     = device

    # ─────────────────────────────────────────
    # Core analogy: A:B :: C:?
    # ─────────────────────────────────────────

    def find_analogical_target(
        self,
        source_a:   Tensor,
        source_b:   Tensor,
        target_c:   Tensor,
        search_k:   int = 16,
    ) -> Tuple[Tensor, float, Tuple[int,int,int]]:
        """
        Given A→B relationship, apply same delta to C to find D.

        Args:
            source_a:  [dim] source concept A
            source_b:  [dim] target concept B
            target_c:  [dim] query concept C
            search_k:  how many neighbors to consider
        Returns:
            (best_feature, similarity, location)
        """
        # Compute the relationship delta in wave space
        # Work on the 432-dim base wave if inputs are 432-dim
        # or on the 608-dim causal wave if they are
        delta = source_b - source_a

        # Project C + delta into field
        projected = target_c + delta

        # Find nearest field attractor to projected point
        # Use wave_to_feature to get the feature-space target
        with torch.no_grad():
            if projected.shape[0] == self.field.wave_dim:
                # 432-dim: pass directly
                target_feature = self.field.wave_to_feature(
                    projected.to(self.device)
                )
                h, w, d = self.field.wave_to_field_coords(projected.to(self.device))
            else:
                # 608-dim causal wave: use the base 432 dims
                base = projected[:self.field.wave_dim]
                target_feature = self.field.wave_to_feature(
                    base.to(self.device)
                )
                h, w, d = self.field.wave_to_field_coords(base.to(self.device))

        feats, sims, locs = self.field.registry.batch_query_neighborhood(
            h, w, d,
            radius         = 24,
            k              = search_k,
            target_feature = target_feature,
        )

        if feats.shape[0] == 0 or sims.max().item() < 0.01:
            return target_feature, 0.0, (h, w, d)

        best_idx  = sims.argmax().item()
        best_feat = feats[best_idx]
        best_sim  = sims[best_idx].item()
        best_loc  = locs[best_idx] if best_idx < len(locs) else (h, w, d)

        return best_feat, best_sim, best_loc

    def analogy_text(
        self,
        cse,
        cwc,
        word_a: str,
        word_b: str,
        word_c: str,
        label_index: Dict[str, Tensor] = None,
    ) -> Tuple[str, float]:
        """
        High-level text analogy: word_a : word_b :: word_c : ?

        Args:
            cse:         Phase 1 CSE
            cwc:         Phase 1.5 CWC
            word_a, word_b, word_c: input words
            label_index: dict mapping text → vector for readable output
        Returns:
            (closest_label, similarity)
        """
        def encode(text):
            with torch.no_grad():
                wave = cse.encode(text)
                cw   = cwc.forward(wave)
            return cw.full.mean(dim=0)

        va = encode(word_a)
        vb = encode(word_b)
        vc = encode(word_c)

        best_feat, best_sim, best_loc = self.find_analogical_target(va, vb, vc)

        if label_index:
            # Find closest label.
            # label_index vectors are 608-dim (causal wave).
            # best_feat is 512-dim (field feature space).
            # Project label vectors to feature space before comparing.
            best_label = f"[field location {best_loc}]"
            best_label_sim = best_sim
            bf_n = F.normalize(best_feat.float().cpu(), dim=-1)  # [512]
            with torch.no_grad():
                for label, lv in label_index.items():
                    # lv is 608-dim causal wave — take base 432 dims and project
                    lv_base = lv[:self.field.wave_dim].to(
                        next(self.field.parameters()).device
                    )
                    lv_feat = self.field.wave_to_feature(lv_base).detach().cpu()
                    lv_n = F.normalize(lv_feat.float(), dim=-1)  # [512]
                    s = F.cosine_similarity(
                        lv_n.unsqueeze(0), bf_n.unsqueeze(0)
                    ).item()
                    if s > best_label_sim:
                        best_label_sim = s
                        best_label     = label
            return best_label, best_label_sim

        return f"field@{best_loc}", best_sim

    # ─────────────────────────────────────────
    # Chain shape matching
    # ─────────────────────────────────────────

    def _compute_chain_signature(
        self,
        chain: List[Tuple[Tensor, float]],
    ) -> Tensor:
        """
        Compute a geometric shape signature for a causal chain.

        The signature captures the DIRECTION of each step,
        not the content. This is why two chains about completely
        different topics can have identical signatures if they
        follow the same logical structure.

        Signature = sequence of normalized deltas between steps.
        """
        if len(chain) < 2:
            if chain:
                return F.normalize(chain[0][0].float().cpu(), dim=-1)
            return torch.zeros(608)

        deltas = []
        for i in range(len(chain) - 1):
            v1 = chain[i][0].float().cpu()
            v2 = chain[i+1][0].float().cpu()
            delta = F.normalize(v2 - v1, dim=-1)
            deltas.append(delta)

        # Mean of all step directions
        return torch.stack(deltas).mean(dim=0)

    def _get_novel_chain_signature(
        self,
        query_vec: Tensor,
        depth:     int = 3,
    ) -> Tensor:
        """
        Build a chain from the query via implication propagation
        and compute its signature.
        """
        chains = self.impl_store.chain_propagate(
            query_vec, depth=depth, k_per_step=3
        )
        if not chains:
            return F.normalize(query_vec.float().cpu(), dim=-1)

        # Use the longest chain
        longest = max(chains, key=len)
        return self._compute_chain_signature(longest)

    def map_causal_chain(
        self,
        novel_problem_wave: Tensor,
        k:                  int = 1,
        depth:              int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Find known reasoning chains with the highest structural
        similarity to the novel problem — regardless of content.

        Args:
            novel_problem_wave: [dim] wave vector for novel problem
            k:                  number of best matches to return
            depth:              chain propagation depth
        Returns:
            List of dicts: [{'chain_signature', 'similarity', 'chain_vecs'}]
        """
        novel_sig = self._get_novel_chain_signature(
            novel_problem_wave, depth=depth
        )

        # Collect all known chains from impl_store
        known_chains = self.impl_store.chain_propagate(
            novel_problem_wave, depth=depth, k_per_step=5
        )

        # Build signatures for all known chains
        scored = []
        for chain in known_chains:
            sig = self._compute_chain_signature(chain)
            sim = F.cosine_similarity(
                novel_sig.unsqueeze(0),
                sig.unsqueeze(0),
            ).item()
            scored.append({
                'chain_signature': sig,
                'similarity':      sim,
                'chain_vecs':      chain,
            })

        # Sort by similarity descending
        scored.sort(key=lambda x: x['similarity'], reverse=True)
        return scored[:k]

    # ─────────────────────────────────────────
    # Batch analogy evaluation
    # ─────────────────────────────────────────

    def evaluate_analogies(
        self,
        cse,
        cwc,
        test_pairs: List[Tuple[str,str,str,str]],
        label_index: Dict[str, Tensor],
    ) -> Dict:
        """
        Evaluate a list of (A, B, C, expected_D) analogy tuples.

        Returns accuracy and per-example results.
        """
        correct = 0
        results = []

        for word_a, word_b, word_c, expected_d in test_pairs:
            predicted, sim = self.analogy_text(
                cse, cwc, word_a, word_b, word_c, label_index
            )
            hit = (
                predicted.lower() == expected_d.lower() or
                expected_d.lower() in predicted.lower()
            )
            if hit:
                correct += 1
            results.append({
                'a': word_a, 'b': word_b, 'c': word_c,
                'expected': expected_d,
                'predicted': predicted,
                'similarity': sim,
                'correct': hit,
            })

        return {
            'accuracy': correct / max(len(test_pairs), 1),
            'correct': correct,
            'total': len(test_pairs),
            'results': results,
        }


# ─────────────────────────────────────────────
# Standard analogy test set
# ─────────────────────────────────────────────

ANALOGY_TEST_PAIRS = [
    # (A, B, C, expected_D)
    ('man',    'king',    'woman',  'queen'),
    ('dog',    'puppy',   'cat',    'kitten'),
    ('france', 'paris',   'japan',  'tokyo'),
    ('hot',    'cold',    'fast',   'slow'),
    ('big',    'small',   'tall',   'short'),
    ('run',    'running', 'swim',   'swimming'),
    ('father', 'son',     'mother', 'daughter'),
]