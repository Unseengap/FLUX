"""
Memory Router — Routes queries between the three memory tiers.

Every forward pass follows this routing logic:
1. Working memory processes current input (immediate context)
2. GR queries episodic store for relevant past facts
3. Retrieved facts are injected into the working field region
4. Semantic field provides background knowledge (always on)

The router coordinates all three tiers and merges their outputs
into a unified memory response.
"""

import sys
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from working_memory import WorkingMemory
from episodic_memory import EpisodicMemory
from semantic_memory import SemanticMemory


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

ROUTER_CONFIG = {
    'working_weight': 0.4,
    'episodic_weight': 0.35,
    'semantic_weight': 0.25,
    'episodic_k': 5,
    'working_k': 10,
}


class MemoryRouter(nn.Module):
    """
    Routes queries across three memory tiers and merges outputs.

    For every query wave:
    1. Retrieves relevant context from working memory (session)
    2. Searches episodic memory for matching facts (permanent)
    3. Reads semantic field state (deep knowledge, always on)
    4. Merges with learned tier weights

    Args:
        working:  WorkingMemory instance
        episodic: EpisodicMemory instance
        semantic: SemanticMemory instance
        wave_dim: full CSE wave dimension (432)
        feature_dim: compressed retrieval dimension (256)
    """

    def __init__(
        self,
        working: WorkingMemory,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        wave_dim: int = 432,
        feature_dim: int = 256,
    ):
        super().__init__()
        self.working = working
        self.episodic = episodic
        self.semantic = semantic
        self.wave_dim = wave_dim
        self.feature_dim = feature_dim

        # Tier weighting (learnable)
        self.tier_weights = nn.Parameter(torch.tensor([0.4, 0.35, 0.25]))

        # Compression for episodic queries (wave_dim → feature_dim)
        self.query_compress = nn.Linear(wave_dim, feature_dim)

        # Merge projection: combines tier outputs → wave_dim
        self.merge_proj = nn.Linear(wave_dim * 2, wave_dim)

    def route_query(
        self,
        query_wave: Tensor,
        episodic_k: int = 5,
        working_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Route a query through all three memory tiers.

        Args:
            query_wave:  [wave_dim] query semantic wave
            episodic_k:  number of episodic results
            working_k:   number of working memory results

        Returns:
            Dict with:
                'working_context':  [k, wave_dim] recent context
                'episodic_facts':   list of (EpisodicEntry, score)
                'semantic_energy':  float — field energy level
                'merged':           [wave_dim] merged memory output
                'tier_weights':     [3] normalized weights
        """
        device = query_wave.device

        # Normalize tier weights
        weights = torch.softmax(self.tier_weights, dim=0)

        # ── Tier 1: Working Memory ──
        wm_waves, wm_scores = self.working.query(query_wave, top_k=working_k)

        # ── Tier 2: Episodic Memory ──
        with torch.no_grad():
            q_compressed = self.query_compress(query_wave.unsqueeze(0)).squeeze(0)
        episodic_results = self.episodic.search(q_compressed, k=episodic_k)

        # ── Tier 3: Semantic Memory (always on) ──
        sem_energy = self.semantic.get_field_energy()

        # ── Merge tier outputs ──
        # Compute working memory contribution
        if wm_waves.numel() > 0:
            wm_contribution = (wm_waves * torch.softmax(wm_scores, dim=0).unsqueeze(-1)).sum(dim=0)
        else:
            wm_contribution = torch.zeros(self.wave_dim)

        # Compute episodic contribution (from retrieved vectors)
        if episodic_results:
            ep_waves = []
            ep_scores_list = []
            for entry, score in episodic_results:
                vec = self.episodic.get_vector(entry.entry_id)
                if vec is not None:
                    # Pad/project to wave_dim
                    t = torch.from_numpy(vec).float()
                    if t.shape[0] < self.wave_dim:
                        t = torch.nn.functional.pad(t, (0, self.wave_dim - t.shape[0]))
                    else:
                        t = t[:self.wave_dim]
                    ep_waves.append(t)
                    ep_scores_list.append(score)

            if ep_waves:
                ep_stack = torch.stack(ep_waves)
                ep_s = torch.tensor(ep_scores_list)
                ep_w = torch.softmax(ep_s, dim=0).unsqueeze(-1)
                ep_contribution = (ep_stack * ep_w).sum(dim=0)
            else:
                ep_contribution = torch.zeros(self.wave_dim)
        else:
            ep_contribution = torch.zeros(self.wave_dim)

        # Weighted merge
        merged = (
            weights[0] * wm_contribution
            + weights[1] * ep_contribution
            + weights[2] * query_wave.cpu()
        )

        return {
            'working_context': wm_waves,
            'episodic_facts': episodic_results,
            'semantic_energy': sem_energy,
            'merged': merged,
            'tier_weights': weights.detach(),
        }

    def write_and_route(
        self,
        wave: Tensor,
        fact: str = "",
        episodic_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Write a wave to working + episodic memory, then route a query.

        This is the standard "process new input" path:
        1. Add to working memory (session context)
        2. Write to episodic memory (permanent)
        3. Route a query to find related existing knowledge

        Args:
            wave:       [wave_dim] incoming semantic wave
            fact:       human-readable fact string
            episodic_k: number of episodic search results

        Returns:
            Same as route_query
        """
        # Write to working memory
        self.working.add_perturbation(wave)

        # Write to episodic memory
        with torch.no_grad():
            vec = self.working.compress(wave.unsqueeze(0)).squeeze(0)
        self.episodic.write(vec, fact=fact)

        # Route the query
        return self.route_query(wave, episodic_k=episodic_k)

    def get_stats(self) -> Dict[str, Any]:
        """Aggregate statistics from all tiers."""
        return {
            'working': self.working.get_stats(),
            'episodic': self.episodic.get_stats(),
            'semantic': self.semantic.get_stats(),
            'tier_weights': torch.softmax(self.tier_weights, dim=0).tolist(),
        }

    def save_state(self) -> Dict[str, Any]:
        """Save router state for checkpointing."""
        return {
            'module_state': self.state_dict(),
            'config': {
                'wave_dim': self.wave_dim,
                'feature_dim': self.feature_dim,
            },
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load router state from checkpoint."""
        self.load_state_dict(state['module_state'])
