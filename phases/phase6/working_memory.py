"""
Tier 1: Working Memory — Rolling field window for recent context.

Analogy: Your desk — what you're looking at right now.

Working memory is session-scoped: it holds the most recent perturbations
(semantic waves) in a bounded rolling window and provides fast context
retrieval for the current interaction.  It sits in front of the resonance
field and feeds the gravitational-relevance query pipeline.

Key properties:
- Fixed capacity (window_size perturbations)
- Immediate read/write (no indexing overhead)
- Session only — cleared on save/load cycle
- Supports importance-weighted eviction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

WORKING_MEMORY_CONFIG = {
    'window_size': 512,
    'feature_dim': 256,       # episodic retrieval dim (compressed wave)
    'wave_dim': 432,          # full CSE wave dim
}


@dataclass
class WorkingMemoryEntry:
    """A single entry in the working memory window."""
    wave: Tensor          # [wave_dim] full semantic wave
    vector: Tensor        # [feature_dim] compressed retrieval vector
    timestamp: int        # insertion order
    importance: float     # computed importance score


class WorkingMemory(nn.Module):
    """
    Tier 1: Rolling field window for recent context.

    Maintains a bounded list of recent semantic wave perturbations,
    with an importance-weighted eviction mechanism and dot-product
    attention retrieval.

    Args:
        window_size:  maximum number of entries in the window
        wave_dim:     dimensionality of full CSE waves (432)
        feature_dim:  compressed retrieval vector dimension (256)
    """

    def __init__(
        self,
        window_size: int = 512,
        wave_dim: int = 432,
        feature_dim: int = 256,
    ):
        super().__init__()
        self.window_size = window_size
        self.wave_dim = wave_dim
        self.feature_dim = feature_dim

        # Compression projection: wave_dim → feature_dim
        self.compress = nn.Linear(wave_dim, feature_dim)

        # Importance scorer: estimates relevance of an entry
        self.importance_scorer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Query / Key projections for attention-based retrieval
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)

        # Internal rolling state (session-only, not checkpointed)
        self._entries: List[WorkingMemoryEntry] = []
        self._clock: int = 0

    # ─────────────────────────────────────────
    # Write path
    # ─────────────────────────────────────────

    def add_perturbation(self, wave: Tensor) -> None:
        """
        Add a semantic wave (or batch of waves) to working memory.

        Args:
            wave: [wave_dim] or [seq_len, wave_dim] semantic wave(s)
        """
        if wave.dim() == 2:
            for i in range(wave.shape[0]):
                self._add_single(wave[i])
        else:
            self._add_single(wave)

    def _add_single(self, wave: Tensor) -> None:
        """Add one wave vector to the rolling window."""
        wave_d = wave.detach()
        with torch.no_grad():
            vec = self.compress(wave_d.unsqueeze(0)).squeeze(0)
            importance = self.importance_scorer(vec.unsqueeze(0)).squeeze().item()

        entry = WorkingMemoryEntry(
            wave=wave_d.cpu(),
            vector=vec.cpu(),
            timestamp=self._clock,
            importance=importance,
        )
        self._entries.append(entry)
        self._clock += 1

        # Evict lowest-importance entry when over capacity
        if len(self._entries) > self.window_size:
            self._evict()

    def _evict(self) -> None:
        """Remove the lowest-importance entry."""
        if not self._entries:
            return
        min_idx = min(range(len(self._entries)),
                      key=lambda i: self._entries[i].importance)
        self._entries.pop(min_idx)

    # ─────────────────────────────────────────
    # Read path
    # ─────────────────────────────────────────

    def get_current_context(self) -> Tensor:
        """
        Return all current entries as a stacked tensor.

        Returns:
            [N, wave_dim] tensor of recent waves, or [0, wave_dim] if empty.
        """
        if not self._entries:
            return torch.empty(0, self.wave_dim)
        return torch.stack([e.wave for e in self._entries])

    def get_retrieval_vectors(self) -> Tensor:
        """
        Return compressed retrieval vectors for all entries.

        Returns:
            [N, feature_dim] or [0, feature_dim] if empty.
        """
        if not self._entries:
            return torch.empty(0, self.feature_dim)
        return torch.stack([e.vector for e in self._entries])

    def query(self, query_wave: Tensor, top_k: int = 10) -> Tuple[Tensor, Tensor]:
        """
        Retrieve the most relevant working memory entries for a query.

        Args:
            query_wave: [wave_dim] or [feature_dim] query vector
            top_k:      number of entries to return

        Returns:
            waves:  [k, wave_dim] most relevant waves
            scores: [k] relevance scores
        """
        if not self._entries:
            return torch.empty(0, self.wave_dim), torch.empty(0)

        with torch.no_grad():
            # Compress query if full-wave dimension
            if query_wave.shape[-1] == self.wave_dim:
                q = self.compress(query_wave.unsqueeze(0))
            else:
                q = query_wave.unsqueeze(0)
            q = self.query_proj(q)  # [1, feature_dim]

            vecs = torch.stack([e.vector for e in self._entries])  # [N, fd]
            keys = self.key_proj(vecs)  # [N, feature_dim]

            scores = (q @ keys.T).squeeze(0) / (self.feature_dim ** 0.5)  # [N]

            k = min(top_k, len(self._entries))
            top_scores, top_idx = scores.topk(k)
            waves = torch.stack([self._entries[i].wave for i in top_idx])

        return waves, top_scores

    # ─────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of entries currently in working memory."""
        return len(self._entries)

    def clear(self) -> None:
        """Clear all entries (session reset)."""
        self._entries.clear()
        self._clock = 0

    def get_stats(self) -> Dict[str, Any]:
        """Return working memory statistics."""
        if not self._entries:
            return {'size': 0, 'avg_importance': 0.0, 'window_size': self.window_size}
        importances = [e.importance for e in self._entries]
        return {
            'size': len(self._entries),
            'window_size': self.window_size,
            'avg_importance': sum(importances) / len(importances),
            'max_importance': max(importances),
            'min_importance': min(importances),
            'clock': self._clock,
        }

    def state_dict_memory(self) -> Dict[str, Any]:
        """Save nn.Module state (entries are session-only, not saved)."""
        return {
            'module_state': self.state_dict(),
            'config': {
                'window_size': self.window_size,
                'wave_dim': self.wave_dim,
                'feature_dim': self.feature_dim,
            },
        }

    def load_state_memory(self, state: Dict[str, Any]) -> None:
        """Load nn.Module state from checkpoint."""
        self.load_state_dict(state['module_state'])
