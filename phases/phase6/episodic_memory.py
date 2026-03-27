"""
Tier 2: Episodic Memory — FAISS vector store + metadata.

Analogy: Your diary — specific events and facts.

Episodic memory provides unlimited, permanent storage of individual facts
and experiences.  Each memory is a compressed retrieval vector paired with
rich metadata (timestamp, confidence, causal source, access count).

Write is one-shot: encode + store, no training needed.
Read is semantic similarity search via FAISS (or brute-force fallback).

Key properties:
- One-shot write (no training loop)
- FAISS-backed fast similarity search
- Access-counting for consolidation decisions
- Never decays unless explicitly contradicted
"""

import time
import torch
import numpy as np
from torch import Tensor
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

EPISODIC_CONFIG = {
    'feature_dim': 256,
    'default_confidence': 0.8,
    'decay_rate': 0.0,          # 0 = never decays
}


@dataclass
class EpisodicEntry:
    """Metadata associated with one episodic memory."""
    entry_id: int
    fact: str
    timestamp: float
    confidence: float
    causal_source: str
    access_count: int = 0
    decay_rate: float = 0.0
    contradicted: bool = False


class EpisodicMemory:
    """
    Tier 2: External vector store for facts and events.

    Uses FAISS IndexFlatIP (inner product) for fast similarity search.
    Falls back to brute-force NumPy search if FAISS is unavailable.

    Args:
        feature_dim: dimensionality of retrieval vectors (must match
                     the compression output from WorkingMemory / CSE)
    """

    def __init__(self, feature_dim: int = 256):
        self.feature_dim = feature_dim
        self._next_id: int = 0
        self._vectors: List[np.ndarray] = []

        # FAISS index (inner-product)
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(feature_dim)
        else:
            self.index = None

        self._metadata: List[EpisodicEntry] = []

    # ─────────────────────────────────────────
    # Write (one-shot)
    # ─────────────────────────────────────────

    def write(
        self,
        vector: Tensor,
        fact: str = "",
        confidence: float = 0.8,
        causal_source: str = "",
    ) -> int:
        """
        Write a fact into episodic memory (one-shot, no training).

        Args:
            vector:        [feature_dim] compressed retrieval vector
            fact:          human-readable fact string
            confidence:    confidence score [0, 1]
            causal_source: why/how this fact was learned

        Returns:
            entry_id: unique identifier for this memory
        """
        v = vector.detach().cpu().numpy().astype('float32')
        if v.ndim == 1:
            # Normalise for inner-product search
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            v = v.reshape(1, -1)

        if self.index is not None:
            self.index.add(v)
        else:
            self._vectors.append(v.squeeze(0))

        entry = EpisodicEntry(
            entry_id=self._next_id,
            fact=fact,
            timestamp=time.time(),
            confidence=confidence,
            causal_source=causal_source,
        )
        self._metadata.append(entry)
        self._next_id += 1
        return entry.entry_id

    # ─────────────────────────────────────────
    # Read (similarity search)
    # ─────────────────────────────────────────

    def search(
        self,
        query_vector: Tensor,
        k: int = 5,
    ) -> List[Tuple[EpisodicEntry, float]]:
        """
        Search episodic memory for the most similar entries.

        Args:
            query_vector: [feature_dim] query vector
            k:            number of results to return

        Returns:
            List of (EpisodicEntry, similarity_score) tuples, best first
        """
        if len(self._metadata) == 0:
            return []

        q = query_vector.detach().cpu().numpy().astype('float32')
        if q.ndim == 1:
            norm = np.linalg.norm(q)
            if norm > 0:
                q = q / norm
            q = q.reshape(1, -1)

        k = min(k, len(self._metadata))

        # Guard: backing store may be empty after a dim-mismatch rebuild
        # (metadata preserved but vectors discarded — nothing to search yet)
        if self.index is not None and self.index.ntotal == 0:
            return []
        if self.index is None and not self._vectors:
            return []

        if self.index is not None:
            scores, indices = self.index.search(q, k)
            results = []
            for rank, idx in enumerate(indices[0]):
                if 0 <= idx < len(self._metadata):
                    entry = self._metadata[idx]
                    entry.access_count += 1
                    results.append((entry, float(scores[0][rank])))
            return results
        else:
            # Brute-force fallback
            mat = np.stack(self._vectors)  # [N, feature_dim]
            sims = (mat @ q.squeeze()).astype(float)  # [N]
            top_idx = np.argsort(-sims)[:k]
            results = []
            for idx in top_idx:
                entry = self._metadata[idx]
                entry.access_count += 1
                results.append((entry, float(sims[idx])))
            return results

    # ─────────────────────────────────────────
    # Contradiction handling
    # ─────────────────────────────────────────

    def contradict(self, entry_id: int) -> None:
        """Mark an episodic entry as contradicted (it will be repelled)."""
        for entry in self._metadata:
            if entry.entry_id == entry_id:
                entry.contradicted = True
                entry.confidence *= 0.1
                return

    # ─────────────────────────────────────────
    # Consolidation helpers
    # ─────────────────────────────────────────

    def get_frequent_entries(self, min_access: int = 3) -> List[EpisodicEntry]:
        """Return entries accessed at least `min_access` times (candidates for consolidation)."""
        return [e for e in self._metadata
                if e.access_count >= min_access and not e.contradicted]

    def get_all_entries(self) -> List[EpisodicEntry]:
        """Return all non-contradicted entries."""
        return [e for e in self._metadata if not e.contradicted]

    def get_vector(self, entry_id: int) -> Optional[np.ndarray]:
        """Retrieve the raw vector for a given entry ID."""
        if entry_id < 0 or entry_id >= len(self._metadata):
            return None
        if self.index is not None:
            return faiss.rev_swig_ptr(
                self.index.get_xb(), self.index.ntotal * self.feature_dim
            ).reshape(-1, self.feature_dim)[entry_id].copy()
        else:
            return self._vectors[entry_id].copy()

    # ─────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of entries stored."""
        return len(self._metadata)

    def get_stats(self) -> Dict[str, Any]:
        """Return episodic memory statistics."""
        if not self._metadata:
            return {'size': 0, 'avg_access': 0.0, 'contradicted': 0}
        accesses = [e.access_count for e in self._metadata]
        return {
            'size': len(self._metadata),
            'avg_access': sum(accesses) / len(accesses),
            'max_access': max(accesses),
            'contradicted': sum(1 for e in self._metadata if e.contradicted),
            'avg_confidence': sum(e.confidence for e in self._metadata) / len(self._metadata),
        }

    def save_state(self) -> Dict[str, Any]:
        """Serialize episodic memory for checkpointing."""
        # Collect all vectors
        if self.index is not None and self.index.ntotal > 0:
            vecs = faiss.rev_swig_ptr(
                self.index.get_xb(), self.index.ntotal * self.feature_dim
            ).reshape(-1, self.feature_dim).copy()
        elif self._vectors:
            vecs = np.stack(self._vectors)
        else:
            vecs = np.empty((0, self.feature_dim), dtype='float32')

        entries = []
        for e in self._metadata:
            entries.append({
                'entry_id': e.entry_id,
                'fact': e.fact,
                'timestamp': e.timestamp,
                'confidence': e.confidence,
                'causal_source': e.causal_source,
                'access_count': e.access_count,
                'decay_rate': e.decay_rate,
                'contradicted': e.contradicted,
            })
        return {
            'vectors': vecs,
            'metadata': entries,
            'feature_dim': self.feature_dim,
            'next_id': self._next_id,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore episodic memory from checkpoint.

        Handles dimension mismatches gracefully: if the checkpoint was
        saved with a different feature_dim (e.g. Phase 7 = 256 loaded
        into Phase 8 = 512), the stale vectors are discarded and the
        index is rebuilt at the current model's feature_dim.  Metadata
        (facts, causal sources) is always preserved.
        """
        ckpt_dim = state['feature_dim']
        self._next_id = state['next_id']

        vecs = state['vectors']

        # Dimension mismatch — discard incompatible vectors, keep metadata
        if ckpt_dim != self.feature_dim:
            print(f"  ⚠ Episodic memory dim mismatch: checkpoint={ckpt_dim}, "
                  f"model={self.feature_dim} — rebuilding index (vectors discarded)")
            if self.index is not None:
                self.index = faiss.IndexFlatIP(self.feature_dim)
            else:
                self._vectors = []
            # Preserve metadata so facts aren't lost (vectors will be
            # re-written on next learn_fact / forward call if needed)
            self._metadata = []
            for d in state['metadata']:
                self._metadata.append(EpisodicEntry(**d))
            return

        # Dimensions match — normal restore
        if self.index is not None:
            self.index = faiss.IndexFlatIP(self.feature_dim)
            if len(vecs) > 0:
                self.index.add(vecs.astype('float32'))
        else:
            self._vectors = [vecs[i] for i in range(len(vecs))]

        self._metadata = []
        for d in state['metadata']:
            self._metadata.append(EpisodicEntry(**d))
