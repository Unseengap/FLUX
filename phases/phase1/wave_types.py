"""
SemanticWave: The fundamental data type of FLUX.
Replaces: token IDs, embeddings, positional encodings.

A SemanticWave represents a window of input as a multi-dimensional
continuous wave with phonetic, syntactic, semantic, temporal, and
intensity components.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────
# Wave Dimension Constants
# ─────────────────────────────────────────────

WAVE_DIMS = {
    'phonetic':  64,    # Character/sound pattern encoding
    'syntactic': 64,    # Grammatical structure signal
    'semantic':  256,   # Core meaning coordinates
    'temporal':  32,    # Sequential position signal
    'intensity': 16,    # Emphasis and importance
}
TOTAL_WAVE_DIM = sum(WAVE_DIMS.values())  # 432


# ─────────────────────────────────────────────
# SemanticWave Dataclass
# ─────────────────────────────────────────────

@dataclass
class SemanticWave:
    """
    A continuous representation of a text window.
    Shape: [seq_len, TOTAL_WAVE_DIM]

    Unlike token embeddings, this is:
    - Continuous (no discrete boundaries)
    - Multi-dimensional (5 separate wave components)
    - Physics-aware (waves interfere with each other)
    - Vocabulary-free (any bytes accepted)
    """
    phonetic:  Tensor    # [seq_len, 64]
    syntactic: Tensor    # [seq_len, 64]
    semantic:  Tensor    # [seq_len, 256]
    temporal:  Tensor    # [seq_len, 32]
    intensity: Tensor    # [seq_len, 16]

    @property
    def full(self) -> Tensor:
        """Concatenate all dimensions → [seq_len, 432]"""
        return torch.cat([
            self.phonetic, self.syntactic, self.semantic,
            self.temporal, self.intensity
        ], dim=-1)

    @property
    def seq_len(self) -> int:
        """Number of positions in the sequence."""
        return self.semantic.shape[0]

    def to_retrieval_vector(self) -> Tensor:
        """
        Compress to single vector for episodic memory storage.
        Uses mean pooling over semantic dimension only.
        Returns: [256]
        """
        return self.semantic.mean(dim=0)

    def cosine_similarity(self, other: 'SemanticWave') -> float:
        """
        Semantic similarity between two waves.
        Uses semantic dimension mean vectors.
        """
        v1 = self.to_retrieval_vector()
        v2 = other.to_retrieval_vector()
        return F.cosine_similarity(
            v1.unsqueeze(0), v2.unsqueeze(0)
        ).item()

    def to(self, device: str) -> 'SemanticWave':
        """Move all tensors to device."""
        return SemanticWave(
            phonetic=self.phonetic.to(device),
            syntactic=self.syntactic.to(device),
            semantic=self.semantic.to(device),
            temporal=self.temporal.to(device),
            intensity=self.intensity.to(device),
        )

    def detach(self) -> 'SemanticWave':
        """Detach all tensors from computation graph."""
        return SemanticWave(
            phonetic=self.phonetic.detach(),
            syntactic=self.syntactic.detach(),
            semantic=self.semantic.detach(),
            temporal=self.temporal.detach(),
            intensity=self.intensity.detach(),
        )
