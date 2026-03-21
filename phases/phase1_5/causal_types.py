"""
CausalWave: Extends SemanticWave with directional causality.

A SemanticWave knows what something means.
A CausalWave knows what something means AND what it tends to cause.

New dimensions added on top of the existing 432-dim wave:
    causal_forward:  64   — what meaning-states this wave tends to produce
    causal_backward: 64   — what meaning-states tend to produce this wave
    tension:         32   — contradiction pressure at this position
    chain_id:        16   — which implication chain this belongs to

Total CausalWave dim: 432 + 176 = 608
The original 432 dims are preserved exactly — Phase 2 still works unchanged.
"""

import torch
from torch import Tensor
from dataclasses import dataclass
from wave_types import SemanticWave, TOTAL_WAVE_DIM

CAUSAL_DIMS = {
    'causal_forward':  64,   # What this wave tends to cause next
    'causal_backward': 64,   # What tends to cause this wave
    'tension':         32,   # Contradiction pressure (high = conflict detected)
    'chain_id':        16,   # Implication chain membership vector
}
CAUSAL_EXTENSION_DIM = sum(CAUSAL_DIMS.values())   # 176
TOTAL_CAUSAL_DIM     = TOTAL_WAVE_DIM + CAUSAL_EXTENSION_DIM  # 608

@dataclass
class CausalWave:
    """
    A SemanticWave extended with causal direction.

    The base wave (432 dims) is untouched — Phase 2 compatibility preserved.
    The causal extension (176 dims) adds reasoning capability.

    Key property: causal_forward of position i should align with
                  causal_backward of position i+1 for coherent sequences.
    High alignment = natural flow of meaning.
    Low alignment  = abrupt shift or contradiction.
    """
    base: SemanticWave          # Original Phase 1 wave — never modified
    causal_forward:  Tensor     # [seq_len, 64]
    causal_backward: Tensor     # [seq_len, 64]
    tension:         Tensor     # [seq_len, 32]
    chain_id:        Tensor     # [seq_len, 16]

    @property
    def full(self) -> Tensor:
        """Full 608-dim causal wave tensor."""
        return torch.cat([
            self.base.full,
            self.causal_forward,
            self.causal_backward,
            self.tension,
            self.chain_id,
        ], dim=-1)

    @property
    def seq_len(self) -> int:
        return self.base.seq_len

    def causal_coherence(self) -> Tensor:
        """
        Measure how well meaning flows from each position to the next.
        coherence[i] = cosine_similarity(causal_forward[i], causal_backward[i+1])

        High coherence = natural causal sequence
        Low coherence  = abrupt shift or topic change
        Near -1.0      = direct contradiction

        Returns: [seq_len - 1] coherence scores
        """
        if self.seq_len < 2:
            return torch.zeros(0)
        fwd = self.causal_forward[:-1]    # All but last
        bwd = self.causal_backward[1:]    # All but first
        return torch.nn.functional.cosine_similarity(fwd, bwd, dim=-1)

    def tension_score(self) -> float:
        """
        Overall contradiction tension in this sequence.
        High score = conflicting meaning detected somewhere in sequence.
        Returns scalar in [0, 1].
        """
        return self.tension.norm(dim=-1).mean().item()

    def to_phase2_wave(self) -> SemanticWave:
        """
        Strip causal extension and return base SemanticWave for Phase 2.
        Phase 2 (Resonance Field) receives exactly what Phase 1 produced.
        """
        return self.base


@dataclass
class CausalArrow:
    """
    A single directional relationship: A tends to cause B.
    Stored in the implication chain registry.
    """
    source_vector:  Tensor   # [608] — the causing wave state
    target_vector:  Tensor   # [608] — the caused wave state
    strength:       float    # How reliably A causes B (0 to 1)
    evidence_count: int      # How many times this arrow was observed
    arrow_type:     str      # 'temporal', 'logical', 'semantic', 'contradiction'

    def as_dict(self) -> dict:
        return {
            'strength': self.strength,
            'evidence_count': self.evidence_count,
            'arrow_type': self.arrow_type,
        }
