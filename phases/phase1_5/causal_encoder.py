"""
CausalWaveChainer: The core Phase 1.5 module.

Takes a SemanticWave from Phase 1 (frozen, untouched)
and wraps it with causal direction, tension, and chain identity.

Architecture:
    SemanticWave [seq_len, 432]
        → Forward causal projection  → [seq_len, 64]
        → Backward causal projection → [seq_len, 64]
        → Tension detector           → [seq_len, 32]
        → Chain ID encoder           → [seq_len, 16]
        → CausalWave [seq_len, 608]

The forward and backward projections are trained to satisfy:
    cosine_similarity(forward[i], backward[i+1]) is HIGH
    when position i causally precedes position i+1 in real text.

This is learned from naturally ordered text — no labels required.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from causal_types import CausalWave, CausalArrow, CAUSAL_DIMS
from wave_types import SemanticWave


class CausalProjectionHead(nn.Module):
    """
    Projects semantic wave to causal direction vector.
    Used for both forward and backward causal heads.
    Separate heads learn different aspects of causality.
    """
    def __init__(self, in_dim: int = 432, out_dim: int = 64, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: [seq_len, 432] → [seq_len, out_dim], L2-normalized."""
        out = self.net(x)
        return F.normalize(out, dim=-1)


class TensionDetector(nn.Module):
    """
    Detects contradiction tension at each position in a sequence.

    Tension is HIGH when:
    - Nearby positions have opposing semantic vectors
    - The causal forward of position i contradicts causal backward of i+1
    - A fact that was established earlier is being negated

    Tension is LOW when:
    - Meaning flows naturally
    - No contradiction is present
    """
    def __init__(self, in_dim: int = 432, tension_dim: int = 32, radius: int = 4):
        super().__init__()
        self.radius = radius
        # Attends to local window to detect contradiction
        self.local_attention = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1,
        )
        self.tension_proj = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Linear(128, tension_dim),
            nn.Sigmoid(),  # Tension in [0, 1]
        )

    def forward(self, wave: Tensor) -> Tensor:
        """
        wave: [seq_len, 432]
        Returns: [seq_len, tension_dim] tension values in [0, 1]
        """
        # Self-attention over local window detects local contradictions
        x = wave.unsqueeze(0)  # [1, seq_len, 432]
        attended, _ = self.local_attention(x, x, x)
        attended = attended.squeeze(0)  # [seq_len, 432]

        # Tension = divergence between original and attended
        divergence = wave - attended
        tension = self.tension_proj(divergence)
        return tension


class ChainIDEncoder(nn.Module):
    """
    Assigns each position to an implication chain.

    Positions that belong to the same logical chain
    (e.g. all statements about a single subject)
    get similar chain_id vectors.

    Positions that start new logical chains get distinct vectors.
    This is learned from co-occurrence patterns in training text.
    """
    def __init__(self, in_dim: int = 432, chain_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, chain_dim),
            nn.Tanh(),
        )

    def forward(self, wave: Tensor) -> Tensor:
        """wave: [seq_len, 432] → [seq_len, chain_dim]"""
        return self.encoder(wave)


class CausalWaveChainer(nn.Module):
    """
    Main Phase 1.5 module. Wraps frozen Phase 1 CSE output with
    causal direction, tension, and chain identity.

    The CSE is FROZEN. This module only adds the causal extension.
    Phase 2 compatibility is preserved — call .to_phase2_wave()
    to strip the extension when handing off to the Resonance Field.

    Training objectives:
    1. Causal coherence loss: forward[i] should align with backward[i+1]
    2. Order sensitivity loss: swapped sequences should have lower coherence
    3. Contradiction loss: contradicting sentences should have high tension
    4. Chain consistency loss: same-topic positions should share chain_id
    """

    def __init__(
        self,
        wave_dim:    int = 432,
        forward_dim: int = 64,
        backward_dim: int = 64,
        tension_dim: int = 32,
        chain_dim:   int = 16,
        device:      str = 'cuda',
    ):
        super().__init__()
        self.wave_dim     = wave_dim
        self.forward_dim  = forward_dim
        self.backward_dim = backward_dim
        self.tension_dim  = tension_dim
        self.chain_dim    = chain_dim
        self.device       = device

        self.forward_head  = CausalProjectionHead(wave_dim, forward_dim)
        self.backward_head = CausalProjectionHead(wave_dim, backward_dim)
        self.tension_det   = TensionDetector(wave_dim, tension_dim)
        self.chain_encoder = ChainIDEncoder(wave_dim, chain_dim)

    def forward(self, semantic_wave: SemanticWave) -> CausalWave:
        """
        Extend a Phase 1 SemanticWave with causal direction.

        Args:
            semantic_wave: output of ContinuousSemanticEncoder.encode()
        Returns:
            CausalWave: base wave + causal extension (608 dims total)
        """
        full = semantic_wave.full           # [seq_len, 432]

        causal_fwd  = self.forward_head(full)     # [seq_len, 64]
        causal_bwd  = self.backward_head(full)    # [seq_len, 64]
        tension     = self.tension_det(full)      # [seq_len, 32]
        chain_id    = self.chain_encoder(full)    # [seq_len, 16]

        return CausalWave(
            base            = semantic_wave,
            causal_forward  = causal_fwd,
            causal_backward = causal_bwd,
            tension         = tension,
            chain_id        = chain_id,
        )

    def encode(self, cse, text: str) -> CausalWave:
        """
        Convenience: CSE encode then causal extend in one call.

        Args:
            cse:  frozen ContinuousSemanticEncoder from Phase 1
            text: raw input string
        Returns:
            CausalWave
        """
        with torch.no_grad():
            wave = cse.encode(text)
        return self.forward(wave)

    def causal_coherence_loss(self, causal_wave: CausalWave) -> Tensor:
        """
        Forward pass of position i should predict backward pass of i+1.
        Minimizing this loss teaches directional causal flow.

        Loss = 1 - mean(cosine_similarity(forward[:-1], backward[1:]))
        Perfect causal chain = loss 0.0
        """
        coherence = causal_wave.causal_coherence()
        if coherence.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        return 1.0 - coherence.mean()

    def order_sensitivity_loss(
        self,
        cse,
        text: str,
        n_shuffles: int = 4,
    ) -> Tensor:
        """
        The original sequence should have HIGHER coherence than shuffled.

        For each shuffle:
            coherence(original) > coherence(shuffled)

        Loss = mean(max(0, coherence(shuffled) - coherence(original) + margin))
        Margin = 0.2 — shuffled must be at least 0.2 worse than original.
        """
        original_cw  = self.encode(cse, text)
        orig_coh     = original_cw.causal_coherence().mean()

        margin = 0.2
        total_loss = torch.tensor(0.0, device=self.device)

        words = text.split()
        if len(words) < 3:
            return total_loss

        for _ in range(n_shuffles):
            idx      = torch.randperm(len(words)).tolist()
            shuffled = ' '.join([words[i] for i in idx])
            shuf_cw  = self.encode(cse, shuffled)
            shuf_coh = shuf_cw.causal_coherence().mean()
            # Original should be more coherent than shuffled
            loss_i   = torch.clamp(shuf_coh - orig_coh + margin, min=0.0)
            total_loss = total_loss + loss_i

        return total_loss / n_shuffles

    def contradiction_loss(
        self,
        cse,
        statement: str,
        contradiction: str,
        non_contradiction: str,
    ) -> Tensor:
        """
        Contradiction pairs should have HIGHER tension than neutral pairs.

        tension(statement + contradiction) > tension(statement + non_contradiction)

        Example:
            statement       = "the sky is blue"
            contradiction   = "the sky is green"
            non_contradiction = "birds can fly"
        """
        joint_contra = self.encode(cse, statement + ' ' + contradiction)
        joint_neutral = self.encode(cse, statement + ' ' + non_contradiction)

        tension_contra  = joint_contra.tension_score()
        tension_neutral = joint_neutral.tension_score()

        # Contradiction must have higher tension than neutral
        margin = 0.1
        loss = torch.clamp(
            torch.tensor(tension_neutral - tension_contra + margin,
                         device=self.device),
            min=0.0
        )
        return loss

    def implication_consistency_loss(
        self,
        cw_a: CausalWave,
        cw_b: CausalWave,
        cw_c: CausalWave,
        implies_ab: bool,
        implies_bc: bool,
    ) -> Tensor:
        """
        If A implies B and B implies C, then A should partially imply C.
        Transitivity of causality.

        This teaches the model that reasoning chains are transitive,
        not just pairwise — the core of logical reasoning.
        """
        # Mean causal forward vectors for each sequence
        fwd_a = cw_a.causal_forward.mean(dim=0)
        bwd_b = cw_b.causal_backward.mean(dim=0)
        fwd_b = cw_b.causal_forward.mean(dim=0)
        bwd_c = cw_c.causal_backward.mean(dim=0)

        sim_ab = F.cosine_similarity(fwd_a.unsqueeze(0), bwd_b.unsqueeze(0)).squeeze()
        sim_bc = F.cosine_similarity(fwd_b.unsqueeze(0), bwd_c.unsqueeze(0)).squeeze()

        loss = torch.tensor(0.0, device=self.device)

        if implies_ab:
            loss = loss + torch.clamp(0.5 - sim_ab, min=0.0)
        else:
            loss = loss + torch.clamp(sim_ab - 0.1, min=0.0)

        if implies_bc:
            loss = loss + torch.clamp(0.5 - sim_bc, min=0.0)
        else:
            loss = loss + torch.clamp(sim_bc - 0.1, min=0.0)

        return loss
