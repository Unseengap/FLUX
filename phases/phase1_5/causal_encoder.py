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

    Uses temperature-scaled output so forward and backward heads
    produce outputs with meaningful angular spread — prevents
    collapse to uniform unit sphere.
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
        # Temperature: learned scaling before normalization
        # Starts at 1.0, allows the model to control output spread
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: Tensor) -> Tensor:
        """x: [seq_len, 432] → [seq_len, out_dim], temperature-scaled L2-normalized."""
        out = self.net(x)
        # Scale by temperature before normalizing — controls angular spread
        # Low temperature = sharp peaks, High temperature = spread out
        out = out / (self.temperature.abs() + 1e-4)
        return F.normalize(out, dim=-1)


class TensionDetector(nn.Module):
    """
    Detects contradiction tension at each position in a sequence.

    Core insight: contradiction creates SEMANTIC OPPOSITION between
    nearby positions. We detect this directly via pairwise cosine
    similarity within the window — no attention needed.

    Tension is HIGH when nearby positions are semantically opposed.
    Tension is LOW when meaning flows consistently.
    """
    def __init__(self, in_dim: int = 432, tension_dim: int = 32, radius: int = 4):
        super().__init__()
        self.radius = radius

        # Projects wave to a opposition-sensitive space
        self.opposition_proj = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
        )

        # Detects mean-field deviation — how far is this position
        # from the average meaning of the window?
        self.deviation_proj = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
        )

        # Final tension projection
        # Input: [opposition_128 + deviation_64 + pairwise_scalar_1] = 193
        self.tension_proj = nn.Sequential(
            nn.Linear(193, 128),
            nn.GELU(),
            nn.Linear(128, tension_dim),
            nn.Sigmoid(),
        )

    def forward(self, wave: Tensor) -> Tensor:
        """
        wave: [seq_len, 432]
        Returns: [seq_len, tension_dim] tension values in [0, 1]

        Tension is computed from:
        1. Pairwise opposition within local window (direct contradiction signal)
        2. Deviation from local mean (consistency signal)
        """
        seq_len = wave.shape[0]

        opp_features = self.opposition_proj(wave)    # [seq_len, 128]
        dev_features = self.deviation_proj(wave)     # [seq_len, 64]

        tension_inputs = []
        for i in range(seq_len):
            # Window boundaries
            lo = max(0, i - self.radius)
            hi = min(seq_len, i + self.radius + 1)

            window = opp_features[lo:hi]             # [w, 128]
            center = opp_features[i].unsqueeze(0)    # [1, 128]

            # Pairwise cosine similarity of center to all window members
            sims = F.cosine_similarity(
                center.expand(window.shape[0], -1), window, dim=-1
            )                                        # [w]

            # Opposition score: how many window members are opposing?
            # Negative cosine similarity = semantic opposition
            opposition = F.relu(-sims).mean()        # scalar in [0, 1]

            # Mean deviation: how far is this position from window mean?
            window_mean = opp_features[lo:hi].mean(dim=0)
            deviation   = (opp_features[i] - window_mean).norm()

            # Pack features
            opp_feat = opp_features[i]               # [128]
            dev_feat = dev_features[i]               # [64]
            pair_scalar = opposition.unsqueeze(0) * deviation.unsqueeze(0)  # [1]

            combined = torch.cat([opp_feat, dev_feat, pair_scalar], dim=-1)  # [193]
            tension_inputs.append(combined)

        tension_in = torch.stack(tension_inputs, dim=0)  # [seq_len, 193]
        tension    = self.tension_proj(tension_in)        # [seq_len, tension_dim]

        # Scale output by opposition magnitude so contradictions are visibly higher
        # Compute per-position opposition score as a scaling factor
        opp_norms = F.normalize(opp_features, dim=-1)
        # Mean anti-correlation within window
        global_sim = F.cosine_similarity(
            opp_norms.unsqueeze(1), opp_norms.unsqueeze(0), dim=-1
        ).mean(dim=1, keepdim=True)                      # [seq_len, 1]
        # More opposition = higher scaling
        opposition_scale = (1.0 - global_sim).clamp(0.1, 2.0)
        tension = tension * opposition_scale

        return tension


class ChainIDEncoder(nn.Module):
    """
    Assigns each position to an implication chain.
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
        return self.encoder(wave)


class CausalWaveChainer(nn.Module):
    """
    Main Phase 1.5 module. Wraps frozen Phase 1 CSE output with
    causal direction, tension, and chain identity.

    The CSE is FROZEN. This module only adds the causal extension.
    Phase 2 compatibility is preserved via .to_phase2_wave().
    """

    def __init__(
        self,
        wave_dim:     int = 432,
        forward_dim:  int = 64,
        backward_dim: int = 64,
        tension_dim:  int = 32,
        chain_dim:    int = 16,
        device:       str = 'cuda',
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

        causal_fwd  = self.forward_head(full)
        causal_bwd  = self.backward_head(full)
        tension     = self.tension_det(full)
        chain_id    = self.chain_encoder(full)

        return CausalWave(
            base            = semantic_wave,
            causal_forward  = causal_fwd,
            causal_backward = causal_bwd,
            tension         = tension,
            chain_id        = chain_id,
        )

    def encode(self, cse, text: str) -> CausalWave:
        """CSE encode then causal extend in one call."""
        with torch.no_grad():
            wave = cse.encode(text)
        return self.forward(wave)

    def causal_coherence_loss(self, causal_wave: CausalWave) -> Tensor:
        """
        Forward[i] should align with backward[i+1].
        Loss = 1 - mean(cosine_similarity(forward[:-1], backward[1:]))
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
        Original sentence must have HIGHER coherence than shuffled.

        Uses a harder margin (0.3) and more shuffles to force
        meaningful directional learning rather than just hitting
        the margin floor.
        """
        original_cw = self.encode(cse, text)
        orig_coh    = original_cw.causal_coherence()
        if orig_coh.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        orig_score = orig_coh.mean()

        # Harder margin than before — forces genuine differentiation
        margin = 0.3
        total_loss = torch.tensor(0.0, device=self.device)

        words = text.split()
        if len(words) < 3:
            return total_loss

        shuf_scores = []
        for _ in range(n_shuffles):
            idx      = torch.randperm(len(words)).tolist()
            shuffled = ' '.join([words[i] for i in idx])
            shuf_cw  = self.encode(cse, shuffled)
            shuf_coh = shuf_cw.causal_coherence()
            if shuf_coh.numel() > 0:
                shuf_scores.append(shuf_coh.mean())

        if not shuf_scores:
            return total_loss

        mean_shuf = torch.stack(shuf_scores).mean()
        # Hinge loss: original must beat shuffled by at least margin
        loss = torch.clamp(mean_shuf - orig_score + margin, min=0.0)
        return loss

    def contradiction_loss(
        self,
        cse,
        statement: str,
        contradiction: str,
        non_contradiction: str,
    ) -> Tensor:
        """
        Contradiction pairs must have HIGHER tension than neutral.

        Uses tension_score() which aggregates the TensionDetector output.
        Harder margin (0.15) to force meaningful separation.
        """
        joint_contra  = self.encode(cse, statement + ' ' + contradiction)
        joint_neutral = self.encode(cse, statement + ' ' + non_contradiction)

        t_contra  = joint_contra.tension_score()
        t_neutral = joint_neutral.tension_score()

        margin = 0.15
        # Convert to tensor for gradient flow
        t_c = joint_contra.tension.norm(dim=-1).mean()
        t_n = joint_neutral.tension.norm(dim=-1).mean()

        loss = torch.clamp(t_n - t_c + margin, min=0.0)
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
        If A implies B and B implies C, chains must be transitive.
        """
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