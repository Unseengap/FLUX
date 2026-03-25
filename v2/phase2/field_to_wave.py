"""
FieldToWave: Inverse projection from field feature space [512] → wave space [432].

This is THE OTHER HALF of the field bridge that must be trained with decode loss.

In the original roadmap:
- wave_to_field and field_to_wave stayed at random init
- Phase 7 FLUXModel used .detach() which permanently cut gradients
- Phase 9 WaveToText had to decode RANDOM FIELD FEATURES — impossible
- Result: mode collapse, context collapse, unembeddable wave space

In v2, FieldToWave is trained from Phase 2 step 1 with a full decode loss:
    WTT(field_to_wave(wave_to_field(chunk_wave))) ≈ original bytes

This creates a trained, invertible bridge between wave space and field space.

Architecture:
    field_vec [512] → LayerNorm → Linear(512, 432) → GELU → Linear(432, 432)
    
Symmetric with WaveToField so reconstruction loss is meaningful.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FieldToWave(nn.Module):
    """
    Project a field feature vector back to wave space.

    Input:  [field_dim]  Field feature vector (512-dim)
    Output: [wave_dim]   Reconstructed wave vector (432-dim)

    The reconstruction should satisfy:
        FieldToWave(WaveToField(wave)) ≈ wave     (reconstruction loss)
        WTT(FieldToWave(WaveToField(wave))) ≈ text (decode loss)

    Args:
        field_dim:  Input field feature dimension (512)
        wave_dim:   Output wave dimension (432)
        hidden_dim: Internal hidden layer size (512)
    """

    def __init__(
        self,
        field_dim:  int = 512,
        wave_dim:   int = 432,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.field_dim = field_dim
        self.wave_dim  = wave_dim

        self.norm = nn.LayerNorm(field_dim)

        self.proj = nn.Sequential(
            nn.Linear(field_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, wave_dim),
        )

        nn.init.xavier_uniform_(self.proj[0].weight, gain=0.5)
        nn.init.xavier_uniform_(self.proj[3].weight, gain=0.5)
        nn.init.zeros_(self.proj[0].bias)
        nn.init.zeros_(self.proj[3].bias)

    def forward(self, field_vec: Tensor) -> Tensor:
        """
        Args:
            field_vec: [..., field_dim] field feature vector(s)
        Returns:
            [..., wave_dim] reconstructed wave vector(s)
        """
        x = self.norm(field_vec)
        return self.proj(x)


# ─────────────────────────────────────────────
# Round-Trip Loss Helpers
# ─────────────────────────────────────────────

def wave_field_reconstruction_loss(
    wtf: 'nn.Module',
    ftw: 'FieldToWave',
    wave: Tensor,
) -> Tensor:
    """
    Compute reconstruction loss for the wave ↔ field bridge.

    loss = MSE(wave, FieldToWave(WaveToField(wave)))
         + (1 - cosine_similarity(wave, FieldToWave(WaveToField(wave))))

    Args:
        wtf: WaveToField projection
        ftw: FieldToWave projection
        wave: [..., wave_dim] input wave(s)
    Returns:
        Scalar reconstruction loss
    """
    field_vec = wtf(wave)
    reconstructed = ftw(field_vec)

    mse_loss = F.mse_loss(reconstructed, wave)
    cos_loss = (1.0 - F.cosine_similarity(
        reconstructed.view(-1, reconstructed.shape[-1]),
        wave.view(-1, wave.shape[-1]),
        dim=-1,
    )).mean()

    return mse_loss + 0.5 * cos_loss
