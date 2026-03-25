"""
WaveToField: Projection from wave space [432] → field feature space [512].

This is ONE HALF of the field bridge that must be trained WITH a decode loss.
In the original roadmap, this stayed at random init until Phase 7's .detach()
permanently cut the gradient path. That's what killed Phase 9.

In v2, WaveToField and FieldToWave are trained together from Phase 2 step 1,
with a decode loss that flows back through the entire pipeline.

Architecture:
    wave [432] → LayerNorm → Linear(432, 512) → GELU → Linear(512, 512)
    
The two-layer design gives it enough capacity to learn a non-trivial
mapping rather than just a linear projection.

Trained jointly with FieldToWave so that:
    FieldToWave(WaveToField(wave)) ≈ wave  (reconstruction)
    WTT(FieldToWave(WaveToField(wave))) ≈ original_text  (decodability)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class WaveToField(nn.Module):
    """
    Project a wave vector from CSE space to field feature space.

    Input:  [wave_dim]    CSE wave vector (432-dim, mean-pooled)
    Output: [field_dim]   Field feature vector (512-dim)

    Must be trained jointly with FieldToWave using:
        - MSE loss: FieldToWave(WaveToField(wave)) ≈ wave
        - Decode loss: WTT(FieldToWave(WaveToField(chunk_wave))) ≈ bytes

    Args:
        wave_dim:   Input wave dimension (432)
        field_dim:  Output field feature dimension (512)
        hidden_dim: Internal hidden layer size (512)
    """

    def __init__(
        self,
        wave_dim:   int = 432,
        field_dim:  int = 512,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.wave_dim  = wave_dim
        self.field_dim = field_dim

        # LayerNorm first: stabilizes training when wave norms vary
        self.norm = nn.LayerNorm(wave_dim)

        # Two-layer MLP with residual bypass when dims match
        self.proj = nn.Sequential(
            nn.Linear(wave_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, field_dim),
        )

        # Initialize final layer with small values so the projection
        # starts close to identity in the directions it can express
        nn.init.xavier_uniform_(self.proj[0].weight, gain=0.5)
        nn.init.xavier_uniform_(self.proj[3].weight, gain=0.5)
        nn.init.zeros_(self.proj[0].bias)
        nn.init.zeros_(self.proj[3].bias)

    def forward(self, wave: Tensor) -> Tensor:
        """
        Args:
            wave: [..., wave_dim] wave vector(s)
        Returns:
            [..., field_dim] field feature vector(s)
        """
        x = self.norm(wave)
        return self.proj(x)
