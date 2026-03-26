"""
WaveRecurrentUnit (WRU): A FLUX-native recurrent cell for next-wave prediction.

Unlike a GRU which operates on opaque hidden vectors, the WRU:
  1. Keeps its hidden state as a valid 432-dim wave AT ALL TIMES
  2. Uses per-sub-band interference gates instead of sigmoid gates
  3. Applies energy conservation to prevent state divergence
  4. Every intermediate state is directly decodable by WaveToText

Physics analogy:
  - GRU gate = sigmoid(Wx + Uh)         → arbitrary learned gate
  - WRU gate = cos_sim(state, context)   → wave interference physics
  - Constructive interference (cos > 0)  → amplify / keep band
  - Destructive interference (cos < 0)   → dampen / forget band

Sub-bands (from SemanticWave):
  phonetic  [0:64]    — sound patterns
  syntactic [64:128]  — grammar signals
  semantic  [128:384] — core meaning
  temporal  [384:416] — position signals
  intensity [416:432] — emphasis

Architecture (single step):
    field_context [512]
      → FieldToWave → ctx_wave [432]
      → Per-band interference with wave_state [432]
      → Learned band transforms (5 small MLPs)
      → Energy-constrained superposition
      → output wave [432]  (decodable by WTT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional, Dict


# ─────────────────────────────────────────────
# Sub-band Dimensions (matches wave_types.py)
# ─────────────────────────────────────────────

BAND_SLICES = {
    'phonetic':  (0,   64),
    'syntactic': (64,  128),
    'semantic':  (128, 384),
    'temporal':  (384, 416),
    'intensity': (416, 432),
}

TOTAL_WAVE_DIM = 432
FIELD_DIM = 512


# ─────────────────────────────────────────────
# Band Transform — per-band learned update
# ─────────────────────────────────────────────

class BandTransform(nn.Module):
    """
    MLP that transforms a sub-band given interference context.

    Each sub-band gets its own BandTransform so the model can
    independently update phonetics without touching semantics, etc.

    Args:
        band_dim: Dimension of this sub-band (e.g. 64 for phonetic)
        hidden_mult: Hidden layer multiplier (4× the band dim by default)
    """

    def __init__(self, band_dim: int, hidden_mult: int = 4):
        super().__init__()
        hidden = band_dim * hidden_mult
        # Input: concat of (state_band, ctx_band) → 2 * band_dim
        self.net = nn.Sequential(
            nn.Linear(band_dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, band_dim),
        )
        # Initialize small so initial output is near-zero
        # (state passes through, transform adds a small correction)
        nn.init.xavier_uniform_(self.net[0].weight, gain=0.3)
        nn.init.zeros_(self.net[0].bias)
        nn.init.xavier_uniform_(self.net[2].weight, gain=0.3)
        nn.init.zeros_(self.net[2].bias)
        nn.init.xavier_uniform_(self.net[4].weight, gain=0.1)
        nn.init.zeros_(self.net[4].bias)

    def forward(self, state_band: Tensor, ctx_band: Tensor) -> Tensor:
        """
        Args:
            state_band: [..., band_dim] current wave state for this band
            ctx_band:   [..., band_dim] context wave for this band
        Returns:
            [..., band_dim] proposed update for this band
        """
        combined = torch.cat([state_band, ctx_band], dim=-1)
        return self.net(combined)


# ─────────────────────────────────────────────
# Wave Recurrent Unit
# ─────────────────────────────────────────────

class WaveRecurrentUnit(nn.Module):
    """
    FLUX-native recurrent cell for next-wave prediction.

    Single-step operation:
      1. Project field_context [512] → ctx_wave [432] via learned projection
      2. Compute per-band interference gates (cosine similarity)
      3. Apply learned band transforms where interference is destructive
      4. Superpose state and context per band
      5. Apply energy conservation constraint

    The hidden state IS a valid SemanticWave at every step.

    Args:
        wave_dim:    Wave dimension (432)
        field_dim:   Field feature dimension (512)
        energy_cap:  Maximum energy (||wave||²) allowed — prevents divergence
        residual_scale: How much of the band transform to mix in (0.0–1.0)
    """

    def __init__(
        self,
        wave_dim: int = TOTAL_WAVE_DIM,
        field_dim: int = FIELD_DIM,
        energy_cap: float = 50.0,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        self.wave_dim = wave_dim
        self.field_dim = field_dim
        self.energy_cap = energy_cap
        self.residual_scale = residual_scale

        # ── Context projection: field [512] → wave [432] ──────────────────
        # This is NOT FieldToWave (which is frozen from Phase 2).
        # This is a deeper learned projection that maps field context
        # to the WRU's wave space. Needs capacity to learn
        # "given this topic, the next word's wave looks like X."
        ctx_hidden = max(field_dim, wave_dim) + 256  # 768
        self.context_proj = nn.Sequential(
            nn.LayerNorm(field_dim),
            nn.Linear(field_dim, ctx_hidden),
            nn.GELU(),
            nn.Linear(ctx_hidden, ctx_hidden),
            nn.GELU(),
            nn.Linear(ctx_hidden, wave_dim),
        )
        nn.init.xavier_uniform_(self.context_proj[1].weight, gain=0.5)
        nn.init.zeros_(self.context_proj[1].bias)
        nn.init.xavier_uniform_(self.context_proj[3].weight, gain=0.5)
        nn.init.zeros_(self.context_proj[3].bias)
        nn.init.xavier_uniform_(self.context_proj[5].weight, gain=0.5)
        nn.init.zeros_(self.context_proj[5].bias)

        # ── Per-band transforms ───────────────────────────────────────────
        self.band_transforms = nn.ModuleDict({
            name: BandTransform(end - start)
            for name, (start, end) in BAND_SLICES.items()
        })

        # ── Learnable interference scale per band ─────────────────────────
        # Controls how much each band is affected by context interference.
        # Initialized to balanced (0.5 for each band).
        self.interference_scale = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(0.5))
            for name in BAND_SLICES
        })

        # ── Output projection (refine after superposition) ────────────────
        # Takes merged interference + ctx_wave skip (2 * wave_dim → wave_dim)
        # Deeper than before — needs to reconcile interference + context.
        out_hidden = wave_dim + 128  # 560
        self.output_norm = nn.LayerNorm(wave_dim * 2)
        self.output_proj = nn.Sequential(
            nn.Linear(wave_dim * 2, out_hidden),
            nn.GELU(),
            nn.Linear(out_hidden, wave_dim),
            nn.Tanh(),  # Keep output bounded — waves are tanh-bounded in CSE
        )
        nn.init.xavier_uniform_(self.output_proj[0].weight, gain=0.5)
        nn.init.zeros_(self.output_proj[0].bias)
        nn.init.xavier_uniform_(self.output_proj[2].weight, gain=0.5)
        nn.init.zeros_(self.output_proj[2].bias)

        # ── Initial wave state (learnable "blank") ────────────────────────
        self.initial_state = nn.Parameter(torch.randn(wave_dim) * 0.1)

    def get_initial_state(self, batch_size: int = 1) -> Tensor:
        """
        Get the initial wave state for a new sequence.

        Args:
            batch_size: Number of sequences in the batch
        Returns:
            [batch_size, 432] initial wave state
        """
        return self.initial_state.unsqueeze(0).expand(batch_size, -1)

    def _split_bands(self, wave: Tensor) -> Dict[str, Tensor]:
        """Split a wave [*, 432] into sub-band dict."""
        bands = {}
        for name, (start, end) in BAND_SLICES.items():
            bands[name] = wave[..., start:end]
        return bands

    def _merge_bands(self, bands: Dict[str, Tensor]) -> Tensor:
        """Merge sub-band dict back into wave [*, 432]."""
        parts = []
        for name in ['phonetic', 'syntactic', 'semantic', 'temporal', 'intensity']:
            parts.append(bands[name])
        return torch.cat(parts, dim=-1)

    def _energy_constraint(self, wave: Tensor, prev_energy: Optional[Tensor] = None) -> Tensor:
        """
        Apply energy conservation: wave energy can't grow unboundedly.

        If the new wave's energy exceeds the cap, scale it down.
        This prevents the recurrent state from exploding — the same
        principle as thermodynamic energy minimization in FLUX.

        Args:
            wave: [..., 432] proposed wave
            prev_energy: [...] previous energy (optional soft cap)
        Returns:
            [..., 432] energy-constrained wave
        """
        energy = (wave ** 2).sum(dim=-1, keepdim=True)  # [..., 1]
        max_energy = torch.tensor(self.energy_cap, device=wave.device, dtype=wave.dtype)
        scale = torch.where(
            energy > max_energy,
            torch.sqrt(max_energy / (energy + 1e-8)),
            torch.ones_like(energy),
        )
        return wave * scale

    def step(
        self,
        field_context: Tensor,
        wave_state: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Single WRU step: predict next wave from field context and current state.

        Args:
            field_context: [B, 512] field-space context vector
            wave_state:    [B, 432] current wave state (None → initial state)
        Returns:
            (predicted_wave [B, 432], new_state [B, 432])
        """
        B = field_context.shape[0]

        # ── Initialize state if needed ────────────────────────────────────
        if wave_state is None:
            wave_state = self.get_initial_state(B).to(field_context.device)

        # ── 1. Project field context to wave space ────────────────────────
        ctx_wave = self.context_proj(field_context)  # [B, 432]

        # ── 2. Split into sub-bands ───────────────────────────────────────
        state_bands = self._split_bands(wave_state)
        ctx_bands = self._split_bands(ctx_wave)

        # ── 3. Per-band interference + transform ─────────────────────────
        new_bands = {}
        for name, (start, end) in BAND_SLICES.items():
            s_band = state_bands[name]    # [B, band_dim]
            c_band = ctx_bands[name]      # [B, band_dim]

            # Interference gate: cosine similarity [-1, 1]
            # Positive = constructive (waves reinforce) → keep state
            # Negative = destructive (waves cancel) → allow transform
            gate = F.cosine_similarity(s_band, c_band, dim=-1).unsqueeze(-1)  # [B, 1]

            # Learned transform: proposes what to replace with
            transform = self.band_transforms[name](s_band, c_band)  # [B, band_dim]

            # Interference scale (learnable per band)
            alpha = torch.sigmoid(self.interference_scale[name])  # scalar in [0, 1]

            # Superposition:
            # - Where gate > 0 (constructive): mostly keep state, small transform
            # - Where gate < 0 (destructive): mostly replace with transform
            # The formula: new = state + alpha * (1 - gate) * (transform - state * residual_scale)
            # When gate ≈ 1 (max constructive): (1-1) = 0 → no change
            # When gate ≈ -1 (max destructive): (1-(-1)) = 2 → full transform
            blend = alpha * (1.0 - gate) * 0.5  # normalize to [0, alpha]
            new_band = (1.0 - blend) * s_band + blend * (s_band + transform * self.residual_scale)

            new_bands[name] = new_band

        # ── 4. Merge bands back ──────────────────────────────────────────
        merged = self._merge_bands(new_bands)  # [B, 432]

        # ── 5. Output refinement with ctx_wave skip connection ────────────
        # Skip connection: context has a direct path to output,
        # bypassing the interference bottleneck. This is critical
        # for single-step prediction where state starts near-zero.
        combined = torch.cat([merged, ctx_wave], dim=-1)  # [B, 864]
        refined = self.output_norm(combined)
        predicted = self.output_proj(refined)  # [B, 432]

        # ── 6. Energy constraint ──────────────────────────────────────────
        predicted = self._energy_constraint(predicted)

        # The new state IS the predicted wave — state = valid wave at all times
        new_state = predicted

        return predicted, new_state

    def forward(
        self,
        field_contexts: Tensor,
        wave_state: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for single-step prediction (Phase 2.5).

        In Phase 2.5 (single-step), this is equivalent to self.step().
        In Phase 3 (multi-step), this will be extended to chain steps.

        Args:
            field_contexts: [B, 512] field-space context vectors
            wave_state:     [B, 432] current wave state (None → initial)
        Returns:
            (predicted_wave [B, 432], new_state [B, 432])
        """
        return self.step(field_contexts, wave_state)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_summary(self) -> Dict[str, int]:
        """Detailed parameter count by component."""
        summary = {}
        summary['context_proj'] = sum(p.numel() for p in self.context_proj.parameters())
        summary['band_transforms'] = sum(p.numel() for p in self.band_transforms.parameters())
        summary['interference_scale'] = sum(p.numel() for p in self.interference_scale.values())
        summary['output'] = sum(p.numel() for p in self.output_norm.parameters()) + \
                           sum(p.numel() for p in self.output_proj.parameters())
        summary['skip_params'] = 0  # Skip connection is concat, no extra params
        summary['initial_state'] = self.initial_state.numel()
        summary['total'] = sum(summary.values())
        return summary
