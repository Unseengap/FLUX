"""
FieldEvolutionGenerator — FLUX-native generation through field physics.

Instead of a separate prediction head (WRU/GRU/MLP), generation happens
through the resonance field itself:

  1. SEED: Place prefix waves into the field (perturb at mapped locations)
  2. EVOLVE: Run learned energy-settling steps (differentiable diffusion)
  3. READ: Extract the field state at the "next" position → decode to wave

This is the SPECIFICATION's core principle made real:
  "Energy minimization settling IS both inference and learning"
  "Input → Perturbation → Field Settles → Output extracted from settled state"

The field accumulates evidence from prefix words. Where waves constructively
interfere, energy drops — creating attractors. The attractor nearest to
position n+1 in the evolved field IS the prediction.

Architecture:
  - FieldSeeder: places waves into a learned local field (not the 64³ dense field —
    that's too expensive. Instead, a compact NxF feature field where N = sequence slots)
  - FieldEvolver: differentiable settling via learned diffusion + energy functions
  - FieldReader: extracts the evolved features at position n+1 → FieldToWave → wave

Trainable parameters: ~3-4M
Frozen: CSE, WaveChunker, WTT, WaveToField, FieldToWave (Phase 1 + 2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

TOTAL_WAVE_DIM = 432
FIELD_DIM = 512
MAX_SEQ_SLOTS = 64   # Max number of positions in the local field


# ─────────────────────────────────────────────
# Sub-band definitions (same as wave_types.py)
# ─────────────────────────────────────────────

BAND_SLICES = {
    'phonetic':  (0,   64),
    'syntactic': (64,  128),
    'semantic':  (128, 384),
    'temporal':  (384, 416),
    'intensity': (416, 432),
}


@dataclass
class EvolutionResult:
    """Result of a single field evolution step."""
    predicted_wave: Tensor       # [432] — the generated next wave
    field_energy: Tensor         # scalar — total field energy after settling
    energy_delta: Tensor         # scalar — energy drop during settling
    settle_steps: int            # how many settle steps ran


# ─────────────────────────────────────────────
# FieldSeeder — places waves into the local field
# ─────────────────────────────────────────────

class FieldSeeder(nn.Module):
    """
    Projects wave-space chunks into the local field representation.

    Each prefix wave gets placed at a position in the field. The seeder
    learns HOW to represent each wave as a field feature, combining:
    - The wave's content (via WaveToField bridge)
    - Positional information (where in the sequence)
    - Interference with previously seeded positions

    Args:
        wave_dim: Wave dimension (432)
        field_dim: Field feature dimension (512)
        max_slots: Maximum sequence positions
    """

    def __init__(
        self,
        wave_dim: int = TOTAL_WAVE_DIM,
        field_dim: int = FIELD_DIM,
        max_slots: int = MAX_SEQ_SLOTS,
    ):
        super().__init__()
        self.wave_dim = wave_dim
        self.field_dim = field_dim
        self.max_slots = max_slots

        # Learned position embeddings for field slots
        self.position_embed = nn.Embedding(max_slots, field_dim)

        # Project wave → field feature (separate from frozen WaveToField)
        # This learns "how a wave should be represented IN the field for generation"
        self.wave_proj = nn.Sequential(
            nn.LayerNorm(wave_dim),
            nn.Linear(wave_dim, field_dim),
            nn.GELU(),
            nn.Linear(field_dim, field_dim),
        )
        nn.init.xavier_uniform_(self.wave_proj[1].weight, gain=0.5)
        nn.init.zeros_(self.wave_proj[1].bias)
        nn.init.xavier_uniform_(self.wave_proj[3].weight, gain=0.5)
        nn.init.zeros_(self.wave_proj[3].bias)

        # Energy initialization per slot (learned prior — starts uniform)
        self.energy_prior = nn.Parameter(torch.ones(max_slots) * 1.0)

    def forward(
        self,
        prefix_waves: Tensor,
        field_contexts: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Seed the local field with prefix waves.

        Args:
            prefix_waves: [B, N, 432] — N prefix chunk waves
            field_contexts: [B, N, 512] — optional pre-projected field features
                           (from frozen WaveToField). If None, uses wave_proj.
        Returns:
            field_state: [B, max_slots, 512] — seeded field
            field_energy: [B, max_slots] — energy at each slot
        """
        B, N, _ = prefix_waves.shape
        device = prefix_waves.device
        assert N <= self.max_slots, f"Prefix length {N} > max_slots {self.max_slots}"

        # Project waves to field features
        if field_contexts is not None:
            features = field_contexts  # [B, N, 512]
        else:
            features = self.wave_proj(prefix_waves)  # [B, N, 512]

        # Add position embeddings
        positions = torch.arange(N, device=device)
        pos_embed = self.position_embed(positions)  # [N, 512]
        features = features + pos_embed.unsqueeze(0)  # [B, N, 512]

        # Initialize full field (empty slots = zero + position only)
        field_state = torch.zeros(B, self.max_slots, self.field_dim, device=device)
        field_state[:, :N, :] = features

        # Add positional priors for empty slots too (they need coordinates)
        all_pos = torch.arange(self.max_slots, device=device)
        all_pos_embed = self.position_embed(all_pos)  # [max_slots, 512]
        # Empty slots get just position (scaled down so they don't dominate)
        empty_mask = torch.zeros(self.max_slots, device=device)
        empty_mask[N:] = 1.0  # 1 for empty, 0 for seeded
        field_state = field_state + 0.1 * all_pos_embed.unsqueeze(0) * empty_mask.unsqueeze(0).unsqueeze(-1)

        # Energy: seeded slots have LOW energy (stable), empty slots have HIGH energy (unstable)
        field_energy = self.energy_prior.unsqueeze(0).expand(B, -1).clone()  # [B, max_slots]
        field_energy[:, :N] = field_energy[:, :N] * 0.1   # Seeded = low energy = stable
        # Empty slots keep high energy = will be shaped by settling

        return field_state, field_energy


# ─────────────────────────────────────────────
# FieldEvolver — differentiable energy settling
# ─────────────────────────────────────────────

class FieldEvolver(nn.Module):
    """
    Differentiable energy settling for the local field.

    Each settle step:
    1. Compute pairwise interference between adjacent slots (diffusion)
    2. Apply energy-weighted gating (high energy = more change, low = stable)
    3. Update features through learned local interactions
    4. Track energy decrease (settling = energy monotonically decreases)

    This is the learned version of ResonanceField.settle() — but differentiable
    and operating on a compact 1D sequence field instead of a 3D spatial field.

    Args:
        field_dim: Feature dimension (512)
        settle_steps: Number of settling iterations
        num_heads: Attention heads for local interaction (NOT global attention —
                  limited receptive field like 1D convolution)
    """

    def __init__(
        self,
        field_dim: int = FIELD_DIM,
        settle_steps: int = 4,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.field_dim = field_dim
        self.settle_steps = settle_steps
        self.kernel_size = kernel_size

        # Per-step settling layers (shared across all steps — like an RNN)
        # Local interaction via depthwise-separable 1D convolution
        # (each slot interacts with its K nearest neighbors — NOT global attention)
        self.local_interact = nn.Sequential(
            nn.Conv1d(field_dim, field_dim, kernel_size=kernel_size,
                     padding=kernel_size // 2, groups=field_dim),  # depthwise
            nn.GELU(),
            nn.Conv1d(field_dim, field_dim, kernel_size=1),  # pointwise
        )
        nn.init.xavier_uniform_(self.local_interact[0].weight, gain=0.3)
        nn.init.zeros_(self.local_interact[0].bias)
        nn.init.xavier_uniform_(self.local_interact[2].weight, gain=0.3)
        nn.init.zeros_(self.local_interact[2].bias)

        # Energy gate: decides how much each slot changes based on its energy
        # High energy → fluid (changes easily), Low energy → frozen (resists change)
        self.energy_gate = nn.Sequential(
            nn.Linear(field_dim + 1, field_dim),  # +1 for energy scalar
            nn.Sigmoid(),
        )
        nn.init.xavier_uniform_(self.energy_gate[0].weight, gain=0.5)
        nn.init.zeros_(self.energy_gate[0].bias)

        # Energy update: predicts new energy after interaction
        self.energy_update = nn.Sequential(
            nn.Linear(field_dim, 1),
            nn.Softplus(),  # Energy is always positive
        )
        nn.init.xavier_uniform_(self.energy_update[0].weight, gain=0.3)
        nn.init.zeros_(self.energy_update[0].bias)

        # Layer norms for stability
        self.pre_norm = nn.LayerNorm(field_dim)
        self.post_norm = nn.LayerNorm(field_dim)

    def forward(
        self,
        field_state: Tensor,
        field_energy: Tensor,
        n_prefix: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Evolve the field through settle_steps of energy minimization.

        Args:
            field_state: [B, S, 512] — seeded field
            field_energy: [B, S] — energy per slot
            n_prefix: number of seeded prefix slots
        Returns:
            evolved_state: [B, S, 512] — settled field
            final_energy: [B, S] — energy per slot after settling
            energy_trace: [B, settle_steps+1] — total energy at each step
        """
        B, S, F = field_state.shape
        device = field_state.device

        energy_trace = [field_energy.sum(dim=-1)]  # [B]

        for step in range(self.settle_steps):
            # ── 1. Local interaction (diffusion) ──────────────────────
            normed = self.pre_norm(field_state)  # [B, S, F]
            # Conv1d expects [B, C, L]
            h = normed.transpose(1, 2)  # [B, F, S]
            interaction = self.local_interact(h)  # [B, F, S]
            interaction = interaction.transpose(1, 2)  # [B, S, F]

            # ── 2. Energy-gated update ────────────────────────────────
            # Combine features + energy for gating
            gate_input = torch.cat([
                field_state,
                field_energy.unsqueeze(-1),  # [B, S, 1]
            ], dim=-1)  # [B, S, F+1]
            gate = self.energy_gate(gate_input)  # [B, S, F] in [0, 1]

            # High energy → gate ≈ 1 → full update
            # Low energy → gate ≈ 0 → resist change (mass = stability)
            delta = gate * interaction
            field_state = field_state + delta

            # ── 3. Post-normalize ─────────────────────────────────────
            field_state = self.post_norm(field_state)

            # ── 4. Update energy ──────────────────────────────────────
            # Energy should decrease (settling = energy minimization)
            new_energy = self.energy_update(field_state).squeeze(-1)  # [B, S]
            # Enforce monotonic decrease: clamp to at most previous energy
            field_energy = torch.minimum(field_energy, new_energy)

            energy_trace.append(field_energy.sum(dim=-1))

        energy_trace = torch.stack(energy_trace, dim=-1)  # [B, steps+1]
        return field_state, field_energy, energy_trace


# ─────────────────────────────────────────────
# FieldReader — extract prediction from evolved field
# ─────────────────────────────────────────────

class FieldReader(nn.Module):
    """
    Read the "next position" from the evolved field and convert to wave space.

    After the field settles, the state at position n (the slot right after
    the last prefix) should represent the field's prediction for the next wave.
    This reader extracts and refines that prediction.

    Args:
        field_dim: Field feature dimension (512)
        wave_dim: Wave output dimension (432)
    """

    def __init__(
        self,
        field_dim: int = FIELD_DIM,
        wave_dim: int = TOTAL_WAVE_DIM,
    ):
        super().__init__()
        self.field_dim = field_dim
        self.wave_dim = wave_dim

        # Contextual attention: the reader looks at the settled field
        # around the target position to gather context
        self.context_attn = nn.MultiheadAttention(
            embed_dim=field_dim,
            num_heads=4,
            dropout=0.0,
            batch_first=True,
        )

        # Project field features → wave space
        # Deeper than FieldToWave because this needs to GENERATE, not just decode
        hidden = field_dim + 128  # 640
        self.wave_proj = nn.Sequential(
            nn.LayerNorm(field_dim),
            nn.Linear(field_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, wave_dim),
            nn.Tanh(),  # Bounded like CSE output
        )
        for i in [1, 3, 5]:
            nn.init.xavier_uniform_(self.wave_proj[i].weight, gain=0.5)
            nn.init.zeros_(self.wave_proj[i].bias)

    def forward(
        self,
        evolved_field: Tensor,
        n_prefix: int,
        field_energy: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Extract the predicted next wave from the evolved field.

        Args:
            evolved_field: [B, S, 512] — settled field state
            n_prefix: position index of the next slot (= number of prefix waves)
            field_energy: [B, S] — optional energy weights
        Returns:
            predicted_wave: [B, 432]
        """
        B, S, F = evolved_field.shape

        # The query is the slot at position n_prefix (next position after prefix)
        query_pos = min(n_prefix, S - 1)
        query = evolved_field[:, query_pos:query_pos+1, :]  # [B, 1, 512]

        # Context: look at a local window around the target position
        # (wider for more context, but always local — not global attention)
        window_start = max(0, query_pos - 8)
        window_end = min(S, query_pos + 8)
        context = evolved_field[:, window_start:window_end, :]  # [B, W, 512]

        # Contextual attention (query attends to local context)
        attended, _ = self.context_attn(query, context, context)  # [B, 1, 512]
        attended = attended.squeeze(1)  # [B, 512]

        # Project to wave space
        predicted_wave = self.wave_proj(attended)  # [B, 432]

        return predicted_wave


# ─────────────────────────────────────────────
# FieldEvolutionGenerator — the full pipeline
# ─────────────────────────────────────────────

class FieldEvolutionGenerator(nn.Module):
    """
    FLUX-native text generation through field physics.

    Pipeline:
      prefix_waves [B, N, 432]
        → FieldSeeder: place waves in local field [B, S, 512]
        → FieldEvolver: settle field through K energy-minimization steps
        → FieldReader: extract prediction at position n+1 → wave [B, 432]

    The field evolution IS the computation. No separate "model" —
    the physics of the field (interference, energy settling, attractor
    formation) produces the output. The trainable parameters control
    HOW the field evolves, not WHAT the answer is.

    This aligns with SPECIFICATION.md:
      "Input → Perturbation → Field Settles → Output extracted from settled state"

    Args:
        wave_dim: Wave dimension (432)
        field_dim: Field feature dimension (512)
        max_slots: Maximum sequence length
        settle_steps: Number of energy-settling iterations
        kernel_size: Local interaction kernel size in evolver
    """

    def __init__(
        self,
        wave_dim: int = TOTAL_WAVE_DIM,
        field_dim: int = FIELD_DIM,
        max_slots: int = MAX_SEQ_SLOTS,
        settle_steps: int = 4,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.wave_dim = wave_dim
        self.field_dim = field_dim
        self.max_slots = max_slots
        self.settle_steps = settle_steps

        self.seeder = FieldSeeder(wave_dim, field_dim, max_slots)
        self.evolver = FieldEvolver(field_dim, settle_steps, kernel_size)
        self.reader = FieldReader(field_dim, wave_dim)

    def forward(
        self,
        prefix_waves: Tensor,
        field_contexts: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Generate the next wave from prefix waves via field evolution.

        Args:
            prefix_waves: [B, N, 432] — prefix chunk waves
            field_contexts: [B, N, 512] — optional pre-projected features
        Returns:
            predicted_wave: [B, 432]
            info: dict with energy_trace, field_energy, etc.
        """
        B, N, _ = prefix_waves.shape

        # ── 1. Seed the field ─────────────────────────────────────────
        field_state, field_energy = self.seeder(prefix_waves, field_contexts)

        # ── 2. Evolve through energy settling ─────────────────────────
        evolved_state, final_energy, energy_trace = self.evolver(
            field_state, field_energy, n_prefix=N,
        )

        # ── 3. Read prediction at next position ──────────────────────
        predicted_wave = self.reader(evolved_state, n_prefix=N, field_energy=final_energy)

        # ── Info for diagnostics ──────────────────────────────────────
        info = {
            'energy_trace': energy_trace,           # [B, steps+1]
            'final_energy': final_energy,           # [B, S]
            'energy_drop': energy_trace[:, 0] - energy_trace[:, -1],  # [B]
            'field_state': evolved_state.detach(),   # [B, S, 512]
        }

        return predicted_wave, info

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_summary(self) -> Dict[str, int]:
        """Detailed parameter count by component."""
        summary = {}
        summary['seeder'] = sum(p.numel() for p in self.seeder.parameters())
        summary['evolver'] = sum(p.numel() for p in self.evolver.parameters())
        summary['reader'] = sum(p.numel() for p in self.reader.parameters())
        summary['total'] = sum(summary.values())
        return summary
