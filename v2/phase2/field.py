"""
ResonanceField: Replaces all weight matrices in FLUX.

The field is a 3D spatial tensor where:
- Each location holds a feature vector (what's stored here)
- Each location has an energy scalar (how stable this location is)
- Each location has a mass scalar (how much evidence supports this)

Learning = perturbation + settling (no backprop through the field)
Memory   = stable attractor locations (energy minima)

State, energy, and mass are register_buffers (not nn.Parameters) because
they are updated by physics (perturbation + settling), not by gradient
descent. Only the projection layers (wave_to_location, wave_to_feature)
require gradients during the brief warmup phase.

v2 changes (wave-first):
  - Imports wave constants from v2/phase1 (not phases/phase1)
  - Topology collapse fix from original Phase 2 v2.1 is preserved
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path

# Import wave constants from v2/phase1
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1'))
from wave_types import TOTAL_WAVE_DIM


# ─────────────────────────────────────────────
# Field Constants
# ─────────────────────────────────────────────

FIELD_H = 64
FIELD_W = 64
FIELD_D = 64
FIELD_FEATURES = 512


# ─────────────────────────────────────────────
# FieldLocation Dataclass
# ─────────────────────────────────────────────

@dataclass
class FieldLocation:
    """A coordinate in the 3D field."""
    h: int
    w: int
    d: int

    def to_tensor(self, device: str = 'cpu') -> Tensor:
        """Convert to float tensor [3]."""
        return torch.tensor([self.h, self.w, self.d], dtype=torch.float, device=device)

    def distance_to(self, other: 'FieldLocation') -> float:
        """L2 distance to another location."""
        return (
            (self.h - other.h) ** 2 +
            (self.w - other.w) ** 2 +
            (self.d - other.d) ** 2
        ) ** 0.5


# ─────────────────────────────────────────────
# Spatial Projection MLP
# ─────────────────────────────────────────────

class SpatialProjection(nn.Module):
    """
    Maps a wave vector to 3D field coordinates.

    Replaces the original single Linear(432→3) layer which caused
    topology collapse — all inputs mapping to the same field region.

    Fix:
      1. L2-normalize input so scale doesn't dominate
      2. 2-layer MLP with hidden dim 64 gives enough capacity to
         learn diverse, spread-out spatial mappings
      3. Output uses tanh (range -1..1) then rescaled to field dims,
         instead of sigmoid (which saturates near 0.5 = field center)
    """

    def __init__(self, wave_dim: int, hidden_dim: int = 64):
        super().__init__()
        # THREE INDEPENDENT heads — each learns its own mapping
        self.head_h = nn.Sequential(
            nn.Linear(wave_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.head_w = nn.Sequential(
            nn.Linear(wave_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.head_d = nn.Sequential(
            nn.Linear(wave_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        # Init each head's final layer differently so they diverge early
        for i, head in enumerate([self.head_h, self.head_w, self.head_d]):
            nn.init.uniform_(head[-1].weight, -0.5 + i*0.1, 0.5 + i*0.1)
            nn.init.constant_(head[-1].bias, (i - 1) * 0.3)

    def forward(self, wave_vector: Tensor) -> Tensor:
        v = F.normalize(wave_vector.unsqueeze(0), dim=-1).squeeze(0)
        h = torch.tanh(self.head_h(v))
        w = torch.tanh(self.head_w(v))
        d = torch.tanh(self.head_d(v))
        return torch.cat([h, w, d], dim=-1)


# ─────────────────────────────────────────────
# ResonanceField
# ─────────────────────────────────────────────

class ResonanceField(nn.Module):
    """
    The living field that replaces all weight matrices.

    Stores knowledge as energy minima (attractors) in a 3D landscape.
    New information creates local perturbations.
    The field settling into new energy minimum = learning.

    Key invariant: Updates are LOCAL. Far regions never change unless
    the perturbation energy is physically large enough to reach them.
    This is how catastrophic forgetting is eliminated by physics.
    """

    def __init__(
        self,
        h: int = FIELD_H,
        w: int = FIELD_W,
        d: int = FIELD_D,
        features: int = FIELD_FEATURES,
        wave_dim: int = TOTAL_WAVE_DIM,
    ):
        super().__init__()
        self.h = h
        self.w = w
        self.d = d
        self.features = features
        self.wave_dim = wave_dim
        self._step_count = 0

        # ── Field state buffers (updated by physics, not gradients) ──
        self.register_buffer('state',  torch.randn(h, w, d, features) * 0.01)
        self.register_buffer('energy', torch.ones(h, w, d, 1))
        self.register_buffer('mass',   torch.zeros(h, w, d, 1))

        # ── Trainable projections (gradient-based warmup + decode loss) ──
        self.wave_to_location = SpatialProjection(wave_dim, hidden_dim=64)
        self.wave_to_feature  = nn.Linear(wave_dim, features)

    @property
    def device(self) -> torch.device:
        return self.state.device

    @property
    def step_count(self) -> int:
        return self._step_count

    def wave_to_field_coords(self, wave_vector: Tensor) -> FieldLocation:
        """
        Map a semantic wave vector to a location in the 3D field.

        Args:
            wave_vector: [wave_dim] semantic wave (mean pooled)
        Returns:
            FieldLocation in [0, H) × [0, W) × [0, D)
        """
        with torch.no_grad():
            coords = self.wave_to_location(wave_vector.detach())  # [3] in (-1, 1)
            h = int(((coords[0].item() + 1) / 2) * (self.h - 1))
            w = int(((coords[1].item() + 1) / 2) * (self.w - 1))
            d = int(((coords[2].item() + 1) / 2) * (self.d - 1))
            h = max(0, min(self.h - 1, h))
            w = max(0, min(self.w - 1, w))
            d = max(0, min(self.d - 1, d))
        return FieldLocation(h, w, d)

    def _get_neighborhood_slices(
        self, loc: FieldLocation, radius: int
    ) -> Tuple[slice, slice, slice]:
        return (
            slice(max(0, loc.h - radius), min(self.h, loc.h + radius + 1)),
            slice(max(0, loc.w - radius), min(self.w, loc.w + radius + 1)),
            slice(max(0, loc.d - radius), min(self.d, loc.d + radius + 1)),
        )

    def _compute_distances(
        self, location: FieldLocation,
        h_slice: slice, w_slice: slice, d_slice: slice,
    ) -> Tensor:
        dev = self.state.device
        hr = torch.arange(h_slice.start, h_slice.stop, device=dev, dtype=torch.float)
        wr = torch.arange(w_slice.start, w_slice.stop, device=dev, dtype=torch.float)
        dr = torch.arange(d_slice.start, d_slice.stop, device=dev, dtype=torch.float)
        hh, ww, dd = torch.meshgrid(hr, wr, dr, indexing='ij')
        return torch.sqrt(
            (hh - location.h) ** 2 +
            (ww - location.w) ** 2 +
            (dd - location.d) ** 2 +
            1e-8
        )

    def get_state_at(self, loc: FieldLocation) -> Tensor:
        """Get feature vector at a specific location. Returns [features]."""
        return self.state[loc.h, loc.w, loc.d]

    def get_mass_at(self, loc: FieldLocation) -> float:
        return self.mass[loc.h, loc.w, loc.d].item()

    def get_energy_at(self, loc: FieldLocation) -> float:
        return self.energy[loc.h, loc.w, loc.d].item()

    def perturb(self, wave_vector: Tensor, strength: float = 1.0) -> FieldLocation:
        """
        Apply a perturbation to the field at the mapped location.
        LOCAL update only — physics prevents catastrophic forgetting.

        Args:
            wave_vector: [wave_dim] input semantic wave (mean pooled)
            strength: perturbation strength
        Returns:
            FieldLocation where perturbation was applied
        """
        location = self.wave_to_field_coords(wave_vector)

        with torch.no_grad():
            target_feature = self.wave_to_feature(wave_vector).detach()

        radius = max(1, int(strength * 4))
        hs, ws, ds = self._get_neighborhood_slices(location, radius)

        distances = self._compute_distances(location, hs, ws, ds)
        weights = torch.exp(-distances / max(radius, 1)).unsqueeze(-1)

        neighborhood = self.state[hs, ws, ds]
        target_expanded = target_feature.view(1, 1, 1, -1).expand_as(neighborhood)
        delta = target_expanded - neighborhood

        local_mass = self.mass[hs, ws, ds]
        resistance = 1.0 / (1.0 + local_mass)

        update = weights * delta * strength * 0.01 * resistance
        self.state[hs, ws, ds] += update
        self.mass[hs, ws, ds] += weights * 0.01

        energy_decrease = weights * strength * 0.005
        self.energy[hs, ws, ds] -= energy_decrease
        self.energy.clamp_(min=0.01)

        self._step_count += 1
        return location

    def query(
        self,
        wave_vector: Tensor,
        k: int = 8,
        search_radius: int = 16,
    ) -> Tuple[Tensor, Tensor, List[FieldLocation]]:
        """
        Retrieve k nearest field features to a query wave.

        Args:
            wave_vector: [wave_dim] query wave
            k: number of neighbors
            search_radius: spatial search radius
        Returns:
            features [k, features], similarities [k], locations
        """
        location = self.wave_to_field_coords(wave_vector)
        hs, ws, ds = self._get_neighborhood_slices(location, search_radius)

        neighborhood = self.state[hs, ws, ds]
        shape = neighborhood.shape[:3]
        flat = neighborhood.reshape(-1, self.features)

        with torch.no_grad():
            target = self.wave_to_feature(wave_vector)

        similarities = F.cosine_similarity(
            flat, target.unsqueeze(0).expand(flat.shape[0], -1)
        )
        actual_k = min(k, flat.shape[0])
        topk = similarities.topk(actual_k)

        locations = []
        for idx in topk.indices:
            i = idx.item()
            dh = i // (shape[1] * shape[2])
            rem = i % (shape[1] * shape[2])
            dw = rem // shape[2]
            dd = rem % shape[2]
            locations.append(FieldLocation(
                h=hs.start + dh, w=ws.start + dw, d=ds.start + dd,
            ))

        return flat[topk.indices], topk.values, locations

    def settle(self, steps: int = 10, dt: float = 0.1):
        """Settle the field toward energy minimum via diffusion."""
        with torch.no_grad():
            for _ in range(steps):
                s = self.state.permute(3, 0, 1, 2).unsqueeze(0)
                neighbor_avg = F.avg_pool3d(s, kernel_size=3, stride=1, padding=1)
                neighbor_avg = neighbor_avg.squeeze(0).permute(1, 2, 3, 0)

                diffusion_rate = torch.sigmoid(self.energy) / (1.0 + self.mass)
                self.state += dt * diffusion_rate * (neighbor_avg - self.state)

                e = self.energy.permute(3, 0, 1, 2).unsqueeze(0)
                energy_avg = F.avg_pool3d(e, kernel_size=3, stride=1, padding=1)
                energy_avg = energy_avg.squeeze(0).permute(1, 2, 3, 0)
                self.energy.copy_(0.9 * self.energy + 0.1 * energy_avg)
                self.energy -= 0.001 * self.mass
                self.energy.clamp_(min=0.01, max=2.0)

    def total_energy(self) -> float:
        return self.energy.sum().item()

    def num_attractors(self, mass_threshold: float = 0.1) -> int:
        return int((self.mass.squeeze(-1) > mass_threshold).sum().item())

    def get_field_stats(self) -> Dict[str, Any]:
        return {
            'total_energy': self.total_energy(),
            'num_attractors': self.num_attractors(),
            'mean_mass': self.mass.mean().item(),
            'max_mass': self.mass.max().item(),
            'state_norm': self.state.norm().item(),
            'state_std': self.state.std().item(),
            'energy_mean': self.energy.mean().item(),
            'energy_std': self.energy.std().item(),
            'step_count': self._step_count,
        }

    def forward(self, wave_vector: Tensor) -> Tuple[Tensor, Tensor, FieldLocation]:
        """Standard forward: perturb then query."""
        location = self.perturb(wave_vector)
        features, similarities, _ = self.query(wave_vector)
        return features, similarities, location
