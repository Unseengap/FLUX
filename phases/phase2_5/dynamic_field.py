"""
SparseResonanceField: Dynamic sparse field for Phase 2.5.

Inherits topology from Phase 2's ResonanceField (SpatialProjection
weights + wave_to_feature) and extends to a sparse address space
that grows on demand up to 256³.

Key differences from Phase 2's ResonanceField:
  - Storage: FieldRegistry dict instead of dense [H,W,D,F] buffer
  - Dimensions: logical address space, not physical allocation
  - Growth: GrowthManager promotes to next tier when capacity fills
  - Memory: proportional to active attractors, not field dimensions

Phase 2 compatibility: wave_to_field_coords, perturb, query,
settle — all same signatures.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
from datetime import datetime

# Phase 1 path for wave types
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1'))
from wave_types import TOTAL_WAVE_DIM

from field_registry import FieldRegistry
from growth_manager import GrowthManager, GROWTH_TIERS


# ─────────────────────────────────────────────
# Spatial Projection (inherited from Phase 2)
# ─────────────────────────────────────────────

class SpatialProjection(nn.Module):
    """
    Identical to Phase 2's SpatialProjection.
    Three independent heads, each learns its own axis mapping.
    Weights are inherited from Phase 2 checkpoint.
    """
    def __init__(self, wave_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.head_h = nn.Sequential(
            nn.Linear(wave_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )
        self.head_w = nn.Sequential(
            nn.Linear(wave_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )
        self.head_d = nn.Sequential(
            nn.Linear(wave_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )
        for i, head in enumerate([self.head_h, self.head_w, self.head_d]):
            nn.init.uniform_(head[-1].weight, -0.5 + i * 0.1, 0.5 + i * 0.1)
            nn.init.constant_(head[-1].bias, (i - 1) * 0.3)

    def forward(self, x: Tensor) -> Tensor:
        v = F.normalize(x, dim=-1)
        h = torch.tanh(self.head_h(v))
        w = torch.tanh(self.head_w(v))
        d = torch.tanh(self.head_d(v))
        return torch.cat([h, w, d], dim=-1)

    def to_field_coords(self, x: Tensor, H: int, W: int, D: int) -> Tuple[int,int,int]:
        coords = self.forward(x)
        h = int(((coords[0].item() + 1) / 2) * (H - 1))
        w = int(((coords[1].item() + 1) / 2) * (W - 1))
        d = int(((coords[2].item() + 1) / 2) * (D - 1))
        return (
            max(0, min(H - 1, h)),
            max(0, min(W - 1, w)),
            max(0, min(D - 1, d)),
        )


# ─────────────────────────────────────────────
# SparseResonanceField
# ─────────────────────────────────────────────

class SparseResonanceField(nn.Module):
    """
    Sparse dynamic resonance field for Phase 2.5.

    Logical address space starts at 64³ (inherited from Phase 2)
    and grows up to 256³ as attractors fill the field.

    Memory usage is proportional to active attractors:
        10,000 attractors × 512 features × 4 bytes ≈ 20 MB
        100,000 attractors × 512 features × 4 bytes ≈ 200 MB

    The field grows when active_locations / theoretical_max > 0.6.
    Each growth saves a tier checkpoint that can be used for
    inference-time real-time learning.
    """

    def __init__(
        self,
        initial_h:       int   = 64,
        initial_w:       int   = 64,
        initial_d:       int   = 64,
        features:        int   = 512,
        wave_dim:        int   = TOTAL_WAVE_DIM,
        max_tier:        int   = 5,
        checkpoint_dir:  str   = 'checkpoints/',
        device:          str   = 'cuda',
    ):
        super().__init__()
        self.features       = features
        self.wave_dim       = wave_dim
        self.device_str     = device
        self._step_count    = 0

        # Sparse storage
        self.registry = FieldRegistry(
            max_h       = initial_h,
            max_w       = initial_w,
            max_d       = initial_d,
            feature_dim = features,
            device      = device,
        )

        # Growth management
        self.growth_manager = GrowthManager(
            checkpoint_dir = checkpoint_dir,
            device         = device,
            max_tier       = max_tier,
        )

        # Trainable projections (inherited from Phase 2)
        self.wave_to_location = SpatialProjection(wave_dim, hidden_dim=64)
        self.wave_to_feature  = nn.Linear(wave_dim, features)

    # ─────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────

    @property
    def h(self) -> int:
        return self.registry.max_h

    @property
    def w(self) -> int:
        return self.registry.max_w

    @property
    def d(self) -> int:
        return self.registry.max_d

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # ─────────────────────────────────────────
    # Phase 2 compatible interface
    # ─────────────────────────────────────────

    def wave_to_field_coords(self, wave_vector: Tensor) -> Tuple[int,int,int]:
        """Map wave vector to (h,w,d) in current field dimensions."""
        with torch.no_grad():
            return self.wave_to_location.to_field_coords(
                wave_vector.detach(), self.h, self.w, self.d
            )

    def perturb(
        self,
        wave_vector:  Tensor,
        strength:     float = 1.0,
    ) -> Tuple[int,int,int]:
        """
        Apply a perturbation at the mapped location.
        Identical semantics to Phase 2's ResonanceField.perturb().

        Automatically checks for growth after each perturbation.
        """
        loc = self.wave_to_field_coords(wave_vector)
        h, w, d = loc

        with torch.no_grad():
            target_feature = self.wave_to_feature(wave_vector).detach()

        radius = max(1, int(strength * 4))

        # Update neighborhood
        for dh in range(-radius, radius + 1):
            for dw in range(-radius, radius + 1):
                for dd in range(-radius, radius + 1):
                    dist = (dh**2 + dw**2 + dd**2) ** 0.5
                    weight = torch.exp(
                        torch.tensor(-dist / max(radius, 1))
                    ).item()

                    ch = max(0, min(self.h - 1, h + dh))
                    cw = max(0, min(self.w - 1, w + dw))
                    cd = max(0, min(self.d - 1, d + dd))

                    current = self.registry.get(ch, cw, cd)
                    if current is None:
                        current = torch.zeros(
                            self.features, device=self.device_str
                        )

                    existing_mass = self.registry.get_mass(ch, cw, cd)
                    resistance    = 1.0 / (1.0 + existing_mass)

                    delta = (
                        (target_feature - current)
                        * weight * strength * 0.01 * resistance
                    )

                    self.registry.update(
                        ch, cw, cd,
                        delta_feature = delta,
                        delta_mass    = weight * 0.01,
                        delta_energy  = -weight * strength * 0.005,
                    )

        self._step_count += 1
        return loc

    def query(
        self,
        wave_vector:   Tensor,
        k:             int = 8,
        search_radius: int = 16,
    ) -> Tuple[Tensor, Tensor, List[Tuple[int,int,int]]]:
        """
        Retrieve k nearest field features to a query wave.
        Same signature as Phase 2's ResonanceField.query().
        """
        h, w, d = self.wave_to_field_coords(wave_vector)

        with torch.no_grad():
            target = self.wave_to_feature(wave_vector)

        feats, sims, locs = self.registry.batch_query_neighborhood(
            h, w, d,
            radius         = search_radius,
            k              = k,
            target_feature = target,
        )
        return feats, sims, locs

    def settle(self, steps: int = 10, dt: float = 0.1):
        """
        Simplified settle for sparse field.
        Diffuses features of active locations toward their neighbors.
        """
        with torch.no_grad():
            active_keys = list(self.registry._features.keys())
            for _ in range(steps):
                for key in active_keys:
                    h, w, d = self.registry._decode_key(key)
                    neighbors = []
                    for dh, dw, dd in [
                        (1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)
                    ]:
                        nf = self.registry.get(
                            h+dh, w+dw, d+dd
                        )
                        if nf is not None:
                            neighbors.append(nf)

                    if not neighbors:
                        continue

                    neighbor_mean = torch.stack(neighbors).mean(dim=0)
                    current = self.registry._features[key]
                    mass    = self.registry._mass.get(key, 0.0)
                    energy  = self.registry._energy.get(key, 1.0)

                    diffusion = torch.sigmoid(
                        torch.tensor(energy)
                    ).item() / (1.0 + mass)
                    self.registry._features[key] = (
                        current + dt * diffusion * (neighbor_mean - current)
                    )

    # ─────────────────────────────────────────
    # Capacity & growth
    # ─────────────────────────────────────────

    def check_and_grow(self, extra_state: Dict[str, Any] = None) -> bool:
        """
        Check if growth is needed and execute if so.

        Args:
            extra_state: additional state to include in tier checkpoint
        Returns:
            True if growth occurred
        """
        if not self.growth_manager.should_grow(self.registry):
            return False

        state = extra_state or {}
        state['spatial_projection'] = self.wave_to_location.state_dict()
        state['wave_to_feature']    = self.wave_to_feature.state_dict()
        state['step_count']         = self._step_count

        new_registry, event = self.growth_manager.grow(
            self.registry, state
        )
        self.registry = new_registry
        return True

    def active_capacity(self) -> Dict[str, Any]:
        """Return current capacity information."""
        return self.growth_manager.full_status(self.registry)

    def num_attractors(self, mass_threshold: float = 0.1) -> int:
        """Count locations with mass above threshold."""
        return sum(
            1 for m in self.registry._mass.values()
            if m > mass_threshold
        )

    def total_energy(self) -> float:
        return sum(self.registry._energy.values())

    def get_field_stats(self) -> Dict[str, Any]:
        stats = self.growth_manager.full_status(self.registry)
        masses = list(self.registry._mass.values()) or [0.0]
        stats.update({
            'num_attractors':  self.num_attractors(),
            'total_energy':    self.total_energy(),
            'max_mass':        max(masses),
            'mean_mass':       sum(masses) / len(masses),
            'step_count':      self._step_count,
        })
        return stats

    # ─────────────────────────────────────────
    # Inherit Phase 2 weights
    # ─────────────────────────────────────────

    def inherit_from_phase2(self, phase2_state_dict: Dict[str, Tensor]):
        """
        Load SpatialProjection and wave_to_feature weights from
        a Phase 2 checkpoint state dict.

        Also migrates existing Phase 2 attractors (mass > 0.1) into
        the sparse registry at their scaled coordinates.
        """
        # Load projection weights
        sp_keys = {
            k.replace('wave_to_location.', ''): v
            for k, v in phase2_state_dict.items()
            if k.startswith('wave_to_location.')
        }
        if sp_keys:
            self.wave_to_location.load_state_dict(sp_keys, strict=False)
            print(f"  ✓ SpatialProjection weights inherited from Phase 2")

        wf_keys = {
            k.replace('wave_to_feature.', ''): v
            for k, v in phase2_state_dict.items()
            if k.startswith('wave_to_feature.')
        }
        if wf_keys:
            self.wave_to_feature.load_state_dict(wf_keys, strict=False)
            print(f"  ✓ wave_to_feature weights inherited from Phase 2")

        # Migrate dense field attractors
        if 'state' in phase2_state_dict:
            dense_state  = phase2_state_dict['state']   # [64,64,64,512]
            dense_mass   = phase2_state_dict.get('mass', None)
            old_h, old_w, old_d = dense_state.shape[:3]
            scale_h = self.h / old_h
            scale_w = self.w / old_w
            scale_d = self.d / old_d

            migrated = 0
            if dense_mass is not None:
                mass_sq = dense_mass.squeeze(-1)  # [H,W,D]
                coords  = (mass_sq > 0.1).nonzero(as_tuple=False)
                for idx in range(coords.shape[0]):
                    oh, ow, od = coords[idx].tolist()
                    feature    = dense_state[oh, ow, od].cpu()
                    mass_val   = mass_sq[oh, ow, od].item()
                    nh = min(self.h - 1, int(oh * scale_h))
                    nw = min(self.w - 1, int(ow * scale_w))
                    nd = min(self.d - 1, int(od * scale_d))
                    self.registry.set(nh, nw, nd, feature, mass=mass_val)
                    migrated += 1

            print(f"  ✓ Migrated {migrated} Phase 2 attractors to sparse registry")

    # ─────────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────────

    def get_state_for_checkpoint(self) -> Dict[str, Any]:
        """Full state dict for checkpointing."""
        return {
            'registry':           self.registry.serialize(),
            'growth_manager':     self.growth_manager.save_state(),
            'wave_to_location':   self.wave_to_location.state_dict(),
            'wave_to_feature':    self.wave_to_feature.state_dict(),
            'step_count':         self._step_count,
            'features':           self.features,
            'wave_dim':           self.wave_dim,
        }

    def load_from_checkpoint(self, state: Dict[str, Any]):
        """Restore full state from checkpoint dict."""
        self.registry = FieldRegistry.deserialize(
            state['registry'], device=self.device_str
        )
        self.growth_manager.load_state(state['growth_manager'])
        self.wave_to_location.load_state_dict(state['wave_to_location'])
        self.wave_to_feature.load_state_dict(state['wave_to_feature'])
        self._step_count = state.get('step_count', 0)
