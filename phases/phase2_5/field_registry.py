"""
FieldRegistry: Hash-based sparse storage engine for SparseResonanceField.

Instead of a dense [H,W,D,F] tensor that allocates memory for every
possible location, the registry only allocates memory where an attractor
actually forms. Empty space costs zero bytes.

Storage:
    key = h * (W * D) + w * D + d  →  integer
    features[key] = [512] tensor
    mass[key]     = float
    energy[key]   = float

This separation from the field logic means the storage layer can be
swapped, inspected, or migrated independently of growth logic.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field as dc_field
from datetime import datetime


@dataclass
class RegistryStats:
    """Snapshot of registry state at a point in time."""
    active_locations:   int
    max_h:              int
    max_w:              int
    max_d:              int
    theoretical_max:    int
    capacity_fraction:  float
    total_mass:         float
    mean_mass:          float
    max_mass:           float
    feature_dim:        int
    memory_mb:          float
    timestamp:          str


class FieldRegistry:
    """
    Sparse storage engine for the SparseResonanceField.

    Keys are flat integers computed from (h, w, d) coordinates.
    Only locations that have been perturbed at least once are stored.

    Thread-safe for single-process use. For multi-GPU, wrap with
    appropriate synchronization (future work).
    """

    def __init__(
        self,
        max_h:       int = 64,
        max_w:       int = 64,
        max_d:       int = 64,
        feature_dim: int = 512,
        device:      str = 'cpu',
    ):
        self.max_h       = max_h
        self.max_w       = max_w
        self.max_d       = max_d
        self.feature_dim = feature_dim
        self.device      = device

        # Core sparse storage
        self._features: Dict[int, Tensor] = {}
        self._mass:     Dict[int, float]  = {}
        self._energy:   Dict[int, float]  = {}

        # Growth history — list of RegistryStats at each resize
        self.growth_log: List[RegistryStats] = []

    # ─────────────────────────────────────────
    # Coordinate encoding
    # ─────────────────────────────────────────

    def _key(self, h: int, w: int, d: int) -> int:
        """Encode (h,w,d) as a single integer key."""
        return h * (self.max_w * self.max_d) + w * self.max_d + d

    def _decode_key(self, key: int) -> Tuple[int, int, int]:
        """Decode integer key back to (h, w, d)."""
        h   = key // (self.max_w * self.max_d)
        rem = key  % (self.max_w * self.max_d)
        w   = rem  // self.max_d
        d   = rem  %  self.max_d
        return h, w, d

    def _clamp(self, h: int, w: int, d: int) -> Tuple[int, int, int]:
        """Clamp coordinates to valid range."""
        return (
            max(0, min(self.max_h - 1, h)),
            max(0, min(self.max_w - 1, w)),
            max(0, min(self.max_d - 1, d)),
        )

    # ─────────────────────────────────────────
    # Read / Write
    # ─────────────────────────────────────────

    def get(self, h: int, w: int, d: int) -> Optional[Tensor]:
        """Get feature vector at location, or None if unallocated."""
        key = self._key(*self._clamp(h, w, d))
        return self._features.get(key, None)

    def get_mass(self, h: int, w: int, d: int) -> float:
        key = self._key(*self._clamp(h, w, d))
        return self._mass.get(key, 0.0)

    def get_energy(self, h: int, w: int, d: int) -> float:
        key = self._key(*self._clamp(h, w, d))
        return self._energy.get(key, 1.0)

    def set(
        self,
        h: int, w: int, d: int,
        feature:  Tensor,
        mass:     float = 0.0,
        energy:   float = 1.0,
    ):
        """Store or update a location."""
        key = self._key(*self._clamp(h, w, d))
        self._features[key] = feature.detach().to(self.device)
        self._mass[key]     = mass
        self._energy[key]   = energy

    def update(
        self,
        h: int, w: int, d: int,
        delta_feature: Tensor,
        delta_mass:    float = 0.01,
        delta_energy:  float = -0.005,
    ):
        """
        Apply an incremental update to a location.
        Allocates if not yet present (starts from near-zero noise).
        """
        key = self._key(*self._clamp(h, w, d))

        if key not in self._features:
            # Allocate new location with small random noise
            self._features[key] = (
                torch.randn(self.feature_dim, device=self.device) * 0.01
            )
            self._mass[key]   = 0.0
            self._energy[key] = 1.0

        self._features[key] = self._features[key] + delta_feature.to(self.device)
        self._mass[key]     = self._mass[key]   + delta_mass
        self._energy[key]   = max(0.01, self._energy[key] + delta_energy)

    # ─────────────────────────────────────────
    # Batch operations
    # ─────────────────────────────────────────

    def batch_query_neighborhood(
        self,
        h: int, w: int, d: int,
        radius: int = 16,
        k:      int = 8,
        target_feature: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, List[Tuple[int,int,int]]]:
        """
        Find k nearest active locations within radius of (h,w,d).

        Returns:
            features:     [k, feature_dim]
            similarities: [k]
            locations:    list of (h,w,d) tuples
        """
        candidates = []
        for dh in range(-radius, radius + 1):
            for dw in range(-radius, radius + 1):
                for dd in range(-radius, radius + 1):
                    ch, cw, cd = self._clamp(h + dh, w + dw, d + dd)
                    key = self._key(ch, cw, cd)
                    if key in self._features:
                        candidates.append((ch, cw, cd, key))

        if not candidates:
            empty_f = torch.zeros(k, self.feature_dim, device=self.device)
            empty_s = torch.zeros(k, device=self.device)
            return empty_f, empty_s, []

        # Build candidate feature matrix
        locs  = [(c[0], c[1], c[2]) for c in candidates]
        feats = torch.stack([self._features[c[3]] for c in candidates])

        if target_feature is not None:
            tf  = target_feature.to(self.device)
            sims = F.cosine_similarity(
                feats, tf.unsqueeze(0).expand_as(feats)
            )
        else:
            sims = torch.ones(len(feats), device=self.device)

        actual_k = min(k, len(feats))
        topk     = sims.topk(actual_k)
        top_locs = [locs[i] for i in topk.indices.tolist()]

        # Pad to k if needed
        pad = k - actual_k
        top_feats = feats[topk.indices]
        top_sims  = topk.values
        if pad > 0:
            top_feats = torch.cat([
                top_feats,
                torch.zeros(pad, self.feature_dim, device=self.device)
            ])
            top_sims = torch.cat([
                top_sims,
                torch.zeros(pad, device=self.device)
            ])
            top_locs += [(h, w, d)] * pad

        return top_feats, top_sims, top_locs

    def get_top_by_mass(self, k: int = 10) -> List[Tuple[Tuple,float,float]]:
        """Return top-k locations by mass: [(loc, mass, energy), ...]"""
        if not self._mass:
            return []
        sorted_keys = sorted(self._mass.keys(), key=lambda x: self._mass[x], reverse=True)
        results = []
        for key in sorted_keys[:k]:
            loc    = self._decode_key(key)
            results.append((loc, self._mass[key], self._energy[key]))
        return results

    # ─────────────────────────────────────────
    # Statistics
    # ─────────────────────────────────────────

    def active_count(self) -> int:
        return len(self._features)

    def theoretical_max(self) -> int:
        return self.max_h * self.max_w * self.max_d

    def capacity_fraction(self) -> float:
        t = self.theoretical_max()
        return self.active_count() / t if t > 0 else 0.0

    def total_mass(self) -> float:
        return sum(self._mass.values()) if self._mass else 0.0

    def memory_mb(self) -> float:
        """Estimate current memory usage in MB."""
        n   = self.active_count()
        f   = self.feature_dim
        # features: n * f * 4 bytes (float32)
        # mass + energy: n * 2 * 8 bytes (python float ≈ 8 bytes)
        return (n * f * 4 + n * 2 * 8) / 1e6

    def snapshot_stats(self) -> RegistryStats:
        """Capture current state as a RegistryStats snapshot."""
        masses = list(self._mass.values()) if self._mass else [0.0]
        return RegistryStats(
            active_locations  = self.active_count(),
            max_h             = self.max_h,
            max_w             = self.max_w,
            max_d             = self.max_d,
            theoretical_max   = self.theoretical_max(),
            capacity_fraction = self.capacity_fraction(),
            total_mass        = self.total_mass(),
            mean_mass         = sum(masses) / len(masses),
            max_mass          = max(masses),
            feature_dim       = self.feature_dim,
            memory_mb         = self.memory_mb(),
            timestamp         = datetime.now().isoformat(),
        )

    # ─────────────────────────────────────────
    # Growth / Migration
    # ─────────────────────────────────────────

    def resize(
        self,
        new_h: int,
        new_w: int,
        new_d: int,
    ) -> 'FieldRegistry':
        """
        Migrate all active locations to a larger coordinate space.

        Coordinates are scaled proportionally:
            new_h_coord = old_h_coord * (new_h / old_h)

        Returns a new FieldRegistry with migrated data.
        The old registry is unchanged (for rollback safety).
        """
        scale_h = new_h / self.max_h
        scale_w = new_w / self.max_w
        scale_d = new_d / self.max_d

        # Record pre-migration stats
        pre_stats = self.snapshot_stats()
        self.growth_log.append(pre_stats)

        new_reg = FieldRegistry(
            max_h       = new_h,
            max_w       = new_w,
            max_d       = new_d,
            feature_dim = self.feature_dim,
            device      = self.device,
        )
        new_reg.growth_log = list(self.growth_log)

        for key, feat in self._features.items():
            old_h, old_w, old_d = self._decode_key(key)
            nh = min(new_h - 1, int(old_h * scale_h))
            nw = min(new_w - 1, int(old_w * scale_w))
            nd = min(new_d - 1, int(old_d * scale_d))
            new_key = new_reg._key(nh, nw, nd)
            # If collision at new key, average the features
            if new_key in new_reg._features:
                new_reg._features[new_key] = (
                    new_reg._features[new_key] + feat
                ) / 2.0
                new_reg._mass[new_key]   += self._mass.get(key, 0.0)
                new_reg._energy[new_key]  = min(
                    new_reg._energy.get(new_key, 1.0),
                    self._energy.get(key, 1.0),
                )
            else:
                new_reg._features[new_key] = feat.clone()
                new_reg._mass[new_key]     = self._mass.get(key, 0.0)
                new_reg._energy[new_key]   = self._energy.get(key, 1.0)

        print(
            f"  FieldRegistry migrated: "
            f"{self.max_h}³ → {new_h}³  |  "
            f"{self.active_count()} → {new_reg.active_count()} locations"
        )
        return new_reg

    # ─────────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────────

    def serialize(self) -> Dict[str, Any]:
        """Convert to checkpoint-safe dict."""
        return {
            'max_h':       self.max_h,
            'max_w':       self.max_w,
            'max_d':       self.max_d,
            'feature_dim': self.feature_dim,
            'features':    {k: v.cpu() for k, v in self._features.items()},
            'mass':        dict(self._mass),
            'energy':      dict(self._energy),
            'growth_log':  [
                {
                    'active_locations':   s.active_locations,
                    'max_h':              s.max_h,
                    'max_w':              s.max_w,
                    'max_d':              s.max_d,
                    'theoretical_max':    s.theoretical_max,
                    'capacity_fraction':  s.capacity_fraction,
                    'total_mass':         s.total_mass,
                    'mean_mass':          s.mean_mass,
                    'max_mass':           s.max_mass,
                    'feature_dim':        s.feature_dim,
                    'memory_mb':          s.memory_mb,
                    'timestamp':          s.timestamp,
                }
                for s in self.growth_log
            ],
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any], device: str = 'cpu') -> 'FieldRegistry':
        """Restore from checkpoint dict."""
        reg = cls(
            max_h       = data['max_h'],
            max_w       = data['max_w'],
            max_d       = data['max_d'],
            feature_dim = data['feature_dim'],
            device      = device,
        )
        reg._features = {
            int(k): v.to(device)
            for k, v in data['features'].items()
        }
        reg._mass   = {int(k): v for k, v in data['mass'].items()}
        reg._energy = {int(k): v for k, v in data['energy'].items()}

        reg.growth_log = [
            RegistryStats(**{
                k: v for k, v in entry.items()
            })
            for entry in data.get('growth_log', [])
        ]
        return reg
