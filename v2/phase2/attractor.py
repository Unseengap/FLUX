"""
AttractorCatalog: Tracks and catalogs stable attractors in the field.

An attractor is a location in the field that has:
- High mass (reinforced by many observations)
- Low energy (stable, resists change)
- Strong feature vector (clear identity)
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass, field as dc_field
from typing import List, Optional, Dict, Any
from datetime import datetime


# ─────────────────────────────────────────────
# Attractor Dataclass
# ─────────────────────────────────────────────

@dataclass
class Attractor:
    """A snapshot of a stable attractor in the resonance field."""
    attractor_id: int
    location: tuple            # (h, w, d)
    feature_snapshot: Tensor   # [features] — frozen copy at registration time
    mass: float
    energy: float
    label: str = ""
    timestamp: str = ""
    alive: bool = True


# ─────────────────────────────────────────────
# AttractorCatalog
# ─────────────────────────────────────────────

class AttractorCatalog:
    """
    Maintains a catalog of known attractors in a ResonanceField.

    Attractors are discovered via mass/energy thresholds, then tracked
    over time to verify persistence (no-forgetting invariant).
    """

    def __init__(self, field: 'ResonanceField'):
        self.field = field
        self.attractors: List[Attractor] = []
        self._next_id = 0

    def _next_attractor_id(self) -> int:
        aid = self._next_id
        self._next_id += 1
        return aid

    def register(self, location: tuple, label: str = "") -> Attractor:
        """Manually register an attractor at a known location."""
        h, w, d = location
        feature = self.field.state[h, w, d].detach().clone()
        mass    = self.field.mass[h, w, d].item()
        energy  = self.field.energy[h, w, d].item()

        attractor = Attractor(
            attractor_id=self._next_attractor_id(),
            location=location,
            feature_snapshot=feature,
            mass=mass,
            energy=energy,
            label=label,
            timestamp=datetime.now().isoformat(),
            alive=True,
        )
        self.attractors.append(attractor)
        return attractor

    def register_from_wave(self, wave_vector: Tensor, label: str = "") -> Attractor:
        """Register an attractor at the location mapped from a wave vector."""
        loc = self.field.wave_to_field_coords(wave_vector)
        return self.register((loc.h, loc.w, loc.d), label=label)

    def scan_and_update(
        self,
        mass_threshold: float = 0.3,
        max_new: int = 50,
    ) -> int:
        """
        Scan the field for new stable attractors and add to catalog.

        Args:
            mass_threshold: minimum mass to qualify as attractor
            max_new: maximum new attractors to register per scan
        Returns:
            Number of new attractors found
        """
        mass = self.field.mass.squeeze(-1)
        candidates = (mass > mass_threshold).nonzero(as_tuple=False)

        if candidates.shape[0] == 0:
            return 0

        masses = mass[candidates[:, 0], candidates[:, 1], candidates[:, 2]]
        order  = masses.argsort(descending=True)
        candidates = candidates[order]

        new_count = 0
        for idx in range(min(candidates.shape[0], max_new * 5)):
            h, w, d = candidates[idx].tolist()
            loc = (h, w, d)

            if self._is_near_existing(loc, min_distance=3.0):
                continue

            self.register(loc, label=f"auto_scan_{self._next_id}")
            new_count += 1

            if new_count >= max_new:
                break

        return new_count

    def _is_near_existing(self, location: tuple, min_distance: float) -> bool:
        h, w, d = location
        for att in self.attractors:
            ah, aw, ad = att.location
            dist = ((h - ah) ** 2 + (w - aw) ** 2 + (d - ad) ** 2) ** 0.5
            if dist < min_distance:
                return True
        return False

    def verify_attractor_persistence(
        self,
        attractor_id: int,
        similarity_threshold: float = 0.7,
    ) -> bool:
        """Check that a previously registered attractor is still intact."""
        att = self.get_by_id(attractor_id)
        if att is None:
            return False

        h, w, d = att.location
        current_feature  = self.field.state[h, w, d].detach()
        original_feature = att.feature_snapshot.to(current_feature.device)

        similarity = F.cosine_similarity(
            current_feature.unsqueeze(0),
            original_feature.unsqueeze(0),
        ).item()

        att.alive = similarity >= similarity_threshold
        return att.alive

    def verify_all(self, similarity_threshold: float = 0.7) -> Dict[str, Any]:
        """Verify all registered attractors and return a summary."""
        results = []
        alive = 0
        for att in self.attractors:
            h, w, d = att.location
            current  = self.field.state[h, w, d].detach()
            original = att.feature_snapshot.to(current.device)
            sim = F.cosine_similarity(
                current.unsqueeze(0), original.unsqueeze(0)
            ).item()
            is_alive = sim >= similarity_threshold
            att.alive = is_alive
            if is_alive:
                alive += 1
            results.append({
                'id': att.attractor_id,
                'label': att.label,
                'similarity': sim,
                'alive': is_alive,
            })

        total = len(self.attractors)
        return {
            'total': total,
            'alive': alive,
            'dead': total - alive,
            'survival_rate': alive / total if total > 0 else 1.0,
            'details': results,
        }

    def get_by_id(self, attractor_id: int) -> Optional[Attractor]:
        for att in self.attractors:
            if att.attractor_id == attractor_id:
                return att
        return None

    def get_all(self) -> List[Attractor]:
        return self.attractors

    def get_alive(self) -> List[Attractor]:
        return [a for a in self.attractors if a.alive]

    def count(self) -> int:
        return len(self.attractors)

    def to_dict(self) -> List[Dict[str, Any]]:
        """Serialize catalog for checkpointing."""
        return [
            {
                'attractor_id': a.attractor_id,
                'location': a.location,
                'feature_snapshot': a.feature_snapshot.cpu(),
                'mass': a.mass,
                'energy': a.energy,
                'label': a.label,
                'timestamp': a.timestamp,
                'alive': a.alive,
            }
            for a in self.attractors
        ]

    def from_dict(self, data: List[Dict[str, Any]]):
        """Load catalog from serialized dicts."""
        self.attractors = []
        for d in data:
            self.attractors.append(Attractor(
                attractor_id=d['attractor_id'],
                location=tuple(d['location']),
                feature_snapshot=d['feature_snapshot'],
                mass=d['mass'],
                energy=d['energy'],
                label=d.get('label', ''),
                timestamp=d.get('timestamp', ''),
                alive=d.get('alive', True),
            ))
        if self.attractors:
            self._next_id = max(a.attractor_id for a in self.attractors) + 1

    def summary(self) -> str:
        total = self.count()
        alive = len(self.get_alive())
        lines = [f"AttractorCatalog: {total} attractors ({alive} alive)"]
        for a in self.attractors:
            status = '✓' if a.alive else '✗'
            lines.append(
                f"  {status} #{a.attractor_id} at {a.location} "
                f"mass={a.mass:.4f} energy={a.energy:.4f} "
                f"label='{a.label}'"
            )
        return "\n".join(lines)
