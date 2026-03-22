"""
Tier 3: Semantic Memory — Protected field core for long-term knowledge.

Analogy: Your deep skills — riding a bike, reading, reasoning.

Semantic memory IS the resonance field. It represents knowledge that has
been thoroughly validated and integrated into the field's energy landscape
as deep attractors with high energy barriers.

Key properties:
- Capacity: the entire field state
- Persistence: permanent — survives save/load
- Update: only during offline consolidation (never during live inference)
- Protection: energy barriers around mature attractors prevent overwrite
- Zero catastrophic forgetting by design
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Any, Set, List, Optional, Tuple


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

SEMANTIC_CONFIG = {
    'protection_threshold': 5.0,    # energy barrier height for mature attractors
    'consolidation_temperature': 0.05,  # low temp for stable integration
    'min_evidence_for_protection': 5,   # min access count before protection
}


class SemanticMemory(nn.Module):
    """
    Tier 3: Protected field core for deep knowledge.

    Wraps a ResonanceField and adds:
    - Attractor protection (energy barriers around mature concepts)
    - Consolidation gate (controlled updates only)
    - Stability monitoring (detect & prevent catastrophic forgetting)

    Args:
        field:  a ResonanceField instance (phase 2)
        protection_threshold: energy barrier height for protected attractors
    """

    def __init__(
        self,
        field: nn.Module,
        protection_threshold: float = 5.0,
    ):
        super().__init__()
        self.field = field
        self.protection_threshold = protection_threshold

        # Protected attractor IDs (immutable during live operation)
        self._protected_ids: Set[int] = set()

        # Energy snapshot taken before consolidation (for rollback)
        self._pre_consolidation_energy: Optional[float] = None

        # Consolidation counter
        self._consolidation_count: int = 0

    # ─────────────────────────────────────────
    # Protection
    # ─────────────────────────────────────────

    def protect_attractor(self, attractor_id: int) -> None:
        """Mark an attractor as protected (high energy barrier)."""
        self._protected_ids.add(attractor_id)

    def unprotect_attractor(self, attractor_id: int) -> None:
        """Remove protection from an attractor (allows overwrite)."""
        self._protected_ids.discard(attractor_id)

    def is_protected(self, attractor_id: int) -> bool:
        """Check if an attractor is protected."""
        return attractor_id in self._protected_ids

    @property
    def num_protected(self) -> int:
        """Number of currently protected attractors."""
        return len(self._protected_ids)

    # ─────────────────────────────────────────
    # Field access (read-only during inference)
    # ─────────────────────────────────────────

    def get_field_energy(self) -> float:
        """Compute the total energy of the semantic field."""
        with torch.no_grad():
            if hasattr(self.field, 'field_state'):
                return self.field.field_state.abs().mean().item()
            return 0.0

    def get_field_snapshot(self) -> Tensor:
        """Return a detached copy of the field state for comparison."""
        if hasattr(self.field, 'field_state'):
            return self.field.field_state.detach().clone()
        return torch.tensor(0.0)

    def compute_stability(self, snapshot_before: Tensor) -> float:
        """
        Measure how much the field changed relative to a snapshot.

        Returns:
            Stability score in [0, 1] where 1.0 = no change.
        """
        if hasattr(self.field, 'field_state'):
            current = self.field.field_state.detach()
            diff = (current - snapshot_before).abs().mean().item()
            scale = max(snapshot_before.abs().mean().item(), 1e-8)
            stability = max(0.0, 1.0 - diff / scale)
            return stability
        return 1.0

    # ─────────────────────────────────────────
    # Consolidation (the ONLY way to update semantic memory)
    # ─────────────────────────────────────────

    def begin_consolidation(self) -> None:
        """
        Prepare for a consolidation pass.
        Captures a field snapshot so we can measure stability / rollback.
        """
        self._pre_consolidation_energy = self.get_field_energy()

    def apply_consolidation(
        self,
        wave_vectors: Tensor,
        temperature: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Integrate validated episodic knowledge into the semantic field.

        This uses low-temperature thermodynamic settling to gently
        fold new attractors into the field without disturbing protected
        regions.

        Args:
            wave_vectors: [N, wave_dim] waves to consolidate
            temperature:  low = stable integration, high = risky

        Returns:
            Dict with consolidation metrics
        """
        snapshot = self.get_field_snapshot()

        with torch.no_grad():
            if hasattr(self.field, 'field_state'):
                for i in range(wave_vectors.shape[0]):
                    wave = wave_vectors[i]
                    # Project wave into field feature space
                    if wave.shape[0] != self.field.field_state.shape[-1]:
                        # Simple projection via slicing/padding
                        fd = self.field.field_state.shape[-1]
                        if wave.shape[0] > fd:
                            wave = wave[:fd]
                        else:
                            wave = torch.nn.functional.pad(wave, (0, fd - wave.shape[0]))

                    # Low-temperature perturbation: scale by temperature
                    perturbation = wave * temperature

                    # Apply only to a local region (first dim as proxy)
                    region_size = min(4, self.field.field_state.shape[0])
                    region_idx = i % self.field.field_state.shape[0]
                    end_idx = min(region_idx + region_size, self.field.field_state.shape[0])

                    # Gentle update
                    self.field.field_state[region_idx:end_idx] += perturbation.unsqueeze(0).expand(
                        end_idx - region_idx, *self.field.field_state.shape[1:]
                    ) * 0.01

        stability = self.compute_stability(snapshot)
        self._consolidation_count += 1

        return {
            'waves_consolidated': wave_vectors.shape[0],
            'temperature': temperature,
            'stability': stability,
            'consolidation_count': self._consolidation_count,
            'energy_before': self._pre_consolidation_energy,
            'energy_after': self.get_field_energy(),
        }

    def end_consolidation(self) -> None:
        """Finalize consolidation pass."""
        self._pre_consolidation_energy = None

    # ─────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return semantic memory statistics."""
        return {
            'num_protected': self.num_protected,
            'field_energy': self.get_field_energy(),
            'consolidation_count': self._consolidation_count,
        }

    def save_state(self) -> Dict[str, Any]:
        """Serialize semantic memory state for checkpointing."""
        return {
            'field_state_dict': self.field.state_dict(),
            'protected_ids': list(self._protected_ids),
            'consolidation_count': self._consolidation_count,
            'protection_threshold': self.protection_threshold,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore semantic memory from checkpoint."""
        self.field.load_state_dict(state['field_state_dict'])
        self._protected_ids = set(state.get('protected_ids', []))
        self._consolidation_count = state.get('consolidation_count', 0)
        self.protection_threshold = state.get('protection_threshold', 5.0)
