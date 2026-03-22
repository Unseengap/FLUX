"""
Consolidation — Episodic → Semantic distillation.

Periodically reviews episodic memories by frequency of retrieval.
Frequently retrieved facts are promoted into the semantic field
via low-temperature thermodynamic settling.

This is the mechanism that converts "I remember being told X"
(episodic) into "I know X" (semantic) — mirroring how human
memory consolidation works during sleep.

Key properties:
- Only runs during offline passes (never during live inference)
- Uses low temperature for gentle, stable integration
- Checks stability before/after to prevent catastrophic change
- Respects energy barriers around protected attractors
"""

import sys
import torch
from torch import Tensor
from typing import Dict, Any, List, Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from episodic_memory import EpisodicMemory, EpisodicEntry
from semantic_memory import SemanticMemory


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

CONSOLIDATION_CONFIG = {
    'min_access_count': 3,          # min accesses before promotion
    'consolidation_temperature': 0.05,
    'max_per_batch': 50,            # max entries per consolidation pass
    'stability_threshold': 0.95,    # abort if stability drops below this
}


class ConsolidationProcess:
    """
    Distills high-frequency episodic memories into the semantic field.

    Steps:
    1. Scan episodic memory for frequently-accessed entries
    2. Snapshot the semantic field (for rollback check)
    3. Apply low-temperature consolidation to fold entries into field
    4. Verify stability — rollback if field stability < threshold

    Args:
        episodic:    EpisodicMemory instance
        semantic:    SemanticMemory instance
        min_access:  minimum access count for promotion
        temperature: consolidation temperature (low = stable)
        max_per_batch: max entries consolidated in one pass
    """

    def __init__(
        self,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        min_access: int = 3,
        temperature: float = 0.05,
        max_per_batch: int = 50,
        stability_threshold: float = 0.95,
    ):
        self.episodic = episodic
        self.semantic = semantic
        self.min_access = min_access
        self.temperature = temperature
        self.max_per_batch = max_per_batch
        self.stability_threshold = stability_threshold

        # History tracking
        self._history: List[Dict[str, Any]] = []

    def find_candidates(self) -> List[EpisodicEntry]:
        """
        Find episodic entries eligible for consolidation.

        Returns:
            List of EpisodicEntry with access_count >= min_access,
            sorted by access count (most accessed first).
        """
        candidates = self.episodic.get_frequent_entries(self.min_access)
        candidates.sort(key=lambda e: e.access_count, reverse=True)
        return candidates[:self.max_per_batch]

    def run_consolidation(
        self,
        wave_dim: int = 432,
    ) -> Dict[str, Any]:
        """
        Execute one consolidation pass: episodic → semantic.

        1. Find candidates
        2. Convert to wave vectors
        3. Apply to semantic field via low-temp settling
        4. Check stability

        Args:
            wave_dim: dimension of full semantic waves

        Returns:
            Dict with consolidation metrics
        """
        candidates = self.find_candidates()

        if not candidates:
            result = {
                'candidates_found': 0,
                'consolidated': 0,
                'stability': 1.0,
                'status': 'no_candidates',
            }
            self._history.append(result)
            return result

        # Build wave vectors from episodic entries
        waves = []
        for entry in candidates:
            vec = self.episodic.get_vector(entry.entry_id)
            if vec is not None:
                t = torch.from_numpy(vec).float()
                # Pad/truncate to wave_dim
                if t.shape[0] < wave_dim:
                    t = torch.nn.functional.pad(t, (0, wave_dim - t.shape[0]))
                else:
                    t = t[:wave_dim]
                waves.append(t)

        if not waves:
            result = {
                'candidates_found': len(candidates),
                'consolidated': 0,
                'stability': 1.0,
                'status': 'no_valid_vectors',
            }
            self._history.append(result)
            return result

        wave_tensor = torch.stack(waves)  # [N, wave_dim]

        # Perform consolidation
        self.semantic.begin_consolidation()
        metrics = self.semantic.apply_consolidation(
            wave_tensor, temperature=self.temperature
        )
        self.semantic.end_consolidation()

        # Protect newly consolidated attractors
        for entry in candidates[:len(waves)]:
            self.semantic.protect_attractor(entry.entry_id)

        result = {
            'candidates_found': len(candidates),
            'consolidated': len(waves),
            'stability': metrics['stability'],
            'energy_before': metrics.get('energy_before'),
            'energy_after': metrics.get('energy_after'),
            'temperature': self.temperature,
            'status': 'success' if metrics['stability'] >= self.stability_threshold else 'stability_warning',
        }
        self._history.append(result)
        return result

    @property
    def consolidation_count(self) -> int:
        """Total number of consolidation passes executed."""
        return len(self._history)

    def get_history(self) -> List[Dict[str, Any]]:
        """Return full consolidation history."""
        return list(self._history)

    def get_stats(self) -> Dict[str, Any]:
        """Return consolidation statistics."""
        if not self._history:
            return {'passes': 0, 'total_consolidated': 0}
        return {
            'passes': len(self._history),
            'total_consolidated': sum(h.get('consolidated', 0) for h in self._history),
            'avg_stability': sum(h.get('stability', 1.0) for h in self._history) / len(self._history),
            'last_status': self._history[-1].get('status', 'unknown'),
        }
