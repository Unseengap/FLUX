"""
ContradictionRegistry: Tracks known contradictions as persistent tension.

When a contradiction is detected between two statements,
it is registered here so future sequences that touch either
statement inherit the tension — the model remembers
that this territory is disputed.

This is different from Phase 2's negative mass (which handles
evidence at the field level). Contradiction here is at the
semantic wave level — it is felt BEFORE field storage.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ContradictionRecord:
    """A registered pair of contradicting wave states."""
    statement_vector:     Tensor   # [608] CausalWave mean vector for statement
    contradiction_vector: Tensor   # [608] CausalWave mean vector for contradiction
    tension_strength:     float    # How severe the contradiction is (0 to 1)
    evidence_count:       int      # How many times this contradiction was observed
    first_seen:           int      # Training step when first detected
    description:          str      # Human-readable (for debugging only)


class ContradictionRegistry:
    """
    Maintains a record of all detected semantic contradictions.

    At inference time, when a new CausalWave arrives:
    1. Check if it is similar to any known contradicted statement
    2. If so, increase its tension vector
    3. This tension propagates to Phase 2 — the field feels it too

    This is how the model builds up a sense of disputed knowledge —
    regions of semantic space where it has seen conflicting evidence.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        max_records: int = 10000,
        device: str = 'cuda',
    ):
        self.threshold   = similarity_threshold
        self.max_records = max_records
        self.device      = device
        self.records: List[ContradictionRecord] = []

    def register(
        self,
        statement_vec: Tensor,
        contradiction_vec: Tensor,
        strength: float,
        description: str = '',
        step: int = 0,
    ):
        """Add a new contradiction pair to the registry."""
        # Check if already registered (merge if very similar)
        for rec in self.records:
            if self._is_duplicate(statement_vec, rec):
                rec.evidence_count += 1
                rec.tension_strength = max(rec.tension_strength, strength)
                return

        record = ContradictionRecord(
            statement_vector     = statement_vec.detach().cpu(),
            contradiction_vector = contradiction_vec.detach().cpu(),
            tension_strength     = strength,
            evidence_count       = 1,
            first_seen           = step,
            description          = description,
        )
        self.records.append(record)

        # Prune oldest low-evidence records if full
        if len(self.records) > self.max_records:
            self.records.sort(key=lambda r: r.evidence_count, reverse=True)
            self.records = self.records[:self.max_records]

    def query_tension(self, wave_vector: Tensor) -> float:
        """
        Check if a wave vector touches any known contradiction.
        Returns tension boost in [0, 1] — 0 if no contradiction found.
        """
        if not self.records:
            return 0.0

        v = F.normalize(wave_vector.cpu().float(), dim=-1)
        max_tension = 0.0

        for rec in self.records:
            stmt = F.normalize(rec.statement_vector.float(), dim=-1)
            sim  = F.cosine_similarity(v.unsqueeze(0), stmt.unsqueeze(0)).item()
            if sim > self.threshold:
                max_tension = max(max_tension, rec.tension_strength)

        return max_tension

    def _is_duplicate(
        self, vec: Tensor, rec: ContradictionRecord
    ) -> bool:
        v  = F.normalize(vec.cpu().float(), dim=-1)
        rv = F.normalize(rec.statement_vector.float(), dim=-1)
        return F.cosine_similarity(v.unsqueeze(0), rv.unsqueeze(0)).item() > 0.9

    def stats(self) -> dict:
        if not self.records:
            return {'count': 0}
        strengths = [r.tension_strength for r in self.records]
        return {
            'count': len(self.records),
            'mean_strength': sum(strengths) / len(strengths),
            'max_strength': max(strengths),
            'total_evidence': sum(r.evidence_count for r in self.records),
        }

    def save(self) -> dict:
        return {
            'records': [
                {
                    'statement_vector':     r.statement_vector,
                    'contradiction_vector': r.contradiction_vector,
                    'tension_strength':     r.tension_strength,
                    'evidence_count':       r.evidence_count,
                    'description':          r.description,
                }
                for r in self.records
            ],
            'config': {
                'similarity_threshold': self.threshold,
                'max_records':          self.max_records,
            }
        }

    def load(self, state: dict):
        cfg = state.get('config', {})
        self.threshold   = cfg.get('similarity_threshold', self.threshold)
        self.max_records = cfg.get('max_records', self.max_records)
        self.records = [
            ContradictionRecord(
                statement_vector     = r['statement_vector'],
                contradiction_vector = r['contradiction_vector'],
                tension_strength     = r['tension_strength'],
                evidence_count       = r['evidence_count'],
                first_seen           = 0,
                description          = r['description'],
            )
            for r in state['records']
        ]
