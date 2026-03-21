"""
NegativeMass: Contradiction detection and repulsion mechanism.
"""
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple

class ContradictionDetector:
    def __init__(self, contradiction_threshold: float = -0.3, min_evidence_strength: float = 0.5, device: str = 'cuda'):
        self.contradiction_threshold = contradiction_threshold
        self.min_evidence_strength = min_evidence_strength
        self.device = device
    def check(self, new_concept: Tensor, existing_concept: Tensor, evidence_strength: float = 1.0) -> Tuple[bool, float]:
        if evidence_strength < self.min_evidence_strength: return False, 0.0
        similarity = F.cosine_similarity(new_concept.unsqueeze(0), existing_concept.unsqueeze(0)).item()
        is_contradiction = similarity < self.contradiction_threshold
        strength = abs(similarity) * evidence_strength if is_contradiction else 0.0
        return is_contradiction, strength
    def find_contradicted_concepts(self, new_concept: Tensor, concept_registry: Tensor, masses: Tensor) -> Tensor:
        if len(concept_registry) == 0: return torch.zeros(0, device=self.device)
        new_expanded = new_concept.unsqueeze(0).expand_as(concept_registry)
        similarities = F.cosine_similarity(new_expanded, concept_registry, dim=-1)
        positive_mask = masses > 0
        contradiction_mask = (similarities < self.contradiction_threshold) & positive_mask
        strengths = torch.where(contradiction_mask, -similarities * 0.1, torch.zeros_like(similarities))
        return strengths
