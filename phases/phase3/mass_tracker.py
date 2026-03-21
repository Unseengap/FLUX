"""
MassTracker: Tracks evidence mass for each concept the model encounters.
"""
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Optional
import numpy as np

class MassTracker:
    def __init__(
        self,
        feature_dim: int = 512,
        base_mass: float = 1.0,
        growth_rate: float = 0.01,
        negative_rate: float = 0.05,
        merge_threshold: float = 0.95,
        max_concepts: int = 100000,
        device: str = 'cuda'
    ):
        self.feature_dim = feature_dim
        self.base_mass = base_mass
        self.growth_rate = growth_rate
        self.negative_rate = negative_rate
        self.merge_threshold = merge_threshold
        self.max_concepts = max_concepts
        self.device = device
        self.concept_vectors = torch.zeros(max_concepts, feature_dim, device=device)
        self.masses = torch.full((max_concepts,), base_mass, device=device)
        self.count = 0

    def observe(self, keys: Tensor):
        for key in keys:
            self._observe_one(key)

    def _observe_one(self, key: Tensor):
        if self.count == 0:
            self._add_concept(key)
            return
        existing = self.concept_vectors[:self.count]
        similarities = F.cosine_similarity(key.unsqueeze(0), existing, dim=-1)
        max_sim, max_idx = similarities.max(dim=0)
        if max_sim.item() > self.merge_threshold:
            self.masses[max_idx] += self.growth_rate
        else:
            self._add_concept(key)

    def _add_concept(self, vector: Tensor):
        if self.count >= self.max_concepts:
            self._prune()
        idx = self.count
        self.concept_vectors[idx] = vector.detach()
        self.masses[idx] = self.base_mass
        self.count += 1

    def get_masses(self, indices: Tensor) -> Tensor:
        valid_indices = indices.clamp(0, max(0, self.count - 1))
        return self.masses[valid_indices]

    def lookup_masses(self, keys: Tensor) -> Tensor:
        if self.count == 0:
            return torch.full((len(keys),), self.base_mass, device=self.device)
        keys_norm = F.normalize(keys, dim=-1)
        existing_norm = F.normalize(self.concept_vectors[:self.count], dim=-1)
        sims = torch.matmul(keys_norm, existing_norm.t())
        max_sims, max_indices = sims.max(dim=1)
        masses = self.masses[max_indices]
        return torch.where(max_sims > self.merge_threshold, masses, torch.tensor(self.base_mass, device=self.device))

    def reinforce(self, concept: Tensor, strength: float = 1.0):
        if self.count == 0: return
        sims = F.cosine_similarity(concept.unsqueeze(0), self.concept_vectors[:self.count], dim=-1)
        self.masses[sims.argmax()] += self.growth_rate * strength

    def contradict(self, concept: Tensor, strength: float = 1.0):
        if self.count == 0: return
        sims = F.cosine_similarity(concept.unsqueeze(0), self.concept_vectors[:self.count], dim=-1)
        self.masses[sims.argmax()] -= self.negative_rate * strength

    def _prune(self, keep_fraction: float = 0.8):
        keep_n = int(self.max_concepts * keep_fraction)
        top_indices = self.masses[:self.count].topk(keep_n).indices
        self.concept_vectors[:keep_n] = self.concept_vectors[top_indices]
        self.masses[:keep_n] = self.masses[top_indices]
        self.count = keep_n

    def stats(self) -> dict:
        if self.count == 0: return {'count': 0}
        active_masses = self.masses[:self.count]
        return {
            'count': self.count,
            'mean_mass': active_masses.mean().item(),
            'max_mass': active_masses.max().item(),
            'min_mass': active_masses.min().item(),
            'negative_count': (active_masses < 0).sum().item(),
        }

    def save_state(self) -> dict:
        return {
            'concept_vectors': self.concept_vectors[:self.count].cpu(),
            'masses': self.masses[:self.count].cpu(),
            'count': self.count,
            'config': {
                'feature_dim': self.feature_dim,
                'base_mass': self.base_mass,
                'growth_rate': self.growth_rate,
                'negative_rate': self.negative_rate,
            }
        }

    def load_state(self, state: dict):
        self.count = state['count']
        self.concept_vectors[:self.count] = state['concept_vectors'].to(self.device)
        self.masses[:self.count] = state['masses'].to(self.device)
