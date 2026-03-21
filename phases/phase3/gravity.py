"""
GravitationalRelevance: Physics-based attention replacement.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Optional
from mass_tracker import MassTracker
from spatial_index import SpatialIndex

class GravitationalRelevance(nn.Module):
    def __init__(self, feature_dim: int = 512, k_neighbors: int = 32, base_mass: float = 1.0,
                 mass_growth_rate: float = 0.01, negative_mass_rate: float = 0.05, device: str = 'cuda'):
        super().__init__()
        self.feature_dim = feature_dim
        self.k_neighbors = k_neighbors
        self.base_mass = base_mass
        self.device = device
        self.mass_tracker = MassTracker(feature_dim=feature_dim, base_mass=base_mass, growth_rate=mass_growth_rate,
                                        negative_rate=negative_mass_rate, device=device)
        self.spatial_index = SpatialIndex(feature_dim=feature_dim, device=device)
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.G = nn.Parameter(torch.ones(1))

    def gravitational_weight(self, queries: Tensor, neighbor_keys: Tensor, neighbor_masses: Tensor) -> Tensor:
        n, k, d = neighbor_keys.shape
        q_exp = queries.unsqueeze(1).expand(n, k, d)
        cosine_sim = F.cosine_similarity(q_exp, neighbor_keys, dim=-1)
        distances = 1.0 - cosine_sim + 1e-8
        forces = self.G * neighbor_masses / (distances ** 2)
        positive_mask = forces > 0
        forces_masked = torch.where(positive_mask, forces, torch.zeros_like(forces))
        sum_forces = forces_masked.sum(dim=1, keepdim=True)
        all_repel = sum_forces < 1e-8
        weights = F.softmax(forces_masked, dim=1)
        if all_repel.any():
            uniform = torch.ones_like(weights) / k
            weights = torch.where(all_repel, uniform, weights)
        return weights

    def forward(self, queries: Tensor, context: Optional[Tensor] = None) -> Tensor:
        batched = queries.dim() == 3
        if batched:
            batch_size, seq_len, _ = queries.shape
            queries = queries.view(-1, self.feature_dim)
        else:
            seq_len = queries.shape[0]
        Q = self.query_proj(queries)
        if context is not None:
            if context.dim() == 3: context = context.view(-1, self.feature_dim)
            K = self.key_proj(torch.cat([queries, context], dim=0))
            V = self.value_proj(torch.cat([queries, context], dim=0))
        else:
            K = self.key_proj(queries)
            V = self.value_proj(queries)
        self.mass_tracker.observe(K.detach())
        K_masses = self.mass_tracker.lookup_masses(K)
        self.spatial_index.reset()
        self.spatial_index.add(K.detach())
        neighbor_indices = self.spatial_index.search(Q, k=self.k_neighbors)
        neighbor_keys = K[neighbor_indices]
        neighbor_values = V[neighbor_indices]
        neighbor_masses = K_masses[neighbor_indices]
        weights = self.gravitational_weight(Q, neighbor_keys, neighbor_masses)
        attended = (weights.unsqueeze(-1) * neighbor_values).sum(dim=1)
        output = self.out_proj(attended)
        if batched: output = output.view(batch_size, seq_len, self.feature_dim)
        return output

    def reinforce(self, concept_vector: Tensor, strength: float = 1.0):
        self.mass_tracker.reinforce(concept_vector, strength)
    def contradict(self, concept_vector: Tensor, strength: float = 1.0):
        self.mass_tracker.contradict(concept_vector, strength)
    def save_state(self) -> dict:
        return {'model_state': self.state_dict(), 'mass_state': self.mass_tracker.save_state(),
                'spatial_index_state': self.spatial_index.save_state(),
                'config': {'feature_dim': self.feature_dim, 'k_neighbors': self.k_neighbors, 'base_mass': self.base_mass}}
    @classmethod
    def load_state(cls, state: dict, device: str = 'cuda') -> 'GravitationalRelevance':
        gr = cls(**state['config'], device=device)
        gr.load_state_dict(state['model_state'])
        gr.mass_tracker.load_state(state['mass_state'])
        gr.spatial_index.load_state(state['spatial_index_state'])
        return gr
