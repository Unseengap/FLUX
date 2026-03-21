import torch
import torch.nn as nn
from typing import List
from cgn import CausalGeometryNode

class MultiTimescaleCoordination(nn.Module):
    """
    Coordinates fast and slow nodes.
    Fast nodes: syntax, patterns.
    Slow nodes: deep abstractions, concepts.
    """
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.feature_dim = feature_dim

        # Fast nodes (surface patterns)
        self.fast_nodes = nn.ModuleList([
            CausalGeometryNode(i, feature_dim, time_const=0.01) for i in range(16)
        ])

        # Slow nodes (deep concepts)
        self.slow_nodes = nn.ModuleList([
            CausalGeometryNode(i + 16, feature_dim, time_const=1.0) for i in range(4)
        ])

    def forward(self, signal: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """
        Processes signal across multiple timescales.
        """
        current_state = signal

        # Fast nodes update every step
        for _ in range(steps):
            fast_out = torch.stack([node(current_state) for node in self.fast_nodes]).mean(dim=0)
            current_state = 0.9 * current_state + 0.1 * fast_out

        # Slow nodes update less frequently (or with smaller influence per step)
        slow_out = torch.stack([node(current_state) for node in self.slow_nodes]).mean(dim=0)
        final_state = 0.95 * current_state + 0.05 * slow_out

        return final_state
