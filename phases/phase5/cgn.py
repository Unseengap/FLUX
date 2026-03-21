import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from manifold import ManifoldPatch

class CausalGeometryNode(nn.Module):
    """
    Geometry-aware node that replaces standard neurons.
    Stores WHAT it knows and WHY it knows it.
    """
    def __init__(
        self,
        node_id: int,
        feature_dim: int = 512,
        time_const: float = 0.1,
        radius: float = 1.0
    ):
        super().__init__()
        self.node_id = node_id
        self.feature_dim = feature_dim
        self.time_const = time_const
        self.radius = radius

        # Geometric properties
        self.curvature = nn.Parameter(torch.randn(1, feature_dim) * 0.01)
        self.orientation = nn.Parameter(torch.randn(1, feature_dim))
        self.mass = nn.Parameter(torch.tensor(1.0))

        self.causal_why: Optional[Tensor] = None
        self.manifold = ManifoldPatch(feature_dim)

    def forward(self, signal: Tensor) -> Tensor:
        """
        Signal processing via geometric bending.
        """
        # Curvature bends the signal path
        # Ensure signal is 2D for apply_curvature
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)

        bent_signal = self.manifold.apply_curvature(signal, self.curvature, self.orientation)

        # Distance check within radius
        dist = torch.norm(bent_signal - self.orientation, dim=-1, keepdim=True)
        influence = torch.exp(-dist / self.radius)

        output = bent_signal * influence * self.mass
        return output.squeeze(0) if signal.size(0) == 1 and signal.dim() == 2 else output

    def set_causal_source(self, source_vector: Tensor):
        self.causal_why = source_vector.detach()
