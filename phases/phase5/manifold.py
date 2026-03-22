"""
ManifoldPatch: Geometric manifold operations for Causal Geometry Nodes.

Each CGN sits on a manifold patch — a local region of curved space.
Curvature bends signal paths, orientation defines sensitivity directions,
and geodesic steps model how signals propagate across the manifold.

Instead of output = activation(W @ input), signals flow through curved
geometry. The curvature IS the computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


class ManifoldPatch(nn.Module):
    """
    Implements geometric manifold patch operations for CGNs.

    Each patch is a local region of a curved manifold characterized by:
    - Metric tensor: defines local distances and angles
    - Curvature: bends signal paths (analogous to spacetime curvature)
    - Connection: how parallel transport works across the patch

    Signals entering the patch are bent by curvature and orientation,
    then flow naturally toward compatible attractors.
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim

        # Learnable local metric tensor (symmetric positive-definite via Cholesky)
        self.metric_L = nn.Parameter(torch.eye(feature_dim) * 0.1)

        # Connection coefficients (simplified Christoffel-like)
        self.connection = nn.Parameter(torch.randn(feature_dim, feature_dim) * 0.01)

    def apply_curvature(self, signal: Tensor, curvature: Tensor, orientation: Tensor) -> Tensor:
        """
        Bends the signal path based on node curvature and orientation.

        The signal is projected onto the orientation to measure alignment,
        then deflected proportionally to curvature. This replaces the
        traditional W @ x + b operation with geometric bending.

        Args:
            signal:      [batch, feature_dim] or [feature_dim] — incoming signal
            curvature:   [1, feature_dim] — local curvature tensor
            orientation: [1, feature_dim] — sensitivity direction

        Returns:
            [batch, feature_dim] — bent signal
        """
        # Project signal onto orientation to find alignment
        alignment = torch.matmul(signal, orientation.T)  # [batch, 1]
        # Bending force is proportional to curvature and alignment
        bending = curvature * alignment  # [batch, feature_dim]
        # Apply metric tensor for local distance correction
        metric = self.metric_L @ self.metric_L.T  # Ensure PD
        metric_correction = F.linear(signal, metric) * 0.01
        return signal + bending + metric_correction

    def compute_geodesic_step(
        self,
        position: Tensor,
        velocity: Tensor,
        curvature: Tensor,
        dt: float = 0.1,
    ) -> Tuple[Tensor, Tensor]:
        """
        Computes next step on the manifold surface using geodesic equation.

        In a Riemannian manifold, geodesics satisfy:
            d²x/dt² + Γ(dx/dt, dx/dt) = 0

        We use a simplified version with learnable connection coefficients
        that bend the velocity based on local curvature.

        Args:
            position:  [feature_dim] — current position on manifold
            velocity:  [feature_dim] — current velocity vector
            curvature: [1, feature_dim] — local curvature
            dt:        time step size

        Returns:
            new_position: [feature_dim] — updated position
            new_velocity: [feature_dim] — updated velocity
        """
        # Connection term (simplified Christoffel symbols)
        christoffel_force = F.linear(velocity, self.connection)
        curvature_force = curvature.squeeze(0) * velocity

        # Geodesic acceleration
        acceleration = -christoffel_force - curvature_force

        # Symplectic Euler integration (preserves energy better)
        new_velocity = velocity + acceleration * dt
        new_position = position + new_velocity * dt

        return new_position, new_velocity

    def parallel_transport(self, vector: Tensor, along: Tensor) -> Tensor:
        """
        Parallel transport a vector along a direction on the manifold.

        This preserves the "meaning" of a vector as it's moved across
        the manifold — crucial for comparing signals at different nodes.

        Args:
            vector: [feature_dim] — vector to transport
            along:  [feature_dim] — direction of transport

        Returns:
            [feature_dim] — transported vector
        """
        # Connection-based transport
        correction = F.linear(along, self.connection) * 0.01
        transported = vector + torch.cross(
            correction[:3].unsqueeze(0),
            vector[:3].unsqueeze(0),
            dim=-1,
        ).squeeze(0).new_zeros(vector.shape)
        # Simplified: small correction from connection
        transported = vector - F.linear(
            vector * along, self.connection
        ) * 0.01
        return transported

    def compute_sectional_curvature(self, v1: Tensor, v2: Tensor) -> Tensor:
        """
        Compute the sectional curvature of the manifold in the plane
        spanned by v1 and v2. Higher curvature = stronger signal bending.

        Args:
            v1: [feature_dim] — first basis vector
            v2: [feature_dim] — second basis vector

        Returns:
            Scalar curvature value
        """
        # Riemann curvature tensor approximation
        R_v1_v2 = F.linear(v1, self.connection) * F.linear(v2, self.connection).sum()
        R_v2_v1 = F.linear(v2, self.connection) * F.linear(v1, self.connection).sum()
        numerator = (R_v1_v2 - R_v2_v1).norm()
        denominator = (v1.norm() * v2.norm()).clamp(min=1e-8)
        return numerator / denominator
