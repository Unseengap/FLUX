"""
CausalGeometryNode: Replaces neurons with geometry-aware causal nodes.

Standard neurons store: "input X maps to output Y"
CGN stores: "input X maps to output Y BECAUSE of causal relationship Z"

Each CGN is not a scalar but a geometric patch characterized by:
- Curvature:   How strongly it bends incoming signals
- Orientation: What directions it is sensitive to
- Radius:      Breadth of its influence
- Causal Why:  Pointer to what caused this node to form
- Time Const:  How fast this node reacts (fast=surface, slow=deep)
- Mass:        Evidence weight (same as GR mass from Phase 3)

Geometry as computation:
    Signal enters the manifold
    → CGN curvature bends the signal path
    → Signal naturally flows toward compatible attractors
    → Output emerges from where signal comes to rest
    → No explicit multiplication — geometry IS the computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field

try:
    from phases.phase5.manifold import ManifoldPatch
except ImportError:
    from manifold import ManifoldPatch


# ─────────────────────────────────────────────
# CGN Configuration
# ─────────────────────────────────────────────

CGN_CONFIG = {
    'num_nodes': 4096,
    'feature_dim': 512,
    'fast_time_const': 0.01,
    'medium_time_const': 0.1,
    'slow_time_const': 1.0,
    'default_radius': 1.0,
}


@dataclass
class CausalArrow:
    """A directed causal link from source to this node."""
    source_id: int
    weight: float
    reason: str = ""


@dataclass
class CGNState:
    """Snapshot of a CGN's internal state for checkpointing."""
    node_id: int
    curvature: Tensor
    orientation: Tensor
    mass: float
    time_const: float
    radius: float
    causal_arrows: List[CausalArrow]
    activation_count: int


@dataclass
class SignalTrace:
    """Trace of how a signal was processed through a CGN."""
    node_id: int
    input_signal: Tensor
    output_signal: Tensor
    bending_magnitude: float
    influence_strength: float
    causal_why: Optional[Tensor]
    causal_arrows: List[CausalArrow]


class CausalGeometryNode(nn.Module):
    """
    Geometry-aware node that replaces standard neurons.
    Stores WHAT it knows and WHY it knows it.

    Instead of output = activation(W @ input + b), a CGN:
    1. Bends the input signal via manifold curvature
    2. Gates by distance-based influence within its radius
    3. Weights by evidence mass
    4. Stores causal provenance (WHY this mapping exists)
    """

    def __init__(
        self,
        node_id: int,
        feature_dim: int = 512,
        time_const: float = 0.1,
        radius: float = 1.0,
    ):
        """
        Args:
            node_id:     unique identifier for this node
            feature_dim: dimensionality of signal space
            time_const:  reaction speed (0.01=fast/syntax, 1.0=slow/concept)
            radius:      spatial extent of influence
        """
        super().__init__()
        self.node_id = node_id
        self.feature_dim = feature_dim
        self.time_const = time_const
        self.radius = radius

        # Geometric properties (learnable)
        self.curvature = nn.Parameter(torch.randn(1, feature_dim) * 0.01)
        self.orientation = nn.Parameter(torch.randn(1, feature_dim))
        self.mass = nn.Parameter(torch.tensor(1.0))

        # Internal state (not gradient-updated)
        self.register_buffer('activation_energy', torch.tensor(0.0))
        self.register_buffer('_activation_count', torch.tensor(0))

        # Causal provenance
        self.causal_why: Optional[Tensor] = None
        self._causal_arrows: List[CausalArrow] = []

        # Manifold geometry
        self.manifold = ManifoldPatch(feature_dim)

    def forward(self, signal: Tensor) -> Tensor:
        """
        Process a signal through geometric bending.

        1. Curvature bends the signal path
        2. Distance check gates influence by radius
        3. Mass scales output by evidence weight

        Args:
            signal: [feature_dim] or [batch, feature_dim]

        Returns:
            [feature_dim] or [batch, feature_dim] — bent signal
        """
        was_1d = signal.dim() == 1
        if was_1d:
            signal = signal.unsqueeze(0)

        # Step 1: Manifold curvature bends the signal path
        bent_signal = self.manifold.apply_curvature(
            signal, self.curvature, self.orientation
        )

        # Step 2: Distance-based influence gating
        # Normalize by sqrt(feature_dim) to avoid curse-of-dimensionality:
        # Raw L2 distance in 512-d ≈ 22, making exp(-22) ≈ 0.
        dist = torch.norm(bent_signal - self.orientation, dim=-1, keepdim=True)
        dist = dist / (self.feature_dim ** 0.5)
        influence = torch.exp(-dist / max(self.radius, 1e-6))

        # Step 3: Mass-weighted output
        output = bent_signal * influence * self.mass

        # Note: temporal gating is handled by MultiTimescaleCoordinator EMA,
        # not at the node level.  time_const remains as a descriptor used
        # by the coordinator to categorise nodes into fast/medium/slow.

        # Track activation
        with torch.no_grad():
            self.activation_energy = output.abs().mean()
            self._activation_count += 1

        return output.squeeze(0) if was_1d else output

    def forward_with_trace(self, signal: Tensor) -> Tuple[Tensor, SignalTrace]:
        """
        Process signal and return a full trace of the computation.
        Used for causal reasoning — tracing WHY a conclusion was reached.

        Args:
            signal: [feature_dim] — input signal

        Returns:
            output: [feature_dim] — processed signal
            trace:  SignalTrace with full provenance info
        """
        was_1d = signal.dim() == 1
        if was_1d:
            signal_2d = signal.unsqueeze(0)
        else:
            signal_2d = signal

        bent_signal = self.manifold.apply_curvature(
            signal_2d, self.curvature, self.orientation
        )
        dist = torch.norm(bent_signal - self.orientation, dim=-1, keepdim=True)
        dist = dist / (self.feature_dim ** 0.5)
        influence = torch.exp(-dist / max(self.radius, 1e-6))
        output = bent_signal * influence * self.mass

        if was_1d:
            output = output.squeeze(0)

        trace = SignalTrace(
            node_id=self.node_id,
            input_signal=signal.detach().clone(),
            output_signal=output.detach().clone(),
            bending_magnitude=torch.norm(bent_signal.squeeze(0) - signal_2d.squeeze(0)).item(),
            influence_strength=influence.mean().item(),
            causal_why=self.causal_why.clone() if self.causal_why is not None else None,
            causal_arrows=list(self._causal_arrows),
        )

        return output, trace

    def set_causal_source(self, source_vector: Tensor, source_id: int = -1, reason: str = ""):
        """
        Record WHY this node was formed / activated.

        Args:
            source_vector: the evidence that caused this node
            source_id:     ID of the source node (for graph tracing)
            reason:        human-readable explanation
        """
        self.causal_why = source_vector.detach()
        if source_id >= 0:
            self._causal_arrows.append(
                CausalArrow(source_id=source_id, weight=1.0, reason=reason)
            )

    def invalidate_cause(self, source_id: int) -> bool:
        """
        Invalidate a causal source. If this removes all evidence,
        the node's mass decays (it becomes less influential).

        Args:
            source_id: the source node ID to invalidate

        Returns:
            True if the node's mass was reduced
        """
        invalidated = False
        for arrow in self._causal_arrows:
            if arrow.source_id == source_id:
                arrow.weight = -1.0  # Repulsion
                invalidated = True

        if invalidated:
            # Reduce mass proportionally to invalidated evidence
            valid_arrows = [a for a in self._causal_arrows if a.weight > 0]
            if len(valid_arrows) == 0:
                with torch.no_grad():
                    self.mass.mul_(0.1)  # Heavy decay if no valid causes remain
            else:
                ratio = len(valid_arrows) / max(len(self._causal_arrows), 1)
                with torch.no_grad():
                    self.mass.mul_(ratio)

        return invalidated

    def get_causal_chain(self) -> List[CausalArrow]:
        """Get all causal arrows pointing to this node."""
        return list(self._causal_arrows)

    def save_state(self) -> Dict[str, Any]:
        """Save node state for checkpointing."""
        return {
            'node_id': self.node_id,
            'state_dict': self.state_dict(),
            'time_const': self.time_const,
            'radius': self.radius,
            'causal_arrows': [
                {'source_id': a.source_id, 'weight': a.weight, 'reason': a.reason}
                for a in self._causal_arrows
            ],
            'activation_count': self._activation_count.item(),
            'causal_why': self.causal_why.cpu() if self.causal_why is not None else None,
        }

    def load_state(self, state: Dict[str, Any]):
        """Load node state from checkpoint."""
        self.load_state_dict(state['state_dict'])
        self.time_const = state['time_const']
        self.radius = state['radius']
        self._causal_arrows = [
            CausalArrow(**a) for a in state.get('causal_arrows', [])
        ]
        if state.get('causal_why') is not None:
            self.causal_why = state['causal_why']
