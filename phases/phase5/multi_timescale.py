"""
MultiTimescaleCoordinator: Coordinates fast and slow CGN nodes.

Multi-timescale operation is a key design principle:
- Fast nodes (time_const ≈ 0.01):  Surface syntax patterns
- Medium nodes (time_const ≈ 0.1): Semantic relationships
- Slow nodes (time_const ≈ 1.0):   Deep conceptual abstractions

All operate simultaneously — no layer separation needed.
Fast nodes respond immediately to new input, while slow nodes
integrate information over many steps to build stable deep attractors.

This mirrors human cognition: fast reflexes + slow reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from cgn import CausalGeometryNode, SignalTrace, CGN_CONFIG


@dataclass
class TimescaleResult:
    """Result of processing through multi-timescale coordination."""
    output: Tensor
    fast_output: Tensor
    medium_output: Tensor
    slow_output: Tensor
    fast_activation: float
    medium_activation: float
    slow_activation: float
    steps_used: int
    traces: List[SignalTrace]


class MultiTimescaleCoordinator(nn.Module):
    """
    Coordinates CGN nodes operating at different timescales.

    Architecture:
    - Fast layer:   16 nodes (time_const=0.01) — syntax, patterns
    - Medium layer: 8 nodes  (time_const=0.1)  — semantics
    - Slow layer:   4 nodes  (time_const=1.0)  — deep concepts

    Fast nodes update at full strength every step.
    Medium nodes accumulate over several steps.
    Slow nodes build stable representations over many steps.

    The final output blends all timescales with learned mixing weights.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        n_fast: int = 16,
        n_medium: int = 8,
        n_slow: int = 4,
    ):
        """
        Args:
            feature_dim: signal dimensionality
            n_fast:      number of fast nodes
            n_medium:    number of medium nodes
            n_slow:      number of slow nodes
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.n_fast = n_fast
        self.n_medium = n_medium
        self.n_slow = n_slow

        # Fast nodes (surface patterns, syntax)
        self.fast_nodes = nn.ModuleList([
            CausalGeometryNode(
                node_id=i,
                feature_dim=feature_dim,
                time_const=CGN_CONFIG['fast_time_const'],
                radius=0.5,
            )
            for i in range(n_fast)
        ])

        # Medium nodes (semantic relationships)
        self.medium_nodes = nn.ModuleList([
            CausalGeometryNode(
                node_id=n_fast + i,
                feature_dim=feature_dim,
                time_const=CGN_CONFIG['medium_time_const'],
                radius=1.0,
            )
            for i in range(n_medium)
        ])

        # Slow nodes (deep conceptual abstractions)
        self.slow_nodes = nn.ModuleList([
            CausalGeometryNode(
                node_id=n_fast + n_medium + i,
                feature_dim=feature_dim,
                time_const=CGN_CONFIG['slow_time_const'],
                radius=2.0,
            )
            for i in range(n_slow)
        ])

        # Learned timescale mixing weights
        self.mix_weights = nn.Parameter(torch.tensor([0.4, 0.35, 0.25]))

        # Running state buffers for accumulation
        self.register_buffer('_fast_state', torch.zeros(feature_dim))
        self.register_buffer('_medium_state', torch.zeros(feature_dim))
        self.register_buffer('_slow_state', torch.zeros(feature_dim))
        self.register_buffer('_step_count', torch.tensor(0))

    def forward(self, signal: Tensor, steps: int = 1) -> Tensor:
        """
        Process signal across multiple timescales.

        Fast nodes fully update every step.
        Medium nodes partially update (accumulated).
        Slow nodes barely change per step (highly accumulated).

        Args:
            signal: [feature_dim] — input signal
            steps:  number of processing steps

        Returns:
            [feature_dim] — blended multi-timescale output
        """
        current_state = signal

        for step in range(steps):
            # Fast nodes: full update each step
            fast_outputs = [node(current_state) for node in self.fast_nodes]
            fast_mean = torch.stack(fast_outputs).mean(dim=0)

            # Fast state updates immediately
            self._fast_state = 0.2 * self._fast_state + 0.8 * fast_mean

            # Medium nodes: moderate accumulation
            medium_outputs = [node(current_state) for node in self.medium_nodes]
            medium_mean = torch.stack(medium_outputs).mean(dim=0)
            self._medium_state = 0.7 * self._medium_state + 0.3 * medium_mean

            # Slow nodes: heavy accumulation
            slow_outputs = [node(current_state) for node in self.slow_nodes]
            slow_mean = torch.stack(slow_outputs).mean(dim=0)
            self._slow_state = 0.95 * self._slow_state + 0.05 * slow_mean

            # Blend for next iteration
            weights = F.softmax(self.mix_weights, dim=0)
            current_state = (
                weights[0] * self._fast_state +
                weights[1] * self._medium_state +
                weights[2] * self._slow_state
            )

            self._step_count += 1

        return current_state

    def forward_with_traces(
        self, signal: Tensor, steps: int = 1
    ) -> TimescaleResult:
        """
        Process signal and return detailed traces for causal reasoning.

        Args:
            signal: [feature_dim] — input signal
            steps:  number of processing steps

        Returns:
            TimescaleResult with per-timescale outputs and traces
        """
        all_traces: List[SignalTrace] = []
        current_state = signal

        for step in range(steps):
            # Fast nodes with traces
            fast_outputs = []
            for node in self.fast_nodes:
                out, trace = node.forward_with_trace(current_state)
                fast_outputs.append(out)
                if step == steps - 1:  # Only record traces from final step
                    all_traces.append(trace)
            fast_mean = torch.stack(fast_outputs).mean(dim=0)
            self._fast_state = 0.2 * self._fast_state + 0.8 * fast_mean

            # Medium nodes with traces
            medium_outputs = []
            for node in self.medium_nodes:
                out, trace = node.forward_with_trace(current_state)
                medium_outputs.append(out)
                if step == steps - 1:
                    all_traces.append(trace)
            medium_mean = torch.stack(medium_outputs).mean(dim=0)
            self._medium_state = 0.7 * self._medium_state + 0.3 * medium_mean

            # Slow nodes with traces
            slow_outputs = []
            for node in self.slow_nodes:
                out, trace = node.forward_with_trace(current_state)
                slow_outputs.append(out)
                if step == steps - 1:
                    all_traces.append(trace)
            slow_mean = torch.stack(slow_outputs).mean(dim=0)
            self._slow_state = 0.95 * self._slow_state + 0.05 * slow_mean

            weights = F.softmax(self.mix_weights, dim=0)
            current_state = (
                weights[0] * self._fast_state +
                weights[1] * self._medium_state +
                weights[2] * self._slow_state
            )

        return TimescaleResult(
            output=current_state,
            fast_output=self._fast_state.clone(),
            medium_output=self._medium_state.clone(),
            slow_output=self._slow_state.clone(),
            fast_activation=self._fast_state.abs().mean().item(),
            medium_activation=self._medium_state.abs().mean().item(),
            slow_activation=self._slow_state.abs().mean().item(),
            steps_used=steps,
            traces=all_traces,
        )

    def measure_timescale_separation(self, signal: Tensor, max_steps: int = 100) -> Dict[str, Any]:
        """
        Measure how many steps each timescale needs to significantly activate.

        The threshold is **per-timescale**: 80 % of each timescale's own
        converged (final) activation after *max_steps*.  This directly
        reflects the EMA time constant for each tier.

        Used by test_phase5_test2 to verify:
        - Fast nodes react in < 5 steps
        - Slow nodes take > 10 steps

        Args:
            signal: [feature_dim] — test signal
            max_steps: maximum steps to check

        Returns:
            Dict with activation curves and threshold-crossing steps
        """
        # Reset states
        self._fast_state.zero_()
        self._medium_state.zero_()
        self._slow_state.zero_()

        fast_activations: list[float] = []
        medium_activations: list[float] = []
        slow_activations: list[float] = []

        for step in range(max_steps):
            self.forward(signal, steps=1)

            fast_activations.append(self._fast_state.abs().mean().item())
            medium_activations.append(self._medium_state.abs().mean().item())
            slow_activations.append(self._slow_state.abs().mean().item())

        # Per-timescale threshold: 80 % of converged (final) value.
        # This measures how quickly each EMA reaches its steady state.
        fast_final = max(fast_activations[-1], 1e-12)
        medium_final = max(medium_activations[-1], 1e-12)
        slow_final = max(slow_activations[-1], 1e-12)

        frac = 0.8  # 80 % of convergence
        fast_thresh = fast_final * frac
        medium_thresh = medium_final * frac
        slow_thresh = slow_final * frac

        fast_crossed = next(
            (i + 1 for i, v in enumerate(fast_activations) if v >= fast_thresh),
            max_steps,
        )
        medium_crossed = next(
            (i + 1 for i, v in enumerate(medium_activations) if v >= medium_thresh),
            max_steps,
        )
        slow_crossed = next(
            (i + 1 for i, v in enumerate(slow_activations) if v >= slow_thresh),
            max_steps,
        )

        return {
            'fast_activations': fast_activations,
            'medium_activations': medium_activations,
            'slow_activations': slow_activations,
            'fast_steps_to_activate': fast_crossed,
            'medium_steps_to_activate': medium_crossed,
            'slow_steps_to_activate': slow_crossed,
            'threshold_fraction': frac,
            'fast_converged': fast_final,
            'medium_converged': medium_final,
            'slow_converged': slow_final,
        }

    def reset_states(self):
        """Reset all accumulated states to zero."""
        self._fast_state.zero_()
        self._medium_state.zero_()
        self._slow_state.zero_()
        self._step_count.zero_()

    def total_nodes(self) -> int:
        """Total number of CGN nodes across all timescales."""
        return self.n_fast + self.n_medium + self.n_slow

    def total_params(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    def save_state(self) -> Dict[str, Any]:
        """Save coordinator state for checkpointing."""
        return {
            'state_dict': self.state_dict(),
            'config': {
                'feature_dim': self.feature_dim,
                'n_fast': self.n_fast,
                'n_medium': self.n_medium,
                'n_slow': self.n_slow,
            },
            'step_count': self._step_count.item(),
        }

    def load_state(self, state: Dict[str, Any]):
        """Load coordinator state from checkpoint."""
        self.load_state_dict(state['state_dict'])

    def stats(self) -> Dict[str, Any]:
        """Summary stats for logging."""
        return {
            'total_nodes': self.total_nodes(),
            'fast_nodes': self.n_fast,
            'medium_nodes': self.n_medium,
            'slow_nodes': self.n_slow,
            'total_params': self.total_params(),
            'steps_processed': self._step_count.item(),
            'fast_activation': self._fast_state.abs().mean().item(),
            'medium_activation': self._medium_state.abs().mean().item(),
            'slow_activation': self._slow_state.abs().mean().item(),
            'mix_weights': F.softmax(self.mix_weights, dim=0).detach().tolist(),
        }
