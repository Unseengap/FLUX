"""
Phase 9 — ThermodynamicWaveSampler: Confidence-Adaptive Wave Sampling

Adapts thermodynamic temperature dynamics for wave-level generation.
Uses the WaveGenerator's confidence head + wave entropy to adaptively
control exploration. Same physics as Phase 4's TemperatureManager
but applied to wave generation.

Non-parametric — no learnable weights. Pure physics.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional


class ThermodynamicWaveSampler:
    """
    Adaptive sampling at the wave level using thermodynamic principles.

    High confidence → low temperature → stay close to predicted wave
    Low confidence → high temperature → add noise to explore alternatives
    Entropy spike → heat spike → escape stuck generation

    Non-parametric — no learnable weights. Pure physics.

    Args:
        base_noise: Base noise level (default 0.1)
        min_noise: Minimum noise floor (default 0.01)
        max_noise: Maximum noise cap (default 0.5)
        momentum: Thermal inertia — smooths noise changes (default 0.7)
    """

    def __init__(
        self,
        base_noise: float = 0.1,
        min_noise: float = 0.01,
        max_noise: float = 0.5,
        momentum: float = 0.7,
    ):
        self.base_noise = base_noise
        self.min_noise = min_noise
        self.max_noise = max_noise
        self.momentum = momentum
        self._current_noise = base_noise
        self._history: List[float] = []

    def sample_wave(
        self,
        predicted_wave: torch.Tensor,
        confidence: float,
    ) -> torch.Tensor:
        """
        Add thermodynamic noise to the predicted wave based on confidence.

        Args:
            predicted_wave: [432] the WaveGenerator's prediction
            confidence: [0, 1] how certain the generator is

        Returns:
            [432] sampled wave (prediction + calibrated noise)
        """
        # Target noise inversely proportional to confidence
        target_noise = self.base_noise * (1.0 - confidence)

        # Apply momentum (thermal inertia)
        self._current_noise = (
            self.momentum * self._current_noise
            + (1.0 - self.momentum) * target_noise
        )
        self._current_noise = max(
            self.min_noise, min(self.max_noise, self._current_noise)
        )
        self._history.append(self._current_noise)

        # Add Gaussian noise scaled by current noise level
        noise = torch.randn_like(predicted_wave) * self._current_noise
        sampled = predicted_wave + noise

        # Normalize to match wave distribution (preserve original norm)
        wave_norm = predicted_wave.norm()
        if wave_norm > 1e-8:
            sampled = F.normalize(sampled, dim=-1) * wave_norm

        return sampled

    def get_history(self) -> List[float]:
        """Return the noise level history for visualization."""
        return list(self._history)

    def get_current_noise(self) -> float:
        """Return the current noise level."""
        return self._current_noise

    def reset(self):
        """Reset internal state for a new generation session."""
        self._current_noise = self.base_noise
        self._history = []
