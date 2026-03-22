"""
Temperature dynamics for thermodynamic settling.

Temperature controls how much the field can change:
    High temperature → field is fluid, learns quickly
    Low temperature  → field is stable, resists change
    Dynamic temperature → adjusts locally based on prediction error

Key insight: temperature is NOT global. Different regions of the field
can have different effective temperatures. Regions with high surprise
heat up (become plastic), while confident regions cool down (become rigid).
"""

import torch
from torch import Tensor
from typing import Dict, Optional
from dataclasses import dataclass, field


@dataclass
class TemperatureState:
    """Snapshot of temperature dynamics for logging/checkpointing."""
    global_temp: float
    min_temp: float
    max_temp: float
    decay_rate: float
    step_count: int
    recent_errors: list = field(default_factory=list)
    history: list = field(default_factory=list)


class TemperatureManager:
    """
    Manages temperature dynamics for thermodynamic settling.

    The temperature schedule combines:
    1. Global cooling: exponential decay from initial → min_temp
    2. Error-reactive heating: when prediction error spikes, temp rises locally
    3. Stability detection: if error is consistently low, cool faster

    Temperature directly controls:
    - Perturbation strength (how much a new sample changes the field)
    - Settle duration (high temp = more settling steps needed)
    - Acceptance threshold (high temp = accept worse states, like simulated annealing)
    """

    def __init__(
        self,
        initial: float = 1.0,
        min_temp: float = 0.01,
        max_temp: float = 2.0,
        decay: float = 0.999,
        error_sensitivity: float = 0.5,
        history_window: int = 50,
    ):
        """
        Args:
            initial:           starting temperature
            min_temp:          minimum temperature (never below this)
            max_temp:          maximum temperature (cap for error spikes)
            decay:             per-step multiplicative decay
            error_sensitivity: how much prediction error heats the system
            history_window:    number of recent errors to track for trend
        """
        self.current_temp = initial
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.decay = decay
        self.error_sensitivity = error_sensitivity
        self.history_window = history_window

        self._step_count = 0
        self._error_history: list = []
        self._temp_history: list = []

    def step(self, prediction_error: Optional[float] = None) -> float:
        """
        Advance one step, optionally reacting to prediction error.

        If prediction_error is provided:
            - High error → heat up (increase temp toward max)
            - Low error  → cool down (accelerate decay)
            - Error trend decreasing → additional cooling bonus

        Args:
            prediction_error: optional float ∈ [0, ∞). The surprise level.

        Returns:
            Current temperature after this step.
        """
        self._step_count += 1

        if prediction_error is not None:
            self._error_history.append(prediction_error)
            if len(self._error_history) > self.history_window:
                self._error_history = self._error_history[-self.history_window:]

            # React to error magnitude
            if prediction_error > 0.5:
                # High surprise → heat up (become plastic)
                heat = self.error_sensitivity * prediction_error
                self.current_temp = min(self.max_temp, self.current_temp + heat)
            elif prediction_error < 0.1 and len(self._error_history) >= 10:
                # Consistently low error → cool faster
                recent_mean = sum(self._error_history[-10:]) / 10
                if recent_mean < 0.15:
                    self.current_temp *= self.decay * 0.99  # Extra cooling
                else:
                    self.current_temp *= self.decay
            else:
                self.current_temp *= self.decay
        else:
            # No error info → just decay
            self.current_temp *= self.decay

        # Clamp to valid range
        self.current_temp = max(self.min_temp, min(self.max_temp, self.current_temp))

        self._temp_history.append(self.current_temp)
        if len(self._temp_history) > 1000:
            self._temp_history = self._temp_history[-500:]

        return self.current_temp

    @property
    def temperature(self) -> float:
        """Current temperature value."""
        return self.current_temp

    @property
    def is_hot(self) -> bool:
        """True if temperature is in the upper third of its range."""
        return self.current_temp > (self.max_temp - self.min_temp) * 0.66 + self.min_temp

    @property
    def is_cold(self) -> bool:
        """True if temperature is in the lower tenth of its range."""
        return self.current_temp < (self.max_temp - self.min_temp) * 0.1 + self.min_temp

    def error_trend(self) -> Optional[float]:
        """
        Compute error trend (negative = improving, positive = degrading).
        Returns None if insufficient history.
        """
        if len(self._error_history) < 10:
            return None
        first_half = sum(self._error_history[-10:-5]) / 5
        second_half = sum(self._error_history[-5:]) / 5
        return second_half - first_half

    def save_state(self) -> Dict:
        """Save temperature state for checkpointing."""
        return {
            'current_temp': self.current_temp,
            'min_temp': self.min_temp,
            'max_temp': self.max_temp,
            'decay': self.decay,
            'error_sensitivity': self.error_sensitivity,
            'step_count': self._step_count,
            'error_history': self._error_history[-self.history_window:],
            'temp_history': self._temp_history[-200:],
        }

    def load_state(self, state: Dict):
        """Load temperature state from checkpoint."""
        self.current_temp = state['current_temp']
        self.min_temp = state['min_temp']
        self.max_temp = state['max_temp']
        self.decay = state['decay']
        self.error_sensitivity = state.get('error_sensitivity', 0.5)
        self._step_count = state['step_count']
        self._error_history = state.get('error_history', [])
        self._temp_history = state.get('temp_history', [])

    def stats(self) -> Dict[str, float]:
        """Summary stats for logging."""
        result = {
            'temperature': self.current_temp,
            'step_count': self._step_count,
            'is_hot': self.is_hot,
            'is_cold': self.is_cold,
        }
        if self._error_history:
            result['recent_mean_error'] = sum(self._error_history[-10:]) / min(10, len(self._error_history))
            result['min_error_seen'] = min(self._error_history)
        trend = self.error_trend()
        if trend is not None:
            result['error_trend'] = trend
        return result
