"""
ThermodynamicLearner: Replaces backpropagation with local energy settling.

Core idea: The system is always trying to reach minimum energy.
When input arrives, it creates a perturbation. The system settling into
a new minimum IS both the forward pass (producing output) and learning
(updating the field) simultaneously.

    Input → Perturbation → Field Settles → Output extracted from settled state
                         ↑
                         This settling = learning
                         No separate backward pass needed

No global gradients. No optimizer. No training loop. No epochs.
Learning happens through local physics — each sample changes
only its neighborhood, guided by temperature dynamics.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))
from temperature import TemperatureManager
from energy_functions import (
    local_energy, coherence_energy, prediction_energy,
    mass_resistance_energy, total_local_energy,
)


@dataclass
class SettleResult:
    """Result of a single settling operation."""
    initial_energy: float
    final_energy: float
    energy_delta: float
    iterations_used: int
    temperature: float
    prediction_error: float
    fact_stored: bool


class ThermodynamicLearner(nn.Module):
    """
    Replaces backpropagation with thermodynamic settling.

    Given a ResonanceField, this module:
    1. Takes a new input (wave vector + target)
    2. Computes prediction error (surprise)
    3. Adjusts temperature based on surprise
    4. Perturbs the field locally
    5. Settles the field to a new energy minimum
    6. The settled state IS the learned representation

    There is no loss.backward(). No optimizer.step(). No gradients.
    Learning = energy minimization through local physics.
    """

    def __init__(
        self,
        field: nn.Module,
        initial_temp: float = 1.0,
        min_temp: float = 0.01,
        max_temp: float = 2.0,
        decay: float = 0.999,
        settle_iterations: int = 10,
        settle_dt: float = 0.1,
        perturbation_radius: int = 4,
        error_sensitivity: float = 0.5,
    ):
        """
        Args:
            field:               ResonanceField instance (from Phase 2)
            initial_temp:        starting temperature
            min_temp:            floor temperature
            max_temp:            ceiling temperature
            decay:               per-step temperature decay
            settle_iterations:   default settling steps after perturbation
            settle_dt:           time step for settling dynamics
            perturbation_radius: spatial radius of influence
            error_sensitivity:   how much surprise heats the system
        """
        super().__init__()
        self.field = field
        self.settle_iterations = settle_iterations
        self.settle_dt = settle_dt
        self.perturbation_radius = perturbation_radius

        self.temp_manager = TemperatureManager(
            initial=initial_temp,
            min_temp=min_temp,
            max_temp=max_temp,
            decay=decay,
            error_sensitivity=error_sensitivity,
        )

        # Tracking
        self._total_samples = 0
        self._total_settles = 0
        self._energy_history: List[float] = []
        self._error_history: List[float] = []

    def compute_prediction(self, wave_vec: Tensor) -> Tensor:
        """
        Field's current prediction for a query: retrieve nearest features.

        Args:
            wave_vec: [wave_dim] — input wave (mean-pooled from CSE)

        Returns:
            [features] — field's current best guess
        """
        with torch.no_grad():
            features, sims, locs = self.field.query(wave_vec, k=4)
            # Similarity-weighted average of nearest neighbors
            weights = F.softmax(sims, dim=0)
            prediction = (weights.unsqueeze(-1) * features).sum(dim=0)
        return prediction

    def compute_surprise(self, wave_vec: Tensor, target: Tensor) -> float:
        """
        How surprised is the field by this input?
        Surprise = distance between prediction and actual target.

        Args:
            wave_vec: [wave_dim] — input wave
            target:   [features] — what the field feature SHOULD be

        Returns:
            Scalar surprise value ∈ [0, ∞)
        """
        prediction = self.compute_prediction(wave_vec)
        error = prediction_energy(prediction, target)
        return error.item()

    def settle_once(
        self,
        wave_vec: Tensor,
        target: Optional[Tensor] = None,
        iterations: Optional[int] = None,
    ) -> SettleResult:
        """
        Process ONE sample: perturb the field and settle to a new minimum.
        This is BOTH inference AND learning in a single operation.

        No gradients. No backward pass. Just physics.

        Args:
            wave_vec: [wave_dim] — input semantic wave (mean-pooled)
            target:   [features] — optional explicit target features.
                      If None, the field computes its own target from wave_vec.
            iterations: override settle iterations for this sample

        Returns:
            SettleResult with energy and convergence info
        """
        iterations = iterations or self.settle_iterations

        with torch.no_grad():
            # Step 1: Compute target features from wave
            if target is None:
                target = self.field.wave_to_feature(wave_vec).detach()

            # Step 2: Check field's current prediction (before learning)
            surprise = self.compute_surprise(wave_vec, target)

            # Step 3: Update temperature based on surprise
            temp = self.temp_manager.step(prediction_error=surprise)

            # Step 4: Compute initial energy
            location = self.field.wave_to_field_coords(wave_vec)
            hs, ws, ds = self.field._get_neighborhood_slices(location, self.perturbation_radius)
            neighborhood = self.field.state[hs, ws, ds]
            local_mass = self.field.mass[hs, ws, ds]
            flat_state = neighborhood.reshape(-1, neighborhood.shape[-1])
            flat_mass = local_mass.reshape(-1, 1)
            initial_e, _ = total_local_energy(flat_state, target, neighborhood, flat_mass)

            # Step 5: Perturb the field (strength modulated by temperature)
            perturbation_strength = temp * (1.0 + surprise)
            self.field.perturb(wave_vec, strength=perturbation_strength)

            # Step 6: Settle — let the field find a new energy minimum
            # More iterations when temperature is high (system is fluid)
            adaptive_iters = max(iterations, int(iterations * temp))
            self.field.settle(steps=adaptive_iters, dt=self.settle_dt)

            # Step 7: Compute final energy
            neighborhood_after = self.field.state[hs, ws, ds]
            flat_after = neighborhood_after.reshape(-1, neighborhood_after.shape[-1])
            final_e, _ = total_local_energy(flat_after, target, neighborhood_after, flat_mass)

            # Track
            self._total_samples += 1
            self._total_settles += adaptive_iters
            self._energy_history.append(final_e.item())
            self._error_history.append(surprise)

            fact_stored = final_e.item() < initial_e.item()

        return SettleResult(
            initial_energy=initial_e.item(),
            final_energy=final_e.item(),
            energy_delta=final_e.item() - initial_e.item(),
            iterations_used=adaptive_iters,
            temperature=temp,
            prediction_error=surprise,
            fact_stored=fact_stored,
        )

    def learn_stream(
        self,
        wave_target_pairs: List[Tuple[Tensor, Optional[Tensor]]],
        verbose: bool = False,
    ) -> List[SettleResult]:
        """
        Process a continuous stream of samples. No epochs. No batches.
        Each sample is settled independently, in order.

        Args:
            wave_target_pairs: list of (wave_vec, optional_target) tuples
            verbose: print progress every 10 samples

        Returns:
            List of SettleResults for each sample
        """
        results = []
        for i, (wave_vec, target) in enumerate(wave_target_pairs):
            result = self.settle_once(wave_vec, target)
            results.append(result)
            if verbose and (i + 1) % 10 == 0:
                recent = results[-10:]
                avg_energy = sum(r.final_energy for r in recent) / len(recent)
                avg_surprise = sum(r.prediction_error for r in recent) / len(recent)
                stored_count = sum(r.fact_stored for r in recent)
                temp = self.temp_manager.temperature
                print(f"    [{i+1:4d}] temp={temp:.4f}  energy={avg_energy:.4f}  "
                      f"surprise={avg_surprise:.4f}  stored={stored_count}/10")
        return results

    def retrieve(self, wave_vec: Tensor, k: int = 4) -> Tuple[Tensor, Tensor]:
        """
        Retrieve knowledge from the field — the "inference" half.
        (In thermodynamic learning, inference and learning are the same
        operation. This is a convenience for explicit query-only use.)

        Args:
            wave_vec: [wave_dim] — query wave
            k: number of neighbors to return

        Returns:
            features: [k, F] — retrieved field features
            similarities: [k] — cosine similarities
        """
        with torch.no_grad():
            features, sims, locs = self.field.query(wave_vec, k=k)
        return features, sims

    def save_state(self) -> Dict[str, Any]:
        """Save ThermodynamicLearner state for checkpointing."""
        return {
            'field_state': self.field.state_dict(),
            'temp_state': self.temp_manager.save_state(),
            'total_samples': self._total_samples,
            'total_settles': self._total_settles,
            'energy_history': self._energy_history[-200:],
            'error_history': self._error_history[-200:],
            'config': {
                'settle_iterations': self.settle_iterations,
                'settle_dt': self.settle_dt,
                'perturbation_radius': self.perturbation_radius,
            },
        }

    def load_state(self, state: Dict[str, Any]):
        """Load ThermodynamicLearner state from checkpoint."""
        self.field.load_state_dict(state['field_state'])
        self.temp_manager.load_state(state['temp_state'])
        self._total_samples = state.get('total_samples', 0)
        self._total_settles = state.get('total_settles', 0)
        self._energy_history = state.get('energy_history', [])
        self._error_history = state.get('error_history', [])

    def stats(self) -> Dict[str, Any]:
        """Summary stats for logging."""
        result = {
            'total_samples': self._total_samples,
            'total_settles': self._total_settles,
            'temperature': self.temp_manager.temperature,
        }
        if self._energy_history:
            result['recent_energy'] = sum(self._energy_history[-10:]) / min(10, len(self._energy_history))
            result['min_energy'] = min(self._energy_history)
        if self._error_history:
            result['recent_surprise'] = sum(self._error_history[-10:]) / min(10, len(self._error_history))
            result['min_surprise'] = min(self._error_history)
        result.update(self.temp_manager.stats())
        return result
