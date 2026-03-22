"""
Local energy computations for thermodynamic settling.

Energy functions define the "landscape" the system settles into.
Lower energy = better fit between the field state and new evidence.
All computations are LOCAL — no global gradient required.

Key functions:
    local_energy:       MSE between a field neighborhood and a target pattern
    coherence_energy:   How well neighbors agree with each other
    prediction_energy:  Surprise — how far the field's prediction is from reality
    total_local_energy: Combined energy of a neighborhood (what gets minimized)
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Tuple


# ─────────────────────────────────────────────
# Energy Components
# ─────────────────────────────────────────────

def local_energy(state: Tensor, target: Tensor) -> Tensor:
    """
    Reconstruction energy: how far the local state is from the target.

    Args:
        state:  [N, F] or [F] — local field features
        target: [N, F] or [F] — what the evidence says it should be

    Returns:
        Scalar energy (lower = better fit)
    """
    return F.mse_loss(state, target.expand_as(state))


def coherence_energy(neighborhood: Tensor) -> Tensor:
    """
    Coherence energy: how well neighbors agree with each other.
    High coherence (low variance) = low energy = stable attractor.

    Args:
        neighborhood: [H, W, D, F] or [N, F] — local field region

    Returns:
        Scalar energy (lower = more coherent)
    """
    if neighborhood.dim() > 2:
        flat = neighborhood.reshape(-1, neighborhood.shape[-1])
    else:
        flat = neighborhood
    return flat.var(dim=0).mean()


def prediction_energy(predicted: Tensor, actual: Tensor) -> Tensor:
    """
    Prediction energy: surprise — how wrong the field's prediction was.
    High surprise → high energy → system needs to change.

    Args:
        predicted: [F] or [N, F] — what the field expected
        actual:    [F] or [N, F] — what actually arrived

    Returns:
        Scalar energy (lower = less surprised)
    """
    return F.mse_loss(predicted, actual)


def mass_resistance_energy(mass: Tensor, delta: Tensor) -> Tensor:
    """
    Mass resistance: high-mass regions resist change.
    Changing an established attractor costs more energy.

    Args:
        mass:  [H, W, D, 1] or [N, 1] — evidence mass at each location
        delta: [H, W, D, F] or [N, F] — proposed change magnitude

    Returns:
        Scalar energy cost of making the proposed change
    """
    resistance = mass / (1.0 + mass)
    change_magnitude = delta.norm(dim=-1, keepdim=True)
    return (resistance * change_magnitude).mean()


def total_local_energy(
    state: Tensor,
    target: Tensor,
    neighborhood: Tensor,
    mass: Tensor,
    alpha_recon: float = 1.0,
    alpha_coherence: float = 0.1,
    alpha_resistance: float = 0.5,
) -> Tuple[Tensor, Dict[str, float]]:
    """
    Combined local energy — this is what thermodynamic settling minimizes.

    Args:
        state:        [N, F] — local field state at target locations
        target:       [N, F] or [F] — evidence target
        neighborhood: [H, W, D, F] — broader neighborhood for coherence
        mass:         [N, 1] — evidence mass at state locations
        alpha_recon:      weight for reconstruction energy
        alpha_coherence:  weight for coherence energy
        alpha_resistance: weight for mass resistance energy

    Returns:
        total: scalar total energy
        components: dict with individual energy values
    """
    e_recon = local_energy(state, target)
    e_coherence = coherence_energy(neighborhood)
    delta = target.expand_as(state) - state
    e_resistance = mass_resistance_energy(mass, delta)

    total = (alpha_recon * e_recon
             + alpha_coherence * e_coherence
             + alpha_resistance * e_resistance)

    components = {
        'reconstruction': e_recon.item(),
        'coherence': e_coherence.item(),
        'resistance': e_resistance.item(),
        'total': total.item(),
    }

    return total, components
