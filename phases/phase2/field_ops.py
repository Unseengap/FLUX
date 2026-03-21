"""
Field operations: energy computation, field analysis, and utilities.

These functions operate on a ResonanceField to compute derived
quantities, analyze the field state, and provide diagnostic tools.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Any, List, Tuple


# ─────────────────────────────────────────────
# Energy Analysis
# ─────────────────────────────────────────────

def compute_local_energy(
    field: 'ResonanceField',
    h: int, w: int, d: int,
    radius: int = 3,
) -> float:
    """
    Compute local energy at a field location based on
    feature variance in the neighborhood.

    Low variance = coherent region = low local energy = attractor.
    High variance = disordered region = high energy = unstable.

    Args:
        field: the ResonanceField
        h, w, d: center coordinates
        radius: neighborhood radius
    Returns:
        Local energy (scalar)
    """
    hs = slice(max(0, h - radius), min(field.h, h + radius + 1))
    ws = slice(max(0, w - radius), min(field.w, w + radius + 1))
    ds = slice(max(0, d - radius), min(field.d, d + radius + 1))

    neighborhood = field.state[hs, ws, ds]  # [h_r, w_r, d_r, F]
    center = field.state[h, w, d].unsqueeze(0).unsqueeze(0).unsqueeze(0)

    # Variance from center
    diff = neighborhood - center
    variance = (diff ** 2).mean().item()
    return variance


def compute_energy_landscape_slice(
    field: 'ResonanceField',
    axis: str = 'd',
    index: int = None,
) -> Tensor:
    """
    Extract a 2D energy slice from the 3D field.
    Useful for visualization.

    Args:
        field: the ResonanceField
        axis: which axis to slice ('h', 'w', or 'd')
        index: which slice to take (default: middle)
    Returns:
        [dim1, dim2] energy values
    """
    energy = field.energy.squeeze(-1)  # [H, W, D]

    if axis == 'h':
        idx = index if index is not None else field.h // 2
        return energy[idx, :, :]
    elif axis == 'w':
        idx = index if index is not None else field.w // 2
        return energy[:, idx, :]
    else:  # 'd'
        idx = index if index is not None else field.d // 2
        return energy[:, :, idx]


def compute_mass_landscape_slice(
    field: 'ResonanceField',
    axis: str = 'd',
    index: int = None,
) -> Tensor:
    """
    Extract a 2D mass slice from the 3D field.

    Args:
        field: the ResonanceField
        axis: which axis to slice ('h', 'w', or 'd')
        index: which slice to take (default: middle)
    Returns:
        [dim1, dim2] mass values
    """
    mass = field.mass.squeeze(-1)  # [H, W, D]

    if axis == 'h':
        idx = index if index is not None else field.h // 2
        return mass[idx, :, :]
    elif axis == 'w':
        idx = index if index is not None else field.w // 2
        return mass[:, idx, :]
    else:
        idx = index if index is not None else field.d // 2
        return mass[:, :, idx]


# ─────────────────────────────────────────────
# Field Analysis
# ─────────────────────────────────────────────

def compute_field_statistics(field: 'ResonanceField') -> Dict[str, Any]:
    """
    Comprehensive statistics of the field state.
    Useful for monitoring training progress and field health.

    Returns:
        Dict with energy, mass, state, and attractor statistics.
    """
    with torch.no_grad():
        state = field.state
        energy = field.energy
        mass = field.mass

        # State statistics
        state_norms = state.norm(dim=-1)  # [H, W, D]
        active_mask = mass.squeeze(-1) > 0.01

        stats = {
            # Energy
            'total_energy': energy.sum().item(),
            'mean_energy': energy.mean().item(),
            'min_energy': energy.min().item(),
            'max_energy': energy.max().item(),
            'energy_std': energy.std().item(),

            # Mass
            'total_mass': mass.sum().item(),
            'mean_mass': mass.mean().item(),
            'max_mass': mass.max().item(),
            'active_locations': int(active_mask.sum().item()),
            'active_fraction': active_mask.float().mean().item(),

            # State
            'state_mean_norm': state_norms.mean().item(),
            'state_max_norm': state_norms.max().item(),
            'state_std': state.std().item(),

            # Attractors
            'num_attractors_01': field.num_attractors(0.1),
            'num_attractors_03': field.num_attractors(0.3),
            'num_attractors_05': field.num_attractors(0.5),

            # Steps
            'step_count': field.step_count,
        }

    return stats


def find_top_attractors(
    field: 'ResonanceField',
    k: int = 10,
    mass_threshold: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    Find the top-k strongest attractors by mass.

    Args:
        field: the ResonanceField
        k: how many to return
        mass_threshold: minimum mass to qualify
    Returns:
        List of dicts with location, mass, energy, feature_norm
    """
    mass = field.mass.squeeze(-1)  # [H, W, D]
    candidates = (mass > mass_threshold).nonzero(as_tuple=False)

    if candidates.shape[0] == 0:
        return []

    masses = mass[candidates[:, 0], candidates[:, 1], candidates[:, 2]]
    order = masses.argsort(descending=True)
    candidates = candidates[order[:min(k, len(order))]]

    results = []
    for idx in range(candidates.shape[0]):
        h, w, d = candidates[idx].tolist()
        results.append({
            'location': (h, w, d),
            'mass': field.mass[h, w, d].item(),
            'energy': field.energy[h, w, d].item(),
            'feature_norm': field.state[h, w, d].norm().item(),
        })

    return results


# ─────────────────────────────────────────────
# Field Normalization
# ─────────────────────────────────────────────

def normalize_field_energy(field: 'ResonanceField', target_mean: float = 1.0):
    """
    Rescale energy to prevent explosion or collapse.
    Call periodically during long training runs.

    Args:
        field: the ResonanceField to normalize
        target_mean: desired mean energy level
    """
    with torch.no_grad():
        current_mean = field.energy.mean()
        if current_mean > 0:
            scale = target_mean / current_mean
            field.energy *= scale
            field.energy.clamp_(min=0.01, max=2.0)


def normalize_field_state(field: 'ResonanceField', max_norm: float = 10.0):
    """
    Clip field state norms to prevent feature explosion.

    Args:
        field: the ResonanceField
        max_norm: maximum L2 norm per location
    """
    with torch.no_grad():
        norms = field.state.norm(dim=-1, keepdim=True)  # [H, W, D, 1]
        mask = norms > max_norm
        if mask.any():
            scale = torch.where(mask, max_norm / (norms + 1e-8), torch.ones_like(norms))
            field.state *= scale


# ─────────────────────────────────────────────
# Perturbation Analysis
# ─────────────────────────────────────────────

def measure_update_locality(
    field: 'ResonanceField',
    wave_vector: Tensor,
    sample_locations: List[Tuple[int, int, int]],
) -> Dict[str, float]:
    """
    Measure how local a perturbation is by recording state changes
    at distant locations.

    Args:
        field: the ResonanceField
        wave_vector: [wave_dim] the wave to perturb with
        sample_locations: list of (h, w, d) to monitor
    Returns:
        Dict with mean_change, max_change, and per-location changes
    """
    # Record state before
    before = {}
    for loc in sample_locations:
        h, w, d = loc
        before[loc] = field.state[h, w, d].detach().clone()

    # Apply perturbation
    field.perturb(wave_vector)

    # Measure changes
    changes = {}
    for loc in sample_locations:
        h, w, d = loc
        after = field.state[h, w, d].detach()
        change = (after - before[loc]).norm().item()
        changes[loc] = change

    change_values = list(changes.values())
    return {
        'mean_change': sum(change_values) / len(change_values) if change_values else 0.0,
        'max_change': max(change_values) if change_values else 0.0,
        'per_location': changes,
    }
