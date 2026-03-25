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
    Compute local energy at a field location based on feature variance.

    Low variance = coherent region = low local energy = attractor.
    High variance = disordered region = high energy = unstable.
    """
    hs = slice(max(0, h - radius), min(field.h, h + radius + 1))
    ws = slice(max(0, w - radius), min(field.w, w + radius + 1))
    ds = slice(max(0, d - radius), min(field.d, d + radius + 1))

    neighborhood = field.state[hs, ws, ds]
    center = field.state[h, w, d].unsqueeze(0).unsqueeze(0).unsqueeze(0)

    diff = neighborhood - center
    variance = (diff ** 2).mean().item()
    return variance


def compute_energy_landscape_slice(
    field: 'ResonanceField',
    axis: str = 'd',
    index: int = None,
) -> Tensor:
    """Extract a 2D energy slice from the 3D field."""
    energy = field.energy.squeeze(-1)

    if axis == 'h':
        idx = index if index is not None else field.h // 2
        return energy[idx, :, :]
    elif axis == 'w':
        idx = index if index is not None else field.w // 2
        return energy[:, idx, :]
    else:
        idx = index if index is not None else field.d // 2
        return energy[:, :, idx]


def compute_mass_landscape_slice(
    field: 'ResonanceField',
    axis: str = 'd',
    index: int = None,
) -> Tensor:
    """Extract a 2D mass slice from the 3D field."""
    mass = field.mass.squeeze(-1)

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
    """Comprehensive statistics of the field state."""
    with torch.no_grad():
        state  = field.state
        energy = field.energy
        mass   = field.mass

        state_norms = state.norm(dim=-1)
        active_mask = mass.squeeze(-1) > 0.01

        return {
            'total_energy':      energy.sum().item(),
            'mean_energy':       energy.mean().item(),
            'min_energy':        energy.min().item(),
            'max_energy':        energy.max().item(),
            'energy_std':        energy.std().item(),
            'total_mass':        mass.sum().item(),
            'mean_mass':         mass.mean().item(),
            'max_mass':          mass.max().item(),
            'active_locations':  int(active_mask.sum().item()),
            'active_fraction':   active_mask.float().mean().item(),
            'state_mean_norm':   state_norms.mean().item(),
            'state_max_norm':    state_norms.max().item(),
            'state_std':         state.std().item(),
            'num_attractors_01': field.num_attractors(0.1),
            'num_attractors_03': field.num_attractors(0.3),
            'num_attractors_05': field.num_attractors(0.5),
            'step_count':        field.step_count,
        }


def find_top_attractors(
    field: 'ResonanceField',
    k: int = 10,
    mass_threshold: float = 0.1,
) -> List[Dict[str, Any]]:
    """Find the top-k strongest attractors by mass."""
    mass = field.mass.squeeze(-1)
    candidates = (mass > mass_threshold).nonzero(as_tuple=False)

    if candidates.shape[0] == 0:
        return []

    masses = mass[candidates[:, 0], candidates[:, 1], candidates[:, 2]]
    order  = masses.argsort(descending=True)
    candidates = candidates[order[:min(k, len(order))]]

    return [
        {
            'location':     (candidates[i][0].item(), candidates[i][1].item(), candidates[i][2].item()),
            'mass':         field.mass[candidates[i][0], candidates[i][1], candidates[i][2]].item(),
            'energy':       field.energy[candidates[i][0], candidates[i][1], candidates[i][2]].item(),
            'feature_norm': field.state[candidates[i][0], candidates[i][1], candidates[i][2]].norm().item(),
        }
        for i in range(candidates.shape[0])
    ]


# ─────────────────────────────────────────────
# Field Normalization
# ─────────────────────────────────────────────

def normalize_field_energy(field: 'ResonanceField', target_mean: float = 1.0):
    """Rescale energy to prevent explosion or collapse."""
    with torch.no_grad():
        current_mean = field.energy.mean()
        if current_mean > 0:
            field.energy *= target_mean / current_mean
            field.energy.clamp_(min=0.01, max=2.0)


def normalize_field_state(field: 'ResonanceField', max_norm: float = 10.0):
    """Clip field state norms to prevent feature explosion."""
    with torch.no_grad():
        norms = field.state.norm(dim=-1, keepdim=True)
        mask  = norms > max_norm
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
    at sample locations before and after the perturbation.
    """
    before = {
        loc: field.state[loc[0], loc[1], loc[2]].detach().clone()
        for loc in sample_locations
    }

    field.perturb(wave_vector)

    changes = {
        loc: (field.state[loc[0], loc[1], loc[2]].detach() - before[loc]).norm().item()
        for loc in sample_locations
    }

    vals = list(changes.values())
    return {
        'mean_change':   sum(vals) / len(vals) if vals else 0.0,
        'max_change':    max(vals) if vals else 0.0,
        'per_location':  changes,
    }
