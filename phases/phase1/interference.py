"""
Wave interference functions.
This is what makes CSE fundamentally different from embeddings.

Two semantic waves interact based on:
- Phase alignment (cosine similarity as proxy for phase difference)
- Distance (exponential decay with sequence distance)
- Amplitude (wave magnitude)

Constructive interference: similar meaning waves reinforce
Destructive interference: opposite meaning waves cancel
"""

import torch
import torch.nn.functional as F
from torch import Tensor
import math


# ─────────────────────────────────────────────
# Core Interference Functions
# ─────────────────────────────────────────────

def compute_phase_difference(w1: Tensor, w2: Tensor) -> Tensor:
    """
    Compute phase difference between two wave vectors.
    Returns angle in [0, π] — 0=aligned, π=opposite.

    Uses cosine similarity of the full vectors as a proxy for
    wave phase alignment. Returns a scalar phase per pair,
    broadcast to all dimensions.

    Args:
        w1: [..., dim] wave vector(s)
        w2: [..., dim] wave vector(s)
    Returns:
        [..., 1] phase difference in radians
    """
    cos_sim = F.cosine_similarity(w1, w2, dim=-1).unsqueeze(-1)
    cos_sim = cos_sim.clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.acos(cos_sim)


def decay_function(distance: int, decay_rate: float = 0.5) -> float:
    """
    Exponential decay of interference strength with distance.
    At distance 0: strength = 1.0
    At distance 4: strength ≈ 0.14

    Args:
        distance: integer distance between wave positions
        decay_rate: how quickly interference drops off
    Returns:
        float in [0, 1]
    """
    return math.exp(-decay_rate * distance)


def wave_interference(
    w1: Tensor,
    w2: Tensor,
    distance: int,
    decay_rate: float = 0.5
) -> Tensor:
    """
    Compute interference of w2 on w1 given their distance.

    Constructive: same phase → waves amplify each other.
    Destructive: opposite phase → waves cancel each other.

    Args:
        w1: [..., dim] primary wave (being affected)
        w2: [..., dim] secondary wave (affecting w1)
        distance: positions apart in sequence
        decay_rate: interference decay with distance
    Returns:
        [..., dim] w1 after w2 interference applied
    """
    cos_sim = F.cosine_similarity(w1, w2, dim=-1).unsqueeze(-1)
    decay = decay_function(distance, decay_rate)
    interference = cos_sim * decay * w2
    return w1 + interference


def apply_neighborhood_interference(
    waves: Tensor,
    radius: int = 4,
    scale: float = 0.1
) -> Tensor:
    """
    Apply interference from all waves within radius of each position.
    Vectorized over the sequence — loops only over offset distances.

    Each position accumulates constructive/destructive interference
    from neighbors. Interference is scaled down to prevent energy
    explosion.

    Args:
        waves: [seq_len, dim] all waves in sequence
        radius: how many positions each wave can affect
        scale: scaling factor for interference (prevents blowup)
    Returns:
        [seq_len, dim] waves after mutual interference
    """
    seq_len, dim = waves.shape
    interference = torch.zeros_like(waves)

    for offset in range(1, radius + 1):
        if offset >= seq_len:
            break

        decay = decay_function(offset)

        # Forward: position i receives interference from position i + offset
        w1_fwd = waves[:-offset]
        w2_fwd = waves[offset:]
        cos_fwd = F.cosine_similarity(w1_fwd, w2_fwd, dim=-1).unsqueeze(-1)
        interference[:-offset] += cos_fwd * decay * w2_fwd

        # Backward: position i receives interference from position i - offset
        w1_bwd = waves[offset:]
        w2_bwd = waves[:-offset]
        cos_bwd = F.cosine_similarity(w1_bwd, w2_bwd, dim=-1).unsqueeze(-1)
        interference[offset:] += cos_bwd * decay * w2_bwd

    return waves + scale * interference
