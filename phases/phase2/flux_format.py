"""
FLUX model file format (.flx).

Unlike .pt files which are just weight dicts,
.flx files capture the full living state of the model:
- Field tensor (the "knowledge landscape")
- Attractor catalog (what stable things are known)
- Causal graph (why things are known) — placeholder until Phase 5
- Learning state (how many steps, temperature, etc.)

A .flx file can be loaded and the model continues from EXACTLY
where it left off — including mid-learning. There is no distinction
between "trained model" and "model in training" in FLUX.
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


def save_flux(
    field: 'ResonanceField',
    cse: 'ContinuousSemanticEncoder',
    path: str,
    attractor_catalog: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Save full FLUX state to .flx file.

    The .flx format captures the ENTIRE model state so that
    learning can continue seamlessly from where it left off.

    Args:
        field: the ResonanceField
        cse: the ContinuousSemanticEncoder (from Phase 1)
        path: output file path (should end in .flx)
        attractor_catalog: optional AttractorCatalog to save
        metadata: optional dict of extra metadata
    """
    if metadata is None:
        metadata = {}

    state = {
        # Format identification
        'format': 'FLUX',
        'version': '0.2',
        'phase': 2,

        # Component states
        'field_state': field.state_dict(),
        'cse_state': cse.state_dict(),

        # Field knowledge (not in state_dict because they're register_buffers
        # which ARE in state_dict, but we also save mass separately for clarity)
        'field_mass': field.mass.cpu(),
        'field_step_count': field.step_count,

        # Attractor catalog
        'attractor_catalog': (
            attractor_catalog.to_dict() if attractor_catalog else []
        ),

        # Causal graph placeholder (Phase 5)
        'causal_graph': {},

        # Learning metadata
        'learning_steps': metadata.get('steps', field.step_count),
        'can_continue_learning': True,  # Always true in FLUX
        'timestamp': datetime.now().isoformat(),

        # Configuration (enough to reconstruct)
        'field_config': {
            'h': field.h,
            'w': field.w,
            'd': field.d,
            'features': field.features,
            'wave_dim': field.wave_dim,
        },
        'cse_config': {
            'wave_dims': cse.wave_dims,
            'byte_window': cse.byte_window,
            'byte_stride': cse.byte_stride,
            'interference_radius': cse.interference_radius,
        },

        # Extra metadata
        'metadata': metadata,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))

    size_mb = path.stat().st_size / 1e6
    print(f"✓ FLUX model saved: {path}")
    print(f"  Format: FLUX v{state['version']}")
    print(f"  File size: {size_mb:.1f} MB")
    print(f"  Learning steps: {state['learning_steps']}")
    print(f"  Can continue learning: {state['can_continue_learning']}")


def load_flux(path: str) -> Dict[str, Any]:
    """
    Load FLUX state from .flx file.
    Returns dict with all components ready to initialize.

    Use the returned config dicts to reconstruct models:
        cse = ContinuousSemanticEncoder(**state['cse_config'])
        cse.load_state_dict(state['cse_state'])
        field = ResonanceField(**state['field_config'])
        field.load_state_dict(state['field_state'])

    Args:
        path: path to .flx file
    Returns:
        Dict with all saved state
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"FLUX model not found: {path}\n"
            f"Run training first to create the model file."
        )

    state = torch.load(str(path), map_location='cpu')

    assert state.get('format') == 'FLUX', (
        f"Not a FLUX model file — got format: {state.get('format')}"
    )

    print(f"✓ FLUX model loaded: {path}")
    print(f"  Version: {state['version']}")
    print(f"  Phase: {state['phase']}")
    print(f"  Learning steps: {state['learning_steps']}")
    print(f"  Can continue: {state['can_continue_learning']}")

    size_mb = path.stat().st_size / 1e6
    print(f"  File size: {size_mb:.1f} MB")

    return state


def verify_flux_integrity(path: str) -> bool:
    """
    Verify that a .flx file is complete and uncorrupted.

    Checks:
    - File loads without error
    - Format field is 'FLUX'
    - All required keys present
    - State dicts are non-empty

    Args:
        path: path to .flx file
    Returns:
        True if file passes all checks
    """
    required_keys = [
        'format', 'version', 'phase',
        'field_state', 'cse_state',
        'field_config', 'cse_config',
    ]

    try:
        state = torch.load(str(path), map_location='cpu')

        if state.get('format') != 'FLUX':
            print(f"  ✗ Wrong format: {state.get('format')}")
            return False

        for key in required_keys:
            if key not in state:
                print(f"  ✗ Missing key: {key}")
                return False

        if len(state['field_state']) == 0:
            print("  ✗ Empty field state dict")
            return False

        if len(state['cse_state']) == 0:
            print("  ✗ Empty CSE state dict")
            return False

        print(f"  ✓ FLUX file integrity verified: {path}")
        return True

    except Exception as e:
        print(f"  ✗ FLUX file corrupt or unreadable: {e}")
        return False
