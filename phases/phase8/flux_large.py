"""
Phase 8: FLUXLarge — Scaled FLUX Configuration for GPT-2 Benchmark

Scales the Phase 7 FLUXModel to approximately GPT-2 small (117M) parameter
count. Uses wider wave dimensions, larger field, and more CGN nodes.

The scaling philosophy follows the physics paradigm:
- Wider waves = richer semantic representation
- Larger field = more knowledge capacity
- More CGN nodes = finer-grained causal reasoning
- Deeper working memory window = longer effective context
"""

import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# ─────────────────────────────────────────────
# Path setup for cross-phase imports
# ─────────────────────────────────────────────
_PHASES_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PHASES_DIR.parent

for _phase in ['phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'phase6', 'phase7']:
    _p = str(_PHASES_DIR / _phase)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cse import ContinuousSemanticEncoder
from wave_types import SemanticWave
from field import ResonanceField
from gravity import GravitationalRelevance
from thermodynamic import ThermodynamicLearner
from cgn import CausalGeometryNode
from multi_timescale import MultiTimescaleCoordinator
from causal_graph import CausalGraph
from working_memory import WorkingMemory
from episodic_memory import EpisodicMemory
from semantic_memory import SemanticMemory
from memory_router import MemoryRouter
from consolidation import ConsolidationProcess
from flux_model import FLUXModel, OutputHead, FLUXResponse, FLUXStats
from flux_utils import (
    get_device, load_checkpoint, save_checkpoint, checkpoint_exists,
    PhaseLogger,
)


# ─────────────────────────────────────────────
# Scaled Wave Dimensions
# ─────────────────────────────────────────────
WAVE_DIMS_LARGE = {
    'phonetic':  96,
    'syntactic': 96,
    'semantic':  384,
    'temporal':  128,
    'intensity': 64,
}
TOTAL_WAVE_DIM_LARGE = sum(WAVE_DIMS_LARGE.values())  # 768

# ─────────────────────────────────────────────
# Scaled Field Dimensions
# ─────────────────────────────────────────────
FIELD_H_LARGE = 96
FIELD_W_LARGE = 96
FIELD_D_LARGE = 96
FIELD_FEATURES_LARGE = 768


# ─────────────────────────────────────────────
# FLUXLarge Configuration
# ─────────────────────────────────────────────
FLUX_LARGE_CONFIG = {
    # CSE (Phase 1) — scaled
    'wave_dim': TOTAL_WAVE_DIM_LARGE,      # 768
    'byte_window': 8,
    'byte_stride': 1,
    'interference_radius': 6,               # wider interference

    # Field (Phase 2) — scaled
    'field_h': FIELD_H_LARGE,               # 96
    'field_w': FIELD_W_LARGE,               # 96
    'field_d': FIELD_D_LARGE,               # 96
    'field_features': FIELD_FEATURES_LARGE,  # 768

    # GR (Phase 3) — scaled
    'gr_k_neighbors': 64,
    'gr_base_mass': 1.0,
    'gr_mass_growth': 0.005,

    # TL (Phase 4) — scaled
    'tl_initial_temp': 1.0,
    'tl_min_temp': 0.005,
    'tl_decay': 0.9999,
    'tl_settle_iterations': 15,

    # CGN (Phase 5) — scaled
    'cgn_feature_dim': FIELD_FEATURES_LARGE,
    'cgn_n_fast': 32,
    'cgn_n_medium': 16,
    'cgn_n_slow': 8,

    # Memory (Phase 6) — scaled
    'wm_window_size': 2048,
    'feature_dim': 512,
    'episodic_dim': 512,
    'consolidation_min_access': 3,
    'consolidation_temperature': 0.05,

    # Generation
    'output_vocab_size': 256,               # byte-level output
    'max_gen_length': 500,
    'temperature': 0.8,
}


@dataclass
class ScaleProfile:
    """Comparison profile between FLUX base and large."""
    name: str
    total_params: int
    wave_dim: int
    field_dims: Tuple[int, int, int]
    field_features: int
    cgn_nodes: int
    wm_window: int


class FLUXLarge(FLUXModel):
    """
    Scaled FLUX model targeting GPT-2 small (~117M) parameter count.

    Inherits from FLUXModel (Phase 7) and overrides configuration
    to use wider waves, larger field, and more nodes.

    Key scaling changes:
    - wave_dim: 432 → 768 (match GPT-2 embed dimension)
    - field: 64³ → 96³ (more knowledge capacity)
    - field_features: 512 → 768
    - CGN nodes: 28 → 56 (finer causal reasoning)
    - Working memory: 512 → 2048 (longer context)

    Usage:
        model = FLUXLarge(device='cuda')
        model = FLUXLarge.from_phase7_checkpoint(device='cuda')
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, device: str = 'cpu'):
        merged = {**FLUX_LARGE_CONFIG, **(config or {})}
        super().__init__(config=merged, device=device)

    def get_scale_profile(self) -> ScaleProfile:
        """Return a profile describing this model's scale."""
        stats = self.get_stats()
        cfg = self.config
        return ScaleProfile(
            name='FLUXLarge',
            total_params=stats.total_params,
            wave_dim=cfg['wave_dim'],
            field_dims=(cfg['field_h'], cfg['field_w'], cfg['field_d']),
            field_features=cfg['field_features'],
            cgn_nodes=cfg['cgn_n_fast'] + cfg['cgn_n_medium'] + cfg['cgn_n_slow'],
            wm_window=cfg['wm_window_size'],
        )

    @classmethod
    def from_phase7_checkpoint(cls, device: str = 'cpu',
                               config: Optional[Dict[str, Any]] = None) -> 'FLUXLarge':
        """
        Build FLUXLarge by loading Phase 7 checkpoint and scaling up.

        The Phase 7 checkpoint has smaller dimensions, so we:
        1. Create a fresh FLUXLarge with larger dimensions
        2. Load Phase 7 weights where shapes are compatible
        3. Initialize new capacity with random weights

        Args:
            device: Target device
            config: Optional config overrides

        Returns:
            FLUXLarge with Phase 7 knowledge + expanded capacity
        """
        # Create fresh large model
        model = cls(config=config, device=device)

        # Try to load Phase 7 checkpoint for knowledge transfer
        if checkpoint_exists(7):
            ckpt7 = load_checkpoint(7)
            print(f"  ✓ Phase 7 checkpoint loaded for knowledge transfer")

            # Transfer compatible weights (partial load)
            transferred = 0

            # CSE — different wave_dim, so we can't directly load
            # Instead we initialize fresh CSE with larger dimensions
            print(f"  ℹ CSE: fresh init (wave_dim scaled 432 → {model.config['wave_dim']})")

            # For components with compatible architectures, try partial transfer
            try:
                if 'gr_state' in ckpt7:
                    # GR state is architecture-independent (mass tracking)
                    print(f"  ℹ GR: fresh init (feature_dim scaled)")
                if 'causal_graph_state' in ckpt7:
                    model.causal_graph.load_state(ckpt7['causal_graph_state'])
                    transferred += 1
                    print(f"  ✓ Causal graph transferred")
            except Exception as e:
                print(f"  ℹ Partial transfer skipped: {e}")

            print(f"  ✓ Knowledge transfer complete: {transferred} components")
        else:
            print(f"  ℹ No Phase 7 checkpoint — starting FLUXLarge from scratch")

        model = model.to(device)
        stats = model.get_stats()
        print(f"\n  ═══ FLUXLarge assembled: {stats.total_params:,} total parameters ═══")
        return model

    @classmethod
    def from_phase8_checkpoint(cls, device: str = 'cpu') -> 'FLUXLarge':
        """
        Load FLUXLarge from a Phase 8 checkpoint.

        Args:
            device: Target device

        Returns:
            FLUXLarge with trained weights
        """
        ckpt = load_checkpoint(8)
        config = ckpt.get('config', FLUX_LARGE_CONFIG)
        model = cls(config=config, device=device)

        # Load all component states
        if 'cse_state_dict' in ckpt:
            try:
                model.cse.load_state_dict(ckpt['cse_state_dict'])
            except Exception:
                print("  ⚠ CSE state incompatible — using fresh init")

        if 'field_state_dict' in ckpt:
            try:
                model.field.load_state_dict(ckpt['field_state_dict'])
            except Exception:
                print("  ⚠ Field state incompatible — using fresh init")

        if 'gr_state' in ckpt:
            try:
                model.gr = GravitationalRelevance.load_state(ckpt['gr_state'], device=device)
            except Exception:
                print("  ⚠ GR state incompatible — using fresh init")

        if 'tl_state' in ckpt:
            try:
                model.tl.load_state(ckpt['tl_state'])
            except Exception:
                print("  ⚠ TL state incompatible — using fresh init")

        if 'cgn_state' in ckpt:
            try:
                model.cgn.load_state(ckpt['cgn_state'])
            except Exception:
                print("  ⚠ CGN state incompatible — using fresh init")

        if 'causal_graph_state' in ckpt:
            model.causal_graph.load_state(ckpt['causal_graph_state'])

        if 'working_memory_state' in ckpt:
            try:
                model.working_memory.load_state_memory(ckpt['working_memory_state'])
            except Exception:
                print("  ⚠ Working memory state incompatible — using fresh init")

        if 'episodic_memory_state' in ckpt:
            model.episodic_memory.load_state(ckpt['episodic_memory_state'])

        if 'semantic_memory_state' in ckpt:
            try:
                model.semantic_memory.load_state(ckpt['semantic_memory_state'])
            except Exception:
                print("  ⚠ Semantic memory state incompatible — using fresh init")

        if 'router_state' in ckpt:
            try:
                model.memory_router.load_state(ckpt['router_state'])
            except Exception:
                pass

        if 'wave_to_field_state' in ckpt:
            try:
                model.wave_to_field.load_state_dict(ckpt['wave_to_field_state'])
            except Exception:
                pass

        if 'field_to_wave_state' in ckpt:
            try:
                model.field_to_wave.load_state_dict(ckpt['field_to_wave_state'])
            except Exception:
                pass

        if 'output_head_state' in ckpt:
            try:
                model.output_head.load_state_dict(ckpt['output_head_state'])
            except Exception:
                pass

        model._learning_steps = ckpt.get('learning_steps', 0)
        model = model.to(device)

        stats = model.get_stats()
        print(f"  ✓ FLUXLarge loaded from Phase 8 checkpoint: {stats.total_params:,} params")
        return model


def compare_scales() -> Dict[str, ScaleProfile]:
    """Compare base (Phase 7) and large (Phase 8) model scales."""
    from flux_model import FLUX_MODEL_CONFIG

    base_profile = ScaleProfile(
        name='FLUXBase (Phase 7)',
        total_params=0,  # computed at runtime
        wave_dim=FLUX_MODEL_CONFIG['wave_dim'],
        field_dims=(
            FLUX_MODEL_CONFIG['field_h'],
            FLUX_MODEL_CONFIG['field_w'],
            FLUX_MODEL_CONFIG['field_d'],
        ),
        field_features=FLUX_MODEL_CONFIG['field_features'],
        cgn_nodes=(
            FLUX_MODEL_CONFIG['cgn_n_fast'] +
            FLUX_MODEL_CONFIG['cgn_n_medium'] +
            FLUX_MODEL_CONFIG['cgn_n_slow']
        ),
        wm_window=FLUX_MODEL_CONFIG['wm_window_size'],
    )

    large_profile = ScaleProfile(
        name='FLUXLarge (Phase 8)',
        total_params=0,  # computed at runtime
        wave_dim=FLUX_LARGE_CONFIG['wave_dim'],
        field_dims=(
            FLUX_LARGE_CONFIG['field_h'],
            FLUX_LARGE_CONFIG['field_w'],
            FLUX_LARGE_CONFIG['field_d'],
        ),
        field_features=FLUX_LARGE_CONFIG['field_features'],
        cgn_nodes=(
            FLUX_LARGE_CONFIG['cgn_n_fast'] +
            FLUX_LARGE_CONFIG['cgn_n_medium'] +
            FLUX_LARGE_CONFIG['cgn_n_slow']
        ),
        wm_window=FLUX_LARGE_CONFIG['wm_window_size'],
    )

    return {'base': base_profile, 'large': large_profile}


# ─────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  FLUXLarge — Scale Profile")
    print("=" * 60)

    profiles = compare_scales()
    for key, p in profiles.items():
        print(f"\n  {p.name}:")
        print(f"    Wave dim:       {p.wave_dim}")
        print(f"    Field dims:     {p.field_dims}")
        print(f"    Field features: {p.field_features}")
        print(f"    CGN nodes:      {p.cgn_nodes}")
        print(f"    WM window:      {p.wm_window}")

    print(f"\n  Building FLUXLarge...")
    device = get_device()
    model = FLUXLarge(device=device)
    stats = model.get_stats()
    print(f"  ✓ FLUXLarge built: {stats.total_params:,} parameters")
    print(f"  ✓ Target: ~117M (GPT-2 small)")
    print(f"  ✓ Ratio: {stats.total_params / 117e6:.2f}x GPT-2 small")
