"""
################################################################################
# PARTIALLY DEPRECATED as of v6.0-autonomous (April 2026)
#
# The WaveDecoder integration in this file is DEPRECATED.
# Text generation is now handled by embedded models.instruct.
#
# STILL USEFUL:
#   - FLUXLarge scaling concepts (larger field, more CGN nodes)
#   - Integration architecture patterns
#
# See: DOCS/FLUX_CONSOLIDATION_ROADMAP.md
################################################################################

Phase 8: FLUXModel + WaveDecoder — Scaled FLUX for GPT-2 Benchmark

Uses the SAME FLUXModel from Phase 7 with compatible field_features=512,
so all Phase 1-7 trained weights transfer directly. Scales capacity via:
- Larger spatial field (96³ vs 64³) — more attractor slots
- More CGN nodes (32/16/8 vs 16/8/4) — finer causal reasoning
- Larger working memory (2048 vs 512) — longer context
- WaveDecoder: new autoregressive byte-level decoder (~5M params)

CRITICAL: field_features stays at 512 to preserve Phase 7 compatibility.
The previous FLUXLarge used 768, which broke all trained weights from
Phases 1-7 and forced retraining from scratch on only 13MB of data.

The scaling philosophy follows the physics paradigm:
- Larger field = more knowledge capacity (96³ vs 64³)
- More CGN nodes = finer-grained causal reasoning (56 vs 28)
- Deeper working memory window = longer effective context (2048 vs 512)
- WaveDecoder = autoregressive byte generation guided by field semantics
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

# Add paths for direct execution
for _phase in ['phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'phase6', 'phase7']:
    _p = str(_PHASES_DIR / _phase)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import with fallback for embedded vs direct execution
try:
    from phases.phase1.cse import ContinuousSemanticEncoder
    from phases.phase1.wave_types import SemanticWave, TOTAL_WAVE_DIM
    from phases.phase2.field import ResonanceField
    from phases.phase3.gravity import GravitationalRelevance
    from phases.phase4.thermodynamic import ThermodynamicLearner
    from phases.phase5.cgn import CausalGeometryNode
    from phases.phase5.multi_timescale import MultiTimescaleCoordinator
    from phases.phase5.causal_graph import CausalGraph
    from phases.phase6.working_memory import WorkingMemory
    from phases.phase6.episodic_memory import EpisodicMemory
    from phases.phase6.semantic_memory import SemanticMemory
    from phases.phase6.memory_router import MemoryRouter
    from phases.phase6.consolidation import ConsolidationProcess
    from flux_model import FLUXModel, OutputHead, FLUXResponse, FLUXStats
    from phases.phase8.wave_decoder import WaveDecoder
    from flux_utils import (
        get_device, load_checkpoint, save_checkpoint, checkpoint_exists,
        PhaseLogger,
    )
except ImportError:
    from cse import ContinuousSemanticEncoder
    from wave_types import SemanticWave, TOTAL_WAVE_DIM
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
    from wave_decoder import WaveDecoder
    from flux_utils import (
        get_device, load_checkpoint, save_checkpoint, checkpoint_exists,
        PhaseLogger,
    )


# ─────────────────────────────────────────────
# CSE output dimension (fixed by Phase 1)
# ─────────────────────────────────────────────
CSE_WAVE_DIM = TOTAL_WAVE_DIM  # 432 — always, cannot be changed

# ─────────────────────────────────────────────
# Field features — MUST stay 512 for Phase 7 compatibility
# ─────────────────────────────────────────────
FIELD_FEATURES_PHASE8 = 512    # Same as Phase 7 — all trained weights transfer

# ─────────────────────────────────────────────
# Scaled Field Spatial Dimensions (capacity scaling)
# ─────────────────────────────────────────────
FIELD_H_LARGE = 96
FIELD_W_LARGE = 96
FIELD_D_LARGE = 96


# ─────────────────────────────────────────────
# Phase 8 Configuration — compatible with Phase 7 weights
# ─────────────────────────────────────────────
FLUX_LARGE_CONFIG = {
    # CSE (Phase 1) — wave_dim stays 432 (CSE is frozen / fixed)
    'wave_dim': CSE_WAVE_DIM,              # 432 — must match CSE output
    'byte_window': 8,
    'byte_stride': 1,
    'interference_radius': 6,               # wider interference

    # Field (Phase 2) — scaled spatial dims, SAME feature width
    'field_h': FIELD_H_LARGE,               # 96 (up from 64)
    'field_w': FIELD_W_LARGE,               # 96 (up from 64)
    'field_d': FIELD_D_LARGE,               # 96 (up from 64)
    'field_features': FIELD_FEATURES_PHASE8, # 512 — SAME as Phase 7

    # GR (Phase 3) — scaled
    'gr_k_neighbors': 64,
    'gr_base_mass': 1.0,
    'gr_mass_growth': 0.005,

    # TL (Phase 4) — scaled
    'tl_initial_temp': 1.0,
    'tl_min_temp': 0.005,
    'tl_decay': 0.9999,
    'tl_settle_iterations': 15,

    # CGN (Phase 5) — scaled node counts, SAME feature dim
    'cgn_feature_dim': FIELD_FEATURES_PHASE8,  # 512 — SAME as Phase 7
    'cgn_n_fast': 32,
    'cgn_n_medium': 16,
    'cgn_n_slow': 8,

    # Memory (Phase 6) — scaled window + episodic dim
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
    Phase 8 FLUX model — FLUXModel + WaveDecoder for generation.

    Uses the SAME field_features=512 as Phase 7, so all trained weights
    from Phases 1-7 transfer perfectly. Scales capacity via:
    - field: 64³ → 96³ (more attractor slots / knowledge capacity)
    - CGN nodes: 28 → 56 (finer causal reasoning)
    - Working memory: 512 → 2048 (longer context)
    - WaveDecoder: autoregressive byte-level generation (~5M params)

    The FLUX pipeline decides WHAT to say (semantic meaning via field).
    The WaveDecoder decides HOW to spell it (byte generation).

    Usage:
        model = FLUXLarge(device='cuda')
        model = FLUXLarge.from_phase7_checkpoint(device='cuda')
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, device: str = 'cpu'):
        merged = {**FLUX_LARGE_CONFIG, **(config or {})}
        super().__init__(config=merged, device=device)

        # The parent FLUXModel creates all components using the config:
        #   - CSE always outputs 432-dim (wave_dim=432, matching Phase 1)
        #   - wave_to_field bridge: nn.Linear(432, 512) — SAME as Phase 7
        #   - field_to_wave bridge: nn.Linear(512, 432) — SAME as Phase 7
        #   - field: 96³ × 512 features (larger spatial, same feature width)
        #   - output_head: accepts field_features=512, wave_context=432
        #
        # All Phase 7 trained weights load without dimension mismatch.

        # ── WaveDecoder: autoregressive byte-level generation ──
        # Config matches Run 2 (the run that produced working broken-English output):
        #   embed_dim=256, hidden_dim=1024, num_layers=4, num_heads=16
        # This makes the WaveDecoder ~38M params and the full model ~75M total.
        self.decoder = WaveDecoder(
            wave_dim=merged['wave_dim'],              # 432
            field_features=merged['field_features'],  # 512
            embed_dim=256,                            # Run 2: was 128 — restored
            hidden_dim=1024,                          # Run 2: was 512 — restored
            num_layers=4,                             # Run 2: was 2  — restored
            num_heads=16,                             # Run 2 default
            vocab_size=256,
        ).to(device)

    # ─────────────────────────────────────────────
    # Override generation to use WaveDecoder
    # ─────────────────────────────────────────────

    def _get_context(self, text: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run FLUX pipeline and return semantic context for generation.

        Returns:
            (wave_sequence [seq, wave_dim], wave_vec [wave_dim], merged [field_features])
        """
        device = self._device_str

        with torch.no_grad():
            wave = self.cse.encode(text)
        wave_sequence = wave.full.to(device)     # [seq, 432] — full sequence!
        wave_vec = wave_sequence.mean(dim=0)     # [432] for field query

        # Field query for context features
        field_features, sims, locs = self.field.query(wave_vec, k=4)
        combined = field_features.mean(dim=0)

        # CGN processing for causal enrichment
        cgn_out = self.cgn(combined)
        merged = combined + cgn_out

        return wave_sequence, wave_vec, merged

    def generate_logits_sequential(self, text: str,
                                   max_len: int = 100) -> torch.Tensor:
        """
        Generate sequential byte logits using the WaveDecoder.

        Args:
            text: Input context text
            max_len: Maximum output length

        Returns:
            [max_len, 256] logits sequence
        """
        wave_sequence, wave_vec, field_feat = self._get_context(text)

        with torch.no_grad():
            target_bytes = torch.tensor(
                list(text.encode('utf-8', errors='replace'))[:max_len],
                dtype=torch.long, device=wave_vec.device,
            )
            logits = self.decoder(target_bytes, wave_sequence, field_feat)

        return logits

    def generate(self, prompt: str, max_length: int = 100,
                 temperature: float = 0.8) -> str:
        """
        Generate text continuation using the WaveDecoder.

        The FLUX pipeline provides semantic context (WHAT to say).
        The WaveDecoder generates coherent bytes (HOW to spell it).

        Args:
            prompt: Starting text
            max_length: Maximum bytes to generate
            temperature: Sampling temperature

        Returns:
            Generated text (prompt + continuation)
        """
        device = self._device_str

        # Run prompt through full FLUX pipeline
        self.forward(prompt, learn=False)

        # Get semantic context (full wave sequence for cross-attention)
        wave_sequence, wave_vec, field_feat = self._get_context(prompt)

        # Encode prompt as bytes for seeding the decoder
        prompt_bytes = torch.tensor(
            list(prompt.encode('utf-8', errors='replace')),
            dtype=torch.long, device=device,
        )

        with torch.no_grad():
            generated_raw = self.decoder.generate(
                wave_sequence=wave_sequence,
                field_features=field_feat,
                max_length=max_length,
                temperature=temperature,
                prompt_bytes=prompt_bytes,
            )

        # Decode: prompt bytes + generated continuation
        try:
            result = (prompt.encode('utf-8') + generated_raw[len(prompt_bytes):]).decode(
                'utf-8', errors='replace'
            )
        except Exception:
            result = prompt + generated_raw[len(prompt_bytes):].decode(
                'latin-1', errors='replace'
            )

        return result

    def get_stats(self) -> FLUXStats:
        """Override to include WaveDecoder parameters in total count."""
        stats = super().get_stats()
        decoder_p = sum(p.numel() for p in self.decoder.parameters())
        return FLUXStats(
            total_params=stats.total_params + decoder_p,
            cse_params=stats.cse_params,
            field_params=stats.field_params,
            gr_params=stats.gr_params,
            tl_params=stats.tl_params,
            cgn_params=stats.cgn_params,
            memory_params=stats.memory_params,
            output_head_params=stats.output_head_params,
            episodic_entries=stats.episodic_entries,
            working_entries=stats.working_entries,
            field_energy=stats.field_energy,
            field_attractors=stats.field_attractors,
            temperature=stats.temperature,
            learning_steps=stats.learning_steps,
            decoder_params=decoder_p,
        )

    def get_scale_profile(self) -> ScaleProfile:
        """Return a profile describing this model's scale."""
        stats = self.get_stats()
        cfg = self.config
        return ScaleProfile(
            name='FLUXModel (Phase 8)',
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
        Build FLUXLarge by loading Phase 7 checkpoint with full weight transfer.

        Since field_features=512 matches Phase 7, ALL component weights
        transfer directly — no dimension mismatch, no retraining needed.
        Only the spatial field size differs (96³ vs 64³), which means field
        attractors don't transfer but all nn.Module weights do.

        Args:
            device: Target device
            config: Optional config overrides

        Returns:
            FLUXLarge with Phase 7 trained weights + WaveDecoder (fresh)
        """
        # Create fresh model with Phase 8 config
        model = cls(config=config, device=device)

        # Try to load Phase 7 checkpoint for knowledge transfer
        ckpt7 = None
        try:
            ckpt7 = load_checkpoint(7)
            print(f"  ✓ Phase 7 checkpoint loaded for knowledge transfer")
        except FileNotFoundError:
            print(f"  ℹ No Phase 7 checkpoint (local or HF Hub) — starting from scratch")

        if ckpt7 is not None:
            transferred = 0

            # CSE — wave_dim=432 is identical (always)
            if 'cse_state_dict' in ckpt7:
                try:
                    model.cse.load_state_dict(ckpt7['cse_state_dict'])
                    transferred += 1
                    print(f"  ✓ CSE transferred (wave_dim=432, trained Phase 1 weights)")
                except Exception as e:
                    print(f"  ⚠ CSE transfer failed: {e}")
            else:
                try:
                    ckpt1 = load_checkpoint(1)
                    if 'state_dict' in ckpt1:
                        model.cse.load_state_dict(ckpt1['state_dict'])
                        transferred += 1
                        print(f"  ✓ CSE transferred from Phase 1 checkpoint")
                except Exception:
                    print(f"  ℹ CSE: fresh init (no trained weights found)")

            # wave_to_field bridge — 432→512, SAME as Phase 7
            if 'wave_to_field_state' in ckpt7:
                try:
                    model.wave_to_field.load_state_dict(ckpt7['wave_to_field_state'])
                    transferred += 1
                    print(f"  ✓ wave_to_field bridge transferred (432→512)")
                except Exception as e:
                    print(f"  ⚠ wave_to_field transfer failed: {e}")

            # field_to_wave bridge — 512→432, SAME as Phase 7
            if 'field_to_wave_state' in ckpt7:
                try:
                    model.field_to_wave.load_state_dict(ckpt7['field_to_wave_state'])
                    transferred += 1
                    print(f"  ✓ field_to_wave bridge transferred (512→432)")
                except Exception as e:
                    print(f"  ⚠ field_to_wave transfer failed: {e}")

            # OutputHead — field_features=512, SAME as Phase 7
            if 'output_head_state' in ckpt7:
                try:
                    model.output_head.load_state_dict(ckpt7['output_head_state'])
                    transferred += 1
                    print(f"  ✓ OutputHead transferred (field_features=512)")
                except Exception as e:
                    print(f"  ⚠ OutputHead transfer failed: {e}")

            # GR — feature_dim=512, SAME as Phase 7
            if 'gr_state' in ckpt7:
                try:
                    model.gr = GravitationalRelevance.load_state(ckpt7['gr_state'], device=device)
                    transferred += 1
                    print(f"  ✓ GravitationalRelevance transferred (feature_dim=512)")
                except Exception as e:
                    print(f"  ⚠ GR transfer failed: {e}")

            # CGN — feature_dim=512, SAME as Phase 7
            # Node counts differ (32/16/8 vs 16/8/4) but feature_dim matches
            if 'cgn_state' in ckpt7:
                try:
                    model.cgn.load_state(ckpt7['cgn_state'])
                    transferred += 1
                    print(f"  ✓ CGN transferred (feature_dim=512)")
                except Exception:
                    print(f"  ℹ CGN: fresh init (node count changed, feature_dim compatible)")

            # TL — architecture-independent
            if 'tl_state' in ckpt7:
                try:
                    model.tl.load_state(ckpt7['tl_state'])
                    transferred += 1
                    print(f"  ✓ ThermodynamicLearner state transferred")
                except Exception:
                    print(f"  ℹ TL: fresh init")

            # Causal graph — architecture-independent
            if 'causal_graph_state' in ckpt7:
                try:
                    model.causal_graph.load_state(ckpt7['causal_graph_state'])
                    transferred += 1
                    print(f"  ✓ Causal graph transferred")
                except Exception:
                    pass

            # Episodic memory — dimension may differ (Phase 7=256, Phase 8=512)
            # load_state handles the mismatch gracefully (discards stale vectors)
            if 'episodic_memory_state' in ckpt7:
                try:
                    model.episodic_memory.load_state(ckpt7['episodic_memory_state'])
                    transferred += 1
                    print(f"  ✓ Episodic memory transferred")
                except Exception:
                    print(f"  ℹ Episodic memory: fresh init")

            # Semantic memory
            if 'semantic_memory_state' in ckpt7:
                try:
                    model.semantic_memory.load_state(ckpt7['semantic_memory_state'])
                    transferred += 1
                    print(f"  ✓ Semantic memory transferred")
                except Exception:
                    print(f"  ℹ Semantic memory: fresh init")

            # Router
            if 'router_state' in ckpt7:
                try:
                    model.memory_router.load_state(ckpt7['router_state'])
                    transferred += 1
                    print(f"  ✓ Memory router transferred")
                except Exception:
                    pass

            # Field attractors — spatial size changed (64³→96³), can't transfer state_dict
            # but individual attractors could be migrated in future
            if 'field_state_dict' in ckpt7:
                print(f"  ℹ Field: spatial size changed (64³→96³), attractors start fresh")
                print(f"    (field_features=512 is compatible — only spatial layout differs)")

            print(f"  ✓ Knowledge transfer complete: {transferred} components transferred")

        model = model.to(device)
        stats = model.get_stats()
        print(f"\n  ═══ FLUXModel (Phase 8) assembled: {stats.total_params:,} total parameters ═══")
        print(f"      field_features=512 (Phase 7 compatible)")
        print(f"      field_spatial=96³ (scaled from 64³)")
        print(f"      WaveDecoder: fresh init (needs training)")
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

        # Handle legacy checkpoints that used field_features=768
        # by overriding to the correct 512
        if config.get('field_features', 512) == 768:
            print("  ⚠ Legacy checkpoint with field_features=768 detected")
            print("    Rebuilding with field_features=512 (Phase 7 compatible)")
            config['field_features'] = 512
            config['cgn_feature_dim'] = 512

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
            try:
                model.episodic_memory.load_state(ckpt['episodic_memory_state'])
            except Exception:
                print("  ⚠ Episodic memory state incompatible — using fresh init")

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

        # ── WaveDecoder state ──
        if 'decoder_state_dict' in ckpt:
            try:
                dec_state = ckpt['decoder_state_dict']
                # torch.compile wraps modules in OptimizedModule, which
                # prefixes all state_dict keys with '_orig_mod.'.  Strip
                # this prefix so the weights load into a non-compiled decoder.
                if any(k.startswith('_orig_mod.') for k in dec_state):
                    dec_state = {k.replace('_orig_mod.', '', 1): v
                                 for k, v in dec_state.items()}
                model.decoder.load_state_dict(dec_state)
                print(f"  ✓ WaveDecoder loaded from checkpoint")
            except Exception as e:
                print(f"  ⚠ WaveDecoder state incompatible — using fresh init (needs retraining)")
                print(f"    Error: {e}")
        else:
            print("  ℹ No WaveDecoder in checkpoint — decoder needs training")

        model._learning_steps = ckpt.get('learning_steps', 0)
        model = model.to(device)

        stats = model.get_stats()
        print(f"  ✓ FLUXModel (Phase 8) loaded from checkpoint: {stats.total_params:,} params")
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
        name='FLUXModel (Phase 8)',
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
    print("  FLUXModel (Phase 8) — Scale Profile")
    print("=" * 60)

    profiles = compare_scales()
    for key, p in profiles.items():
        print(f"\n  {p.name}:")
        print(f"    Wave dim:       {p.wave_dim}")
        print(f"    Field dims:     {p.field_dims}")
        print(f"    Field features: {p.field_features}")
        print(f"    CGN nodes:      {p.cgn_nodes}")
        print(f"    WM window:      {p.wm_window}")

    print(f"\n  Building FLUXModel (Phase 8)...")
    device = get_device()
    model = FLUXLarge(device=device)
    stats = model.get_stats()
    print(f"  ✓ Built: {stats.total_params:,} parameters")
    print(f"  ✓ field_features=512 (Phase 7 compatible)")
    print(f"  ✓ field_spatial=96³ (scaled from 64³)")
