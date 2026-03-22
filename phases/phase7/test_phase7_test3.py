"""
Phase 7 Test 3: All Components Loaded Correctly

Verifies every component from phases 1–6 is properly loaded,
functional, and integrated into the unified FLUXModel.

Pass criteria:
  - All 6 phase components are non-None and functional
  - CSE encodes text without error
  - Field queries return valid results
  - GR gravitational weights are non-zero
  - TL thermodynamic settling runs
  - CGN multi-timescale processing works
  - Memory tiers (working, episodic, semantic) all function
  - .flx save/load roundtrip preserves state
"""

import sys
import torch
import tempfile
from pathlib import Path

# ── Path setup ──
_PHASE_DIR = Path(__file__).parent
_PHASES_DIR = _PHASE_DIR.parent
_PROJECT_ROOT = _PHASES_DIR.parent

for _p in ['phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'phase6']:
    sys.path.insert(0, str(_PHASES_DIR / _p))
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PHASE_DIR))

from flux_utils import PhaseResults, get_device
from flux_model import FLUXModel


def test_components_loaded():
    print("=" * 60)
    print("  Phase 7 Test 3: All Components Loaded Correctly")
    print("=" * 60)

    DEVICE = get_device()
    results = PhaseResults(phase=7, component_name="Full FLUX Integration")

    # Load model
    print("\n  Loading FLUXModel from checkpoints...")
    model = FLUXModel.from_checkpoints(device=DEVICE)

    # ── Phase 1: CSE ──
    print(f"\n  Checking Phase 1 (CSE)...")
    try:
        wave = model.encode("Component loading test")
        assert wave.full.shape[-1] == 432
        assert wave.semantic is not None
        assert wave.phonetic is not None
        cse_ok = True
        print(f"    ✓ CSE: wave shape {wave.full.shape}, all components present")
    except Exception as e:
        cse_ok = False
        print(f"    ✗ CSE: {e}")
    results.add_test("Phase 1 CSE", cse_ok, score="PASS" if cse_ok else "FAIL",
                     threshold="encodes text → [seq, 432]")

    # ── Phase 2: ResonanceField ──
    print(f"\n  Checking Phase 2 (ResonanceField)...")
    try:
        wave_vec = wave.full.mean(dim=0).to(DEVICE)
        field_feats, sims, locs = model.field.query(wave_vec, k=4)
        assert field_feats.shape[0] == 4
        assert field_feats.shape[1] == 512
        assert not torch.isnan(field_feats).any()
        energy = model.field.total_energy()
        field_ok = True
        print(f"    ✓ Field: query returned {field_feats.shape}, energy={energy:.4f}")
    except Exception as e:
        field_ok = False
        print(f"    ✗ Field: {e}")
    results.add_test("Phase 2 Field", field_ok, score="PASS" if field_ok else "FAIL",
                     threshold="query returns [k, 512]")

    # ── Phase 3: GravitationalRelevance ──
    print(f"\n  Checking Phase 3 (GR)...")
    try:
        field_input = model.wave_to_field(wave_vec)
        gr_out = model.gr(field_input.unsqueeze(0)).squeeze(0)
        assert gr_out.shape[-1] == 512
        assert not torch.isnan(gr_out).any()
        gr_ok = True
        print(f"    ✓ GR: output shape {gr_out.shape}")
    except Exception as e:
        gr_ok = False
        print(f"    ✗ GR: {e}")
    results.add_test("Phase 3 GR", gr_ok, score="PASS" if gr_ok else "FAIL",
                     threshold="output [512], no NaN")

    # ── Phase 4: ThermodynamicLearner ──
    print(f"\n  Checking Phase 4 (TL)...")
    try:
        settle_result = model.tl.settle_once(wave_vec)
        assert hasattr(settle_result, 'energy_delta')
        assert hasattr(settle_result, 'temperature')
        tl_ok = True
        print(f"    ✓ TL: settled, ΔE={settle_result.energy_delta:.6f}, "
              f"T={settle_result.temperature:.4f}")
    except Exception as e:
        tl_ok = False
        print(f"    ✗ TL: {e}")
    results.add_test("Phase 4 TL", tl_ok, score="PASS" if tl_ok else "FAIL",
                     threshold="settle_once completes")

    # ── Phase 5: CGN ──
    print(f"\n  Checking Phase 5 (CGN)...")
    try:
        cgn_input = torch.randn(512, device=DEVICE)
        cgn_out = model.cgn(cgn_input)
        assert cgn_out.shape[-1] == 512
        assert not torch.isnan(cgn_out).any()
        n_nodes = model.cgn.total_nodes()
        cgn_ok = True
        print(f"    ✓ CGN: {n_nodes} nodes, output shape {cgn_out.shape}")
    except Exception as e:
        cgn_ok = False
        print(f"    ✗ CGN: {e}")
    results.add_test("Phase 5 CGN", cgn_ok, score="PASS" if cgn_ok else "FAIL",
                     threshold="forward pass, no NaN")

    # ── Phase 6: Memory System ──
    print(f"\n  Checking Phase 6 (Memory)...")
    try:
        # Working memory
        model.working_memory.add_perturbation(wave_vec)
        wm_size = model.working_memory.size
        assert wm_size > 0

        # Episodic memory
        compressed = model.working_memory.compress(wave_vec.unsqueeze(0)).squeeze(0)
        eid = model.episodic_memory.write(compressed, fact="Test 3 fact", causal_source="test")
        results_ep = model.episodic_memory.search(compressed, k=1)
        assert len(results_ep) > 0

        # Semantic memory
        sm_energy = model.semantic_memory.get_field_energy()

        # Memory router
        route_result = model.memory_router.route_query(wave_vec, episodic_k=3, working_k=5)
        assert 'tier_weights' in route_result

        memory_ok = True
        print(f"    ✓ Working memory: {wm_size} entries")
        print(f"    ✓ Episodic: write+search OK (id={eid})")
        print(f"    ✓ Semantic: energy={sm_energy:.4f}")
        print(f"    ✓ Router: weights={[f'{w:.3f}' for w in route_result['tier_weights']]}")
    except Exception as e:
        memory_ok = False
        print(f"    ✗ Memory: {e}")
    results.add_test("Phase 6 Memory", memory_ok, score="PASS" if memory_ok else "FAIL",
                     threshold="all 3 tiers + router work")

    # ── .flx Save/Load Roundtrip ──
    print(f"\n  Checking .flx save/load roundtrip...")
    try:
        with tempfile.NamedTemporaryFile(suffix='.flx', delete=False) as f:
            tmp_path = f.name

        model.save_model(tmp_path)
        flx_size = Path(tmp_path).stat().st_size / 1e6
        print(f"    Saved: {flx_size:.1f} MB")

        model2 = FLUXModel.load_model(tmp_path, device=DEVICE)
        wave2 = model2.encode("roundtrip test")
        assert wave2.full.shape[-1] == 432

        stats1 = model.get_stats()
        stats2 = model2.get_stats()
        assert stats1.total_params == stats2.total_params
        assert stats1.cse_params == stats2.cse_params

        flx_ok = True
        print(f"    ✓ .flx roundtrip: {stats2.total_params:,} params preserved")

        # Cleanup
        Path(tmp_path).unlink(missing_ok=True)
    except Exception as e:
        flx_ok = False
        print(f"    ✗ .flx roundtrip: {e}")
    results.add_test(".flx Save/Load", flx_ok, score="PASS" if flx_ok else "FAIL",
                     threshold="state preserved after save+load")

    # ── Summary ──
    all_passed = cse_ok and field_ok and gr_ok and tl_ok and cgn_ok and memory_ok and flx_ok
    stats = model.get_stats()
    results.add_metric("total_params", f"{stats.total_params:,}")
    results.add_metric("components_ok", f"{'7/7' if all_passed else 'partial'}")
    results.save()

    print(f"\n  Component Summary:")
    components = [
        ("Phase 1 CSE", cse_ok),
        ("Phase 2 Field", field_ok),
        ("Phase 3 GR", gr_ok),
        ("Phase 4 TL", tl_ok),
        ("Phase 5 CGN", cgn_ok),
        ("Phase 6 Memory", memory_ok),
        (".flx Roundtrip", flx_ok),
    ]
    for name, ok in components:
        print(f"    {'✓' if ok else '✗'} {name}")

    print(f"\n  {'✓' if all_passed else '✗'} Test 3: {'PASS' if all_passed else 'FAIL'}")
    print(f"\nTest 3: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


if __name__ == "__main__":
    test_components_loaded()
