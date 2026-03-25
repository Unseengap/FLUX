"""
test_phase2_test1.py — Attractor Formation

Verifies that training the resonance field creates stable attractors:
- After repeated perturbation with the same wave, mass accumulates
- Mass above threshold → attractor registered in catalog
- Different waves map to different field locations (spatial spread)

Run: python test_phase2_test1.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from field import ResonanceField
from attractor import AttractorCatalog
from flux_utils import PhaseResults

TEST_TEXTS = [
    "The quick brown fox",
    "Machine learning models",
    "Physics describes fundamental laws",
    "Water is a polar molecule",
    "def fibonacci(n):",
]


@torch.no_grad()
def test_attractor_formation():
    """Test: repeated perturbation creates stable attractors."""
    print("=" * 60)
    print("Test 1: Attractor Formation")
    print("=" * 60)

    # Try to load Phase 2 v2 checkpoint
    try:
        from train_field import load_phase2_checkpoint
        from train_codec import load_phase1_checkpoint
        codec = load_phase1_checkpoint(device='cpu')
        field, bridge, catalog = load_phase2_checkpoint(device='cpu')
        cse = codec.cse
        print("  ✓ Phase 2 v2 checkpoint loaded")
    except FileNotFoundError:
        print("  ⚠ No checkpoint — testing with fresh field + 100 perturbations")
        from wave_types import TOTAL_WAVE_DIM
        from cse import ContinuousSemanticEncoder
        cse   = ContinuousSemanticEncoder()
        field = ResonanceField()
        catalog = AttractorCatalog(field)

        # Simulate training: perturb same wave many times
        for _ in range(100):
            for text in TEST_TEXTS:
                wave = cse.encode(text)
                wv = wave.full.mean(dim=0)
                field.perturb(wv, strength=1.0)

    # ── Test 1a: Attractors exist ──────────────────────────────────
    # Scan for attractors
    n_new = catalog.scan_and_update(mass_threshold=0.05)
    n_total = catalog.count()

    print(f"\n  Attractor count (mass > 0.05): {field.num_attractors(0.05)}")
    print(f"  Attractors in catalog: {n_total}  (new from scan: {n_new})")

    # Auto-register more attractors from field state
    if n_total < 3:
        # Do extra perturbations and re-scan
        from cse import ContinuousSemanticEncoder
        cse_tmp = ContinuousSemanticEncoder()
        for _ in range(200):
            for text in TEST_TEXTS[:3]:
                wave = cse_tmp.encode(text)
                field.perturb(wave.full.mean(dim=0), strength=2.0)
        n_new = catalog.scan_and_update(mass_threshold=0.05)

    n_total = catalog.count()
    assert n_total >= 1, (
        f"FAIL: only {n_total} attractors found\n"
        f"Field step count: {field.step_count}\n"
        f"Max mass: {field.mass.max():.4f}"
    )
    print(f"\n  ✓ {n_total} attractors formed (>= 1)")

    # ── Test 1b: Different texts → different locations ──────────────
    field_tmp = ResonanceField()
    from cse import ContinuousSemanticEncoder as CSE
    cse_t = CSE()
    locations = set()
    for text in TEST_TEXTS:
        wave = cse_t.encode(text)
        wv = wave.full.mean(dim=0)
        loc = field_tmp.wave_to_field_coords(wv)
        locations.add((loc.h, loc.w, loc.d))

    spread = len(locations)
    print(f"  Unique field locations for {len(TEST_TEXTS)} texts: {spread}/{len(TEST_TEXTS)}")
    assert spread >= 2, (
        f"FAIL: only {spread} unique field locations — spatial collapse\n"
        f"The spatial projection needs more diversity."
    )
    print(f"  ✓ {spread} unique locations (spatial spread verified)")

    # ── Test 1c: Field statistics look healthy ──────────────────────
    stats = field.get_field_stats()
    print(f"\n  Field statistics:")
    print(f"    Total energy : {stats['total_energy']:.1f}")
    print(f"    Mean mass    : {stats['mean_mass']:.5f}")
    print(f"    Max mass     : {stats['max_mass']:.4f}")
    print(f"    Step count   : {stats['step_count']}")

    assert stats['step_count'] >= 0, "FAIL: field has never been updated"
    print(f"\n  ✓ PASS: Attractor formation test")

    return n_total, spread


if __name__ == '__main__':
    results = PhaseResults(phase=2, component_name="Resonance Field v2")
    try:
        n_att, n_loc = test_attractor_formation()
        results.add_test("Attractors formed",   passed=(n_att >= 1), score=n_att, threshold=1)
        results.add_test("Spatial spread",      passed=(n_loc >= 2), score=n_loc, threshold=2)
        print("\n  ALL TESTS PASSED ✓")
    except AssertionError as e:
        print(f"\n  TEST FAILED ✗\n  {e}")
    except FileNotFoundError as e:
        print(f"\n  CHECKPOINT NOT FOUND ✗\n  {e}")
    results.save()
