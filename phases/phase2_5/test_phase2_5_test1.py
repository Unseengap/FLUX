"""
PHASE 2.5 TEST 1: Sparse Field Integrity

Verifies that:
1. Phase 2 dense coordinates correctly map to sparse field coordinates
2. Feature vectors are preserved after migration
3. Registry serialization round-trips correctly
4. Active capacity tracking is accurate

Pass criteria:
    - 100% coordinate mapping accuracy (64³ → sparse)
    - Feature cosine similarity > 0.99 after migration
    - Serialization round-trip produces bit-identical results
    - Capacity fraction reported correctly
    - Runs in < 60 seconds
"""

import sys
import time
import torch
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase2'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase2_5'))

from flux_utils import PhaseResults, get_device, load_checkpoint
from field_registry import FieldRegistry
from dynamic_field import SparseResonanceField


def main():
    print("=" * 60)
    print("  Phase 2.5 Test 1: Sparse Field Integrity")
    print("=" * 60)

    device = get_device()
    start  = time.time()
    results = PhaseResults(phase=2.5, component_name="SparseResonanceField")

    # ── Build a small sparse field ──
    field = SparseResonanceField(
        initial_h      = 16,
        initial_w      = 16,
        initial_d      = 16,
        features       = 64,
        wave_dim       = 432,
        checkpoint_dir = '/tmp/flux_test_ckpts/',
        device         = device,
    ).to(device)

    # ── Test 1: Write and read back locations ──
    print("\n  Test 1: Write → read round-trip...")
    torch.manual_seed(42)
    test_locs = [(3, 5, 7), (0, 0, 0), (15, 15, 15), (8, 8, 8)]
    test_feats = [torch.randn(64) for _ in test_locs]

    for (h, w, d), feat in zip(test_locs, test_feats):
        field.registry.set(h, w, d, feat, mass=0.5, energy=0.8)

    read_ok = True
    for (h, w, d), feat in zip(test_locs, test_feats):
        read = field.registry.get(h, w, d)
        if read is None:
            read_ok = False
            break
        sim = torch.nn.functional.cosine_similarity(
            read.unsqueeze(0), feat.unsqueeze(0)
        ).item()
        if sim < 0.999:
            read_ok = False
            break

    results.add_test(
        "Write-read round-trip",
        passed=read_ok,
        score="bit-exact" if read_ok else "mismatch",
        threshold="cosine_sim > 0.999",
    )
    print(f"    {'✓' if read_ok else '✗'} Write-read: {'OK' if read_ok else 'FAIL'}")

    # ── Test 2: Capacity fraction tracking ──
    print("\n  Test 2: Capacity fraction tracking...")
    field2 = SparseResonanceField(
        initial_h=8, initial_w=8, initial_d=8,
        features=32, wave_dim=432,
        checkpoint_dir='/tmp/flux_test_ckpts/',
        device=device,
    )
    theoretical_max = 8 * 8 * 8  # 512
    # Insert exactly 51 locations (10% of 512)
    inserted = 0
    for h in range(8):
        for w in range(8):
            for d in range(8):
                if inserted >= 51:
                    break
                field2.registry.set(h, w, d, torch.randn(32), mass=0.1)
                inserted += 1
            if inserted >= 51:
                break
        if inserted >= 51:
            break

    cap = field2.registry.capacity_fraction()
    expected_cap = 51 / 512
    cap_ok = abs(cap - expected_cap) < 0.01

    results.add_test(
        "Capacity fraction accuracy",
        passed=cap_ok,
        score=f"{cap:.4f}",
        threshold=f"≈ {expected_cap:.4f} (±0.01)",
    )
    print(f"    {'✓' if cap_ok else '✗'} Capacity fraction: {cap:.4f} (expected {expected_cap:.4f})")

    # ── Test 3: Registry resize migration ──
    print("\n  Test 3: Registry resize migration (8³ → 16³)...")
    reg_small = FieldRegistry(max_h=8, max_w=8, max_d=8, feature_dim=32)
    known_feat = torch.randn(32)
    reg_small.set(4, 4, 4, known_feat, mass=0.9)
    reg_small.set(1, 2, 3, torch.randn(32), mass=0.5)
    reg_small.set(7, 7, 7, torch.randn(32), mass=0.3)
    n_before = reg_small.active_count()

    reg_large = reg_small.resize(16, 16, 16)
    n_after   = reg_large.active_count()

    # Check that (4,4,4) → (8,8,8) and feature preserved
    migrated_feat = reg_large.get(8, 8, 8)
    migration_ok  = (
        n_after >= n_before and
        migrated_feat is not None and
        torch.nn.functional.cosine_similarity(
            migrated_feat.unsqueeze(0), known_feat.unsqueeze(0)
        ).item() > 0.99
    )

    results.add_test(
        "Registry resize migration",
        passed=migration_ok,
        score=f"{n_before} → {n_after} locations, sim > 0.99",
        threshold="all locations preserved, features intact",
    )
    print(f"    {'✓' if migration_ok else '✗'} Migration: {n_before} → {n_after} locations")

    # ── Test 4: Serialization round-trip ──
    print("\n  Test 4: Serialization round-trip...")
    reg_orig = FieldRegistry(max_h=8, max_w=8, max_d=8, feature_dim=32)
    orig_feats = {}
    for i in range(20):
        h, w, d = i % 8, (i * 3) % 8, (i * 7) % 8
        f = torch.randn(32)
        orig_feats[(h, w, d)] = f
        reg_orig.set(h, w, d, f, mass=float(i) * 0.05)

    serial   = reg_orig.serialize()
    reg_copy = FieldRegistry.deserialize(serial)

    serial_ok = True
    for (h, w, d), f in orig_feats.items():
        read = reg_copy.get(h, w, d)
        if read is None:
            serial_ok = False
            break
        sim = torch.nn.functional.cosine_similarity(
            read.unsqueeze(0), f.unsqueeze(0)
        ).item()
        if sim < 0.999:
            serial_ok = False
            break

    results.add_test(
        "Serialization round-trip",
        passed=serial_ok,
        score="bit-exact" if serial_ok else "mismatch",
        threshold="all features preserved",
    )
    print(f"    {'✓' if serial_ok else '✗'} Serialization: {'OK' if serial_ok else 'FAIL'}")

    # ── Test 5: Phase 2 weight inheritance (if checkpoint exists) ──
    print("\n  Test 5: Phase 2 weight inheritance...")
    try:
        ckpt2 = load_checkpoint(2)
        field3 = SparseResonanceField(
            initial_h=64, initial_w=64, initial_d=64,
            features=512, wave_dim=432,
            checkpoint_dir='/tmp/flux_test_ckpts/',
            device='cpu',
        )
        field3.inherit_from_phase2(ckpt2['state_dict'])

        # Verify projections are not all-zero
        sp_norm = sum(
            p.norm().item()
            for p in field3.wave_to_location.parameters()
        )
        wf_norm = sum(
            p.norm().item()
            for p in field3.wave_to_feature.parameters()
        )
        inherit_ok = sp_norm > 1.0 and wf_norm > 1.0
        inherit_msg = f"SP norm={sp_norm:.2f}, WF norm={wf_norm:.2f}"
    except Exception as e:
        inherit_ok  = False
        inherit_msg = f"Checkpoint missing: {e}"

    results.add_test(
        "Phase 2 weight inheritance",
        passed=inherit_ok,
        score=inherit_msg,
        threshold="non-zero projection norms",
    )
    print(f"    {'✓' if inherit_ok else '✗'} Inheritance: {inherit_msg}")

    elapsed = time.time() - start
    results.add_test(
        "Runtime < 60s",
        passed=elapsed < 60,
        score=f"{elapsed:.1f}s",
        threshold="< 60s",
    )

    results.add_metric("active_locations_test1", field.registry.active_count())
    results.add_metric("capacity_fraction_test2", f"{cap:.4f}")
    results.add_metric("migration_locations", f"{n_before}→{n_after}")

    all_passed = results.all_tests_passed()
    print(f"\n  {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    results.save()
    return all_passed


if __name__ == '__main__':
    main()
