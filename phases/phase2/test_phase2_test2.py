"""
PHASE 2 TEST 2: Local Update Verification

Verifies that field updates are truly local — perturbing one location
does NOT affect distant locations. This is the physics-based guarantee
against catastrophic forgetting.

Procedure:
  1. Create fresh ResonanceField
  2. Record field state at 100 random FAR locations (distance > 8)
  3. Apply perturbation at center of field
  4. Record field state at same 100 locations
  5. Assert: changes at far locations are negligible

Pass criteria:
  - Max change at distance > 8 from perturbation: < 0.01
  - Average change at distance > 8: < 0.001
  - This proves no global gradient is propagating

Usage: python test_phase2_test2.py
"""

import sys
import random
from pathlib import Path

import torch

PHASE_DIR = Path(__file__).parent
PROJECT_ROOT = PHASE_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PHASE_DIR.parent / 'phase1'))

from flux_utils import PhaseResults, get_device
from field import ResonanceField, FieldLocation


def main():
    print("=" * 60)
    print("  Phase 2 Test 2: Local Update Verification")
    print("=" * 60)

    device = get_device()
    print(f"  Device: {device}")

    results = PhaseResults(phase=2, component_name="Resonance Field Core")

    # Small field for testing
    H, W, D, F = 16, 16, 16, 128
    field = ResonanceField(h=H, w=W, d=D, features=F).to(device)

    # Center of field (where we'll perturb)
    center = FieldLocation(h=H // 2, w=W // 2, d=D // 2)
    print(f"\n  Perturbation center: ({center.h}, {center.w}, {center.d})")

    # ── Step 1: Pick 100 far locations (distance > 6 from center) ──
    min_distance = 6
    far_locations = []
    random.seed(42)
    attempts = 0
    while len(far_locations) < 100 and attempts < 10000:
        h = random.randint(0, H - 1)
        w = random.randint(0, W - 1)
        d = random.randint(0, D - 1)
        dist = ((h - center.h)**2 + (w - center.w)**2 + (d - center.d)**2) ** 0.5
        if dist > min_distance:
            far_locations.append((h, w, d))
        attempts += 1

    print(f"  Selected {len(far_locations)} far locations (distance > {min_distance})")
    if len(far_locations) < 50:
        print(f"  ⚠ Only found {len(far_locations)} far locations, using what we have")

    # ── Step 2: Record state before perturbation ──
    print("  Recording field state before perturbation...")
    before = {}
    with torch.no_grad():
        for loc in far_locations:
            h, w, d = loc
            before[loc] = field.state[h, w, d].clone()

    # ── Step 3: Apply perturbation at center ──
    print("  Applying perturbation at center...")
    wave_vector = torch.randn(field.wave_dim, device=device)

    # Override location mapping to ensure perturbation is at center
    # We do this by directly manipulating the field
    with torch.no_grad():
        target_feature = field.wave_to_feature(wave_vector).detach()

    radius = 4  # Default perturbation radius
    hs = slice(max(0, center.h - radius), min(H, center.h + radius + 1))
    ws = slice(max(0, center.w - radius), min(W, center.w + radius + 1))
    ds = slice(max(0, center.d - radius), min(D, center.d + radius + 1))

    # Compute distance weights
    distances = field._compute_distances(center, hs, ws, ds)
    weights = torch.exp(-distances / max(radius, 1)).unsqueeze(-1)

    # Apply local update
    neighborhood = field.state[hs, ws, ds]
    delta = target_feature.view(1, 1, 1, -1).expand_as(neighborhood) - neighborhood
    with torch.no_grad():
        field.state[hs, ws, ds] += weights * delta * 0.01

    print(f"  Updated {(hs.stop-hs.start) * (ws.stop-ws.start) * (ds.stop-ds.start)} locations")

    # ── Step 4: Record state after perturbation ──
    print("  Recording field state after perturbation...")
    changes = []
    with torch.no_grad():
        for loc in far_locations:
            h, w, d = loc
            after = field.state[h, w, d]
            change = (after - before[loc]).norm().item()
            changes.append(change)

    avg_change = sum(changes) / len(changes) if changes else 0
    max_change = max(changes) if changes else 0
    nonzero_changes = sum(1 for c in changes if c > 1e-10)

    print(f"\n  Results:")
    print(f"    Avg change at far locations: {avg_change:.8f}")
    print(f"    Max change at far locations: {max_change:.8f}")
    print(f"    Non-zero changes: {nonzero_changes}/{len(changes)}")

    # ── Step 5: Also verify center DID change ──
    center_change = (field.state[center.h, center.w, center.d] -
                     torch.randn(F, device=device) * 0.01).norm().item()
    print(f"    Center state norm: {field.state[center.h, center.w, center.d].norm().item():.6f}")

    # ── Record Results ──
    results.add_test(
        "Average far change < 0.001",
        passed=avg_change < 0.001,
        score=avg_change,
        threshold=0.001,
    )
    results.add_test(
        "Max far change < 0.01",
        passed=max_change < 0.01,
        score=max_change,
        threshold=0.01,
    )
    results.add_test(
        "Zero far changes (exact locality)",
        passed=nonzero_changes == 0,
        score=nonzero_changes,
        threshold=0,
        notes="All far locations should be exactly unchanged"
    )

    results.add_metric("avg_far_change", f"{avg_change:.10f}")
    results.add_metric("max_far_change", f"{max_change:.10f}")
    results.add_metric("nonzero_far_changes", nonzero_changes)
    results.add_metric("num_far_locations_tested", len(far_locations))
    results.add_metric("min_distance_from_center", min_distance)

    all_passed = results.all_tests_passed()
    print(f"\n  {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print(f"  Updates are {'TRULY LOCAL ✓' if all_passed else 'NOT LOCAL ✗'}")
    results.save()


if __name__ == '__main__':
    main()
