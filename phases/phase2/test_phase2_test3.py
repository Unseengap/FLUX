"""
PHASE 2 TEST 3: Field Stability Over 1000 Steps

Verifies that the field remains stable through extended operation:
- Energy doesn't explode or collapse
- Attractors grow monotonically (never lost)
- Performance stays within time bounds

Procedure:
  1. Create fresh ResonanceField
  2. Run 1000 random perturbations
  3. Track total_energy() at each step
  4. Track num_attractors() every 100 steps
  5. Settle periodically

Pass criteria:
  - total_energy never exceeds initial × 10 (no explosion)
  - total_energy never drops below initial × 0.01 (no collapse)
  - num_attractors grows monotonically (never decreases between checks)
  - Completes in < 120 seconds on GPU

Usage: python test_phase2_test3.py
"""

import sys
import time
from pathlib import Path

import torch

PHASE_DIR = Path(__file__).parent
PROJECT_ROOT = PHASE_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PHASE_DIR.parent / 'phase1'))

from flux_utils import PhaseResults, get_device
from field import ResonanceField


def main():
    print("=" * 60)
    print("  Phase 2 Test 3: Field Stability Over 1000 Steps")
    print("=" * 60)

    device = get_device()
    print(f"  Device: {device}")

    results = PhaseResults(phase=2, component_name="Resonance Field Core")

    # Small field for testing speed
    field = ResonanceField(h=16, w=16, d=16, features=128).to(device)

    # Record initial state
    initial_energy = field.total_energy()
    energy_ceiling = initial_energy * 10.0
    energy_floor = initial_energy * 0.01
    print(f"\n  Initial energy: {initial_energy:.1f}")
    print(f"  Ceiling (×10): {energy_ceiling:.1f}")
    print(f"  Floor (×0.01): {energy_floor:.4f}")

    # ── Run 1000 random perturbations ──
    print("\n  Running 1000 perturbations...")
    start_time = time.time()

    energy_history = []
    attractor_history = []
    energy_exploded = False
    energy_collapsed = False
    attractors_decreased = False

    # Use a fixed set of 20 random wave vectors (repeated)
    # This simulates learning from a fixed set of patterns
    torch.manual_seed(42)
    wave_pool = [torch.randn(field.wave_dim, device=device) for _ in range(20)]

    prev_attractors = 0

    for step in range(1000):
        # Pick a random wave from the pool
        wave = wave_pool[step % len(wave_pool)]
        field.perturb(wave)

        # Track energy every step
        energy = field.total_energy()
        energy_history.append(energy)

        if energy > energy_ceiling:
            energy_exploded = True
        if energy < energy_floor:
            energy_collapsed = True

        # Settle and check attractors every 100 steps
        if (step + 1) % 100 == 0:
            field.settle(steps=5)
            attractors = field.num_attractors()
            attractor_history.append(attractors)

            if attractors < prev_attractors:
                attractors_decreased = True
            prev_attractors = attractors

            elapsed = time.time() - start_time
            print(f"    Step {step+1}: energy={energy:.1f}, attractors={attractors}, "
                  f"max_mass={field.mass.max().item():.4f}, "
                  f"elapsed={elapsed:.1f}s")

    elapsed = time.time() - start_time

    # ── Compute statistics ──
    final_energy = field.total_energy()
    final_attractors = field.num_attractors()

    min_energy = min(energy_history)
    max_energy = max(energy_history)
    energy_range = max_energy - min_energy

    print(f"\n  Results:")
    print(f"    Total time: {elapsed:.1f}s")
    print(f"    Energy range: [{min_energy:.2f}, {max_energy:.2f}]")
    print(f"    Energy exploded: {energy_exploded}")
    print(f"    Energy collapsed: {energy_collapsed}")
    print(f"    Final attractors: {final_attractors}")
    print(f"    Attractors decreased at any point: {attractors_decreased}")
    print(f"    Attractor history: {attractor_history}")

    # ── Record Results ──
    results.add_test(
        "Energy never explodes (< init×10)",
        passed=not energy_exploded,
        score=max_energy,
        threshold=energy_ceiling,
    )
    results.add_test(
        "Energy never collapses (> init×0.01)",
        passed=not energy_collapsed,
        score=min_energy,
        threshold=energy_floor,
    )
    results.add_test(
        "Attractors grow monotonically",
        passed=not attractors_decreased,
        score="monotonic" if not attractors_decreased else "decreased",
        threshold="monotonic",
    )
    results.add_test(
        "Completes in < 120 seconds",
        passed=elapsed < 120,
        score=f"{elapsed:.1f}s",
        threshold="120s",
    )
    results.add_test(
        "At least 5 attractors after 1000 steps",
        passed=final_attractors >= 5,
        score=final_attractors,
        threshold=5,
    )

    results.add_metric("total_time", f"{elapsed:.1f}s")
    results.add_metric("initial_energy", f"{initial_energy:.1f}")
    results.add_metric("final_energy", f"{final_energy:.1f}")
    results.add_metric("min_energy", f"{min_energy:.2f}")
    results.add_metric("max_energy", f"{max_energy:.2f}")
    results.add_metric("final_attractors", final_attractors)
    results.add_metric("attractor_history", str(attractor_history))
    results.add_metric("steps_per_second", f"{1000/elapsed:.1f}")

    all_passed = results.all_tests_passed()
    print(f"\n  {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print(f"  Field is {'STABLE ✓' if all_passed else 'UNSTABLE ✗'}")
    results.save()


if __name__ == '__main__':
    main()
