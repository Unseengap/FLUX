"""
PHASE 2 TEST 1: Attractor Formation and No-Forgetting

Verifies that:
1. Repeated inputs create stable attractors in the field
2. New patterns don't destroy old attractors (no catastrophic forgetting)

Procedure:
  1. Load Phase 1 CSE (or use random vectors as fallback)
  2. Create fresh ResonanceField (small for speed)
  3. Feed Pattern A 20 times → verify attractor A forms
  4. Feed Pattern B 20 times → verify attractor B forms
  5. CRITICAL: Verify attractor A is still intact after B training

Pass criteria:
  - Pattern A similarity to original > 0.7 after B training
  - Both attractors exist in the field
  - Forgetting score < 0.2

Usage: python test_phase2_test1.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PHASE_DIR = Path(__file__).parent
PROJECT_ROOT = PHASE_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PHASE_DIR.parent / 'phase1'))

from flux_utils import PhaseResults, get_device
from field import ResonanceField
from attractor import AttractorCatalog


def get_wave_vectors(device: str):
    """Get wave vectors from CSE or fallback to random."""
    try:
        from flux_utils import load_checkpoint
        from cse import ContinuousSemanticEncoder
        checkpoint = load_checkpoint(1)
        config = checkpoint.get('config', {})
        cse = ContinuousSemanticEncoder(**config)
        cse.load_state_dict(checkpoint['state_dict'])
        cse = cse.to(device)
        cse.eval()
        with torch.no_grad():
            wave_a = cse.encode("The cat sat on the mat and watched the birds fly past.").full.mean(dim=0)
            wave_b = cse.encode("Quantum mechanics describes the behavior of subatomic particles.").full.mean(dim=0)
            wave_c = cse.encode("The economy grew significantly during the fiscal year report.").full.mean(dim=0)
        print("  ✓ Using CSE-encoded wave vectors")
        return wave_a, wave_b, wave_c
    except Exception as e:
        print(f"  ⚠ CSE unavailable ({e}), using random vectors")
        wave_a = torch.randn(432, device=device)
        wave_b = torch.randn(432, device=device)
        wave_c = torch.randn(432, device=device)
        return wave_a, wave_b, wave_c


def main():
    print("=" * 60)
    print("  Phase 2 Test 1: Attractor Formation & No-Forgetting")
    print("=" * 60)

    device = get_device()
    print(f"  Device: {device}")

    results = PhaseResults(phase=2, component_name="Resonance Field Core")

    # Use small field for testing speed
    field = ResonanceField(h=16, w=16, d=16, features=128).to(device)
    catalog = AttractorCatalog(field)

    wave_a, wave_b, wave_c = get_wave_vectors(device)

    # ── Step 1: Feed Pattern A 20 times ──
    print("\n  Step 1: Training Pattern A (20 repetitions)...")
    for _ in range(20):
        field.perturb(wave_a)
    field.settle(steps=5)

    # Register attractor A
    att_a = catalog.register_from_wave(wave_a, label="Pattern A")
    sim_a_initial = F.cosine_similarity(
        field.get_state_at(field.wave_to_field_coords(wave_a)).unsqueeze(0),
        field.wave_to_feature(wave_a).unsqueeze(0),
    ).item()
    print(f"    Attractor A formed at {att_a.location}, similarity: {sim_a_initial:.4f}")

    # ── Step 2: Feed Pattern B 20 times ──
    print("  Step 2: Training Pattern B (20 repetitions)...")
    for _ in range(20):
        field.perturb(wave_b)
    field.settle(steps=5)

    att_b = catalog.register_from_wave(wave_b, label="Pattern B")
    print(f"    Attractor B formed at {att_b.location}")

    # ── Step 3: Feed Pattern C 20 times ──
    print("  Step 3: Training Pattern C (20 repetitions)...")
    for _ in range(20):
        field.perturb(wave_c)
    field.settle(steps=5)

    att_c = catalog.register_from_wave(wave_c, label="Pattern C")
    print(f"    Attractor C formed at {att_c.location}")

    # ── Step 4: Verify all attractors persist ──
    print("\n  Step 4: Verifying attractor persistence...")
    verification = catalog.verify_all(similarity_threshold=0.7)
    print(f"    Total: {verification['total']}")
    print(f"    Alive: {verification['alive']}")
    print(f"    Dead:  {verification['dead']}")
    print(f"    Survival rate: {verification['survival_rate']:.2%}")
    for d in verification['details']:
        status = "✓" if d['alive'] else "✗"
        print(f"    {status} {d['label']}: similarity = {d['similarity']:.4f}")

    # ── Step 5: Compute forgetting score ──
    a_alive = catalog.verify_attractor_persistence(att_a.attractor_id, 0.7)
    b_alive = catalog.verify_attractor_persistence(att_b.attractor_id, 0.7)
    c_alive = catalog.verify_attractor_persistence(att_c.attractor_id, 0.7)

    forgetting_score = 1.0 - verification['survival_rate']

    # ── Record Results ──
    results.add_test(
        "Attractor A survives B+C training",
        passed=a_alive,
        score=verification['details'][0]['similarity'],
        threshold=0.7,
    )
    results.add_test(
        "Attractor B survives C training",
        passed=b_alive,
        score=verification['details'][1]['similarity'],
        threshold=0.7,
    )
    results.add_test(
        "All 3 attractors coexist",
        passed=verification['survival_rate'] >= 0.9,
        score=verification['survival_rate'],
        threshold=0.9,
    )
    results.add_test(
        "Forgetting score < 0.2",
        passed=forgetting_score < 0.2,
        score=forgetting_score,
        threshold=0.2,
    )
    results.add_test(
        "Field has attractors",
        passed=field.num_attractors() >= 3,
        score=field.num_attractors(),
        threshold=3,
    )

    results.add_metric("num_attractors", field.num_attractors())
    results.add_metric("total_energy", f"{field.total_energy():.1f}")
    results.add_metric("forgetting_score", f"{forgetting_score:.4f}")

    # ── Report ──
    all_passed = results.all_tests_passed()
    print(f"\n  {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print(f"  Forgetting score: {forgetting_score:.4f} (target: < 0.2)")
    results.save()


if __name__ == '__main__':
    main()
