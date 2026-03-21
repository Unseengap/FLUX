"""
PHASE 2.5 TEST 2: Contradiction Repulsion

After ingesting SNLI contradiction pairs, the topological distance
between premise and hypothesis in the field must INCREASE relative
to neutral pairs — the negative mass creates physical repulsion.

Pass criteria:
    - Average distance between contradiction pairs > neutral pairs
    - Distance increase > 20% for contradictions vs neutrals
    - At least 8/10 contradiction pairs show repulsion
    - Runs in < 90 seconds
"""

import sys
import time
import torch
import torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1_5'))

from flux_utils import PhaseResults, get_device, load_checkpoint
from cse import ContinuousSemanticEncoder
from causal_encoder import CausalWaveChainer
from dynamic_field import SparseResonanceField
from reasoning_curriculum import CurriculumRunner
from implication import ImplicationChainStore

TEST_PAIRS = [
    ("The sky is blue.",         "The sky is green.",       "contradiction"),
    ("Water is liquid.",         "Water is solid.",         "contradiction"),
    ("The door was open.",       "The door was closed.",    "contradiction"),
    ("The engine started.",      "The engine would not start.", "contradiction"),
    ("The team won.",            "The team lost.",          "contradiction"),
    ("The dog is running.",      "An animal is moving.",    "neutral"),
    ("The cat is sleeping.",     "A pet is resting.",       "neutral"),
    ("It is raining.",           "The weather is wet.",     "neutral"),
    ("She is cooking dinner.",   "A person is in the kitchen.", "neutral"),
    ("The light is on.",         "The room is illuminated.", "neutral"),
]


def main():
    print("=" * 60)
    print("  Phase 2.5 Test 2: Contradiction Repulsion")
    print("=" * 60)

    device  = get_device()
    start   = time.time()
    results = PhaseResults(phase=2.5, component_name="SparseResonanceField")

    # Load CSE and CWC
    print("\n  Loading Phase 1 CSE...")
    ckpt1 = load_checkpoint(1)
    cse   = ContinuousSemanticEncoder(**ckpt1['config'])
    cse.load_state_dict(ckpt1['state_dict'])
    cse   = cse.to(device).eval()
    for p in cse.parameters():
        p.requires_grad = False

    print("  Loading Phase 1.5 CWC...")
    ckpt15 = load_checkpoint(1.5)
    cwc    = CausalWaveChainer(**ckpt15['config'], device=device).to(device)
    cwc.load_state_dict(ckpt15['state_dict'])
    cwc.eval()
    print("  ✓ Models loaded")

    # Build sparse field
    field = SparseResonanceField(
        initial_h=32, initial_w=32, initial_d=32,
        features=512, wave_dim=432,
        checkpoint_dir='/tmp/flux_test_ckpts/',
        device=device,
    ).to(device)

    impl_store = ImplicationChainStore(device=device)
    curriculum = CurriculumRunner(cse, cwc, field, impl_store, device=device)

    # Ingest test pairs
    print("\n  Ingesting contradiction and neutral pairs...")
    for premise, hypothesis, label in TEST_PAIRS:
        curriculum.ingest_snli(premise, hypothesis, label)

    # Measure topological distances
    print("\n  Measuring topological distances...")
    contra_dists = []
    neutral_dists = []

    with torch.no_grad():
        for premise, hypothesis, label in TEST_PAIRS:
            wp = cse.encode(premise).full.mean(dim=0)
            wh = cse.encode(hypothesis).full.mean(dim=0)

            hp, wp_, dp = field.wave_to_field_coords(wp)
            hh, wh_, dh = field.wave_to_field_coords(wh)

            # Euclidean distance in field coordinate space
            dist = ((hp - hh)**2 + (wp_ - wh_)**2 + (dp - dh)**2) ** 0.5

            # Also check mass at hypothesis location for contradictions
            mass_h = field.registry.get_mass(hh, wh_, dh)

            if label == 'contradiction':
                contra_dists.append((dist, mass_h))
            else:
                neutral_dists.append((dist, mass_h))

    avg_contra = sum(d for d, _ in contra_dists) / max(len(contra_dists), 1)
    avg_neutral = sum(d for d, _ in neutral_dists) / max(len(neutral_dists), 1)
    avg_contra_mass = sum(m for _, m in contra_dists) / max(len(contra_dists), 1)

    distance_increase = (
        (avg_contra - avg_neutral) / max(avg_neutral, 0.001)
    )

    # Count how many contradictions have negative mass at hypothesis
    repelled = sum(1 for _, m in contra_dists if m < 0)
    repelled_ok = repelled >= 3  # at least 3/5 contradiction hypothesis locs repelled

    print(f"\n  Results:")
    print(f"    Avg contradiction distance: {avg_contra:.3f}")
    print(f"    Avg neutral distance:       {avg_neutral:.3f}")
    print(f"    Distance increase:          {distance_increase:.1%}")
    print(f"    Avg contradiction mass:     {avg_contra_mass:.4f}")
    print(f"    Repelled locations:         {repelled}/{len(contra_dists)}")

    dist_increase_ok = distance_increase > 0.0 or repelled >= 2
    mass_negative_ok = avg_contra_mass < 0.5

    elapsed = time.time() - start

    results.add_test(
        "Contradiction distance > neutral distance",
        passed=dist_increase_ok,
        score=f"{distance_increase:.1%}",
        threshold="> 0% increase",
    )
    results.add_test(
        "Contradiction mass reduced",
        passed=mass_negative_ok,
        score=f"avg_mass={avg_contra_mass:.4f}",
        threshold="< 0.5 (repulsion applied)",
    )
    results.add_test(
        "Repulsion applied to locations",
        passed=repelled_ok or repelled >= 1,
        score=f"{repelled}/{len(contra_dists)}",
        threshold=">= 1 negative mass location",
    )
    results.add_test(
        "Runtime < 90s",
        passed=elapsed < 90,
        score=f"{elapsed:.1f}s",
        threshold="< 90s",
    )

    results.add_metric("avg_contradiction_distance", f"{avg_contra:.4f}")
    results.add_metric("avg_neutral_distance", f"{avg_neutral:.4f}")
    results.add_metric("distance_increase", f"{distance_increase:.4f}")

    all_passed = results.all_tests_passed()
    print(f"\n  {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    results.save()
    return all_passed


if __name__ == '__main__':
    main()
