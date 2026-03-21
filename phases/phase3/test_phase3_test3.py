"""
Test 3: Negative Mass Repulsion

Creates 10 concept vectors, contradicts them (negative mass),
then verifies that the gravitational weight assigned to each
contradicted concept is lower than its pre-contradiction weight.

Pass criteria: 10/10 concepts correctly repelled (negative mass
causes reduced gravitational pull).
"""
import sys, torch
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from flux_utils import get_device, PhaseResults
from gravity import GravitationalRelevance

N_CONCEPTS = 10
FEATURE_DIM = 512


def main():
    device = get_device()
    results = PhaseResults(phase=3, component_name="Negative Mass Repulsion")

    gr = GravitationalRelevance(feature_dim=FEATURE_DIM, device=device).to(device)
    gr.eval()

    # ── Create 10 concept vectors ──
    concepts = torch.randn(N_CONCEPTS, FEATURE_DIM, device=device)
    concepts = torch.nn.functional.normalize(concepts, dim=-1)

    # Observe them (positive mass)
    gr.mass_tracker.observe(concepts)

    # ── Check masses are positive initially ──
    print(f"\n  Negative Mass Repulsion Test ({N_CONCEPTS} concepts):")
    print(f"  {'-' * 50}")
    print(f"  {'Concept':<12} {'Before':<12} {'After':<12} {'Repelled?':<10}")
    print(f"  {'-' * 50}")

    repelled_count = 0
    for i in range(N_CONCEPTS):
        concept = concepts[i]

        # Get mass before contradiction
        mass_before = gr.mass_tracker.lookup_masses(concept.unsqueeze(0)).item()

        # Contradict the concept
        gr.contradict(concept, strength=50.0)

        # Get mass after contradiction
        mass_after = gr.mass_tracker.lookup_masses(concept.unsqueeze(0)).item()

        repelled = mass_after < mass_before
        if repelled:
            repelled_count += 1

        icon = "✓" if repelled else "✗"
        print(f"  {icon} Concept {i:<4} {mass_before:>10.4f} {mass_after:>10.4f}   {'YES' if repelled else 'NO'}")

    # ── Verify negative masses exist ──
    stats = gr.mass_tracker.stats()
    neg_count = stats.get('negative_count', 0)

    print(f"\n  Repelled: {repelled_count}/{N_CONCEPTS}")
    print(f"  Negative mass concepts in tracker: {neg_count}")

    passed = repelled_count == N_CONCEPTS
    print(f"\n  {'✓ PASS' if passed else '✗ FAIL'}: Negative Mass Repulsion ({repelled_count}/{N_CONCEPTS})")

    results.add_test(
        "Negative Mass Repulsion",
        passed=passed,
        score=repelled_count,
        threshold=N_CONCEPTS,
    )
    results.save()


if __name__ == "__main__":
    main()
