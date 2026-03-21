"""
Test 2: Retrieval Precision@k

Inserts known concept vectors into GR's spatial index,
then queries and checks if the correct concept is retrieved.

Pass criteria:
  - Precision@1  > 0.8
  - Precision@10 > 0.7
"""
import sys, torch
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from flux_utils import get_device, PhaseResults
from gravity import GravitationalRelevance


N_CONCEPTS = 50
FEATURE_DIM = 512


def main():
    device = get_device()
    results = PhaseResults(phase=3, component_name="Retrieval Precision@k")

    gr = GravitationalRelevance(feature_dim=FEATURE_DIM, device=device).to(device)
    gr.eval()

    # ── Create N distinct concept vectors ──
    # Use well-separated vectors (orthogonal-ish) for clear retrieval
    concepts = torch.randn(N_CONCEPTS, FEATURE_DIM, device=device)
    concepts = torch.nn.functional.normalize(concepts, dim=-1)

    # Add all concepts to spatial index
    gr.spatial_index.reset()
    gr.spatial_index.add(concepts)

    # ── Precision@1: query each concept, check if top-1 is itself ──
    correct_at_1 = 0
    correct_at_10 = 0

    for i in range(N_CONCEPTS):
        query = concepts[i:i+1]  # [1, D]

        # Add small noise so it's not an exact match
        noisy_query = query + torch.randn_like(query) * 0.05
        noisy_query = torch.nn.functional.normalize(noisy_query, dim=-1)

        indices = gr.spatial_index.search(noisy_query.squeeze(0), k=min(10, N_CONCEPTS))

        # Precision@1: is the original concept in top-1?
        if indices[0].item() == i:
            correct_at_1 += 1

        # Precision@10: is the original concept in top-10?
        if i in indices[:10].tolist():
            correct_at_10 += 1

    precision_at_1 = correct_at_1 / N_CONCEPTS
    precision_at_10 = correct_at_10 / N_CONCEPTS

    print(f"\n  Retrieval Precision Results ({N_CONCEPTS} concepts):")
    print(f"  {'Precision@1:':<20} {precision_at_1:.3f} (need > 0.8)")
    print(f"  {'Precision@10:':<20} {precision_at_10:.3f} (need > 0.7)")

    passed_1 = precision_at_1 > 0.8
    passed_10 = precision_at_10 > 0.7
    all_passed = passed_1 and passed_10

    print(f"\n  {'✓ PASS' if all_passed else '✗ FAIL'}: Retrieval Precision@k")

    results.add_test(
        "Precision@1",
        passed=passed_1,
        score=precision_at_1,
        threshold=0.8,
    )
    results.add_test(
        "Precision@10",
        passed=passed_10,
        score=precision_at_10,
        threshold=0.7,
    )
    results.save()


if __name__ == "__main__":
    main()
