"""
test_phase2_5_test2.py — Test: Different contexts produce different predictions

Loads Phase 2.5 FieldEvolutionGenerator checkpoint from HuggingFace.
Feeds different prefix sequences to the generator and verifies:
  1. Different prompts → different predicted waves (cosine < 0.90)
  2. Same prompt → same predicted wave (deterministic)

This tests that the FieldEvolutionGenerator is NOT collapsing all
contexts to the same output — the field must evolve differently
based on the prefix content.

Threshold: avg pairwise cosine between different contexts < 0.90
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase2'))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F
from flux_utils import PhaseResults

from train_field_evolution import load_phase1_and_phase2, load_phase2_5_checkpoint


DIVERSITY_TEXTS = [
    "The cat sat on the mat",
    "Quantum mechanics describes",
    "def fibonacci(n):",
    "Water freezes at zero degrees",
    "The stock market crashed",
    "Photosynthesis converts sunlight",
]


@torch.no_grad()
def main():
    print("\n" + "=" * 60)
    print("  TEST 2: Different contexts → different predictions")
    print("=" * 60)

    device = 'cpu'
    hf_token = os.environ.get('HF_TOKEN', '')

    # Load components
    print("\n  Loading Phase 1 + 2 (frozen)...")
    components = load_phase1_and_phase2(device=device, hf_token=hf_token)
    cse = components['cse']
    chunker = components['chunker']

    print("  Loading Phase 2.5 FieldEvolutionGenerator checkpoint...")
    generator = load_phase2_5_checkpoint(device=device, hf_token=hf_token)
    generator.eval()

    # Generate predictions from different contexts
    predictions = []
    energy_drops = []
    for text in DIVERSITY_TEXTS:
        wave = cse.encode(text)
        chunk_waves, _ = chunker(wave.full)

        if chunk_waves.shape[0] < 3:
            continue

        n = max(2, chunk_waves.shape[0] // 2)
        prefix = chunk_waves[:n].unsqueeze(0)  # [1, n, 432]

        pred, info = generator(prefix)
        predictions.append((text, pred.squeeze(0)))
        energy_drops.append(info['energy_drop'].item())

    # ── Check diversity: pairwise cosine between different contexts ────
    print(f"\n  Pairwise cosine similarities:")
    cosines = []
    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            cos = F.cosine_similarity(
                predictions[i][1].unsqueeze(0),
                predictions[j][1].unsqueeze(0),
                dim=-1,
            ).item()
            cosines.append(cos)
            t1 = predictions[i][0][:25]
            t2 = predictions[j][0][:25]
            status = '✓' if cos < 0.90 else '✗'
            print(f"  {status} cos({t1}, {t2}) = {cos:.4f}")

    avg_cos = sum(cosines) / max(len(cosines), 1)
    max_cos = max(cosines) if cosines else 1.0
    print(f"\n  Avg pairwise cosine: {avg_cos:.4f}  (want < 0.90)")
    print(f"  Max pairwise cosine: {max_cos:.4f}  (want < 0.95)")

    # ── Check energy drops ────────────────────────────────────────────
    print(f"\n  Energy drops during settling:")
    for i, (text, _) in enumerate(predictions):
        edrop = energy_drops[i]
        status = '✓' if edrop > 0 else '✗'
        print(f"  {status} '{text[:30]}' → ΔE = {edrop:.4f}")

    avg_edrop = sum(energy_drops) / max(len(energy_drops), 1)
    print(f"  Avg energy drop: {avg_edrop:.4f}  (want > 0)")

    # ── Check determinism: same input → same output ───────────────────
    print(f"\n  Determinism check:")
    text = DIVERSITY_TEXTS[0]
    wave = cse.encode(text)
    chunk_waves, _ = chunker(wave.full)
    n = max(2, chunk_waves.shape[0] // 2)
    prefix = chunk_waves[:n].unsqueeze(0)

    pred1, _ = generator(prefix)
    pred2, _ = generator(prefix)
    determ_cos = F.cosine_similarity(pred1, pred2, dim=-1).item()
    determ_ok = determ_cos > 0.999
    print(f"  {'✓' if determ_ok else '✗'} Same input cosine: {determ_cos:.6f}  (want > 0.999)")

    # ── Record results ────────────────────────────────────────────────
    results = PhaseResults(phase=2, component_name="Field Evolution Diversity (Phase 2.5)")
    results.add_test(
        "Avg Pairwise Cosine < 0.90",
        passed=avg_cos < 0.90,
        score=avg_cos,
        threshold=0.90,
    )
    results.add_test(
        "Deterministic Output",
        passed=determ_ok,
        score=determ_cos,
        threshold=0.999,
    )
    results.add_test(
        "Energy Decreases During Settling",
        passed=avg_edrop > 0,
        score=avg_edrop,
        threshold=0.0,
    )
    results.save()

    assert avg_cos < 0.90, f"FAILED: avg pairwise cosine {avg_cos:.4f} ≥ 0.90 (mode collapse)"
    assert determ_ok, f"FAILED: determinism cosine {determ_cos:.6f} < 0.999"
    print("\n  ✓ TEST 2 PASSED\n")


if __name__ == '__main__':
    main()
