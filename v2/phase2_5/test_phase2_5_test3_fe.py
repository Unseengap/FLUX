"""
test_phase2_5_test3.py — Test: FieldEvolutionGenerator output is a valid wave

Loads Phase 2.5 FieldEvolutionGenerator checkpoint from HuggingFace.
Verifies physics invariants:
  1. Output shape is always [B, 432]
  2. Sub-bands are non-degenerate (std > 0 for each band)
  3. Output wave is bounded (tanh gives [-1, 1])
  4. Energy trace is monotonically non-increasing (settling works)
  5. Field state is finite and non-NaN

Thresholds:
  - Per-band std > 0.001 (not collapsed)
  - Max absolute value < 2.0 (tanh-bounded)
  - Energy never increases during settling
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase2'))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from flux_utils import PhaseResults

from field_evolution_generator import BAND_SLICES, TOTAL_WAVE_DIM
from train_field_evolution import load_phase1_and_phase2, load_phase2_5_checkpoint, PHASE2_5_CONFIG


TEST_TEXTS = [
    "The cat sat on the mat",
    "Quantum computing uses qubits to process information",
    "def factorial(n): return 1 if n == 0 else n * factorial(n-1)",
    "café résumé naïve coöperate",
    "Machine learning models translate patterns in data",
    "Water is a polar molecule consisting of two hydrogen atoms",
]


@torch.no_grad()
def main():
    print("\n" + "=" * 60)
    print("  TEST 3: FieldEvolutionGenerator output is a valid wave")
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

    all_pass = True

    for text in TEST_TEXTS:
        wave = cse.encode(text)
        chunk_waves, _ = chunker(wave.full)

        if chunk_waves.shape[0] < 3:
            print(f"  ⚠ Skipping '{text[:30]}' — too few chunks")
            continue

        n = max(2, chunk_waves.shape[0] // 2)
        prefix = chunk_waves[:n].unsqueeze(0)  # [1, n, 432]

        pred, info = generator(prefix)  # [1, 432]

        # ── Check 1: Shape ────────────────────────────────────────────
        assert pred.shape == (1, TOTAL_WAVE_DIM), \
            f"Shape mismatch: {pred.shape} vs (1, {TOTAL_WAVE_DIM})"

        # ── Check 2: Bounded output (tanh) ────────────────────────────
        max_abs = pred.abs().max().item()
        bounded_ok = max_abs < 2.0  # tanh should give [-1, 1], small tolerance

        # ── Check 3: Sub-band non-degeneracy ──────────────────────────
        band_ok = True
        band_stds = {}
        for name, (start, end) in BAND_SLICES.items():
            band = pred[0, start:end]
            std = band.std().item()
            band_stds[name] = std
            if std < 0.001:
                band_ok = False

        # ── Check 4: Finite (no NaN/Inf) ─────────────────────────────
        finite_ok = torch.isfinite(pred).all().item()
        field_finite = torch.isfinite(info['field_state']).all().item()

        # ── Check 5: Energy trace monotonically non-increasing ────────
        energy_trace = info['energy_trace'][0]  # [settle_steps + 1]
        diffs = energy_trace[1:] - energy_trace[:-1]
        monotonic_ok = (diffs <= 1e-3).all().item()  # small tolerance for float

        text_ok = bounded_ok and band_ok and finite_ok and field_finite and monotonic_ok
        status = '✓' if text_ok else '✗'
        print(f"\n  {status} '{text[:45]}'")
        print(f"    shape={pred.shape}  max_abs={max_abs:.3f}  bounded={bounded_ok}")
        print(f"    finite={finite_ok}  field_finite={field_finite}")
        print(f"    bands: {', '.join(f'{k}={v:.4f}' for k, v in band_stds.items())}")
        print(f"    energy_trace: {' → '.join(f'{e:.2f}' for e in energy_trace.tolist())}")
        print(f"    monotonic={monotonic_ok}")

        if not text_ok:
            all_pass = False

    # ── Record results ────────────────────────────────────────────────
    results = PhaseResults(phase=2, component_name="Field Evolution Wave Validity (Phase 2.5)")
    results.add_test(
        "Output Bounded (tanh)",
        passed=bounded_ok,
        score=max_abs,
        threshold=2.0,
    )
    results.add_test(
        "Sub-bands Non-Degenerate",
        passed=band_ok,
        score=min(band_stds.values()) if band_stds else 0.0,
        threshold=0.001,
    )
    results.add_test(
        "Energy Monotonically Decreasing",
        passed=monotonic_ok,
        score=diffs.max().item() if len(diffs) > 0 else 0.0,
        threshold=0.001,
    )
    results.add_test(
        "All Outputs Finite",
        passed=finite_ok and field_finite,
        score=1.0 if (finite_ok and field_finite) else 0.0,
        threshold=1.0,
    )
    results.save()

    assert all_pass, "FAILED: Some wave validity checks failed"
    print("\n  ✓ TEST 3 PASSED\n")


if __name__ == '__main__':
    main()
