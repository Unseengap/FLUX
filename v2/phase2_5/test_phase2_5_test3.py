"""
test_phase2_5_test3.py — Test: WRU output is always a valid wave

Loads Phase 2.5 checkpoint from HuggingFace.
Verifies physics invariants of the WRU:
  1. Output shape is always [B, 432]
  2. Output energy is bounded (energy constraint works)
  3. Sub-bands are non-degenerate (std > 0 for each band)
  4. Output wave is in the same range as CSE waves (approximately tanh-bounded)

This tests the structural soundness of the WRU — its output must be
a valid SemanticWave that subsequent components can consume.

Thresholds:
  - Energy ≤ energy_cap (50.0)
  - Per-band std > 0.001 (not collapsed)
  - Max absolute value < 5.0 (bounded output — tanh gives [-1, 1])
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

from wave_recurrent_unit import BAND_SLICES, TOTAL_WAVE_DIM
from train_wru import load_phase1_and_phase2, load_phase2_5_checkpoint, PHASE2_5_CONFIG


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
    print("  TEST 3: WRU output is always a valid wave")
    print("=" * 60)

    device = 'cpu'
    hf_token = os.environ.get('HF_TOKEN', '')

    # Load components
    print("\n  Loading Phase 1 + 2 (frozen)...")
    components = load_phase1_and_phase2(device=device, hf_token=hf_token)
    cse = components['cse']
    chunker = components['chunker']
    w2f = components['w2f']

    print("  Loading Phase 2.5 WRU checkpoint...")
    wru = load_phase2_5_checkpoint(device=device, hf_token=hf_token)
    wru.eval()

    energy_cap = PHASE2_5_CONFIG['energy_cap']
    all_pass = True

    for text in TEST_TEXTS:
        wave = cse.encode(text)
        chunk_waves, _ = chunker(wave.full)

        if chunk_waves.shape[0] < 2:
            print(f"  ⚠ Skipping '{text[:30]}' — too few chunks")
            continue

        n = max(1, chunk_waves.shape[0] // 2)
        prefix_mean = chunk_waves[:n].mean(dim=0)
        ctx = w2f(prefix_mean).unsqueeze(0)  # [1, 512]

        pred, _ = wru(ctx)  # [1, 432]

        # ── Check 1: Shape ────────────────────────────────────────────
        assert pred.shape == (1, TOTAL_WAVE_DIM), \
            f"Shape mismatch: {pred.shape} vs (1, {TOTAL_WAVE_DIM})"

        # ── Check 2: Energy bounded ───────────────────────────────────
        energy = (pred ** 2).sum().item()
        energy_ok = energy <= energy_cap * 1.1  # small tolerance

        # ── Check 3: Sub-band non-degeneracy ──────────────────────────
        band_ok = True
        band_stds = {}
        for name, (start, end) in BAND_SLICES.items():
            band = pred[0, start:end]
            std = band.std().item()
            band_stds[name] = std
            if std < 0.001:
                band_ok = False

        # ── Check 4: Bounded output ───────────────────────────────────
        max_abs = pred.abs().max().item()
        bounded_ok = max_abs < 5.0

        # ── Report ────────────────────────────────────────────────────
        text_preview = text[:35]
        status = '✓' if (energy_ok and band_ok and bounded_ok) else '✗'
        print(f"  {status} '{text_preview}'")
        print(f"      energy={energy:.2f}{'✓' if energy_ok else '✗'}  "
              f"max_abs={max_abs:.3f}{'✓' if bounded_ok else '✗'}  "
              f"bands: " + " ".join(f"{n}={s:.4f}" for n, s in band_stds.items()))

        if not (energy_ok and band_ok and bounded_ok):
            all_pass = False

    # ── Also test batch mode ──────────────────────────────────────────
    print(f"\n  Batch mode check:")
    batch_ctx = torch.randn(16, 512)
    pred_batch, state_batch = wru(batch_ctx)
    assert pred_batch.shape == (16, TOTAL_WAVE_DIM)
    assert state_batch.shape == (16, TOTAL_WAVE_DIM)
    batch_energy = (pred_batch ** 2).sum(dim=-1)
    max_batch_energy = batch_energy.max().item()
    print(f"  ✓ Batch shape: {list(pred_batch.shape)}  max_energy={max_batch_energy:.2f}")

    # ── Record results ────────────────────────────────────────────────
    results = PhaseResults(phase=2, component_name="WRU Wave Validity (Phase 2.5)")
    results.add_test("Output shape correct", passed=True, score=1.0, threshold=1.0)
    results.add_test("Energy bounded", passed=max_batch_energy <= energy_cap * 1.1,
                     score=max_batch_energy, threshold=energy_cap)
    results.add_test("Sub-bands non-degenerate", passed=all_pass, score=1.0 if all_pass else 0.0,
                     threshold=1.0)
    results.save()

    assert all_pass, "FAILED: some wave validity checks failed"
    print("\n  ✓ TEST 3 PASSED\n")


if __name__ == '__main__':
    main()
