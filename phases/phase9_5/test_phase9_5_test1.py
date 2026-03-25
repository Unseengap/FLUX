"""
Phase 9.5 — Test 1: Context Collapse Fix Verification

Verifies that the six fixes resolved context collapse:
    - Wave 0 cross-context cosine < 0.85 (was 1.000 in Phase 9)
    - Hidden init cross-context cosine < 0.90 (was 0.996 in Phase 9)
    - Processed context diversity improved (was 0.980 avg cosine)

Usage:
    python test_phase9_5_test1.py
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path

# ─────────────────────────────────────────────
# Path setup
# ─────────────────────────────────────────────
_PHASES_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PHASES_DIR.parent

for _phase in ['phase1', 'phase2', 'phase3', 'phase4',
               'phase5', 'phase6', 'phase7', 'phase8', 'phase9']:
    _p = str(_PHASES_DIR / _phase)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from flux_utils import get_device, PhaseResults
from train_wave_gen_v2 import (
    load_phase9_5_modules,
    compute_merged_context,
    evaluate_context_diversity,
    precompute_wg_data,
    load_training_data,
)


def main() -> None:
    """Test context collapse is fixed."""
    device = get_device()
    print("=" * 60)
    print("  Phase 9.5 Test 1: Context Collapse Fix")
    print("=" * 60)

    # Load Phase 9.5
    model, chunker, generator, wtt = load_phase9_5_modules(device=device)
    generator.eval()

    # Load a small set of texts for precomputing
    texts = load_training_data(max_docs=200)
    precomputed = precompute_wg_data(model, chunker, texts, max_samples=50, device=device)
    assert len(precomputed) >= 10, f"Need ≥10 samples, got {len(precomputed)}"

    # Evaluate context diversity
    metrics = evaluate_context_diversity(generator, precomputed, n_samples=20, device=device)

    # ── Assertions ──
    w0_cos = metrics['wave0_cross_ctx_cosine']
    h_cos = metrics['hidden_init_cross_ctx_cosine']
    ctx_cos = metrics['processed_ctx_avg_cosine']

    print(f"\n  Test Results:")
    print(f"    Wave 0 cross-context cosine: {w0_cos:.3f} (threshold < 0.85)")
    w0_pass = w0_cos < 0.85
    print(f"    {'✓' if w0_pass else '✗'} Wave 0 test: {'PASS' if w0_pass else 'FAIL'}")

    print(f"    Hidden init cross-ctx cosine: {h_cos:.3f} (threshold < 0.90)")
    h_pass = h_cos < 0.90
    print(f"    {'✓' if h_pass else '✗'} Hidden init test: {'PASS' if h_pass else 'FAIL'}")

    print(f"    Processed context avg cosine: {ctx_cos:.3f} (Phase 9 was 0.980)")
    ctx_improved = ctx_cos < 0.98
    print(f"    {'✓' if ctx_improved else '✗'} Context diversity: {'improved' if ctx_improved else 'no improvement'}")

    # ── Also test that Wave 0 is different for very different prompts ──
    different_prompts = [
        "The cat sat on the mat",
        "Quantum mechanics describes particle behavior",
        "Chocolate cake recipe ingredients",
    ]
    wave0s = []
    with torch.no_grad():
        for prompt in different_prompts:
            wave = model.cse.encode(prompt)
            wave_seq = wave.full.to(device)
            wave_vec = wave_seq.mean(dim=0)
            merged = compute_merged_context(model, wave_vec, device)
            ctx = generator.process_context(merged)
            hidden = generator.init_hidden(device, field_context=merged)
            context_wave = generator.context_to_wave(ctx)
            wave0, _, _ = generator.forward_step(
                generator.bos_wave, context_wave, hidden
            )
            wave0s.append(wave0)

    w0_stack = torch.stack(wave0s)
    explicit_cos = F.cosine_similarity(
        w0_stack[0].unsqueeze(0), w0_stack[1].unsqueeze(0)
    ).item()
    explicit_cos2 = F.cosine_similarity(
        w0_stack[0].unsqueeze(0), w0_stack[2].unsqueeze(0)
    ).item()
    print(f"\n    Explicit Wave 0 test (3 diverse prompts):")
    print(f"      [cat] vs [quantum]: {explicit_cos:.3f}")
    print(f"      [cat] vs [chocolate]: {explicit_cos2:.3f}")

    all_pass = w0_pass and h_pass
    print(f"\n  {'='*58}")
    print(f"  Test 1 overall: {'✓ PASS' if all_pass else '✗ FAIL'}")
    assert all_pass, (
        f"Context collapse not fixed: "
        f"Wave0={w0_cos:.3f} (need <0.85), Hidden={h_cos:.3f} (need <0.90)"
    )
    print("=" * 60)


if __name__ == '__main__':
    main()
