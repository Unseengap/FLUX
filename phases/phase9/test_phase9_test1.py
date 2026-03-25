"""
Phase 9 — Test 1: Generated Waves Match CSE Distribution

Setup: Generate 100 wave sequences from diverse prompts.
Verify:
- Generated waves have similar L2 norm distribution as real CSE waves
- Cosine similarity between consecutive generated waves > 0.3 (coherent)
- Wave dimension statistics (mean, std per component) within 2σ of CSE stats
- Pass criterion: KL divergence between generated and real wave stats < 0.5
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────
# Path setup
# ─────────────────────────────────────────────
_PHASES_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PHASES_DIR.parent

for _phase in ['phase1', 'phase2', 'phase3', 'phase4',
               'phase5', 'phase6', 'phase7', 'phase8']:
    _p = str(_PHASES_DIR / _phase)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from flux_utils import get_device, PhaseResults
from train_wave_gen import load_phase9_modules, build_phase9_modules, PHASE9_CONFIG


def collect_cse_wave_stats(model, prompts, device):
    """Collect L2 norm and per-dim stats from real CSE waves."""
    all_norms = []
    all_waves = []
    for text in prompts:
        try:
            with torch.no_grad():
                wave = model.cse.encode(text)
            wave_seq = wave.full.to(device)  # [seq, 432]
            norms = wave_seq.norm(dim=-1)  # [seq]
            all_norms.extend(norms.cpu().tolist())
            all_waves.append(wave_seq.cpu())
        except Exception as e:
            print(f"    ⚠ CSE encode failed for '{text[:30]}...': {e}")
            continue
    all_waves_cat = torch.cat(all_waves, dim=0)  # [total, 432]
    return all_norms, all_waves_cat


def collect_generated_wave_stats(model, chunker, generator, wtt, prompts, device):
    """Collect stats from generated waves."""
    all_norms = []
    all_waves = []
    all_consec_cos = []

    generator.eval()
    for text in prompts:
        try:
            with torch.no_grad():
                wave_seq, wave_vec, merged = model._get_context(text)
                gen_waves, confs = generator.generate(
                    field_context=merged,
                    max_waves=20,
                    flux_model=None,  # Static context for consistency
                )
            # L2 norms
            norms = gen_waves.norm(dim=-1)
            all_norms.extend(norms.cpu().tolist())
            all_waves.append(gen_waves.cpu())

            # Consecutive cosine similarity
            if gen_waves.shape[0] > 1:
                cos_sim = F.cosine_similarity(
                    gen_waves[:-1], gen_waves[1:], dim=-1
                )
                all_consec_cos.extend(cos_sim.cpu().tolist())
        except Exception as e:
            print(f"    ⚠ Generation failed for '{text[:30]}...': {e}")
            continue

    all_waves_cat = torch.cat(all_waves, dim=0) if all_waves else torch.zeros(1, 432)
    return all_norms, all_waves_cat, all_consec_cos


def compute_kl_divergence(p_values, q_values, n_bins=50):
    """Compute KL divergence between two sets of values using histogram binning."""
    all_vals = p_values + q_values
    min_val = min(all_vals) - 0.01
    max_val = max(all_vals) + 0.01

    p_hist, _ = np.histogram(p_values, bins=n_bins, range=(min_val, max_val), density=True)
    q_hist, _ = np.histogram(q_values, bins=n_bins, range=(min_val, max_val), density=True)

    # Add small epsilon to avoid log(0)
    eps = 1e-8
    p_hist = p_hist + eps
    q_hist = q_hist + eps

    # Normalize
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()

    # KL divergence
    kl = np.sum(p_hist * np.log(p_hist / q_hist))
    return kl


def main():
    print("=" * 60)
    print("  Phase 9 — Test 1: Wave Distribution Match")
    print("=" * 60)

    device = get_device()
    results = PhaseResults(phase=9, component_name="Wave-Level Generation")

    # Diverse prompts for testing
    prompts = [
        "The future of artificial intelligence is",
        "In the beginning there was",
        "Scientists have discovered that",
        "The relationship between energy and matter",
        "Modern technology relies on advanced",
        "Once upon a time in a distant land",
        "The quantum nature of reality suggests",
        "Education is the foundation of society",
        "Climate change affects global ecosystems",
        "The history of mathematics reveals that",
        "Philosophers have long debated the nature of consciousness",
        "Neural networks learn through gradient descent",
        "The ocean covers most of the earth surface",
        "Music has the power to transform emotions",
        "The development of language shaped human civilization",
        "Gravity is one of the fundamental forces",
        "The speed of light is a universal constant",
        "Democracy requires active participation from citizens",
        "The structure of DNA was discovered in",
        "Innovation drives economic growth and prosperity",
    ] * 5  # 100 prompts total

    # Try to load Phase 9 checkpoint
    try:
        model, chunker, generator, wtt = load_phase9_modules(device=device)
        print("  ✓ Phase 9 checkpoint loaded")
    except Exception as e:
        print(f"  ⚠ No Phase 9 checkpoint: {e}")
        print("  ℹ Loading best available checkpoint + fresh Phase 9 modules")
        from train_wave_gen import build_flux_for_phase9
        model = build_flux_for_phase9(device=device)
        chunker, generator, wtt = build_phase9_modules(device=device)

    # Collect stats
    print("\n  Collecting CSE wave statistics...")
    cse_norms, cse_waves = collect_cse_wave_stats(model, prompts[:20], device)

    print("  Collecting generated wave statistics...")
    gen_norms, gen_waves, consec_cos = collect_generated_wave_stats(
        model, chunker, generator, wtt, prompts, device
    )

    # ── Test 1a: L2 norm distribution similarity ──
    print("\n  ── Test 1a: L2 Norm Distribution ──")
    if len(cse_norms) > 0 and len(gen_norms) > 0:
        kl_norm = compute_kl_divergence(cse_norms, gen_norms)
        cse_mean_norm = np.mean(cse_norms)
        gen_mean_norm = np.mean(gen_norms)
        print(f"    CSE mean L2 norm:  {cse_mean_norm:.4f}")
        print(f"    Gen mean L2 norm:  {gen_mean_norm:.4f}")
        print(f"    KL divergence:     {kl_norm:.4f}")
        norm_pass = kl_norm < 0.5
    else:
        kl_norm = float('inf')
        norm_pass = False
    print(f"    {'✓' if norm_pass else '✗'} KL div < 0.5: {norm_pass}")

    # ── Test 1b: Consecutive cosine similarity ──
    print("\n  ── Test 1b: Consecutive Wave Coherence ──")
    if len(consec_cos) > 0:
        mean_cos = np.mean(consec_cos)
        min_cos = np.min(consec_cos)
        print(f"    Mean consecutive cos sim:  {mean_cos:.4f}")
        print(f"    Min consecutive cos sim:   {min_cos:.4f}")
        coherence_pass = mean_cos > 0.3
    else:
        mean_cos = 0.0
        coherence_pass = False
    print(f"    {'✓' if coherence_pass else '✗'} Mean cos sim > 0.3: {coherence_pass}")

    # ── Test 1c: Per-dimension statistics ──
    print("\n  ── Test 1c: Per-Dimension Statistics ──")
    if cse_waves.shape[0] > 1 and gen_waves.shape[0] > 1:
        cse_mean = cse_waves.mean(dim=0)  # [432]
        cse_std = cse_waves.std(dim=0)    # [432]
        gen_mean = gen_waves.mean(dim=0)  # [432]

        # Check if gen_mean is within 2σ of CSE mean
        within_2sigma = ((gen_mean - cse_mean).abs() < 2 * cse_std + 1e-6).float().mean()
        print(f"    Dims within 2σ of CSE: {within_2sigma:.1%}")
        dim_pass = within_2sigma > 0.5  # At least 50% of dims within 2σ
    else:
        within_2sigma = 0.0
        dim_pass = False
    print(f"    {'✓' if dim_pass else '✗'} >50% dims within 2σ: {dim_pass}")

    # ── Overall KL divergence ──
    print("\n  ── Overall Score ──")
    overall_kl = kl_norm if len(cse_norms) > 0 else float('inf')
    overall_pass = overall_kl < 0.5
    print(f"    Overall KL divergence: {overall_kl:.4f}")
    print(f"    {'✓' if overall_pass else '✗'} PASS: KL div < 0.5")

    # Record results
    results.add_test(
        "Wave L2 Norm Distribution",
        passed=norm_pass,
        score=kl_norm,
        threshold=0.5,
        notes=f"KL divergence between CSE and generated wave norms",
    )
    results.add_test(
        "Consecutive Wave Coherence",
        passed=coherence_pass,
        score=mean_cos,
        threshold=0.3,
        notes=f"Mean cosine similarity between consecutive generated waves",
    )
    results.add_test(
        "Per-Dimension Statistics",
        passed=dim_pass,
        score=float(within_2sigma) if isinstance(within_2sigma, float) else within_2sigma.item(),
        threshold=0.5,
        notes=f"Fraction of wave dimensions within 2σ of CSE stats",
    )
    results.add_test(
        "Overall Wave Distribution Match",
        passed=overall_pass,
        score=overall_kl,
        threshold=0.5,
        notes=f"Overall KL divergence < 0.5",
    )
    results.save()

    # Summary
    all_passed = norm_pass and coherence_pass and dim_pass and overall_pass
    print(f"\n  {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")

    # Assert all criteria (project convention: standalone tests use assert)
    assert norm_pass, f"KL divergence {kl_norm:.4f} >= 0.5 threshold"
    assert coherence_pass, f"Mean consecutive cosine {mean_cos:.4f} <= 0.3 threshold"
    assert dim_pass, f"Only {within_2sigma:.1%} dims within 2σ (need >50%)"
    assert overall_pass, f"Overall KL divergence {overall_kl:.4f} >= 0.5 threshold"

    return all_passed


if __name__ == '__main__':
    passed = main()
    sys.exit(0 if passed else 1)
