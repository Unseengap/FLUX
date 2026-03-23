"""
Phase 9 — Demo 3: Thermodynamic Sampling Temperature Trace

Generate with thermodynamic sampling and plot:
- Top: generated text with confidence per word
- Middle: noise level history
- Bottom: confidence history
- Output: matplotlib figure saved to thermo_wave_sampling.png
"""

import sys
import torch
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

from flux_utils import get_device
from train_wave_gen import load_phase9_modules, build_phase9_modules
from wave_sampler import ThermodynamicWaveSampler


def main():
    print("=" * 60)
    print("  Phase 9 — Demo 3: Thermodynamic Wave Sampling Trace")
    print("=" * 60)

    device = get_device()

    # Load modules
    try:
        model, chunker, generator, wtt = load_phase9_modules(device=device)
        print("  ✓ Phase 9 loaded from checkpoint")
    except Exception as e:
        print(f"  ⚠ No Phase 9 checkpoint ({e})")
        print("  ℹ Using fresh FLUXLarge + fresh Phase 9 modules")
        from flux_large import FLUXLarge
        model = FLUXLarge(device=device)
        for param in model.parameters():
            param.requires_grad = False
        chunker, generator, wtt = build_phase9_modules(device=device)

    prompt = "The history of science reveals the importance of curiosity and persistence"
    print(f"\n  Prompt: \"{prompt}\"")

    generator.eval()
    wtt.eval()
    chunker.eval()

    # Generate waves
    with torch.no_grad():
        wave_seq, wave_vec, merged = model._get_context(prompt)
        gen_waves, confs = generator.generate(
            field_context=merged,
            max_waves=30,
            flux_model=None,
            temperature=0.8,
        )

    n_waves = gen_waves.shape[0]
    print(f"  Generated {n_waves} waves")

    # Apply thermodynamic sampling
    sampler = ThermodynamicWaveSampler(
        base_noise=0.1,
        min_noise=0.01,
        max_noise=0.5,
        momentum=0.7,
    )

    decoded_words = []
    for wave, conf in zip(gen_waves, confs):
        sampled = sampler.sample_wave(wave, conf)
        chunk_bytes = wtt.decode(sampled, temperature=0.8)
        decoded_words.append(chunk_bytes.decode('utf-8', errors='replace'))

    noise_history = sampler.get_history()
    confidences = confs

    # Text output
    full_text = prompt + ' ' + ' '.join(decoded_words)
    print(f"\n  Generated text:")
    print(f"  {full_text[:300]}")

    print(f"\n  ── Per-Word Detail ──")
    print(f"  {'Pos':>3} {'Conf':>8} {'Noise':>8} {'Word':<15}")
    print(f"  {'─'*3} {'─'*8} {'─'*8} {'─'*15}")
    for i in range(n_waves):
        noise = noise_history[i] if i < len(noise_history) else 0.0
        print(f"  {i:3d} {confidences[i]:8.3f} {noise:8.4f} {decoded_words[i]:<15}")

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(
            f'Phase 9: Thermodynamic Wave Sampling\n"{prompt[:60]}..."',
            fontsize=12, fontweight='bold',
        )

        x = np.arange(n_waves)

        # ── Panel 1: Words with confidence color coding ──
        ax1 = axes[0]
        colors = [(1.0 - c, c, 0.2) for c in confidences]
        bars = ax1.bar(x, confidences, color=colors, edgecolor='black', linewidth=0.5)
        ax1.set_ylabel('Confidence')
        ax1.set_title('Generated Words (green = confident, red = uncertain)')
        ax1.set_ylim(0, 1.1)
        for i, word in enumerate(decoded_words):
            ax1.text(
                i, confidences[i] + 0.02, word[:10],
                ha='center', va='bottom', fontsize=6, rotation=45,
            )

        # ── Panel 2: Noise level history ──
        ax2 = axes[1]
        if len(noise_history) > 0:
            ax2.plot(
                x[:len(noise_history)], noise_history,
                'r-o', markersize=4, linewidth=1.5, label='Noise level',
            )
            ax2.fill_between(
                x[:len(noise_history)], noise_history,
                alpha=0.2, color='red',
            )
            ax2.axhline(
                y=sampler.base_noise, color='gray', linestyle='--',
                alpha=0.7, label=f'Base noise ({sampler.base_noise})',
            )
        ax2.set_ylabel('Noise Level')
        ax2.set_title('Thermodynamic Noise — Inversely Proportional to Confidence')
        ax2.legend(fontsize=8)
        ax2.set_ylim(0, sampler.max_noise + 0.05)

        # ── Panel 3: Confidence history ──
        ax3 = axes[2]
        ax3.plot(
            x, confidences, 'g-o', markersize=4, linewidth=1.5,
            label='Confidence',
        )
        ax3.fill_between(x, confidences, alpha=0.2, color='green')
        ax3.axhline(
            y=0.1, color='r', linestyle='--', alpha=0.7,
            label='Stop threshold (0.1)',
        )
        ax3.set_xlabel('Wave Position')
        ax3.set_ylabel('Confidence')
        ax3.set_title('Generator Confidence Trace')
        ax3.set_ylim(-0.05, 1.05)
        ax3.legend(fontsize=8)

        plt.tight_layout()
        out_path = Path(__file__).parent / 'thermo_wave_sampling.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"\n  ✓ Figure saved: {out_path}")
        plt.close()

    except ImportError:
        print("\n  ⚠ matplotlib not available — text summary only")

    # Summary stats
    if len(noise_history) > 0:
        import numpy as np
        print(f"\n  ── Thermodynamic Summary ──")
        print(f"    Mean noise:       {np.mean(noise_history):.4f}")
        print(f"    Min noise:        {np.min(noise_history):.4f}")
        print(f"    Max noise:        {np.max(noise_history):.4f}")
        print(f"    Mean confidence:  {np.mean(confidences):.4f}")
        print(f"    Noise-confidence correlation: inverse (expected)")

    print(f"\n  Physics: High confidence → low noise (stay close to prediction)")
    print(f"           Low confidence → high noise (explore alternatives)")


if __name__ == '__main__':
    main()
