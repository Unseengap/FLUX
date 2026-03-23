"""
Phase 9 — Demo 2: Wave Sequence Visualization

For a single generation:
- Plot wave cosine similarities (consecutive) — shows coherence
- Plot wave norms — shows energy per concept
- Color-code by confidence — green=confident, red=uncertain
- Output: matplotlib figure saved to wave_sequence.png
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
    print("  Phase 9 — Demo 2: Wave Sequence Visualization")
    print("=" * 60)

    device = get_device()

    # Load modules
    try:
        model, chunker, generator, wtt = load_phase9_modules(device=device)
        print("  ✓ Phase 9 loaded from checkpoint")
    except Exception as e:
        print(f"  ⚠ No Phase 9 checkpoint ({e})")
        print("  ℹ Loading best available checkpoint + fresh Phase 9 modules")
        from train_wave_gen import build_flux_for_phase9
        model = build_flux_for_phase9(device=device)
        chunker, generator, wtt = build_phase9_modules(device=device)

    prompt = "The future of artificial intelligence is transforming the world"
    print(f"\n  Prompt: \"{prompt}\"")

    generator.eval()
    wtt.eval()
    chunker.eval()

    with torch.no_grad():
        wave_seq, wave_vec, merged = model._get_context(prompt)
        gen_waves, confs = generator.generate(
            field_context=merged,
            max_waves=25,
            flux_model=None,
            temperature=0.8,
        )

    n_waves = gen_waves.shape[0]
    print(f"  Generated {n_waves} waves")

    # Compute metrics
    norms = gen_waves.norm(dim=-1).cpu().numpy()
    confidences = confs

    if n_waves > 1:
        cos_sims = F.cosine_similarity(
            gen_waves[:-1], gen_waves[1:], dim=-1
        ).cpu().numpy()
    else:
        cos_sims = []

    # Decode waves for labels
    labels = []
    sampler = ThermodynamicWaveSampler()
    for w, c in zip(gen_waves, confidences):
        sampled = sampler.sample_wave(w, c)
        decoded = wtt.decode(sampled, temperature=0.5)
        labels.append(decoded.decode('utf-8', errors='replace')[:12])

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
        fig.suptitle(f'Phase 9: Wave Sequence Visualization\n"{prompt}"',
                     fontsize=12, fontweight='bold')

        x = np.arange(n_waves)

        # ── Panel 1: Wave norms colored by confidence ──
        ax1 = axes[0]
        colors = [(1.0 - c, c, 0.2) for c in confidences]  # red=low, green=high
        ax1.bar(x, norms, color=colors, edgecolor='black', linewidth=0.5)
        ax1.set_ylabel('Wave L2 Norm')
        ax1.set_title('Energy per Concept (color = confidence: green=high, red=low)')
        for i, lbl in enumerate(labels):
            ax1.text(i, norms[i] + 0.02, lbl, ha='center', va='bottom',
                     fontsize=7, rotation=45)

        # ── Panel 2: Consecutive cosine similarity ──
        ax2 = axes[1]
        if len(cos_sims) > 0:
            x_cos = np.arange(len(cos_sims))
            ax2.plot(x_cos, cos_sims, 'b-o', markersize=4, linewidth=1.5)
            ax2.axhline(y=0.3, color='r', linestyle='--', alpha=0.7, label='Coherence threshold (0.3)')
            ax2.fill_between(x_cos, cos_sims, alpha=0.2, color='blue')
            ax2.legend(fontsize=8)
        ax2.set_ylabel('Cosine Similarity')
        ax2.set_title('Coherence: Consecutive Wave Cosine Similarity')
        ax2.set_ylim(-0.2, 1.05)

        # ── Panel 3: Confidence trace ──
        ax3 = axes[2]
        ax3.plot(x, confidences, 'g-o', markersize=4, linewidth=1.5)
        ax3.fill_between(x, confidences, alpha=0.2, color='green')
        ax3.axhline(y=0.1, color='r', linestyle='--', alpha=0.7, label='Stop threshold (0.1)')
        ax3.set_xlabel('Wave Position')
        ax3.set_ylabel('Confidence')
        ax3.set_title('Generator Confidence per Wave')
        ax3.set_ylim(-0.05, 1.05)
        ax3.legend(fontsize=8)

        plt.tight_layout()
        out_path = Path(__file__).parent / 'wave_sequence.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"\n  ✓ Figure saved: {out_path}")
        plt.close()

    except ImportError:
        print("\n  ⚠ matplotlib not available — printing text summary instead")

    # Text summary regardless
    print(f"\n  ── Wave Sequence Summary ──")
    print(f"  {'Pos':>3} {'Norm':>8} {'Conf':>8} {'Decoded':<15}")
    print(f"  {'─'*3} {'─'*8} {'─'*8} {'─'*15}")
    for i in range(n_waves):
        print(f"  {i:3d} {norms[i]:8.4f} {confidences[i]:8.3f} {labels[i]:<15}")

    if len(cos_sims) > 0:
        import numpy as np
        print(f"\n  Consecutive cosine similarity:")
        print(f"    Mean: {np.mean(cos_sims):.4f}")
        print(f"    Min:  {np.min(cos_sims):.4f}")
        print(f"    Max:  {np.max(cos_sims):.4f}")

    print(f"\n  Decoded text: {prompt} {' '.join(labels)}")


if __name__ == '__main__':
    main()
