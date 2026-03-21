"""
PHASE 1.5 DEMO 1: Causal coherence of ordered vs shuffled sequences.
Saves: demo1_5_order_sensitivity.png
"""

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'phases' / 'phase1'))
sys.path.insert(0, str(ROOT / 'phases' / 'phase1_5'))

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from flux_utils import load_checkpoint
from cse import ContinuousSemanticEncoder
from causal_encoder import CausalWaveChainer

DEMO_SENTENCES = [
    "the dog chased the cat across the yard into the garden",
    "scientists discovered a new species in the deep ocean",
    "the storm moved quickly across the mountains into the valley",
    "she carefully read the instructions before starting the assembly",
    "the chef prepared the meal with fresh ingredients from the market",
]


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 60)
    print("FLUX Phase 1.5 Demo 1: Order Sensitivity Visualization")
    print("=" * 60)

    ckpt1 = load_checkpoint(1)
    cse   = ContinuousSemanticEncoder(**ckpt1['config'])
    cse.load_state_dict(ckpt1['state_dict'])
    cse   = cse.to(device).eval()
    for p in cse.parameters(): p.requires_grad = False

    ckpt15 = load_checkpoint(1.5)
    cwc    = CausalWaveChainer(**ckpt15['config'], device=device).to(device)
    cwc.load_state_dict(ckpt15['state_dict'])
    cwc.eval()

    fig, axes = plt.subplots(len(DEMO_SENTENCES), 2,
                              figsize=(14, 3.5 * len(DEMO_SENTENCES)),
                              constrained_layout=True)
    fig.suptitle("Phase 1.5: Causal Coherence — Ordered vs Shuffled",
                 fontsize=15, fontweight='bold')

    all_gaps = []

    with torch.no_grad():
        for i, sentence in enumerate(DEMO_SENTENCES):
            words  = sentence.split()
            idx    = torch.randperm(len(words)).tolist()
            shuffled = ' '.join([words[j] for j in idx])

            cw_orig  = cwc.encode(cse, sentence)
            cw_shuf  = cwc.encode(cse, shuffled)
            coh_orig = cw_orig.causal_coherence().cpu().numpy()
            coh_shuf = cw_shuf.causal_coherence().cpu().numpy()

            mean_orig = float(coh_orig.mean()) if len(coh_orig) else 0
            mean_shuf = float(coh_shuf.mean()) if len(coh_shuf) else 0
            gap = mean_orig - mean_shuf
            all_gaps.append(gap)

            print(f"\n  Sentence {i+1}: \"{sentence[:50]}...\"")
            print(f"    Original  coherence: {mean_orig:.4f}")
            print(f"    Shuffled  coherence: {mean_shuf:.4f}")
            print(f"    Gap: {gap:+.4f}  {'✓' if gap > 0 else '✗'}")

            # Plot original
            ax_orig = axes[i][0]
            x_orig  = np.arange(len(coh_orig))
            colors_orig = ['#2ecc71' if v > 0.5 else '#e74c3c' if v < 0.2 else '#f39c12'
                           for v in coh_orig]
            ax_orig.bar(x_orig, coh_orig, color=colors_orig, alpha=0.85, edgecolor='white')
            ax_orig.axhline(mean_orig, color='navy', linestyle='--', linewidth=1.5,
                            label=f'mean={mean_orig:.3f}')
            ax_orig.set_ylim(-0.1, 1.1)
            ax_orig.set_title(f"Original (mean={mean_orig:.3f})", fontsize=10)
            ax_orig.set_xlabel("Position")
            ax_orig.set_ylabel("Coherence")
            ax_orig.legend(fontsize=8)
            sent_display = sentence[:45] + ".." if len(sentence) > 45 else sentence
            ax_orig.text(0.5, -0.22, f'"{sent_display}"',
                         transform=ax_orig.transAxes, ha='center', fontsize=7,
                         style='italic', color='#555')

            # Plot shuffled
            ax_shuf = axes[i][1]
            x_shuf  = np.arange(len(coh_shuf))
            colors_shuf = ['#2ecc71' if v > 0.5 else '#e74c3c' if v < 0.2 else '#f39c12'
                           for v in coh_shuf]
            ax_shuf.bar(x_shuf, coh_shuf, color=colors_shuf, alpha=0.85, edgecolor='white')
            ax_shuf.axhline(mean_shuf, color='darkred', linestyle='--', linewidth=1.5,
                            label=f'mean={mean_shuf:.3f}')
            ax_shuf.set_ylim(-0.1, 1.1)
            ax_shuf.set_title(f"Shuffled (mean={mean_shuf:.3f})  Gap={gap:+.3f}",
                              fontsize=10, color='darkred' if gap < 0.3 else 'darkgreen')
            ax_shuf.set_xlabel("Position")
            ax_shuf.legend(fontsize=8)
            shuf_display = shuffled[:45] + ".." if len(shuffled) > 45 else shuffled
            ax_shuf.text(0.5, -0.22, f'"{shuf_display}"',
                         transform=ax_shuf.transAxes, ha='center', fontsize=7,
                         style='italic', color='#555')

    mean_all_gaps = sum(all_gaps) / len(all_gaps)
    fig.text(0.5, -0.01,
             f"Mean coherence gap across all sentences: {mean_all_gaps:+.4f}  "
             f"({'✓ Original > Shuffled' if mean_all_gaps > 0.3 else '⚠ Gap below threshold'})",
             ha='center', fontsize=11,
             color='darkgreen' if mean_all_gaps > 0.3 else 'darkred',
             fontweight='bold')

    legend_patches = [
        mpatches.Patch(color='#2ecc71', label='High coherence (> 0.5)'),
        mpatches.Patch(color='#f39c12', label='Medium coherence'),
        mpatches.Patch(color='#e74c3c', label='Low coherence (< 0.2)'),
    ]
    fig.legend(handles=legend_patches, loc='upper right', fontsize=9)

    out = Path(__file__).parent / 'demo1_5_order_sensitivity.png'
    plt.savefig(str(out), dpi=120, bbox_inches='tight')
    print(f"\n  ✓ Saved: {out}")
    print(f"  Mean coherence gap: {mean_all_gaps:.4f}")
    print("  ✓ Demo 1 complete")


if __name__ == '__main__':
    main()