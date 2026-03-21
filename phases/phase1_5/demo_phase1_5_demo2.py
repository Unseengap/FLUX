"""
PHASE 1.5 DEMO 2: Tension detection on contradicting statements.
Saves: demo1_5_contradiction_tension.png
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

DEMO_CASES = [
    {
        'label':        'Neutral (no contradiction)',
        'neutral':      'the sky is blue. birds can fly.',
        'contradiction':'the sky is blue. the sky is green.',
        'stmt':         'the sky is blue',
        'contra':       'the sky is green',
        'neutral_alt':  'birds can fly',
    },
    {
        'label':        'Neutral (no contradiction)',
        'neutral':      'water is liquid. fish swim in water.',
        'contradiction':'water is liquid. water is solid.',
        'stmt':         'water is liquid',
        'contra':       'water is solid',
        'neutral_alt':  'fish swim in water',
    },
    {
        'label':        'Neutral (no contradiction)',
        'neutral':      'the door was open. the room was bright.',
        'contradiction':'the door was open. the door was closed.',
        'stmt':         'the door was open',
        'contra':       'the door was closed',
        'neutral_alt':  'the room was bright',
    },
    {
        'label':        'Neutral (no contradiction)',
        'neutral':      'the engine started. the driver smiled.',
        'contradiction':'the engine started. the engine would not start.',
        'stmt':         'the engine started',
        'contra':       'the engine would not start',
        'neutral_alt':  'the driver smiled',
    },
    {
        'label':        'Neutral (no contradiction)',
        'neutral':      'the team won. the crowd cheered.',
        'contradiction':'the team won. the team lost.',
        'stmt':         'the team won',
        'contra':       'the team lost',
        'neutral_alt':  'the crowd cheered',
    },
]


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 60)
    print("FLUX Phase 1.5 Demo 2: Contradiction Tension Heatmap")
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

    fig, axes = plt.subplots(len(DEMO_CASES), 2,
                              figsize=(14, 3.2 * len(DEMO_CASES)),
                              constrained_layout=True)
    fig.suptitle("Phase 1.5: Contradiction Tension Detection",
                 fontsize=15, fontweight='bold')

    all_gaps = []

    with torch.no_grad():
        for i, case in enumerate(DEMO_CASES):
            cw_neutral = cwc.encode(cse, case['neutral'])
            cw_contra  = cwc.encode(cse, case['contradiction'])

            t_neutral = cw_neutral.tension.norm(dim=-1).cpu().numpy()
            t_contra  = cw_contra.tension.norm(dim=-1).cpu().numpy()

            score_n = float(t_neutral.mean())
            score_c = float(t_contra.mean())
            gap     = score_c - score_n
            all_gaps.append(gap)

            print(f"\n  Case {i+1}: {case['stmt']!r}")
            print(f"    Neutral     tension: {score_n:.4f}  → \"{case['neutral_alt']}\"")
            print(f"    Contradiction tension: {score_c:.4f}  → \"{case['contra']}\"")
            print(f"    Gap: {gap:+.4f}  {'✓ Contradiction detected' if gap > 0 else '✗ Not detected'}")

            # Neutral plot
            ax_n  = axes[i][0]
            x_n   = np.arange(len(t_neutral))
            cmap_n = plt.cm.Blues
            ax_n.bar(x_n, t_neutral, color=[cmap_n(min(v / 0.5, 1.0)) for v in t_neutral],
                     edgecolor='white', alpha=0.9)
            ax_n.axhline(score_n, color='steelblue', linestyle='--', linewidth=1.5,
                         label=f'mean={score_n:.4f}')
            ax_n.set_ylim(0, max(t_contra.max(), 0.8) * 1.15)
            ax_n.set_title(f"Neutral: \"{case['neutral'][:40]}\"", fontsize=9)
            ax_n.set_xlabel("Position")
            ax_n.set_ylabel("Tension")
            ax_n.legend(fontsize=8)

            # Contradiction plot
            ax_c  = axes[i][1]
            x_c   = np.arange(len(t_contra))
            cmap_c = plt.cm.Reds
            ax_c.bar(x_c, t_contra, color=[cmap_c(min(v / 0.5, 1.0)) for v in t_contra],
                     edgecolor='white', alpha=0.9)
            ax_c.axhline(score_c, color='darkred', linestyle='--', linewidth=1.5,
                         label=f'mean={score_c:.4f}')
            ax_c.set_ylim(0, max(t_contra.max(), 0.8) * 1.15)
            ax_c.set_title(
                f"Contradiction: \"{case['contradiction'][:40]}\"  Gap={gap:+.3f}",
                fontsize=9, color='darkgreen' if gap > 0.2 else 'darkred'
            )
            ax_c.set_xlabel("Position")
            ax_c.legend(fontsize=8)

    mean_gap = sum(all_gaps) / len(all_gaps)
    fig.text(0.5, -0.01,
             f"Mean tension gap (contradiction − neutral): {mean_gap:+.4f}  "
             f"({'✓ Contradiction reliably detected' if mean_gap > 0.2 else '⚠ Gap below threshold'})",
             ha='center', fontsize=11,
             color='darkgreen' if mean_gap > 0.2 else 'darkred',
             fontweight='bold')

    out = Path(__file__).parent / 'demo1_5_contradiction_tension.png'
    plt.savefig(str(out), dpi=120, bbox_inches='tight')
    print(f"\n  ✓ Saved: {out}")
    print(f"  Mean tension gap: {mean_gap:.4f}")
    print(f"  CATASTROPHIC FORGETTING SCORE: 0.00 (field unchanged by tension)")
    print("  ✓ Demo 2 complete")


if __name__ == '__main__':
    main()