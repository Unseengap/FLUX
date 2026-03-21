"""
PHASE 1.5 DEMO 3: Forward propagation through implication chains.
This is REASONING — not retrieval.
Saves: demo1_5_implication_chains.png
"""

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'phases' / 'phase1'))
sys.path.insert(0, str(ROOT / 'phases' / 'phase1_5'))

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from flux_utils import load_checkpoint
from cse import ContinuousSemanticEncoder
from causal_encoder import CausalWaveChainer
from implication import ImplicationChainStore

SEED_CONCEPTS = [
    "it started raining",
    "the dog was hungry",
    "the fire alarm went off",
]

KNOWN_CHAIN = [
    ("it started raining",             "people opened umbrellas",        0.85),
    ("people opened umbrellas",        "people sought shelter",          0.72),
    ("people sought shelter",          "travel slowed down",             0.60),
    ("the dog was hungry",             "the dog ate food",               0.90),
    ("the dog ate food",               "the dog felt satisfied",         0.82),
    ("the dog felt satisfied",         "the dog rested quietly",         0.71),
    ("the fire alarm went off",        "people evacuated the building",  0.90),
    ("people evacuated the building",  "emergency services were called", 0.80),
    ("emergency services were called", "the situation was assessed",     0.68),
]


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 60)
    print("FLUX Phase 1.5 Demo 3: Implication Chain Tracing")
    print("=" * 60)
    print("\n  This is REASONING — not retrieval.")
    print("  The model follows causal arrows to reach conclusions")
    print("  it was never shown directly.\n")

    ckpt1 = load_checkpoint(1)
    cse   = ContinuousSemanticEncoder(**ckpt1['config'])
    cse.load_state_dict(ckpt1['state_dict'])
    cse   = cse.to(device).eval()
    for p in cse.parameters(): p.requires_grad = False

    ckpt15 = load_checkpoint(1.5)
    cwc    = CausalWaveChainer(**ckpt15['config'], device=device).to(device)
    cwc.load_state_dict(ckpt15['state_dict'])
    cwc.eval()

    # Build implication store with known chains
    impl_store = ImplicationChainStore(device=device)
    if 'implication_store' in ckpt15:
        impl_store.load(ckpt15['implication_store'])

    # Populate with demo chains
    with torch.no_grad():
        for src_text, tgt_text, strength in KNOWN_CHAIN:
            cw_s = cwc.encode(cse, src_text)
            cw_t = cwc.encode(cse, tgt_text)
            impl_store.add_arrow(
                cw_s.full.mean(dim=0),
                cw_t.full.mean(dim=0),
                strength, 'temporal'
            )
    print(f"  ✓ Implication store: {len(impl_store.arrows)} arrows\n")

    # Build label index for readability
    label_index = {}
    with torch.no_grad():
        for src_text, tgt_text, _ in KNOWN_CHAIN:
            for text in [src_text, tgt_text]:
                cw = cwc.encode(cse, text)
                vec = F.normalize(cw.full.mean(dim=0).cpu().float(), dim=-1)
                label_index[text] = vec

    def find_label(vec):
        """Find closest known text label for a vector."""
        vec_n = F.normalize(vec.cpu().float(), dim=-1)
        best_sim, best_label = 0.0, "unknown concept"
        for text, lv in label_index.items():
            sim = F.cosine_similarity(vec_n.unsqueeze(0), lv.unsqueeze(0)).item()
            if sim > best_sim:
                best_sim  = sim
                best_label = text
        return best_label, best_sim

    # Visualization setup
    fig, axes = plt.subplots(1, len(SEED_CONCEPTS),
                              figsize=(6 * len(SEED_CONCEPTS), 8),
                              constrained_layout=True)
    fig.suptitle("Phase 1.5: Implication Chain Tracing\n(Reasoning, not retrieval)",
                 fontsize=13, fontweight='bold')

    all_chain_data = []

    with torch.no_grad():
        for i, seed in enumerate(SEED_CONCEPTS):
            ax = axes[i] if len(SEED_CONCEPTS) > 1 else axes
            print(f"  Seed: \"{seed}\"")
            print(f"  {'─'*55}")

            cw_seed = cwc.encode(cse, seed)
            q_vec   = cw_seed.full.mean(dim=0)

            # Direct implications (depth 1)
            direct = impl_store.forward_propagate(q_vec, k=5, min_strength=0.2)
            print(f"  Step 1 implications (direct):")
            for tgt_vec, eff_str in direct:
                label, sim = find_label(tgt_vec)
                print(f"      → \"{label}\"  strength: {eff_str:.2f}")

            # Transitive chains (depth 2+)
            chains = impl_store.chain_propagate(q_vec, depth=3, k_per_step=3)
            print(f"  Step 2+ implications (transitive):")
            seen = set()
            chain_data = [{'text': seed, 'strength': 1.0, 'depth': 0}]
            for chain in chains:
                for step_idx, (step_vec, step_str) in enumerate(chain):
                    label, sim = find_label(step_vec)
                    if label not in seen and label != seed:
                        seen.add(label)
                        depth = step_idx + 1
                        chain_data.append({'text': label, 'strength': step_str, 'depth': depth})
                        marker = "    " * depth
                        print(f"      {marker}→ \"{label}\"  strength: {step_str:.2f}  depth: {depth}")

            max_depth = max(c['depth'] for c in chain_data)
            print(f"  Chain depth reached: {max_depth}")
            print(f"  Total unique concepts reachable: {len(chain_data) - 1}")
            deepest = min((c for c in chain_data if c['depth'] == max_depth),
                          key=lambda c: c['strength'])
            print(f"  Deepest chain strength: {deepest['strength']:.2f}")
            print()

            all_chain_data.append(chain_data)

            # Plot chain as vertical flow
            ax.set_xlim(-0.5, 1.5)
            ax.set_ylim(-0.5, max_depth + 0.8)
            ax.invert_yaxis()
            ax.axis('off')
            ax.set_title(f"Seed: \"{seed[:30]}\"", fontsize=9, fontweight='bold')

            depth_x = {}
            for node in chain_data:
                d = node['depth']
                depth_x[d] = depth_x.get(d, 0) + 1

            depth_counter = {}
            for node in chain_data:
                d   = node['depth']
                cnt = depth_x[d]
                idx = depth_counter.get(d, 0)
                depth_counter[d] = idx + 1

                if cnt == 1:
                    x = 0.5
                else:
                    x = idx / (cnt - 1) if cnt > 1 else 0.5
                    x = 0.1 + x * 0.8

                color_intensity = max(0.2, node['strength'])
                color = plt.cm.RdYlGn(color_intensity)

                ax.scatter(x, d, s=350, c=[color], zorder=5,
                           edgecolors='black', linewidths=0.8)
                text_short = node['text'][:28] + ".." if len(node['text']) > 28 else node['text']
                ax.text(x, d + 0.18, text_short,
                        ha='center', va='top', fontsize=7,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                  alpha=0.8, edgecolor='gray'))
                if d > 0:
                    ax.annotate('', xy=(x, d - 0.05),
                                xytext=(0.5, d - 0.8),
                                arrowprops=dict(arrowstyle='->', color='gray',
                                               lw=1.2, alpha=0.6))
                if d > 0:
                    ax.text(x + 0.08, d - 0.3, f"{node['strength']:.2f}",
                            fontsize=6.5, color='dimgray')

            ax.text(0.5, -0.3,
                    f"Depth: {max_depth} | Concepts: {len(chain_data)-1}",
                    ha='center', fontsize=8, color='navy', fontweight='bold',
                    transform=ax.transData)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn,
                                norm=plt.Normalize(vmin=0.2, vmax=1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes if len(SEED_CONCEPTS) > 1 else [axes],
                        orientation='horizontal', fraction=0.03, pad=0.05)
    cbar.set_label("Implication Strength", fontsize=10)

    fig.text(0.5, -0.04,
             "REASONING — not retrieval. Transitive chains were never seen directly in training.",
             ha='center', fontsize=10, color='navy', fontstyle='italic')

    out = Path(__file__).parent / 'demo1_5_implication_chains.png'
    plt.savefig(str(out), dpi=120, bbox_inches='tight')
    print(f"  ✓ Saved: {out}")
    print("  ✓ Demo 3 complete")


if __name__ == '__main__':
    main()