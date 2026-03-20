"""
PHASE 1 DEMO 1: Visualize semantic waves for sample text.

Run: python demo_phase1_demo1.py

Produces a matplotlib heatmap showing wave components for a sample sentence.
Each row = one byte position in the sequence.
Each column group = one wave dimension (phonetic, syntactic, semantic, temporal, intensity).
Color intensity = wave amplitude.
"""

import sys
from pathlib import Path

PHASE_DIR = Path(__file__).parent
PROJECT_ROOT = PHASE_DIR.parent.parent
sys.path.insert(0, str(PHASE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from cse import ContinuousSemanticEncoder
from wave_types import WAVE_DIMS
from flux_utils import load_checkpoint


def main():
    print("=" * 60)
    print("FLUX Phase 1 Demo 1: Semantic Wave Visualization")
    print("=" * 60)

    # ── Load trained CSE ──
    print("\n  Loading Phase 1 checkpoint...")
    checkpoint = load_checkpoint(1)
    config = checkpoint['config']
    cse = ContinuousSemanticEncoder(**config, device='cpu')
    cse.load_state_dict(checkpoint['state_dict'])
    cse.eval()

    # ── Encode sample texts ──
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Hello, world!",
        "AI is transforming technology",
    ]

    fig = plt.figure(figsize=(18, 4 * len(texts)))
    gs = gridspec.GridSpec(len(texts), 5, hspace=0.4, wspace=0.3)

    dim_names = list(WAVE_DIMS.keys())
    dim_sizes = list(WAVE_DIMS.values())

    for row, text in enumerate(texts):
        print(f"\n  Encoding: \"{text}\"")
        with torch.no_grad():
            wave = cse.encode(text)

        wave_parts = [wave.phonetic, wave.syntactic, wave.semantic, wave.temporal, wave.intensity]
        print(f"    Wave shape: [{wave.seq_len}, {sum(dim_sizes)}]")

        for col, (name, part) in enumerate(zip(dim_names, wave_parts)):
            ax = fig.add_subplot(gs[row, col])
            data = part.cpu().numpy()
            im = ax.imshow(data, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_title(f"{name} [{dim_sizes[col]}]", fontsize=10)
            if col == 0:
                # Label y-axis with byte characters
                byte_labels = list(text.encode('utf-8'))
                tick_positions = list(range(0, len(byte_labels), max(1, len(byte_labels) // 10)))
                ax.set_yticks(tick_positions)
                ax.set_yticklabels([chr(byte_labels[i]) if 32 <= byte_labels[i] < 127 else '·'
                                    for i in tick_positions], fontsize=7)
            else:
                ax.set_yticks([])
            ax.set_xticks([])

    plt.suptitle("FLUX CSE: Semantic Wave Components per Byte Position", fontsize=14, y=1.02)

    out_path = PHASE_DIR / 'demo1_wave_visualization.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Visualization saved to: {out_path}")
    print("  ✓ Demo 1 complete")

    # Also show statistics
    print("\n  Wave Statistics:")
    for text in texts:
        with torch.no_grad():
            wave = cse.encode(text)
        full = wave.full
        print(f"    \"{text[:40]}...\"")
        print(f"      Mean: {full.mean().item():.4f}  Std: {full.std().item():.4f}  "
              f"Min: {full.min().item():.4f}  Max: {full.max().item():.4f}")


if __name__ == '__main__':
    main()
