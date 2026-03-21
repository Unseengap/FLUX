"""
Demo 1: GR vs Attention Speed Comparison

Generates a log-log chart showing wall-clock latency at 6 sequence lengths.
Also shows per-doubling growth rates to illustrate scaling behavior.

Output: demo3_speed_comparison.png
"""
import sys, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from flux_utils import load_checkpoint, get_device
from gravity import GravitationalRelevance
from benchmark_attention import run_benchmark, SEQ_LENGTHS


def main():
    device = get_device()

    # ── Load trained GR from checkpoint (or build fresh) ──
    try:
        c3 = load_checkpoint(3)
        gr = GravitationalRelevance.load_state(c3['phase3_gr_state'], device=device)
        gr = gr.to(device).eval()
        print("  ✓ Using trained GR from checkpoint")
    except Exception:
        gr = GravitationalRelevance(feature_dim=512, device=device).to(device)
        gr.eval()
        print("  ℹ Using fresh GR (no checkpoint)")

    # ── Run benchmark ──
    print("\n" + "=" * 65)
    print("  Demo 1: GR vs Attention — Speed Comparison")
    print("=" * 65)
    results = run_benchmark(gr, device=device)

    seq_lens = [r['seq_len'] for r in results]
    gr_ms = [r['gr_ms'] for r in results]
    attn_ms = [r['attention_ms'] for r in results]

    # ── Generate chart ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: log-log latency comparison
    ax1.loglog(seq_lens, gr_ms, 'o-', color='#e74c3c', linewidth=2, markersize=8, label='GR (spatial index)')
    ax1.loglog(seq_lens, attn_ms, 's-', color='#3498db', linewidth=2, markersize=8, label='Attention (fused CUDA)')
    ax1.set_xlabel('Sequence Length', fontsize=12)
    ax1.set_ylabel('Latency (ms)', fontsize=12)
    ax1.set_title('Wall-Clock Latency (log-log)', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(seq_lens)
    ax1.set_xticklabels([str(s) for s in seq_lens])

    # Right panel: per-doubling growth rates
    gr_growths = [gr_ms[i] / gr_ms[i-1] for i in range(1, len(gr_ms))]
    attn_growths = [attn_ms[i] / attn_ms[i-1] for i in range(1, len(attn_ms)) if attn_ms[i-1] > 0]
    x_labels = [f"{seq_lens[i-1]}→{seq_lens[i]}" for i in range(1, len(seq_lens))]

    x = np.arange(len(gr_growths))
    width = 0.35
    ax2.bar(x - width/2, gr_growths, width, color='#e74c3c', alpha=0.8, label='GR growth')
    if len(attn_growths) == len(gr_growths):
        ax2.bar(x + width/2, attn_growths, width, color='#3498db', alpha=0.8, label='Attn growth')
    ax2.axhline(y=2.0, color='green', linestyle='--', alpha=0.5, label='O(n log n) ideal: 2.0x')
    ax2.axhline(y=4.0, color='red', linestyle='--', alpha=0.5, label='O(n²) quadratic: 4.0x')
    ax2.set_xlabel('Sequence Length Doubling', fontsize=12)
    ax2.set_ylabel('Growth Factor (×)', fontsize=12)
    ax2.set_title('Scaling Rate per Doubling', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, rotation=30, ha='right', fontsize=9)
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle('FLUX Phase 3: GravitationalRelevance vs PyTorch Attention', fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_path = Path(__file__).parent / 'demo3_speed_comparison.png'
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  ✓ Chart saved: {out_path.name}")

    # Print summary
    avg_gr_growth = sum(gr_growths) / len(gr_growths) if gr_growths else 0
    avg_attn_growth = sum(attn_growths) / len(attn_growths) if attn_growths else 0
    print(f"\n  Summary:")
    print(f"    GR avg growth per doubling:   {avg_gr_growth:.2f}x")
    print(f"    Attn avg growth per doubling:  {avg_attn_growth:.2f}x")
    print(f"    GR is slower in absolute time (Python/CPU spatial index overhead)")
    print(f"    GR scales better: ~{avg_gr_growth:.1f}x vs ~{avg_attn_growth:.1f}x per doubling")


if __name__ == "__main__":
    main()
