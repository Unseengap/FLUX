"""
Phase 5 Demo 2: Fast vs Slow Node Activation

Demonstrates the multi-timescale architecture:
- Fast nodes (time_const=0.01): react immediately to input patterns
- Medium nodes (time_const=0.1): accumulate semantic relationships
- Slow nodes (time_const=1.0): build deep conceptual abstractions

Shows activation curves over 100 steps, visualizing how
different timescales respond at different rates.
"""

import sys
import torch
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from multi_timescale import MultiTimescaleCoordinator


def demo_timescale_activation():
    print("=" * 65)
    print("  FLUX Phase 5 Demo 2: Fast vs Slow Node Activation")
    print("=" * 65)

    feature_dim = 512
    mtc = MultiTimescaleCoordinator(
        feature_dim=feature_dim,
        n_fast=16,
        n_medium=8,
        n_slow=4,
    )

    # ─────────────────────────────────────────
    # Measure activation over time
    # ─────────────────────────────────────────
    signal = torch.randn(feature_dim) * 0.5
    mtc.reset_states()

    print(f"\n  Input signal: norm={signal.norm().item():.4f}")
    print(f"  Architecture: {mtc.n_fast} fast + {mtc.n_medium} medium + {mtc.n_slow} slow = {mtc.total_nodes()} nodes")
    print(f"  Total parameters: {mtc.total_params():,}")

    sep = mtc.measure_timescale_separation(signal, max_steps=100)

    # ─────────────────────────────────────────
    # Print activation timeline
    # ─────────────────────────────────────────
    print(f"\n  Activation Timeline (threshold={sep['threshold']:.4f}):")
    print(f"  " + "-" * 60)
    print(f"  {'Step':>6}  {'Fast':>10}  {'Medium':>10}  {'Slow':>10}")
    print(f"  " + "-" * 60)

    steps_to_show = [1, 2, 3, 5, 10, 20, 30, 50, 75, 100]
    for step in steps_to_show:
        if step <= len(sep['fast_activations']):
            fa = sep['fast_activations'][step - 1]
            ma = sep['medium_activations'][step - 1]
            sa = sep['slow_activations'][step - 1]

            # Visual indicators
            f_bar = "█" * min(int(fa / sep['threshold'] * 10), 20)
            m_bar = "█" * min(int(ma / sep['threshold'] * 10), 20)
            s_bar = "█" * min(int(sa / sep['threshold'] * 10), 20)

            print(f"  {step:>6}  {fa:>10.4f}  {ma:>10.4f}  {sa:>10.4f}  | {f_bar}")

    # ─────────────────────────────────────────
    # Threshold crossing report
    # ─────────────────────────────────────────
    print(f"\n  Activation Threshold Crossing (50% of input magnitude):")
    print(f"  " + "-" * 45)
    print(f"    Fast nodes activate at step:   {sep['fast_steps_to_activate']}")
    print(f"    Medium nodes activate at step: {sep['medium_steps_to_activate']}")
    print(f"    Slow nodes activate at step:   {sep['slow_steps_to_activate']}")

    # ─────────────────────────────────────────
    # Generate matplotlib plot if available
    # ─────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        steps = list(range(1, len(sep['fast_activations']) + 1))
        ax.plot(steps, sep['fast_activations'], 'r-', linewidth=2, label=f'Fast (τ=0.01)', alpha=0.9)
        ax.plot(steps, sep['medium_activations'], 'g-', linewidth=2, label=f'Medium (τ=0.1)', alpha=0.9)
        ax.plot(steps, sep['slow_activations'], 'b-', linewidth=2, label=f'Slow (τ=1.0)', alpha=0.9)
        ax.axhline(y=sep['threshold'], color='gray', linestyle='--', alpha=0.5, label='Threshold')

        ax.set_xlabel('Processing Steps', fontsize=12)
        ax.set_ylabel('Activation Magnitude', fontsize=12)
        ax.set_title('FLUX Phase 5: Multi-Timescale CGN Activation', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Annotate threshold crossings
        for name, step_val, color in [
            ('Fast', sep['fast_steps_to_activate'], 'red'),
            ('Medium', sep['medium_steps_to_activate'], 'green'),
            ('Slow', sep['slow_steps_to_activate'], 'blue'),
        ]:
            if 0 < step_val <= 100:
                ax.axvline(x=step_val, color=color, linestyle=':', alpha=0.4)

        plt.tight_layout()
        save_path = Path(__file__).parent / 'demo5_timescale_activation.png'
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"\n  ✓ Chart saved: {save_path.name}")
    except ImportError:
        print(f"\n  ⚠ matplotlib not available — chart skipped")

    # ─────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────
    print(f"\n  Key Insights:")
    print(f"    • Fast nodes (syntax) respond almost immediately")
    print(f"    • Medium nodes (semantics) accumulate over ~10 steps")
    print(f"    • Slow nodes (concepts) build gradually over many steps")
    print(f"    • This mirrors human cognition: fast reflexes + slow reasoning")
    print(f"    • All timescales operate simultaneously — no layer separation")

    print(f"\n  Node Statistics:")
    stats = mtc.stats()
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"    {k:<25}: {v:.4f}")
        else:
            print(f"    {k:<25}: {v}")

    print(f"\n  ✓ Demo 2 complete")


if __name__ == "__main__":
    demo_timescale_activation()

