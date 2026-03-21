"""
Phase 2 Demo 1: Attractor Formation Visualization

Shows the resonance field forming attractors in real-time as
patterns are repeatedly fed in. Visualizes:
  - Energy landscape as 2D heatmap (summed over D axis)
  - Mass accumulation showing where attractors form
  - Statistics printed at each step

Usage: python demo_phase2_demo1.py
"""

import sys
from pathlib import Path

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PHASE_DIR = Path(__file__).parent
PROJECT_ROOT = PHASE_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PHASE_DIR.parent / 'phase1'))

from flux_utils import get_device
from field import ResonanceField
from field_ops import compute_energy_landscape_slice, compute_mass_landscape_slice


def main():
    print("=" * 60)
    print("  Phase 2 Demo 1: Attractor Formation Visualization")
    print("=" * 60)

    device = get_device()
    print(f"  Device: {device}")

    # Small field for fast demo
    H, W, D, F = 16, 16, 16, 128
    field = ResonanceField(h=H, w=W, d=D, features=F).to(device)

    # Create 5 distinct patterns
    torch.manual_seed(42)
    patterns = {
        'Science': torch.randn(field.wave_dim, device=device),
        'Music': torch.randn(field.wave_dim, device=device),
        'Sports': torch.randn(field.wave_dim, device=device),
        'Cooking': torch.randn(field.wave_dim, device=device),
        'History': torch.randn(field.wave_dim, device=device),
    }

    # Try to use CSE for more meaningful patterns
    try:
        from flux_utils import load_checkpoint
        from cse import ContinuousSemanticEncoder
        checkpoint = load_checkpoint(1)
        cse = ContinuousSemanticEncoder(**checkpoint.get('config', {}))
        cse.load_state_dict(checkpoint['state_dict'])
        cse = cse.to(device).eval()
        with torch.no_grad():
            patterns = {
                'Science': cse.encode("Physics describes the fundamental laws of nature and the universe.").full.mean(0),
                'Music': cse.encode("The symphony orchestra performed a beautiful concerto in the hall.").full.mean(0),
                'Sports': cse.encode("The football team scored a winning touchdown in the final quarter.").full.mean(0),
                'Cooking': cse.encode("The chef prepared a delicious gourmet meal with fresh ingredients.").full.mean(0),
                'History': cse.encode("Ancient civilizations built remarkable structures that still stand today.").full.mean(0),
            }
        print("  ✓ Using CSE-encoded patterns")
    except Exception:
        print("  ⚠ Using random patterns (CSE not available)")

    # ── Formation: Feed each pattern 15 times ──
    snapshots = []  # (step, energy_slice, mass_slice, stats)
    step = 0
    repetitions = 15

    print(f"\n  Feeding {len(patterns)} patterns × {repetitions} repetitions = {len(patterns) * repetitions} perturbations")

    for rep in range(repetitions):
        for name, wave in patterns.items():
            field.perturb(wave)
            step += 1

        if (rep + 1) % 3 == 0:
            field.settle(steps=5)

        # Snapshot every 5 reps
        if (rep + 1) % 5 == 0 or rep == 0 or rep == repetitions - 1:
            with torch.no_grad():
                energy_slice = compute_energy_landscape_slice(field, axis='d', index=D // 2).cpu()
                mass_slice = compute_mass_landscape_slice(field, axis='d', index=D // 2).cpu()
            stats = field.get_field_stats()
            snapshots.append((step, energy_slice, mass_slice, stats))
            print(f"  Rep {rep+1}/{repetitions}: attractors={stats['num_attractors']}, "
                  f"energy={stats['total_energy']:.1f}, max_mass={stats['max_mass']:.4f}")

    # ── Final settle ──
    field.settle(steps=20)
    with torch.no_grad():
        energy_final = compute_energy_landscape_slice(field, axis='d', index=D // 2).cpu()
        mass_final = compute_mass_landscape_slice(field, axis='d', index=D // 2).cpu()
    stats_final = field.get_field_stats()
    snapshots.append((step, energy_final, mass_final, stats_final))

    # ── Visualization ──
    print("\n  Generating visualization...")
    n = len(snapshots)
    fig = plt.figure(figsize=(4 * n, 8))
    gs = gridspec.GridSpec(2, n, hspace=0.3, wspace=0.3)

    for i, (s, energy, mass, stats) in enumerate(snapshots):
        # Energy row
        ax_e = fig.add_subplot(gs[0, i])
        im_e = ax_e.imshow(energy.numpy(), cmap='hot', interpolation='nearest')
        ax_e.set_title(f"Energy (step {s})", fontsize=9)
        ax_e.set_xlabel(f"E={stats['total_energy']:.0f}", fontsize=8)
        plt.colorbar(im_e, ax=ax_e, fraction=0.046)

        # Mass row
        ax_m = fig.add_subplot(gs[1, i])
        im_m = ax_m.imshow(mass.numpy(), cmap='Blues', interpolation='nearest')
        ax_m.set_title(f"Mass (step {s})", fontsize=9)
        ax_m.set_xlabel(f"A={stats['num_attractors']}", fontsize=8)
        plt.colorbar(im_m, ax=ax_m, fraction=0.046)

    fig.suptitle("Resonance Field: Attractor Formation Over Time", fontsize=14, fontweight='bold')

    out_path = PHASE_DIR / 'demo1_attractor_formation.png'
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out_path}")

    # ── Print attractor locations ──
    print(f"\n  Final field state:")
    print(f"    Attractors: {stats_final['num_attractors']}")
    print(f"    Total energy: {stats_final['total_energy']:.1f}")
    print(f"    Max mass: {stats_final['max_mass']:.4f}")
    print(f"    State std: {stats_final['state_std']:.6f}")

    # Show where patterns mapped to
    print(f"\n  Pattern locations:")
    for name, wave in patterns.items():
        loc = field.wave_to_field_coords(wave)
        mass_val = field.get_mass_at(loc)
        energy_val = field.get_energy_at(loc)
        print(f"    {name:10s} → ({loc.h:2d}, {loc.w:2d}, {loc.d:2d}) "
              f"mass={mass_val:.4f} energy={energy_val:.4f}")

    print(f"\n  ✓ Demo complete")


if __name__ == '__main__':
    main()
