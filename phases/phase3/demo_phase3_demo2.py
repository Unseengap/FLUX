"""
Demo 2: Evidence Mass Accumulation Visualization

Feeds astronomy-themed texts into GR, showing how mass tracker
accumulates evidence. Then contradicts one concept to demonstrate
negative mass repulsion.

Output: demo3_mass_distribution.png
"""
import sys, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'phase1'))
sys.path.append(str(Path(__file__).parent.parent / 'phase2'))
from flux_utils import load_checkpoint, get_device
from cse import ContinuousSemanticEncoder
from field import ResonanceField
from gravity import GravitationalRelevance

ASTRONOMY_TEXTS = [
    "Stars are giant balls of hydrogen plasma fusing into helium.",
    "The Sun is a medium-sized yellow dwarf star.",
    "Black holes have gravitational pull so strong light cannot escape.",
    "Neutron stars are incredibly dense collapsed stellar cores.",
    "Galaxies contain billions of stars orbiting a central mass.",
    "The Milky Way is a barred spiral galaxy.",
    "Supernovae are violent stellar explosions that seed heavy elements.",
    "White dwarfs are the remnants of medium-mass stars.",
    "Pulsars emit beams of electromagnetic radiation.",
    "Dark matter makes up about 27 percent of the universe.",
    "The cosmic microwave background is the oldest light in the universe.",
    "Planetary nebulae are shells of gas expelled by dying stars.",
    "Red giants form when hydrogen in the core is exhausted.",
    "Binary star systems contain two stars orbiting each other.",
    "Quasars are extremely luminous active galactic nuclei.",
]

CONTRADICT_TEXT = "Stars are cold rocks floating in empty space."


def main():
    device = get_device()

    # ── Load pipeline ──
    c1 = load_checkpoint(1)
    cse = ContinuousSemanticEncoder(**c1.get('config', {})).to(device)
    cse.load_state_dict(c1['state_dict'])
    cse.eval()

    c2 = load_checkpoint(2)
    field = ResonanceField(**c2.get('config', {}).get('field', {})).to(device)
    field.load_state_dict(c2['state_dict'])
    field.eval()

    gr = GravitationalRelevance(feature_dim=512, device=device).to(device)

    print("\n" + "=" * 65)
    print("  Demo 2: Evidence Mass Accumulation")
    print("=" * 65)

    # ── Feed texts and track mass growth ──
    mass_history = []
    concept_counts = []

    with torch.no_grad():
        for i, text in enumerate(ASTRONOMY_TEXTS):
            wave = cse.encode(text)
            vec = wave.full.mean(dim=0).to(device)
            field_out, _, _ = field.query(vec, k=8)
            _ = gr(field_out.unsqueeze(0))

            stats = gr.mass_tracker.stats()
            mass_history.append(stats.get('mean_mass', 1.0))
            concept_counts.append(stats.get('count', 0))

            if (i + 1) % 5 == 0 or i == 0:
                print(f"    Text {i+1:2d}/{len(ASTRONOMY_TEXTS)}: "
                      f"concepts={stats.get('count', 0)}, "
                      f"mean_mass={stats.get('mean_mass', 0):.4f}, "
                      f"max_mass={stats.get('max_mass', 0):.4f}")

    # ── Contradict a concept ──
    print(f"\n  Contradicting: '{CONTRADICT_TEXT}'")
    contradict_vec = cse.encode(CONTRADICT_TEXT).full.mean(dim=0).to(device)
    stats_before = gr.mass_tracker.stats()
    gr.contradict(contradict_vec, strength=50.0)
    stats_after = gr.mass_tracker.stats()

    print(f"    Before: mean_mass={stats_before.get('mean_mass', 0):.4f}, "
          f"negative={stats_before.get('negative_count', 0)}")
    print(f"    After:  mean_mass={stats_after.get('mean_mass', 0):.4f}, "
          f"negative={stats_after.get('negative_count', 0)}")

    # ── Generate chart ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Mass growth over texts
    axes[0].plot(range(1, len(mass_history) + 1), mass_history,
                 'o-', color='#e74c3c', linewidth=2, markersize=6)
    axes[0].set_xlabel('Texts Fed', fontsize=11)
    axes[0].set_ylabel('Mean Mass', fontsize=11)
    axes[0].set_title('Evidence Mass Growth', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Concept count growth
    axes[1].plot(range(1, len(concept_counts) + 1), concept_counts,
                 's-', color='#3498db', linewidth=2, markersize=6)
    axes[1].set_xlabel('Texts Fed', fontsize=11)
    axes[1].set_ylabel('Concept Count', fontsize=11)
    axes[1].set_title('Concept Accumulation', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Mass distribution histogram
    active_masses = gr.mass_tracker.masses[:gr.mass_tracker.count].detach().cpu().numpy()
    axes[2].hist(active_masses, bins=50, color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[2].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero (negative = contradicted)')
    neg_count = (active_masses < 0).sum()
    axes[2].set_xlabel('Mass', fontsize=11)
    axes[2].set_ylabel('Count', fontsize=11)
    axes[2].set_title(f'Mass Distribution ({neg_count} negative)', fontsize=12)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3, axis='y')

    fig.suptitle('FLUX Phase 3: Evidence Mass Accumulation & Contradiction',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_path = Path(__file__).parent / 'demo3_mass_distribution.png'
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  ✓ Chart saved: {out_path.name}")


if __name__ == "__main__":
    main()
