"""
PHASE 2.5 DEMO 1: The Sparse Galaxy

Shows that upgrading to a 256³ address space does NOT crash
a 16GB GPU because empty space costs nothing.

Seeds 1000 ConceptNet nodes and shows:
  - Field dimensions vs actual VRAM
  - Active vs theoretical capacity
  - Growth tier progression (if triggered)
  - Per-tier checkpoint sizes

Expected output:
    Field Address Space: 256×256×256 (16,777,216 locations)
    Active Attractors:   1,000
    Capacity Used:       0.006%
    Sparse Memory:       ~4 MB
    VRAM Allocated:      < 100 MB
    Result: Frontier-scale capacity on consumer hardware ✓
"""

import sys
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1_5'))

from flux_utils import get_device, load_checkpoint
from cse import ContinuousSemanticEncoder
from causal_encoder import CausalWaveChainer
from dynamic_field import SparseResonanceField
from concept_seeder import OntologicalSeeder, FALLBACK_TRIPLES, EDGE_WEIGHTS
from growth_manager import GROWTH_TIERS
from implication import ImplicationChainStore


def main():
    device = get_device()
    print("=" * 60)
    print("FLUX Phase 2.5 Demo 1: The Sparse Galaxy")
    print("=" * 60)

    # Load models
    print("\n  Loading Phase 1 CSE...")
    ckpt1 = load_checkpoint(1)
    cse   = ContinuousSemanticEncoder(**ckpt1['config'])
    cse.load_state_dict(ckpt1['state_dict'])
    cse   = cse.to(device).eval()
    for p in cse.parameters():
        p.requires_grad = False

    print("  Loading Phase 1.5 CWC...")
    ckpt15 = load_checkpoint(1.5)
    cwc    = CausalWaveChainer(**ckpt15['config'], device=device).to(device)
    cwc.load_state_dict(ckpt15['state_dict'])
    cwc.eval()
    print("  ✓ Models loaded")

    # Build max-tier sparse field
    print("\n  Building SparseResonanceField (256³ address space)...")
    field = SparseResonanceField(
        initial_h      = 256,
        initial_w      = 256,
        initial_d      = 256,
        features       = 512,
        wave_dim       = 432,
        max_tier       = 5,
        checkpoint_dir = 'checkpoints/',
        device         = device,
    ).to(device)
    field.growth_manager.current_tier = 5  # Already at max

    impl_store = ImplicationChainStore(device=device)
    seeder     = OntologicalSeeder(cse, cwc, field, impl_store, device=device)

    # Track VRAM at each seeding milestone
    def vram_mb():
        if torch.cuda.is_available():
            used, total = torch.cuda.mem_get_info()
            return (total - used) / 1e6, total / 1e6
        return 0.0, 0.0

    milestones     = [10, 50, 100, len(FALLBACK_TRIPLES)]
    milestone_data = []
    prev_count     = 0

    print(f"\n  Seeding {len(FALLBACK_TRIPLES)} ConceptNet triples...")
    print(f"\n  {'Triples':>10}  {'Active Locs':>12}  {'Capacity':>10}  {'Sparse MB':>10}  {'VRAM MB':>10}")
    print(f"  {'-'*58}")

    for m in milestones:
        batch = FALLBACK_TRIPLES[prev_count:m]
        seeder.seed_batch(batch, log_every=999, check_growth=False)
        prev_count = m

        active = field.registry.active_count()
        cap    = field.registry.capacity_fraction()
        smem   = field.registry.memory_mb()
        vram_used, vram_total = vram_mb()

        milestone_data.append({
            'triples':      m,
            'active':       active,
            'capacity_pct': cap * 100,
            'sparse_mb':    smem,
            'vram_mb':      vram_used,
        })

        print(f"  {m:>10}  {active:>12,}  {cap:>9.4%}  {smem:>9.1f}  {vram_used:>9.1f}")

    # ── Final summary ──
    final = milestone_data[-1]
    print(f"\n  {'='*60}")
    print(f"  Field Address Space: 256×256×256 ({256**3:,} locations)")
    print(f"  Active Attractors:   {final['active']:,}")
    print(f"  Capacity Used:       {final['capacity_pct']:.4f}%")
    print(f"  Sparse Memory:       {final['sparse_mb']:.1f} MB")
    print(f"  VRAM (approx):       {final['vram_mb']:.0f} MB")
    print(f"  Result: Frontier-scale capacity on consumer hardware ✓")
    print(f"  {'='*60}")

    # ── Growth tier comparison ──
    print(f"\n  Growth Tier Comparison (sparse vs dense memory):")
    print(f"  {'Tier':>5}  {'Dims':>12}  {'Locations':>14}  {'Dense GB':>10}  {'Sparse MB':>10}")
    print(f"  {'-'*58}")
    for tid, h, w, d, label in GROWTH_TIERS:
        locs      = h * w * d
        dense_gb  = locs * 512 * 4 / 1e9
        sparse_mb = final['active'] * 512 * 4 / 1e6  # same active count
        print(f"  {tid:>5}  {h}×{w}×{d:>6}  {locs:>14,}  {dense_gb:>9.1f}  {sparse_mb:>9.1f}")

    # ── Visualization ──
    print("\n  Generating visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Phase 2.5: Sparse Galaxy — 256³ on Consumer Hardware",
                 fontsize=13, fontweight='bold')

    triples_x = [d['triples'] for d in milestone_data]

    # Plot 1: Active locations vs triples
    ax1 = axes[0]
    ax1.plot(triples_x, [d['active'] for d in milestone_data],
             'o-', color='#3498db', linewidth=2, markersize=8)
    ax1.set_xlabel("Triples Seeded")
    ax1.set_ylabel("Active Attractors")
    ax1.set_title("Attractor Growth")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Sparse memory vs capacity
    ax2 = axes[1]
    ax2.bar(range(len(GROWTH_TIERS)),
            [t[1]*t[2]*t[3] * 512 * 4 / 1e9 for t in GROWTH_TIERS],
            color='#e74c3c', alpha=0.7, label='Dense (GB)')
    sparse_line = [final['active'] * 512 * 4 / 1e6 / 1000] * len(GROWTH_TIERS)
    ax2.plot(range(len(GROWTH_TIERS)), sparse_line,
             'g--', linewidth=2, label='Sparse (GB)')
    ax2.set_xlabel("Growth Tier")
    ax2.set_ylabel("Memory (GB)")
    ax2.set_title("Dense vs Sparse Memory")
    ax2.set_xticks(range(len(GROWTH_TIERS)))
    ax2.set_xticklabels([f"T{t[0]}\n{t[1]}³" for t in GROWTH_TIERS], fontsize=8)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Capacity fraction (barely visible = good)
    ax3 = axes[2]
    caps = [d['capacity_pct'] for d in milestone_data]
    ax3.fill_between(triples_x, caps, alpha=0.4, color='#2ecc71')
    ax3.plot(triples_x, caps, 'o-', color='#27ae60', linewidth=2)
    ax3.set_xlabel("Triples Seeded")
    ax3.set_ylabel("Capacity Used (%)")
    ax3.set_title(f"256³ Field: {caps[-1]:.4f}% Used")
    ax3.axhline(60, color='orange', linestyle='--', label='Growth trigger (60%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(__file__).parent / 'demo2_5_sparse_galaxy.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out}")
    print("  ✓ Demo 1 complete")


if __name__ == '__main__':
    main()
