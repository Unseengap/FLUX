"""
demo_phase3_demo2.py — Context-Dependent Generation

Shows that FLUX generates differently for different contexts:
    - 4 semantic clusters (science, code, geography, abstract)
    - Each cluster gets 3 prompts
    - Visualizes the generated wave similarity matrix:
        intra-cluster similarity should be HIGHER than inter-cluster

This demo is the evidence that context collapse did NOT happen.
In the legacy Phase 9, all outputs had cosine > 0.98 — they were all
identical regardless of prompt. Here we should see structure.

Run: python demo_phase3_demo2.py
(Saves similarity matrix to demo_phase3_demo2.png)
"""

import sys
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────
# Path setup
# ─────────────────────────────────────────────
_V2_DIR       = Path(__file__).parent.parent
_PHASE1_DIR   = _V2_DIR / 'phase1'
_PHASE2_DIR   = _V2_DIR / 'phase2'
_PROJECT_ROOT = _V2_DIR.parent

for _p in [str(_PHASE1_DIR), str(_PHASE2_DIR), str(Path(__file__).parent), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cse            import ContinuousSemanticEncoder
from wave_to_field  import WaveToField
from wave_generator import WaveGenerator
from flux_utils     import get_device


# ─────────────────────────────────────────────
# Prompts organised into semantic clusters
# ─────────────────────────────────────────────
CLUSTERS = {
    'Science': [
        "Water freezes at zero degrees Celsius",
        "The speed of light is 299,792,458 m/s",
        "Photosynthesis converts sunlight into energy",
    ],
    'Code': [
        "def hello(): return 'world'",
        "import torch; x = torch.randn(3, 432)",
        "for i in range(10): print(i)",
    ],
    'Geography': [
        "The capital of France is Paris",
        "The Amazon river flows through Brazil",
        "Mount Everest is the tallest mountain",
    ],
    'Abstract': [
        "The future of artificial intelligence",
        "Consciousness emerges from complexity",
        "Language models predict the next token",
    ],
}

ALL_LABELS  = [f"{c[:4]}-{i+1}" for c, prompts in CLUSTERS.items() for i, _ in enumerate(prompts)]
ALL_PROMPTS = [p for prompts in CLUSTERS.values() for p in prompts]
CLUSTER_SIZES = [len(v) for v in CLUSTERS.values()]


def load_components(ckpt_dir: Path, device: str):
    p1   = torch.load(ckpt_dir / 'phase1_v2.phase.pt', map_location='cpu')
    cfg1 = p1['config']
    cse  = ContinuousSemanticEncoder(
        wave_dim=cfg1.get('wave_dim', 432),
        window_size=cfg1.get('window_size', 8),
        stride=cfg1.get('stride', 1),
    )
    cse.load_state_dict(p1['state_dict']['cse'])
    cse.to(device).eval()

    p2   = torch.load(ckpt_dir / 'phase2_v2.phase.pt', map_location='cpu')
    cfg2 = p2['config']
    w2f  = WaveToField(
        wave_dim=cfg2.get('wave_dim', 432),
        field_dim=cfg2.get('field_features', 512),
    )
    w2f.load_state_dict(p2['state_dict']['bridge_wtf'])
    w2f.to(device).eval()

    p3   = torch.load(ckpt_dir / 'phase3_v2.phase.pt', map_location='cpu')
    cfg3 = p3['config']
    gen  = WaveGenerator(
        wave_dim=cfg3.get('wave_dim', 432),
        field_features=cfg3.get('field_features', 512),
        gru_hidden=cfg3.get('gru_hidden', 512),
        gru_layers=cfg3.get('gru_layers', 1),
        dropout=0.0,
    )
    gen.load_state_dict(p3['state_dict']['generator'])
    gen.to(device).eval()

    return cse, w2f, gen


@torch.no_grad()
def get_mean_gen_wave(
    prompt:    str,
    cse:       ContinuousSemanticEncoder,
    w2f:       WaveToField,
    generator: WaveGenerator,
    device:    str,
    n_waves:   int = 10,
) -> torch.Tensor:
    """Get mean wave of generated sequence for a prompt."""
    wave      = cse.encode(prompt)
    mean_wave = wave.full.mean(dim=0).to(device)
    ctx       = w2f(mean_wave)
    waves, _  = generator.generate(field_context=ctx, max_waves=n_waves)
    return F.normalize(waves.mean(dim=0), dim=-1)


def main():
    device   = get_device()
    ckpt_dir = _PROJECT_ROOT / 'checkpoints'

    assert (ckpt_dir / 'phase3_v2.phase.pt').exists(), \
        "Phase 3 checkpoint not found. Run train_generator.py first."

    print("\n")
    print("=" * 65)
    print("  FLUX v2 Phase 3 — Demo 2: Context-Dependent Generation")
    print("=" * 65)
    print(f"  Device: {device}")
    print()

    cse, w2f, generator = load_components(ckpt_dir, device)

    # ── Generate mean wave for each prompt ──
    print("  Generating waves for 12 prompts across 4 semantic clusters...\n")
    gen_waves = []
    for i, (prompt, label) in enumerate(zip(ALL_PROMPTS, ALL_LABELS)):
        wave = get_mean_gen_wave(prompt, cse, w2f, generator, device)
        gen_waves.append(wave)
        print(f"  [{label}] {prompt[:55]}")

    gen_matrix = torch.stack(gen_waves)  # [12, 432]

    # ── Cosine similarity matrix ──
    sim_matrix = (gen_matrix @ gen_matrix.T).cpu().numpy()
    N = len(ALL_PROMPTS)

    # ── Compute intra vs inter cluster statistics ──
    intra_sims, inter_sims = [], []
    idx = 0
    offset = 0
    for size in CLUSTER_SIZES:
        for i in range(offset, offset + size):
            for j in range(offset, offset + size):
                if i != j:
                    intra_sims.append(sim_matrix[i, j])
            for j in range(N):
                if j < offset or j >= offset + size:
                    inter_sims.append(sim_matrix[i, j])
        offset += size

    avg_intra = np.mean(intra_sims) if intra_sims else 0.0
    avg_inter = np.mean(inter_sims) if inter_sims else 0.0
    structure_ok = avg_intra > avg_inter

    print(f"\n  Intra-cluster avg cosine:  {avg_intra:.4f}")
    print(f"  Inter-cluster avg cosine:  {avg_inter:.4f}")
    print(f"  Structure preserved:       {'✓ YES' if structure_ok else '✗ NO'}")
    print()

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Context-Dependent Generation — Wave Similarity\nFLUX v2 Phase 3",
        fontsize=13,
    )

    # Heatmap
    ax1 = axes[0]
    im  = ax1.imshow(sim_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    ax1.set_xticks(range(N))
    ax1.set_yticks(range(N))
    ax1.set_xticklabels(ALL_LABELS, rotation=45, ha='right', fontsize=8)
    ax1.set_yticklabels(ALL_LABELS, fontsize=8)
    ax1.set_title("Generated Wave Cosine Similarity")
    plt.colorbar(im, ax=ax1, label='Cosine Similarity')

    # Draw cluster boundary lines
    boundaries = np.cumsum([0] + CLUSTER_SIZES)[1:-1]
    for b in boundaries:
        ax1.axhline(b - 0.5, color='black', linewidth=1.5)
        ax1.axvline(b - 0.5, color='black', linewidth=1.5)

    # Bar chart: intra vs inter
    ax2 = axes[1]
    cluster_names = list(CLUSTERS.keys())
    x = np.arange(len(cluster_names) + 1)

    # Per-cluster intra similarity
    per_cluster_intra = []
    offset = 0
    for size in CLUSTER_SIZES:
        sims = []
        for i in range(offset, offset + size):
            for j in range(offset, offset + size):
                if i != j:
                    sims.append(sim_matrix[i, j])
        per_cluster_intra.append(np.mean(sims) if sims else 0.0)
        offset += size

    bar_labels  = cluster_names + ['Inter-cluster']
    bar_values  = per_cluster_intra + [avg_inter]
    bar_colors  = ['#2ecc71'] * len(cluster_names) + ['#e74c3c']

    bars = ax2.bar(bar_labels, bar_values, color=bar_colors, alpha=0.85, edgecolor='white')
    ax2.set_ylabel("Avg Cosine Similarity")
    ax2.set_title("Intra vs Inter Cluster Similarity")
    ax2.set_ylim(0, 1.0)
    ax2.axhline(avg_inter, color='#e74c3c', linestyle='--', linewidth=1, alpha=0.5)

    for bar, val in zip(bars, bar_values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha='center', va='bottom', fontsize=9,
        )

    ax2.text(
        0.98, 0.02,
        f"Structure: {'PRESERVED ✓' if structure_ok else 'COLLAPSED ✗'}",
        transform=ax2.transAxes,
        ha='right', va='bottom',
        fontsize=10,
        color='green' if structure_ok else 'red',
        fontweight='bold',
    )

    plt.tight_layout()
    out_path = Path(__file__).parent / 'demo_phase3_demo2.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    plt.show()

    print(f"\n  {'✓ Context structure preserved' if structure_ok else '✗ Context collapsed'}")
    print(f"  (intra={avg_intra:.4f} > inter={avg_inter:.4f})" if structure_ok else
          f"  (intra={avg_intra:.4f} ≤ inter={avg_inter:.4f}) — retrain may help")


if __name__ == '__main__':
    main()
