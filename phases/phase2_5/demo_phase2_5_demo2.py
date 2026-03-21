"""
PHASE 2.5 DEMO 2: Analogical Leaps

Zero-shot analogical reasoning on relationships FLUX was never
explicitly trained on.

Shows:
1. Classic word analogies (King:Man::Queen:?)
2. Cross-domain chain matching (physics → math structure)
3. The analogical mapper working in real time
"""

import sys
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
from concept_seeder import OntologicalSeeder, FALLBACK_TRIPLES
from reasoning_curriculum import CurriculumRunner, FALLBACK_GSM8K
from analogical_mapper import AnalogicalMapper, ANALOGY_TEST_PAIRS
from implication import ImplicationChainStore


def main():
    device = get_device()
    print("=" * 60)
    print("FLUX Phase 2.5 Demo 2: Analogical Leaps")
    print("=" * 60)

    # Load models
    ckpt1 = load_checkpoint(1)
    cse   = ContinuousSemanticEncoder(**ckpt1['config'])
    cse.load_state_dict(ckpt1['state_dict'])
    cse   = cse.to(device).eval()
    for p in cse.parameters():
        p.requires_grad = False

    ckpt15 = load_checkpoint(1.5)
    cwc    = CausalWaveChainer(**ckpt15['config'], device=device).to(device)
    cwc.load_state_dict(ckpt15['state_dict'])
    cwc.eval()
    print("  ✓ Models loaded")

    # Build field
    field = SparseResonanceField(
        initial_h=64, initial_w=64, initial_d=64,
        features=512, wave_dim=432,
        checkpoint_dir='checkpoints/',
        device=device,
    ).to(device)
    impl_store = ImplicationChainStore(device=device)

    # Seed field
    print("\n  Seeding ConceptNet triples...")
    seeder = OntologicalSeeder(cse, cwc, field, impl_store, device=device)
    seeder.seed_batch(FALLBACK_TRIPLES, log_every=999, check_growth=False)

    print("\n  Running GSM8K curriculum...")
    curriculum = CurriculumRunner(cse, cwc, field, impl_store, device=device)
    curriculum.run_gsm8k(
        problems=FALLBACK_GSM8K, max_problems=20,
        log_every=999, check_growth=False
    )

    mapper = AnalogicalMapper(field, impl_store, device=device)

    # Build label index
    all_words = set()
    for a, b, c, d in ANALOGY_TEST_PAIRS:
        all_words.update([a, b, c, d])
    for head, rel, tail in FALLBACK_TRIPLES[:20]:
        all_words.update([head, tail])

    label_index = {}
    with torch.no_grad():
        for word in all_words:
            try:
                wave = cse.encode(word)
                cw   = cwc.forward(wave)
                label_index[word] = cw.full.mean(dim=0).cpu()
            except Exception:
                pass
    print(f"\n  ✓ Label index: {len(label_index)} words")

    # ── Demo 1: Classic analogies ──
    print("\n  ── Classic Analogies (A:B::C:?) ──")
    print(f"  {'A':>10} {'B':>10} {'C':>10}  {'Expected':>12}  {'Predicted':>12}  {'Sim':>8}  Result")
    print(f"  {'-'*72}")

    hits = 0
    results_data = []
    for word_a, word_b, word_c, expected in ANALOGY_TEST_PAIRS:
        predicted, sim = mapper.analogy_text(
            cse, cwc, word_a, word_b, word_c, label_index
        )
        correct = (
            predicted.lower() == expected.lower() or
            expected.lower() in predicted.lower()
        )
        if correct:
            hits += 1
        status = "✓" if correct else "~"
        print(f"  {word_a:>10} {word_b:>10} {word_c:>10}  "
              f"{expected:>12}  {predicted[:12]:>12}  {sim:>8.4f}  {status}")
        results_data.append({
            'a': word_a, 'b': word_b, 'c': word_c,
            'expected': expected, 'predicted': predicted,
            'sim': sim, 'correct': correct,
        })

    print(f"\n  Accuracy: {hits}/{len(ANALOGY_TEST_PAIRS)}")

    # ── Demo 2: Cross-domain chain matching ──
    print("\n  ── Cross-Domain Chain Matching ──")
    test_queries = [
        "If the glorp decreases by 5, what happens to the total?",
        "A zorpf moves at constant speed for 3 time units.",
        "Half of the snurps are removed from the container.",
    ]

    print(f"\n  Query → Best chain shape similarity:")
    chain_sims = []
    for query in test_queries:
        with torch.no_grad():
            wave   = cse.encode(query)
            causal = cwc.forward(wave)
        q_vec = causal.full.mean(dim=0)

        matches = mapper.map_causal_chain(q_vec, k=1, depth=2)
        top_sim = matches[0]['similarity'] if matches else 0.0
        chain_sims.append(top_sim)
        print(f"    '{query[:50]}...'")
        print(f"      → Chain shape sim: {top_sim:.4f}")

    # ── Visualization ──
    print("\n  Generating visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Phase 2.5: Analogical Leaps", fontsize=13, fontweight='bold')

    # Plot 1: Analogy results heatmap
    ax1 = axes[0]
    sims = [r['sim'] for r in results_data]
    colors = ['#2ecc71' if r['correct'] else '#e74c3c' for r in results_data]
    labels = [f"{r['a']}:{r['b']}::{r['c']}:?" for r in results_data]
    y_pos  = np.arange(len(results_data))
    ax1.barh(y_pos, sims, color=colors, alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlabel("Similarity to Target")
    ax1.set_title(f"Word Analogies ({hits}/{len(ANALOGY_TEST_PAIRS)} correct)")
    ax1.axvline(0.5, color='gray', linestyle='--', alpha=0.5)

    green_patch = mpatches.Patch(color='#2ecc71', label='Correct')
    red_patch   = mpatches.Patch(color='#e74c3c', label='Near miss')
    ax1.legend(handles=[green_patch, red_patch])

    # Plot 2: Chain matching similarity
    ax2 = axes[1]
    x_pos = np.arange(len(test_queries))
    ax2.bar(x_pos, chain_sims, color='#9b59b6', alpha=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(
        [f"Query {i+1}" for i in range(len(test_queries))],
        fontsize=10,
    )
    ax2.set_ylabel("Chain Shape Similarity")
    ax2.set_title("Cross-Domain Chain Matching\n(fictional words, real logic structure)")
    ax2.axhline(0.5, color='orange', linestyle='--', label='Similarity threshold')
    ax2.legend()
    ax2.set_ylim(0, 1.1)
    for i, sim in enumerate(chain_sims):
        ax2.text(i, sim + 0.03, f"{sim:.3f}", ha='center', fontsize=10)

    plt.tight_layout()
    out = Path(__file__).parent / 'demo2_5_analogical_leaps.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out}")
    print("  ✓ Demo 2 complete")


if __name__ == '__main__':
    main()
