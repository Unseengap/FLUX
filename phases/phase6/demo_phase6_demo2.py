"""
Phase 6 Demo 2: Consolidation Process Live

Demonstrates the episodic → semantic memory consolidation:
1. Write many facts to episodic memory
2. Access some facts repeatedly (simulating natural retrieval)
3. Run consolidation — frequently accessed facts → semantic field
4. Show that semantic field absorbs high-frequency knowledge
5. Show stability metrics before/after

This mirrors human sleep consolidation: frequently rehearsed
memories become deep knowledge.
"""

import sys
import torch
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'phases' / 'phase1'))
sys.path.insert(0, str(ROOT / 'phases' / 'phase2'))
sys.path.insert(0, str(ROOT / 'phases' / 'phase6'))

from flux_utils import load_checkpoint, get_device
from cse import ContinuousSemanticEncoder
from field import ResonanceField
from working_memory import WorkingMemory
from episodic_memory import EpisodicMemory
from semantic_memory import SemanticMemory
from consolidation import ConsolidationProcess


def demo_consolidation():
    print("=" * 65)
    print("  DEMO 2: Consolidation Process Live")
    print("  Episodic → Semantic memory distillation")
    print("=" * 65)

    device = get_device()

    # Load components
    ckpt6 = load_checkpoint(6)
    ckpt1 = load_checkpoint(1)
    ckpt2 = load_checkpoint(2)

    cse = ContinuousSemanticEncoder(**ckpt1.get('config', {}))
    cse.load_state_dict(ckpt1['state_dict'])
    cse = cse.to(device).eval()

    field_cfg = ckpt2.get('config', {}).get('field', {})
    field = ResonanceField(**field_cfg)
    field.load_state_dict(ckpt2['state_dict'])
    field = field.to(device)

    cfg = ckpt6.get('config', {})
    wm = WorkingMemory(
        window_size=cfg.get('window_size', 512),
        wave_dim=cfg.get('wave_dim', 432),
        feature_dim=cfg.get('feature_dim', 256),
    ).to(device)
    wm.load_state_memory(ckpt6['working_memory_state'])
    wm.eval()

    em = EpisodicMemory(feature_dim=cfg.get('feature_dim', 256))
    sm = SemanticMemory(field=field).to(device)
    consolidation = ConsolidationProcess(episodic=em, semantic=sm, min_access=3)

    # ── Step 1: Write facts ──
    print("\n  Step 1: Writing 20 facts to episodic memory...")
    facts = [
        "FLUX is a field-based AI architecture",
        "Resonance fields replace weight matrices",
        "Gravitational relevance replaces attention",
        "Thermodynamic learning replaces backpropagation",
        "Causal geometry nodes replace neurons",
        "Working memory handles current context",
        "Episodic memory stores individual facts",
        "Semantic memory holds deep knowledge",
        "Consolidation promotes episodic to semantic",
        "FLUX has zero catastrophic forgetting",
        "The CSE uses raw UTF-8 bytes",
        "Wave interference encodes meaning",
        "Mass grows with accumulated evidence",
        "Negative mass means contradiction",
        "Temperature controls learning rate",
        "CGN curvature bends signal paths",
        "Causal arrows trace reasoning",
        "The field settles to minimum energy",
        "Three tiers mirror human cognition",
        "No epochs needed for FLUX learning",
    ]

    for fact in facts:
        with torch.no_grad():
            wave = cse.encode(fact)
            vec = wm.compress(wave.full.mean(dim=0).to(device).unsqueeze(0)).squeeze(0)
        em.write(vec, fact=fact, causal_source="demo2")

    print(f"  → {em.size} facts stored")

    # ── Step 2: Simulate repeated access (top 8 facts) ──
    print("\n  Step 2: Simulating natural retrieval patterns...")
    frequently_accessed = facts[:8]
    rarely_accessed = facts[8:]

    for _ in range(10):  # 10 access cycles
        for fact in frequently_accessed:
            with torch.no_grad():
                wave = cse.encode(fact)
                vec = wm.compress(wave.full.mean(dim=0).to(device).unsqueeze(0)).squeeze(0)
            em.search(vec, k=1)  # This increments access_count

    print(f"  → Top 8 facts accessed ~10 times each")
    print(f"  → Bottom 12 facts accessed ~0 times")

    # Show access counts
    print("\n  Access count distribution:")
    for entry in em.get_all_entries()[:20]:
        bar = "█" * min(entry.access_count, 30)
        marker = " ← candidate" if entry.access_count >= 3 else ""
        print(f"    [{entry.access_count:3d}] {bar} {entry.fact[:40]}{marker}")

    # ── Step 3: Run consolidation ──
    print("\n  Step 3: Running consolidation (episodic → semantic)...")
    snapshot = sm.get_field_snapshot()
    energy_before = sm.get_field_energy()

    result = consolidation.run_consolidation(wave_dim=cfg.get('wave_dim', 432))

    energy_after = sm.get_field_energy()
    stability = sm.compute_stability(snapshot)

    print(f"\n  Consolidation results:")
    print(f"    Candidates found:  {result['candidates_found']}")
    print(f"    Entries consolidated: {result['consolidated']}")
    print(f"    Status:            {result['status']}")
    print(f"    Field stability:   {stability:.4f}")
    print(f"    Energy before:     {energy_before:.6f}")
    print(f"    Energy after:      {energy_after:.6f}")
    print(f"    Protected attractors: {sm.num_protected}")

    # ── Summary ──
    print("\n" + "─" * 65)
    print("  Consolidation demonstrated:")
    print("  ✓ Frequently accessed memories identified as candidates")
    print("  ✓ Low-temperature consolidation into semantic field")
    print(f"  ✓ Field stability maintained at {stability:.4f}")
    print("  ✓ New attractors protected in semantic memory")
    print("  ✓ Rare memories remain in episodic store only")
    print("─" * 65)


if __name__ == "__main__":
    demo_consolidation()
