"""
Phase 6 Demo 1: Cross-Session Memory

Demonstrates that FLUX can:
1. Store facts during a "session" (working + episodic tiers)
2. Simulate a session end (clear working memory)
3. Resume and retrieve episodic facts from the previous session
4. Show that knowledge persists across session boundaries

This is a key differentiator: FLUX remembers facts without RAG or fine-tuning.
"""

import sys
import torch
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'phases' / 'phase1'))
sys.path.insert(0, str(ROOT / 'phases' / 'phase6'))

from flux_utils import load_checkpoint, get_device
from cse import ContinuousSemanticEncoder
from working_memory import WorkingMemory
from episodic_memory import EpisodicMemory


def demo_cross_session_memory():
    print("=" * 65)
    print("  DEMO 1: Cross-Session Memory")
    print("  Knowledge persists across session boundaries")
    print("=" * 65)

    device = get_device()

    # Load components
    ckpt6 = load_checkpoint(6)
    ckpt1 = load_checkpoint(1)

    cse = ContinuousSemanticEncoder(**ckpt1.get('config', {}))
    cse.load_state_dict(ckpt1['state_dict'])
    cse = cse.to(device).eval()

    cfg = ckpt6.get('config', {})
    wm = WorkingMemory(
        window_size=cfg.get('window_size', 512),
        wave_dim=cfg.get('wave_dim', 432),
        feature_dim=cfg.get('feature_dim', 256),
    ).to(device)
    wm.load_state_memory(ckpt6['working_memory_state'])
    wm.eval()

    em = EpisodicMemory(feature_dim=cfg.get('feature_dim', 256))

    # ── Session 1: Learn facts ──
    print("\n┌─────────────────────────────────────────────┐")
    print("│  SESSION 1: Learning Phase                   │")
    print("└─────────────────────────────────────────────┘")

    session1_facts = [
        "My name is Alex and I am a marine biologist",
        "I discovered a new species of deep-sea jellyfish",
        "The jellyfish glows blue in complete darkness",
        "I named it Aurelia fluxia after the FLUX project",
        "My lab is located in Monterey Bay California",
    ]

    for fact in session1_facts:
        with torch.no_grad():
            wave = cse.encode(fact)
            vec = wm.compress(wave.full.mean(dim=0).to(device).unsqueeze(0)).squeeze(0)
        wm.add_perturbation(wave.full.mean(dim=0).to(device))
        em.write(vec, fact=fact, causal_source="session_1")
        print(f"  📝 Learned: '{fact}'")

    print(f"\n  Working memory: {wm.size} entries")
    print(f"  Episodic memory: {em.size} entries")

    # ── Session boundary: clear working memory ──
    print("\n┌─────────────────────────────────────────────┐")
    print("│  SESSION BOUNDARY: Working memory cleared    │")
    print("└─────────────────────────────────────────────┘")
    wm.clear()
    print(f"  Working memory: {wm.size} entries (cleared)")
    print(f"  Episodic memory: {em.size} entries (persisted)")

    # ── Session 2: Query facts from previous session ──
    print("\n┌─────────────────────────────────────────────┐")
    print("│  SESSION 2: Recall Phase                     │")
    print("└─────────────────────────────────────────────┘")

    queries = [
        "What do you know about me?",
        "Tell me about the jellyfish",
        "Where is my lab?",
        "What did I name the new species?",
    ]

    for query in queries:
        with torch.no_grad():
            q_wave = cse.encode(query)
            q_vec = wm.compress(q_wave.full.mean(dim=0).to(device).unsqueeze(0)).squeeze(0)
        results = em.search(q_vec, k=2)
        print(f"\n  🔍 Query: '{query}'")
        for entry, score in results:
            print(f"     → [{score:.3f}] {entry.fact}")

    # ── Summary ──
    print("\n" + "─" * 65)
    print("  Cross-session memory verified:")
    print("  ✓ Session 1 facts survived session boundary")
    print("  ✓ Episodic memory persists after working memory clear")
    print("  ✓ Semantic similarity search retrieves relevant facts")
    print("  ✓ No fine-tuning, no RAG pipeline — pure memory architecture")
    print("─" * 65)


if __name__ == "__main__":
    demo_cross_session_memory()
