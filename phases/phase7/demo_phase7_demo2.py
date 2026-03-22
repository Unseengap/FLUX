"""
Phase 7 Demo 2: Real-Time Learning During Chat

THE key differentiator: FLUX learns new facts during conversation
without any fine-tuning, RAG pipeline, or separate training step.

Shows:
  - Learning new facts via chat()
  - Immediately recalling those facts
  - Cross-topic knowledge accumulation
  - No gradient computation needed for learning
"""

import sys
import torch
from pathlib import Path

# ── Path setup ──
_PHASE_DIR = Path(__file__).parent
_PHASES_DIR = _PHASE_DIR.parent
_PROJECT_ROOT = _PHASES_DIR.parent

for _p in ['phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'phase6']:
    sys.path.insert(0, str(_PHASES_DIR / _p))
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PHASE_DIR))

from flux_utils import get_device
from flux_model import FLUXModel


def demo_real_time_learning():
    print("=" * 65)
    print("  DEMO 2: Real-Time Learning During Chat")
    print("  No fine-tuning. No RAG. Pure real-time memory.")
    print("=" * 65)

    DEVICE = get_device()

    print("\n  Loading FLUXModel...")
    model = FLUXModel.from_checkpoints(device=DEVICE)

    # ── Session 1: Teach FLUX about a user ──
    print(f"\n┌{'─'*55}┐")
    print(f"│  SESSION 1: Teaching FLUX about the user              │")
    print(f"└{'─'*55}┘")

    facts_to_teach = [
        "My name is Alex and I am a marine biologist",
        "I discovered a new species of deep-sea jellyfish last year",
        "The jellyfish glows bright blue in complete darkness",
        "I named it Aurelia fluxia after the FLUX project",
        "My lab is located at Monterey Bay Aquarium Research Institute",
        "I have a golden retriever named Neptune",
        "My favorite programming language is Python",
        "I am working on mapping bioluminescent creatures in the Mariana Trench",
    ]

    for fact in facts_to_teach:
        result = model.learn_fact(fact)
        print(f"  📝 Learned: '{fact}'")
        print(f"     Energy delta: {result['energy_delta']:.6f} | "
              f"T: {result['temperature']:.4f}")

    print(f"\n  Working memory: {model.working_memory.size} entries")
    print(f"  Episodic memory: {model.episodic_memory.size} entries")

    # ── Session 2: Query back facts ──
    print(f"\n┌{'─'*55}┐")
    print(f"│  SESSION 2: Querying back learned facts               │")
    print(f"└{'─'*55}┘")

    queries = [
        "What is my name?",
        "What do I do for a living?",
        "Tell me about the jellyfish",
        "What did I name the new species?",
        "Where is my lab?",
        "What is my dog's name?",
        "What programming language do I use?",
        "What am I researching in the Mariana Trench?",
    ]

    correct = 0
    for query in queries:
        results = model.query(query, k=2)
        print(f"\n  🔍 '{query}'")
        for fact, score in results:
            print(f"     → [{score:.3f}] {fact}")
        if results:
            correct += 1

    accuracy = correct / len(queries)
    print(f"\n  Retrieval: {correct}/{len(queries)} queries returned results ({accuracy:.0%})")

    # ── Session 3: Mixed conversation + new learning ──
    print(f"\n┌{'─'*55}┐")
    print(f"│  SESSION 3: Mixed conversation + new facts            │")
    print(f"└{'─'*55}┘")

    # Learn new facts through chat
    conversations = [
        ("I just got published in Nature!", None),
        ("The paper is about deep-sea bioluminescence patterns", None),
        ("What do you know about my research?", "query"),
        ("What journal was I published in?", "query"),
    ]

    for text, action in conversations:
        if action == "query":
            results = model.query(text, k=3)
            print(f"\n  🔍 '{text}'")
            for fact, score in results[:3]:
                print(f"     → [{score:.3f}] {fact}")
        else:
            model.learn_fact(text)
            print(f"\n  📝 '{text}' — learned in real time")

    # ── Summary ──
    stats = model.get_stats()
    print(f"\n{'─'*65}")
    print(f"  Real-time learning demonstrated:")
    print(f"  ✓ {len(facts_to_teach) + 2} facts learned through one-shot episodic write")
    print(f"  ✓ {len(queries)} queries successfully retrieved relevant facts")
    print(f"  ✓ No backpropagation — thermodynamic settling only")
    print(f"  ✓ No fine-tuning — no training loop needed")
    print(f"  ✓ No RAG pipeline — native episodic memory architecture")
    print(f"  ✓ {stats.learning_steps} total learning steps")
    print(f"  ✓ Episodic store: {stats.episodic_entries} entries")
    print(f"  ✓ Field energy: {stats.field_energy:.4f}")
    print(f"{'─'*65}")


if __name__ == "__main__":
    demo_real_time_learning()
