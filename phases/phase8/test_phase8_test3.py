"""
Phase 8 — Test 3: Continual Learning (FLUX wins)

Verifies that FLUXLarge retains knowledge after learning new facts.
GPT-2 cannot learn at inference time at all — FLUX should have
a forgetting score near 0.0 (perfect retention).

Pass Criteria:
  - FLUX forgetting score < 0.10 (less than 10% degradation)
  - FLUX successfully recalls facts A after learning facts B
  - Episodic memory grows correctly
"""

import sys
import torch
from pathlib import Path

_PHASES_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PHASES_DIR.parent
for _phase in ['phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'phase6', 'phase7']:
    _p = str(_PHASES_DIR / _phase)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from flux_large import FLUXLarge
from benchmark_gpt2 import FLUXBenchmark
from flux_utils import get_device, checkpoint_exists


def main():
    print("=" * 60)
    print("  Test 3: Continual Learning — Zero Forgetting")
    print("=" * 60)

    device = get_device()

    # Load model
    if checkpoint_exists(8):
        model = FLUXLarge.from_phase8_checkpoint(device=device)
    else:
        print("  ⚠ No Phase 8 checkpoint — testing with fresh FLUXLarge")
        model = FLUXLarge(device=device)

    # ── Facts A (learn first) ──
    facts_a = [
        "The capital of France is Paris",
        "Water freezes at zero degrees Celsius",
        "Light travels at approximately 300000 km per second",
        "The human genome contains about 3 billion base pairs",
        "Pi is approximately 3.14159",
        "The Eiffel Tower is in Paris France",
        "Oxygen has atomic number 8",
        "The speed of sound is about 343 meters per second",
    ]

    # ── Facts B (learn second — should NOT erase A) ──
    facts_b = [
        "Mars has two moons named Phobos and Deimos",
        "Jupiter is the largest planet in the solar system",
        "DNA was first isolated by Friedrich Miescher in 1869",
        "The deepest point in the ocean is the Challenger Deep",
        "Euler's number e is approximately 2.71828",
        "Mount Everest is the tallest mountain on Earth",
        "Gold has atomic number 79",
        "Saturn has prominent rings made of ice and rock",
    ]

    # Step 1: Learn facts A
    print("\n  Step 1: Learning Facts A...")
    episodic_before = model.episodic_memory.size
    for fact in facts_a:
        model.learn_fact(fact)
    print(f"    Learned {len(facts_a)} facts (episodic: {episodic_before} → {model.episodic_memory.size})")

    # Step 2: Verify recall of facts A
    print("\n  Step 2: Verifying recall of Facts A...")
    recall_before = 0
    for fact in facts_a:
        results = model.query(fact, k=3)
        found = any(fact[:20] in r[0] for r in results) if results else False
        recall_before += int(found)
        print(f"    {'✓' if found else '✗'} {fact[:50]}")
    recall_before_rate = recall_before / len(facts_a)
    print(f"    Recall: {recall_before}/{len(facts_a)} = {recall_before_rate:.1%}")

    # Step 3: Learn facts B
    print("\n  Step 3: Learning Facts B...")
    for fact in facts_b:
        model.learn_fact(fact)
    print(f"    Learned {len(facts_b)} facts (episodic: {model.episodic_memory.size})")

    # Step 4: Re-check recall of facts A
    print("\n  Step 4: Re-checking recall of Facts A (after learning B)...")
    recall_after = 0
    for fact in facts_a:
        results = model.query(fact, k=5)
        found = any(fact[:20] in r[0] for r in results) if results else False
        recall_after += int(found)
        print(f"    {'✓' if found else '✗'} {fact[:50]}")
    recall_after_rate = recall_after / len(facts_a)
    print(f"    Recall: {recall_after}/{len(facts_a)} = {recall_after_rate:.1%}")

    # Compute forgetting score
    if recall_before_rate > 0:
        forgetting = (recall_before_rate - recall_after_rate) / recall_before_rate
    else:
        forgetting = 0.0

    print(f"\n  Results:")
    print(f"    Recall before B: {recall_before_rate:.1%}")
    print(f"    Recall after B:  {recall_after_rate:.1%}")
    print(f"    Forgetting score: {forgetting:.4f}")
    print(f"    Threshold:        < 0.10")

    # Verify episodic memory grew
    expected_entries = len(facts_a) + len(facts_b)
    mem_grew = model.episodic_memory.size >= episodic_before + expected_entries

    passed_forgetting = forgetting < 0.10
    passed_recall = recall_before_rate > 0.5
    passed_memory = mem_grew

    print(f"\n  Checks:")
    print(f"    Forgetting < 0.10:      {'✓' if passed_forgetting else '✗'} ({forgetting:.4f})")
    print(f"    Initial recall > 50%:   {'✓' if passed_recall else '✗'} ({recall_before_rate:.1%})")
    print(f"    Episodic memory grew:   {'✓' if passed_memory else '✗'} ({model.episodic_memory.size} entries)")

    passed = passed_forgetting and passed_recall and passed_memory
    print(f"\n  {'✓ TEST PASSED' if passed else '✗ TEST FAILED'}")

    assert passed_forgetting, f"Forgetting too high: {forgetting:.4f} (threshold: 0.10)"
    assert passed_recall, f"Initial recall too low: {recall_before_rate:.1%}"
    assert passed_memory, "Episodic memory did not grow"

    return forgetting


if __name__ == '__main__':
    main()
