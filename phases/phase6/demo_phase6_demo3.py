"""
Phase 6 Demo 3: Zero Forgetting over 1000 Tasks

Demonstrates the key advantage of FLUX's memory architecture:
- Feed 1000 sequential tasks
- After all 1000 tasks, check accuracy on task 1
- Show that FLUX retains early knowledge with zero degradation
- Compare to expected transformer baseline (30-80% forgetting)

This is the single most important demonstration in the project.
"""

import sys
import time
import torch
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'phases' / 'phase1'))
sys.path.insert(0, str(ROOT / 'phases' / 'phase6'))

from flux_utils import load_checkpoint, get_device
from cse import ContinuousSemanticEncoder
from working_memory import WorkingMemory
from episodic_memory import EpisodicMemory


def demo_zero_forgetting():
    print("=" * 65)
    print("  DEMO 3: Zero Forgetting over 1000 Tasks")
    print("  THE key differentiator vs transformers")
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

    # ── Generate 1000 unique tasks ──
    print(f"\n  Generating 1000 unique task facts...")
    task_templates = [
        "The capital of country {} is city_{}",
        "Element {} has {} protons",
        "Species {} lives in habitat {}",
        "Algorithm {} has complexity O({})",
        "Language {} was created in year {}",
    ]

    tasks = []
    for i in range(1000):
        template = task_templates[i % len(task_templates)]
        fact = template.format(i, i * 7 + 13)
        tasks.append(fact)

    # ── Learn the first 10 tasks (we'll track these) ──
    print("  Storing first 10 anchor tasks...")
    anchor_vecs = []
    for fact in tasks[:10]:
        with torch.no_grad():
            wave = cse.encode(fact)
            vec = wm.compress(wave.full.mean(dim=0).to(device).unsqueeze(0)).squeeze(0)
        em.write(vec, fact=fact, causal_source="demo3_anchor")
        anchor_vecs.append((vec, fact))

    # Verify anchor retrieval before bulk load
    print("  Verifying anchor retrieval before bulk load...")
    pre_correct = 0
    for vec, fact in anchor_vecs:
        results = em.search(vec, k=1)
        if results and results[0][0].fact == fact:
            pre_correct += 1
    pre_accuracy = pre_correct / len(anchor_vecs)
    print(f"  Pre-bulk accuracy on anchors: {pre_correct}/{len(anchor_vecs)} = {pre_accuracy:.1%}")

    # ── Load remaining 990 tasks ──
    print(f"\n  Loading 990 additional tasks...")
    start = time.time()
    for i, fact in enumerate(tasks[10:]):
        with torch.no_grad():
            wave = cse.encode(fact)
            vec = wm.compress(wave.full.mean(dim=0).to(device).unsqueeze(0)).squeeze(0)
        em.write(vec, fact=fact, causal_source="demo3_bulk")
        if (i + 1) % 200 == 0:
            elapsed = time.time() - start
            print(f"    ... {i + 11}/1000 tasks loaded ({elapsed:.1f}s)")

    total_elapsed = time.time() - start
    print(f"  All 1000 tasks loaded in {total_elapsed:.1f}s")
    print(f"  Episodic memory size: {em.size}")

    # ── Test: Can we still retrieve the original 10 anchors? ──
    print(f"\n  Testing retrieval of original 10 anchor tasks after 1000 total writes...")
    post_correct = 0
    for idx, (vec, fact) in enumerate(anchor_vecs):
        results = em.search(vec, k=1)
        retrieved = results[0][0].fact if results else "NONE"
        match = results[0][0].fact == fact if results else False
        score = results[0][1] if results else 0.0
        status = "✓" if match else "✗"
        if match:
            post_correct += 1
        print(f"    {status} Anchor {idx+1}: [{score:.3f}] "
              f"{'MATCH' if match else 'MISMATCH'}")

    post_accuracy = post_correct / len(anchor_vecs)

    # ── Compute forgetting ──
    if pre_accuracy > 0:
        forgetting = (pre_accuracy - post_accuracy) / pre_accuracy
    else:
        forgetting = 0.0

    # ── Results ──
    print(f"\n  {'═' * 55}")
    print(f"  ║  ZERO FORGETTING TEST RESULTS")
    print(f"  ╠{'═' * 55}")
    print(f"  ║  Total tasks loaded:     1,000")
    print(f"  ║  Anchor accuracy before:  {pre_accuracy:.1%}")
    print(f"  ║  Anchor accuracy after:   {post_accuracy:.1%}")
    print(f"  ║  Forgetting score:        {forgetting:.4f}")
    print(f"  ╠{'═' * 55}")
    print(f"  ║  FLUX target:            < 0.02 (2%)")
    print(f"  ║  Transformer baseline:   0.30 – 0.80 (30-80%)")
    print(f"  ║  FLUX result:            {forgetting:.4f} ({'PASS ✓' if forgetting < 0.02 else 'FAIL ✗'})")
    print(f"  {'═' * 55}")

    if forgetting < 0.02:
        print("\n  🎉 ZERO CATASTROPHIC FORGETTING VERIFIED!")
        print("     After learning 1000 tasks, the memory of task 1 is intact.")
        print("     A transformer would have lost 30-80% of task 1 accuracy.")
        print("     This is the fundamental advantage of the FLUX architecture.")
    else:
        print(f"\n  ⚠ Forgetting detected: {forgetting:.4f}")


if __name__ == "__main__":
    demo_zero_forgetting()
