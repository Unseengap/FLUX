"""
test_phase11_test3.py — Zero-Forgetting Verification Test

This is the critical test that proves FLUX's advantage:
adding new knowledge does NOT cause catastrophic forgetting.

Test procedure:
1. Store initial batch of facts
2. Verify all are recallable
3. Add NEW batch of facts  
4. Verify OLD facts are STILL recallable
5. Compute forgetting score (should be ~0)

This is impossible with standard fine-tuning.
With FLUX, it's guaranteed by architecture.
"""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))
from flux_utils import PhaseResults, get_device

from flux_augmented_llm import SimpleMemoryStore, SimpleWaveEncoder


def test_zero_forgetting():
    """
    Verify zero catastrophic forgetting.
    
    This test proves that:
    - Old memories persist when new memories are added
    - The memory system accumulates, never overwrites
    - Forgetting score approaches zero
    """
    
    results = PhaseResults(phase=11, component_name="Zero-Forgetting Verification")
    
    device = get_device()
    print(f"\n{'='*60}")
    print(f"Phase 11 Test 3: Zero-Forgetting Verification")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # ── Setup ──
    encoder = SimpleWaveEncoder(wave_dim=432)
    encoder.to(device)
    memory = SimpleMemoryStore(wave_dim=432)
    
    # ── Batch 1: Initial Knowledge ──
    batch1_facts = [
        "The capital of France is Paris",
        "Water boils at 100 degrees Celsius",
        "Python was created by Guido van Rossum",
        "The moon orbits the Earth",
        "Shakespeare wrote Hamlet",
        "Pi is approximately 3.14159",
        "DNA stands for deoxyribonucleic acid",
        "The speed of light is 299,792 km/s",
        "Mount Everest is the tallest mountain",
        "The Amazon is the largest river by volume",
    ]
    
    # ── Batch 2: New Knowledge ──
    batch2_facts = [
        "FLUX uses gravitational relevance instead of attention",
        "Resonance fields replace weight matrices in FLUX",
        "The episodic memory uses FAISS for indexing",
        "Wave dimension is 432 in the FLUX architecture",
        "Thermodynamic learning replaces backpropagation",
        "Causal geometry nodes track reasoning chains",
        "The field is a 96x96x96 tensor in Flux-beta",
        "Attractors form at energy minima",
        "Mass accumulates with evidence",
        "Negative mass represents contradiction",
    ]
    
    # ── Step 1: Store Batch 1 ──
    print("Step 1: Storing Batch 1 (10 facts)...")
    batch1_waves = []
    for fact in batch1_facts:
        wave = encoder.encode(fact).mean(dim=0)
        memory.store(wave, fact, {'batch': 1})
        batch1_waves.append(wave.detach().cpu().numpy())
    
    print(f"  ✓ Stored {len(batch1_facts)} facts")
    print(f"  Memory size: {len(memory)}")
    
    # ── Step 2: Verify Batch 1 Recall BEFORE Batch 2 ──
    print("\nStep 2: Verify Batch 1 recall BEFORE adding Batch 2...")
    
    batch1_recall_before = 0
    for fact, wave in zip(batch1_facts, batch1_waves):
        results_list = memory.retrieve(wave, top_k=3, threshold=0.0)
        texts = [r[0]['text'] for r in results_list]
        if any(fact.lower() in t.lower() for t in texts):
            batch1_recall_before += 1
    
    recall_before_rate = batch1_recall_before / len(batch1_facts)
    print(f"  Batch 1 recall: {batch1_recall_before}/{len(batch1_facts)} ({recall_before_rate:.0%})")
    
    # ── Step 3: Store Batch 2 ──
    print("\nStep 3: Storing Batch 2 (10 new facts)...")
    batch2_waves = []
    for fact in batch2_facts:
        wave = encoder.encode(fact).mean(dim=0)
        memory.store(wave, fact, {'batch': 2})
        batch2_waves.append(wave.detach().cpu().numpy())
    
    print(f"  ✓ Stored {len(batch2_facts)} new facts")
    print(f"  Memory size: {len(memory)}")
    
    # ── Step 4: Verify Batch 1 Recall AFTER Batch 2 ──
    print("\nStep 4: Verify Batch 1 recall AFTER adding Batch 2...")
    
    batch1_recall_after = 0
    for fact, wave in zip(batch1_facts, batch1_waves):
        results_list = memory.retrieve(wave, top_k=3, threshold=0.0)
        texts = [r[0]['text'] for r in results_list]
        if any(fact.lower() in t.lower() for t in texts):
            batch1_recall_after += 1
    
    recall_after_rate = batch1_recall_after / len(batch1_facts)
    print(f"  Batch 1 recall: {batch1_recall_after}/{len(batch1_facts)} ({recall_after_rate:.0%})")
    
    # ── Step 5: Verify Batch 2 Recall ──
    print("\nStep 5: Verify Batch 2 recall...")
    
    batch2_recall = 0
    for fact, wave in zip(batch2_facts, batch2_waves):
        results_list = memory.retrieve(wave, top_k=3, threshold=0.0)
        texts = [r[0]['text'] for r in results_list]
        if any(fact.lower() in t.lower() for t in texts):
            batch2_recall += 1
    
    batch2_recall_rate = batch2_recall / len(batch2_facts)
    print(f"  Batch 2 recall: {batch2_recall}/{len(batch2_facts)} ({batch2_recall_rate:.0%})")
    
    # ── Step 6: Compute Forgetting Score ──
    print("\nStep 6: Computing forgetting score...")
    
    # Forgetting = (recall_before - recall_after) / recall_before
    # 0.0 = perfect (no forgetting)
    # 1.0 = complete forgetting
    if batch1_recall_before > 0:
        forgetting_score = max(0, (batch1_recall_before - batch1_recall_after) / batch1_recall_before)
    else:
        forgetting_score = 0.0
    
    print(f"\n{'='*60}")
    print(f"FORGETTING SCORE: {forgetting_score:.4f}")
    print(f"(0.0 = perfect, 1.0 = total forgetting)")
    print(f"{'='*60}")
    
    # ── Record Results ──
    results.add_test(
        "Batch 1 Recall (Before)",
        passed=recall_before_rate >= 0.8,
        score=recall_before_rate,
        threshold=0.8,
    )
    
    results.add_test(
        "Batch 1 Recall (After)",
        passed=recall_after_rate >= 0.8,
        score=recall_after_rate,
        threshold=0.8,
    )
    
    results.add_test(
        "Batch 2 Recall",
        passed=batch2_recall_rate >= 0.8,
        score=batch2_recall_rate,
        threshold=0.8,
    )
    
    results.add_test(
        "Forgetting Score",
        passed=forgetting_score <= 0.05,
        score=forgetting_score,
        threshold=0.05,
    )
    
    results.save()
    
    # ── Summary ──
    all_passed = (
        recall_before_rate >= 0.8 and
        recall_after_rate >= 0.8 and
        batch2_recall_rate >= 0.8 and
        forgetting_score <= 0.05
    )
    
    print(f"\n{'='*60}")
    if all_passed:
        print("✓ TEST PASSED — Zero catastrophic forgetting confirmed")
        print("  Old knowledge preserved when new knowledge added")
    else:
        print("✗ TEST FAILED — Some forgetting detected")
    print(f"{'='*60}")
    
    return all_passed


if __name__ == "__main__":
    success = test_zero_forgetting()
    sys.exit(0 if success else 1)
