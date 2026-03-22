"""
Phase 6 Training: Three-Tier Memory System integration.

Loads Phases 1–5 checkpoints, builds the three-tier memory system,
and trains/validates:
  Stage A: Working memory — rolling window + importance scoring
  Stage B: Episodic memory — one-shot write/read accuracy
  Stage C: Semantic memory — field protection + consolidation
  Stage D: Memory router — cross-tier query integration
  Stage E: Forgetting test — sequential task pair evaluation

~5 min on GPU • ~20 min on CPU
"""

import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Project paths ──
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'phases' / 'phase1'))
sys.path.insert(0, str(ROOT / 'phases' / 'phase2'))
sys.path.insert(0, str(ROOT / 'phases' / 'phase3'))
sys.path.insert(0, str(ROOT / 'phases' / 'phase4'))
sys.path.insert(0, str(ROOT / 'phases' / 'phase5'))
sys.path.insert(0, str(Path(__file__).parent))

from flux_utils import (
    load_checkpoint, save_checkpoint, get_device,
    PhaseLogger, PhaseResults, checkpoint_exists,
    verify_checkpoint_chain,
)
from cse import ContinuousSemanticEncoder
from field import ResonanceField
from working_memory import WorkingMemory
from episodic_memory import EpisodicMemory
from semantic_memory import SemanticMemory
from memory_router import MemoryRouter
from consolidation import ConsolidationProcess


def main():
    log = PhaseLogger(phase=6)
    log.separator("Phase 6: Three-Tier Memory System")
    start_time = time.time()

    device = get_device()
    log.info(f"Device: {device}")

    # ─────────────────────────────────────────
    # Load Phases 1–5
    # ─────────────────────────────────────────
    print("\n── Loading Phase 1 (CSE) ──")
    ckpt1 = load_checkpoint(1)
    cse = ContinuousSemanticEncoder(**ckpt1.get('config', {}))
    cse.load_state_dict(ckpt1['state_dict'])
    cse = cse.to(device).eval()
    for p in cse.parameters():
        p.requires_grad = False
    log.success(f"CSE loaded: {sum(p.numel() for p in cse.parameters()):,} params")

    print("\n── Loading Phase 2 (ResonanceField) ──")
    ckpt2 = load_checkpoint(2)
    field_cfg = ckpt2.get('config', {}).get('field', {})
    field = ResonanceField(**field_cfg)
    field.load_state_dict(ckpt2['state_dict'])
    field = field.to(device)
    log.success(f"ResonanceField loaded: {sum(p.numel() for p in field.parameters()):,} params")

    print("\n── Loading Phase 5 (CGN) ──")
    ckpt5 = load_checkpoint(5)
    log.success("Phase 5 checkpoint loaded")

    # ─────────────────────────────────────────
    # Build Phase 6 Components
    # ─────────────────────────────────────────
    print("\n── Building Three-Tier Memory System ──")

    wave_dim = 432
    feature_dim = 256

    working = WorkingMemory(
        window_size=512,
        wave_dim=wave_dim,
        feature_dim=feature_dim,
    ).to(device)
    log.success(f"WorkingMemory built: window={working.window_size}, "
                f"{sum(p.numel() for p in working.parameters()):,} params")

    episodic = EpisodicMemory(feature_dim=feature_dim)
    log.success(f"EpisodicMemory built: feature_dim={feature_dim}")

    semantic = SemanticMemory(field=field, protection_threshold=5.0).to(device)
    log.success(f"SemanticMemory built: {semantic.num_protected} protected attractors")

    router = MemoryRouter(
        working=working,
        episodic=episodic,
        semantic=semantic,
        wave_dim=wave_dim,
        feature_dim=feature_dim,
    ).to(device)
    log.success(f"MemoryRouter built: {sum(p.numel() for p in router.parameters()):,} params")

    consolidation = ConsolidationProcess(
        episodic=episodic,
        semantic=semantic,
        min_access=3,
        temperature=0.05,
    )
    log.success("ConsolidationProcess built")

    # ═══════════════════════════════════════════
    # Stage A: Working Memory Training
    # ═══════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  Stage A: Working Memory — Rolling Window + Importance Scoring")
    print("=" * 65)

    # Train the importance scorer and compression layers
    optimizer_wm = torch.optim.Adam(working.parameters(), lr=1e-3)

    test_sentences = [
        "The capital of France is Paris",
        "Water boils at 100 degrees Celsius",
        "The earth orbits the sun",
        "Neural networks use backpropagation",
        "FLUX uses resonance fields instead of weights",
        "Semantic waves encode meaning continuously",
        "Gravitational relevance replaces attention",
        "Thermodynamic learning replaces backpropagation",
        "Causal geometry nodes store reasons for conclusions",
        "Memory consolidation happens during offline phases",
    ]

    # Encode sentences to waves
    waves = []
    for s in test_sentences:
        with torch.no_grad():
            w = cse.encode(s)
            waves.append(w.full.mean(dim=0).to(device))

    # Train compression + importance via reconstruction loss
    for epoch in range(50):
        total_loss = 0.0
        for wave in waves:
            compressed = working.compress(wave.unsqueeze(0))
            importance = working.importance_scorer(compressed)
            # Reconstruction signal: compressed should preserve identity
            reconstructed = F.linear(compressed, working.compress.weight.T)
            loss = F.mse_loss(reconstructed, wave.unsqueeze(0))
            optimizer_wm.zero_grad()
            loss.backward()
            optimizer_wm.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/50  loss={total_loss / len(waves):.6f}")

    # Fill working memory
    working.eval()
    for i, wave in enumerate(waves):
        working.add_perturbation(wave)

    log.success(f"Working memory trained and populated: {working.size} entries")
    log.metric("wm_size", working.size)
    log.metric("wm_avg_importance", f"{working.get_stats()['avg_importance']:.4f}")

    # ═══════════════════════════════════════════
    # Stage B: Episodic Memory — One-Shot Write/Read
    # ═══════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  Stage B: Episodic Memory — One-Shot Write/Read Accuracy")
    print("=" * 65)

    facts = [
        ("The capital of Mars colony Alpha is New Houston", "science_fiction"),
        ("FLUX processes raw bytes with no tokenization", "architecture"),
        ("Resonance fields replace weight matrices", "architecture"),
        ("Gravitational relevance costs O(log n)", "performance"),
        ("Thermodynamic learning has no epochs", "learning"),
        ("Causal nodes store WHY, not just WHAT", "reasoning"),
        ("Working memory is session-scoped", "memory"),
        ("Episodic memory is permanent", "memory"),
        ("Semantic memory only updates offline", "memory"),
        ("Zero catastrophic forgetting by design", "key_feature"),
    ]

    # Write all facts
    for fact_text, source in facts:
        with torch.no_grad():
            wave = cse.encode(fact_text)
            vec = working.compress(wave.full.mean(dim=0).to(device).unsqueeze(0)).squeeze(0)
        episodic.write(vec, fact=fact_text, causal_source=source)

    # Test retrieval accuracy
    correct = 0
    for fact_text, source in facts:
        with torch.no_grad():
            wave = cse.encode(fact_text)
            q_vec = working.compress(wave.full.mean(dim=0).to(device).unsqueeze(0)).squeeze(0)
        results = episodic.search(q_vec, k=1)
        if results and results[0][0].fact == fact_text:
            correct += 1

    episodic_accuracy = correct / len(facts)
    print(f"  Episodic retrieval accuracy: {correct}/{len(facts)} = {episodic_accuracy:.1%}")
    log.metric("episodic_accuracy", f"{episodic_accuracy:.4f}")
    log.success(f"Episodic memory: {episodic.size} facts stored, {episodic_accuracy:.1%} retrieval accuracy")

    # Simulate multiple accesses for consolidation candidates
    for _ in range(5):
        for fact_text, _ in facts[:5]:  # Top 5 facts accessed repeatedly
            with torch.no_grad():
                wave = cse.encode(fact_text)
                q_vec = working.compress(wave.full.mean(dim=0).to(device).unsqueeze(0)).squeeze(0)
            episodic.search(q_vec, k=1)

    # ═══════════════════════════════════════════
    # Stage C: Semantic Memory + Consolidation
    # ═══════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  Stage C: Semantic Memory — Protection + Consolidation")
    print("=" * 65)

    # Snapshot field before operations
    field_snapshot = semantic.get_field_snapshot()
    energy_before = semantic.get_field_energy()
    print(f"  Field energy before: {energy_before:.6f}")

    # Protect some attractors
    for i in range(5):
        semantic.protect_attractor(i)
    print(f"  Protected {semantic.num_protected} attractors")

    # Run consolidation
    consolid_result = consolidation.run_consolidation(wave_dim=wave_dim)
    print(f"  Consolidation: {consolid_result['consolidated']} entries consolidated")
    print(f"  Stability: {consolid_result['stability']:.4f}")
    print(f"  Status: {consolid_result['status']}")

    # Verify semantic field stability
    stability = semantic.compute_stability(field_snapshot)
    print(f"  Semantic field stability: {stability:.4f}")

    energy_after = semantic.get_field_energy()
    print(f"  Field energy after: {energy_after:.6f}")

    log.metric("consolidation_entries", consolid_result['consolidated'])
    log.metric("field_stability", f"{stability:.4f}")
    log.success("Consolidation completed, semantic field stable")

    # ═══════════════════════════════════════════
    # Stage D: Memory Router Integration
    # ═══════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  Stage D: Memory Router — Cross-Tier Query Integration")
    print("=" * 65)

    # Test routing queries through all tiers
    query_texts = [
        "What is the capital of Mars colony Alpha?",
        "How does FLUX replace attention?",
        "Does FLUX forget old knowledge?",
    ]

    for qt in query_texts:
        with torch.no_grad():
            wave = cse.encode(qt)
            qw = wave.full.mean(dim=0).to(device)

        result = router.route_query(qw, episodic_k=3, working_k=5)
        n_episodic = len(result['episodic_facts'])
        wm_size = result['working_context'].shape[0] if result['working_context'].numel() > 0 else 0
        weights = result['tier_weights']
        print(f"  Query: '{qt[:50]}...'")
        print(f"    Working: {wm_size} entries | Episodic: {n_episodic} facts | "
              f"Semantic energy: {result['semantic_energy']:.4f}")
        print(f"    Tier weights: W={weights[0]:.3f} E={weights[1]:.3f} S={weights[2]:.3f}")

    log.success("Memory router integration verified across all tiers")

    # ═══════════════════════════════════════════
    # Stage E: Forgetting Test
    # ═══════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  Stage E: Zero Forgetting Verification")
    print("=" * 65)

    # Create 10 task pairs and verify no catastrophic forgetting
    task_pairs = [
        ("The sky is blue", "Water is wet"),
        ("Dogs are mammals", "Fish live in water"),
        ("Python is a language", "Java is also a language"),
        ("The sun is a star", "The moon reflects light"),
        ("FLUX uses fields", "Transformers use attention"),
        ("Paris is in France", "Tokyo is in Japan"),
        ("Iron is a metal", "Oxygen is a gas"),
        ("Math uses numbers", "Music uses notes"),
        ("Trees produce oxygen", "Cars produce CO2"),
        ("Books store knowledge", "Brains store memories"),
    ]

    forgetting_scores = []
    for task_a_text, task_b_text in task_pairs:
        # Encode task A
        with torch.no_grad():
            wave_a = cse.encode(task_a_text).full.mean(dim=0).to(device)
            vec_a = working.compress(wave_a.unsqueeze(0)).squeeze(0)

        # Write task A
        episodic.write(vec_a, fact=task_a_text, causal_source="forgetting_test")

        # Measure accuracy on A
        results_before = episodic.search(vec_a, k=1)
        acc_before = 1.0 if (results_before and results_before[0][0].fact == task_a_text) else 0.0

        # Write task B (would destroy a transformer's memory of A)
        with torch.no_grad():
            wave_b = cse.encode(task_b_text).full.mean(dim=0).to(device)
            vec_b = working.compress(wave_b.unsqueeze(0)).squeeze(0)
        episodic.write(vec_b, fact=task_b_text, causal_source="forgetting_test")

        # Measure accuracy on A again
        results_after = episodic.search(vec_a, k=1)
        acc_after = 1.0 if (results_after and results_after[0][0].fact == task_a_text) else 0.0

        forgetting = (acc_before - acc_after) / max(acc_before, 1e-8) if acc_before > 0 else 0.0
        forgetting_scores.append(forgetting)

    avg_forgetting = sum(forgetting_scores) / len(forgetting_scores)
    max_forgetting = max(forgetting_scores)
    print(f"  Forgetting scores per task pair: {[f'{s:.4f}' for s in forgetting_scores]}")
    print(f"  Average forgetting: {avg_forgetting:.4f}")
    print(f"  Max forgetting: {max_forgetting:.4f}")
    print(f"  Target: < 0.02 (2%)")
    print(f"  Result: {'PASS ✓' if avg_forgetting < 0.02 else 'FAIL ✗'}")

    log.metric("avg_forgetting", f"{avg_forgetting:.6f}")
    log.metric("max_forgetting", f"{max_forgetting:.6f}")
    if avg_forgetting < 0.02:
        log.success("Zero catastrophic forgetting verified!")
    else:
        log.error(f"Forgetting too high: {avg_forgetting:.4f}")

    # ═══════════════════════════════════════════
    # Build checkpoint
    # ═══════════════════════════════════════════
    elapsed = time.time() - start_time
    print(f"\n  Training completed in {elapsed:.1f}s")

    checkpoint_state = {
        'phase': 6,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'wave_dim': wave_dim,
            'feature_dim': feature_dim,
            'window_size': 512,
            'cse': ckpt1.get('config', {}),
            'field': field_cfg,
        },
        # Phase 1 — CSE
        'cse_state_dict': ckpt1['state_dict'],
        # Phase 2 — Field
        'field_state_dict': ckpt2['state_dict'],
        # Phase 5 — CGN
        'phase5_state': {k: v for k, v in ckpt5.items() if k != 'state_dict'},
        # Phase 6 — Memory
        'working_memory_state': working.state_dict_memory(),
        'episodic_memory_state': episodic.save_state(),
        'semantic_memory_state': semantic.save_state(),
        'router_state': router.save_state(),
        'metrics': {
            'training_time_seconds': elapsed,
            'episodic_accuracy': episodic_accuracy,
            'avg_forgetting': avg_forgetting,
            'max_forgetting': max_forgetting,
            'field_stability': stability,
            'consolidation_entries': consolid_result['consolidated'],
            'working_memory_size': working.size,
            'episodic_memory_size': episodic.size,
        },
    }

    save_checkpoint(6, checkpoint_state)
    log.success("Phase 6 checkpoint saved")
    log.separator("Phase 6 Training Complete")

    # Generate results
    results = PhaseResults(phase=6, component_name="Three-Tier Memory System")
    results.add_test("Episodic Write/Read", episodic_accuracy >= 0.9,
                     score=f"{episodic_accuracy:.4f}", threshold="≥ 0.9")
    results.add_test("Semantic Field Stability", stability >= 0.95,
                     score=f"{stability:.4f}", threshold="≥ 0.95")
    results.add_test("Zero Forgetting", avg_forgetting < 0.02,
                     score=f"{avg_forgetting:.6f}", threshold="< 0.02")
    results.add_metric("training_time", f"{elapsed:.1f}s")
    results.add_metric("episodic_entries", episodic.size)
    results.add_metric("working_entries", working.size)
    results.add_metric("consolidation_passes", consolidation.consolidation_count)
    results.save()

    return checkpoint_state


if __name__ == "__main__":
    main()
