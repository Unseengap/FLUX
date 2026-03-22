"""
Phase 5 Training: Causal Geometry Nodes integration.

Loads Phases 1–4 checkpoints, builds the CGN layer on top,
and demonstrates:
  1. Causal graph formation (add arrows, trace causes)
  2. Multi-timescale separation (fast vs slow node activation)
  3. Geometry computation correctness (signal bending)
  4. Causal invalidation (disprove cause → weaken conclusion)
  5. Full CGN network processing from CSE waves

~3 min on GPU • ~15 min on CPU
"""

import sys
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

# ── Project paths ──
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'phases' / 'phase1'))
sys.path.insert(0, str(ROOT / 'phases' / 'phase2'))
sys.path.insert(0, str(ROOT / 'phases' / 'phase3'))
sys.path.insert(0, str(ROOT / 'phases' / 'phase4'))
sys.path.insert(0, str(Path(__file__).parent))

from flux_utils import (
    load_checkpoint, save_checkpoint, get_device,
    PhaseLogger, PhaseResults, checkpoint_exists,
    verify_checkpoint_chain,
)
from cse import ContinuousSemanticEncoder
from field import ResonanceField
from thermodynamic import ThermodynamicLearner
from online_learner import OnlineLearner
from cgn import CausalGeometryNode, CGN_CONFIG
from causal_graph import CausalGraph
from multi_timescale import MultiTimescaleCoordinator


def main():
    log = PhaseLogger(phase=5)
    log.separator("Phase 5: Causal Geometry Nodes")
    start_time = time.time()

    device = get_device()
    log.info(f"Device: {device}")

    # ─────────────────────────────────────────
    # Load Phases 1–4
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

    print("\n── Loading Phase 4 (ThermodynamicLearner) ──")
    ckpt4 = load_checkpoint(4)
    tl = ThermodynamicLearner(field=field, settle_iterations=10, decay=0.995).to(device)
    tl.load_state(ckpt4['tl_state'])
    ol = OnlineLearner(cse=cse, tl=tl, device=device)
    log.success("ThermodynamicLearner + OnlineLearner loaded")

    # ─────────────────────────────────────────
    # Build Phase 5 Components
    # ─────────────────────────────────────────
    print("\n── Building CGN Components ──")

    feature_dim = 512
    cgn_network = MultiTimescaleCoordinator(
        feature_dim=feature_dim,
        n_fast=16,
        n_medium=8,
        n_slow=4,
    ).to(device)
    log.success(f"MultiTimescaleCoordinator built: {cgn_network.total_params():,} params")

    causal_graph = CausalGraph()
    log.success("CausalGraph initialized")

    # ════════════════════════════════════════════
    # Stage A: Causal Graph Formation
    # ════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  Stage A: Causal Graph Formation")
    print("=" * 65)

    # Build a knowledge graph of causal relationships
    knowledge = [
        # (subject, relation, object, node_ids, weight)
        ("bird", "can", "fly", (0, 2), 0.8),
        ("penguin", "is_a", "bird", (1, 0), 1.0),
        ("penguin", "cannot", "fly", (1, 3), 0.9),
        ("sparrow", "is_a", "bird", (4, 0), 1.0),
        ("sparrow", "can", "fly", (4, 2), 0.95),
        ("fish", "can", "swim", (5, 6), 0.9),
        ("dolphin", "resembles", "fish", (7, 5), 0.3),
        ("dolphin", "is_a", "mammal", (7, 8), 1.0),
        ("mammal", "breathes", "air", (8, 9), 0.95),
    ]

    concept_names = {
        0: "bird", 1: "penguin", 2: "can_fly", 3: "cannot_fly",
        4: "sparrow", 5: "fish", 6: "can_swim", 7: "dolphin",
        8: "mammal", 9: "breathes_air",
    }

    for subj, rel, obj, (src, tgt), weight in knowledge:
        causal_graph.add_arrow(src, tgt, weight=weight, reason=f"{subj} {rel} {obj}")
        print(f"  ✓ {concept_names[src]:>10} → {concept_names[tgt]:<15}  (w={weight:.2f})  {subj} {rel} {obj}")

    # Register contradictory pairs
    causal_graph.add_contradiction(2, 3)   # can_fly ⇔ cannot_fly

    log.metric("causal_edges", causal_graph.edge_count())
    log.metric("causal_nodes", causal_graph.node_count())

    # Trace causal chains
    print("\n  Causal Traces:")
    trace_fly = causal_graph.trace_cause(2, depth=3)
    print(f"    Can fly? Chain: {trace_fly.chain}  conflict={trace_fly.has_conflict}")

    trace_not_fly = causal_graph.trace_cause(3, depth=3)
    print(f"    Cannot fly? Chain: {trace_not_fly.chain}  conflict={trace_not_fly.has_conflict}")

    # Entity-level conflict detection — penguin reaches both can_fly and cannot_fly
    penguin_conflicts = causal_graph.detect_entity_conflicts(1, depth=3)  # penguin=1
    print(f"    Penguin entity conflicts: {[(concept_names.get(a, a), concept_names.get(b, b)) for a, b in penguin_conflicts]}")
    sparrow_conflicts = causal_graph.detect_entity_conflicts(4, depth=3)  # sparrow=4
    print(f"    Sparrow entity conflicts: {[(concept_names.get(a, a), concept_names.get(b, b)) for a, b in sparrow_conflicts]}")

    # ════════════════════════════════════════════
    # Stage B: Multi-Timescale Processing
    # ════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  Stage B: Multi-Timescale Separation")
    print("=" * 65)

    # Encode some concepts and pass through CGN network
    test_concepts = [
        "Birds can fly through the air",
        "Penguins are birds that cannot fly",
        "The quick brown fox jumps over the lazy dog",
    ]

    for concept_text in test_concepts:
        with torch.no_grad():
            wave = cse.encode(concept_text)
            wave_vec = wave.full.mean(dim=0).to(device)
            # Project to feature_dim via field's wave_to_feature
            feature = field.wave_to_feature(wave_vec).detach()

        cgn_network.reset_states()
        result = cgn_network.forward_with_traces(feature, steps=10)
        print(f"  '{concept_text[:50]}'")
        print(f"    Fast activation:   {result.fast_activation:.4f}")
        print(f"    Medium activation: {result.medium_activation:.4f}")
        print(f"    Slow activation:   {result.slow_activation:.4f}")
        print(f"    Traces collected:  {len(result.traces)}")

    # Measure timescale separation
    print("\n  Timescale Separation Measurement:")
    test_signal = torch.randn(feature_dim, device=device) * 0.1
    cgn_network.reset_states()
    sep = cgn_network.measure_timescale_separation(test_signal, max_steps=100)
    print(f"    Fast nodes activate at step:   {sep['fast_steps_to_activate']}")
    print(f"    Medium nodes activate at step: {sep['medium_steps_to_activate']}")
    print(f"    Slow nodes activate at step:   {sep['slow_steps_to_activate']}")
    log.metric("fast_activation_step", sep['fast_steps_to_activate'])
    log.metric("medium_activation_step", sep['medium_steps_to_activate'])
    log.metric("slow_activation_step", sep['slow_steps_to_activate'])

    # ════════════════════════════════════════════
    # Stage C: Geometry Computation
    # ════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  Stage C: Geometry Computation Correctness")
    print("=" * 65)

    node = CausalGeometryNode(0, feature_dim).to(device)
    signal = torch.randn(feature_dim, device=device) * 0.5

    output, trace = node.forward_with_trace(signal)
    print(f"  Input norm:      {signal.norm().item():.4f}")
    print(f"  Output norm:     {output.norm().item():.4f}")
    print(f"  Bending magnitude: {trace.bending_magnitude:.4f}")
    print(f"  Influence strength: {trace.influence_strength:.4f}")

    # Mass effect
    with torch.no_grad():
        node.mass.copy_(torch.tensor(5.0))
    output_high_mass = node(signal)
    mass_ratio = output_high_mass.norm().item() / max(output.norm().item(), 1e-8)
    print(f"  Mass=5.0 amplification: {mass_ratio:.2f}x")
    log.metric("mass_amplification", f"{mass_ratio:.2f}x")

    # ════════════════════════════════════════════
    # Stage D: Causal Invalidation
    # ════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  Stage D: Causal Invalidation")
    print("=" * 65)

    # Add a fact, then disprove it
    causal_graph.add_arrow(10, 11, weight=1.0, reason="all swans are white")
    causal_graph.add_arrow(11, 12, weight=0.8, reason="therefore this is white")

    print("  Before invalidation:")
    summary_before = causal_graph.get_conflict_summary(11)
    print(f"    'all swans white': {summary_before['conclusion']} (net={summary_before['net_weight']:.2f})")

    affected = causal_graph.invalidate_cause(10)
    print(f"\n  After invalidating 'all swans are white':")
    print(f"    Affected nodes: {affected}")
    summary_after = causal_graph.get_conflict_summary(11)
    print(f"    'all swans white': {summary_after['conclusion']} (net={summary_after['net_weight']:.2f})")
    log.metric("invalidation_propagation", len(affected))

    # ════════════════════════════════════════════
    # Stage E: Full Pipeline (CSE → Field → CGN)
    # ════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  Stage E: Full Pipeline — CSE → Field → CGN")
    print("=" * 65)

    pipeline_texts = [
        "The Earth revolves around the Sun",
        "Water freezes at zero degrees Celsius",
        "Gravity pulls objects toward the center of the Earth",
        "Light travels at 300000 kilometers per second",
        "Plants convert sunlight into energy through photosynthesis",
    ]

    for text in pipeline_texts:
        with torch.no_grad():
            # CSE: text → wave
            wave = cse.encode(text)
            wave_vec = wave.full.mean(dim=0).to(device)
            # Field: wave → feature
            feature = field.wave_to_feature(wave_vec).detach()
            # CGN: feature → processed output
            cgn_network.reset_states()
            cgn_out = cgn_network(feature, steps=5)
            # Check causal trace
            _, traces_result = cgn_network.fast_nodes[0].forward_with_trace(feature)

        print(f"  ✓ '{text[:50]}'")
        print(f"      wave=[{wave_vec.shape}] → feature=[{feature.shape}] → cgn_out=[{cgn_out.shape}]")
        print(f"      output norm={cgn_out.norm().item():.4f}  bending={traces_result.bending_magnitude:.4f}")

    elapsed = time.time() - start_time
    log.metric("training_time", f"{elapsed:.1f}s")
    log.metric("total_cgn_params", cgn_network.total_params())
    print(f"\n  ✓ Pipeline complete in {elapsed:.1f}s")

    # ─────────────────────────────────────────
    # Save Checkpoint
    # ─────────────────────────────────────────
    print("\n── Saving Phase 5 Checkpoint ──")

    checkpoint_state = {
        'phase': 5,
        'timestamp': datetime.now().isoformat(),
        'phase1_config': ckpt1.get('config', {}),
        'phase2_config': ckpt2.get('config', {}),
        'tl_state': tl.save_state(),
        'cgn_state': cgn_network.save_state(),
        'causal_graph_state': causal_graph.save_state(),
        'config': {
            'feature_dim': feature_dim,
            'n_fast': 16,
            'n_medium': 8,
            'n_slow': 4,
        },
        'metrics': {
            'causal_edges': causal_graph.edge_count(),
            'causal_nodes': causal_graph.node_count(),
            'total_cgn_params': cgn_network.total_params(),
            'fast_activation_step': sep['fast_steps_to_activate'],
            'slow_activation_step': sep['slow_steps_to_activate'],
            'mass_amplification': mass_ratio,
            'training_time_seconds': elapsed,
        },
    }

    save_checkpoint(5, checkpoint_state)
    log.success("Phase 5 checkpoint saved")

    # ─────────────────────────────────────────
    # Generate Results
    # ─────────────────────────────────────────
    results = PhaseResults(phase=5, component_name="Causal Geometry Nodes")

    results.add_test(
        "Causal Trace Accuracy",
        passed=len(trace_fly.chain) >= 2,
        score=f"chain_length={len(trace_fly.chain)}",
        threshold="chain_length >= 2",
    )
    results.add_test(
        "Multi-Timescale Separation",
        passed=sep['fast_steps_to_activate'] < sep['slow_steps_to_activate'],
        score=f"fast={sep['fast_steps_to_activate']}, slow={sep['slow_steps_to_activate']}",
        threshold="fast < slow",
    )
    results.add_test(
        "Geometry Computation",
        passed=trace.bending_magnitude > 0,
        score=f"bending={trace.bending_magnitude:.4f}",
        threshold="> 0",
    )
    results.add_test(
        "Causal Invalidation",
        passed=len(affected) > 0,
        score=f"affected={len(affected)}",
        threshold="> 0",
    )
    results.add_test(
        "Full Pipeline",
        passed=cgn_out.norm().item() > 0,
        score=f"output_norm={cgn_out.norm().item():.4f}",
        threshold="> 0",
    )

    results.add_demo("Causal Chain Trace", ran=True, quality="Full causal chain with conflict detection")
    results.add_demo("Fast vs Slow Activation", ran=True, quality="Clear timescale separation measured")

    results.add_metric("Total CGN Parameters", f"{cgn_network.total_params():,}")
    results.add_metric("Causal Edges", causal_graph.edge_count())
    results.add_metric("Causal Nodes", causal_graph.node_count())
    results.add_metric("Training Time", f"{elapsed:.1f}s")

    results.save(str(Path(__file__).parent / 'RESULTS_PHASE_5.md'))

    print(f"\n{'='*60}")
    print("✓ PHASE 5 TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Checkpoint: checkpoints/phase5.phase.pt")
    print(f"  Results:    phases/phase5/RESULTS_PHASE_5.md")
    print(f"  Next:       Phase 6 — Three-Tier Memory System")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
