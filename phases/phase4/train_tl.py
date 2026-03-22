"""
Phase 4 Training: Thermodynamic Learning integration.

Loads Phases 1–3 checkpoints, builds ThermodynamicLearner on
top of the Phase 2 ResonanceField, and demonstrates:
  1. One-shot fact learning (single settle pass)
  2. Retention after 100+ subsequent updates
  3. Temperature annealing behaviour
  4. Comparison to SGD convergence
  5. No global gradients anywhere

~5 min on GPU • ~20 min on CPU
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


def main():
    log = PhaseLogger(phase=4)
    log.separator("Phase 4: Thermodynamic Learning")
    start_time = time.time()

    device = get_device()
    log.info(f"Device: {device}")

    # ─────────────────────────────────────────
    # Load Phases 1 & 2
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

    # ─────────────────────────────────────────
    # Build ThermodynamicLearner
    # ─────────────────────────────────────────
    print("\n── Building ThermodynamicLearner ──")
    tl = ThermodynamicLearner(
        field=field,
        initial_temp=1.0,
        min_temp=0.01,
        max_temp=2.0,
        decay=0.995,
        settle_iterations=10,
        settle_dt=0.1,
        perturbation_radius=4,
        error_sensitivity=0.5,
    )
    tl = tl.to(device)
    log.success("ThermodynamicLearner built")

    ol = OnlineLearner(cse=cse, tl=tl, device=device)
    log.success("OnlineLearner ready")

    # ─────────────────────────────────────────
    # Stage A: One-Shot Fact Learning
    # ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Stage A: One-Shot Fact Learning")
    print("=" * 65)

    test_facts = [
        "The capital of Mars colony Alpha is New Houston",
        "Water boils at 100 degrees Celsius at sea level",
        "The speed of light is approximately 300000 km per second",
        "Photosynthesis converts carbon dioxide into oxygen",
        "The deepest ocean trench is the Mariana Trench",
        "DNA carries genetic information in all living organisms",
        "The Earth orbits the Sun once every 365 days",
        "Gravity on the Moon is one sixth of Earth gravity",
        "The human brain contains approximately 86 billion neurons",
        "Sound travels faster in water than in air",
    ]

    fact_results = []
    for i, fact in enumerate(test_facts):
        result = ol.learn_fact(fact)
        fact_results.append(result)
        icon = "✓" if result.fact_stored else "✗"
        print(f"  {icon} [{i+1:2d}] energy: {result.initial_energy:.4f} → {result.final_energy:.4f} "
              f"(Δ={result.energy_delta:+.4f})  temp={result.temperature:.4f}  "
              f"surprise={result.prediction_error:.4f}")

    stored_count = sum(r.fact_stored for r in fact_results)
    log.metric("facts_stored", f"{stored_count}/{len(test_facts)}")
    print(f"\n  Facts stored (energy decreased): {stored_count}/{len(test_facts)}")

    # ─────────────────────────────────────────
    # Stage B: Retention Test
    # ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Stage B: Retention After 100 Distractor Updates")
    print("=" * 65)

    distractors = [
        "The weather today is partly cloudy with a chance of rain",
        "Apples are a popular fruit grown in temperate climates",
        "Chess is a strategic board game played by two opponents",
        "Mount Everest is the tallest mountain above sea level",
        "The Pacific Ocean is the largest ocean on Earth",
        "Electricity flows through conductors like copper wire",
        "Volcanoes form at tectonic plate boundaries",
        "The Sahara is the largest hot desert in the world",
        "Penguins are flightless birds found in the southern hemisphere",
        "The Amazon rainforest produces 20 percent of the world oxygen",
    ] * 12  # 120 distractors

    # Test retention of the first fact
    retention = ol.test_retention(
        fact_text="The capital of Mars colony Alpha is New Houston",
        distractor_texts=distractors,
        n_distractors=100,
    )
    print(f"  Fact: '{retention['fact']}'")
    print(f"  Similarity immediately after learning: {retention['sim_immediately']:.4f}")
    print(f"  Similarity after {retention['n_distractors']} distractors: {retention['sim_after_distractors']:.4f}")
    print(f"  Similarity drop: {retention['similarity_drop']:.4f}")
    print(f"  Retained: {retention['retained']}")
    log.metric("retention_sim_immediately", f"{retention['sim_immediately']:.4f}")
    log.metric("retention_sim_after_100", f"{retention['sim_after_distractors']:.4f}")
    log.metric("retention_passed", str(retention['retained']))

    # ─────────────────────────────────────────
    # Stage C: Temperature Annealing Observation
    # ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Stage C: Temperature Annealing Over 200 Steps")
    print("=" * 65)

    # Feed a stream and observe temperature
    stream_texts = distractors[:200]
    stream_pairs = []
    for text in stream_texts:
        wave_vec = ol._text_to_wave(text)
        stream_pairs.append((wave_vec, None))

    stream_results = tl.learn_stream(stream_pairs, verbose=True)

    temp_stats = tl.temp_manager.stats()
    print(f"\n  Final temperature:    {temp_stats['temperature']:.6f}")
    print(f"  Steps completed:      {temp_stats['step_count']}")
    print(f"  Is cold:              {temp_stats['is_cold']}")
    if 'error_trend' in temp_stats:
        print(f"  Error trend:          {temp_stats['error_trend']:+.6f}")
    log.metric("final_temperature", f"{temp_stats['temperature']:.6f}")

    # ─────────────────────────────────────────
    # Stage D: SGD Comparison
    # ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Stage D: Thermodynamic Learning vs SGD Comparison")
    print("=" * 65)

    comparison_facts = test_facts[:5]
    comparison = ol.compare_to_sgd(comparison_facts, sgd_steps=100)
    print(f"  Facts compared:       {comparison['n_facts']}")
    print(f"  TL time:              {comparison['tl_time']:.3f}s")
    print(f"  SGD time:             {comparison['sgd_time']:.3f}s")
    print(f"  TL mean energy:       {comparison['tl_mean_energy']:.6f}")
    print(f"  SGD mean energy:      {comparison['sgd_mean_energy']:.6f}")
    print(f"  TL faster:            {comparison['tl_faster']}")
    print(f"  Speedup factor:       {comparison['speedup']:.2f}x")
    log.metric("tl_vs_sgd_speedup", f"{comparison['speedup']:.2f}x")
    log.metric("tl_faster_than_sgd", str(comparison['tl_faster']))

    # ─────────────────────────────────────────
    # Stage E: Verify No Global Gradients
    # ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Stage E: Verify No Global Gradients")
    print("=" * 65)

    # Check: no parameter in the field should have accumulated gradients
    has_grads = False
    for name, param in field.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grads = True
            print(f"  ✗ UNEXPECTED gradient in {name}: {param.grad.abs().sum():.6f}")

    if not has_grads:
        print("  ✓ No global gradients found in field parameters")
        print("  ✓ All updates were local (through physics, not backprop)")
    else:
        print("  ✗ WARNING: Global gradients detected — needs investigation")
    log.metric("no_global_gradients", str(not has_grads))

    # ─────────────────────────────────────────
    # Save Checkpoint
    # ─────────────────────────────────────────
    elapsed = time.time() - start_time

    checkpoint_state = {
        'phase': 4,
        'timestamp': datetime.now().isoformat(),
        'phase1_config': ckpt1.get('config', {}),
        'phase2_config': ckpt2.get('config', {}),
        'tl_state': tl.save_state(),
        'metrics': {
            'facts_stored': stored_count,
            'retention_sim': retention['sim_after_distractors'],
            'retention_passed': retention['retained'],
            'final_temperature': tl.temp_manager.temperature,
            'no_global_gradients': not has_grads,
            'tl_faster_than_sgd': comparison['tl_faster'],
            'sgd_speedup': comparison['speedup'],
            'training_time_seconds': elapsed,
        },
    }
    save_checkpoint(4, checkpoint_state)

    # ─────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  Phase 4 training complete!")
    print(f"  Facts learned (one-shot): {stored_count}/{len(test_facts)}")
    print(f"  Retention after 100 updates: {retention['retained']}")
    print(f"  No global gradients: {not has_grads}")
    print(f"  TL faster than SGD: {comparison['tl_faster']} ({comparison['speedup']:.1f}x)")
    print(f"  Final temperature: {tl.temp_manager.temperature:.6f}")
    print(f"  Duration: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Next: run test scripts to verify acceptance criteria")
    print("=" * 65)

    log.success(f"Phase 4 complete in {elapsed:.1f}s")
    return tl, ol


if __name__ == "__main__":
    main()
