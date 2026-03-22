"""
Phase 7 Test 1: Full Pipeline Integration

Verifies the complete FLUX pipeline works end-to-end:
  text → CSE → wave → GR → CGN → Field → TL → Memory → output

Pass criteria:
  - Model loads all 6 phase checkpoints
  - Forward pass completes without error
  - All intermediate representations have correct shapes
  - Latency < 5 seconds per sample
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

from flux_utils import load_checkpoint, PhaseResults, get_device, checkpoint_exists
from flux_model import FLUXModel, FLUX_MODEL_CONFIG


def test_full_pipeline():
    print("=" * 60)
    print("  Phase 7 Test 1: Full Pipeline Integration")
    print("=" * 60)

    DEVICE = get_device()
    results = PhaseResults(phase=7, component_name="Full FLUX Integration")

    # ── Step 1: Load all checkpoints ──
    print("\n  Loading FLUXModel from phase checkpoints 1–6...")
    try:
        model = FLUXModel.from_checkpoints(device=DEVICE)
        load_ok = True
    except Exception as e:
        print(f"  ✗ Failed to load: {e}")
        load_ok = False
        results.add_test("Checkpoint Loading", False, score="FAIL", threshold="all 6 phases")
        results.save()
        print("\nTest 1: FAIL")
        return

    results.add_test("Checkpoint Loading", True, score="PASS", threshold="all 6 phases")
    print(f"  ✓ FLUXModel loaded on {DEVICE}")

    # ── Step 2: Forward pass ──
    test_texts = [
        "The quick brown fox jumps over the lazy dog",
        "FLUX uses resonance fields instead of weight matrices",
        "E = mc²",
        "def hello():\n    print('Hello FLUX!')",
        "你好世界",  # Chinese: Hello World
    ]

    all_forward_ok = True
    latencies = []
    print(f"\n  Running forward pass on {len(test_texts)} test texts...")
    for text in test_texts:
        try:
            response = model.forward(text, learn=False)

            # Validate shapes
            assert response.wave is not None, "Wave is None"
            assert response.wave.full.shape[-1] == 432, f"Wave dim wrong: {response.wave.full.shape}"
            assert response.relevance_output is not None, "GR output is None"
            assert response.cgn_output is not None, "CGN output is None"
            assert response.field_features is not None, "Field features are None"
            assert response.memory_result is not None, "Memory result is None"
            assert response.latency_ms < 5000, f"Too slow: {response.latency_ms:.0f}ms"

            latencies.append(response.latency_ms)
            print(f"    ✓ '{text[:40]}...' — {response.latency_ms:.1f}ms")

        except Exception as e:
            print(f"    ✗ '{text[:40]}...' — {e}")
            all_forward_ok = False

    avg_latency = sum(latencies) / max(len(latencies), 1)
    results.add_test("Forward Pass", all_forward_ok,
                     score=f"{avg_latency:.1f}ms avg",
                     threshold="< 5000ms, no errors")

    # ── Step 3: Stats validation ──
    stats = model.get_stats()
    print(f"\n  Model stats:")
    print(f"    Total params:     {stats.total_params:,}")
    print(f"    CSE params:       {stats.cse_params:,}")
    print(f"    Field params:     {stats.field_params:,}")
    print(f"    GR params:        {stats.gr_params:,}")
    print(f"    TL params:        {stats.tl_params:,}")
    print(f"    CGN params:       {stats.cgn_params:,}")
    print(f"    Memory params:    {stats.memory_params:,}")
    print(f"    Output head:      {stats.output_head_params:,}")
    print(f"    Field energy:     {stats.field_energy:.4f}")
    print(f"    Learning steps:   {stats.learning_steps}")

    stats_ok = stats.total_params > 0 and stats.cse_params > 0
    results.add_test("Model Statistics", stats_ok,
                     score=f"{stats.total_params:,} params",
                     threshold="> 0 for all components")

    # ── Step 4: Memory integration ──
    print(f"\n  Testing memory integration...")
    model.learn_fact("Phase 7 integration test fact")
    query_results = model.query("Phase 7 integration test", k=3)
    memory_ok = len(query_results) > 0
    if memory_ok:
        top_fact, top_score = query_results[0]
        print(f"    ✓ Query returned: [{top_score:.3f}] {top_fact[:60]}")
    else:
        print(f"    ✗ Query returned empty results")

    results.add_test("Memory Integration", memory_ok,
                     score=f"{len(query_results)} results",
                     threshold="> 0 results")

    # ── Step 5: Generation smoke test ──
    print(f"\n  Testing text generation...")
    try:
        generated = model.generate("The future of AI is", max_length=30, temperature=1.0)
        gen_ok = len(generated) > len("The future of AI is")
        print(f"    ✓ Generated: {generated[:80]}...")
    except Exception as e:
        gen_ok = False
        print(f"    ✗ Generation failed: {e}")

    results.add_test("Generation Smoke Test", gen_ok,
                     score="PASS" if gen_ok else "FAIL",
                     threshold="produces output > prompt")

    # ── Summary ──
    all_passed = all_forward_ok and stats_ok and memory_ok and gen_ok and load_ok
    results.add_metric("avg_latency_ms", f"{avg_latency:.1f}")
    results.add_metric("total_params", f"{stats.total_params:,}")
    results.save()

    print(f"\n  {'✓' if all_passed else '✗'} Test 1: {'PASS' if all_passed else 'FAIL'}")
    print(f"\nTest 1: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


if __name__ == "__main__":
    test_full_pipeline()
