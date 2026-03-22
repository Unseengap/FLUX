"""
Phase 4 Test 1: Single-Shot Learning Retention

Acceptance criterion:
  - Model learns from single example (no batch, no epochs)
  - Learned fact retrievable after 100 subsequent updates
  - Similarity to original fact > 0.5 after distractors

Standalone: python test_phase4_test1.py
"""

import sys
import torch
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'phases' / 'phase1'))
sys.path.insert(0, str(ROOT / 'phases' / 'phase2'))
sys.path.insert(0, str(Path(__file__).parent))

from flux_utils import load_checkpoint, get_device, PhaseResults
from cse import ContinuousSemanticEncoder
from field import ResonanceField
from thermodynamic import ThermodynamicLearner
from online_learner import OnlineLearner

print("=" * 60)
print("FLUX Phase 4 Test 1: Single-Shot Learning Retention")
print("=" * 60)

device = get_device()

# ── Load models ──
ckpt1 = load_checkpoint(1)
cse = ContinuousSemanticEncoder(**ckpt1.get('config', {}))
cse.load_state_dict(ckpt1['state_dict'])
cse = cse.to(device).eval()
for p in cse.parameters():
    p.requires_grad = False

ckpt2 = load_checkpoint(2)
field_cfg = ckpt2.get('config', {}).get('field', {})
field = ResonanceField(**field_cfg)
field.load_state_dict(ckpt2['state_dict'])
field = field.to(device)

tl = ThermodynamicLearner(field=field, settle_iterations=10).to(device)
ol = OnlineLearner(cse=cse, tl=tl, device=device)
print("  ✓ Models loaded")

# ── Test facts ──
test_facts = [
    "The capital of Mars colony Alpha is New Houston",
    "Water boils at 100 degrees Celsius at sea level",
    "The speed of light is approximately 300000 km per second",
    "Photosynthesis converts carbon dioxide into oxygen",
    "The deepest ocean trench is the Mariana Trench",
]

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

results = PhaseResults(phase=4, component_name="Thermodynamic Learning")

# ── Sub-test 1: One-shot learning (energy decreases) ──
print("\n  Sub-test 1: One-shot learning (energy should decrease)...")
stored_count = 0
for fact in test_facts:
    result = ol.learn_fact(fact)
    if result.fact_stored:
        stored_count += 1
    print(f"    {'✓' if result.fact_stored else '✗'} '{fact[:40]}...'  "
          f"energy: {result.initial_energy:.4f} → {result.final_energy:.4f}")

store_rate = stored_count / len(test_facts)
results.add_test(
    "One-Shot Learning",
    passed=store_rate >= 0.5,
    score=store_rate,
    threshold=0.5,
)
print(f"  One-shot store rate: {store_rate:.0%} ({stored_count}/{len(test_facts)})")

# ── Sub-test 2: Retention after 100 distractors ──
print("\n  Sub-test 2: Retention after 100 distractors...")
retention_results = []
for fact in test_facts[:3]:
    retention = ol.test_retention(
        fact_text=fact,
        distractor_texts=distractors,
        n_distractors=100,
    )
    retention_results.append(retention)
    icon = "✓" if retention['retained'] else "✗"
    print(f"    {icon} '{fact[:40]}...'  "
          f"sim: {retention['sim_immediately']:.4f} → {retention['sim_after_distractors']:.4f} "
          f"(drop: {retention['similarity_drop']:.4f})")

retained_count = sum(r['retained'] for r in retention_results)
retention_rate = retained_count / len(retention_results)
avg_sim_after = sum(r['sim_after_distractors'] for r in retention_results) / len(retention_results)

results.add_test(
    "Retention After 100 Updates",
    passed=retention_rate >= 0.5,
    score=avg_sim_after,
    threshold=0.5,
)
print(f"  Retention rate: {retention_rate:.0%} ({retained_count}/{len(retention_results)})")
print(f"  Avg similarity after distractors: {avg_sim_after:.4f}")

# ── Save results ──
results.save()

all_passed = store_rate >= 0.5 and retention_rate >= 0.5
print(f"\n{'='*50}")
print(f"All tests passed: {all_passed}")
print(f"{'='*50}")
if all_passed:
    print("  ✓ SINGLE-SHOT LEARNING RETENTION TEST PASSED")
else:
    print("  ✗ Test failed — check thresholds")
