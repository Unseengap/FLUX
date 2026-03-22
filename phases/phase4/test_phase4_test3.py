"""
Phase 4 Test 3: Temperature Annealing Behavior

Acceptance criterion:
  - Temperature decreases as error decreases
  - High prediction error → temperature increases (heating)
  - Low prediction error → temperature decreases (cooling)
  - Temperature stays within [min_temp, max_temp] bounds

Standalone: python test_phase4_test3.py
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
from temperature import TemperatureManager

print("=" * 60)
print("FLUX Phase 4 Test 3: Temperature Annealing Behavior")
print("=" * 60)

device = get_device()
results = PhaseResults(phase=4, component_name="Thermodynamic Learning")

# ── Sub-test 1: Temperature decays without error input ──
print("\n  Sub-test 1: Base temperature decay...")
tm = TemperatureManager(initial=1.0, min_temp=0.01, decay=0.99)

temps = []
for i in range(200):
    t = tm.step()
    temps.append(t)

# Temperature should be monotonically decreasing without error input
monotonic_decreasing = all(temps[i] >= temps[i+1] for i in range(len(temps)-1))
final_temp = temps[-1]
print(f"    Initial: {temps[0]:.4f}")
print(f"    After 200 steps: {final_temp:.4f}")
print(f"    Monotonically decreasing: {monotonic_decreasing}")

results.add_test(
    "Base Temperature Decay",
    passed=monotonic_decreasing and final_temp < 0.5,
    score=final_temp,
    threshold=0.5,
)
print(f"    {'✓' if monotonic_decreasing else '✗'} Temperature decays correctly")

# ── Sub-test 2: High error → temperature increases ──
print("\n  Sub-test 2: High error causes heating...")
tm2 = TemperatureManager(initial=0.3, min_temp=0.01, max_temp=2.0, decay=0.99, error_sensitivity=0.5)

# Let it cool a bit
for _ in range(50):
    tm2.step(prediction_error=0.05)

temp_before_spike = tm2.temperature
# Now feed high error
tm2.step(prediction_error=2.0)
temp_after_spike = tm2.temperature

heated_up = temp_after_spike > temp_before_spike
print(f"    Temp before error spike: {temp_before_spike:.4f}")
print(f"    Temp after error spike:  {temp_after_spike:.4f}")
print(f"    Heated up: {heated_up}")

results.add_test(
    "High Error Causes Heating",
    passed=heated_up,
    score=temp_after_spike - temp_before_spike,
    threshold=0.0,
)
print(f"    {'✓' if heated_up else '✗'} Temperature rises on high error")

# ── Sub-test 3: Low error → accelerated cooling ──
print("\n  Sub-test 3: Consistently low error accelerates cooling...")
tm3a = TemperatureManager(initial=0.5, min_temp=0.01, decay=0.99)
tm3b = TemperatureManager(initial=0.5, min_temp=0.01, decay=0.99)

# Both start at the same temp. Feed low errors to one, nothing to the other.
for _ in range(100):
    tm3a.step()  # No error info — just base decay
    tm3b.step(prediction_error=0.02)  # Consistently low error — extra cooling

low_error_cooler = tm3b.temperature <= tm3a.temperature
print(f"    No-error temp after 100 steps: {tm3a.temperature:.6f}")
print(f"    Low-error temp after 100 steps: {tm3b.temperature:.6f}")
print(f"    Low error cooled faster: {low_error_cooler}")

results.add_test(
    "Low Error Accelerates Cooling",
    passed=low_error_cooler,
    score=tm3a.temperature - tm3b.temperature,
    threshold=0.0,
)
print(f"    {'✓' if low_error_cooler else '✗'} Low error accelerates cooling")

# ── Sub-test 4: Temperature bounds respected ──
print("\n  Sub-test 4: Temperature stays in bounds...")
tm4 = TemperatureManager(initial=1.0, min_temp=0.01, max_temp=2.0, error_sensitivity=1.0)

bounds_ok = True
for _ in range(500):
    # Alternate between high and low errors to stress-test bounds
    import random
    error = random.choice([0.01, 0.05, 0.5, 2.0, 5.0])
    t = tm4.step(prediction_error=error)
    if t < tm4.min_temp - 1e-8 or t > tm4.max_temp + 1e-8:
        bounds_ok = False
        print(f"    ✗ Out of bounds: {t:.6f} (min={tm4.min_temp}, max={tm4.max_temp})")
        break

print(f"    Final temp: {tm4.temperature:.6f}")
print(f"    Bounds respected over 500 steps: {bounds_ok}")

results.add_test(
    "Temperature Bounds Respected",
    passed=bounds_ok,
    score=1.0 if bounds_ok else 0.0,
    threshold=1.0,
)
print(f"    {'✓' if bounds_ok else '✗'} Temperature always in [{tm4.min_temp}, {tm4.max_temp}]")

# ── Sub-test 5: Integration — temperature anneals during real learning ──
print("\n  Sub-test 5: Temperature anneals during actual learning...")

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

tl = ThermodynamicLearner(field=field, initial_temp=1.0, decay=0.995).to(device)
ol = OnlineLearner(cse=cse, tl=tl, device=device)

# Feed 50 facts
texts = [
    "The Sun is a star", "Water is H2O", "Gravity pulls objects down",
    "Light travels in waves", "Plants need sunlight", "Fish live in water",
    "Birds have feathers", "Ice is frozen water", "Fire needs oxygen",
    "Wind is moving air",
] * 5

initial_temp = tl.temp_manager.temperature
for text in texts:
    ol.learn_fact(text)
final_temp = tl.temp_manager.temperature

temp_decreased = final_temp < initial_temp
print(f"    Initial temperature: {initial_temp:.4f}")
print(f"    After 50 facts:     {final_temp:.6f}")
print(f"    Temperature decreased: {temp_decreased}")

results.add_test(
    "Temperature Decrease During Learning",
    passed=temp_decreased,
    score=initial_temp - final_temp,
    threshold=0.0,
)
print(f"    {'✓' if temp_decreased else '✗'} Temperature annealed during real learning")

# ── Save results ──
results.save()

all_passed = all([
    monotonic_decreasing and final_temp < 0.5,
    heated_up,
    low_error_cooler,
    bounds_ok,
    temp_decreased,
])

print(f"\n{'='*50}")
print(f"All tests passed: {all_passed}")
print(f"{'='*50}")
if all_passed:
    print("  ✓ TEMPERATURE ANNEALING TEST PASSED")
else:
    print("  ✗ Test failed — check sub-tests above")
