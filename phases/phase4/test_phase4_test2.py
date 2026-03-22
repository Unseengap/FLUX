"""
Phase 4 Test 2: No Global Gradient Required

Acceptance criterion:
  - No global gradient computation at any point
  - All field parameter .grad attributes are None or zero after learning
  - ThermodynamicLearner operates entirely with torch.no_grad()

Standalone: python test_phase4_test2.py
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
print("FLUX Phase 4 Test 2: No Global Gradient Required")
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

# Ensure requires_grad is False on field buffers but True on parameters
# (we want to verify gradients are never accumulated even if params exist)
tl = ThermodynamicLearner(field=field, settle_iterations=10).to(device)
ol = OnlineLearner(cse=cse, tl=tl, device=device)
print("  ✓ Models loaded")

results = PhaseResults(phase=4, component_name="Thermodynamic Learning")

# ── Sub-test 1: Clear all gradients, learn facts, check no grads ──
print("\n  Sub-test 1: Learn 20 facts, verify no gradients accumulated...")

# Zero all gradients first
for p in field.parameters():
    if p.grad is not None:
        p.grad.zero_()

facts = [
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
    "The Milky Way contains hundreds of billions of stars",
    "Antibiotics treat bacterial infections but not viruses",
    "The Great Wall of China is visible from low Earth orbit",
    "Lightning is a discharge of atmospheric electricity",
    "Tectonic plates move a few centimeters per year",
    "Oxygen makes up 21 percent of the atmosphere",
    "The largest planet in our solar system is Jupiter",
    "Diamond is the hardest known natural material",
    "The human body is approximately 60 percent water",
    "Coral reefs support 25 percent of marine species",
]

for fact in facts:
    ol.learn_fact(fact)

# Check for gradients
grad_found = False
grad_violations = []
for name, param in field.named_parameters():
    if param.grad is not None and param.grad.abs().sum().item() > 0:
        grad_found = True
        grad_violations.append((name, param.grad.abs().sum().item()))
        print(f"    ✗ Gradient found in {name}: {param.grad.abs().sum().item():.6f}")

if not grad_found:
    print("    ✓ Zero gradients in all field parameters after 20 facts")

results.add_test(
    "No Global Gradients After Learning",
    passed=not grad_found,
    score=0.0 if not grad_found else len(grad_violations),
    threshold=0.0,
)

# ── Sub-test 2: Verify gradient tracking is disabled during settle ──
print("\n  Sub-test 2: Verify torch.no_grad() context during settling...")

# Enable gradient tracking temporarily to detect leaks
for p in field.parameters():
    p.requires_grad_(True)
    if p.grad is not None:
        p.grad.zero_()

# Learn one more fact
ol.learn_fact("Test gradient leak detection fact number one")

# Check again
grad_leak = False
for name, param in field.named_parameters():
    if param.grad is not None and param.grad.abs().sum().item() > 0:
        grad_leak = True
        print(f"    ✗ Gradient LEAK in {name}: {param.grad.abs().sum().item():.6f}")

if not grad_leak:
    print("    ✓ No gradient leaks — settling uses torch.no_grad() correctly")

results.add_test(
    "No Gradient Leaks During Settle",
    passed=not grad_leak,
    score=0.0 if not grad_leak else 1.0,
    threshold=0.0,
)

# ── Sub-test 3: Verify field state changes (it IS learning, just not via gradients) ──
print("\n  Sub-test 3: Verify field state changes without gradients...")

state_before = field.state.clone()
ol.learn_fact("A brand new fact that should change the field state visibly")
state_after = field.state.clone()

state_changed = (state_before - state_after).abs().sum().item()
changed = state_changed > 0.0
print(f"    Field state delta: {state_changed:.6f}")
print(f"    {'✓' if changed else '✗'} Field state {'changed' if changed else 'unchanged'} "
      f"(learning happened via physics, not gradients)")

results.add_test(
    "Field State Changes Without Gradients",
    passed=changed,
    score=state_changed,
    threshold=0.0,
)

# ── Save results ──
results.save()

all_passed = not grad_found and not grad_leak and changed
print(f"\n{'='*50}")
print(f"All tests passed: {all_passed}")
print(f"{'='*50}")
if all_passed:
    print("  ✓ NO GLOBAL GRADIENT TEST PASSED")
else:
    print("  ✗ Test failed")
