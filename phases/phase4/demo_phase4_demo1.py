"""
Phase 4 Demo 1: Learn a Fact in One Shot

Shows the thermodynamic learning process visually:
  - Feed a single fact
  - Watch energy decrease during settling
  - Query the fact back immediately
  - Show temperature dynamics

Generates: demo4_oneshot_learning.png
Standalone: python demo_phase4_demo1.py
"""

import sys
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'phases' / 'phase1'))
sys.path.insert(0, str(ROOT / 'phases' / 'phase2'))
sys.path.insert(0, str(Path(__file__).parent))

from flux_utils import load_checkpoint, get_device
from cse import ContinuousSemanticEncoder
from field import ResonanceField
from thermodynamic import ThermodynamicLearner
from online_learner import OnlineLearner

print("=" * 60)
print("FLUX Phase 4 Demo 1: One-Shot Fact Learning")
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

tl = ThermodynamicLearner(field=field, settle_iterations=10, decay=0.995).to(device)
ol = OnlineLearner(cse=cse, tl=tl, device=device)

# ── Learn facts and track everything ──
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
    "Lightning is a discharge of atmospheric electricity",
    "Tectonic plates move a few centimeters per year",
    "Oxygen makes up 21 percent of the atmosphere",
]

energies_initial = []
energies_final = []
temperatures = []
surprises = []
stored_flags = []

for fact in facts:
    result = ol.learn_fact(fact)
    energies_initial.append(result.initial_energy)
    energies_final.append(result.final_energy)
    temperatures.append(result.temperature)
    surprises.append(result.prediction_error)
    stored_flags.append(result.fact_stored)
    icon = "✓" if result.fact_stored else "✗"
    print(f"  {icon} '{fact[:50]}...'")
    print(f"    energy: {result.initial_energy:.4f} → {result.final_energy:.4f}  "
          f"temp={result.temperature:.4f}  surprise={result.prediction_error:.4f}")

# ── Query facts back ──
print("\n── Querying facts back ──")
query_sims = []
for fact in facts[:5]:
    qr = ol.query_fact(fact)
    query_sims.append(qr.top_similarity)
    print(f"  Query: '{fact[:40]}...'  → similarity: {qr.top_similarity:.4f}")

# ── Plot ──
fig = plt.figure(figsize=(16, 10))
fig.suptitle("FLUX Phase 4: One-Shot Thermodynamic Learning", fontsize=16, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

# Plot 1: Energy before/after
ax1 = fig.add_subplot(gs[0, 0])
x = range(1, len(facts) + 1)
ax1.plot(x, energies_initial, 'o-', color='#e74c3c', label='Before settle', alpha=0.8)
ax1.plot(x, energies_final, 's-', color='#27ae60', label='After settle', alpha=0.8)
ax1.set_xlabel('Fact #')
ax1.set_ylabel('Local Energy')
ax1.set_title('Energy Drop Per Fact')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Plot 2: Temperature over time
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(x, temperatures, 'o-', color='#e67e22', linewidth=2)
ax2.set_xlabel('Fact #')
ax2.set_ylabel('Temperature')
ax2.set_title('Temperature Annealing')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=tl.temp_manager.min_temp, color='blue', linestyle='--', alpha=0.5, label='min_temp')
ax2.legend(fontsize=8)

# Plot 3: Surprise (prediction error)
ax3 = fig.add_subplot(gs[0, 2])
ax3.bar(x, surprises, color=['#e74c3c' if s > 0.5 else '#f39c12' if s > 0.1 else '#27ae60' for s in surprises])
ax3.set_xlabel('Fact #')
ax3.set_ylabel('Prediction Error')
ax3.set_title('Surprise Level Per Fact')
ax3.grid(True, alpha=0.3)

# Plot 4: Energy delta (how much energy decreased)
ax4 = fig.add_subplot(gs[1, 0])
deltas = [ef - ei for ei, ef in zip(energies_initial, energies_final)]
colors = ['#27ae60' if d < 0 else '#e74c3c' for d in deltas]
ax4.bar(x, deltas, color=colors)
ax4.axhline(y=0, color='black', linewidth=0.5)
ax4.set_xlabel('Fact #')
ax4.set_ylabel('Energy Delta')
ax4.set_title('Energy Change (negative = learned)')
ax4.grid(True, alpha=0.3)

# Plot 5: Query similarity for first 5 facts
ax5 = fig.add_subplot(gs[1, 1])
bars = ax5.bar(range(1, 6), query_sims,
               color=['#27ae60' if s > 0.5 else '#e74c3c' for s in query_sims])
ax5.set_xlabel('Fact #')
ax5.set_ylabel('Cosine Similarity')
ax5.set_title('Retrieval Similarity After Learning')
ax5.set_ylim(0, 1.1)
ax5.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='threshold')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# Plot 6: Summary text
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
stored_count = sum(stored_flags)
summary = (
    f"Facts fed: {len(facts)}\n"
    f"Stored (energy↓): {stored_count}/{len(facts)}\n"
    f"Avg surprise: {sum(surprises)/len(surprises):.4f}\n"
    f"Final temperature: {temperatures[-1]:.6f}\n"
    f"Avg query sim: {sum(query_sims)/len(query_sims):.4f}\n\n"
    f"No epochs. No batches.\n"
    f"No gradients. No optimizer.\n"
    f"Just physics."
)
ax6.text(0.1, 0.5, summary, fontsize=12, fontfamily='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax6.set_title('Summary')

# Add caption
fig.text(0.5, 0.01,
         "ONE-SHOT LEARNING — each fact learned in a single settling pass, no training loop.",
         ha='center', fontsize=11, fontstyle='italic', color='#2c3e50')

out_path = Path(__file__).parent / 'demo4_oneshot_learning.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\n  ✓ Saved: {out_path}")
print("  ✓ Demo complete")
