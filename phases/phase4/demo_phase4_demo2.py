"""
Phase 4 Demo 2: Compare Thermodynamic Learning to SGD Convergence

Shows side-by-side:
  - Thermodynamic settling: 1 pass per fact, no gradients
  - SGD: N steps per fact, requires gradients
  - Convergence speed and final energy comparison

Generates: demo4_tl_vs_sgd.png
Standalone: python demo_phase4_demo2.py
"""

import sys
import time
import torch
import torch.nn.functional as F
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
print("FLUX Phase 4 Demo 2: Thermodynamic Learning vs SGD")
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

# ── Test facts ──
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
]

# ── Thermodynamic Learning ──
print("\n── Thermodynamic Learning (1 settle pass per fact) ──")
tl_energies = []
tl_times = []

for fact in facts:
    t0 = time.time()
    result = ol.learn_fact(fact)
    elapsed = time.time() - t0
    tl_energies.append(result.final_energy)
    tl_times.append(elapsed)
    print(f"  TL: {fact[:40]:<40} energy={result.final_energy:.4f}  time={elapsed:.4f}s")

# ── SGD Baseline (for comparison) ──
print("\n── SGD Baseline (100 steps per fact) ──")
sgd_step_counts = [1, 5, 10, 25, 50, 100, 200]
sgd_results = {n: [] for n in sgd_step_counts}
sgd_curves = {}

for fact in facts:
    wave_vec = ol._text_to_wave(fact)
    with torch.no_grad():
        target = field.wave_to_feature(wave_vec).detach()

    # Track convergence curve for the first fact
    if fact == facts[0]:
        param = torch.randn_like(target, requires_grad=True)
        opt = torch.optim.SGD([param], lr=0.01)
        curve = []
        for step in range(200):
            loss = F.mse_loss(param, target)
            curve.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
        sgd_curves['first_fact'] = curve

    for n_steps in sgd_step_counts:
        param = torch.randn_like(target, requires_grad=True)
        opt = torch.optim.SGD([param], lr=0.01)
        t0 = time.time()
        for _ in range(n_steps):
            loss = F.mse_loss(param, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
        elapsed = time.time() - t0
        sgd_results[n_steps].append({
            'energy': loss.item(),
            'time': elapsed,
        })

# Print SGD summary
for n_steps in sgd_step_counts:
    avg_e = sum(r['energy'] for r in sgd_results[n_steps]) / len(facts)
    avg_t = sum(r['time'] for r in sgd_results[n_steps]) / len(facts)
    print(f"  SGD ({n_steps:3d} steps): avg_energy={avg_e:.4f}  avg_time={avg_t:.4f}s")

# ── Plot ──
fig = plt.figure(figsize=(16, 10))
fig.suptitle("FLUX Phase 4: Thermodynamic Learning vs SGD", fontsize=16, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

# Plot 1: Energy comparison bar chart
ax1 = fig.add_subplot(gs[0, 0])
tl_avg = sum(tl_energies) / len(tl_energies)
sgd_avgs = [sum(r['energy'] for r in sgd_results[n]) / len(facts) for n in sgd_step_counts]
labels = ['TL\n(1 pass)'] + [f'SGD\n({n} steps)' for n in sgd_step_counts]
values = [tl_avg] + sgd_avgs
colors = ['#27ae60'] + ['#3498db'] * len(sgd_step_counts)
ax1.bar(range(len(values)), values, color=colors)
ax1.set_xticks(range(len(values)))
ax1.set_xticklabels(labels, fontsize=7)
ax1.set_ylabel('Mean Energy')
ax1.set_title('Final Energy (lower = better)')
ax1.grid(True, alpha=0.3)

# Plot 2: Time comparison
ax2 = fig.add_subplot(gs[0, 1])
tl_avg_time = sum(tl_times) / len(tl_times)
sgd_avg_times = [sum(r['time'] for r in sgd_results[n]) / len(facts) for n in sgd_step_counts]
labels_t = ['TL'] + [f'SGD-{n}' for n in sgd_step_counts]
values_t = [tl_avg_time] + sgd_avg_times
colors_t = ['#27ae60'] + ['#3498db'] * len(sgd_step_counts)
ax2.bar(range(len(values_t)), [v * 1000 for v in values_t], color=colors_t)
ax2.set_xticks(range(len(values_t)))
ax2.set_xticklabels(labels_t, fontsize=7, rotation=45)
ax2.set_ylabel('Time (ms)')
ax2.set_title('Time Per Fact (lower = faster)')
ax2.grid(True, alpha=0.3)

# Plot 3: SGD convergence curve vs TL instant
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(sgd_curves['first_fact'], color='#3498db', label='SGD convergence', linewidth=2)
ax3.axhline(y=tl_energies[0], color='#27ae60', linestyle='--', linewidth=2,
            label=f'TL (1 pass) = {tl_energies[0]:.4f}')
ax3.set_xlabel('SGD Step')
ax3.set_ylabel('Energy (MSE)')
ax3.set_title('Convergence: SGD vs Instant TL')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Plot 4: Per-fact energy (TL)
ax4 = fig.add_subplot(gs[1, 0])
ax4.bar(range(1, len(facts) + 1), tl_energies, color='#27ae60')
ax4.set_xlabel('Fact #')
ax4.set_ylabel('Final Energy')
ax4.set_title('TL: Energy Per Fact (1 pass each)')
ax4.grid(True, alpha=0.3)

# Plot 5: Gradient count comparison
ax5 = fig.add_subplot(gs[1, 1])
tl_grads = 0
sgd_grads = [n * len(facts) for n in sgd_step_counts]
all_grads = [tl_grads] + sgd_grads
all_labels = ['TL'] + [f'SGD-{n}' for n in sgd_step_counts]
all_colors = ['#27ae60'] + ['#e74c3c'] * len(sgd_step_counts)
ax5.bar(range(len(all_grads)), all_grads, color=all_colors)
ax5.set_xticks(range(len(all_grads)))
ax5.set_xticklabels(all_labels, fontsize=7, rotation=45)
ax5.set_ylabel('Total .backward() Calls')
ax5.set_title('Gradient Computations Required')
ax5.grid(True, alpha=0.3)

# Plot 6: Summary
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
summary = (
    f"Thermodynamic Learning:\n"
    f"  Passes per fact: 1\n"
    f"  Gradient calls:  0\n"
    f"  Avg energy:      {tl_avg:.4f}\n"
    f"  Avg time/fact:   {tl_avg_time*1000:.1f}ms\n\n"
    f"SGD (100 steps):\n"
    f"  Passes per fact: 100\n"
    f"  Gradient calls:  {100 * len(facts)}\n"
    f"  Avg energy:      {sgd_avgs[sgd_step_counts.index(100)]:.4f}\n"
    f"  Avg time/fact:   {sgd_avg_times[sgd_step_counts.index(100)]*1000:.1f}ms\n\n"
    f"Learning = Physics\n"
    f"Not Gradient Descent"
)
ax6.text(0.05, 0.5, summary, fontsize=10, fontfamily='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax6.set_title('Comparison Summary')

fig.text(0.5, 0.01,
         "THERMODYNAMIC SETTLING — inference and learning are the same operation.",
         ha='center', fontsize=11, fontstyle='italic', color='#2c3e50')

out_path = Path(__file__).parent / 'demo4_tl_vs_sgd.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\n  ✓ Saved: {out_path}")
print("  ✓ Demo complete")
