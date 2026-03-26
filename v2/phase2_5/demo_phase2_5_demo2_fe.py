"""
demo_phase2_5_demo2.py — Demo: Field Evolution Energy Landscape Visualization

Shows the FLUX-native physics inside the FieldEvolutionGenerator:
  1. Energy trace during settling (should monotonically decrease)
  2. Per-slot energy distribution in the evolved field
  3. How different contexts create different field patterns
  4. Sub-band analysis of generated waves

This demonstrates what makes Field Evolution different from an MLP/GRU:
  - The field physically settles — energy decreases = computation happening
  - Different contexts create different attractor landscapes
  - The "model" is the physics, not the weights
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase2'))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F
from field_evolution_generator import BAND_SLICES
from train_field_evolution import load_phase1_and_phase2, load_phase2_5_checkpoint


DEMO_TEXTS = [
    "The cat sat on the mat",
    "def fibonacci(n): return n",
    "∫₀^∞ e^(-x²) dx = √π/2",
]


@torch.no_grad()
def main():
    print("\n" + "=" * 65)
    print("  DEMO 2: Field Evolution Energy Landscape Analysis")
    print("=" * 65)

    device = 'cpu'
    hf_token = os.environ.get('HF_TOKEN', '')

    # Load everything
    print("\n  Loading Phase 1 + 2 + 2.5...")
    components = load_phase1_and_phase2(device=device, hf_token=hf_token)
    cse = components['cse']
    chunker = components['chunker']
    wtt = components['wtt']

    generator = load_phase2_5_checkpoint(device=device, hf_token=hf_token)
    generator.eval()

    # ── Architecture summary ──────────────────────────────────────────
    summary = generator.parameter_summary()
    print(f"\n  FieldEvolutionGenerator Architecture:")
    for component, count in summary.items():
        print(f"    {component:>12s}: {count:>8,} params")
    print(f"\n  Settle steps: {generator.settle_steps}")
    print(f"  Max slots: {generator.max_slots}")
    print()

    for text in DEMO_TEXTS:
        byte_data = text.encode('utf-8')
        wave = cse.encode(text)
        pairs = chunker.chunk_with_bytes(wave.full, byte_data)

        if len(pairs) < 3:
            continue

        chunk_waves = torch.stack([p[0] for p in pairs]).to(device)
        chunk_bytes = [p[1] for p in pairs]
        chunk_texts = [b.decode('utf-8', errors='replace') for b in chunk_bytes]

        n = max(2, len(pairs) // 2)
        prefix = chunk_waves[:n].unsqueeze(0)  # [1, n, 432]

        pred_wave, info = generator(prefix)

        # Decode
        decoded = wtt.decode(pred_wave.squeeze(0), temperature=0.5)
        decoded_text = decoded.decode('utf-8', errors='replace')
        gt_text = chunk_texts[n] if n < len(chunk_texts) else "?"

        print(f"  ┌─ \"{text}\"")
        print(f"  │  prefix = {chunk_texts[:n]}")
        print(f"  │  predicted = \"{decoded_text}\"   ground truth = \"{gt_text}\"")

        # ── Energy trace ──────────────────────────────────────────────
        energy_trace = info['energy_trace'][0].tolist()
        trace_str = ' → '.join(f'{e:.2f}' for e in energy_trace)
        total_drop = energy_trace[0] - energy_trace[-1]
        print(f"  │")
        print(f"  │  Energy trace: {trace_str}")
        print(f"  │  Total ΔE = {total_drop:.4f}  ({'settling ✓' if total_drop > 0 else 'NOT settling ✗'})")

        # ── Per-slot energy in evolved field ──────────────────────────
        final_energy = info['final_energy'][0]  # [S]
        seeded_energy = final_energy[:n].tolist()
        next_energy = final_energy[n].item()
        print(f"  │")
        print(f"  │  Slot energies (seeded): {', '.join(f'{e:.3f}' for e in seeded_energy[:8])}")
        print(f"  │  Slot energy (next pos): {next_energy:.3f}")

        # ── Sub-band analysis of predicted wave ───────────────────────
        print(f"  │")
        print(f"  │  Sub-band analysis:")
        pred = pred_wave.squeeze(0)
        for name, (start, end) in BAND_SLICES.items():
            band = pred[start:end]
            std = band.std().item()
            mean = band.mean().item()
            energy = (band ** 2).sum().item()
            print(f"  │    {name:>10s}: mean={mean:+.4f}  std={std:.4f}  energy={energy:.4f}")

        # ── Field state norm profile ──────────────────────────────────
        field_state = info['field_state'][0]  # [S, 512]
        norms = field_state.norm(dim=-1)  # [S]
        seeded_norms = norms[:n].tolist()
        next_norm = norms[n].item() if n < norms.shape[0] else 0.0
        print(f"  │")
        print(f"  │  Field norms (seeded): {', '.join(f'{x:.3f}' for x in seeded_norms[:8])}")
        print(f"  │  Field norm (next pos): {next_norm:.3f}")
        print(f"  └─\n")


if __name__ == '__main__':
    main()
