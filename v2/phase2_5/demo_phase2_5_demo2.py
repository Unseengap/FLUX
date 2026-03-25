"""
demo_phase2_5_demo2.py — Demo: WRU sub-band interference visualization

Shows the FLUX-native physics inside the WRU:
  1. Per-band interference gates (cosine similarity between state and context)
  2. How different bands respond to different contexts
  3. Energy of the predicted wave across bands

This demonstrates what makes the WRU different from a GRU:
  - Independent per-band gating via physics (not learned sigmoids)
  - Constructive vs destructive interference visible per sub-band
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
from wave_recurrent_unit import BAND_SLICES
from train_wru import load_phase1_and_phase2, load_phase2_5_checkpoint


DEMO_TEXTS = [
    "The cat sat on the mat",
    "def fibonacci(n): return n",
    "∫₀^∞ e^(-x²) dx = √π/2",
]


@torch.no_grad()
def main():
    print("\n" + "=" * 65)
    print("  DEMO 2: WRU Sub-Band Interference Analysis")
    print("=" * 65)

    device = 'cpu'
    hf_token = os.environ.get('HF_TOKEN', '')

    # Load everything
    print("\n  Loading Phase 1 + 2 + 2.5...")
    components = load_phase1_and_phase2(device=device, hf_token=hf_token)
    cse = components['cse']
    chunker = components['chunker']
    w2f = components['w2f']
    wtt = components['wtt']

    wru = load_phase2_5_checkpoint(device=device, hf_token=hf_token)
    wru.eval()

    print(f"  ✓ All loaded\n")

    # ── Parameter summary ─────────────────────────────────────────────
    summary = wru.parameter_summary()
    print(f"  WRU Architecture:")
    for component, count in summary.items():
        print(f"    {component:>20s}: {count:>8,} params")
    print()

    for text in DEMO_TEXTS:
        byte_data = text.encode('utf-8')
        wave = cse.encode(text)
        pairs = chunker.chunk_with_bytes(wave.full, byte_data)

        if len(pairs) < 2:
            continue

        chunk_waves = torch.stack([p[0] for p in pairs]).to(device)
        chunk_bytes = [p[1] for p in pairs]

        n = max(1, len(pairs) // 2)
        prefix_mean = chunk_waves[:n].mean(dim=0)
        ctx = w2f(prefix_mean).unsqueeze(0)  # [1, 512]

        # Get prediction
        pred, state = wru(ctx)

        # Decode
        decoded = wtt.decode(pred.squeeze(0), temperature=0.5)
        decoded_text = decoded.decode('utf-8', errors='replace')
        gt_text = chunk_bytes[n].decode('utf-8', errors='replace')

        print(f"  ┌─ Text: \"{text}\"")
        prefix_words = [p[1].decode('utf-8', errors='replace') for p in pairs[:n]]
        print(f"  │  Prefix: {prefix_words}  →  predicted: \"{decoded_text}\"  gt: \"{gt_text}\"")

        # ── Analyze sub-band interference ─────────────────────────────
        ctx_wave = wru.context_proj(ctx).squeeze(0)  # [432]
        initial = wru.get_initial_state(1).squeeze(0).to(device)  # [432]

        print(f"  │")
        print(f"  │  Sub-band interference analysis:")
        print(f"  │  {'Band':<12s} {'Gate':>8s}  {'State σ':>8s}  {'Ctx σ':>8s}  {'Pred σ':>8s}  {'Type'}")
        print(f"  │  {'─'*65}")

        for name, (start, end) in BAND_SLICES.items():
            s_band = initial[start:end]
            c_band = ctx_wave[start:end]
            p_band = pred.squeeze(0)[start:end]

            gate = F.cosine_similarity(s_band.unsqueeze(0), c_band.unsqueeze(0), dim=-1).item()
            s_std = s_band.std().item()
            c_std = c_band.std().item()
            p_std = p_band.std().item()

            kind = "constructive ↑" if gate > 0.1 else "destructive ↓" if gate < -0.1 else "neutral ─"
            print(f"  │  {name:<12s} {gate:>+8.4f}  {s_std:>8.4f}  {c_std:>8.4f}  {p_std:>8.4f}  {kind}")

        # ── Energy analysis ───────────────────────────────────────────
        pred_energy = (pred ** 2).sum().item()
        print(f"  │")
        print(f"  │  Output energy: {pred_energy:.2f}  (cap: {wru.energy_cap})")
        print(f"  └─\n")

    print("  Done.\n")


if __name__ == '__main__':
    main()
