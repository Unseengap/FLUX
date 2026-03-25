"""
Phase 9.5 — Demo 2: Context Diversity Visualization

Demonstrates that different prompts produce different Wave 0 outputs,
unlike Phase 9 where cross-context Wave 0 cosine was 1.000 (identical
outputs for every input regardless of content).

Shows:
    1. Processed context pairwise cosines
    2. Wave 0 cross-context cosines
    3. Hidden init diversity

Usage:
    python demo_phase9_5_demo2.py
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path

# ─────────────────────────────────────────────
# Path setup
# ─────────────────────────────────────────────
_PHASES_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PHASES_DIR.parent

for _phase in ['phase1', 'phase2', 'phase3', 'phase4',
               'phase5', 'phase6', 'phase7', 'phase8', 'phase9']:
    _p = str(_PHASES_DIR / _phase)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from flux_utils import get_device
from train_wave_gen_v2 import (
    load_phase9_5_modules,
    compute_merged_context,
)


def main() -> None:
    """Run context diversity visualization."""
    device = get_device()
    print("=" * 60)
    print("  Phase 9.5 Demo 2: Context Diversity")
    print("=" * 60)

    # Load Phase 9.5
    print("\n  Loading Phase 9.5 checkpoint...")
    model, chunker, generator, wtt = load_phase9_5_modules(device=device)
    generator.eval()

    # Diverse prompts
    prompts = [
        "The future of artificial intelligence",
        "Energy equals mass times the speed of light squared",
        "Quantum mechanics describes the behavior of particles",
        "The cat sat on the mat",
        "Photosynthesis converts sunlight into chemical energy",
    ]

    print(f"\n  📊 Context Diversity Test ({len(prompts)} prompts)")
    print("  " + "=" * 58)

    # Compute merged contexts and Wave 0 for each prompt
    contexts = []
    wave0s = []
    hiddens = []

    with torch.no_grad():
        for prompt in prompts:
            wave = model.cse.encode(prompt)
            wave_seq = wave.full.to(device)
            wave_vec = wave_seq.mean(dim=0)
            merged = compute_merged_context(model, wave_vec, device)

            ctx = generator.process_context(merged)
            contexts.append(ctx)

            hidden = generator.init_hidden(device, field_context=merged)
            hiddens.append(hidden.squeeze())

            context_wave = generator.context_to_wave(ctx)
            wave0, _, _ = generator.forward_step(
                generator.bos_wave, context_wave, hidden
            )
            wave0s.append(wave0)

    # Pairwise context cosines
    print("\n  📐 Processed Context Pairwise Cosines:")
    ctx_stack = torch.stack(contexts)
    for i in range(len(prompts)):
        for j in range(i + 1, len(prompts)):
            cos = F.cosine_similarity(
                ctx_stack[i].unsqueeze(0), ctx_stack[j].unsqueeze(0)
            ).item()
            print(f"    [{i}] vs [{j}]: {cos:.3f}")

    # Cross-context Wave 0 cosines
    print("\n  🌊 Wave 0 Cross-Context Cosines (must be < 0.85):")
    w0_stack = torch.stack(wave0s)
    w0_cosines = []
    for i in range(len(prompts)):
        for j in range(i + 1, len(prompts)):
            cos = F.cosine_similarity(
                w0_stack[i].unsqueeze(0), w0_stack[j].unsqueeze(0)
            ).item()
            w0_cosines.append(cos)
            print(f"    [{i}] vs [{j}]: {cos:.3f}")
    avg_w0 = sum(w0_cosines) / len(w0_cosines)
    print(f"    Average: {avg_w0:.3f} (Phase 9 was 1.000)")

    # Hidden init cosines
    print("\n  🧠 Hidden Init Cross-Context Cosines (must be < 0.90):")
    h_stack = torch.stack(hiddens)
    h_cosines = []
    for i in range(len(prompts)):
        for j in range(i + 1, len(prompts)):
            cos = F.cosine_similarity(
                h_stack[i].unsqueeze(0), h_stack[j].unsqueeze(0)
            ).item()
            h_cosines.append(cos)
            print(f"    [{i}] vs [{j}]: {cos:.3f}")
    avg_h = sum(h_cosines) / len(h_cosines)
    print(f"    Average: {avg_h:.3f} (Phase 9 was 0.996)")

    # Summary
    print(f"\n  {'='*58}")
    print(f"  Wave 0 avg cosine: {avg_w0:.3f}  {'✓ < 0.85' if avg_w0 < 0.85 else '✗ ≥ 0.85'}")
    print(f"  Hidden avg cosine: {avg_h:.3f}  {'✓ < 0.90' if avg_h < 0.90 else '✗ ≥ 0.90'}")

    # Prompt labels for reference
    print("\n  Prompt labels:")
    for i, p in enumerate(prompts):
        print(f"    [{i}] {p}")
    print("=" * 60)


if __name__ == '__main__':
    main()
