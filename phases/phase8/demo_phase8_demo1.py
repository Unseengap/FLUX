"""
Phase 8 — Demo 1: FLUX vs GPT-2 Generation Quality

Side-by-side comparison of text generation from FLUXModel (Phase 8)
and GPT-2 small on the same prompts.

Shows:
  - Generation quality comparison
  - Speed comparison
  - Unique FLUX properties (byte-level, no tokenizer)
"""

import sys
import time
import torch
from pathlib import Path

_PHASES_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PHASES_DIR.parent
for _phase in ['phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'phase6', 'phase7']:
    _p = str(_PHASES_DIR / _phase)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from flux_large import FLUXLarge
from benchmark_gpt2 import GPT2Baseline
from flux_utils import get_device, checkpoint_exists


def main():
    print("=" * 70)
    print("  Demo 1: FLUX vs GPT-2 — Generation Quality Comparison")
    print("=" * 70)

    device = get_device()

    # Load FLUX
    print("\n  Loading FLUXModel (Phase 8)...")
    if checkpoint_exists(8):
        flux = FLUXLarge.from_phase8_checkpoint(device=device)
    else:
        flux = FLUXLarge(device=device)
        print("  ⚠ Using untrained FLUXModel (no checkpoint)")

    stats = flux.get_stats()
    print(f"  ✓ FLUXModel: {stats.total_params:,} params")

    # Load GPT-2
    print("\n  Loading GPT-2 small...")
    gpt2 = GPT2Baseline(device=device)

    # ── Generation Comparison ──
    prompts = [
        "The future of artificial intelligence is",
        "In a world where machines can think,",
        "The discovery of gravitational waves proved that",
        "Deep learning has revolutionized how we",
        "The relationship between physics and computation",
    ]

    print("\n" + "═" * 70)
    print("  Side-by-Side Generation")
    print("═" * 70)

    for prompt in prompts:
        print(f"\n  Prompt: \"{prompt}\"")
        print(f"  {'─'*60}")

        # FLUX generation
        t0 = time.time()
        flux_text = flux.generate(prompt, max_length=60, temperature=0.9)
        flux_time = (time.time() - t0) * 1000
        flux_continuation = flux_text[len(prompt):]
        print(f"  FLUX:  {flux_continuation[:80]}")
        print(f"         ({flux_time:.0f}ms, {len(flux_continuation)} bytes)")

        # GPT-2 generation
        if gpt2.available:
            try:
                t0 = time.time()
                inputs = gpt2.tokenizer(prompt, return_tensors='pt').to(device)
                with torch.no_grad():
                    outputs = gpt2.model.generate(
                        inputs.input_ids, max_new_tokens=60,
                        do_sample=True, temperature=0.9,
                        pad_token_id=gpt2.tokenizer.eos_token_id,
                    )
                gpt2_text = gpt2.tokenizer.decode(outputs[0], skip_special_tokens=True)
                gpt2_time = (time.time() - t0) * 1000
                gpt2_continuation = gpt2_text[len(prompt):]
                print(f"  GPT-2: {gpt2_continuation[:80]}")
                print(f"         ({gpt2_time:.0f}ms, {len(gpt2_continuation)} chars)")
            except Exception as e:
                print(f"  GPT-2: [error: {e}]")
        else:
            print(f"  GPT-2: [not available — install transformers]")

    # ── FLUX Unique Properties ──
    print("\n" + "═" * 70)
    print("  FLUX Unique Properties")
    print("═" * 70)

    print("\n  1. Byte-Level (No Tokenizer):")
    multilingual = "Hello 你好 مرحبا 🌍 → works without any vocabulary"
    wave = flux.encode(multilingual)
    print(f"     Input:  \"{multilingual}\"")
    print(f"     Wave shape: {wave.full.shape}")
    print(f"     ✓ Any UTF-8 input — no OOV errors ever")

    print(f"\n  2. Real-Time Learning:")
    fact = "FLUXLarge has been benchmarked against GPT-2 in Phase 8"
    flux.learn_fact(fact)
    results = flux.query("What was benchmarked in Phase 8?", k=2)
    print(f"     Learned: \"{fact}\"")
    if results:
        print(f"     Query:   \"What was benchmarked in Phase 8?\"")
        print(f"     Result:  [{results[0][1]:.3f}] {results[0][0][:60]}")

    print(f"\n  3. Model Statistics:")
    stats = flux.get_stats()
    print(f"     Total params:     {stats.total_params:,}")
    print(f"     Learning steps:   {stats.learning_steps}")
    print(f"     Episodic entries: {stats.episodic_entries}")
    print(f"     Field energy:     {stats.field_energy:.4f}")

    print("\n" + "═" * 70)
    print("  ✓ Demo 1 Complete")
    print("═" * 70)


if __name__ == '__main__':
    main()
