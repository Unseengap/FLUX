"""
Phase 7 Demo 1: End-to-End Text Generation

Demonstrates the full FLUX pipeline: text input → CSE encoding →
field interaction → gravitational relevance → causal processing →
thermodynamic settling → memory storage → byte-level text generation.

Shows:
  - Multiple prompts with different styles
  - Generation latency and throughput
  - Model statistics across all components
"""

import sys
import time
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

from flux_utils import get_device, load_checkpoint
from flux_model import FLUXModel
from flux_generate import TextGenerator


def demo_end_to_end_generation():
    print("=" * 65)
    print("  DEMO 1: End-to-End Text Generation")
    print("  Complete FLUX pipeline: text → wave → field → output")
    print("=" * 65)

    DEVICE = get_device()

    print("\n  Loading FLUXModel from phase checkpoints 1–6...")
    model = FLUXModel.from_checkpoints(device=DEVICE)
    generator = TextGenerator(model)

    stats = model.get_stats()
    print(f"\n  Model assembled:")
    print(f"    Total parameters: {stats.total_params:,}")
    print(f"    Field energy:     {stats.field_energy:.4f}")
    print(f"    CGN nodes:        {stats.cgn_params:,} params")
    print(f"    Memory entries:   {stats.episodic_entries} episodic, {stats.working_entries} working")

    # ── Generation prompts ──
    prompts = [
        ("Natural language",   "The future of artificial intelligence is"),
        ("Science",            "In physics, the speed of light"),
        ("Code",               "def fibonacci(n):"),
        ("Philosophy",         "The meaning of consciousness"),
        ("FLUX architecture",  "Resonance fields replace weight matrices because"),
    ]

    print(f"\n{'─'*65}")
    print(f"  Generating text for {len(prompts)} prompts")
    print(f"{'─'*65}")

    for category, prompt in prompts:
        result = generator.generate(
            prompt, max_length=60, temperature=0.9, top_k=40
        )
        print(f"\n  [{category}]")
        print(f"  Prompt:    {prompt}")
        print(f"  Generated: {result.full_text[:120]}")
        print(f"  Stats:     +{result.num_bytes_generated} bytes, "
              f"{result.latency_ms:.0f}ms, "
              f"{result.bytes_per_second:.1f} bytes/s")

    # ── Greedy vs Sampling comparison ──
    print(f"\n{'─'*65}")
    print(f"  Greedy vs Sampling comparison")
    print(f"{'─'*65}")

    test_prompt = "The quick brown"
    greedy = generator.generate_greedy(test_prompt, max_length=30)
    sampled = generator.generate(test_prompt, max_length=30, temperature=1.0)

    print(f"\n  Prompt: '{test_prompt}'")
    print(f"  Greedy:  {greedy.full_text[:80]}")
    print(f"  Sampled: {sampled.full_text[:80]}")

    # ── Final stats ──
    final_stats = model.get_stats()
    print(f"\n{'─'*65}")
    print(f"  Generation demonstrated:")
    print(f"  ✓ {len(prompts)} prompts processed through full FLUX pipeline")
    print(f"  ✓ CSE → GR → CGN → Field → TL → Memory → OutputHead → bytes")
    print(f"  ✓ Byte-level autoregressive generation working")
    print(f"  ✓ Both greedy and sampling decoding supported")
    print(f"  ✓ Model: {final_stats.total_params:,} params, "
          f"{final_stats.learning_steps} learning steps")
    print(f"{'─'*65}")


if __name__ == "__main__":
    demo_end_to_end_generation()
