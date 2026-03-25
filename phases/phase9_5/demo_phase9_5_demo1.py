"""
Phase 9.5 — Demo 1: Free Generation Quality Comparison

Demonstrates that the retrained WaveGenerator produces recognizable
English text, unlike Phase 9 which output gibberish like
"a s y  y   u u m t Fk  u  u n h  u  u at t  t".

Usage:
    python demo_phase9_5_demo1.py
"""

import sys
import torch
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
    generate_text,
)


def main() -> None:
    """Run free generation demo with multiple prompts."""
    device = get_device()
    print("=" * 60)
    print("  Phase 9.5 Demo 1: Free Generation Quality")
    print("=" * 60)

    # Load Phase 9.5 checkpoint
    print("\n  Loading Phase 9.5 checkpoint...")
    model, chunker, generator, wtt = load_phase9_5_modules(device=device)

    # Test prompts
    prompts = [
        "The future of artificial intelligence",
        "Scientists have discovered",
        "In the beginning",
        "The relationship between energy and matter",
        "Modern technology relies on",
        "The history of mathematics reveals",
        "Research shows that quantum",
        "Water freezes at zero degrees",
    ]

    print("\n  📊 Free Generation Results")
    print("  " + "=" * 58)

    total_valid = 0
    total_words = 0

    for prompt in prompts:
        try:
            result = generate_text(
                prompt, model, chunker, generator, wtt,
                max_waves=15, temperature=0.8,
            )
            continuation = result[len(prompt):].strip()

            # Count valid words
            words = continuation.split()
            valid = 0
            for w in words:
                clean = w.strip('.,;:!?"\'-()[]{}').lower()
                if clean.isalpha() and len(clean) >= 2 and len(clean) <= 15:
                    valid += 1
                    total_valid += 1
                total_words += 1

            # Display
            print(f"\n  Prompt: \"{prompt}\"")
            print(f"  Output: \"{result[:200]}\"")
            print(f"  Valid words: {valid}/{len(words)}")

        except Exception as e:
            print(f"\n  Prompt: \"{prompt}\"")
            print(f"  ✗ Error: {e}")

    valid_rate = total_valid / max(total_words, 1)
    print(f"\n  {'='*58}")
    print(f"  Overall valid word rate: {total_valid}/{total_words} ({valid_rate:.1%})")
    print(f"  Phase 9 comparison: gibberish (\"a s y  y   u u m t Fk\")")
    print(f"  Target: > 30% valid words")
    print(f"  Result: {'✓ PASS' if valid_rate > 0.3 else '✗ FAIL'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
