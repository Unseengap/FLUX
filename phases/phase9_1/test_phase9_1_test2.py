"""
Phase 9.1 — Test 2: Hard Word Spelling Accuracy

Validates that ContextWaveToText achieves ≥60% on hard multi-chunk
word spelling, up from Phase 9's 33% (5/15). Tests words embedded
in carrier sentences.

Run: python test_phase9_1_test2.py
"""

import sys
import torch
from pathlib import Path

# Path setup
_PHASES_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PHASES_DIR.parent
for _p in ['phase1', 'phase9', 'phase9_1']:
    _path = str(_PHASES_DIR / _p)
    if _path not in sys.path:
        sys.path.insert(0, _path)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from flux_utils import PhaseResults, get_device
from train_context_wtt import load_phase9_1_modules, HARD_SPELLING_WORDS


def main():
    print("=" * 60)
    print("  Phase 9.1 — Test 2: Hard Word Spelling Accuracy")
    print("=" * 60)

    device = get_device()
    print(f"  Device: {device}")

    # Load Phase 9.1 checkpoint
    cse, chunker, wtt, ckpt = load_phase9_1_modules(device=device)
    wtt.eval()

    # Test each word in a carrier sentence
    print(f"\n  Testing {len(HARD_SPELLING_WORDS)} words in carrier sentences...\n")
    correct = 0
    total = len(HARD_SPELLING_WORDS)

    for word in HARD_SPELLING_WORDS:
        carrier = f"This is {word} here"
        try:
            wave = cse.encode(carrier)
            wave_seq = wave.full.to(device)

            chunk_waves, spans = chunker(wave_seq)

            # Decode with context
            decoded_chunks = wtt.decode_sequence(chunk_waves, temperature=0.3)
            decoded_text = b''.join(decoded_chunks)

            try:
                decoded_str = decoded_text.decode('utf-8', errors='replace')
            except Exception:
                decoded_str = decoded_text.decode('latin-1', errors='replace')

            found = word in decoded_str
            if found:
                correct += 1
                status = '✓'
            else:
                status = '✗'

            chunks_used = chunk_waves.shape[0]
            print(
                f"  {status} {word:<20} → '{decoded_str:<40}' "
                f"GT='{carrier:<30}' ({chunks_used} chunks)"
            )

        except Exception as e:
            print(f"  ✗ {word:<20} → ERROR: {e}")

    accuracy = correct / max(total, 1)
    passed = accuracy >= 0.60

    print(f"\n  Hard spelling accuracy: {correct}/{total} ({accuracy:.1%})")
    print(f"  Threshold: ≥60% (9/15)")
    print(f"  Phase 9 baseline: 33% (5/15)")
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")

    # Save result
    results = PhaseResults(phase=9, component_name="ContextWaveToText Hard Spelling")
    results.add_test(
        "Hard Spelling",
        passed=passed,
        score=f"{accuracy:.1%} ({correct}/{total})",
        threshold="≥60%",
    )

    assert passed, f"Hard spelling {accuracy:.1%} < 60% threshold"
    print("\n  ✓ Test 2 passed!")


if __name__ == '__main__':
    main()
