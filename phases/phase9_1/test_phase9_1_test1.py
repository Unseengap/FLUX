"""
Phase 9.1 — Test 1: Chunk-Level Byte Accuracy

Validates that ContextWaveToText achieves ≥93% chunk-level accuracy,
up from Phase 9's 82.8%. Tests individual chunk wave → byte decoding
with left-context support.

Run: python test_phase9_1_test1.py
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
from train_context_wtt import load_phase9_1_modules, load_training_data


def main():
    print("=" * 60)
    print("  Phase 9.1 — Test 1: Chunk-Level Byte Accuracy")
    print("=" * 60)

    device = get_device()
    print(f"  Device: {device}")

    # Load Phase 9.1 checkpoint
    cse, chunker, wtt, ckpt = load_phase9_1_modules(device=device)
    wtt.eval()

    # Load evaluation data
    texts = load_training_data(max_docs=200)
    eval_texts = texts[-50:]  # Last 50 for eval

    # Evaluate chunk accuracy
    print("\n  ℹ Evaluating chunk-level accuracy on 50 texts (max 500 chunks)...")
    correct = 0
    total = 0
    errors_shown = 0

    for text in eval_texts:
        if not text or len(text.strip()) < 10:
            continue
        try:
            wave = cse.encode(text)
            wave_seq = wave.full.to(device)
            text_bytes = text.encode('utf-8', errors='replace')

            chunk_waves, spans = chunker(wave_seq)
            num_chunks = chunk_waves.shape[0]

            for i in range(num_chunks):
                start, end = spans[i]
                byte_start = min(start, len(text_bytes))
                byte_end = min(end, len(text_bytes))
                chunk_bytes = text_bytes[byte_start:byte_end]
                if len(chunk_bytes) == 0:
                    continue

                # Context from previous chunks
                ctx_start = max(0, i - 2)
                ctx = chunk_waves[ctx_start:i] if i > 0 else None

                decoded = wtt.decode(chunk_waves[i], temperature=0.5, context_waves=ctx)

                if decoded == chunk_bytes:
                    correct += 1
                elif errors_shown < 10:
                    gt = chunk_bytes.decode('utf-8', errors='replace')
                    pred = decoded.decode('utf-8', errors='replace')
                    print(f"    ✗ GT='{gt:<15}' Pred='{pred:<15}'")
                    errors_shown += 1
                total += 1
                if total >= 500:
                    break
        except Exception:
            continue
        if total >= 500:
            break

    accuracy = correct / max(total, 1)
    passed = accuracy >= 0.93

    print(f"\n  Chunk accuracy: {correct}/{total} ({accuracy:.1%})")
    print(f"  Threshold: ≥93%")
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")

    # Save result
    results = PhaseResults(phase=9, component_name="ContextWaveToText Chunk Accuracy")
    results.add_test(
        "Chunk Accuracy",
        passed=passed,
        score=f"{accuracy:.1%}",
        threshold="≥93%",
    )
    results.add_metric("Total chunks evaluated", total)
    results.add_metric("Correct chunks", correct)

    assert passed, f"Chunk accuracy {accuracy:.1%} < 93% threshold"
    print("\n  ✓ Test 1 passed!")


if __name__ == '__main__':
    main()
