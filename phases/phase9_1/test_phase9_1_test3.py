"""
Phase 9.1 — Test 3: UTF-8 Multi-Byte Character Accuracy

Validates that ContextWaveToText can handle non-ASCII characters
(accented letters, special characters). Phase 9 scored 0% on these.
Target: ≥50%.

Run: python test_phase9_1_test3.py
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
from train_context_wtt import load_phase9_1_modules


# UTF-8 test words — all multi-byte
UTF8_TEST_WORDS = [
    ("café", "The café on the corner is open late."),
    ("naïve", "She was naïve about the situation."),
    ("résumé", "He updated his résumé for the interview."),
    ("piñata", "The piñata was filled with candy."),
    ("Zürich", "They traveled to Zürich by train."),
    ("château", "The château overlooked the valley."),
    ("crème", "The crème brûlée was delicious."),
    ("über", "The über driver arrived on time."),
    ("señor", "Buenos días señor how are you today."),
    ("François", "François studied at the university."),
]


def main():
    print("=" * 60)
    print("  Phase 9.1 — Test 3: UTF-8 Multi-Byte Accuracy")
    print("=" * 60)

    device = get_device()
    print(f"  Device: {device}")

    # Load Phase 9.1 checkpoint
    cse, chunker, wtt, ckpt = load_phase9_1_modules(device=device)
    wtt.eval()

    # Test each UTF-8 word in its carrier sentence
    print(f"\n  Testing {len(UTF8_TEST_WORDS)} multi-byte words...\n")
    correct = 0
    total = len(UTF8_TEST_WORDS)

    for word, carrier in UTF8_TEST_WORDS:
        try:
            wave = cse.encode(carrier)
            wave_seq = wave.full.to(device)

            chunk_waves, spans = chunker(wave_seq)
            decoded_chunks = wtt.decode_sequence(chunk_waves, temperature=0.3)
            decoded_text = b''.join(decoded_chunks)

            try:
                decoded_str = decoded_text.decode('utf-8', errors='replace')
            except Exception:
                decoded_str = decoded_text.decode('latin-1', errors='replace')

            # Check if the word appears in decoded output
            found = word in decoded_str
            if found:
                correct += 1
                status = '✓'
            else:
                status = '✗'

            # Show hex of the multi-byte chars for debugging
            word_hex = word.encode('utf-8').hex()
            print(
                f"  {status} {word:<15} (hex:{word_hex:<20}) → "
                f"'{decoded_str[:60]}'"
            )

        except Exception as e:
            print(f"  ✗ {word:<15} → ERROR: {e}")

    accuracy = correct / max(total, 1)
    passed = accuracy >= 0.50

    print(f"\n  UTF-8 accuracy: {correct}/{total} ({accuracy:.1%})")
    print(f"  Threshold: ≥50%")
    print(f"  Phase 9 baseline: 0% (café, naïve both failed)")
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")

    # Also test some single-byte edge cases
    print(f"\n  Additional: ASCII edge cases...")
    edge_cases = [
        ("xyz", "The letters xyz appear rarely."),
        ("42", "The answer is 42 to everything."),
        ("Dr.", "Dr. Smith arrived at the hospital."),
        ("U.S.A.", "She traveled across the U.S.A. last summer."),
    ]
    edge_correct = 0
    for word, carrier in edge_cases:
        try:
            wave = cse.encode(carrier)
            wave_seq = wave.full.to(device)
            chunk_waves, spans = chunker(wave_seq)
            decoded_chunks = wtt.decode_sequence(chunk_waves, temperature=0.3)
            decoded_text = b''.join(decoded_chunks)
            decoded_str = decoded_text.decode('utf-8', errors='replace')
            found = word in decoded_str
            if found:
                edge_correct += 1
            status = '✓' if found else '✗'
            print(f"    {status} '{word}' in: '{decoded_str[:60]}'")
        except Exception as e:
            print(f"    ✗ '{word}' → ERROR: {e}")

    print(f"  ASCII edge cases: {edge_correct}/{len(edge_cases)}")

    # Save result
    results = PhaseResults(phase=9, component_name="ContextWaveToText UTF-8")
    results.add_test(
        "UTF-8 Multi-Byte",
        passed=passed,
        score=f"{accuracy:.1%} ({correct}/{total})",
        threshold="≥50%",
    )
    results.add_metric("ASCII edge cases", f"{edge_correct}/{len(edge_cases)}")

    assert passed, f"UTF-8 accuracy {accuracy:.1%} < 50% threshold"
    print("\n  ✓ Test 3 passed!")


if __name__ == '__main__':
    main()
