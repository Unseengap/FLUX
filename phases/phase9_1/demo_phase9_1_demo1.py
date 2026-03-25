"""
Phase 9.1 — Demo 1: Spelling Comparison (Phase 9 vs Phase 9.1)

Side-by-side comparison showing how context-aware decoding improves
spelling accuracy over the original WaveToText. Uses the same
diagnostic words from Phase 9's WTT Diag cell.

Run: python demo_phase9_1_demo1.py
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

from flux_utils import get_device
from train_context_wtt import load_phase9_1_modules


# Test sentences with increasing complexity
TEST_PHRASES = [
    "The quick brown fox jumps over the lazy dog.",
    "Energy equals mass times the speed of light squared.",
    "Quantum mechanics describes the behavior of particles.",
    "The fundamental relationship between science and technology.",
    "Photosynthesis converts sunlight into chemical energy efficiently.",
    "Neural networks learn from data through backpropagation.",
    "The café on the corner serves excellent crème brûlée.",
    "François updated his résumé before the interview at Zürich.",
]


def main():
    print("=" * 60)
    print("  Phase 9.1 — Demo 1: Spelling Comparison")
    print("=" * 60)

    device = get_device()
    print(f"  Device: {device}\n")

    # Load Phase 9.1
    cse, chunker, wtt, ckpt = load_phase9_1_modules(device=device)
    wtt.eval()

    # Also try loading Phase 9 WTT for comparison (if available)
    phase9_wtt = None
    try:
        from wave_to_text import WaveToText
        p9cfg = ckpt.get('phase9_config', ckpt.get('phase9_1_config', {}))
        phase9_wtt = WaveToText(
            wave_dim=432,
            hidden_dim=p9cfg.get('wtt_hidden_dim', 256),
            max_bytes=p9cfg.get('wtt_max_bytes', 20),
        ).to(device)
        if 'wave_to_text_state_dict' in ckpt:
            # Try old WTT weights (might match or might be ContextWTT weights)
            try:
                phase9_wtt.load_state_dict(ckpt['wave_to_text_state_dict'])
                print("  ✓ Phase 9 WTT loaded for comparison")
            except Exception:
                phase9_wtt = None
                print("  ℹ Phase 9 WTT not compatible (using 9.1 only)")
        else:
            phase9_wtt = None
    except Exception:
        print("  ℹ Phase 9 WTT not available for comparison")

    print()
    print("  Phrase-by-phrase spelling test")
    print("  " + "=" * 58)

    total_9_1 = 0
    correct_9_1 = 0
    total_9 = 0
    correct_9 = 0

    for phrase in TEST_PHRASES:
        print(f"\n  Original: \"{phrase}\"")

        try:
            wave = cse.encode(phrase)
            wave_seq = wave.full.to(device)
            chunk_waves, spans = chunker(wave_seq)
            num_chunks = chunk_waves.shape[0]

            # Phase 9.1 decode (with context)
            decoded_9_1 = wtt.decode_sequence(chunk_waves, temperature=0.3)
            text_9_1 = b''.join(decoded_9_1)
            try:
                str_9_1 = text_9_1.decode('utf-8', errors='replace')
            except Exception:
                str_9_1 = text_9_1.decode('latin-1', errors='replace')

            # Word accuracy for 9.1
            orig_words = phrase.split()
            dec_words = str_9_1.split()
            for j, w in enumerate(orig_words):
                total_9_1 += 1
                if j < len(dec_words) and w == dec_words[j]:
                    correct_9_1 += 1

            print(f"  9.1    : \"{str_9_1}\"  ({num_chunks} chunks)")

            # Phase 9 decode (no context) — if available
            if phase9_wtt is not None:
                phase9_wtt.eval()
                decoded_9 = []
                for i in range(num_chunks):
                    d = phase9_wtt.decode(chunk_waves[i], temperature=0.3)
                    decoded_9.append(d)
                text_9 = b''.join(decoded_9)
                try:
                    str_9 = text_9.decode('utf-8', errors='replace')
                except Exception:
                    str_9 = text_9.decode('latin-1', errors='replace')

                dec_words_9 = str_9.split()
                for j, w in enumerate(orig_words):
                    total_9 += 1
                    if j < len(dec_words_9) and w == dec_words_9[j]:
                        correct_9 += 1

                print(f"  9.0    : \"{str_9}\"")

        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n  " + "=" * 58)
    print(f"  Summary:")
    print(f"    Phase 9.1 word accuracy: {correct_9_1}/{total_9_1} "
          f"({correct_9_1/max(total_9_1,1):.1%})")
    if total_9 > 0:
        print(f"    Phase 9.0 word accuracy: {correct_9}/{total_9} "
              f"({correct_9/max(total_9,1):.1%})")
    else:
        print(f"    Phase 9.0: not available for comparison")

    print(f"\n  ✓ Demo 1 complete")


if __name__ == '__main__':
    main()
