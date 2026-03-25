"""
Phase 9.1 — Demo 2: Live Decode Visualization

Visualizes the chunk-by-chunk decoding process, showing how
left-context influences each chunk's byte predictions.

Run: python demo_phase9_1_demo2.py
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


DEMO_TEXTS = [
    "The future of artificial intelligence is profoundly exciting.",
    "Scientists have discovered a fundamental relationship in quantum physics.",
    "The café serves excellent crème brûlée and espresso every morning.",
]


def main():
    print("=" * 60)
    print("  Phase 9.1 — Demo 2: Live Decode Visualization")
    print("=" * 60)

    device = get_device()
    print(f"  Device: {device}\n")

    cse, chunker, wtt, ckpt = load_phase9_1_modules(device=device)
    wtt.eval()

    for text in DEMO_TEXTS:
        print(f"\n  {'─'*56}")
        print(f"  Input: \"{text}\"")
        print(f"  {'─'*56}")

        try:
            wave = cse.encode(text)
            wave_seq = wave.full.to(device)
            text_bytes = text.encode('utf-8', errors='replace')

            chunk_waves, spans = chunker(wave_seq)
            num_chunks = chunk_waves.shape[0]

            print(f"  CSE → {wave_seq.shape[0]} byte-waves → {num_chunks} chunks\n")

            # Decode chunk by chunk, showing context influence
            reconstructed = []
            for i in range(num_chunks):
                start, end = spans[i]
                byte_start = min(start, len(text_bytes))
                byte_end = min(end, len(text_bytes))
                gt_bytes = text_bytes[byte_start:byte_end]
                gt_str = gt_bytes.decode('utf-8', errors='replace')

                # Context info
                ctx_start = max(0, i - 2)
                if i > 0:
                    ctx = chunk_waves[ctx_start:i]
                    ctx_desc = f"ctx=[{ctx_start}..{i-1}]"
                else:
                    ctx = None
                    ctx_desc = "ctx=None"

                # Decode with context
                decoded_ctx = wtt.decode(
                    chunk_waves[i], temperature=0.3, context_waves=ctx
                )
                pred_str = decoded_ctx.decode('utf-8', errors='replace')

                # Also decode WITHOUT context for comparison
                decoded_no_ctx = wtt.decode(
                    chunk_waves[i], temperature=0.3, context_waves=None
                )
                pred_no_ctx = decoded_no_ctx.decode('utf-8', errors='replace')

                match_ctx = '✓' if decoded_ctx == gt_bytes else '✗'
                match_no = '✓' if decoded_no_ctx == gt_bytes else '✗'

                # Show context effect
                ctx_helped = (decoded_ctx == gt_bytes) and (decoded_no_ctx != gt_bytes)
                ctx_marker = ' ★' if ctx_helped else ''

                print(
                    f"  Chunk {i:>2} [{start:>3}:{end:>3}]  "
                    f"GT='{gt_str:<12}'  "
                    f"{match_ctx} ctx='{pred_str:<12}' ({ctx_desc})  "
                    f"{match_no} no_ctx='{pred_no_ctx:<12}'"
                    f"{ctx_marker}"
                )

                reconstructed.append(pred_str)

            # Full reconstruction
            full_text = ''.join(reconstructed)
            print(f"\n  Reconstructed: \"{full_text}\"")

            # Word-level comparison
            orig_words = text.split()
            recon_words = full_text.split()
            word_matches = sum(
                1 for j, w in enumerate(orig_words)
                if j < len(recon_words) and w == recon_words[j]
            )
            print(
                f"  Word accuracy: {word_matches}/{len(orig_words)} "
                f"({word_matches/max(len(orig_words),1):.0%})"
            )

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n  {'─'*56}")
    print(f"  ★ = context improved the decode (wrong without, correct with)")
    print(f"\n  ✓ Demo 2 complete")


if __name__ == '__main__':
    main()
