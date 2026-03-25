"""
demo_phase1_demo1.py — Text → Waves → Text Round-Trip

Shows the complete wave codec pipeline with a rich visual walkthrough:
  1. Encode text to SemanticWave
  2. Chunk the wave sequence
  3. Decode each chunk back to bytes
  4. Reconstruct the original text

Run: python demo_phase1_demo1.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from decode_gate import byte_accuracy

DEMO_TEXTS = [
    "The future of artificial intelligence is wave-first.",
    "café naïve résumé",
    "def encode(text): return waves",
    "∫₀^∞ e^(-x²) dx = √π/2",
    "Hello 👋 World 🌍",
    "水 water 물 вода مياه",
]


def print_bar(width: int = 60, char: str = '─'):
    print(char * width)


@torch.no_grad()
def run_demo():
    print()
    print_bar(60, '═')
    print("  FLUX v2 Phase 1 — Text ↔ Wave ↔ Text Round-Trip Demo")
    print_bar(60, '═')

    # ── Load checkpoint ───────────────────────────────────────────
    try:
        from train_codec import load_phase1_checkpoint
        codec = load_phase1_checkpoint(device='cpu')
        print("  ✓ Phase 1 v2 checkpoint loaded\n")
    except FileNotFoundError:
        print("  ⚠ No checkpoint found — running with UNTRAINED model")
        print("  To train: python train_codec.py --steps 30000\n")
        from train_codec import WaveCodec
        codec = WaveCodec(device='cpu')

    for i, text in enumerate(DEMO_TEXTS, 1):
        print_bar()
        print(f"  Example {i}: '{text}'")
        print_bar()

        # Encode
        wave = codec.cse.encode(text)
        print(f"  Input bytes    : {len(text.encode('utf-8'))}")
        print(f"  CSE wave shape : {list(wave.full.shape)}  (seq_len × 432)")

        # Chunk
        chunk_waves, spans = codec.chunker(wave.full)
        print(f"  Chunks         : {chunk_waves.shape[0]}  (spans: {spans[:5]}{'...' if len(spans) > 5 else ''})")

        # Decode
        decoded_text = codec.wtt.decode_to_text(chunk_waves, temperature=0.3)
        acc = byte_accuracy(text, decoded_text)

        print(f"  Decoded        : '{decoded_text}'")
        print(f"  Byte accuracy  : {acc:.1%}")

        # Wave statistics
        full = wave.full
        print(f"  Wave stats     : mean={full.mean():.3f}  std={full.std():.3f}  "
              f"norm={full.norm(dim=-1).mean():.3f}")

        status = '✓ PASS' if acc >= 0.90 else ('⚠ PARTIAL' if acc >= 0.60 else '✗ FAIL')
        print(f"  Status         : {status}")
        print()

    # ── Summary statistics ────────────────────────────────────────
    print_bar(60, '═')
    print("  Summary")
    print_bar()

    accs = []
    for text in DEMO_TEXTS:
        wave = codec.cse.encode(text)
        chunks, _ = codec.chunker(wave.full)
        dec = codec.wtt.decode_to_text(chunks, temperature=0.3)
        accs.append(byte_accuracy(text, dec))

    print(f"  Avg byte accuracy : {sum(accs)/len(accs):.1%}")
    print(f"  Min byte accuracy : {min(accs):.1%}")
    print(f"  Max byte accuracy : {max(accs):.1%}")
    print()

    overall_pass = sum(accs) / len(accs) >= 0.90
    if overall_pass:
        print("  ✓ Phase 1 v2 decode gate: PASSED")
    else:
        print("  ⚠ Phase 1 v2 decode gate: NOT YET (train more steps)")
    print_bar(60, '═')


if __name__ == '__main__':
    run_demo()
