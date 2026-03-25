"""
demo_phase2_5_demo1.py — Demo: Prompt → WRU next-wave prediction → decoded text

Shows the complete pipeline:
  text → CSE → WaveChunker → prefix[:n] → mean → WaveToField → WRU → predicted_wave → WTT → bytes

For each demo text, splits at multiple positions and shows what the WRU predicts
as the next word given increasing amounts of context.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase2'))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from train_wru import load_phase1_and_phase2, load_phase2_5_checkpoint


DEMO_TEXTS = [
    "The cat sat on the mat",
    "Energy equals mass times the speed of light squared",
    "Photosynthesis converts sunlight into chemical energy",
    "Machine learning models translate patterns into predictions",
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
]


@torch.no_grad()
def main():
    print("\n" + "=" * 65)
    print("  DEMO 1: WRU Next-Wave Prediction → Decoded Text")
    print("=" * 65)

    device = 'cpu'
    hf_token = os.environ.get('HF_TOKEN', '')

    # Load everything
    print("\n  Loading Phase 1 + 2 + 2.5...")
    components = load_phase1_and_phase2(device=device, hf_token=hf_token)
    cse = components['cse']
    chunker = components['chunker']
    wtt = components['wtt']
    w2f = components['w2f']

    wru = load_phase2_5_checkpoint(device=device, hf_token=hf_token)
    wru.eval()
    print(f"  ✓ All components loaded\n")

    for text in DEMO_TEXTS:
        byte_data = text.encode('utf-8')
        wave = cse.encode(text)
        pairs = chunker.chunk_with_bytes(wave.full, byte_data)

        if len(pairs) < 2:
            print(f"  ⚠ Skipping '{text[:40]}' — too few chunks\n")
            continue

        chunk_waves = torch.stack([p[0] for p in pairs]).to(device)
        chunk_bytes = [p[1] for p in pairs]
        chunk_texts = [b.decode('utf-8', errors='replace') for b in chunk_bytes]

        print(f"  ┌─ Text: \"{text}\"")
        print(f"  │  Chunks: {chunk_texts}")
        print(f"  │")

        # Try multiple prefix lengths
        for n in range(1, len(pairs)):
            prefix_text = ''.join(chunk_texts[:n])
            gt_text = chunk_texts[n] if n < len(chunk_texts) else "?"

            prefix_mean = chunk_waves[:n].mean(dim=0)
            ctx = w2f(prefix_mean).unsqueeze(0)
            pred_wave, _ = wru(ctx)

            decoded = wtt.decode(pred_wave.squeeze(0), temperature=0.5)
            decoded_text = decoded.decode('utf-8', errors='replace')

            match = '✓' if decoded_text.strip() == gt_text.strip() else '·'
            print(f"  │  [{n}/{len(pairs)}] prefix=\"{prefix_text}\"  "
                  f"→  pred=\"{decoded_text}\"  gt=\"{gt_text}\"  {match}")

        print(f"  └─\n")

    print("  Done.\n")


if __name__ == '__main__':
    main()
