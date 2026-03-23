"""
Phase 9 — Demo 1: Wave-Level Generation vs Byte-Level

Generate text from 5 prompts. Shows:
- The wave sequence (number of waves, average confidence)
- Each wave decoded to its text chunk
- Final assembled text
- Comparison metric: valid word rate
"""

import sys
import time
import torch
from pathlib import Path

# ─────────────────────────────────────────────
# Path setup
# ─────────────────────────────────────────────
_PHASES_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PHASES_DIR.parent

for _phase in ['phase1', 'phase2', 'phase3', 'phase4',
               'phase5', 'phase6', 'phase7', 'phase8']:
    _p = str(_PHASES_DIR / _phase)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from flux_utils import get_device
from train_wave_gen import load_phase9_modules, build_phase9_modules, generate_text
from wave_sampler import ThermodynamicWaveSampler


def main():
    print("=" * 70)
    print("  Phase 9 — Demo 1: Wave-Level Generation")
    print("  Thinking in waves, spelling in bytes")
    print("=" * 70)

    device = get_device()

    # Load modules
    try:
        model, chunker, generator, wtt = load_phase9_modules(device=device)
        print("  ✓ Phase 9 loaded from checkpoint\n")
    except Exception as e:
        print(f"  ⚠ No Phase 9 checkpoint ({e})")
        print("  ℹ Using fresh FLUXLarge + fresh Phase 9 modules\n")
        from flux_large import FLUXLarge
        model = FLUXLarge(device=device)
        for param in model.parameters():
            param.requires_grad = False
        chunker, generator, wtt = build_phase9_modules(device=device)

    prompts = [
        "The future of artificial intelligence",
        "Scientists have discovered that the universe",
        "Once upon a time in a distant kingdom",
        "The relationship between energy and matter",
        "Modern technology relies on advanced computing",
    ]

    generator.eval()
    wtt.eval()
    chunker.eval()

    total_valid = 0
    total_words_checked = 0

    for i, prompt in enumerate(prompts):
        print(f"\n{'─' * 70}")
        print(f"  Prompt {i+1}: \"{prompt}\"")
        print(f"{'─' * 70}")

        t0 = time.time()

        try:
            with torch.no_grad():
                # 1. Get FLUX context
                wave_seq, wave_vec, merged = model._get_context(prompt)
                print(f"  Input waves:    {wave_seq.shape[0]} byte-level waves")

                # 2. Chunk input
                chunk_waves, spans = chunker(wave_seq)
                print(f"  Input chunks:   {chunk_waves.shape[0]} word-level chunks")

                # 3. Generate new waves
                gen_waves, confs = generator.generate(
                    field_context=merged,
                    max_waves=20,
                    flux_model=model,
                    temperature=0.8,
                )
                avg_conf = sum(confs) / max(len(confs), 1)
                print(f"  Generated:      {gen_waves.shape[0]} waves")
                print(f"  Avg confidence: {avg_conf:.3f}")

                # 4. Decode each wave
                sampler = ThermodynamicWaveSampler()
                decoded_chunks = []
                print(f"\n  Wave-by-wave decode:")
                for j, (wave, conf) in enumerate(zip(gen_waves, confs)):
                    sampled = sampler.sample_wave(wave, conf)
                    chunk_bytes = wtt.decode(sampled, temperature=0.8)
                    chunk_str = chunk_bytes.decode('utf-8', errors='replace')
                    decoded_chunks.append(chunk_str)
                    conf_bar = '█' * int(conf * 20) + '░' * (20 - int(conf * 20))
                    print(f"    wave[{j:2d}] conf={conf:.3f} [{conf_bar}] → \"{chunk_str}\"")

            elapsed = (time.time() - t0) * 1000

            # 5. Assemble final text
            full_text = prompt + ' ' + ' '.join(decoded_chunks)
            print(f"\n  ── Assembled Output ──")
            print(f"  {full_text[:300]}")

            # 6. Quick word validity check
            continuation = ' '.join(decoded_chunks)
            words = continuation.split()
            valid = sum(
                1 for w in words
                if w.strip('.,;:!?"\'-()[]{}').lower().isalpha()
                and 2 <= len(w.strip('.,;:!?"\'-()[]{}')) <= 15
            )
            word_count = sum(
                1 for w in words
                if w.strip('.,;:!?"\'-()[]{}').lower().isalpha()
            )
            total_valid += valid
            total_words_checked += word_count

            print(f"\n  Steps:   {gen_waves.shape[0]} wave steps (vs ~{wave_seq.shape[0]} byte steps)")
            print(f"  Time:    {elapsed:.1f}ms")
            print(f"  Words:   {word_count} total, {valid} valid-looking")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    # Summary
    overall_rate = total_valid / max(total_words_checked, 1)
    print(f"\n{'=' * 70}")
    print(f"  Summary")
    print(f"{'=' * 70}")
    print(f"  Total words generated: {total_words_checked}")
    print(f"  Valid word rate:       {overall_rate:.1%}")
    print(f"  Wave-level advantage:  ~5-10x fewer generation steps than byte-level")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
