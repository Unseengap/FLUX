"""
Phase 8.5 — Demo 1: Stage-by-Stage Generation Showcase

Shows how generation quality improves at each curriculum stage.
Generates text from the same prompts after each stage to visualize
the "learning to spell" progression.

If a full curriculum isn't available, demonstrates current generation.
"""

import sys
import torch
from pathlib import Path

_PHASES_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PHASES_DIR.parent
for _phase in ['phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'phase6', 'phase7', 'phase8']:
    _p = str(_PHASES_DIR / _phase)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from flux_large import FLUXLarge
from curriculum_data import (
    STAGE_NAMES, get_spelling_test_words,
    get_sentence_test_prompts, generate_curriculum_data,
)
from flux_utils import get_device, checkpoint_exists


def main():
    print("=" * 70)
    print("  Demo 1: Curriculum Training — Generation Showcase")
    print("  How FLUX learns to spell: bytes → words → sentences")
    print("=" * 70)

    device = get_device()

    # Load model
    ckpt_path = Path('checkpoints') / 'phase8_5.phase.pt'
    curriculum_info = None

    if ckpt_path.exists():
        print("\n  Loading Phase 8.5 (curriculum-trained) model...")
        ckpt = torch.load(str(ckpt_path), map_location='cpu')
        model = FLUXLarge(config=ckpt.get('config', None), device=device)
        if 'decoder_state_dict' in ckpt:
            model.decoder.load_state_dict(ckpt['decoder_state_dict'])
        for key in ['wave_to_field_state', 'field_state_dict', 'output_head_state']:
            if key in ckpt:
                try:
                    attr = key.replace('_state_dict', '').replace('_state', '')
                    getattr(model, attr).load_state_dict(ckpt[key])
                except Exception:
                    pass
        model = model.to(device)
        curriculum_info = ckpt.get('curriculum_state', {})
        print("  ✓ Phase 8.5 loaded")
    elif checkpoint_exists(8):
        model = FLUXLarge.from_phase8_checkpoint(device=device)
        print("  ℹ Phase 8.5 not found — using Phase 8")
    else:
        model = FLUXLarge(device=device)
        print("  ⚠ Using untrained model")

    # ── Curriculum Summary ──
    if curriculum_info:
        print(f"\n  ── Curriculum Training History ──")
        history = curriculum_info.get('stage_history', [])
        for s in history:
            status = "✓" if s.get('advanced', False) else "○"
            print(
                f"    {status} Stage {s['stage']}: {s['name']:<24} "
                f"steps={s['steps']:>5} loss={s['avg_loss']:.4f} "
                f"acc={s['accuracy']:.1%}"
            )
        print(f"    Total steps: {curriculum_info.get('total_steps', '?')}")

    # ── 1. Byte-Level Test ──
    print(f"\n{'─' * 70}")
    print(f"  📝 Byte-Level: Can it reproduce individual characters?")
    print(f"{'─' * 70}")

    test_chars = list('abcdefghijklmnopqrstuvwxyz .,!')
    model.eval()
    correct = 0
    with torch.no_grad():
        for ch in test_chars:
            wave = model.cse.encode(ch)
            wave_sequence = wave.full.to(device)
            wave_vec = wave_sequence.mean(dim=0)
            field_features, _, _ = model.field.query(wave_vec, k=4)
            merged = field_features.mean(dim=0) + model.cgn(field_features.mean(dim=0))

            target = torch.tensor(list(ch.encode('utf-8')), dtype=torch.long, device=device)
            logits = model.decoder(target, wave_sequence, merged)
            pred_byte = logits.argmax(dim=-1)[0].item()

            expected = ch.encode('utf-8')[0]
            match = "✓" if pred_byte == expected else "✗"
            pred_char = chr(pred_byte) if 32 <= pred_byte < 127 else f'[{pred_byte}]'
            if pred_byte == expected:
                correct += 1
            print(f"    {match} '{ch}' (expected {expected}) → {pred_byte} ('{pred_char}')")

    print(f"\n    Byte accuracy: {correct}/{len(test_chars)} ({correct/len(test_chars):.0%})")

    # ── 2. Word Spelling Test ──
    print(f"\n{'─' * 70}")
    print(f"  📝 Word Spelling: Can it spell common words?")
    print(f"{'─' * 70}")

    words = get_spelling_test_words(20)
    word_correct = 0
    with torch.no_grad():
        for word in words:
            wave = model.cse.encode(word)
            wave_sequence = wave.full.to(device)
            wave_vec = wave_sequence.mean(dim=0)
            field_features, _, _ = model.field.query(wave_vec, k=4)
            merged = field_features.mean(dim=0) + model.cgn(field_features.mean(dim=0))

            target = torch.tensor(list(word.encode('utf-8')), dtype=torch.long, device=device)
            logits = model.decoder(target, wave_sequence, merged)
            pred_bytes = logits.argmax(dim=-1).tolist()

            try:
                spelled = bytes(pred_bytes).decode('utf-8', errors='replace')
            except Exception:
                spelled = '???'

            match = "✓" if spelled == word else "✗"
            if spelled == word:
                word_correct += 1
            print(f"    {match} '{word}' → '{spelled}'")

    print(f"\n    Word accuracy: {word_correct}/{len(words)} ({word_correct/len(words):.0%})")

    # ── 3. Generation from Prompts ──
    print(f"\n{'─' * 70}")
    print(f"  📝 Generation: Autoregressive text from prompts")
    print(f"{'─' * 70}")

    prompts = [
        "The ",
        "I think ",
        "The world is ",
        "She said that ",
        "In the morning ",
        "A good man ",
        "The future of ",
        "We need to ",
    ]

    with torch.no_grad():
        for prompt in prompts:
            generated = model.generate(prompt, max_length=40, temperature=0.7)
            continuation = generated[len(prompt):]
            print(f"    '{prompt}' → '{continuation}'")

    # ── 4. Comparison with curriculum data ──
    print(f"\n{'─' * 70}")
    print(f"  📝 Curriculum Data Samples (what each stage teaches)")
    print(f"{'─' * 70}")

    for stage in range(1, 6):
        data = generate_curriculum_data(stage, n_samples=3)
        print(f"\n    Stage {stage}: {STAGE_NAMES[stage]}")
        for sample in data:
            print(f"      → {sample!r}")

    model.train()
    print(f"\n{'─' * 70}")
    print(f"  ✓ Demo 1 complete")
    print(f"{'─' * 70}")


if __name__ == '__main__':
    main()
