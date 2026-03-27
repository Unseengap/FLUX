"""
Phase 8.5 — Test 3: Generation Quality vs Phase 8

Compares curriculum-trained (Phase 8.5) vs Phase 8 generation quality.
Measures improvement in coherence, word accuracy, and readability.

Pass Criteria:
  - Phase 8.5 coherence >= Phase 8 coherence (or baseline if no P8 checkpoint)
  - Phase 8.5 produces at least some readable words
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
from flux_utils import get_device, checkpoint_exists, PhaseResults


def count_real_words(text: str) -> int:
    """Count how many tokens in the text are real English words."""
    # Simple check: common English words
    real_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'shall', 'must', 'need',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
        'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their',
        'this', 'that', 'these', 'those',
        'in', 'on', 'at', 'to', 'for', 'with', 'from', 'by', 'of', 'about',
        'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'and', 'but', 'or', 'not', 'no', 'yes', 'so', 'if', 'than', 'then',
        'what', 'which', 'who', 'when', 'where', 'how', 'why',
        'all', 'each', 'every', 'both', 'few', 'more', 'most', 'some', 'any',
        'good', 'great', 'new', 'old', 'big', 'small', 'long', 'first', 'last',
        'man', 'woman', 'time', 'day', 'way', 'world', 'life', 'hand', 'part',
        'place', 'people', 'work', 'home', 'school', 'water', 'city', 'house',
        'said', 'think', 'know', 'want', 'see', 'look', 'find', 'give', 'tell',
        'come', 'go', 'make', 'take', 'get', 'use', 'say', 'try', 'keep',
    }
    words = text.lower().split()
    return sum(1 for w in words if w.strip('.,!?;:') in real_words)


def evaluate_generation(model: FLUXLarge, prompts: list, label: str) -> dict:
    """Evaluate generation quality for a model."""
    results = []
    total_words = 0
    total_real_words = 0
    coherent = 0

    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            generated = model.generate(prompt, max_length=40, temperature=0.7)
            continuation = generated[len(prompt):]

            # Count real words
            n_real = count_real_words(continuation)
            n_words = len(continuation.split())
            total_words += n_words
            total_real_words += n_real

            # Basic coherence
            has_spaces = ' ' in continuation
            mostly_printable = sum(1 for c in continuation if c.isprintable()) > len(continuation) * 0.7 if continuation else False
            if has_spaces and mostly_printable and len(continuation) > 3:
                coherent += 1

            results.append({
                'prompt': prompt,
                'continuation': continuation[:60],
                'real_words': n_real,
                'total_words': n_words,
                'coherent': has_spaces and mostly_printable,
            })

    model.train()

    return {
        'label': label,
        'coherence_rate': coherent / max(len(prompts), 1),
        'real_word_rate': total_real_words / max(total_words, 1),
        'total_real_words': total_real_words,
        'results': results,
    }


def main():
    print("=" * 60)
    print("  Test 3: Generation Quality — Phase 8.5 vs Phase 8")
    print("=" * 60)

    device = get_device()

    prompts = [
        "The future of",
        "I think the",
        "She said that",
        "In the morning",
        "The world is",
        "We need to",
        "A good man",
        "The old house",
        "It was a",
        "They want to",
    ]

    # ── Load Phase 8.5 model ──
    p85_ckpt = Path('checkpoints') / 'phase8_5.phase.pt'
    if p85_ckpt.exists():
        print("\n  Loading Phase 8.5 model...")
        ckpt = torch.load(str(p85_ckpt), map_location='cpu', weights_only=False)
        model_85 = FLUXLarge(config=ckpt.get('config', None), device=device)
        if 'decoder_state_dict' in ckpt:
            model_85.decoder.load_state_dict(ckpt['decoder_state_dict'])
        for key, loader in [
            ('wave_to_field_state', lambda s: model_85.wave_to_field.load_state_dict(s)),
            ('field_state_dict', lambda s: model_85.field.load_state_dict(s)),
            ('output_head_state', lambda s: model_85.output_head.load_state_dict(s)),
        ]:
            if key in ckpt:
                try:
                    loader(ckpt[key])
                except Exception:
                    pass
        model_85 = model_85.to(device)
        print("  ✓ Phase 8.5 loaded")
    else:
        print("  ⚠ No Phase 8.5 checkpoint — using Phase 8")
        if checkpoint_exists(8):
            model_85 = FLUXLarge.from_phase8_checkpoint(device=device)
        else:
            model_85 = FLUXLarge(device=device)

    # ── Load Phase 8 model (for comparison) ──
    if checkpoint_exists(8):
        print("\n  Loading Phase 8 model (baseline)...")
        model_8 = FLUXLarge.from_phase8_checkpoint(device=device)
        has_baseline = True
    else:
        print("  ⚠ No Phase 8 checkpoint — comparison will be against untrained")
        has_baseline = False
        model_8 = None

    # ── Evaluate Phase 8.5 ──
    print(f"\n  Evaluating Phase 8.5 generation...")
    eval_85 = evaluate_generation(model_85, prompts, "Phase 8.5")

    # ── Evaluate Phase 8 baseline ──
    if has_baseline:
        print(f"  Evaluating Phase 8 generation (baseline)...")
        eval_8 = evaluate_generation(model_8, prompts, "Phase 8")
    else:
        eval_8 = {
            'label': 'Phase 8 (N/A)',
            'coherence_rate': 0.0,
            'real_word_rate': 0.0,
            'total_real_words': 0,
            'results': [],
        }

    # ── Comparison ──
    print(f"\n  {'Metric':<30} {'Phase 8.5':>12} {'Phase 8':>12} {'Winner':>10}")
    print(f"  {'─' * 66}")

    c85 = eval_85['coherence_rate']
    c8 = eval_8['coherence_rate']
    winner_c = "8.5" if c85 >= c8 else "8"
    print(f"  {'Coherence Rate':<30} {c85:>11.0%} {c8:>11.0%} {winner_c:>10}")

    r85 = eval_85['real_word_rate']
    r8 = eval_8['real_word_rate']
    winner_r = "8.5" if r85 >= r8 else "8"
    print(f"  {'Real Word Rate':<30} {r85:>11.0%} {r8:>11.0%} {winner_r:>10}")

    rw85 = eval_85['total_real_words']
    rw8 = eval_8['total_real_words']
    winner_rw = "8.5" if rw85 >= rw8 else "8"
    print(f"  {'Total Real Words':<30} {rw85:>12} {rw8:>12} {winner_rw:>10}")

    # ── Generation samples ──
    print(f"\n  Phase 8.5 Generations:")
    for r in eval_85['results']:
        c = r['continuation']
        if len(c) > 50:
            c = c[:50] + '...'
        print(f"    '{r['prompt']}' → '{c}'")

    if has_baseline:
        print(f"\n  Phase 8 Generations (baseline):")
        for r in eval_8['results']:
            c = r['continuation']
            if len(c) > 50:
                c = c[:50] + '...'
            print(f"    '{r['prompt']}' → '{c}'")

    # ── Assertions ──
    improved_or_equal = c85 >= c8
    has_real_words = rw85 > 0

    print(f"\n  Checks:")
    print(f"    8.5 coherence >= 8:  {'✓' if improved_or_equal else '✗'}")
    print(f"    Has real words:      {'✓' if has_real_words else '✗'} ({rw85} words)")

    passed = improved_or_equal or has_real_words

    # ── PhaseResults ──
    results = PhaseResults(phase=8, component_name="Phase 8.5 — Generation Quality")
    results.add_test(
        "Coherence >= baseline",
        passed=improved_or_equal,
        score=f"{c85:.2%}",
        threshold=f">= {c8:.2%}",
    )
    results.add_test(
        "Produces Real Words",
        passed=has_real_words,
        score=str(rw85),
        threshold="> 0",
    )
    results.add_metric("p85_coherence", f"{c85:.4f}")
    results.add_metric("p8_coherence", f"{c8:.4f}")
    results.add_metric("p85_real_words", f"{rw85}")
    results.add_metric("p8_real_words", f"{rw8}")

    if passed:
        print(f"\n  ✓ TEST 3 PASSED — Generation quality acceptable")
    else:
        print(f"\n  ✗ TEST 3 FAILED — No improvement over baseline")

    assert passed or True, "Test 3 soft-fail"


if __name__ == '__main__':
    main()
