"""
Phase 9 — Test 3: Full Pipeline Produces Valid English Words

Setup: Generate from 20 diverse prompts using the full pipeline.
Verify:
- ≥50% of output tokens are valid English words (dictionary check)
- Average word length is between 3 and 8 characters (English-like)
- Output contains spaces between words
- Pass criterion: Valid word rate ≥ 50%
"""

import sys
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

from flux_utils import get_device, PhaseResults
from train_wave_gen import load_phase9_modules, build_phase9_modules, generate_text


# ─────────────────────────────────────────────
# Simple English Word Dictionary
# ─────────────────────────────────────────────

# Top 2000 common English words (condensed set for validation)
ENGLISH_WORDS = set("""
a about above after again against all am an and any are as at be because
been before being below between both but by can could did do does doing
down during each few for from further get got had has have having he her
here hers herself him himself his how i if in into is it its itself just
know let like make me might more most much must my myself no nor not now
of off on once only or other our ours ourselves out over own part per
please put quite rather really right said same say see seem shall she
should so some still such take tell than that the their theirs them
themselves then there these they this those through to too under until up
upon us very want was we well were what when where which while who whom
why will with without would yes yet you your yours yourself yourselves
able about above across actually after again against ago ahead almost
along already also always among another answer any anything apart appear
area around ask away back bad become been before began begin behind
believe below best better big bit black body book both bring build
business call came can case change child children city close come company
could country course cut day dead deal dear decide deep did different
difficult door down draw drink during each early earth east eat eight end
enough even every example eye face fact family far feel field find first
five flower follow food foot for form found four free friend from front
full further game gave get girl give go going gone good great green grew
ground group grow had half hand happen hard has head hear heart help her
here high him hold home hope hot house how however hundred idea if
important in interest into just keep kind king knew know land large last
late later lay lead learn leave left let letter life light like line list
little live long look lost lot love low made make man many may mean men
might mile mind miss money month more morning most mother move much music
must name near need never new next night no north not note nothing now
number off often old on once one only open or order other our out over
own page paper part pass past people person picture place plan plant play
point power problem product program put question quite ran read ready
really red remember rest right river road room round run same saw say
school second see seem sentence set several she short should show side
since sit six small so some something song soon south space stand start
state still stop story street strong student study such sun sure system
take talk tell ten than that the them then there these they thing think
this those though thought three through time to together too top toward
tree turn two type under until up upon us use very voice walk want war
was watch water way we well went were what when where which while white
who whole why will with without woman won word work world would write
year you young your
""".split())


def check_valid_word(word: str) -> bool:
    """Check if a word is a valid English word."""
    clean = word.strip('.,;:!?"\'-()[]{}/*#@$%^&+=~`<>|\\').lower()
    if not clean:
        return False
    if not clean.isalpha():
        return False
    if len(clean) < 2:
        return False

    # Check against dictionary
    if clean in ENGLISH_WORDS:
        return True

    # Common suffixes heuristic
    for suffix in ['s', 'ed', 'ing', 'ly', 'er', 'est', 'tion', 'ment', 'ness', 'able', 'ible']:
        stem = clean.rstrip(suffix) if clean.endswith(suffix) else None
        if stem and stem in ENGLISH_WORDS:
            return True

    return False


def main():
    print("=" * 60)
    print("  Phase 9 — Test 3: Full Pipeline English Words")
    print("=" * 60)

    device = get_device()
    results = PhaseResults(phase=9, component_name="Wave-Level Generation")

    # Load modules
    try:
        model, chunker, generator, wtt = load_phase9_modules(device=device)
        print("  ✓ Phase 9 checkpoint loaded")
    except Exception as e:
        print(f"  ⚠ No Phase 9 checkpoint: {e}")
        print("  ℹ Using fresh FLUXLarge + fresh Phase 9 modules")
        from flux_large import FLUXLarge
        model = FLUXLarge(device=device)
        for param in model.parameters():
            param.requires_grad = False
        chunker, generator, wtt = build_phase9_modules(device=device)

    # 20 diverse prompts
    prompts = [
        "The future of artificial intelligence is",
        "In the beginning there was nothing but",
        "Scientists have discovered that the universe",
        "The relationship between energy and matter was",
        "Modern technology relies on advanced computing",
        "Once upon a time in a distant kingdom",
        "The quantum nature of reality suggests that",
        "Education is the foundation of a thriving",
        "Climate change affects global ecosystems through",
        "The history of mathematics reveals important",
        "Philosophers have long debated whether free will",
        "The human brain contains billions of neurons",
        "Water is essential for all forms of life",
        "The speed of light determines the nature of",
        "Democracy requires participation from every citizen",
        "The structure of language reflects deep patterns",
        "Innovation and creativity drive economic progress",
        "Music has the power to transform our emotions",
        "The development of writing changed civilization",
        "Gravity shapes the structure of the cosmos",
    ]

    all_valid = 0
    all_total = 0
    all_word_lengths = []
    has_spaces = 0
    generation_results = []

    generator.eval()
    wtt.eval()
    chunker.eval()

    print(f"\n  Generating from {len(prompts)} prompts...")

    for i, prompt in enumerate(prompts):
        try:
            output = generate_text(
                prompt, model, chunker, generator, wtt,
                max_waves=20, temperature=0.8, use_sampler=True,
            )

            # Extract continuation (skip prompt)
            continuation = output[len(prompt):].strip()
            words = continuation.split()

            # Check for spaces
            if len(words) > 1:
                has_spaces += 1

            # Check word validity
            valid_count = 0
            for w in words:
                clean = w.strip('.,;:!?"\'-()[]{}').lower()
                if clean and clean.isalpha():
                    all_word_lengths.append(len(clean))
                    all_total += 1
                    if check_valid_word(w):
                        valid_count += 1
                        all_valid += 1

            generation_results.append({
                'prompt': prompt,
                'output': output[:200],
                'n_words': len(words),
                'valid': valid_count,
            })

            print(f"    [{i+1:2d}] {prompt[:40]}... → {len(words)} words, {valid_count} valid")

        except Exception as e:
            print(f"    [{i+1:2d}] ✗ Failed: {e}")
            generation_results.append({
                'prompt': prompt,
                'output': f"ERROR: {e}",
                'n_words': 0,
                'valid': 0,
            })

    # Compute metrics
    valid_rate = all_valid / max(all_total, 1)
    avg_word_len = sum(all_word_lengths) / max(len(all_word_lengths), 1)
    spaces_rate = has_spaces / max(len(prompts), 1)

    print(f"\n  ── Results ──")
    print(f"    Total generated words: {all_total}")
    print(f"    Valid English words:   {all_valid}")
    print(f"    Valid word rate:       {valid_rate:.1%}")
    print(f"    Average word length:   {avg_word_len:.1f}")
    print(f"    Outputs with spaces:   {has_spaces}/{len(prompts)}")

    # Pass criteria
    valid_pass = valid_rate >= 0.50
    length_pass = 3.0 <= avg_word_len <= 8.0
    spaces_pass = spaces_rate >= 0.5

    print(f"\n    {'✓' if valid_pass else '✗'} Valid word rate ≥ 50%: {valid_pass}")
    print(f"    {'✓' if length_pass else '✗'} Avg word length 3-8: {length_pass}")
    print(f"    {'✓' if spaces_pass else '✗'} ≥50% outputs have spaces: {spaces_pass}")

    overall_pass = valid_pass and length_pass and spaces_pass

    # Show some example outputs
    print(f"\n  ── Example Outputs ──")
    for r in generation_results[:5]:
        print(f"    Prompt: {r['prompt'][:50]}...")
        print(f"    Output: {r['output'][:150]}")
        print()

    # Record results
    results.add_test(
        "Valid English Word Rate",
        passed=valid_pass,
        score=valid_rate,
        threshold=0.50,
        notes=f"Valid: {all_valid}/{all_total}",
    )
    results.add_test(
        "Average Word Length (3-8)",
        passed=length_pass,
        score=avg_word_len,
        threshold=3.0,
        notes=f"Average word length (target 3-8 chars)",
    )
    results.add_test(
        "Output Contains Spaces",
        passed=spaces_pass,
        score=spaces_rate,
        threshold=0.5,
        notes=f"{has_spaces}/{len(prompts)} outputs have spaces between words",
    )
    results.save()

    print(f"\n  {'✓ ALL TESTS PASSED' if overall_pass else '✗ SOME TESTS FAILED'}")
    return overall_pass


if __name__ == '__main__':
    passed = main()
    sys.exit(0 if passed else 1)
