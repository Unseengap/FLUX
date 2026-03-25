"""
test_phase3_test3.py — Valid Word Rate: Generated text contains real words

Generates text from 20 diverse prompts and checks what fraction of the
decoded output consists of real English words (not garbage bytes).

This tests that the decode loss worked during training — the generator
is producing waves that decode to real dictionary words, not noise.

Pass criteria:
    valid_word_rate > 0.50   (at least half the decoded tokens are real words)
    avg_output_length > 3    (produces substantial output, not just BOS/EOS)
    no_crash_rate == 1.0     (all 20 prompts generate without error)
"""

import sys
import re
import torch
from pathlib import Path
from typing import List, Tuple, Set

# ─────────────────────────────────────────────
# Path setup
# ─────────────────────────────────────────────
_V2_DIR       = Path(__file__).parent.parent
_PHASE1_DIR   = _V2_DIR / 'phase1'
_PHASE2_DIR   = _V2_DIR / 'phase2'
_PROJECT_ROOT = _V2_DIR.parent

for _p in [str(_PHASE1_DIR), str(_PHASE2_DIR), str(Path(__file__).parent), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cse            import ContinuousSemanticEncoder
from wave_chunker   import WaveChunker
from wave_to_text   import WaveToText
from wave_to_field  import WaveToField
from wave_generator import WaveGenerator
from flux_utils     import PhaseResults, get_device

# ─────────────────────────────────────────────
# 20 diverse test prompts
# ─────────────────────────────────────────────
TEST_PROMPTS = [
    "The cat sat on the mat",
    "Water freezes at zero degrees Celsius",
    "The capital of France is Paris",
    "Dogs are friendly animals",
    "The sky is blue during the day",
    "Python is a programming language",
    "The Earth orbits the Sun",
    "Machine learning requires data",
    "Books contain knowledge",
    "Music has rhythm and melody",
    "The ocean covers most of Earth",
    "Humans need sleep to function",
    "Light travels faster than sound",
    "Trees produce oxygen",
    "The heart pumps blood",
    "Mathematics is universal",
    "Fire requires oxygen to burn",
    "Gravity pulls objects downward",
    "Languages have grammar rules",
    "Stars are distant suns",
]

VALID_WORD_THRESHOLD  = 0.50
MIN_OUTPUT_LEN        = 3
CRASH_TOLERANCE       = 0       # zero crashes allowed


def _load_word_list() -> Set[str]:
    """
    Load a simple word list. Falls back to a built-in common word set.
    Tries: /usr/share/dict/words, then nltk, then hardcoded top-1000.
    """
    # Try system word list
    for path in ['/usr/share/dict/words', '/usr/dict/words']:
        p = Path(path)
        if p.exists():
            return {w.strip().lower() for w in p.read_text().splitlines() if len(w) > 1}

    # Try NLTK
    try:
        import nltk
        nltk.download('words', quiet=True)
        from nltk.corpus import words as nltk_words
        return {w.lower() for w in nltk_words.words()}
    except Exception:
        pass

    # Hardcoded common words fallback
    common = (
        "the a an and or but in on at to for of with by from is are was were "
        "be been being have has had do does did will would shall should may might "
        "can could not no yes all any each every some this that these those it "
        "he she we they you i me him her us them my your his her our their its "
        "what which who how when where why which than then so as if up out over "
        "after before into through about between against during without within "
        "cat dog fox mat sun moon star sky tree water fire earth air wind light "
        "time day night year man woman child life world way hand eye head word "
        "place thing work house school city road car people human brain mind "
        "good bad big small long short old new high low hot cold fast slow "
        "true false first last next know think say go come see get take make "
        "give look use find tell ask seem feel try leave call keep run turn "
        "music book art science math language code data model neural network "
        "machine learning intelligence system energy matter physics chemistry "
        "biology memory compute function class type return import def pass "
        "python javascript java rust swift program algorithm process result "
        "capital france paris water freezes celsius degrees equal mass speed "
        "light squared photosynthesis converts sunlight chemical future "
    )
    return {w.strip().lower() for w in common.split() if w.strip()}


WORD_LIST = _load_word_list()


def is_real_word(token: str) -> bool:
    """Check if a string token is a real dictionary word."""
    t = re.sub(r"[^a-z]", "", token.lower())
    return len(t) >= 2 and t in WORD_LIST


def tokenize(text: str) -> List[str]:
    """Split text into word-like tokens."""
    return re.findall(r"[a-zA-Z]+", text)


def valid_word_rate(text: str) -> Tuple[float, int]:
    """
    Compute fraction of alphabetic tokens that are real words.

    Returns:
        (rate, total_tokens)
    """
    tokens = tokenize(text)
    if not tokens:
        return 0.0, 0
    valid  = sum(1 for t in tokens if is_real_word(t))
    return valid / len(tokens), len(tokens)


def load_components(ckpt_dir: Path, device: str):
    """Load all components from checkpoints."""
    p1  = torch.load(ckpt_dir / 'phase1_v2.phase.pt', map_location='cpu')
    cfg1 = p1['config']
    cse = ContinuousSemanticEncoder(
        wave_dim=cfg1.get('wave_dim', 432),
        window_size=cfg1.get('window_size', 8),
        stride=cfg1.get('stride', 1),
    )
    cse.load_state_dict(p1['state_dict']['cse'])
    cse.to(device).eval()

    chunker = WaveChunker(
        wave_dim=cfg1.get('wave_dim', 432),
        min_chunk=cfg1.get('min_chunk', 2),
        max_chunk=cfg1.get('max_chunk', 20),
    )
    chunker.load_state_dict(p1['state_dict']['chunker'])
    chunker.to(device).eval()

    wtt = WaveToText(
        wave_dim=cfg1.get('wave_dim', 432),
        hidden_dim=cfg1.get('wtt_hidden_dim', 256),
        max_bytes=cfg1.get('max_bytes', 20),
    )
    wtt.load_state_dict(p1['state_dict']['wtt'])
    wtt.to(device).eval()

    p2  = torch.load(ckpt_dir / 'phase2_v2.phase.pt', map_location='cpu')
    cfg2 = p2['config']
    w2f = WaveToField(
        wave_dim=cfg2.get('wave_dim', 432),
        field_dim=cfg2.get('field_features', 512),
    )
    w2f.load_state_dict(p2['state_dict']['wave_to_field'])
    w2f.to(device).eval()

    p3  = torch.load(ckpt_dir / 'phase3_v2.phase.pt', map_location='cpu')
    cfg3 = p3['config']
    generator = WaveGenerator(
        wave_dim=cfg3.get('wave_dim', 432),
        field_features=cfg3.get('field_features', 512),
        gru_hidden=cfg3.get('gru_hidden', 512),
        gru_layers=cfg3.get('gru_layers', 1),
        dropout=0.0,
    )
    generator.load_state_dict(p3['state_dict']['generator'])
    generator.to(device).eval()

    return cse, chunker, wtt, w2f, generator


@torch.no_grad()
def generate_text(
    prompt:    str,
    cse:       ContinuousSemanticEncoder,
    chunker:   WaveChunker,
    wtt:       WaveToText,
    w2f:       WaveToField,
    generator: WaveGenerator,
    device:    str,
    max_waves: int = 20,
) -> Tuple[str, int]:
    """
    Generate text from a prompt.

    Returns:
        (generated_text, num_waves_generated)
    """
    wave      = cse.encode(prompt)
    mean_wave = wave.full.mean(dim=0).to(device)
    ctx       = w2f(mean_wave)

    waves, confs = generator.generate(field_context=ctx, max_waves=max_waves)
    n_waves      = waves.shape[0]

    decoded_bytes = b''
    for i in range(n_waves):
        chunk_bytes = wtt.decode(waves[i])
        if chunk_bytes:
            decoded_bytes += bytes(chunk_bytes)

    return decoded_bytes.decode('utf-8', errors='replace'), n_waves


def main():
    device   = get_device()
    ckpt_dir = _PROJECT_ROOT / 'checkpoints'

    assert (ckpt_dir / 'phase3_v2.phase.pt').exists(), \
        "Phase 3 checkpoint not found. Run train_generator.py first."

    cse, chunker, wtt, w2f, generator = load_components(ckpt_dir, device)
    results = PhaseResults(phase=3, component_name="Wave Generator")

    print(f"\n── Test 3: Valid Word Rate (20 prompts) ──\n")
    print(f"  Word list size: {len(WORD_LIST):,}\n")

    all_rates    = []
    all_lengths  = []
    crash_count  = 0
    show_limit   = 10  # only print first 10 results

    for i, prompt in enumerate(TEST_PROMPTS):
        try:
            generated, n_waves = generate_text(
                prompt, cse, chunker, wtt, w2f, generator, device
            )
            rate, n_tokens = valid_word_rate(generated)
            all_rates.append(rate)
            all_lengths.append(n_tokens)

            if i < show_limit:
                print(f"  [{rate:.0%}|{n_tokens}tok] {prompt[:30]!r}")
                print(f"         → {generated[:60]!r}")
        except Exception as e:
            crash_count += 1
            print(f"  ✗ CRASH on {prompt[:30]!r}: {e}")
            all_rates.append(0.0)
            all_lengths.append(0)

    if len(TEST_PROMPTS) > show_limit:
        print(f"  ... ({len(TEST_PROMPTS) - show_limit} more not shown)")

    avg_rate   = sum(all_rates)  / len(all_rates)  if all_rates  else 0.0
    avg_length = sum(all_lengths) / len(all_lengths) if all_lengths else 0.0
    no_crashes = (crash_count == 0)

    print(f"\n  avg_valid_word_rate={avg_rate:.1%}  avg_token_len={avg_length:.1f}  crashes={crash_count}")

    results.add_test(
        f"Valid word rate > {VALID_WORD_THRESHOLD:.0%}",
        passed=(avg_rate >= VALID_WORD_THRESHOLD),
        score=avg_rate,
        threshold=VALID_WORD_THRESHOLD,
    )
    results.add_test(
        f"Avg output length > {MIN_OUTPUT_LEN} tokens",
        passed=(avg_length >= MIN_OUTPUT_LEN),
        score=avg_length,
        threshold=float(MIN_OUTPUT_LEN),
    )
    results.add_test(
        "No crashes (all 20 prompts generate)",
        passed=no_crashes,
        score=1.0 - crash_count / len(TEST_PROMPTS),
        threshold=1.0,
    )

    results.save()

    assert avg_rate   >= VALID_WORD_THRESHOLD, \
        f"FAIL valid_word_rate={avg_rate:.1%} < {VALID_WORD_THRESHOLD:.0%}"
    assert avg_length >= MIN_OUTPUT_LEN, \
        f"FAIL avg_output_length={avg_length:.1f} < {MIN_OUTPUT_LEN}"
    assert no_crashes, f"FAIL {crash_count} crashes"

    print("\n  ✓ Test 3 PASSED")


if __name__ == '__main__':
    main()
