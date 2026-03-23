"""
Phase 9 — Test 2: WaveToText Reconstructs Known Words

Setup: Take 500 known words, encode via CSE, chunk, decode via WaveToText.
Verify:
- Word reconstruction accuracy > 80%
- Character error rate < 0.2
- Pass criterion: ≥80% of common English words correctly spelled
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
from train_wave_gen import load_phase9_modules, build_phase9_modules


# Common English words for testing
COMMON_WORDS = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
    "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
    "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "people", "into", "year", "your", "good", "some", "could",
    "them", "see", "other", "than", "then", "now", "look", "only", "come",
    "its", "over", "think", "also", "back", "after", "use", "two", "how",
    "our", "work", "first", "well", "way", "even", "new", "want", "because",
    "any", "these", "give", "day", "most", "us", "great", "world", "need",
    "house", "life", "system", "place", "while", "where", "right", "still",
    "water", "never", "old", "number", "off", "help", "line", "turn",
    "move", "live", "found", "long", "school", "ask", "small", "home",
    "hand", "high", "keep", "point", "play", "own", "same", "tell", "under",
    "last", "read", "call", "run", "change", "big", "end", "put", "try",
    "name", "head", "start", "show", "may", "might", "large", "must",
    "between", "every", "near", "add", "food", "here", "done", "being",
    "many", "far", "each", "state", "both", "few", "book", "much",
    "children", "important", "answer", "learn", "letter", "mother",
    "father", "animal", "country", "different", "together", "another",
    "example", "picture", "always", "family", "music", "story", "paper",
    "often", "earth", "thought", "city", "tree", "cross", "river",
    "science", "second", "listen", "begin", "idea", "enough", "white",
    "close", "open", "seem", "next", "hard", "light", "above", "along",
    "door", "power", "morning", "question", "simple", "several", "watch",
    "color", "word", "group", "problem", "table", "night", "room",
    "young", "human", "money", "order", "stop", "miles", "fact", "three",
    "field", "nothing", "rest", "car", "mind", "strong", "area", "study",
    "person", "result", "side", "stand", "voice", "bring", "face", "space",
    "best", "hour", "better", "early", "food", "hold", "body", "half",
    "real", "fish", "plant", "north", "south", "feet", "west", "fire",
    "part", "kind", "class", "given", "level", "plan", "four", "five",
    "six", "seven", "eight", "nine", "ten", "air", "art", "age", "base",
    "eye", "bit", "war", "rate", "free", "dark", "past", "edge", "cold",
    "heat", "note", "form", "land", "sure", "rock", "road", "deep",
    "full", "clear", "step", "able", "data", "game", "true", "test",
    "love", "care", "cost", "fall", "girl", "late", "hard", "lead",
    "mean", "meet", "miss", "pull", "role", "rule", "save", "sell",
    "sort", "type", "unit", "view", "wall", "wide", "wish", "boat",
    "deal", "drop", "fair", "grow", "join", "king", "mark", "pain",
    "pick", "race", "rise", "risk", "safe", "shot", "sign", "sing",
    "site", "skin", "soft", "soon", "team", "term", "wave", "west",
    "wind", "wire", "wood", "zero", "baby", "bird", "blue", "bone",
    "burn", "camp", "card", "cave", "chip", "coat", "cook", "copy",
    "crew", "crop", "dear", "diet", "dirt", "disk", "draw", "earn",
    "ease", "east", "fail", "farm", "fast", "fear", "feed", "feel",
    "fill", "film", "flag", "flat", "flow", "fold", "fool", "foot",
    "gain", "gate", "gift", "glad", "gold", "grab", "gray", "hall",
    "hang", "harm", "hate", "hide", "hill", "hole", "hope", "host",
    "hurt", "iron", "item", "jump", "jury", "keen", "lake", "lamp",
    "lean", "lift", "link", "loan", "lock", "lose", "luck", "mail",
    "mine", "mood", "moon", "nail", "neat", "nose", "pack", "page",
    "pair", "park", "path", "peak", "peer", "pile", "pink", "pipe",
    "plus", "poem", "poll", "pool", "poor", "port", "pour", "pray",
]


def compute_char_error_rate(reference: str, hypothesis: str) -> float:
    """Compute character error rate (edit distance / reference length)."""
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0

    # Simple Levenshtein distance
    m, n = len(reference), len(hypothesis)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n] / max(m, 1)


def main():
    print("=" * 60)
    print("  Phase 9 — Test 2: WaveToText Word Reconstruction")
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

    # Test on words
    test_words = COMMON_WORDS[:500]
    print(f"\n  Testing {len(test_words)} words...")

    correct = 0
    total = 0
    cer_scores = []
    examples_correct = []
    examples_wrong = []

    wtt.eval()
    chunker.eval()

    for word in test_words:
        try:
            # Encode the word via CSE
            with torch.no_grad():
                wave = model.cse.encode(word)
            wave_seq = wave.full.to(device)  # [seq, 432]

            # Compress to single chunk wave via chunker
            chunk_waves, spans = chunker(wave_seq)
            # Use the first chunk (the word itself)
            chunk_wave = chunk_waves[0]

            # Decode
            decoded_bytes = wtt.decode(chunk_wave, temperature=0.5)
            decoded_str = decoded_bytes.decode('utf-8', errors='replace')

            # Check
            word_bytes = word.encode('utf-8')
            is_correct = decoded_bytes == word_bytes
            cer = compute_char_error_rate(word, decoded_str)
            cer_scores.append(cer)

            if is_correct:
                correct += 1
                if len(examples_correct) < 10:
                    examples_correct.append(word)
            else:
                if len(examples_wrong) < 10:
                    examples_wrong.append((word, decoded_str))

            total += 1

        except Exception:
            total += 1
            cer_scores.append(1.0)

    # Compute metrics
    accuracy = correct / max(total, 1)
    avg_cer = sum(cer_scores) / max(len(cer_scores), 1)

    print(f"\n  ── Results ──")
    print(f"    Words tested:         {total}")
    print(f"    Exact matches:        {correct}")
    print(f"    Word accuracy:        {accuracy:.1%}")
    print(f"    Character error rate: {avg_cer:.4f}")

    if examples_correct:
        print(f"\n    Correct examples: {', '.join(examples_correct[:10])}")
    if examples_wrong:
        print(f"    Wrong examples:")
        for ref, hyp in examples_wrong[:5]:
            print(f"      '{ref}' → '{hyp}'")

    # Pass criteria
    acc_pass = accuracy >= 0.80
    cer_pass = avg_cer < 0.2

    print(f"\n    {'✓' if acc_pass else '✗'} Word accuracy ≥ 80%: {acc_pass}")
    print(f"    {'✓' if cer_pass else '✗'} CER < 0.2: {cer_pass}")

    overall_pass = acc_pass and cer_pass

    # Record results
    results.add_test(
        "Word Reconstruction Accuracy",
        passed=acc_pass,
        score=accuracy,
        threshold=0.80,
        notes=f"Exact word match: {correct}/{total}",
    )
    results.add_test(
        "Character Error Rate",
        passed=cer_pass,
        score=avg_cer,
        threshold=0.2,
        notes=f"Mean CER across {total} words",
    )
    results.save()

    print(f"\n  {'✓ ALL TESTS PASSED' if overall_pass else '✗ SOME TESTS FAILED'}")
    return overall_pass


if __name__ == '__main__':
    passed = main()
    sys.exit(0 if passed else 1)
