"""
Phase 8.5: Curriculum Data — Synthetic Training Data for ABC School

Generates progressively harder training material:
  Stage 1: Individual printable ASCII bytes (learn byte distribution)
  Stage 2: Common bigrams and trigrams (learn byte co-occurrence)
  Stage 3: Top-1000 English words (learn word spelling)
  Stage 4: Common phrases / collocations (learn word combinations)
  Stage 5: Simple sentences (learn grammar and structure)
  Stage 6: Real OpenWebText (handled externally)

Each stage returns a list of training strings. The curriculum trainer
feeds these through the WaveDecoder with teacher forcing.
"""

import random
from typing import List, Dict, Tuple


# ─────────────────────────────────────────────
# Stage 1: Printable ASCII Bytes
# ─────────────────────────────────────────────

# Printable ASCII range: 32 (space) to 126 (~)
PRINTABLE_ASCII = [chr(i) for i in range(32, 127)]

# English byte frequency (approximate, from large corpora)
# Higher weight = more training samples
BYTE_FREQUENCIES = {
    ' ': 18.0, 'e': 12.7, 't': 9.1, 'a': 8.2, 'o': 7.5, 'i': 7.0,
    'n': 6.7, 's': 6.3, 'h': 6.1, 'r': 6.0, 'd': 4.3, 'l': 4.0,
    'c': 2.8, 'u': 2.8, 'm': 2.4, 'w': 2.4, 'f': 2.2, 'g': 2.0,
    'y': 2.0, 'p': 1.9, 'b': 1.5, 'v': 1.0, 'k': 0.8, 'j': 0.2,
    'x': 0.2, 'q': 0.1, 'z': 0.1,
    '.': 1.3, ',': 1.0, "'": 0.5, '"': 0.3, '!': 0.1, '?': 0.2,
    '-': 0.2, ':': 0.1, ';': 0.1, '(': 0.1, ')': 0.1,
    '0': 0.3, '1': 0.3, '2': 0.2, '3': 0.2, '4': 0.2,
    '5': 0.2, '6': 0.2, '7': 0.2, '8': 0.2, '9': 0.2,
}


def generate_stage1(n_samples: int = 500, seed: int = 42) -> List[str]:
    """
    Stage 1: Single bytes and short byte sequences.

    Teaches the decoder to produce individual printable ASCII characters
    with correct frequency distribution.

    Args:
        n_samples: Number of training samples
        seed: Random seed

    Returns:
        List of short strings (1-5 characters each)
    """
    rng = random.Random(seed)
    texts = []

    # Build weighted character pool
    pool = []
    for ch in PRINTABLE_ASCII:
        weight = BYTE_FREQUENCIES.get(ch.lower(), 0.1)
        pool.extend([ch] * int(weight * 10))

    for _ in range(n_samples):
        length = rng.randint(1, 5)
        text = ''.join(rng.choice(pool) for _ in range(length))
        texts.append(text)

    return texts


# ─────────────────────────────────────────────
# Stage 2: Common Bigrams and Trigrams
# ─────────────────────────────────────────────

# Most common English bigrams (by frequency)
COMMON_BIGRAMS = [
    'th', 'he', 'in', 'er', 'an', 'on', 'en', 'at', 'es', 'ed',
    'or', 'ti', 'is', 'it', 'al', 'ar', 'st', 'to', 'nt', 'ng',
    'se', 'ha', 'as', 'ou', 're', 'of', 'le', 'de', 'hi', 'ri',
    'ro', 'ic', 'ne', 'me', 'nd', 'te', 'co', 'li', 'ra', 'io',
    'be', 'ce', 'om', 'pe', 've', 'ma', 'el', 'ta', 'ec', 'si',
]

# Most common English trigrams
COMMON_TRIGRAMS = [
    'the', 'and', 'ing', 'ion', 'tio', 'ent', 'ati', 'for', 'her',
    'ter', 'hat', 'tha', 'ere', 'ate', 'his', 'con', 'res', 'ver',
    'all', 'ons', 'nce', 'men', 'ith', 'ted', 'ers', 'pro', 'thi',
    'wit', 'are', 'ess', 'not', 'ive', 'was', 'ect', 'rea', 'com',
    'eve', 'per', 'int', 'est', 'sta', 'cti', 'ica', 'ist', 'ear',
    'ain', 'one', 'our', 'iti', 'rat', 'ous',
]


def generate_stage2(n_samples: int = 1000, seed: int = 42) -> List[str]:
    """
    Stage 2: Bigram and trigram sequences.

    Teaches common 2-3 byte patterns that appear in English.
    Strings are formed by concatenating bigrams/trigrams with spaces.

    Args:
        n_samples: Number of training samples
        seed: Random seed

    Returns:
        List of strings (5-20 characters each)
    """
    rng = random.Random(seed)
    texts = []

    all_ngrams = COMMON_BIGRAMS + COMMON_TRIGRAMS

    for _ in range(n_samples):
        # Mix of approaches: raw ngrams, spaced ngrams, repeated patterns
        approach = rng.randint(0, 2)

        if approach == 0:
            # Concatenated ngrams (no spaces)
            n = rng.randint(2, 6)
            text = ''.join(rng.choice(all_ngrams) for _ in range(n))
        elif approach == 1:
            # Spaced ngrams
            n = rng.randint(2, 5)
            text = ' '.join(rng.choice(all_ngrams) for _ in range(n))
        else:
            # Repeated pattern (reinforcement)
            ngram = rng.choice(all_ngrams)
            n = rng.randint(2, 5)
            text = ' '.join([ngram] * n)

        texts.append(text)

    return texts


# ─────────────────────────────────────────────
# Stage 3: Common English Words
# ─────────────────────────────────────────────

# Top 200 most common English words (by frequency in text corpora)
COMMON_WORDS = [
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
    'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
    'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
    'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
    'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other',
    'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
    'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
    'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us',
    'great', 'between', 'need', 'large', 'must', 'big', 'high', 'such', 'long', 'here',
    'thing', 'many', 'still', 'find', 'hand', 'old', 'life', 'tell', 'write', 'become',
    'much', 'keep', 'help', 'call', 'through', 'being', 'last', 'turn', 'same', 'right',
    'try', 'leave', 'part', 'real', 'own', 'should', 'world', 'never', 'start', 'three',
    'show', 'every', 'change', 'place', 'where', 'again', 'off', 'always', 'move', 'number',
    'might', 'open', 'point', 'form', 'small', 'end', 'house', 'group', 'under', 'few',
    'seem', 'run', 'state', 'head', 'each', 'both', 'man', 'child', 'next', 'name',
    'side', 'water', 'question', 'power', 'school', 'system', 'program', 'city', 'follow', 'line',
    'set', 'study', 'feel', 'home', 'night', 'story', 'young', 'read', 'second', 'order',
    'eye', 'best', 'game', 'kind', 'body', 'family', 'face', 'done', 'since', 'early',
]

# Extended word list for variety (next 800 words)
EXTENDED_WORDS = [
    'while', 'case', 'idea', 'fact', 'area', 'down', 'against', 'problem',
    'different', 'ask', 'important', 'country', 'general', 'already', 'stand', 'before',
    'during', 'without', 'possible', 'another', 'enough', 'begin', 'something',
    'example', 'develop', 'include', 'believe', 'however', 'provide', 'certain',
    'member', 'maybe', 'moment', 'result', 'reason', 'interest', 'level',
    'continue', 'bring', 'company', 'history', 'whether', 'together', 'social',
    'often', 'children', 'public', 'almost', 'able', 'present', 'experience',
    'market', 'question', 'action', 'building', 'political', 'health', 'common',
    'morning', 'chance', 'research', 'money', 'several', 'business', 'police',
    'community', 'local', 'language', 'national', 'report', 'decision', 'event',
    'music', 'information', 'nothing', 'behind', 'air', 'far', 'across',
    'toward', 'close', 'special', 'paper', 'human', 'service', 'table',
    'office', 'simple', 'street', 'matter', 'dark', 'field', 'inside',
    'today', 'hold', 'west', 'north', 'south', 'east', 'hundred',
    'along', 'though', 'listen', 'window', 'teacher', 'near', 'land',
    'letter', 'mother', 'father', 'answer', 'learn', 'plant', 'food',
    'cover', 'door', 'complete', 'record', 'ground', 'picture', 'force',
    'plan', 'rock', 'room', 'animal', 'color', 'fish', 'bird',
    'river', 'tree', 'earth', 'fire', 'thought', 'car', 'book',
    'light', 'above', 'often', 'below', 'around', 'sun', 'sea',
    'draw', 'left', 'late', 'watch', 'hard', 'list', 'stop',
    'talk', 'mountain', 'walk', 'play', 'sing', 'song', 'war',
    'love', 'minute', 'clear', 'true', 'white', 'black', 'four',
    'half', 'road', 'class', 'rest', 'sure', 'step', 'fast',
    'deep', 'nothing', 'front', 'full', 'gone', 'measure', 'free',
]

ALL_WORDS = COMMON_WORDS + EXTENDED_WORDS


def generate_stage3(n_samples: int = 2000, seed: int = 42) -> List[str]:
    """
    Stage 3: Individual words and short word sequences.

    Teaches the decoder to spell common English words correctly.

    Args:
        n_samples: Number of training samples
        seed: Random seed

    Returns:
        List of strings (single words and 2-3 word sequences)
    """
    rng = random.Random(seed)
    texts = []

    for _ in range(n_samples):
        approach = rng.randint(0, 3)

        if approach == 0:
            # Single word
            texts.append(rng.choice(ALL_WORDS))
        elif approach == 1:
            # Two words
            texts.append(rng.choice(ALL_WORDS) + ' ' + rng.choice(ALL_WORDS))
        elif approach == 2:
            # Three words
            words = [rng.choice(ALL_WORDS) for _ in range(3)]
            texts.append(' '.join(words))
        else:
            # Word repeated (reinforcement)
            word = rng.choice(COMMON_WORDS[:100])
            texts.append(' '.join([word] * rng.randint(2, 4)))

    return texts


# ─────────────────────────────────────────────
# Stage 4: Common Phrases and Collocations
# ─────────────────────────────────────────────

COMMON_PHRASES = [
    # Determiners + nouns
    'the man', 'the woman', 'the child', 'the house', 'the world',
    'the time', 'the people', 'the water', 'the way', 'the day',
    'a man', 'a woman', 'a good', 'a great', 'a new',
    'an old', 'an important', 'an example',
    # Prepositions
    'in the', 'of the', 'to the', 'on the', 'at the',
    'for the', 'with the', 'from the', 'by the', 'into the',
    'in a', 'of a', 'to a', 'on a', 'at a',
    # Pronouns + verbs
    'it is', 'it was', 'he is', 'he was', 'she is', 'she was',
    'they are', 'they were', 'we are', 'we were', 'I am', 'I was',
    'there is', 'there are', 'there was', 'there were',
    'he said', 'she said', 'they said',
    'I think', 'I know', 'I want', 'I need', 'I have',
    # Adjective + noun
    'good morning', 'good night', 'last night', 'next time',
    'first time', 'long time', 'old man', 'young woman',
    'new world', 'real world', 'great work', 'hard work',
    'high school', 'small town', 'big city', 'good people',
    # Verb phrases
    'want to', 'have to', 'need to', 'going to', 'able to',
    'used to', 'try to', 'start to', 'begin to', 'seem to',
    'come from', 'look at', 'go to', 'get to', 'come to',
    'turn around', 'find out', 'take care', 'make sure', 'look like',
    # Multi-word
    'a lot of', 'one of the', 'some of the', 'part of the',
    'in front of', 'on top of', 'out of the', 'at the end',
    'as well as', 'such as the', 'more than the',
    'the end of the', 'the rest of the', 'the beginning of',
]


def generate_stage4(n_samples: int = 3000, seed: int = 42) -> List[str]:
    """
    Stage 4: Common phrases and collocations.

    Teaches the decoder to produce natural multi-word sequences.

    Args:
        n_samples: Number of training samples
        seed: Random seed

    Returns:
        List of phrase strings (2-6 words each)
    """
    rng = random.Random(seed)
    texts = []

    for _ in range(n_samples):
        approach = rng.randint(0, 3)

        if approach == 0:
            # Single phrase
            texts.append(rng.choice(COMMON_PHRASES))
        elif approach == 1:
            # Two phrases joined
            p1 = rng.choice(COMMON_PHRASES)
            p2 = rng.choice(COMMON_PHRASES)
            texts.append(p1 + ' ' + p2)
        elif approach == 2:
            # Phrase + common word
            phrase = rng.choice(COMMON_PHRASES)
            word = rng.choice(COMMON_WORDS[:50])
            if rng.random() < 0.5:
                texts.append(word + ' ' + phrase)
            else:
                texts.append(phrase + ' ' + word)
        else:
            # Article + adjective + noun pattern
            articles = ['the', 'a', 'an', 'this', 'that', 'my', 'your', 'his', 'her']
            adjectives = ['good', 'great', 'new', 'old', 'big', 'small', 'long',
                          'first', 'last', 'real', 'important', 'simple', 'different']
            nouns = ['man', 'woman', 'child', 'world', 'time', 'day', 'way',
                     'people', 'house', 'water', 'hand', 'life', 'story', 'place',
                     'city', 'school', 'family', 'work', 'system', 'room']
            text = f"{rng.choice(articles)} {rng.choice(adjectives)} {rng.choice(nouns)}"
            texts.append(text)

    return texts


# ─────────────────────────────────────────────
# Stage 5: Simple Sentences
# ─────────────────────────────────────────────

# Sentence templates with slots
SENTENCE_TEMPLATES = [
    # SVO patterns
    "The {noun} {verb} the {noun2}.",
    "A {adj} {noun} {verb} the {adj2} {noun2}.",
    "{name} {verb} the {noun}.",
    "The {adj} {noun} is {adj2}.",
    "It was a {adj} {noun}.",
    "There is a {noun} in the {place}.",
    "The {noun} and the {noun2} are {adj}.",

    # More complex
    "{name} said that the {noun} was {adj}.",
    "I think the {noun} is very {adj}.",
    "We need to {verb_inf} the {noun}.",
    "They want to {verb_inf} a {adj} {noun}.",
    "She can {verb_inf} the {noun} well.",
    "He has a {adj} {noun} at {place}.",
    "The {noun} will {verb_inf} the {noun2} soon.",

    # Questions
    "What is the {noun}?",
    "Where is the {adj} {noun}?",
    "How does the {noun} {verb_inf}?",
    "Why is the {noun} so {adj}?",

    # Compound
    "The {noun} is {adj}, but the {noun2} is {adj2}.",
    "I like the {noun} because it is {adj}.",
    "The {adj} {noun} {verb} and the {noun2} {verb2}.",
]

TEMPLATE_NOUNS = [
    'man', 'woman', 'child', 'dog', 'cat', 'house', 'car', 'book',
    'tree', 'water', 'sun', 'moon', 'bird', 'fish', 'rock', 'door',
    'road', 'city', 'school', 'story', 'world', 'hand', 'eye', 'head',
    'room', 'table', 'light', 'fire', 'food', 'game', 'song', 'river',
    'mountain', 'system', 'program', 'letter', 'question', 'answer',
]

TEMPLATE_VERBS = [
    'sees', 'finds', 'takes', 'makes', 'gives', 'tells', 'shows',
    'knows', 'helps', 'uses', 'follows', 'opens', 'closes', 'moves',
    'starts', 'turns', 'leaves', 'reads', 'writes', 'draws', 'plays',
]

TEMPLATE_VERBS_INF = [
    'see', 'find', 'take', 'make', 'give', 'tell', 'show', 'know',
    'help', 'use', 'follow', 'open', 'close', 'move', 'start', 'turn',
    'leave', 'read', 'write', 'draw', 'play', 'build', 'learn', 'change',
]

TEMPLATE_ADJS = [
    'good', 'great', 'new', 'old', 'big', 'small', 'long', 'short',
    'high', 'low', 'fast', 'slow', 'dark', 'bright', 'warm', 'cold',
    'hard', 'soft', 'strong', 'young', 'clean', 'clear', 'simple',
    'important', 'beautiful', 'different', 'special', 'real',
]

TEMPLATE_NAMES = [
    'John', 'Mary', 'David', 'Sarah', 'James', 'Anna', 'Michael',
    'Emma', 'Robert', 'Lisa', 'Thomas', 'Jane', 'William', 'Helen',
]

TEMPLATE_PLACES = [
    'home', 'school', 'the park', 'the city', 'the garden',
    'work', 'the store', 'the river', 'the mountain', 'the office',
]


def generate_stage5(n_samples: int = 3000, seed: int = 42) -> List[str]:
    """
    Stage 5: Simple grammatically correct sentences.

    Teaches the decoder to produce coherent sentences with proper
    punctuation and capitalization.

    Args:
        n_samples: Number of training samples
        seed: Random seed

    Returns:
        List of sentence strings
    """
    rng = random.Random(seed)
    texts = []

    for _ in range(n_samples):
        template = rng.choice(SENTENCE_TEMPLATES)

        text = template.format(
            noun=rng.choice(TEMPLATE_NOUNS),
            noun2=rng.choice(TEMPLATE_NOUNS),
            verb=rng.choice(TEMPLATE_VERBS),
            verb2=rng.choice(TEMPLATE_VERBS),
            verb_inf=rng.choice(TEMPLATE_VERBS_INF),
            adj=rng.choice(TEMPLATE_ADJS),
            adj2=rng.choice(TEMPLATE_ADJS),
            name=rng.choice(TEMPLATE_NAMES),
            place=rng.choice(TEMPLATE_PLACES),
        )
        texts.append(text)

    return texts


# ─────────────────────────────────────────────
# Master generator
# ─────────────────────────────────────────────

STAGE_GENERATORS = {
    1: generate_stage1,
    2: generate_stage2,
    3: generate_stage3,
    4: generate_stage4,
    5: generate_stage5,
}

STAGE_NAMES = {
    1: 'Bytes (ASCII)',
    2: 'Bigrams & Trigrams',
    3: 'Common Words',
    4: 'Phrases & Collocations',
    5: 'Simple Sentences',
    6: 'Real Text (OpenWebText)',
}

STAGE_DEFAULTS = {
    1: {'n_samples': 500},
    2: {'n_samples': 1000},
    3: {'n_samples': 2000},
    4: {'n_samples': 3000},
    5: {'n_samples': 3000},
}


def generate_curriculum_data(stage: int, n_samples: int = None,
                             seed: int = 42) -> List[str]:
    """
    Generate training data for a curriculum stage.

    Args:
        stage: Stage number (1-5). Stage 6 uses real data.
        n_samples: Override default sample count
        seed: Random seed

    Returns:
        List of training strings

    Raises:
        ValueError: If stage is out of range
    """
    if stage < 1 or stage > 5:
        raise ValueError(f"Stage must be 1-5 (stage 6 uses real data). Got: {stage}")

    generator = STAGE_GENERATORS[stage]
    n = n_samples or STAGE_DEFAULTS[stage]['n_samples']

    return generator(n_samples=n, seed=seed)


# ─────────────────────────────────────────────
# Evaluation data generators
# ─────────────────────────────────────────────

def get_spelling_test_words(n: int = 100) -> List[str]:
    """Get top-N words for spelling accuracy evaluation."""
    return COMMON_WORDS[:n]


def get_phrase_test_prompts() -> List[Tuple[str, str]]:
    """
    Get prompt→expected pairs for phrase completion testing.

    Returns:
        List of (prompt, expected_continuation) tuples
    """
    return [
        ('the ', 'man'),
        ('in the ', 'world'),
        ('it is ', 'a'),
        ('I think ', 'the'),
        ('a good ', 'time'),
        ('he said ', 'that'),
        ('one of the ', 'people'),
        ('at the ', 'end'),
        ('she was ', 'a'),
        ('they are ', 'the'),
        ('in a ', 'new'),
        ('of the ', 'world'),
        ('want to ', 'make'),
        ('there is ', 'a'),
        ('I have ', 'a'),
        ('the old ', 'man'),
        ('a new ', 'world'),
        ('we need ', 'to'),
        ('this is ', 'the'),
        ('good ', 'morning'),
    ]


def get_sentence_test_prompts() -> List[str]:
    """Get prompts for sentence coherence testing."""
    return [
        "The man",
        "She said",
        "I think the",
        "There is a",
        "The old house",
        "We need to",
        "It was a",
        "They want to",
        "The world is",
        "He has a",
    ]


# ─────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  Curriculum Data — Stage Preview")
    print("=" * 60)

    for stage in range(1, 6):
        data = generate_curriculum_data(stage, n_samples=5)
        print(f"\n  Stage {stage}: {STAGE_NAMES[stage]}")
        for sample in data:
            print(f"    → {sample!r}")

    print(f"\n  Spelling test words: {get_spelling_test_words(10)}")
    print(f"\n  Phrase prompts: {get_phrase_test_prompts()[:3]}")
    print(f"\n  Sentence prompts: {get_sentence_test_prompts()[:3]}")
    print("\n  ✓ Curriculum data OK")
