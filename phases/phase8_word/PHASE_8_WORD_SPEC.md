# Phase 8-Word Specification: Word-Level Continuous Semantic Encoder
## From Character Bytes to Semantic Word Units

> Prerequisites: Phase 1 CSE (byte-level encoder) must exist.
> Copilot: Open SPECIFICATION.md + PHASE_1_SPEC.md while building.
>
> **Key Insight:** Words are the natural unit of human meaning.
> Characters are spelling artifacts. This phase upgrades FLUX to
> operate on word-level waves while remaining vocabulary-free.

---

## The Problem with Byte-Level Encoding

```
Current (Phase 1):
"The cat sat" → 11 byte positions → 11 waves

The word "cat" is spread across 3 separate wave positions [c, a, t].
Each position captures only partial meaning of the concept.
```

### Issues:

| Problem | Impact |
|---------|--------|
| **Semantic dilution** | "cat" split into c-a-t loses conceptual unity |
| **Long sequences** | 100-word sentence = ~600 byte waves |
| **Gravitational noise** | Character fragments create spurious attractors |
| **Memory inefficiency** | Episodic memory stores character shards |
| **Coherency loss** | Field resonance on fragments, not concepts |

---

## The Solution: Word-Level Wave Aggregation

```
Phase 8-Word:
"The cat sat" → 11 bytes → 11 byte waves → word boundary detection → 3 word waves
                                                                      ↓
                                                          [The] [cat] [sat]
                                                          Each wave = complete concept
```

### Benefits:

| Benefit | Mechanism |
|---------|-----------|
| **Semantic unity** | One wave = one meaning unit |
| **Shorter sequences** | 100 words = 100 waves (not ~600) |
| **Clean attractors** | Each field attractor = actual concept |
| **Efficient memory** | Store meaningful chunks |
| **Better coherency** | Resonance on whole words |

---

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │         WordLevelCSE                │
                    └─────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
            ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
            │ Byte-Level  │ │   Word      │ │   Word      │
            │    CSE      │ │ Boundary    │ │ Pooling     │
            │ (Phase 1)   │ │ Detection   │ │ Attention   │
            └─────────────┘ └─────────────┘ └─────────────┘
                    │               │               │
                    │       bytes → word indices    │
                    │               │               │
                    └───────────────┴───────────────┘
                                    │
                                    ▼
                         Word-Level SemanticWave
                         [num_words, 432]
```

### Components:

1. **Byte-Level CSE** (from Phase 1)
   - Encodes raw UTF-8 bytes to 432-dim waves
   - Captures sub-word patterns (phonetics, letter n-grams)
   - Vocabulary-free: any bytes accepted

2. **Word Boundary Detector**
   - Identifies word boundaries from byte stream
   - Uses whitespace + punctuation as delimiters
   - Groups byte indices by word membership
   - Language-agnostic (works on any UTF-8)

3. **Word Pooling Attention**
   - Aggregates byte waves within each word
   - Learned attention over byte positions
   - Each word → single 432-dim wave
   - Preserves sub-word information in the aggregation

---

## Key Design Decisions

### 1. Still Vocabulary-Free

We do NOT create a word vocabulary. Instead:
- Byte CSE encodes all characters
- Word boundaries are detected algorithmically
- Pooling is learned but content-agnostic

Any word in any language works: "café", "日本", "🚀rocket"

### 2. Word Boundary Strategy

Simple but effective:
```python
WORD_BOUNDARIES = set(' \t\n\r.,;:!?()[]{}"\'-')

def detect_word_boundaries(text: str) -> List[Tuple[int, int]]:
    """Returns list of (start_byte, end_byte) for each word."""
```

This handles:
- English: "Hello world" → ["Hello", "world"]
- CJK: Each character = word (no spaces)
- Mixed: "I love 寿司" → ["I", "love", "寿司"]

### 3. Pooling Method: Attention

```python
class WordPoolingAttention(nn.Module):
    """
    Learns which byte positions matter most for word meaning.
    
    Example: In "running"
    - 'r' and 'u' might get high attention (word start, vowel)
    - 'ing' suffix gets attention (grammatical marker)
    - Middle 'nn' might get less attention
    """
```

### 4. Variable-Length Words

Words vary in length (1 to 20+ bytes). Pooling handles this:
- Short words (1-2 bytes): Nearly all attention on those bytes
- Long words: Attention spreads, learns what matters

---

## Implementation Files

```
phases/phase8_word/
├── PHASE_8_WORD_SPEC.md          ← This file
├── word_cse.py                   ← WordLevelCSE + WordPooling
├── word_types.py                 ← WordWave dataclass
├── train_word_cse.py             ← Training script
├── demo_phase8_word_demo1.py     ← Basic encoding demo
├── demo_phase8_word_demo2.py     ← Comparison: byte vs word
├── test_phase8_word_test1.py     ← Reconstruction test
├── test_phase8_word_test2.py     ← Similarity test
├── test_phase8_word_test3.py     ← Coherency test
└── RESULTS_PHASE_8_WORD.md       ← Auto-generated results
```

---

## word_types.py

```python
"""
WordWave: Word-level semantic wave representation.
Extension of SemanticWave for word-aggregated encoding.
"""

from dataclasses import dataclass
from typing import List
import torch
from torch import Tensor

@dataclass
class WordWave:
    """
    Word-level continuous representation.
    Shape: [num_words, 432]
    
    Unlike byte-level SemanticWave:
    - Each position = complete word concept
    - Much shorter sequences
    - Direct semantic units
    """
    words: List[str]      # Original words (for debugging/display)
    waves: Tensor         # [num_words, 432] wave representations
    word_boundaries: List[Tuple[int, int]]  # Byte spans for each word
    
    @property
    def num_words(self) -> int:
        return len(self.words)
    
    @property
    def full(self) -> Tensor:
        """The wave tensor itself."""
        return self.waves
    
    def to_retrieval_vector(self) -> Tensor:
        """Mean pool all word waves for document-level retrieval."""
        return self.waves.mean(dim=0)
```

---

## word_cse.py Core Classes

### WordBoundaryDetector

```python
class WordBoundaryDetector:
    """
    Detects word boundaries in UTF-8 byte stream.
    Returns byte index spans for each word.
    
    Handles:
    - Whitespace-delimited languages (English, etc.)
    - CJK (each character = word)
    - Punctuation (separate tokens)
    - Mixed content
    """
    
    DELIMITERS = set(' \t\n\r.,;:!?()[]{}"\'-/\\@#$%^&*+=<>|~`')
    
    def detect(self, text: str) -> List[Tuple[int, int]]:
        """
        Args:
            text: Input string
        Returns:
            List of (start_byte_idx, end_byte_idx) tuples
        """
```

### WordPoolingAttention

```python
class WordPoolingAttention(nn.Module):
    """
    Learns to aggregate byte waves into word waves.
    Uses attention mechanism to weight byte contributions.
    """
    
    def __init__(self, wave_dim: int = 432):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(wave_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
    
    def forward(
        self, 
        byte_waves: Tensor,  # [seq_len, 432]
        word_spans: List[Tuple[int, int]]
    ) -> Tensor:  # [num_words, 432]
        """
        Pool byte waves into word waves using learned attention.
        """
```

### WordLevelCSE

```python
class WordLevelCSE(nn.Module):
    """
    Complete word-level encoder.
    
    Pipeline:
    1. Text → bytes (UTF-8)
    2. Bytes → byte waves (Phase 1 CSE)
    3. Detect word boundaries
    4. Pool byte waves → word waves (attention)
    5. Apply word-to-word interference
    6. Return WordWave
    """
    
    def __init__(
        self,
        byte_cse: ContinuousSemanticEncoder,
        interference_radius: int = 3,
        device: str = 'cpu',
    ):
        super().__init__()
        self.byte_cse = byte_cse
        self.boundary_detector = WordBoundaryDetector()
        self.pooling = WordPoolingAttention(wave_dim=432)
        self.interference_radius = interference_radius
        
    def encode(self, text: str) -> WordWave:
        """
        Encode text to word-level waves.
        
        Args:
            text: Any UTF-8 string
        Returns:
            WordWave with one position per word
        """
```

---

## Training Strategy

### Reconstruction Objective

Train the pooling attention to preserve byte-level information:

```
Loss = reconstruction_loss + contrastive_loss

reconstruction_loss:
    - Word wave → decode to original bytes
    - Each word wave should reconstruct its constituent bytes

contrastive_loss:
    - Similar words → similar waves
    - Different words → different waves
```

### Training Data

Use same data as Phase 1 (WikiText, OpenWebText) but:
- Segment into words during preprocessing
- Train word pooling on word reconstruction task

---

## Acceptance Criteria

| Test | Metric | Threshold |
|------|--------|-----------|
| Word reconstruction | Byte accuracy | > 80% |
| Semantic similarity | Same word pair cosine | > 0.9 |
| Different words | Different word pair cosine | < 0.3 |
| Sequence compression | Bytes/words ratio | > 4x |
| Encode speed | Words/second | > 10,000 |

---

## Integration with FLUX

Once word-level CSE is trained:

1. **Resonance Field** (Phase 2): Each attractor = word concept
2. **Gravitational Relevance** (Phase 3): Word-to-word attention in O(log n)
3. **Memory** (Phase 6): Store word-level episodes
4. **Full Model**: Replace byte CSE with word CSE throughout

The rest of FLUX remains unchanged — it just operates on
word waves instead of byte waves.

---

## Future Extensions

- **Subword modes**: Character bigrams for rare words
- **Morphological awareness**: Prefix/suffix detection
- **Language-specific boundaries**: Better CJK segmentation
- **Dynamic granularity**: Switch byte/word/phrase based on context
