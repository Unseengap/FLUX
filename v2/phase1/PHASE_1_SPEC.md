# Phase 1 v2 — Wave Codec (Bidirectional)

> **Roadmap:** ROADMAP_WAVE_FIRST.md  
> **Checkpoint:** `checkpoints/phase1_v2.phase.pt`  
> **Previous phase:** None (this is the foundation)

---

## Goal

Prove that text converts to waves AND waves convert back to text.

This is the atomic unit of the wave-first architecture.
**If waves aren't decodable, nothing that follows is physical.**

---

## What's New in v2

| Original Phase 1 | Phase 1 v2 |
|-----------------|------------|
| Train CSE (encoder only) | Train CSE + WaveChunker + WaveToText jointly |
| Wave space never constrained to be decodable | Wave space learns to be decodable from step 1 |
| WaveToText added in Phase 9 (8 phases too late) | WaveToText is here, trained together |
| Mode collapse discovered at Phase 9 | Mode collapse caught at step 1 of training |

---

## Components

### ContinuousSemanticEncoder (`cse.py`)
- Input: raw UTF-8 string
- Output: `SemanticWave` — `[seq_len, 432]`
- Byte sliding window → conv bank → per-dimension projection → wave interference
- No tokenizer. No vocabulary. Any bytes accepted.

### WaveChunker (`wave_chunker.py`)
- Input: `[seq_len, 432]` CSE wave output
- Output: `[N, 432]` word-level chunks + `[(start, end), ...]` byte spans
- Uses wave coherence drop to detect word boundaries
- Learned compression: variable-length byte waves → single 432-dim wave

### WaveToText (`wave_to_text.py`)
- Input: `[432]` chunk wave
- Output: `bytes` (decoded byte span)
- Tiny 1-layer GRU: only needs to spell one word, not generate prose
- Autoregressive byte generation with EOS detection

### DecodeGate (`decode_gate.py`)
- The non-negotiable invariant shared across all v2 phases
- Must pass at the end of every phase before moving on
- Tests: 8 standard texts across English, UTF-8, code, math

---

## Training Strategy

```python
# Joint training — encoder and decoder learn TOGETHER
# This is THE key difference from the original roadmap
for step in range(max_steps):
    text = sample_text()
    
    # Encode
    wave = cse.encode(text)                      # [seq, 432]
    pairs = chunker.chunk_with_bytes(wave.full, text.encode('utf-8'))
    
    # Decode — TRAINED FROM STEP 1
    chunk_waves = stack([pair[0] for pair in pairs])
    target_list  = [pair[1] as tensor for pair in pairs]
    
    decode_loss = wtt.forward_batch(chunk_waves, target_list)
    coherence_loss = coherence_penalty(wave)
    
    total_loss = decode_loss + 0.1 * coherence_loss
    total_loss.backward()
    optimizer.step()
```

---

## Wave Dimensions

| Dimension | Size | Meaning |
|-----------|------|---------|
| phonetic  |  64  | Character/sound patterns |
| syntactic |  64  | Grammatical structure |
| semantic  | 256  | Core meaning coordinates |
| temporal  |  32  | Sequential position |
| intensity |  16  | Emphasis and importance |
| **Total** | **432** | |

---

## Acceptance Criteria

- [ ] Any UTF-8 string encodes to waves without errors
- [ ] Waves decode back to text with > 95% byte accuracy (avg)
- [ ] Round-trip works for English, multi-byte UTF-8, code, math
- [ ] Similar words produce similar waves (cosine > 0.65 in at least 60% of pairs)
- [ ] Opposite/unrelated words produce dissimilar waves
- [ ] Chunker segments waves into 2–20 byte spans
- [ ] WaveToText decodes each chunk correctly
- [ ] Training converges in < 30K steps
- [ ] All three tests pass
- [ ] Both demos run in < 60 seconds

---

## Decode Gate (Mandatory)

```python
DECODE_GATE_TEXTS = [
    "The future of artificial intelligence",
    "Energy equals mass times the speed of light squared",
    "Photosynthesis converts sunlight into chemical energy",
    "Water freezes at zero degrees Celsius",
    "The cat sat on the mat",
    "café naïve résumé",
    "def hello(): return 'world'",
    "∫₀^∞ e^(-x²) dx = √π/2",
]
# Must achieve avg_accuracy > 90%, min_accuracy > 70%
```

---

## Checkpoint Format

```python
{
    'phase': 1,
    'version': 'v2',
    'timestamp': str,         # ISO format
    'step': int,
    'is_final': bool,
    'config': {
        'wave_dim': 432,
        'chunker_min': 2,
        'chunker_max': 20,
        'wtt_hidden_dim': 256,
        'wtt_max_bytes': 20,
    },
    'state_dict': {
        'cse': OrderedDict,
        'chunker': OrderedDict,
        'wtt': OrderedDict,
    },
    'metrics': {
        'best_decode_loss': float,
    },
}
```

---

## File Structure

```
v2/phase1/
├── __init__.py
├── wave_types.py            ← SemanticWave dataclass + WAVE_DIMS constants
├── interference.py          ← Wave interference functions
├── cse.py                   ← ContinuousSemanticEncoder
├── wave_chunker.py          ← WaveChunker
├── wave_to_text.py          ← WaveToText (GRU decoder)
├── decode_gate.py           ← DecodeGate invariant (ALL v2 phases use this)
├── train_codec.py           ← Joint training script
├── demo_phase1_demo1.py     ← Round-trip demo
├── demo_phase1_demo2.py     ← Interference patterns demo
├── test_phase1_test1.py     ← Round-trip byte accuracy > 95%
├── test_phase1_test2.py     ← Language-agnostic encode+decode
├── test_phase1_test3.py     ← Similar words → similar waves
└── PHASE_1_SPEC.md          ← This file
```

---

## Running

```bash
# Train
python v2/phase1/train_codec.py --steps 30000

# Test (requires trained checkpoint)
python v2/phase1/test_phase1_test1.py
python v2/phase1/test_phase1_test2.py
python v2/phase1/test_phase1_test3.py

# Demo
python v2/phase1/demo_phase1_demo1.py
python v2/phase1/demo_phase1_demo2.py
```
