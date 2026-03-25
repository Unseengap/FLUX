# Phase 9.1 — Context-Aware WaveToText (Spelling Enhancement)

## Overview

Phase 9.1 is a focused enhancement of the WaveToText (WTT) decoder from Phase 9.
The WTT is the last-mile component that converts 432-dim wave chunks back into
UTF-8 byte sequences (spelling). Phase 9 achieved 82.8% chunk accuracy and 79.8%
word accuracy, but only 33% on hard multi-chunk word spelling. Errors compound
across chunks — an 8-chunk word at 83% per-chunk accuracy only has ~24% whole-word
probability.

Phase 9.1 fixes this by:
1. Adding **left-context awareness** (previous 2 chunks) to resolve boundary ambiguity
2. **Scaling training** from 25k to 150k steps with cosine LR decay
3. **Doubling hidden dim** from 256 to 512 (~1.6M params, still tiny)
4. **Explicit UTF-8 multi-byte training** for non-ASCII characters
5. **Random chunk offset augmentation** for position diversity

Everything else from Phase 9 stays frozen. Only WTT is retrained.

## What We Keep from Phase 9 (FROZEN)

| Component | Status | Source |
|-----------|--------|--------|
| FLUXModel (Ph1-7) | Frozen | phase9.phase.pt |
| WaveChunker | Frozen | phase9.phase.pt (374,112 params) |
| WaveGenerator | Frozen (mode-collapsed, ignored) | phase9.phase.pt |
| Field (75,561 attractors) | Frozen | phase9.phase.pt |
| Bridges (wave_to_field, field_to_wave) | Frozen | phase9.phase.pt |

## New Architecture: ContextWaveToText

### Changes from Phase 9 WaveToText

| Aspect | Phase 9 WTT | Phase 9.1 ContextWTT |
|--------|-------------|----------------------|
| Input | Single chunk wave [432] | Current + 2 left-context chunks [3, 432] |
| Hidden dim | 256 | 512 |
| Params | 440,705 | ~1,600,000 |
| Context fusion | None | Cross-attention over left chunks |
| GRU layers | 1 | 2 (with dropout 0.1) |
| Byte embedding | 64-dim | 128-dim |
| Training steps | 25,000 | 150,000 |
| LR schedule | Flat 3e-4 | Warmup + cosine decay |
| UTF-8 | Broken on multi-byte | Explicit multi-byte training data |

### Architecture Detail

```
Input: current_wave [432], left_context [0-2, 432]

1. Context Fusion:
   - Project each wave through shared Linear(432, 512) → [N, 512]
   - Cross-attention: current queries, left-context keys/values
   - Output: context_enriched [512]

2. GRU Decoder:
   - wave_to_hidden: Linear(512, 512) → initial hidden [2, 1, 512]
   - byte_embed: Embedding(258, 128) — 256 bytes + BOS + EOS
   - gru: GRU(128, 512, num_layers=2, dropout=0.1)
   - output_proj: Linear(512, 257) — 256 bytes + EOS

3. Decoding:
   - Teacher-forced during training (same as Phase 9)
   - Autoregressive at inference (same as Phase 9)
   - Temperature sampling with EOS stopping
```

### Why Left-Context Helps

Phase 9's WTT diagnostic showed errors like:
- "the" → "the ee" (boundary bled into next chunk)
- "and" → "an d" (split in wrong place)
- "energy" → "te engry ene" (no context about what came before)

With 2 left-context chunks, the decoder knows:
- What bytes the previous chunk ended with → better boundary handling
- What word prefix might carry over → "en" from prev chunk + "ergy" in current
- Positional context → first chunk vs mid-word chunk behave differently

## Training Plan

### Data Pipeline
1. Load phase9.phase.pt, restore FLUXModel + WaveChunker (frozen)
2. CSE-encode training texts → wave sequences
3. WaveChunker → chunk waves + byte spans
4. For each chunk: extract (current_wave, left_2_waves, target_bytes)
5. Random augmentation: occasionally drop one/both context chunks (25% each)

### Training Config
```python
PHASE9_1_CONFIG = {
    'context_waves': 2,          # Number of left-context chunks
    'hidden_dim': 512,           # GRU hidden (up from 256)
    'byte_embed_dim': 128,       # Byte embedding (up from 64)
    'gru_layers': 2,             # GRU depth (up from 1)
    'dropout': 0.1,              # GRU dropout
    'max_bytes': 20,             # Max bytes per chunk (same)
    'lr': 3e-4,                  # Peak learning rate
    'warmup_steps': 2000,        # LR warmup
    'max_steps': 150000,         # Total training steps
    'batch_size': 64,            # Batch size (up from 32)
    'grad_accum': 2,             # Gradient accumulation
    'log_interval': 5000,        # Log every N steps
    'eval_interval': 25000,      # Eval every N steps
    'context_drop_prob': 0.25,   # Probability of dropping each context chunk
}
```

### LR Schedule
- Warmup: linear 0 → 3e-4 over 2,000 steps
- Decay: cosine 3e-4 → 1e-5 over remaining 148,000 steps

### UTF-8 Augmentation
- 10% of training batches include texts with non-ASCII characters
- Source: synthetic sentences with accented characters (café, naïve, résumé, etc.)
- Also include CJK, emoji, and other multi-byte UTF-8 sequences

## Checkpoint Format

Saved as `phase9_1.phase.pt` with:
```python
{
    'phase': 9.1,
    'config': <FLUXModel config from phase9>,
    'phase9_1_config': PHASE9_1_CONFIG,
    # All Phase 1-7 frozen states (copied from phase9.phase.pt)
    'cse_state_dict': ...,
    'field_state_dict': ...,
    'gr_state': ..., 'tl_state': ..., 'cgn_state': ...,
    'causal_graph_state': ...,
    'working_memory_state': ..., 'episodic_memory_state': ...,
    'semantic_memory_state': ..., 'router_state': ...,
    'wave_to_field_state': ..., 'field_to_wave_state': ...,
    'output_head_state': ...,
    # Phase 9 frozen modules
    'wave_chunker_state_dict': ...,      # KEPT from Phase 9
    'wave_generator_state_dict': ...,    # KEPT from Phase 9 (mode-collapsed, for Phase 9.5)
    # Phase 9.1 NEW module
    'context_wtt_state_dict': ...,       # NEW trained ContextWaveToText
    'wave_to_text_state_dict': ...,      # Also saved under old key for backward compat
    'metrics': {
        'chunk_accuracy': float,
        'word_accuracy': float,
        'hard_spelling_accuracy': float,
        'utf8_accuracy': float,
        'training_steps': int,
        'final_loss': float,
    },
}
```

## Success Criteria

| Metric | Phase 9 | Phase 9.1 Target |
|--------|---------|------------------|
| Chunk accuracy | 82.8% | >93% |
| Word accuracy (easy) | 79.8% | >88% |
| Hard word spelling | 33% (5/15) | >60% (9/15) |
| UTF-8 multi-byte | 0% (café, naïve) | >50% |
| Training time | 5.6 min | ~45 min (150k steps) |
| Params | 440,705 | ~1,600,000 |

## File Structure

```
phases/phase9_1/
├── PHASE_9_1_SPEC.md           # This file
├── context_wave_to_text.py     # ContextWaveToText module
├── train_context_wtt.py        # Training script
├── test_phase9_1_test1.py      # Chunk accuracy test
├── test_phase9_1_test2.py      # Hard word spelling test
├── test_phase9_1_test3.py      # UTF-8 multi-byte test
├── demo_phase9_1_demo1.py      # Spelling comparison demo
├── demo_phase9_1_demo2.py      # Live decode visualization
└── RESULTS_PHASE_9_1.md        # Auto-generated results
```

## Dependencies on Phase 9.5

Phase 9.5 will load `phase9_1.phase.pt` instead of `phase9.phase.pt`.
The ContextWaveToText replaces WaveToText as the decoder.
Phase 9.5 only retrains WaveGenerator — ContextWTT stays frozen.
