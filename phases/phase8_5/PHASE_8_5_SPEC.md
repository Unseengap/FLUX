# Phase 8.5: Curriculum Learning — ABC School for FLUX

## Overview

Phase 8.5 takes the WaveDecoder (introduced in Phase 8) and trains it through a **staged curriculum** — teaching FLUX to generate text the way humans learn: alphabet → bigrams → words → phrases → sentences → paragraphs.

Phase 8 proved FLUX can **understand** text (retrieval, memory, relevance all work). Phase 8.5 teaches it to **articulate** — spelling coherent bytes through progressively harder material.

## Motivation

Phase 8 trained the WaveDecoder on raw OpenWebText. The problem: throwing 256-class byte prediction at a randomly-initialized GRU with no prior knowledge of English structure is like teaching a child calculus before they know numbers.

The curriculum approach:
1. **Stage 1 — Bytes**: Learn the 95 printable ASCII bytes and their frequencies
2. **Stage 2 — Bigrams/Trigrams**: Learn common 2-3 byte sequences (th, he, in, er, an...)
3. **Stage 3 — Words**: Learn the 1000 most common English words
4. **Stage 4 — Phrases**: Learn common 2-4 word collocations ("in the", "of the", "it is")
5. **Stage 5 — Sentences**: Learn grammatically correct simple sentences
6. **Stage 6 — Paragraphs**: Graduate to real OpenWebText text

## Architecture

No new neural components. Phase 8.5 reuses:
- **FLUXLarge** from Phase 8 (loaded from checkpoint)
- **WaveDecoder** from Phase 8 (the GRU decoder)
- **CurriculumTrainer** (new) — orchestrates 6-stage training
- **CurriculumData** (new) — generates training data for each stage

### Energy-Based Grade Advancement

FLUX's thermodynamic system provides a natural "readiness" signal. Each stage monitors:
- **Decoder loss** on that stage's material (must drop below threshold)
- **Field energy delta** (field must stabilize — knowledge is "settled")
- **Generation accuracy** on stage-specific tests (e.g., can it spell top-100 words?)

When all three criteria are met, the curriculum advances to the next stage.

### Stage Thresholds

| Stage | Material | Max Steps | Loss Threshold | Accuracy Test |
|-------|----------|-----------|----------------|---------------|
| 1 | Printable ASCII bytes | 200 | < 3.0 | 80% byte accuracy |
| 2 | Common bigrams/trigrams | 500 | < 2.5 | 70% bigram accuracy |
| 3 | Top-1000 English words | 2000 | < 2.0 | 60% word spelling |
| 4 | Common phrases | 3000 | < 1.8 | 50% phrase completion |
| 5 | Simple sentences | 3000 | < 1.5 | 40% sentence coherence |
| 6 | Real OpenWebText | remainder | < 1.2 | Qualitative check |

## Key Classes

### `CurriculumData`
Generates synthetic training data for stages 1-5. Stage 6 uses real OpenWebText.

### `CurriculumTrainer`
Wraps `OpenWebTextTrainer` from Phase 8 with stage management:
- Monitors loss per stage
- Runs advancement tests
- Tracks stage history
- Provides detailed progress output

## Checkpoint Format

```python
{
    'phase': 8.5,
    'timestamp': str,
    'config': dict,
    'curriculum_state': {
        'current_stage': int,
        'stage_history': List[Dict],
        'total_steps': int,
        'stage_losses': Dict[int, List[float]],
    },
    # All Phase 8 component states...
    'decoder_state_dict': OrderedDict,
    'metrics': dict,
}
```

## Files

| File | Purpose |
|------|---------|
| `PHASE_8_5_SPEC.md` | This spec |
| `curriculum_data.py` | Stage 1-5 synthetic data generation |
| `curriculum_trainer.py` | 6-stage curriculum orchestrator |
| `train_curriculum.py` | Main training entry point |
| `test_phase8_5_test1.py` | Word spelling accuracy test |
| `test_phase8_5_test2.py` | Sentence coherence test |
| `test_phase8_5_test3.py` | Generation quality vs Phase 8 |
| `demo_phase8_5_demo1.py` | Stage-by-stage generation showcase |
| `demo_phase8_5_demo2.py` | FLUX vs GPT-2 generation quality |
| `RESULTS_PHASE_8_5.md` | Auto-generated results |

## Acceptance Criteria

| Criterion | Target | Method |
|-----------|--------|--------|
| All 6 stages complete | Stage 6 reached | Training log |
| Word spelling accuracy | > 50% on top-100 words | Test 1 |
| Sentence coherence | > 30% readable | Test 2 |
| Generation vs Phase 8 | Measurably better | Test 3 |
| Demo shows real words | Not gibberish | Demo 1 |
| RESULTS_PHASE_8_5.md generated | File exists | Auto |
| Checkpoint saved | phase8_5.phase.pt | Cell 6 |

## Dependencies

- Phase 8 checkpoint (phase8.phase.pt) with WaveDecoder
- All Phase 8 modules (flux_large.py, wave_decoder.py, etc.)
- OpenWebText (HuggingFace datasets) for Stage 6
