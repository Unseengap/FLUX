# FLUX Roadmap v2: Wave-First Rebuild
## The Observation-First Architecture

> **Core principle:** If the wave is the fundamental object in FLUX physics,
> then proving waves can round-trip (text → wave → text) must be Phase 1 —
> not Phase 9. Everything else is structure that operates ON waves.

---

## Why Rebuild Wave-First

The original roadmap built the universe before proving its atoms were observable:

```
Original order (encoder-first):
  Phase 1: CSE (encode only — no decode)
  Phase 2: Field
  Phase 3: Gravity
  Phase 4: Thermodynamics
  Phase 5: Causal Nodes
  Phase 6: Memory
  Phase 7: Integration
  Phase 8: Scale
  Phase 9: FINALLY try to decode waves → text... mode collapse

Problem: 8 phases of frozen assumptions before testing decodability.
```

The wave-first rebuild makes every subsequent phase accountable to
**round-trip fidelity** from the start:

```
Wave-first order:
  Phase 1: Wave ↔ Text (bidirectional, round-trip proven)
  Phase 2: Wave → Field ↔ Wave (field stores waves, round-trip proven)
  Phase 3: Wave Generation (next-wave prediction, decoded to text immediately)
  Phase 4: Gravitational Relevance (operates on waves, output decoded)
  Phase 5: Thermodynamic Learning (learns on waves, output decoded)
  Phase 6: Causal Geometry (reasons with waves, output decoded)
  Phase 7: Memory (stores/retrieves waves, output decoded)
  Phase 8: Full FLUX Integration
  Phase 9: Scale & Benchmark

Every phase proves: input → waves → [component] → waves → text still works.
```

---

## What We Keep

Not everything needs rewriting. Some components are solid:

| Component | Status | Reuse? |
|-----------|--------|--------|
| CSE encoder (bytes → waves) | Phase 1 proven: 99.99% reconstruction | **Keep as-is** |
| WaveChunker | Phase 9 trained: segments waves into chunks | **Keep as-is** |
| Wave interference functions | Phase 1 proven | **Keep as-is** |
| SemanticWave dataclass | Clean abstraction | **Keep as-is** |
| ResonanceField (sparse 64³) | Phase 2 trained: 75K+ attractors | **Keep, retrain bridges** |
| flux_utils.py | Infrastructure | **Keep as-is** |
| WaveToText | Phase 9.1 improved | **Keep, make Phase 1** |
| WaveGenerator | Phase 9.5 rewritten (V3) | **Keep, make Phase 3** |
| GR, TL, CGN, Memory | Phases 3-6 trained | **Keep, re-sequence** |

The code exists. The rebuild is about **re-ordering**, not re-coding.

---

## Phase Overview (Wave-First)

```
Phase 1  ──► Wave Codec: CSE + WaveChunker + WaveToText (bidirectional)
Phase 2  ──► Resonance Field (waves live in a field, round-trip preserved)
Phase 3  ──► Wave Generation (predict next wave, decode to text)
Phase 4  ──► Gravitational Relevance (O(log n) retrieval over wave field)
Phase 5  ──► Thermodynamic Learning (local energy settling on wave field)
Phase 6  ──► Causal Geometry Nodes (causal reasoning over waves)
Phase 7  ──► Three-Tier Memory (working/episodic/semantic wave stores)
Phase 8  ──► Full FLUX Integration
Phase 9  ──► Scale & GPT-2 Benchmark
```

### Key difference: Every phase gate-checks decode quality

```python
# Run at the END of every phase:
def phase_gate_check(model, test_texts: List[str]) -> bool:
    """Verify waves still decode to readable text after adding new component."""
    for text in test_texts:
        wave = model.cse.encode(text)
        chunks, spans = model.chunker(wave.full)
        decoded = model.wtt.decode_sequence(chunks)
        decoded_text = b''.join(decoded).decode('utf-8', errors='replace')
        similarity = text_similarity(text, decoded_text)
        assert similarity > 0.90, f"Round-trip degraded: {similarity:.2f}"
    return True
```

---

## Phase 1: Wave Codec (Bidirectional)

### Goal
Prove that text converts to waves AND waves convert back to text.
This is the atomic unit of FLUX — if waves aren't decodable, nothing works.

### What Gets Built
```
phases/phase1/
├── PHASE_1_SPEC.md
├── cse.py                   ← CSE: bytes → SemanticWave [seq, 432]
├── wave_types.py            ← SemanticWave dataclass
├── interference.py          ← Wave interference functions
├── wave_chunker.py          ← Segment continuous waves into chunks
├── wave_to_text.py          ← WaveToText: chunk wave [432] → bytes
├── train_codec.py           ← Joint CSE + WaveChunker + WTT training
│                              (encode, chunk, decode in one loop)
├── demo_phase1_demo1.py     ← Demo: Text → waves → text round-trip
├── demo_phase1_demo2.py     ← Demo: Interference patterns + decode
├── test_phase1_test1.py     ← Test: Round-trip byte accuracy > 95%
├── test_phase1_test2.py     ← Test: Language-agnostic encode+decode
├── test_phase1_test3.py     ← Test: Similar words → similar waves → similar decode
└── RESULTS_PHASE_1.md
```

### Acceptance Criteria
- [ ] Any UTF-8 string encodes to waves without errors
- [ ] Waves decode back to text with > 95% byte accuracy
- [ ] Round-trip works for English, Chinese, code, math, emoji
- [ ] Similar words produce similar waves (cosine > 0.7)
- [ ] Opposite words produce dissimilar waves (cosine < 0.2)
- [ ] Chunker segments waves into 2–20 byte spans
- [ ] WaveToText decodes each chunk correctly
- [ ] Training converges in < 30K steps
- [ ] All tests pass, both demos run

### Training Strategy
```python
# Joint training — encoder and decoder learn TOGETHER
# This is THE key difference from the original roadmap
for step in range(max_steps):
    text = sample_text()
    
    # Encode
    wave = cse.encode(text)                     # [seq, 432]
    chunks, byte_spans = chunker(wave.full)      # [N, 432], [N, bytes]
    
    # Decode — TRAINED FROM STEP 1
    for chunk_wave, gt_bytes in zip(chunks, byte_spans):
        logits = wtt(chunk_wave)
        loss += cross_entropy(logits, gt_bytes)
    
    # Reconstruction signal flows back through entire pipeline
    loss.backward()
    optimizer.step()
```

### Why This Order Matters
The original Phase 1 only trained CSE (encode). WaveToText wasn't added until
Phase 9 — 8 phases later. By then, the wave space had frozen into a shape that
WTT had to decode *after the fact*. Training them together means:
- The wave space learns to be **decodable** from step 1
- WTT learns to decode the *actual* wave space, not an approximation
- Round-trip errors surface immediately, not 8 phases later

---

## Phase 2: Resonance Field (Waves in a Field)

### Goal
Waves now live in a resonance field. The field stores wave patterns as
attractors. Prove that: wave → field → wave preserves decodability.

### Prerequisites
- Phase 1 codec: text ↔ waves proven ✓

### What Gets Built
```
phases/phase2/
├── PHASE_2_SPEC.md
├── field.py                 ← ResonanceField class (sparse 64³×512)
├── attractor.py             ← Attractor detection/catalog
├── field_ops.py             ← Local perturbation, settling, energy
├── wave_to_field.py         ← Projection: wave [432] → field [512]
├── field_to_wave.py         ← Inverse: field [512] → wave [432]
├── train_field.py           ← Field + projection training WITH decode check
├── demo_phase2_demo1.py     ← Demo: Wave → field → wave → TEXT round-trip
├── demo_phase2_demo2.py     ← Demo: No forgetting (old attractors survive)
├── test_phase2_test1.py     ← Test: Attractor formation
├── test_phase2_test2.py     ← Test: wave→field→wave cosine > 0.85
├── test_phase2_test3.py     ← Test: wave→field→wave→TEXT byte accuracy > 90%
└── RESULTS_PHASE_2.md
```

### Acceptance Criteria
- [ ] Field forms stable attractors from repeated wave patterns
- [ ] wave_to_field and field_to_wave trained TOGETHER (not separately)
- [ ] Round-trip cosine: wave → field → wave > 0.85
- [ ] **Decode gate:** field → wave → WTT → text still readable (> 90% byte acc)
- [ ] New attractors don't destroy old ones
- [ ] Local update only — no global field change
- [ ] All tests pass

### Critical Fix from Original
The original Phase 2 trained `field.wave_to_feature` but never trained
`wave_to_field` or `field_to_wave` on FLUXModel. Those stayed random until
Phase 7's `.detach()` prevented gradients from ever reaching them.
Wave-first Phase 2 trains ALL projections with a decode loss:

```python
# Every field training step includes a decode check
wave = cse.encode(text)
field_vec = wave_to_field(wave.mean(dim=0))     # [432] → [512]
reconstructed = field_to_wave(field_vec)         # [512] → [432]
decoded = wtt.decode(reconstructed)              # [432] → bytes

loss = mse(wave.mean(dim=0), reconstructed)      # Wave fidelity
     + cross_entropy(wtt(reconstructed), gt_bytes)  # Decode fidelity
```

---

## Phase 3: Wave Generation (Think in Waves, Speak in Text)

### Goal
Predict the NEXT wave given context, then immediately decode it to text.
This is the first phase that produces novel output.

### Prerequisites
- Phase 1 codec ✓
- Phase 2 field with decode-preserving projections ✓

### What Gets Built
```
phases/phase3/
├── PHASE_3_SPEC.md
├── wave_generator.py        ← GRU-based next-wave predictor
├── train_generator.py       ← Batched training with SS + decode loss
├── demo_phase3_demo1.py     ← Demo: Prompt → wave generation → text
├── demo_phase3_demo2.py     ← Demo: Context-dependent generation
├── test_phase3_test1.py     ← Test: Generated waves decode to real words
├── test_phase3_test2.py     ← Test: Different contexts → different outputs
├── test_phase3_test3.py     ← Test: Valid word rate > 50%
└── RESULTS_PHASE_3.md
```

### Acceptance Criteria
- [ ] Generator predicts next waves from field context
- [ ] Generated waves decode to readable text (valid word rate > 50%)
- [ ] Different prompts produce different continuations (cosine < 0.85)
- [ ] Teacher-forced cosine accuracy > 0.5
- [ ] Training speed > 100 steps/s with batch_size=128
- [ ] **Decode gate:** prompt + generated text is coherent to human reader
- [ ] All tests pass

### Why Phase 3 (Not Phase 9)
In the original roadmap, wave generation was Phase 9 — the LAST thing.
By then, the context pipeline (GR, TL, CGN, Memory) had all been designed
without considering decodability. Phase 9 discovered the context collapse
problem because no earlier phase tested whether contexts were diverse enough
to drive different generations.

In wave-first, generation is Phase 3. Every subsequent phase (GR, TL, CGN,
Memory) must prove it doesn't break generation quality.

### Training Loss
```python
# Multi-objective loss from day 1
loss = mse_loss(predicted_wave, target_wave)           # Wave fidelity
     + (1 - cosine_sim(predicted_wave, target_wave))   # Direction fidelity
     + contrastive_loss(wave_0_across_contexts)         # Context sensitivity
     + decode_loss(wtt(predicted_wave), gt_bytes)       # TEXT fidelity ← NEW
```

---

## Phase 4: Gravitational Relevance (O(log n) Wave Retrieval)

### Goal
Replace attention with gravitational search over the wave field.
Faster than O(n²) attention, and output still decodes to text.

### Prerequisites
- Phases 1–3 ✓ (encode, field, generate all decode-checked)

### What Gets Built
```
phases/phase4/
├── PHASE_4_SPEC.md
├── gravity.py               ← GravitationalRelevance class
├── mass_tracker.py          ← Evidence mass accumulation
├── spatial_index.py         ← KD-tree / FAISS spatial index
├── negative_mass.py         ← Contradiction → repulsion
├── train_gravity.py         ← GR training with decode gate
├── benchmark_attention.py   ← Speed comparison vs attention
├── demo_phase4_demo1.py     ← Demo: Speed at various seq lengths
├── demo_phase4_demo2.py     ← Demo: Query → GR → generate → text
├── test_phase4_test1.py     ← Test: O(log n) complexity verified
├── test_phase4_test2.py     ← Test: Precision@10 > 0.8
├── test_phase4_test3.py     ← Test: Generation quality preserved after GR
└── RESULTS_PHASE_4.md
```

### Acceptance Criteria
- [ ] Faster than PyTorch attention at seq_len=1024
- [ ] Precision@10 > 0.8 on retrieval
- [ ] Negative mass repels contradicted concepts
- [ ] **Decode gate:** GR context → generate → decode still produces valid text
- [ ] Generation quality ≥ Phase 3 baseline (no regression)
- [ ] All tests pass

---

## Phase 5: Thermodynamic Learning (Learn Without Backprop)

### Goal
Replace batch gradient descent with local energy settling.
Every input both produces output AND updates the field.

### Prerequisites
- Phases 1–4 ✓

### What Gets Built
```
phases/phase5/
├── PHASE_5_SPEC.md
├── thermodynamic.py         ← ThermodynamicLearner class
├── temperature.py           ← Temperature dynamics
├── energy_functions.py      ← Local energy computation
├── online_learner.py        ← Single-sample real-time learning
├── demo_phase5_demo1.py     ← Demo: Learn fact → immediately generate with it
├── demo_phase5_demo2.py     ← Demo: Temperature visualization
├── test_phase5_test1.py     ← Test: One-shot learning retention
├── test_phase5_test2.py     ← Test: No global gradient at any point
├── test_phase5_test3.py     ← Test: Generation quality after online updates
└── RESULTS_PHASE_5.md
```

### Acceptance Criteria
- [ ] Learns from single example (no batch, no epochs)
- [ ] Learned fact retrievable after 100 subsequent updates
- [ ] No global gradient computation (verified)
- [ ] **Decode gate:** After online learning, generation still decodes properly
- [ ] All tests pass

---

## Phase 6: Causal Geometry Nodes (Why, Not Just What)

### Goal
Replace neurons with manifold patches that store what they know AND
why they know it. Enable causal tracing of any generated output.

### Prerequisites
- Phases 1–5 ✓

### What Gets Built
```
phases/phase6/
├── PHASE_6_SPEC.md
├── cgn.py                   ← CausalGeometryNode class
├── manifold.py              ← Manifold patch operations
├── causal_graph.py          ← Causal arrow storage/tracing
├── multi_timescale.py       ← Fast/slow node coordination
├── demo_phase6_demo1.py     ← Demo: "Why did you say that?" trace
├── demo_phase6_demo2.py     ← Demo: Invalidate cause → conclusion changes
├── test_phase6_test1.py     ← Test: Every output has causal trace
├── test_phase6_test2.py     ← Test: Causal invalidation works
├── test_phase6_test3.py     ← Test: Generation quality with CGN context
└── RESULTS_PHASE_6.md
```

### Acceptance Criteria
- [ ] Every generated wave has a traceable causal chain
- [ ] Disproving a cause invalidates conclusions derived from it
- [ ] Multi-timescale nodes separate fast/slow patterns
- [ ] **Decode gate:** CGN-routed generation still produces valid text
- [ ] All tests pass

---

## Phase 7: Three-Tier Memory (No Forgetting, No Context Limit)

### Goal
Working memory + episodic memory + semantic memory. The model remembers
across sessions, consolidates knowledge, and NEVER forgets.

### Prerequisites
- Phases 1–6 ✓

### What Gets Built
```
phases/phase7/
├── PHASE_7_SPEC.md
├── working_memory.py        ← Rolling field window
├── episodic_memory.py       ← FAISS vector store + metadata
├── semantic_memory.py       ← Protected field core
├── memory_router.py         ← Routes between tiers
├── consolidation.py         ← Episodic → Semantic distillation
├── demo_phase7_demo1.py     ← Demo: Cross-session memory recall
├── demo_phase7_demo2.py     ← Demo: 1000-task zero-forgetting
├── test_phase7_test1.py     ← Test: One-shot episodic write/read
├── test_phase7_test2.py     ← Test: Forgetting score = 0.0
├── test_phase7_test3.py     ← Test: Memory-augmented generation quality
└── RESULTS_PHASE_7.md
```

### Acceptance Criteria
- [ ] Write fact → retrieve correctly 100 steps later
- [ ] Forgetting score = 0.0 across 10 sequential task pairs
- [ ] Consolidation promotes frequent episodic → semantic
- [ ] Memory persists across save/load cycle
- [ ] **Decode gate:** Memory-augmented generation still decodes properly
- [ ] All tests pass

### The Forgetting Test (Most Important Test in the Project)
```python
for task_A, task_B in generate_10_task_pairs():
    train_on(task_A)
    acc_before = eval_generation_quality(task_A)
    
    train_on(task_B)                           # Would destroy transformer
    acc_after = eval_generation_quality(task_A)
    
    forgetting = (acc_before - acc_after) / acc_before
    assert forgetting < 0.02  # < 2% degradation
    # Transformer baseline: 30-80% degradation
```

---

## Phase 8: Full FLUX Integration

### Goal
Combine all components into a single unified model. Run end-to-end
text generation with the complete pipeline.

### What Gets Built
```
phases/phase8/
├── PHASE_8_SPEC.md
├── flux_model.py            ← FLUXModel — unified class
├── flux_generate.py         ← End-to-end generation pipeline
├── flux_trainer.py          ← Unified training
├── demo_phase8_demo1.py     ← Demo: Complete generation pipeline
├── demo_phase8_demo2.py     ← Demo: Real-time learning during chat
├── demo_phase8_demo3.py     ← Demo: FLUX vs LSTM quality
├── test_phase8_test1.py     ← Test: All components loaded correctly
├── test_phase8_test2.py     ← Test: End-to-end generation coherence
├── test_phase8_test3.py     ← Test: Real-time learning works
└── RESULTS_PHASE_8.md
```

### Acceptance Criteria
- [ ] All phase checkpoints load into single model
- [ ] End-to-end: prompt → CSE → Field → GR → TL → CGN → Memory → Generate → Decode → text
- [ ] Generation quality ≥ small LSTM baseline
- [ ] Real-time learning: new fact → immediately usable in generation
- [ ] All tests pass

---

## Phase 9: Scale & GPT-2 Benchmark

### Goal
Scale FLUX to GPT-2 equivalent size and benchmark head-to-head.

### Benchmark Suite
```
Standard NLP:
  Penn Treebank perplexity     → Target: ≤ GPT-2 small
  WikiText-2 perplexity        → Target: ≤ GPT-2 small

FLUX-Specific (Where FLUX Should Win):
  Continual learning retention  → Target: FLUX > GPT-2 by 50%+
  One-shot fact learning        → Target: FLUX succeeds, GPT-2 fails
  Long sequence speed (16K)     → Target: FLUX > GPT-2 by 5x
  Real-time adaptation          → Target: FLUX adapts, GPT-2 cannot
  Cross-session memory          → Target: FLUX retains, GPT-2 cannot
```

---

## Migration Strategy: How to Rebuild Without Losing Work

The existing Phase 1–9 code doesn't get deleted — it gets **re-sequenced**.

### Phase mapping (old → new)

| Old Phase | Old Component | New Phase | Notes |
|-----------|-------------|-----------|-------|
| Phase 1 | CSE (encode) | Phase 1 | Keep + add WTT + Chunker |
| Phase 9 | WaveChunker | Phase 1 | Move forward — it's part of the codec |
| Phase 9/9.1 | WaveToText / ContextWTT | Phase 1 | Move forward — decode from day 1 |
| Phase 2 | ResonanceField | Phase 2 | Keep, retrain bridges with decode loss |
| Phase 9/9.5 | WaveGeneratorV3 | Phase 3 | Move forward — generation early |
| Phase 3 | GravitationalRelevance | Phase 4 | Shift back one slot |
| Phase 4 | ThermodynamicLearner | Phase 5 | Shift back one slot |
| Phase 5 | CausalGeometryNodes | Phase 6 | Shift back one slot |
| Phase 6 | Memory (3-tier) | Phase 7 | Shift back one slot |
| Phase 7 | FLUXModel integration | Phase 8 | Shift back one slot |
| Phase 8 | Scale & benchmark | Phase 9 | Shift back one slot |

### What actually changes in code
1. **Phase 1 gets WaveChunker + WaveToText** added to its training loop
2. **Phase 2 gets decode-loss** on wave_to_field/field_to_wave projections
3. **Phase 3 IS wave generation** (was Phase 9) — with decode loss included
4. **Phases 4–7** add decode gate checks to their test suites
5. **Phase 8** is the old Phase 7 integration
6. **Phase 9** is the old Phase 8 benchmark

### Estimated rewrite scope
- Phase 1: ~40% new code (add WTT + chunker training to existing CSE)
- Phase 2: ~20% new code (add decode loss to field training)
- Phase 3: ~10% new code (wave_generator_v3.py already exists)
- Phases 4–7: ~5% each (add decode gate check to tests)
- Phases 8–9: ~0% (just renumbered)
- Total: ~2–3 weeks of focused work to re-sequence

---

## The Decode Gate: The Invariant That Prevents Phase 9 Disasters

Every phase must pass this before completing:

```python
DECODE_GATE_TEXTS = [
    "The future of artificial intelligence",
    "Energy equals mass times the speed of light squared",
    "Photosynthesis converts sunlight into chemical energy",
    "Water freezes at zero degrees Celsius",
    "The cat sat on the mat",
    "café naïve résumé",            # UTF-8 multi-byte
    "def hello(): return 'world'",  # Code
    "∫₀^∞ e^(-x²) dx = √π/2",     # Math
]

def run_decode_gate(model, phase: int) -> bool:
    """Must pass at the end of every phase. Non-negotiable."""
    results = []
    for text in DECODE_GATE_TEXTS:
        wave = model.cse.encode(text)
        chunks, spans = model.chunker(wave.full)
        decoded = model.wtt.decode_sequence(chunks)
        decoded_text = b''.join(decoded).decode('utf-8', errors='replace')
        byte_acc = byte_accuracy(text, decoded_text)
        results.append(byte_acc)
        
    avg_acc = sum(results) / len(results)
    min_acc = min(results)
    
    print(f"  Decode Gate (Phase {phase}):")
    print(f"    Avg byte accuracy: {avg_acc:.1%}")
    print(f"    Min byte accuracy: {min_acc:.1%}")
    assert avg_acc > 0.90, f"Decode gate FAILED: avg {avg_acc:.1%} < 90%"
    assert min_acc > 0.70, f"Decode gate FAILED: min {min_acc:.1%} < 70%"
    print(f"    ✓ PASSED")
    return True
```

This single invariant would have caught Phase 9's failure at Phase 2 —
because the bridges (`wave_to_field` / `field_to_wave`) staying at random
init would have failed the decode gate immediately.

---

## Checkpoint Chain (Wave-First)

```
phase1.phase.pt → CSE + WaveChunker + WaveToText (full codec)
phase2.phase.pt → codec + ResonanceField + trained projections
phase3.phase.pt → all above + WaveGeneratorV3 (generation works)
phase4.phase.pt → all above + GravitationalRelevance
phase5.phase.pt → all above + ThermodynamicLearner
phase6.phase.pt → all above + CausalGeometryNodes
phase7.phase.pt → all above + Three-Tier Memory
phase8.phase.pt → Full FLUX integrated model
phase9.phase.pt → Scaled FLUX trained on OpenWebText
```

Every checkpoint can generate text. Not just Phase 8+.
From Phase 1 onward, FLUX can encode, chunk, decode, and verify.
From Phase 3 onward, FLUX can GENERATE novel text.

---

## Definition of Success

**Minimum:** Round-trip byte accuracy > 95% at Phase 1, never drops below 90%.

**Target:** Generation quality improves monotonically from Phase 3 → Phase 9,
with no mode collapse, no context collapse, no silent failures.

**Stretch:** Beat GPT-2 small on perplexity + zero catastrophic forgetting +
real-time learning + O(log n) attention — all in one model.

---

## Summary: The Physics Argument

In physics:
- **Waves** are fundamental objects
- **Fields** are where waves live
- **Forces** (gravity, thermodynamics) operate on waves in fields
- **Measurement** (observation) converts waves to classical values

The original FLUX roadmap built forces and fields first, then tried to observe
waves at the end. Wave-first puts observation (decode) alongside the wave
itself — because a wave that can't be observed isn't physical.

```
                    Original                    Wave-First
                    ────────                    ──────────
Step 1:             Create waves                Create waves + observation
Step 2:             Build field                 Put waves in field + observe
Step 3:             Add forces                  Generate waves + observe
Step 4-7:           More forces + memory        Add forces + memory + observe
Step 8:             Integrate                   Integrate (already works)
Step 9:             Try to observe → FAIL       Scale (everything decodes)
```
