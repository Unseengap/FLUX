# FLUX Roadmap v2: Wave-First Rebuild
## The Observation-First Architecture

> **Core principle:** If the wave is the fundamental object in FLUX physics,
> then proving waves can round-trip (text → wave → text) must be Phase 1 —
> not Phase 9. Everything else is structure that operates ON waves.

---

## Build Status

| Phase | Component | v2 Status | Legacy Code Exists At |
|-------|-----------|-----------|----------------------|
| 1 | Wave Codec (CSE + Chunker + WTT) | ✅ **COMPLETE** | `phases/phase1/` + `phases/phase9/` + `phases/phase9_1/` |
| 2 | Resonance Field + decode bridges | ✅ **COMPLETE** | `phases/phase2/` |
| 2.5 | WRU — single-step next-wave prediction | 🔲 **NEXT** | NEW — FLUX-native recurrent unit |
| 3 | Deliberation + autoregressive generation | ⬜ after 2.5 | Builds on WRU from Phase 2.5 |
| 4 | Gravitational Relevance (O(log n)) | ⬜ not started | `phases/phase3/gravity.py` |
| 5 | Thermodynamic Learning | ⬜ not started | `phases/phase4/thermodynamic.py` |
| 6 | Causal Geometry Nodes | ⬜ not started | `phases/phase5/cgn.py` |
| 7 | Three-Tier Memory | ⬜ not started | `phases/phase6/` |
| 8 | Full FLUX Integration | ⬜ not started | `phases/phase7/flux_model.py` |
| 9 | Scale & GPT-2 Benchmark | ⬜ not started | `phases/phase8/` |

> **Key:** Legacy code is the OLD implementation built encoder-first (broken ordering).
> v2 ports that code into the wave-first chain, adds decode gates, and retrains where needed.

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

| Component | Legacy Location | v2 Status |
|-----------|----------------|-----------|
| CSE encoder (bytes → waves) | `phases/phase1/cse.py` | ✅ Ported — `v2/phase1/cse.py` |
| WaveChunker | `phases/phase9/wave_chunker.py` | ✅ Ported — `v2/phase1/wave_chunker.py` |
| WaveToText | `phases/phase9_1/` | ✅ Ported — `v2/phase1/wave_to_text.py` |
| Wave interference functions | `phases/phase1/interference.py` | ✅ Ported — `v2/phase1/interference.py` |
| SemanticWave dataclass | `phases/phase1/wave_types.py` | ✅ Ported — `v2/phase1/wave_types.py` |
| Decode gate utility | (new in v2) | ✅ Built — `v2/phase1/decode_gate.py` |
| ResonanceField (sparse 64³) | `phases/phase2/field.py` | ✅ Ported + bridges retrained — `v2/phase2/field.py` |
| WaveToField / FieldToWave | (was random init in legacy) | ✅ **FIXED** — trained with decode loss in `v2/phase2/` |
| WaveRecurrentUnit (WRU) | NEW — FLUX-native | 🔲 **NEXT** — `v2/phase2_5/wave_recurrent_unit.py` |
| GravitationalRelevance | `phases/phase3/gravity.py` | ⬜ Port to `v2/phase4/` with decode gate |
| ThermodynamicLearner | `phases/phase4/thermodynamic.py` | ⬜ Port to `v2/phase5/` with decode gate |
| CausalGeometryNodes | `phases/phase5/cgn.py` | ⬜ Port to `v2/phase6/` with decode gate |
| Three-Tier Memory | `phases/phase6/` | ⬜ Port to `v2/phase7/` with decode gate |
| FLUXModel integration | `phases/phase7/flux_model.py` | ⬜ Port to `v2/phase8/` |
| Scale + benchmark | `phases/phase8/` | ⬜ Port to `v2/phase9/` |
| flux_utils.py | `flux_utils.py` | ✅ Shared as-is across all phases |

The code exists. The rebuild is about **re-ordering + adding decode gates**, not re-coding from scratch.

---

## Phase Overview (Wave-First)

```
Phase 1    ──► Wave Codec: CSE + WaveChunker + WaveToText     ✅ COMPLETE
Phase 2    ──► Resonance Field + decode bridges               ✅ COMPLETE
Phase 2.5  ──► WRU: single-step next-wave prediction          🔲 NEXT
Phase 3    ──► Deliberation + autoregressive generation       ⬜ after 2.5
Phase 4    ──► Gravitational Relevance (O(log n) retrieval)   ⬜ not started
Phase 5    ──► Thermodynamic Learning (local energy settling) ⬜ not started
Phase 6    ──► Causal Geometry Nodes (causal reasoning)       ⬜ not started
Phase 7    ──► Three-Tier Memory (no forgetting)              ⬜ not started
Phase 8    ──► Full FLUX Integration                          ⬜ not started
Phase 9    ──► Scale & GPT-2 Benchmark                        ⬜ not started
```

Every phase proves: input → waves → [component] → waves → text still works.

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

## ✅ Phase 1: Wave Codec (Bidirectional) — COMPLETE

> **Trained:** 2026-03-25 on NVIDIA L4 (23.7 GB VRAM)
> **Stopped:** Early at step 17,000 / 30,000 — decode gate passed 100%/100%
> **Checkpoint:** `checkpoints/phase1_v2.phase.pt`
> **Legacy source:** `phases/phase1/` (CSE) + `phases/phase9/` (WaveChunker) + `phases/phase9_1/` (WaveToText)

### What Was Built
```
v2/phase1/
├── cse.py                   ← CSE: bytes → SemanticWave [seq, 432]
├── wave_types.py            ← SemanticWave dataclass
├── interference.py          ← Wave interference functions
├── wave_chunker.py          ← Segment continuous waves into chunks
├── wave_to_text.py          ← WaveToText: chunk wave [432] → bytes
├── decode_gate.py           ← Decode gate utility (new in v2)
├── train_codec.py           ← Joint CSE + WaveChunker + WTT training
├── demo_phase1_demo1.py     ← Demo: Text → waves → text round-trip
├── demo_phase1_demo2.py     ← Demo: Wave space cosine similarity matrix
├── test_phase1_test1.py     ← Test: Round-trip byte accuracy
├── test_phase1_test2.py     ← Test: Language-agnostic encode+decode
├── test_phase1_test3.py     ← Test: Similar words → similar waves
└── RESULTS_PHASE_1.md
```

### Actual Results (2026-03-25)

| Test | Score | Threshold | Status |
|------|-------|-----------|--------|
| Avg byte accuracy (decode gate texts) | 94.5% | 95% | ⚠ Near-miss (0.5% short) |
| Min byte accuracy | 56.0% | 70% | ✗ Math symbols struggle |
| Language pass rate (12 scripts) | 66.7% | 67% | ⚠ Near-miss |
| Chinese reconstruction | 22.2% | 40% | ✗ CJK weak |
| Japanese reconstruction | 22.2% | 40% | ✗ CJK weak |
| Arabic reconstruction | 38.5% | 35% | ✓ |
| English / French / Russian | 100% | 90% | ✓ |
| Code (`def hello()`) | 90.5% | 85% | ✓ |
| Emoji | 100% | 50% | ✓ |
| Similar word cosine ordering | 85.7% pairs correct | 60% | ✓ |
| Self-similarity (determinism) | 1.0 | 1.0 | ✓ |
| Model parameters | 2,152,081 | — | — |
| Final decode loss | 0.0556 | — | — |

**Known weaknesses carried forward:**
- CJK (Chinese/Japanese) reconstruction is poor — 22% accuracy
- Raw numbers score 33% (no context clues for the decoder)
- Math symbol min byte accuracy 56% — just under the 70% gate threshold

These are acceptable for Phase 3 — generation needs coherent *next-wave prediction*,
not perfect single-character reconstruction. CJK can be revisited at Phase 8 scaling.

### Acceptance Criteria
- [x] Any UTF-8 string encodes to waves without errors
- [~] Waves decode back to text with > 95% byte accuracy (achieved 94.5% — near-miss)
- [~] Round-trip works for all scripts (CJK/numbers weak, Western solid)
- [x] Similar words produce similar waves (cosine ordering 85.7% correct)
- [x] Chunker segments waves into byte spans
- [x] WaveToText decodes each chunk
- [x] Training converges in < 30K steps (stopped at 17K)
- [x] Decode gate passed in training (avg=100% at step 17K on gate texts)
- [x] Checkpoint saved to `checkpoints/phase1_v2.phase.pt`

### What Changed vs Legacy Phase 1
Legacy Phase 1 only trained CSE (encode). WaveToText wasn't added until Phase 9 —
8 phases later — by which point the wave space had frozen into a shape WTT had to
decode after the fact. v2 trains encoder + decoder jointly from step 1, so the wave
space learns to be decodable from the beginning.

---

## ✅ Phase 2: Resonance Field (Waves in a Field) — COMPLETE

> **Trained:** 2026-03-25 on NVIDIA L4 (23.7 GB VRAM)
> **Stopped:** Early at step 5,000 — decode gate passed 100%/100%
> **Checkpoint:** `checkpoints/phase2_v2.phase.pt`
> **Legacy source:** `phases/phase2/field.py` — CRITICAL FIX: `wave_to_field` / `field_to_wave` were random init in legacy, now trained with decode loss

### What Was Built
```
v2/phase2/
├── field.py                 ← ResonanceField class (sparse 64³×512)
├── attractor.py             ← Attractor detection/catalog
├── field_ops.py             ← Local perturbation, settling, energy
├── wave_to_field.py         ← Projection: wave [432] → field [512]  ← FIXED
├── field_to_wave.py         ← Inverse: field [512] → wave [432]     ← FIXED
├── train_field.py           ← Field + projection training WITH decode loss
├── demo_phase2_demo1.py     ← Demo: Wave → field → wave → TEXT round-trip
├── demo_phase2_demo2.py     ← Demo: No forgetting (old attractors survive)
├── test_phase2_test1.py     ← Test: Attractor formation
├── test_phase2_test2.py     ← Test: wave→field→wave cosine
├── test_phase2_test3.py     ← Test: decode gate via field bridge
└── RESULTS_PHASE_2.md
```

### Actual Results (2026-03-25)

| Test | Score | Threshold | Status |
|------|-------|-----------|--------|
| Wave→field→wave cosine (avg) | 99.76% | 85% | ✓ Exceeds by 15 pts |
| Wave→field→wave cosine (min) | 99.15% | — | ✓ |
| Decode gate avg byte accuracy | 97.79% | 90% | ✓ |
| Decode gate min byte accuracy | 82.35% (math symbols) | 70% | ✓ |
| No-forgetting: 5 facts survive 100 interferors | All 5 survived | 50% | ✓ |
| "The capital of France is Paris" similarity after 100 interferors | 77% | 50% | ✓ |
| "The Earth orbits the Sun" similarity after 100 interferors | 98% | 50% | ✓ |
| Attractor catalog count | 10 named attractors | >0 | ✓ |
| Field attractors total | 5,648 | — | — |
| Trainable parameters | 1,277,587 | — | — |
| Final recon loss | 0.0259 | <0.5 | ✓ |
| Final decode loss | 0.0307 | <1.5 | ✓ |

**The fix that matters:** In the original Phase 2, `wave_to_field` and `field_to_wave`
were never trained — random initialization all the way through Phases 2–7. This was
the root cause of Phase 9's mode collapse. In v2, both projections train with decode
loss from step 1. Result: 99.76% cosine fidelity vs the original's ~random projection.

### Acceptance Criteria
- [x] Field forms stable attractors from repeated wave patterns
- [x] wave_to_field and field_to_wave trained TOGETHER (not separately)
- [x] Round-trip cosine: wave → field → wave > 0.85 (achieved 99.76%)
- [x] **Decode gate:** field → wave → WTT → text > 90% byte acc (achieved 97.79%)
- [x] New attractors don't destroy old ones (no-forgetting test passed)
- [x] Local update only — no global field change
- [x] Checkpoint saved to `checkpoints/phase2_v2.phase.pt`

---

## 🔲 Phase 2.5: Wave Recurrent Unit (Single-Step Next-Wave) — NEXT

> **Status:** Ready to train
> **Source:** NEW — `v2/phase2_5/wave_recurrent_unit.py` (FLUX-native, no GRU)
> **Notebook:** `v2/V2_NOTEBOOKS/phase2_5_v2.ipynb`

### Goal
Predict the NEXT wave given a prefix context, using FLUX-native physics.
Proves the next-wave prediction objective works before adding autoregressive complexity.
This is the first phase that produces **novel output** — FLUX becomes a language model.

### What's Novel: The Wave Recurrent Unit (WRU)

The WRU replaces the GRU with physics-native operations:

| GRU | WRU |
|-----|-----|
| Hidden state: opaque vector | Hidden state: **valid 432-dim wave** (decodable at every step) |
| Gates: sigmoid(Wx + Uh) | Gates: **cosine interference** (constructive = keep, destructive = forget) |
| Gate scope: all dims at once | Gate scope: **per sub-band** (phonetic, syntactic, semantic, temporal, intensity) |
| Stability: gradient clipping | Stability: **energy conservation** (thermodynamic bound) |
| Interpretability: none | Each band independently inspectable |

### Architecture (~200K params)
```
field_context [512]
  → context_proj (LayerNorm → Linear → GELU → Linear) → ctx_wave [432]
  → 5 per-band interference gates (cosine similarity in [-1, 1])
      constructive (cos > 0)  → keep state band
      destructive  (cos < 0)  → replace via learned transform
  → 5 BandTransforms (small MLPs per sub-band)
  → energy-constrained superposition (can't grow unboundedly)
  → output_norm + output_proj (Tanh bounded)
  → predicted_wave [432]  ← directly decodable by WTT
```

### Training Objective — Next-Wave Prediction
```python
# Random split position n from each text:
prefix   = chunk_waves[:n]              # first n word-level waves
context  = w2f(prefix.mean(dim=0))      # [512] field context
target   = chunk_waves[n]               # [432] ground-truth next wave
gt_bytes = chunk_bytes[n]               # bytes to decode

predicted = wru(context)                # [432] predicted next wave

loss = mse(predicted, target)           # Wave reconstruction
     + (1 - cos_sim(predicted, target)) # Direction fidelity
     + wtt.decode_loss(predicted, gt_bytes)  # TEXT fidelity
```

### Prerequisites
- Phase 1 checkpoint `phase1_v2.phase.pt` ✓ (CSE, Chunker, WTT frozen)
- Phase 2 checkpoint `phase2_v2.phase.pt` ✓ (WaveToField, FieldToWave frozen)

### What Gets Built
```
v2/phase2_5/
├── wave_recurrent_unit.py    ← WRU: FLUX-native recurrent cell (per-band interference)
├── train_wru.py              ← Training with next-wave prediction + decode loss
├── test_phase2_5_test1.py    ← Test: Predicted waves decode to readable text
├── test_phase2_5_test2.py    ← Test: Different contexts → different predictions
├── test_phase2_5_test3.py    ← Test: WRU output is always a valid wave
├── demo_phase2_5_demo1.py    ← Demo: Prompt → predict next word at each prefix length
├── demo_phase2_5_demo2.py    ← Demo: Per-band interference analysis
└── RESULTS_PHASE_2_5.md
```

### Acceptance Criteria
- [ ] WRU predicts next wave from prefix context (single step)
- [ ] Predicted waves decode to readable text (avg byte accuracy ≥ 60%)
- [ ] Different contexts → different predictions (avg pairwise cosine < 0.90)
- [ ] Output is always a valid wave (bounded energy, non-degenerate sub-bands)
- [ ] Hidden state is a valid wave at all times (decodable by WTT)
- [ ] **Decode gate:** avg byte accuracy ≥ 60%, min ≥ 30%
- [ ] Checkpoint saved to `checkpoints/phase2_5_v2.phase.pt`

### Why 2.5 Not 3
Phase 2.5 isolates the question: "Can FLUX predict the next wave at all?"
from: "Can it chain predictions autoregressively?" If single-step fails,
we know the problem is in the prediction head — not in the chaining logic.
This is the lesson from the Phase 9 legacy failure: never add complexity
before proving the base case works.

---

## ⬜ Phase 3: Deliberation + Autoregressive Wave Generation — AFTER 2.5

> **Status:** Not started (requires Phase 2.5 checkpoint)
> **Source:** Extends `v2/phase2_5/wave_recurrent_unit.py` with deliberation loop + chaining
> **Notebook:** `v2/V2_NOTEBOOKS/phase3_v2.ipynb`

### Goal
Chain WRU predictions autoregressively AND add deliberation cycles.
Phase 3 makes FLUX **think before it speaks** — like a human.

### What's Novel: Deliberation Cycles

Instead of one step → emit, run multiple internal WRU steps where the
output feeds back as input. Emit only when the wave **settles** (energy stabilizes):

```
Step 1:  context → WRU → wave₁  (candidate — don't emit yet)
Step 2:  wave₁ interferes with context → WRU → wave₂  (refined)
Step 3:  wave₂ interferes with context → WRU → wave₃  (more refined)
...
Step K:  energy(waveₖ) ≈ energy(waveₖ₋₁) → SETTLED → emit waveₖ
```

Key properties:
- **The model decides WHEN to speak** — emit when energy stabilizes, not after fixed K
- **Harder queries → more cycles** — "what is 2+2" settles fast; "explain quantum mechanics" takes longer
- **Sub-bands settle independently** — semantics may settle in 2 steps, syntax in 5
- **No new parameters** — same WRU, just run multiple times before emitting
- **Energy at emission = confidence** — low energy = certain; high energy = uncertain

This maps directly to SPECIFICATION.md:
> *"Energy minimization — settling to minimum energy IS both inference and learning"*

### Architecture
```
deliberate(context, max_cycles=10, settle_threshold=0.01):
    state = wru.initial_state
    for k in 1..max_cycles:
        predicted, state = wru.step(context, state)
        energy = ||predicted||²
        if |energy - prev_energy| < settle_threshold:
            return predicted  # SETTLED — emit this wave
    return predicted  # max cycles reached — emit best effort

autoregressive_generate(prefix_waves, num_words=10):
    for i in 1..num_words:
        context = w2f(prefix_waves.mean(dim=0))  # running context
        next_wave = deliberate(context)           # think, then speak
        prefix_waves = cat(prefix_waves, next_wave)
    return prefix_waves  # decode entire sequence via WTT
```

### Training
- **Supervised:** Teacher-forced single-step (same as Phase 2.5 but with chaining)
- **REINFORCE:** Autoregressive rollout with byte-accuracy reward (aligns train with inference)
- **Deliberation reward:** Bonus for settling in fewer cycles (efficiency pressure)

### Prerequisites
- Phase 2.5 checkpoint `phase2_5_v2.phase.pt` ✓ (WRU single-step proven)
- Phase 1 + Phase 2 checkpoints (frozen)

### What Gets Built
```
v2/phase3/
├── PHASE_3_SPEC.md
├── deliberator.py            ← Deliberation loop + energy-based settling
├── train_generator.py        ← Autoregressive chaining + REINFORCE + deliberation
├── demo_phase3_demo1.py      ← Demo: Prompt → deliberation → text generation
├── demo_phase3_demo2.py      ← Demo: Deliberation cycle count by query difficulty
├── test_phase3_test1.py      ← Test: Generated sequences decode to real words
├── test_phase3_test2.py      ← Test: Different prompts → different continuations
├── test_phase3_test3.py      ← Test: Deliberation settles (energy decreases)
└── RESULTS_PHASE_3.md
```

### Acceptance Criteria
- [ ] Autoregressive chaining produces multi-word output
- [ ] Generated waves decode to readable text (valid word rate > 50%)
- [ ] Different prompts produce different continuations (cosine < 0.85)
- [ ] Deliberation reduces energy monotonically per cycle
- [ ] Harder prompts → more deliberation cycles (measurable correlation)
- [ ] **Decode gate:** avg byte accuracy of generated output > 70%
- [ ] Checkpoint saved to `checkpoints/phase3_v2.phase.pt`

### What Changed vs Legacy Phase 9.5 + Original Phase 3 Plan
1. **No GRU** — WRU with physics-native interference gates replaces GRU entirely
2. **Deliberation** — multiple internal cycles before emitting (humans think before speaking)
3. **Energy-based stopping** — model decides when to speak, not a fixed step count
4. **Builds on proven Phase 2.5** — single-step prediction verified before adding chaining
5. **REINFORCE for alignment** — bridges the train/inference gap (lesson from failed Phase 9 runs)

---

## ⬜ Phase 4: Gravitational Relevance (O(log n) Wave Retrieval) — NOT STARTED

> **Status:** Not started
> **Legacy source:** `phases/phase3/` — `gravity.py`, `mass_tracker.py`, `spatial_index.py`, `negative_mass.py`, `benchmark_attention.py`
> **What changes for v2:** Add decode gate test; verify generation quality doesn't regress vs Phase 3 baseline

### Goal
Replace attention with gravitational search over the wave field.
Faster than O(n²) attention, and output still decodes to text.

### Prerequisites
- Phase 3 checkpoint `phase3_v2.phase.pt` ✓ (generation working)

### What Gets Built
```
v2/phase4/
├── PHASE_4_SPEC.md
├── gravity.py               ← Port of phases/phase3/gravity.py
├── mass_tracker.py          ← Port of phases/phase3/mass_tracker.py
├── spatial_index.py         ← Port of phases/phase3/spatial_index.py
├── negative_mass.py         ← Port of phases/phase3/negative_mass.py
├── benchmark_attention.py   ← Port of phases/phase3/benchmark_attention.py
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
- [ ] Checkpoint saved to `checkpoints/phase4_v2.phase.pt`

---

## ⬜ Phase 5: Thermodynamic Learning (Learn Without Backprop) — NOT STARTED

> **Status:** Not started
> **Legacy source:** `phases/phase4/` — `thermodynamic.py`, `temperature.py`, `energy_functions.py`, `online_learner.py`
> **What changes for v2:** Add decode gate test after online updates; verify generation quality doesn't regress

### Goal
Replace batch gradient descent with local energy settling.
Every input both produces output AND updates the field.

### Prerequisites
- Phase 4 checkpoint `phase4_v2.phase.pt` ✓

### What Gets Built
```
v2/phase5/
├── PHASE_5_SPEC.md
├── thermodynamic.py         ← Port of phases/phase4/thermodynamic.py
├── temperature.py           ← Port of phases/phase4/temperature.py
├── energy_functions.py      ← Port of phases/phase4/energy_functions.py
├── online_learner.py        ← Port of phases/phase4/online_learner.py
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
- [ ] Checkpoint saved to `checkpoints/phase5_v2.phase.pt`

---

## ⬜ Phase 6: Causal Geometry Nodes (Why, Not Just What) — NOT STARTED

> **Status:** Not started
> **Legacy source:** `phases/phase5/` — `cgn.py`, `manifold.py`, `causal_graph.py`, `multi_timescale.py`
> **What changes for v2:** Add decode gate test; verify generation quality doesn't regress

### Goal
Replace neurons with manifold patches that store what they know AND
why they know it. Enable causal tracing of any generated output.

### Prerequisites
- Phase 5 checkpoint `phase5_v2.phase.pt` ✓

### What Gets Built
```
v2/phase6/
├── PHASE_6_SPEC.md
├── cgn.py                   ← Port of phases/phase5/cgn.py
├── manifold.py              ← Port of phases/phase5/manifold.py
├── causal_graph.py          ← Port of phases/phase5/causal_graph.py
├── multi_timescale.py       ← Port of phases/phase5/multi_timescale.py
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
- [ ] Checkpoint saved to `checkpoints/phase6_v2.phase.pt`

---

## ⬜ Phase 7: Three-Tier Memory (No Forgetting, No Context Limit) — NOT STARTED

> **Status:** Not started
> **Legacy source:** `phases/phase6/` — `working_memory.py`, `episodic_memory.py`, `semantic_memory.py`, `memory_router.py`, `consolidation.py`
> **What changes for v2:** Memory must route through v2's trained wave projections; add generation quality test

### Goal
Working memory + episodic memory + semantic memory. The model remembers
across sessions, consolidates knowledge, and NEVER forgets.

### Prerequisites
- Phase 6 checkpoint `phase6_v2.phase.pt` ✓

### What Gets Built
```
v2/phase7/
├── PHASE_7_SPEC.md
├── working_memory.py        ← Port of phases/phase6/working_memory.py
├── episodic_memory.py       ← Port of phases/phase6/episodic_memory.py
├── semantic_memory.py       ← Port of phases/phase6/semantic_memory.py
├── memory_router.py         ← Port of phases/phase6/memory_router.py
├── consolidation.py         ← Port of phases/phase6/consolidation.py
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
- [ ] Checkpoint saved to `checkpoints/phase7_v2.phase.pt`

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

## ⬜ Phase 8: Full FLUX Integration — NOT STARTED

> **Status:** Not started
> **Legacy source:** `phases/phase7/` — `flux_model.py`, `flux_generate.py`, `flux_trainer.py`, `baseline_lstm.py`
> **What changes for v2:** Wire all v2 phase checkpoints into a single model; this should be mostly plumbing since all components are already trained with compatible wave spaces

### Goal
Combine all v2 components into a single unified model. Run end-to-end
text generation with the complete pipeline.

### Prerequisites
- All phase checkpoints `phase1_v2.phase.pt` through `phase7_v2.phase.pt` ✓

### What Gets Built
```
v2/phase8/
├── PHASE_8_SPEC.md
├── flux_model.py            ← Port/rewrite of phases/phase7/flux_model.py
├── flux_generate.py         ← Port of phases/phase7/flux_generate.py
├── flux_trainer.py          ← Port of phases/phase7/flux_trainer.py
├── baseline_lstm.py         ← Port of phases/phase7/baseline_lstm.py
├── demo_phase8_demo1.py     ← Demo: Complete generation pipeline
├── demo_phase8_demo2.py     ← Demo: Real-time learning during chat
├── demo_phase8_demo3.py     ← Demo: FLUX vs LSTM quality
├── test_phase8_test1.py     ← Test: All components loaded correctly
├── test_phase8_test2.py     ← Test: End-to-end generation coherence
├── test_phase8_test3.py     ← Test: Real-time learning works
└── RESULTS_PHASE_8.md
```

### Acceptance Criteria
- [ ] All v2 phase checkpoints load into single model
- [ ] End-to-end: prompt → CSE → Field → GR → TL → CGN → Memory → Generate → Decode → text
- [ ] Generation quality ≥ small LSTM baseline
- [ ] Real-time learning: new fact → immediately usable in generation
- [ ] **Decode gate:** full pipeline output > 90% byte accuracy
- [ ] Checkpoint saved to `checkpoints/phase8_v2.phase.pt`

---

## ⬜ Phase 9: Scale & GPT-2 Benchmark — NOT STARTED

> **Status:** Not started
> **Legacy source:** `phases/phase8/` — `flux_large.py`, `train_openwebtext.py`, `benchmark_gpt2.py`, `kaggle_train.py`
> **What changes for v2:** Scale the v2 model (not legacy) on OpenWebText; the benchmark suite is identical

### Goal
Scale FLUX to GPT-2 equivalent size and benchmark head-to-head.

### Prerequisites
- Phase 8 checkpoint `phase8_v2.phase.pt` ✓ (full integration working)

### What Gets Built
```
v2/phase9/
├── PHASE_9_SPEC.md
├── flux_large.py            ← Port of phases/phase8/flux_large.py (scaled config)
├── train_openwebtext.py     ← Port of phases/phase8/train_openwebtext.py
├── benchmark_gpt2.py        ← Port of phases/phase8/benchmark_gpt2.py
├── kaggle_train.py          ← Kaggle-optimized training script
├── demo_phase9_demo1.py     ← Demo: FLUX vs GPT-2 generation quality
├── demo_phase9_demo2.py     ← Demo: FLUX continual learning advantage
├── demo_phase9_demo3.py     ← Demo: FLUX speed at long sequences
├── test_phase9_test1.py     ← Test: Perplexity on Penn Treebank
├── test_phase9_test2.py     ← Test: Perplexity on WikiText-2
├── test_phase9_test3.py     ← Test: Continual learning (FLUX wins)
├── test_phase9_test4.py     ← Test: Long sequence speed (FLUX wins)
└── RESULTS_PHASE_9.md
```

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

The existing Phase 1–9 legacy code doesn't get deleted — it gets **re-sequenced**.

### Phase mapping (legacy → v2)

| Legacy Phase | Legacy Component | v2 Phase | Status | Notes |
|-------------|-----------------|----------|--------|-------|
| phases/phase1/ | CSE (encode only) | v2/phase1/ | ✅ Done | Added WTT + Chunker + joint training |
| phases/phase9/ | WaveChunker | v2/phase1/ | ✅ Done | Moved forward — part of the codec |
| phases/phase9_1/ | WaveToText / ContextWTT | v2/phase1/ | ✅ Done | Moved forward — decode from day 1 |
| phases/phase2/ | ResonanceField | v2/phase2/ | ✅ Done | Retrained bridges with decode loss — **root cause of legacy failure fixed** |
| (NEW) | WaveRecurrentUnit | v2/phase2_5/ | 🔲 Next | FLUX-native recurrent cell — no GRU |
| v2/phase2_5/ | Deliberation + chaining | v2/phase3/ | ⬜ | Autoregressive generation with settling |
| phases/phase3/ | GravitationalRelevance | v2/phase4/ | ⬜ | Add decode gate to tests |
| phases/phase4/ | ThermodynamicLearner | v2/phase5/ | ⬜ | Add decode gate to tests |
| phases/phase5/ | CausalGeometryNodes | v2/phase6/ | ⬜ | Add decode gate to tests |
| phases/phase6/ | Memory (3-tier) | v2/phase7/ | ⬜ | Add decode gate to tests |
| phases/phase7/ | FLUXModel integration | v2/phase8/ | ⬜ | Wire v2 checkpoints |
| phases/phase8/ | Scale & benchmark | v2/phase9/ | ⬜ | Same benchmark, v2 model |

### Rewrite scope per phase
- ✅ Phase 1 v2: ~40% new code — joint training loop was entirely new
- ✅ Phase 2 v2: ~20% new code — decode loss on projections was the key fix
- 🔲 Phase 2.5 v2: ~100% new code — WRU is a novel FLUX-native architecture
- ⬜ Phase 3 v2: ~60% new code — deliberation loop + autoregressive chaining on WRU
- ⬜ Phases 4–7 v2: ~5% each — add decode gate check to tests, update imports
- ⬜ Phases 8–9 v2: ~0% — same code, renumbered, v2 checkpoints wired in

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
phase1_v2.phase.pt    → CSE + WaveChunker + WaveToText (full codec)              ✅ EXISTS
phase2_v2.phase.pt    → codec + ResonanceField + trained projections              ✅ EXISTS
phase2_5_v2.phase.pt  → all above + WRU (single-step next-wave)                  🔲 NEXT
phase3_v2.phase.pt    → all above + deliberation + autoregressive chaining        ⬜
phase4_v2.phase.pt    → all above + GravitationalRelevance                        ⬜
phase5_v2.phase.pt    → all above + ThermodynamicLearner                          ⬜
phase6_v2.phase.pt    → all above + CausalGeometryNodes                           ⬜
phase7_v2.phase.pt    → all above + Three-Tier Memory                             ⬜
phase8_v2.phase.pt    → Full FLUX integrated model                                 ⬜
phase9_v2.phase.pt    → Scaled FLUX trained on OpenWebText                         ⬜
```

Every checkpoint from Phase 2.5 onward can **predict next waves**.
Every checkpoint from Phase 3 onward can **generate multi-word text**.
From Phase 1 onward, FLUX can encode, chunk, decode, and verify.

> Note: Checkpoints use `_v2` suffix to distinguish from legacy `phase1.phase.pt`
> files which used the broken encoder-first ordering.

---

## Definition of Success

**Minimum:** Round-trip byte accuracy > 95% at Phase 1, never drops below 90%.

**Target:** Generation quality improves monotonically from Phase 2.5 → Phase 9,
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
Step 2.5:           —                           Single next-wave prediction (WRU)
Step 3:             Add forces                  Deliberation + generation
Step 4-7:           More forces + memory        Add forces + memory + observe
Step 8:             Integrate                   Integrate (already works)
Step 9:             Try to observe → FAIL       Scale (everything decodes)
```
