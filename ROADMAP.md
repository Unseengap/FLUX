# FLUX Roadmap
## Phase-by-Phase Build Plan

> Each phase builds on the last. Each phase saves a checkpoint.
> Each phase loads and verifies the previous checkpoint before starting.
> Each phase has demos, tests, and a RESULTS file.

---

## Phase Overview

```
Phase 1  ──► Continuous Semantic Encoder (CSE)
Phase 2  ──► Resonance Field Core (RFC)
Phase 3  ──► Gravitational Relevance (GR)
Phase 4  ──► Thermodynamic Learning (TL)
Phase 5  ──► Causal Geometry Nodes (CGN)
Phase 6  ──► Three-Tier Memory System
Phase 7  ──► Full FLUX Integration
Phase 8  ──► Scale & GPT-2 Benchmark
```

Each phase is a complete unit:
- Loads previous checkpoint ✓
- Builds new component ✓
- Saves new checkpoint ✓
- Passes all tests ✓
- Has working demo ✓
- Has RESULTS.md ✓
- Has Kaggle notebook (`notebooks/phaseN_kaggle.ipynb`) ✓
- Uploads checkpoint to HuggingFace Hub (`UnseenGAP/FLUX`) ✓
- Writes logs to `logs/phaseN.log` via `PhaseLogger` ✓

---

## Phase 1: Continuous Semantic Encoder (CSE)

### Goal
Replace the tokenizer and embedding layer with a continuous wave-based encoder that requires no vocabulary.

### What Gets Built
```
phases/phase1/
├── PHASE_1_SPEC.md          ← Detailed spec for this phase
├── cse.py                   ← Core CSE implementation
├── wave_types.py            ← SemanticWave dataclass and ops
├── interference.py          ← Wave interference functions
├── train_cse.py             ← CSE training script
├── demo_phase1_demo1.py     ← Demo: Encode text, visualize waves
├── demo_phase1_demo2.py     ← Demo: Show interference patterns
├── test_phase1_test1.py     ← Test: Reconstruction quality
├── test_phase1_test2.py     ← Test: Language agnostic encoding
├── test_phase1_test3.py     ← Test: Similar words → similar waves
└── RESULTS_PHASE_1.md       ← Auto-generated results

notebooks/
└── phase1_kaggle.ipynb      ← Kaggle notebook: train + test + upload
```

### Acceptance Criteria
- [ ] Any UTF-8 string encodes without errors
- [ ] Similar words produce cosine similarity > 0.7
- [ ] Opposite words produce cosine similarity < 0.2
- [ ] Encoding is deterministic (same input = same wave)
- [ ] Reconstruction loss < 0.1 on validation set
- [ ] Encodes English, Chinese, code, and math equally
- [ ] All tests pass
- [ ] Both demos run without error
- [ ] Checkpoint saved to `checkpoints/phase1.phase.pt`
- [ ] Checkpoint uploaded to HuggingFace Hub (`UnseenGAP/FLUX`)
- [ ] Phase log written to `logs/phase1.log`

### Key Implementation Notes
```python
# CSE uses byte-level input — no tokenization at all
# Sliding window over bytes → convolutional filters → wave space
# Wave interference computed over local neighborhood only
# Output is SemanticWave tensor [seq_len, 432]
# Training signal: reconstruction + interference consistency loss
```

### Demo 1: Wave Visualization
```
Input: "The quick brown fox"
Output: Visual plot of semantic wave per character window
        Shows how meaning builds up across the sequence
        Shows interference between nearby windows
```

### Demo 2: Interference Patterns
```
Input pair: ("king", "queen")  → should show constructive interference
Input pair: ("hot", "cold")    → should show destructive interference
Input pair: ("cat", "xyz123")  → should show near-zero interference
```

### Test Checklist
```
test_phase1_test1.py: Reconstruction
  - Encode → decode 1000 sentences
  - Average reconstruction loss < 0.1
  - No sentence fails completely

test_phase1_test2.py: Language Agnostic
  - Encode English sentence
  - Encode French translation
  - Encode Chinese translation
  - Cosine similarity of semantic dimension > 0.5 across all

test_phase1_test3.py: Semantic Proximity
  - "dog" vs "cat" similarity > "dog" vs "democracy" similarity
  - "Paris" vs "France" similarity > "Paris" vs "banana" similarity
  - 20 pairs tested, 18/20 must pass ordering check
```

---

## Phase 2: Resonance Field Core (RFC)

### Goal
Build the field state tensor that replaces all weight matrices. The field should form stable attractors when shown repeated patterns and maintain them when shown new patterns.

### Prerequisites
- Phase 1 checkpoint loaded and verified ✓
- CSE producing valid SemanticWave tensors ✓

### What Gets Built
```
phases/phase2/
├── PHASE_2_SPEC.md
├── field.py                 ← ResonanceField class
├── attractor.py             ← Attractor detection and catalog
├── field_ops.py             ← Field perturbation, settling, energy
├── flux_format.py           ← .flx file save/load
├── train_field.py           ← Field training on simple patterns
├── demo_phase2_demo1.py     ← Demo: Field forming attractors live
├── demo_phase2_demo2.py     ← Demo: New info doesn't destroy old
├── test_phase2_test1.py     ← Test: Attractor formation
├── test_phase2_test2.py     ← Test: Local update (no global change)
├── test_phase2_test3.py     ← Test: Field stability over 1000 steps
└── RESULTS_PHASE_2.md
```

### Acceptance Criteria
- [ ] Field forms stable attractor after 10 repetitions of same input
- [ ] New pattern creates new attractor without destroying old ones
- [ ] Field energy converges (does not explode or collapse)
- [ ] Local update only affects neighborhood (verified by test)
- [ ] .flx file saves and loads correctly
- [ ] Loaded field continues from exact saved state
- [ ] All tests pass

### Demo 1: Attractor Formation Live
```
Show field energy landscape updating in real time as patterns are fed in
Visualize energy wells forming at repeated patterns
Show that each new pattern creates a new well, old wells persist
```

### Demo 2: No Forgetting Demonstration
```
Step 1: Feed pattern A 20 times → attractor A forms
Step 2: Feed pattern B 20 times → attractor B forms
Step 3: Query for pattern A → field correctly returns A
Step 4: Print: "Catastrophic forgetting score: 0.0"
```

---

## Phase 3: Gravitational Relevance (GR)

### Goal
Replace attention with gravitational relevance. Must be faster than attention at sequence length > 512 and equally accurate.

### Prerequisites
- Phase 1 checkpoint: CSE working ✓
- Phase 2 checkpoint: Field core working ✓

### What Gets Built
```
phases/phase3/
├── PHASE_3_SPEC.md
├── gravity.py               ← GravitationalRelevance class
├── mass_tracker.py          ← Evidence mass accumulation
├── spatial_index.py         ← KD-tree / FAISS spatial index
├── negative_mass.py         ← Contradiction → repulsion
├── benchmark_attention.py   ← Speed comparison vs torch attention
├── demo_phase3_demo1.py     ← Demo: Speed vs attention at seq lengths
├── demo_phase3_demo2.py     ← Demo: Mass accumulation visualization
├── test_phase3_test1.py     ← Test: O(log n) complexity verified
├── test_phase3_test2.py     ← Test: Retrieval precision@k
├── test_phase3_test3.py     ← Test: Negative mass repulsion
└── RESULTS_PHASE_3.md
```

### Acceptance Criteria
- [ ] Faster than PyTorch attention at seq_len = 1024
- [ ] Faster than PyTorch attention at seq_len = 4096
- [ ] Precision@10 > 0.8 on similarity retrieval task
- [ ] Negative mass correctly repels contradicted concepts
- [ ] Mass grows correctly with repeated evidence
- [ ] All tests pass

### Speed Benchmark Target
```
seq_len=512:    GR ≤ attention latency (acceptable parity)
seq_len=1024:   GR < attention by 20%
seq_len=4096:   GR < attention by 60%
seq_len=16384:  GR < attention by 90% (attention likely OOM, GR runs fine)
```

---

## Phase 4: Thermodynamic Learning (TL)

### Goal
Replace backpropagation with local thermodynamic settling. The system should learn from single examples in real time without a training loop.

### Prerequisites
- Phases 1–3 checkpoints loaded ✓

### What Gets Built
```
phases/phase4/
├── PHASE_4_SPEC.md
├── thermodynamic.py         ← ThermodynamicLearner class
├── temperature.py           ← Temperature schedule and dynamics
├── energy_functions.py      ← Local energy computations
├── online_learner.py        ← Real-time single-sample learning
├── demo_phase4_demo1.py     ← Demo: Learn a fact in one shot
├── demo_phase4_demo2.py     ← Demo: Compare to SGD convergence
├── test_phase4_test1.py     ← Test: Single-shot learning retention
├── test_phase4_test2.py     ← Test: No global gradient required
├── test_phase4_test3.py     ← Test: Temperature annealing behavior
└── RESULTS_PHASE_4.md
```

### Acceptance Criteria
- [ ] Model learns from single example (no batch, no epochs)
- [ ] Learned fact retrievable after 100 subsequent updates
- [ ] No global gradient computation at any point (verified)
- [ ] Temperature decreases as error decreases
- [ ] Learning is faster than equivalent SGD on simple tasks
- [ ] All tests pass

### Demo 1: One-Shot Fact Learning
```
Input: "The capital of Mars colony Alpha is New Houston"
→ One settling pass
Query: "What is the capital of Mars colony Alpha?"
→ System should return: "New Houston"
→ No training loop. No epochs. One example.
```

---

## Phase 5: Causal Geometry Nodes (CGN)

### Goal
Build the CGN layer that replaces neurons with geometry-aware nodes that store causal relationships, not just mappings.

### Prerequisites
- Phases 1–4 checkpoints loaded ✓

### What Gets Built
```
phases/phase5/
├── PHASE_5_SPEC.md
├── cgn.py                   ← CausalGeometryNode class
├── manifold.py              ← Manifold patch operations
├── causal_graph.py          ← Causal arrow storage and tracing
├── multi_timescale.py       ← Fast/slow node coordination
├── demo_phase5_demo1.py     ← Demo: Trace why a conclusion was reached
├── demo_phase5_demo2.py     ← Demo: Fast vs slow node activation
├── test_phase5_test1.py     ← Test: Causal trace accuracy
├── test_phase5_test2.py     ← Test: Multi-timescale separation
├── test_phase5_test3.py     ← Test: Geometry computation correctness
└── RESULTS_PHASE_5.md
```

### Acceptance Criteria
- [ ] Every output has a traceable causal chain
- [ ] Fast nodes react in < 5 steps, slow nodes in > 50 steps
- [ ] Geometry computation produces correct signal bending
- [ ] Causal invalidation works (disprove cause → invalidate conclusion)
- [ ] All tests pass

### Demo 1: Why Did You Say That?
```
Feed model: "Birds can fly. Penguins are birds."
Query: "Can penguins fly?"
Response includes causal trace:
  → Conclusion: "Generally yes, but penguins are an exception"
  → Causal chain: [penguin→bird, bird→fly, penguin→cannot_fly]
  → Conflict detected and resolved via evidence weight
```

---

## Phase 6: Three-Tier Memory System

### Goal
Integrate working memory, episodic memory, and semantic memory into a unified system with no catastrophic forgetting at any tier.

### Prerequisites
- Phases 1–5 checkpoints loaded ✓

### What Gets Built
```
phases/phase6/
├── PHASE_6_SPEC.md
├── working_memory.py        ← Rolling field window
├── episodic_memory.py       ← FAISS vector store + metadata
├── semantic_memory.py       ← Protected field core (semantic tier)
├── memory_router.py         ← Routes between tiers
├── consolidation.py         ← Episodic → Semantic distillation
├── demo_phase6_demo1.py     ← Demo: Cross-session memory
├── demo_phase6_demo2.py     ← Demo: Consolidation process live
├── demo_phase6_demo3.py     ← Demo: Zero forgetting over 1000 tasks
├── test_phase6_test1.py     ← Test: One-shot episodic write/read
├── test_phase6_test2.py     ← Test: Semantic memory protection
├── test_phase6_test3.py     ← Test: Forgetting score = 0.0
└── RESULTS_PHASE_6.md
```

### Acceptance Criteria
- [ ] Write fact to episodic memory → retrieve correctly 100 steps later
- [ ] Semantic memory unchanged after 1000 episodic writes
- [ ] Forgetting score = 0.0 across 10 sequential task pairs
- [ ] Consolidation promotes high-frequency episodic → semantic
- [ ] Memory persists across save/load cycle
- [ ] All tests pass

### The Forgetting Test (Most Important Test in Project)
```python
# test_phase6_test3.py — THE key differentiator vs transformers
tasks = generate_10_task_pairs()  # 10 pairs of unrelated tasks

for task_A, task_B in tasks:
    train_on(task_A)
    accuracy_before = eval(task_A)
    
    train_on(task_B)              # Would destroy transformer
    accuracy_after = eval(task_A)
    
    forgetting = (accuracy_before - accuracy_after) / accuracy_before
    assert forgetting < 0.02, f"Forgetting too high: {forgetting}"
    # Target: < 2% degradation
    # Transformer baseline: 30-80% degradation
```

---

## Phase 7: Full FLUX Integration

### Goal
Combine all components into a single unified FLUX model. Run end-to-end text generation. Compare quality to a small LSTM baseline.

### Prerequisites
- All phase checkpoints 1–6 loaded ✓

### What Gets Built
```
phases/phase7/
├── PHASE_7_SPEC.md
├── flux_model.py            ← FLUXModel — full integration class
├── flux_generate.py         ← Text generation pipeline
├── flux_trainer.py          ← Unified training on text data
├── baseline_lstm.py         ← Small LSTM for comparison
├── demo_phase7_demo1.py     ← Demo: End-to-end text generation
├── demo_phase7_demo2.py     ← Demo: Real-time learning during chat
├── demo_phase7_demo3.py     ← Demo: FLUX vs LSTM quality comparison
├── test_phase7_test1.py     ← Test: Full pipeline integration
├── test_phase7_test2.py     ← Test: Generation coherence
├── test_phase7_test3.py     ← Test: All components loaded correctly
└── RESULTS_PHASE_7.md
```

### Acceptance Criteria
- [ ] Full model loads all phase checkpoints correctly
- [ ] End-to-end text generation works
- [ ] Generation quality >= small LSTM baseline (perplexity)
- [ ] Real-time learning demonstrated (new fact → immediately usable)
- [ ] .flx model file saves and loads for full inference
- [ ] All tests pass

### Demo 1: Real-Time Learning Chat
```
# This demonstrates the key advantage over all existing models
flux = FLUXModel.load('checkpoints/phase7.flx')

flux.chat("My name is Alex and I am a marine biologist")
# → FLUX writes this to episodic memory in real time

flux.chat("What do you know about me?")
# → "You are Alex, a marine biologist"
# → No fine-tuning. No RAG pipeline setup. Pure real-time learning.
```

---

## Phase 8: Scale and GPT-2 Benchmark

### Goal
Scale FLUX to GPT-2 equivalent size, train on OpenWebText, and benchmark head-to-head.

### Prerequisites
- Full FLUX model from Phase 7 working ✓
- Kaggle GPU quota or equivalent ✓

### What Gets Built
```
phases/phase8/
├── PHASE_8_SPEC.md
├── flux_large.py            ← Scaled FLUX configuration
├── train_openwebtext.py     ← Training on OpenWebText dataset
├── benchmark_gpt2.py        ← Full benchmark suite vs GPT-2
├── kaggle_train.py          ← Kaggle-optimized training script
├── demo_phase8_demo1.py     ← Demo: FLUX vs GPT-2 generation quality
├── demo_phase8_demo2.py     ← Demo: FLUX continual learning advantage
├── demo_phase8_demo3.py     ← Demo: FLUX speed at long sequences
├── test_phase8_test1.py     ← Test: Perplexity on Penn Treebank
├── test_phase8_test2.py     ← Test: Perplexity on WikiText-2
├── test_phase8_test3.py     ← Test: Continual learning (FLUX wins)
├── test_phase8_test4.py     ← Test: Long sequence speed (FLUX wins)
└── RESULTS_PHASE_8.md
```

### Benchmark Suite

#### Standard NLP Benchmarks
```
Penn Treebank perplexity      → Target: ≤ GPT-2 small (117M)
WikiText-2 perplexity         → Target: ≤ GPT-2 small
Text generation human eval    → Target: ≥ GPT-2 small quality
```

#### FLUX-Specific Benchmarks (Where FLUX Should Win)
```
Continual Learning Retention  → Target: FLUX > GPT-2 by 50%+ 
One-Shot Fact Learning        → Target: FLUX succeeds, GPT-2 fails
Long Sequence Speed (16k tok) → Target: FLUX > GPT-2 by 5x
Real-Time Adaptation          → Target: FLUX adapts, GPT-2 cannot
Cross-Session Memory          → Target: FLUX retains, GPT-2 cannot
```

### Kaggle Setup
```python
# kaggle_train.py — optimized for Kaggle P100/T4 GPUs
# Dataset: openwebtext (available on HuggingFace)
# Estimated training time: 48-72 hours on dual T4
# Checkpointing every 5000 steps to Kaggle output
# Resume-safe: loads latest checkpoint if interrupted
```

---

## Results Tracking

### RESULTS Template (Auto-generated each phase)
Each phase generates `RESULTS_PHASE_N.md` automatically at end of test run:

```markdown
# Results: Phase N — [Component Name]
Generated: [timestamp]
Hardware: [GPU info]
Duration: [training time]

## Test Results
| Test | Status | Score | Notes |
|------|--------|-------|-------|
| test_phaseN_test1 | PASS/FAIL | metric | notes |

## Demo Outputs
### demo_phaseN_demo1
[captured output]

## Metrics
[phase-specific metrics]

## Comparison to Previous Phase
[delta from last phase]

## Issues Encountered
[any problems and how resolved]

## Next Phase Readiness
Checkpoint saved: YES/NO
All tests passing: YES/NO
Ready for Phase N+1: YES/NO
```

---

## Checkpoint Chain

Every phase checkpoint is verified to contain:
```
phase1.phase.pt → CSE encoder state
phase2.phase.pt → CSE + Field state
phase3.phase.pt → CSE + Field + GR state
phase4.phase.pt → CSE + Field + GR + TL state
phase5.phase.pt → CSE + Field + GR + TL + CGN state
phase6.phase.pt → all above + Memory system state
phase7.phase.pt → Full FLUX model state (.flx format)
phase8.phase.pt → Scaled FLUX trained on OpenWebText
```

Each checkpoint loads the previous one and verifies integrity before building on it:
```python
def load_and_verify(phase_n: int) -> Dict:
    checkpoint = torch.load(f'checkpoints/phase{phase_n}.phase.pt')
    assert checkpoint['phase'] == phase_n
    assert all_components_present(checkpoint)
    assert run_smoke_test(checkpoint)  # Quick forward pass
    print(f"✓ Phase {phase_n} checkpoint verified")
    return checkpoint
```

### Checkpoint Storage

| Location | Contents | Access |
|---|---|---|
| `checkpoints/` (local) | `.phase.pt` files | Fastest; gitignored |
| HuggingFace Hub (`UnseenGAP/FLUX`) | `checkpoints/phaseN.phase.pt` | Persistent cloud backup |
| HuggingFace Hub (`UnseenGAP/FLUX`) | `logs/phaseN.log` | Training logs |
| GitHub (`Unseengap/FLUX`) | Code + logs + results | Version controlled |

Checkpoints are uploaded via `upload_checkpoint_to_hf()` after training.
If a local checkpoint is missing, `load_checkpoint()` automatically falls back to downloading from HuggingFace Hub.

---

## Kaggle Notebook Standard

Every phase notebook (`notebooks/phaseN_kaggle.ipynb`) follows this cell structure:

| Cell | Purpose |
|---|---|
| 1 | Clone / pull repo from GitHub |
| 2 | Install dependencies + `setup.py` |
| 3 | Init `PhaseLogger`, detect hardware, load `HF_TOKEN` from Kaggle secrets |
| 4 | Smoke test (build model, verify gradients) |
| 5 | Training run |
| 6 | Upload checkpoint to HuggingFace Hub |
| 7–9 | Run 3 phase tests |
| 10–11 | Run 2 demos |
| 12 | Interactive exploration |
| 13 | View RESULTS_PHASE_N.md |
| 14 | View full phase log |
| 15 | Final upload (logs → HF + GitHub) |
| 16 | Save artifacts to Kaggle output |

Every code cell calls `log.cell_start()` / `log.cell_end()` for persistent logging.

---

## Timeline Estimate

| Phase | Estimated Duration | Bottleneck |
|---|---|---|
| 1 CSE | 1–2 weeks | Novel encoder math |
| 2 RFC | 1–2 weeks | Field stability tuning |
| 3 GR | 1 week | KD-tree integration |
| 4 TL | 2–3 weeks | Convergence without backprop |
| 5 CGN | 2–3 weeks | Manifold geometry implementation |
| 6 Memory | 1–2 weeks | FAISS integration + consolidation |
| 7 Integration | 2–3 weeks | Component interaction bugs |
| 8 Scale | 4–8 weeks | Training time dependent |

**Total: 3–6 months of focused daily work**

This is research, not engineering. Each phase may require multiple iterations before passing its acceptance criteria. That is expected and correct.

---

## Definition of Success

**Minimum success:** FLUX beats a transformer of equal size on continual learning retention (forgetting score < 0.05 vs transformer's 0.3+)

**Target success:** FLUX matches GPT-2 small on perplexity AND wins on all FLUX-specific benchmarks

**Stretch success:** FLUX beats GPT-2 small on perplexity while being faster at long sequences and showing zero catastrophic forgetting

Any one of these outcomes is publishable research.
