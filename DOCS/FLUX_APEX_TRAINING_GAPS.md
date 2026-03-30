# Flux-Apex Training Gaps — Data Requirements & Training Plan

**Document Purpose:** Identify data-starved components and create actionable training plan  
**Target Model:** Flux-Apex-V2.flx  
**Created:** March 30, 2026

---

## Executive Summary

Flux-Apex-V1 has **1.9B parameters** but many components lack adequate training:

| Severity | Component | Current Data | Impact |
|----------|-----------|--------------|--------|
| 🔴 Critical | VisionToWave | 0 (missing) | No visual understanding |
| 🔴 Critical | WaveToGrid | ~400 synthetic | Poor grid reconstruction |
| 🔴 Critical | AudioToWave | 0 (stub) | No speech capability |
| 🟠 Moderate | Decoder | 13MB text | Limited fluency |
| 🟠 Moderate | Causal Graph | 463 links | Game-specific only |
| 🟡 Minor | Field | 155k samples | Domain-limited |

**Key Constraint:** ARC-AGI-3 has **NO public training dataset**. All ARC-3 data must come from:
1. Gameplay experience (learning while playing)
2. Synthetically generated ARC-style grids
3. Transfer from ARC-AGI-1 (400 public training tasks)

---

## Component Analysis

### 🔴 1. VisionToWave (MISSING — Highest Priority)

**Current State:** Does not exist  
**Problem:** FLUX cannot see images, relies on external Qwen2.5-VL  
**Solution:** SVD-compressed vision encoder from Qwen2.5-VL

```
Qwen2.5-VL Vision Encoder (~400M params)
    │
    ▼ SVD Compression (keep top-k singular values)
    │
Vision-SVD (~100-200M params)
    │
    ▼ Projection layer
    │
vision_wave [432]
```

**Training Data Options:**
| Dataset | Size | Type | Availability |
|---------|------|------|--------------|
| **COCO Captions** | 330k images | Image + text | ✓ Public |
| **Conceptual Captions** | 3M images | Image + text | ✓ Public |
| **LLaVA-Instruct** | 150k samples | Vision QA | ✓ Public |
| **ARC Screenshots** | Self-generated | Game frames | ✓ Create ourselves |

**Training Plan:**
```python
# Phase 13A: Extract SVD vision encoder
1. Load Qwen2.5-VL vision encoder
2. For each weight matrix W:
   U, S, Vt = svd(W)
   k = rank // 4  # Keep 25% of singular values
   W_svd = U[:,:k] @ diag(S[:k]) @ Vt[:k,:]
3. Add projection: vision_latent → vision_wave [432]
4. Save to .flx

# Phase 13B: Fine-tune on FLUX wave space
1. Load COCO/CC3M images
2. For each image:
   vision_wave = vision_encoder(image)
   text_wave = cse.encode(caption)
   loss = 1 - cosine_sim(vision_wave, text_wave)
3. Backprop through projection layer only
```

**Estimated Size Addition:** 200-500MB to .flx  
**Training Time:** ~4-8 hours on T4 GPU

---

### 🔴 2. WaveToGrid (Poor Reconstruction)

**Current State:** 192,256 params, trained on ~400 synthetic grids  
**Problem:** Can encode grids but barely reconstructs them  
**Metric:** Round-trip fidelity ~60% (should be >95%)

**Data Sources:**
| Dataset | Size | Notes |
|---------|------|-------|
| **ARC-AGI-1 Training** | 400 tasks | Public, official |
| **ARC-AGI-1 Evaluation** | 400 tasks | Public, official |
| **Re-ARC** | 1000+ tasks | Community-generated |
| **Synthetic Grids** | Unlimited | Generate on-the-fly |

**⚠️ ARC-AGI-3 Note:** No public training set exists. Must rely on:
1. ARC-1 transfer (patterns should generalize)
2. Synthetic generation (random valid grids)
3. Online learning (learn from gameplay)

**Training Plan:**
```python
# Phase 13C: Grid adapter training
1. Load ARC-AGI-1 training set (400 tasks)
2. For each task:
   for example in task['train'] + task['test']:
       input_grid = example['input']
       output_grid = example['output']
       
       # Encode-decode round trip
       wave = grid_to_wave(input_grid)
       reconstructed = wave_to_grid(wave)
       
       # Loss: exact cell matching
       loss = cross_entropy(reconstructed, input_grid)
       
3. Generate 10,000 synthetic grids for augmentation
4. Train until round-trip accuracy > 95%
```

**Estimated Training Time:** ~2 hours on T4 GPU

---

### 🔴 3. AudioToWave (Stub Only)

**Current State:** Empty stub, 34,992 params (untrained)  
**Problem:** Cannot process speech or audio

**Data Sources:**
| Dataset | Size | Type |
|---------|------|------|
| **LibriSpeech** | 1000 hours | Speech + transcripts |
| **Common Voice** | 17k hours | Speech + transcripts |
| **AudioCaps** | 46k clips | Audio + captions |

**Training Plan:**
```python
# Phase 13D: Audio encoder training
1. Load LibriSpeech-100
2. For each (audio, transcript) pair:
   mel = librosa.melspectrogram(audio)
   audio_wave = audio_to_wave(mel)
   text_wave = cse.encode(transcript)
   loss = 1 - cosine_sim(audio_wave, text_wave)
3. Train projection layer
```

**Estimated Size Addition:** ~50MB to .flx  
**Training Time:** ~6 hours on T4 GPU

---

### 🟠 4. Decoder (Limited Fluency)

**Current State:** 65M params, trained on 13MB text  
**Problem:** Basic fluency but struggles with complex generation

**Current Training:**
- WikiText-2: ~4MB
- Penn Treebank: ~5MB
- Misc injections: ~4MB

**Additional Data Needed:**
| Dataset | Size | Purpose |
|---------|------|---------|
| **OpenWebText** | 38GB | General fluency |
| **The Pile (subset)** | 10GB | Diverse knowledge |
| **Code (StarCoder)** | 5GB | Programming |
| **Dialogue (ShareGPT)** | 500MB | Conversation |

**Training Plan:**
```python
# Phase 13E: Decoder enrichment
1. Load OpenWebText sample (1GB)
2. Stream through FLUX (thermodynamic settling)
3. Periodically evaluate on held-out text
4. Target: perplexity < 50 on WikiText-2
```

**Note:** FLUX uses thermodynamic settling, not backprop epochs. Data streams continuously.

---

### 🟠 5. Causal Graph (Game-Specific)

**Current State:** 463 causal links from ARC gameplay  
**Problem:** Knowledge is ARC-specific, no general reasoning

**Data Sources:**
| Dataset | Size | Type |
|---------|------|------|
| **ConceptNet** | 34M edges | Commonsense relations |
| **ATOMIC** | 877k events | If-then reasoning |
| **CausalNet** | 500k rules | Causal extraction |

**Training Plan:**
```python
# Phase 13F: Causal knowledge injection
1. Load ConceptNet subset (top 100k relations)
2. For each (concept, relation, target):
   # e.g., ("bird", "CapableOf", "fly")
   causal_link = {
       'source': encode_concept(concept),
       'target': encode_concept(target),
       'relation': relation,
       'weight': confidence,
   }
   causal_graph.add_link(causal_link)
3. Prune low-weight edges
```

---

### 🟡 6. Field Knowledge (Domain-Limited)

**Current State:** 155k samples from WikiText + custom injection  
**Problem:** Strong on text, weak on other domains

**Additional Domains Needed:**
| Domain | Dataset | Purpose |
|--------|---------|---------|
| **Math** | GSM8K, MATH | Numerical reasoning |
| **Logic** | FOLIO, LogiQA | Formal logic |
| **Spatial** | NLVR2, GQA | Spatial relations |
| **ARC patterns** | ARC-1 solutions | Grid transformations |

---

## ARC-AGI-3 Specific Strategy

**Critical Constraint:** ARC-AGI-3 has **NO public training data**

### Online Learning Architecture

FLUX must learn **while playing** the competition:

```
┌─────────────────────────────────────────────────────────────┐
│                  ARC-AGI-3 Competition                       │
│                                                              │
│  Game Start                                                  │
│      │                                                       │
│      ▼                                                       │
│  ┌─────────┐     ┌─────────────┐     ┌─────────────┐        │
│  │ Observe │ ──▶ │ GridToWave  │ ──▶ │ Compare to  │        │
│  │  Frame  │     │   Encode    │     │ Wave Memory │        │
│  └─────────┘     └─────────────┘     └─────────────┘        │
│                                              │               │
│                                              ▼               │
│  ┌─────────────────────────────────────────────────┐        │
│  │ IF similar wave seen before:                     │        │
│  │   → Apply previously successful action           │        │
│  │ ELSE:                                            │        │
│  │   → Explore via spatial memory (curiosity/ice)   │        │
│  └─────────────────────────────────────────────────┘        │
│                        │                                     │
│                        ▼                                     │
│  ┌─────────┐     ┌─────────────┐     ┌─────────────┐        │
│  │ Execute │ ──▶ │ Record      │ ──▶ │ Update      │        │
│  │ Action  │     │ Effect      │     │ Wave Memory │        │
│  └─────────┘     └─────────────┘     └─────────────┘        │
│                                              │               │
│                                              ▼               │
│                                    ┌─────────────────┐      │
│                                    │ Induce Rules    │      │
│                                    │ from Wave Δ     │      │
│                                    └─────────────────┘      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Transfer Learning Strategy

Since ARC-3 has no training data, leverage ARC-1:

```python
# Pre-competition preparation
1. Train GridToWave on ARC-1 (800 tasks)
2. Train WaveToGrid on ARC-1 reconstructions
3. Extract transformation patterns:
   - Rotation, reflection, translation
   - Color mapping, filling, expansion
   - Object detection, boundary tracing
4. Store patterns as "transformation waves" in field

# During competition
5. For each ARC-3 game:
   - Encode frame to wave
   - Find nearest transformation waves
   - Apply most similar transformation
   - Learn from feedback (causal tracker)
```

---

## Training Schedule

### Phase 13: Integrated Vision + Training Gaps

| Week | Task | Data | GPU Hours |
|------|------|------|-----------|
| 1 | SVD Vision Encoder extraction | Qwen2.5-VL | 2 |
| 1 | Vision-wave alignment | COCO 50k | 4 |
| 2 | GridToWave/WaveToGrid | ARC-1 + synthetic | 2 |
| 2 | Decoder enrichment | OpenWebText 1GB | 8 |
| 3 | Causal knowledge | ConceptNet 100k | 4 |
| 3 | Audio encoder | LibriSpeech-100 | 6 |
| 4 | Integration + testing | All | 4 |

**Total:** ~30 GPU hours (~$30 on Kaggle/Colab)

---

## Expected Outcomes

### Flux-Apex-V2 Target Specs

| Component | V1 Current | V2 Target |
|-----------|------------|-----------|
| **Vision** | ✗ External | ✓ Integrated (SVD) |
| **Grid round-trip** | 60% | 95%+ |
| **Audio** | ✗ Stub | ✓ Basic speech |
| **Decoder PPL** | ~80 | <50 |
| **Causal links** | 463 | 100k+ |
| **File size** | 5.79 GB | ~6.5 GB |

### ARC-AGI-3 Impact

| Capability | V1 | V2 |
|------------|-----|-----|
| See game screenshots | ✗ | ✓ |
| Learn from gameplay | ✓ | ✓✓ (faster) |
| Transfer ARC-1 patterns | Partial | Full |
| Rule induction | Wave-based | Wave + causal |

---

## Implementation Priority

```
Priority 1 (Blocks everything):
├── SVD Vision Encoder (enables visual reasoning)
└── GridToWave training (enables grid understanding)

Priority 2 (Major improvement):
├── WaveToGrid training (enables generation)
└── Decoder enrichment (enables fluent output)

Priority 3 (Nice to have):
├── Causal knowledge (general reasoning)
└── Audio encoder (speech capability)
```

---

## Quick Start Commands

```bash
# Create Phase 13 structure
mkdir -p phases/phase13_training_gaps
touch phases/phase13_training_gaps/PHASE_13_SPEC.md
touch phases/phase13_training_gaps/train_vision_svd.py
touch phases/phase13_training_gaps/train_grid_adapters.py
touch phases/phase13_training_gaps/train_decoder.py

# Download ARC-1 training data
wget https://github.com/fchollet/ARC-AGI/raw/master/data/training.tar.gz
tar -xzf training.tar.gz -C data/arc1/

# Download COCO captions (subset)
pip install datasets
python -c "from datasets import load_dataset; load_dataset('HuggingFaceM4/COCO', split='train[:50000]')"
```

---

## Appendix: ARC-AGI-3 Data Constraints

### What We Know
- Competition with hidden test set
- No public training data
- Games are interactive (not static puzzles)
- Multiple game types (movement, click, hybrid)

### What We Can Use
| Source | Legality | Usefulness |
|--------|----------|------------|
| ARC-1 training (400) | ✓ MIT License | High - pattern transfer |
| ARC-1 evaluation (400) | ✓ MIT License | High - pattern transfer |
| Re-ARC (community) | ✓ CC-BY | Medium - augmentation |
| Synthetic grids | ✓ Generated | Medium - regularization |
| Gameplay experience | ✓ Our own | Critical - online learning |

### What We Cannot Use
- ARC-3 hidden test set (competition rules)
- Any leaked ARC-3 data
- Solutions from other competitors

---

*Document maintained by FLUX development team*
*Last updated: March 30, 2026*
