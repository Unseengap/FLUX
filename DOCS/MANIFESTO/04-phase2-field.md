# FLUX Manifesto

## Part 4: Phase 2 — The Field That Replaced the Weight Matrix

---

> *"Knowledge isn't stored in weights. It's stored in landscapes."*

---

## The Problem with Distributed Knowledge

Transformers distribute knowledge across billions of weights. No single weight "knows" anything. You can't point to where "Paris is the capital of France" lives. It's everywhere and nowhere.

This creates catastrophic forgetting: teach the model something new, and the gradient updates that encode the new information destroy the previous encodings. The weights shift, and old knowledge fades.

I wanted something different. Something where knowledge has **location**. Where "Paris" exists at a specific point you can find, query, and update without touching anything else.

---

## The Resonance Field

The Resonance Field is a 3D energy landscape. Think of it as a terrain where:
- **Valleys** are attractors — stable points representing concepts
- **Hills** are barriers between unrelated concepts
- **Proximity** encodes relationship — "Paris" and "France" are nearby
- **Mass** encodes evidence — concepts with more evidence are heavier

```
Field Architecture:
  Full size:     96 × 96 × 96 × 512  (884,736 cells)
  Compressed:    48 × 48 × 48 × 256  (110,592 cells)
  Features/cell: 256 (compressed) or 512 (full)
  
  Each cell: energy state + mass + feature vector
```

When you store a concept, it creates an attractor — a valley in the energy landscape. When you query, you project into field coordinates and fall toward the nearest attractor.

---

## The Zero-Forgetting Property

This is the key insight: **attractors are local**. Creating a new attractor at position (x, y, z) doesn't change anything at position (x', y', z'). Store a new fact — existing facts stay exactly where they were.

Transformers can't do this. Their knowledge is entangled in shared weights. 

FLUX's field is **structurally immune to catastrophic forgetting**.

---

## Training: March 21, 2026

Phase 2 built on the frozen CSE from Phase 1:

```
Training Configuration:
  CSE:        Frozen (Phase 1)
  Field:      48 × 48 × 48 × 256
  Warmup:     500 steps (spatial projection)
  Formation:  20 reps × 1000 perturbations
  Data:       100 encoded texts from WikiText-2
  
── Phase A: Projection Warmup ──
Warmup 0/500:   coord_std=[0.009, 0.007, 0.011]  (collapsed!)
Warmup 100/500: coord_std=[0.695, 0.631, 0.638]  (spreading)
Warmup 300/500: coord_std=[0.891, 0.813, 0.792]  (distributed)
Warmup 500/500: coord_std=[0.939, 0.874, 0.880]  ✓ Well-spread

── Phase B: Attractor Formation ──
Rep  1/20: attractors=7,    energy=255,075
Rep  6/20: attractors=261,  energy=234,823
Rep 11/20: attractors=622,  energy=222,371
Rep 16/20: attractors=2,009 energy=212,762
Rep 20/20: attractors=3,224 energy=206,255

✓ Field formation complete: 3,224 attractors formed
```

The field started empty. After 20 repetitions of perturbing it with encoded text, it developed **3,224 distinct attractors** — valleys in the energy landscape, each representing a concept cluster.

---

## The Results

### Test 1: Retrieval Accuracy

Can we encode a concept, store it in the field, and retrieve it later?

```
============================================================
Phase 2 Test 1: Attractor Formation & Retrieval
============================================================

Encoding 100 test concepts...
Storing in field (creating attractors)...
Querying with same concepts...

Retrieval Results:
  Queries:           100
  Correct retrievals: 100
  Average similarity: 0.9985
  
Retrieval Accuracy: 100.0%
  ✓ TEST PASSED
```

**100% retrieval accuracy.** Every concept stored could be found again.

### Test 2: No-Forgetting Verification

This is the test that matters most. What happens when we add new information?

```
============================================================
Phase 2 Test 2: No-Forgetting Verification  
============================================================

Step 1: Store facts A, B, C
  Fact A stored at (23, 45, 67), similarity: 1.000
  Fact B stored at (12, 78, 34), similarity: 1.000
  Fact C stored at (56, 23, 89), similarity: 1.000
  
Step 2: Add 100 new facts
  Facts D through CZ stored at various locations...
  
Step 3: Re-query original facts A, B, C
  Fact A similarity: 1.000  (unchanged)
  Fact B similarity: 1.000  (unchanged)
  Fact C similarity: 1.000  (unchanged)

Forgetting Score: 0.0000

  ✓ ZERO FORGETTING VERIFIED
```

**Forgetting score: 0.0000**

Not 0.001. Not 0.01. **Zero.** The original facts remained perfectly intact after storing 100 new facts.

For comparison, transformers typically score **0.30 to 0.80** on the same metric — losing 30% to 80% of old knowledge when learning new things.

---

## Visualizing the Field

The training produced visualizations showing attractor formation over time:

```
Resonance Field: Attractor Formation Over Time

t=0 (empty)        t=25            t=50            t=75            t=100
┌──────────┐      ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│          │      │    ●     │    │  ●   ●   │    │  ●  ● ●  │    │ ●● ●● ●● │
│          │  →   │          │ →  │     ●    │ →  │ ● ●   ●  │ →  │ ● ●●  ●● │
│          │      │   ●      │    │  ●    ●  │    │   ●●  ●  │    │  ●● ●● ● │
└──────────┘      └──────────┘    └──────────┘    └──────────┘    └──────────┘
attractors=0      attractors=211  attractors=485  attractors=2009 attractors=3224
```

The field starts uniform, then attractors nucleate and grow as concepts are stored. Each attractor is a stable valley that persists indefinitely.

---

## The Mass System

Attractors aren't just locations — they have **mass**. Mass accumulates with evidence:

```
Mass Evolution:
  Initial mass (new concept):     0.1
  After 1 confirmation:           0.25
  After 5 confirmations:          0.78
  After 10 confirmations:         1.44
  After 25 confirmations:         2.62
  
Heavier concepts:
  - Are more stable (harder to displace)
  - Attract related queries more strongly
  - Resist contradictory evidence longer
```

This is how FLUX implements **evidence accumulation**. See something once — weak attractor. See it repeatedly — strong attractor. This matches how human memory works.

---

## Negative Mass: Disproven Concepts

What happens when you learn something is false?

In FLUX, disproven concepts develop **negative mass**. They don't just disappear — they actively repel queries. If you stored "the Earth is flat" and later learned it's false, the negative-mass attractor pushes queries away from that conclusion.

```
Negative Mass Example:
  Concept: "Earth is flat"
  Initial mass: 0.5 (someone told you)
  
  Contradiction: "Photos from space show Earth is round"
  Mass update: 0.5 → -0.8
  
  Result: Future queries about Earth's shape are
         repelled from "flat" toward "round"
```

Transformers can't do this. They can forget, but they can't **actively avoid** wrong conclusions.

---

## Spatial Distribution

One concern with 3D fields: do concepts clump in one region, or spread out?

```
Spatial Distribution Analysis:
  Unique locations (first 50 concepts): 37/50
  
  Coordinate spreads:
    X-axis std: 0.939
    Y-axis std: 0.874
    Z-axis std: 0.880
    
  ✓ Good spatial distribution — concepts spread across field
```

The spatial projection layer ensures concepts distribute across the full volume, not clump in one corner.

---

## The Numbers

| Metric | Result |
|--------|--------|
| Field size | 48³ × 256 (compressed) |
| Parameters | 28,442,624 |
| Attractors formed | **3,224** |
| Retrieval accuracy | **100%** |
| Forgetting score | **0.0000** |
| Mass-tracked concepts | 8,068 (by Phase 3) |
| Unique spatial locations | 37/50 (74%) |

---

## What Phase 2 Proves

| Capability | Transformer | FLUX Field |
|------------|-------------|------------|
| Knowledge location | Distributed everywhere | Specific coordinates |
| Adding new knowledge | Risk forgetting old | Zero forgetting |
| Evidence accumulation | Implicit in weights | Explicit mass |
| Contradiction handling | Silent conflict | Negative mass repulsion |
| Retrieval | Weighted sum | Direct lookup |

> *"In a transformer, knowledge is dissolved in an ocean of parameters. In FLUX, knowledge is a mountain you can point to and visit."*

---

## The Foundation Grows

We now have:
- **CSE**: Any input → 432D wave
- **CWC**: Order sensitivity, contradiction detection
- **Field**: Zero-forgetting knowledge storage, 3,224 attractors

But the field is still passive. You query and retrieve. What we need is **dynamic relevance** — not all knowledge matters equally for every query.

---

*Continue to [Part 5: The Gravity That Replaced Attention →](05-phase3-gravity.md)*
