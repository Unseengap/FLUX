# FLUX Manifesto

## Part 3: Phase 1.5 — Causal Wave Chaining

---

> *"Waves don't just encode. They reason about order, detect contradiction, and trace implications."*

---

## The Missing Link

After Phase 1, I had waves that encoded meaning. But waves alone are passive — they represent, they don't reason.

I needed something that could:
1. Detect when order matters ("A then B" vs "B then A")
2. Recognize contradictions ("X is true" vs "X is false")
3. Trace implications ("If A causes B, and B causes C, then A causes C")

This is the gap between perception and reasoning. The CSE perceives. Causal Wave Chaining **reasons**.

---

## Causal Wave Chaining (CWC)

The CWC adds three capabilities to waves:

**1. Forward/Backward Heads**
Given two waves, predict which comes first. This captures temporal and causal ordering.

**2. Tension Detection**
Measure the "tension" between waves — how much they oppose or contradict each other.

**3. Chain Encoding**
Encode sequences of implications: A→B→C... as a single chain representation.

```
Architecture:
  forward_head:     132K params
  backward_head:    132K params  
  tension_detector: 173K params
  chain_encoder:    133K params
  ─────────────────────────────
  Total:            570,162 params
```

---

## Training: March 21, 2026

CWC trained on top of the frozen Phase 1 CSE:

```
Training Configuration:
  CSE:       Frozen (1,337,264 params from Phase 1)
  CWC:       570,162 params (trainable)
  Steps:     3,000
  Data:      WikiText-2 + synthetic implication pairs
  Duration:  28.6 minutes
  
  Implication store pre-populated: 20 arrows
```

The training focused on two tasks:
1. Coherence ordering — is this sequence in the right order?
2. Contradiction detection — does statement B contradict statement A?

```
── Training Progress ──
Step  500: coherence=0.82, tension_accuracy=0.71
Step 1000: coherence=0.91, tension_accuracy=0.84
Step 1500: coherence=0.96, tension_accuracy=0.93
Step 2000: coherence=0.98, tension_accuracy=0.97
Step 3000: coherence=0.9998, tension_accuracy=0.99

── Training Complete ──
Duration: 28.6 min
Best coherence gap: 0.0000
```

---

## Test 1: Order Sensitivity

Can CWC detect when statements are in the wrong order?

```
============================================================
FLUX Phase 1.5 Test 1: Order Sensitivity (Tension-Based)
============================================================

Test pairs (correct vs shuffled):

1. "The sun rises in the east" → "It sets in the west"
   Correct order tension:  0.12  ✓
   Reversed order tension: 0.67  ✓ (higher = more wrong)

2. "First mix the ingredients" → "Then bake for 30 minutes"
   Correct order tension:  0.08  ✓
   Reversed order tension: 0.71  ✓
   
3. "She graduated from college" → "She started her career"
   Correct order tension:  0.15  ✓
   Reversed order tension: 0.58  ✓

Order Sensitivity Accuracy: 93%
  ✓ ORDER SENSITIVITY TEST PASSED
```

**93% accuracy** at detecting wrong order. The CWC understands that "bake then mix" doesn't make sense.

---

## Test 2: Contradiction Detection

This is the critical test. Can CWC detect when statements contradict each other?

```
============================================================
FLUX Phase 1.5 Test 2: Contradiction Detection
============================================================

Contradictory pairs:

1. "The door is open" vs "The door is closed"
   Tension: 0.89  → CONTRADICTION ✓

2. "Water boils at 100°C" vs "Water freezes at 100°C"  
   Tension: 0.92  → CONTRADICTION ✓

3. "The cat is alive" vs "The cat is dead"
   Tension: 0.87  → CONTRADICTION ✓

4. "Paris is in France" vs "Paris is in Germany"
   Tension: 0.84  → CONTRADICTION ✓

5. "He is married" vs "He is single"
   Tension: 0.91  → CONTRADICTION ✓

Non-contradictory pairs (control):

6. "The sky is blue" vs "Grass is green"
   Tension: 0.21  → NO CONTRADICTION ✓

7. "I like coffee" vs "She likes tea"
   Tension: 0.18  → NO CONTRADICTION ✓

8. "It's raining" vs "I need an umbrella"
   Tension: 0.11  → NO CONTRADICTION ✓ (related but not contradictory)

Contradiction Detection: 20/20 correct
  ✓ CONTRADICTION DETECTION TEST PASSED
```

**20/20 on contradiction detection.** Perfect score. The CWC can tell the difference between:
- Statements that contradict (open/closed)
- Statements that are merely different (coffee/tea)
- Statements that are related but consistent (raining/umbrella)

---

## Why This Matters

This isn't just academic. Contradiction detection is the foundation for:

**1. Belief Revision**
When new information contradicts old information, we need to detect that conflict. CWC provides the detection mechanism.

**2. Logical Reasoning**
"If X then Y, if Y then Z, therefore if X then Z" — this requires tracking implications. CWC encodes chains.

**3. Inconsistency Prevention**
LLMs regularly contradict themselves within the same conversation. CWC would catch "I said X earlier but now I'm saying not-X."

**4. Debugging Causal Graphs**
When a conclusion depends on contradictory premises, CWC flags the tension.

---

## The Implication Store

CWC also maintains an **implication store** — a persistent record of causal arrows:

```
Implication Store (20 arrows):
  bird → can_fly         (weight: 0.80)
  penguin → cannot_fly   (weight: 0.95)
  fish → lives_in_water  (weight: 0.90)
  mammal → breathes_air  (weight: 0.95)
  ...
```

These arrows are used later by the Causal Geometry Nodes (Phase 5) for multi-hop reasoning. CWC creates the arrows; CGN uses them.

---

## The Numbers

| Metric | Result |
|--------|--------|
| Parameters | 570,162 |
| Training time | 28.6 minutes |
| Order sensitivity | **93%** |
| Contradiction detection | **20/20 (100%)** |
| Implication arrows | 20 |

---

## What Phase 1.5 Adds

| Capability | Before | After |
|------------|--------|-------|
| Order sensitivity | ✗ | ✓ (93%) |
| Contradiction detection | ✗ | ✓ (100%) |
| Implication chaining | ✗ | ✓ |
| Tension measurement | ✗ | ✓ |

The CSE + CWC together give us:
- **Encoding** (any text → 432D wave)
- **Ordering** (detect correct sequence)
- **Contradiction** (detect conflicts)
- **Implication** (A→B→C chains)

Now we need somewhere to store knowledge that accumulates over time.

> *"A transformer can hold contradictory beliefs in the same breath. FLUX feels the tension."*

---

*Continue to [Part 4: The Resonance Field →](04-phase2-field.md)*
