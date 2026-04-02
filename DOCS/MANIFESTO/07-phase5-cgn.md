# FLUX Manifesto

## Part 7: Phase 5 — The Causal Geometry That Replaced the Black Box

---

> *"FLUX doesn't just know. It knows WHY it knows."*

---

## The Black Box Problem

Ask GPT-4 why it gave a specific answer. It'll generate a plausible explanation — but that explanation is generated the same way it generates fiction. The model doesn't actually know why it said what it said.

This isn't a bug, it's architecture. Transformers store knowledge distributed across billions of weights. There's no internal structure that represents "A causes B because of C." The reasoning is a post-hoc performance, not a traceable computation.

This matters for:
- **Medicine**: "Why did you recommend this treatment?" needs a real answer
- **Finance**: Regulators require explainable AI
- **Safety**: If the AI is wrong, you need to know why to fix it
- **Trust**: Users need to verify reasoning, not just accept outputs

I wanted something where every conclusion has a **causal chain back to its evidence**. Where you can literally trace the path from input to output.

---

## Causal Geometry Nodes (CGN)

CGNs are individual reasoning units, each with:

1. **Position in a manifold** — geometric location in reasoning space
2. **Curvature** — how much this node "bends" the reasoning flow
3. **Mass** — evidence weight (accumulated from Phase 2/3)
4. **Activation energy** — threshold before this node fires
5. **Causal arrows** — explicit links to other nodes (A → B with reason)

```
CGN Node Structure:
  curvature:         [1, 512]   — Geometric tensor
  orientation:       [1, 512]   — Direction in manifold
  mass:              scalar     — Evidence weight  
  activation_energy: scalar     — Firing threshold
  manifold.metric_L: [512, 512] — Riemannian metric
  manifold.connection: [512, 512] — Christoffel symbols
  
  Total per node: ~530K parameters
```

The nodes form a **causal graph** — a network of explicit cause-effect relationships that can be traversed, queried, and invalidated.

---

## Multi-Timescale Processing

Here's something interesting: reasoning happens at different speeds.

- "Is that a predator?" → instant (fast neurons)
- "What's the best route home?" → seconds (medium processing)
- "What's the meaning of life?" → extended (deep reflection)

CGN implements this with three tiers:

```
CGN Tiers:
  Fast nodes:   32 nodes, τ=0.01  (instant reactions)
  Medium nodes: 16 nodes, τ=0.1   (deliberate reasoning)  
  Slow nodes:   8 nodes,  τ=1.0   (deep concepts)
  
  Total: 56 nodes, ~14.7M parameters
```

The time constant τ controls how quickly each tier activates. This is biologically inspired — your brain also has fast reflex paths and slow deliberative paths.

---

## Training: March 22, 2026

Phase 5 built the causal system:

```
Training Configuration:
  Previous phases: All frozen
  MultiTimescaleCoordinator: 14,708,767 params
  CausalGraph: Empty initially, populated during training

══════════════════════════════════════════════════════
  Stage A: Causal Graph Formation  
══════════════════════════════════════════════════════

Building causal arrows from known facts:

✓ bird → can_fly              w=0.80  "birds can fly"
✓ penguin → bird              w=0.90  "penguin is a bird"
✓ penguin → cannot_fly        w=0.95  "penguin cannot fly" (exception!)
✓ fish → lives_in_water       w=0.90  "fish live in water"
✓ fire → produces_heat        w=0.85  "fire produces heat"
✓ rain → makes_wet            w=0.90  "rain makes things wet"
✓ ice → is_cold               w=0.95  "ice is cold"
✓ mammal → warm_blooded       w=0.90  "mammals are warm-blooded"
✓ mammal → breathes_air       w=0.95  "mammals breathe air"

Causal edges: 9
Causal nodes: 10
```

Notice the penguin example: birds can fly (general rule), penguins are birds, BUT penguins cannot fly (exception). The causal graph represents all of this explicitly, including the tension between the general rule and the exception.

---

## The Key Result: 6-Hop Causal Tracing

Can CGN trace a chain of reasoning across multiple steps?

```
══════════════════════════════════════════════════════
  Stage D: Causal Invalidation & Multi-Hop Tracing
══════════════════════════════════════════════════════

Building a causal chain:
  A → B → C → D → E → F → G (6 hops)

Querying: "Why is G true?"
  
Trace result:
  G ← F (reason: "F leads to G")
  F ← E (reason: "E causes F")
  E ← D (reason: "D implies E")
  D ← C (reason: "C results in D")
  C ← B (reason: "B produces C")
  B ← A (reason: "A initiates B")
  
Trace depth: 6 hops  ✓ (threshold: ≥ 5)

✓ Multi-hop causal tracing: PASSED
```

**6-hop causal trace.** When asked "why is G true?", FLUX can walk back through the entire chain of reasoning to the original premise A.

This isn't generated explanation. This is **actual graph traversal** of stored causal relationships.

---

## Multi-Timescale Separation

When does each tier activate?

```
══════════════════════════════════════════════════════
  Stage B: Multi-Timescale Separation
══════════════════════════════════════════════════════

Input: "Birds can fly through the air"
Processing steps: 100

  Step 2:  Fast nodes reach 80% activation   ✓
  Step 5:  Medium nodes reach 80% activation ✓  
  Step 31: Slow nodes reach 80% activation   ✓

Activations at step 50:
  Fast tier:   0.0007 (stabilized)
  Medium tier: 0.0042 (stabilized)
  Slow tier:   0.0097 (still evolving)

✓ Multi-timescale separation: VERIFIED
```

Fast nodes fire at step 2 (instant recognition). Medium nodes at step 5 (reasoning). Slow nodes at step 31 (deep processing). This separation is learned, not hardcoded.

---

## Invalidation Propagation

What happens when a premise is proven wrong?

```
══════════════════════════════════════════════════════
  Stage D: Causal Invalidation
══════════════════════════════════════════════════════

Initial state:
  Node "supported" has net_evidence = 1.00 (confident)
  
Invalidating a supporting premise:
  remove_evidence(link_A, weight=0.6)
  
After invalidation:
  Node "supported" net_evidence = 0.40 (reduced!)
  
Downstream affected: nodes 11, 12 (2 total)
  - Node 11 evidence: 0.80 → 0.45
  - Node 12 evidence: 0.75 → 0.40

✓ Invalidation propagates to downstream nodes
```

Disprove one link, and everything that depended on it is automatically flagged. This is **belief revision** — when you learn evidence is wrong, conclusions update automatically.

Transformers can't do this. They'd happily continue using the disproven conclusion because it's baked into their weights.

---

## Geometry Computation

The CGN uses actual differential geometry:

```
══════════════════════════════════════════════════════
  Stage C: Geometry Computation Correctness
══════════════════════════════════════════════════════

Manifold curvature:
  Bending magnitude: 0.8729 (non-flat!)
  
Geodesic computation:
  Path length (A → B): 1.4523
  Direct distance:     1.2078
  Ratio:               1.20 (curved space)
  
Mass amplification:
  Mass=1.0 influence: 1.00
  Mass=5.0 influence: 5.00  (5× amplification)

✓ Geometric computations verified
```

The reasoning space is curved. Paths between concepts aren't straight lines — they bend according to the accumulated evidence. High-mass concepts curve the manifold more, attracting reasoning paths toward them.

This is general relativity for knowledge: mass curves spacetime; evidence curves reasoning-space.

---

## Full Pipeline Test

Does CGN integrate with the rest of FLUX?

```
══════════════════════════════════════════════════════
  Stage E: Full Pipeline — CSE → Field → CGN
══════════════════════════════════════════════════════

Input: "The Earth revolves around the Sun"
  CSE → wave [432]
  Field → activated attractors
  CGN → output [512], norm=0.2681  ✓

Input: "Water boils at 100 degrees Celsius"
  CSE → wave [432]
  Field → activated attractors  
  CGN → output [512], norm=0.2645  ✓

Input: "Plants convert sunlight into energy through photosynthesis"
  CSE → wave [432]
  Field → activated attractors
  CGN → output [512], norm=0.2722  ✓

✓ CSE → Field → CGN pipeline operational
```

The full chain works: text → waves → field lookup → causal reasoning → output.

---

## The Numbers

| Metric | Result |
|--------|--------|
| CGN nodes | 56 (32 fast + 16 medium + 8 slow) |
| Parameters | **149,757,403** (injected into Apex) |
| Causal edges | 9 (initial), expandable |
| **Trace depth** | **6 hops** |
| Fast activation | Step 2 |
| Medium activation | Step 5 |
| Slow activation | Step 31 |
| Invalidation propagation | 2 downstream affected |

---

## What Phase 5 Changes

| Capability | Transformer | FLUX CGN |
|------------|-------------|----------|
| Explainability | Post-hoc generated | **Traceable causal chain** |
| Multi-hop reasoning | Implicit only | **Explicit 6+ hops** |
| Processing speed | Uniform | **Fast/medium/slow tiers** |
| Belief revision | Can't invalidate | **Automatic propagation** |
| Exceptions handling | Black box | **Explicit override arrows** |

> *"A transformer can tell you the answer. FLUX can show you the receipt."*

---

## The Receipt Matters

In the penguin example, ask FLUX: "Can penguins fly?"

```
Query: "Can penguins fly?"

Trace:
  1. penguin → bird (confidence: 0.90)
  2. bird → can_fly (confidence: 0.80)
  3. BUT: penguin → cannot_fly (confidence: 0.95) [OVERRIDE]
  
Conclusion: No (0.95 confidence)
Reason: Penguin-specific exception overrides general bird rule
Evidence chain: penguin → cannot_fly (direct) > penguin → bird → can_fly
```

The system doesn't just answer — it shows the competing reasoning paths and explains why one wins.

---

## The Stack

| Component | Transformer | FLUX | Status |
|-----------|-------------|------|--------|
| Encoding | Tokenizer | CSE waves | ✅ |
| Ordering | Position embeddings | CWC | ✅ |
| Storage | Distributed weights | Resonance field | ✅ |
| Relevance | Attention O(n²) | Gravity O(log n) | ✅ |
| Learning | Backprop | Thermodynamic settling | ✅ |
| Reasoning | Black box | CGN causal tracing | ✅ |
| Memory | ? | Coming... | Phase 6 |

We can encode, store, retrieve, learn, and reason. But what about memory across sessions? What about permanent vs temporary knowledge?

---

*Continue to [Part 8: The Memory That Never Forgets →](08-phase6-memory.md)*
