# FLUX Technical Specification
## Version 0.1 — Foundation Document

> This document is the source of truth for all implementation decisions.
> Copilot should have this file open at all times during development.

---

## 1. Architecture Overview

FLUX is a field-based AI architecture with five novel components that replace all traditional neural network primitives:

```
INPUT (raw text / bytes / signal)
    │
    ▼
┌─────────────────────────────────┐
│  CONTINUOUS SEMANTIC ENCODER    │  Phase 1
│  Replaces: Tokenizer + Embeddings│
│  Output: Semantic Wave Tensor   │
└──────────────┬──────────────────┘
               │
    ▼
┌─────────────────────────────────┐
│  CAUSAL GEOMETRY NODES          │  Phase 5
│  Replaces: Neurons / Layers     │
│  Structure: Manifold patches    │
└──────────────┬──────────────────┘
               │
    ▼
┌─────────────────────────────────┐
│  GRAVITATIONAL RELEVANCE        │  Phase 3
│  Replaces: Attention mechanism  │
│  Cost: O(log n) not O(n²)       │
└──────────────┬──────────────────┘
               │
    ▼
┌─────────────────────────────────┐
│  RESONANCE FIELD CORE           │  Phase 2
│  Replaces: Weight matrices      │
│  Type: Dynamic field tensor     │
└──────────────┬──────────────────┘
               │
    ▼
┌─────────────────────────────────┐
│  THERMODYNAMIC LEARNING         │  Phase 4
│  Replaces: Backpropagation      │
│  Method: Local energy settling  │
└──────────────┬──────────────────┘
               │
    ▼
┌─────────────────────────────────┐
│  THREE-TIER MEMORY              │  Phase 6
│  Working / Episodic / Semantic  │
│  No catastrophic forgetting     │
└──────────────┬──────────────────┘
               │
    ▼
OUTPUT (text / embeddings / actions)
```

---

## 2. Component Specifications

---

### 2.1 Continuous Semantic Encoder (CSE)

**Purpose:** Convert raw input into continuous semantic wave representations without discrete tokenization.

**Replaces:** BPE tokenizer, word embeddings, positional encodings

**Input:** Raw UTF-8 bytes or characters (no vocabulary required)

**Output:** `SemanticWave` tensor of shape `[seq_len, wave_dim]`

#### Wave Dimensions
Each position in the sequence is represented by a wave with:

```python
wave_dim = {
    'phonetic':   64,   # Sound/character pattern encoding
    'syntactic':  64,   # Grammatical tension/structure signal
    'semantic':   256,  # Meaning-space coordinates
    'temporal':   32,   # Position in time/sequence signal
    'intensity':  16,   # Emphasis / importance signal
}
# Total wave_dim = 432
```

#### Encoding Process
```
Raw input → byte-level sliding window (window=8, stride=1)
          → learned convolutional filters (capture n-gram patterns)
          → continuous projection into 432-dim wave space
          → interference pass (nearby waves interact)
          → normalize to unit wave amplitude
```

#### Key Properties
- **No vocabulary:** Any byte sequence is valid input
- **Constructive interference:** Similar meaning → waves add together
- **Destructive interference:** Contradictory meaning → waves cancel
- **Language agnostic:** English, Chinese, code, math all use same encoder
- **Analogy:** Language treated as physics, not as lookup table

#### Interference Function
```python
def wave_interference(w1: Tensor, w2: Tensor, distance: int) -> Tensor:
    """
    Two nearby semantic waves interfere based on distance.
    Close waves interact strongly, distant waves weakly.
    Phase alignment determines constructive vs destructive.
    """
    phase_diff = compute_phase_difference(w1, w2)
    amplitude = torch.cos(phase_diff) * decay_function(distance)
    return w1 + amplitude * w2
```

#### Save Format (Phase 1 Checkpoint)
```python
{
    'encoder_filters': Tensor,      # Convolutional filters
    'wave_projections': Tensor,     # Projection matrices per dimension
    'interference_weights': Tensor, # Learned interference patterns
    'vocab_free': True,             # Flag confirming no vocabulary
    'wave_dim': 432,
    'phase': 1
}
```

---

### 2.2 Resonance Field Core (RFC)

**Purpose:** Store all learned knowledge as a dynamic field that reshapes itself continuously rather than as static weight matrices.

**Replaces:** All weight matrices, parameter tensors

**Structure:** A high-dimensional energy landscape stored as a field state tensor

#### Field Representation
```python
field_shape = {
    'spatial_dims': [H, W, D],    # 3D spatial layout (e.g. 64x64x64)
    'feature_dim':  512,           # Per-location feature vector
    'energy':       1,             # Scalar energy at each location
    'mass':         1,             # Evidence mass (grows with learning)
    'velocity':     512,           # Current rate of change
}
# Field tensor: [H, W, D, 512 + 1 + 1 + 512] = [64, 64, 64, 1026]
```

#### Field Dynamics
The field evolves according to local energy minimization:

```
New input arrives
    → Encoded as SemanticWave
    → Wave perturbs field at nearest spatial location
    → Perturbation propagates outward via wave equation
    → Field settles to local minimum energy
    → Settled state IS the output AND the memory
```

#### Attractor Stability
Learned concepts form stable attractors — energy minima in the field:
```
Strong attractor  = deep energy well = well-learned concept
Weak attractor    = shallow well     = recently seen concept
Destroyed         = only if new info directly contradicts AND
                    is much stronger evidence
```

This means:
- New facts create new attractors locally
- Old attractors persist unless directly contradicted
- Catastrophic forgetting is physically impossible by design

#### Field Update Rule (No Backprop)
```python
def update_field(field: Tensor, perturbation: Tensor, 
                 location: Tensor) -> Tensor:
    """
    Local field update — only affects neighborhood of location.
    No global gradient computation required.
    """
    radius = compute_influence_radius(perturbation.energy)
    neighborhood = get_neighborhood(field, location, radius)
    delta = compute_local_gradient(neighborhood, perturbation)
    field[neighborhood] -= learning_rate * delta
    field = normalize_energy(field)  # Prevent energy explosion
    return field
```

#### Save Format (.flx file)
```python
# FLUX model file format
{
    'field_state': Tensor,          # Full field tensor [H,W,D,F]
    'attractor_catalog': List,      # All stable attractors
    'field_energy': float,          # Total field energy
    'field_topology': Dict,         # Compressed topology description
    'causal_arrows': Dict,          # Why each attractor exists
    'phase': int,                   # Which phase created this
    'timestamp': str,               # When last updated
    'learning_steps': int           # Total updates applied
}
```

---

### 2.3 Gravitational Relevance (GR)

**Purpose:** Replace attention with a physics-inspired relevance mechanism that costs O(log n) instead of O(n²).

**Replaces:** Multi-head self-attention

#### Core Concept
Concepts with more accumulated evidence become "heavier" — they exert more gravitational pull on incoming queries. New input naturally falls toward the most relevant heavy concepts.

```python
mass(concept) = base_mass + evidence_count * evidence_weight
              # Grows as concept is reinforced by data
              # More evidence = stronger pull on related queries

gravity(query, concept) = mass(concept) / distance(query, concept)²
                        # Inverse square law — same as physics
```

#### O(log n) Implementation
Uses spatial partitioning (KD-tree or BVH) to avoid all-pairs comparison:

```
Traditional attention:  Compare query to ALL n keys → O(n²)
Gravitational:         Query falls to nearest heavy attractors
                       Spatial tree finds neighbors → O(log n)
```

#### Negative Mass (Contradiction)
Concepts that have been contradicted by evidence develop negative mass — they actively repel related queries:
```python
if evidence_contradicts(new_evidence, concept):
    concept.mass -= contradiction_weight
    # Negative mass = repulsion = model avoids wrong answers
```

#### Multi-Scale Gravity
```python
# Fast reactions: surface-level pattern attractors (small mass, short range)
# Slow reactions: deep concept attractors (large mass, long range)
# Both operate simultaneously — no hard separation
```

---

### 2.4 Thermodynamic Learning (TL)

**Purpose:** Replace backpropagation with a local energy-settling process that learns in real-time without a separate training phase.

**Replaces:** Backpropagation, gradient descent, training loop

#### Core Principle
The system is always trying to reach minimum energy. When input arrives, it creates a perturbation. The system settling into a new minimum IS both the forward pass (producing output) and learning (updating the field) simultaneously.

```
Input → Perturbation → Field Settles → Output extracted from settled state
                     ↑
                     This settling = learning
                     No separate backward pass needed
```

#### Learning Without Epochs
```python
# Traditional:
for epoch in range(N):
    for batch in dataloader:
        loss = forward(batch)
        loss.backward()     # Global gradient
        optimizer.step()    # Update all weights

# FLUX Thermodynamic:
for sample in stream:       # No epochs — continuous stream
    perturb_field(sample)   # Local perturbation
    settle_field()          # Energy minimization
    # Done — both inference and learning happened
```

#### Temperature Annealing
Controls how much the field can change:
```python
# High temperature (early): Field is fluid, learns quickly
# Low temperature (mature): Field is stable, resists change
# Dynamic temperature: Adjusts based on prediction error
#   High error → increase temperature locally
#   Low error  → decrease temperature locally
```

#### GPU Implementation
Thermodynamic settling maps to parallel iterative solvers:
```python
# Each GPU thread handles one field region
# No global synchronization until output extraction
# Perfect for CUDA parallel execution
# Same hardware as transformers, different math
```

---

### 2.5 Causal Geometry Nodes (CGN)

**Purpose:** Replace neurons with geometric manifold patches that store both WHAT they know and WHY they know it.

**Replaces:** Neurons, layers, activation functions

#### Node Structure
Each CGN is not a scalar but a geometric patch:
```python
class CausalGeometryNode:
    curvature:   Tensor  # How strongly it bends incoming signals
    orientation: Tensor  # What directions it is sensitive to
    radius:      float   # Breadth of its influence
    causal_why:  Tensor  # Pointer to what caused this node to form
    time_const:  float   # How fast this node reacts (fast=surface, slow=deep)
    mass:        float   # Evidence weight (same as GR mass)
```

#### Why "Causal"
Standard neurons store: "input X maps to output Y"
CGN stores: "input X maps to output Y **because** of causal relationship Z"

This enables:
- Genuine causal reasoning (not just correlation)
- Tracing why a conclusion was reached
- Invalidating conclusions when their cause is disproven

#### Geometry as Computation
Instead of `output = activation(W @ input)`:
```
Signal enters the manifold
    → CGN curvature bends the signal path
    → Signal naturally flows toward compatible attractors
    → Output emerges from where signal comes to rest
    → No explicit multiplication — geometry IS the computation
```

#### Multi-Timescale Operation
```python
# Fast nodes (time_const ≈ 0.01):  Surface syntax patterns
# Medium nodes (time_const ≈ 0.1):  Semantic relationships
# Slow nodes (time_const ≈ 1.0):   Deep conceptual abstractions
# All operating simultaneously — no layer separation needed
```

---

### 2.6 Three-Tier Memory System

**Purpose:** Separate working memory, episodic memory, and semantic memory — mirroring human cognitive architecture.

**Replaces:** Context window as sole memory mechanism

#### Tier 1: Working Memory
```python
capacity:    Current field state window (rolling)
persistence: Session only — cleared on reload  
speed:       Immediate
tech:        Active field region + attention over recent perturbations
analogy:     Your desk — what you're looking at right now
```

#### Tier 2: Episodic Memory
```python
capacity:    Unlimited (external vector store)
persistence: Permanent until explicitly deleted
speed:       Fast lookup (FAISS / similar)
write:       One-shot — encode + store, no training needed
read:        Semantic similarity search → inject into working memory
analogy:     Your diary — specific events and facts
```

Episodic memory write (one-shot):
```python
def write_episodic(fact: str, context: Dict) -> None:
    wave = encode_to_wave(fact)           # CSE encoding
    vector = wave.to_retrieval_vector()   # Compress for storage
    metadata = {
        'timestamp': now(),
        'confidence': compute_confidence(wave),
        'causal_source': context['source'],
        'decay_rate': 0.0  # Never decays unless contradicted
    }
    episodic_store.add(vector, metadata)
```

#### Tier 3: Semantic Memory
```python
capacity:    Entire field state (compressed)
persistence: Permanent — survives reload
speed:       Always on (it IS the field)
update:      Only during offline consolidation
protection:  Energy barriers around mature attractors
analogy:     Your deep skills — riding a bike, reading, reasoning
```

#### Memory Interaction Loop
```
Every forward pass:
1. Working memory processes current input
2. GR queries episodic store for relevant past facts
3. Retrieved facts injected into working field region
4. Semantic field provides background knowledge always

Periodically (offline consolidation):
1. Review episodic memories by frequency of retrieval
2. Frequently retrieved → distill into semantic field
3. Update field with thermodynamic settling (low temperature)
4. High-confidence episodic facts → upgrade to semantic
```

---

## 3. File Formats

### 3.1 Phase Checkpoint (.phase.pt)
Standard PyTorch checkpoint containing phase-specific components:
```python
torch.save({
    'phase': N,
    'component': 'component_name',
    'state': component_state_dict,
    'metrics': phase_metrics,
    'config': phase_config,
    'timestamp': datetime.now().isoformat()
}, f'checkpoints/phase{N}.phase.pt')
```

### 3.2 FLUX Model File (.flx)
Full model snapshot loadable for inference and continued learning:
```python
{
    'format': 'FLUX',
    'version': '0.1',
    'field_state': field_tensor,
    'cse_state': encoder_state,
    'episodic_index': faiss_index_bytes,
    'attractor_catalog': attractors,
    'causal_graph': causal_arrows,
    'config': full_config,
    'learning_steps': int,
    'can_continue_learning': True  # Always true — no static mode
}
```

---

## 4. Evaluation Metrics

### Per-Phase Metrics
Each phase adds to the metric stack:

| Phase | Primary Metric | Secondary Metric |
|---|---|---|
| 1 (CSE) | Reconstruction loss | Interference quality score |
| 2 (RFC) | Field stability | Attractor formation rate |
| 3 (GR) | Retrieval precision@k | Speed vs attention baseline |
| 4 (TL) | Loss convergence rate | Forgetting score (lower=better) |
| 5 (CGN) | Causal trace accuracy | Multi-timescale separation |
| 6 (Memory) | One-shot retention | Cross-session continuity |
| 7 (Full) | Perplexity | Generation quality |
| 8 (Scale) | vs GPT-2 perplexity | Continual learning retention |

### Forgetting Score
```python
def forgetting_score(model, task_A_data, task_B_data) -> float:
    """
    Train on A. Train on B. Test on A again.
    Score = (accuracy_after_B - accuracy_before_B) / accuracy_before_B
    Perfect score = 0.0 (no forgetting)
    Transformer baseline typically = -0.3 to -0.8 (30-80% degradation)
    """
```

---

## 5. Configuration

### Master Config
```python
FLUX_CONFIG = {
    # CSE
    'wave_dim': 432,
    'byte_window': 8,
    'byte_stride': 1,
    'interference_radius': 4,
    
    # Field
    'field_dims': [64, 64, 64],
    'field_features': 512,
    'field_update_radius': 8,
    
    # GR
    'gravity_k_neighbors': 32,
    'base_mass': 1.0,
    'mass_growth_rate': 0.01,
    
    # TL
    'initial_temperature': 1.0,
    'min_temperature': 0.01,
    'temperature_decay': 0.9999,
    
    # CGN
    'num_nodes': 4096,
    'fast_time_const': 0.01,
    'slow_time_const': 1.0,
    
    # Memory
    'working_memory_window': 512,
    'episodic_dim': 256,
    'consolidation_interval': 10000,
    
    # Training
    'device': 'cuda',
    'dtype': torch.float32,
    'checkpoint_dir': 'checkpoints/',
}
```

---

## 6. Dependencies

```txt
# Core
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0

# Field operations
torch-geometric>=2.3.0    # Graph/manifold operations

# Episodic memory
faiss-gpu>=1.7.4          # Vector similarity search (GPU)

# Evaluation
datasets>=2.14.0          # HuggingFace datasets (OpenWebText, etc.)
evaluate>=0.4.0           # Metrics

# Visualization
matplotlib>=3.7.0
tensorboard>=2.13.0

# Utils
tqdm>=4.65.0
pyyaml>=6.0
rich>=13.0.0              # Pretty terminal output
```
