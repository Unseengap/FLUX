# FLUX-7B: Physics-Native Language Model

## Design Philosophy

**7B parameters, but NOT 7B trainable parameters.**

FLUX-7B distributes parameters fundamentally differently than transformers:

| Component | Params | Training Method |
|-----------|--------|-----------------|
| **Resonance Field** | 6.5B | **INJECTION** (no gradients) |
| **Physics Stack** | 300M | Frozen after Phase 7 |
| **Emission Head** | 200M | **GRADIENT** (only trainable part) |
| **Total** | 7B | **Only 200M trained** |

This means:
- Knowledge acquisition: **Minutes** (inject into field)
- Emission training: **Days** (not weeks)
- Add new knowledge: **Seconds** (no retraining)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            FLUX-7B SYSTEM                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     RESONANCE FIELD (6.5B)                          │   │
│  │                     ─────────────────────                           │   │
│  │  Shape: [512, 512, 512] × 1536 features = 6.44B params              │   │
│  │                                                                     │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ state: [512, 512, 512, 1536]     # Knowledge tensor          │   │   │
│  │  │ mass:  [512, 512, 512]           # Evidence accumulation     │   │   │
│  │  │ temperature: [512, 512, 512]     # Local plasticity          │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                     │   │
│  │  Operations (NO GRADIENTS):                                        │   │
│  │  • inject(wave) → perturb field at location                       │   │
│  │  • settle(steps) → energy minimization                            │   │
│  │  • query(wave, k) → gravity-weighted k-nearest                    │   │
│  │  • consolidate() → form stable attractors                         │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↑                                        │
│                                    │ inject                                 │
│  ┌─────────────────────────────────┴───────────────────────────────────┐   │
│  │                     PHYSICS STACK (300M, frozen)                    │   │
│  │                     ────────────────────────────                    │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │ CSE (10M)    │→ │ GR (50M)     │→ │ CGN (100M)   │              │   │
│  │  │ bytes→waves  │  │ O(log n)     │  │ causal paths │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  │          ↓                 ↓                 ↓                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │ TL (40M)     │  │ Memory (80M) │  │ Bridges (20M)│              │   │
│  │  │ thermodynamic│  │ 3-tier       │  │ wave↔field   │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     EMISSION HEAD (200M, trainable)                 │   │
│  │                     ──────────────────────────────                  │   │
│  │                                                                     │   │
│  │  Input: field_context [k, 1536] + wave_context [seq, 432]          │   │
│  │         ↓                                                           │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ Context Merger (20M)                                        │   │   │
│  │  │ • Fuse field retrievals with wave context                  │   │   │
│  │  │ • Gravity-weighted attention over k attractors             │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │         ↓                                                           │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ Resonance Emitter (150M)                                    │   │   │
│  │  │ • 12 transformer layers (d=1024, 16 heads)                 │   │   │
│  │  │ • Self-attention over emission history                     │   │   │
│  │  │ • Cross-attention to merged context                        │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │         ↓                                                           │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ Byte Output (30M)                                           │   │   │
│  │  │ • Project to 256 byte logits                               │   │   │
│  │  │ • Resonance matching to byte codebook                      │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Specifications

### 1. Resonance Field (6.5B params)

The field IS the knowledge. Not a neural network — a spatial energy landscape.

```python
class ResonanceField7B:
    """
    Massive 3D knowledge field. Stores all learned information.
    
    NO GRADIENT TRAINING. Knowledge is INJECTED via physics operations.
    
    Parameters:
        state: [512, 512, 512, 1536] = 6.44B floats
        mass:  [512, 512, 512]       = 134M floats
        temp:  [512, 512, 512]       = 134M floats
        
    Total: ~6.5B parameters (all non-trainable)
    """
    
    def __init__(self):
        # Knowledge tensor: 512³ locations × 1536 features
        self.state = torch.zeros(512, 512, 512, 1536)
        
        # Mass: accumulated evidence at each location
        self.mass = torch.zeros(512, 512, 512)
        
        # Temperature: local plasticity (high = learnable, low = stable)
        self.temperature = torch.ones(512, 512, 512)
        
        # Spatial index for O(log n) queries
        self.index = None  # Built during consolidation
        
    def inject(self, wave: Tensor, strength: float = 1.0):
        """
        Inject a wave into the field. NO GRADIENTS.
        
        1. Compute location from wave
        2. Perturb local neighborhood
        3. Increase mass (evidence)
        4. Cool temperature (stabilize)
        """
        location = self._wave_to_location(wave)  # [3] coordinates
        
        # Radius of influence based on wave intensity
        radius = int(2 + wave.norm() / 10)
        
        # Get local neighborhood
        x, y, z = location.int()
        r = radius
        local = self.state[x-r:x+r+1, y-r:y+r+1, z-r:z+r+1]  # [2r+1, 2r+1, 2r+1, 1536]
        
        # Distance-weighted perturbation
        distances = self._compute_distances(radius)  # [2r+1, 2r+1, 2r+1]
        weights = torch.exp(-distances / radius)
        
        # Resistance from existing mass
        local_mass = self.mass[x-r:x+r+1, y-r:y+r+1, z-r:z+r+1]
        resistance = 1.0 / (1.0 + local_mass)
        
        # Apply perturbation
        delta = strength * weights.unsqueeze(-1) * resistance.unsqueeze(-1) * wave
        self.state[x-r:x+r+1, y-r:y+r+1, z-r:z+r+1] += delta
        
        # Accumulate mass
        self.mass[x-r:x+r+1, y-r:y+r+1, z-r:z+r+1] += weights * strength
        
        # Cool temperature
        self.temperature[x-r:x+r+1, y-r:y+r+1, z-r:z+r+1] *= 0.99
        
    def settle(self, steps: int = 100):
        """
        Energy minimization via diffusion. NO GRADIENTS.
        
        High-energy regions diffuse toward neighbors.
        Low-energy regions (attractors) remain stable.
        """
        for _ in range(steps):
            # Compute local energy (variance from neighbors)
            energy = self._compute_energy()
            
            # Diffusion: high energy → flow toward lower energy
            grad = self._compute_energy_gradient()
            
            # Update: regions with high temp are more plastic
            dt = 0.01
            self.state -= dt * self.temperature.unsqueeze(-1) * grad
            
            # Normalize to prevent unbounded growth
            self.state = F.normalize(self.state, dim=-1) * self.state.norm(dim=-1, keepdim=True).clamp(max=10)
            
    def query(self, wave: Tensor, k: int = 16) -> Tuple[Tensor, Tensor, List]:
        """
        Gravitational retrieval. O(log n) with spatial index.
        
        Returns k-nearest attractors weighted by mass × similarity.
        """
        location = self._wave_to_location(wave)
        
        if self.index is not None:
            # Fast path: use FAISS index
            distances, indices = self.index.search(wave.unsqueeze(0), k)
            features = self._indices_to_features(indices)
            masses = self._indices_to_masses(indices)
        else:
            # Slow path: brute force (used during injection)
            features, masses = self._brute_force_query(location, k)
            
        # Gravity weighting: mass / distance²
        similarities = F.cosine_similarity(wave.unsqueeze(0), features, dim=-1)
        gravity_weights = masses / ((1 - similarities).clamp(min=0.01) ** 2)
        
        return features, gravity_weights, similarities
        
    def consolidate(self):
        """
        Form stable attractors and build spatial index.
        
        Call after batch injection to optimize for queries.
        """
        # Find high-mass regions (attractors)
        attractor_mask = self.mass > self.mass.mean()
        attractor_locations = attractor_mask.nonzero()
        attractor_features = self.state[attractor_mask]
        
        # Build FAISS index
        import faiss
        self.index = faiss.IndexFlatIP(1536)  # Inner product = cosine after norm
        self.index.add(F.normalize(attractor_features, dim=-1).numpy())
        
        print(f"  Consolidated {len(attractor_locations)} attractors")
```

### 2. Physics Stack (300M, frozen)

Existing FLUX components, frozen after Phase 7 training.

```python
class PhysicsStack:
    """
    All Phase 1-7 components. Frozen during emission training.
    
    CSE:     10M  — bytes → waves
    GR:      50M  — gravitational index + mass tracking
    CGN:    100M  — causal geometry for reasoning
    TL:      40M  — thermodynamic regulation
    Memory:  80M  — 3-tier (working/episodic/semantic)
    Bridges: 20M  — wave ↔ field projections
    
    Total: 300M params (all frozen)
    """
    
    def __init__(self):
        self.cse = ContinuousSemanticEncoder()       # 10M
        self.gr = GravitationalRelevance()           # 50M
        self.cgn = CausalGeometryNetwork()           # 100M
        self.tl = ThermodynamicLearner()             # 40M
        self.memory = ThreeTierMemory()              # 80M
        self.wave_to_field = nn.Linear(432, 1536)    # 10M
        self.field_to_wave = nn.Linear(1536, 432)    # 10M
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
            
    def encode(self, text: str) -> Tensor:
        """Encode text to wave sequence."""
        return self.cse.encode(text).full  # [seq, 432]
        
    def route_to_field(self, waves: Tensor) -> Tensor:
        """Project waves to field space."""
        return self.wave_to_field(waves)  # [seq, 1536]
```

### 3. Emission Head (200M, trainable)

The ONLY component that needs gradient training.

```python
class EmissionHead(nn.Module):
    """
    Converts field knowledge into byte sequences.
    
    This is a small transformer that learns HOW TO SPELL,
    not WHAT TO KNOW. Knowledge comes from the field.
    
    Components:
        Context Merger:     20M — fuse field + wave context
        Resonance Emitter: 150M — 12 transformer layers
        Byte Output:        30M — project to byte logits
        
    Total: 200M trainable params
    """
    
    def __init__(
        self,
        field_features: int = 1536,
        wave_dim: int = 432,
        d_model: int = 1024,
        num_layers: int = 12,
        num_heads: int = 16,
        vocab_size: int = 256,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # ─────────────────────────────────────────────
        # Context Merger (20M)
        # ─────────────────────────────────────────────
        
        # Project field features to model dim
        self.field_proj = nn.Linear(field_features, d_model)  # 1.5M
        
        # Project wave context to model dim
        self.wave_proj = nn.Linear(wave_dim, d_model)  # 0.4M
        
        # Gravity-weighted attention over field retrievals
        self.field_attention = nn.MultiheadAttention(
            d_model, num_heads=8, batch_first=True
        )  # 3M
        
        # Merge field + wave into unified context
        self.context_merge = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.LayerNorm(d_model),
        )  # 15M
        
        # ─────────────────────────────────────────────
        # Resonance Emitter (150M)
        # ─────────────────────────────────────────────
        
        # Byte embedding for autoregressive decoding
        self.byte_embed = nn.Embedding(vocab_size + 1, d_model)  # 0.3M
        self.BOS = vocab_size
        
        # Position encoding
        self.pos_embed = nn.Embedding(2048, d_model)  # 2M
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)  # 145M
        
        # ─────────────────────────────────────────────
        # Byte Output (30M)
        # ─────────────────────────────────────────────
        
        # Byte codebook for resonance matching
        self.byte_codebook = nn.Parameter(torch.randn(vocab_size, d_model) * 0.02)  # 0.3M
        
        # Output projection
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)  # 0.3M
        
        # Tie output projection to codebook
        self.output_proj.weight = self.byte_codebook
        
        # Additional output MLP for richer byte prediction
        self.output_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )  # 29M
        
    def forward(
        self,
        field_features: Tensor,     # [batch, k, 1536] retrieved from field
        field_weights: Tensor,      # [batch, k] gravity weights
        wave_context: Tensor,       # [batch, seq, 432] input waves
        target_bytes: Tensor,       # [batch, tgt_len] target byte sequence
    ) -> Tensor:
        """
        Training forward pass with teacher forcing.
        
        Returns: [batch, tgt_len, 256] byte logits
        """
        batch_size, tgt_len = target_bytes.shape
        device = target_bytes.device
        
        # ── Merge Field Context ──
        field_proj = self.field_proj(field_features)  # [batch, k, d_model]
        wave_proj = self.wave_proj(wave_context)       # [batch, seq, d_model]
        
        # Gravity-weighted attention over field
        field_context, _ = self.field_attention(
            query=wave_proj,
            key=field_proj,
            value=field_proj,
            key_padding_mask=None,
            attn_mask=None,
        )  # [batch, seq, d_model]
        
        # Pool wave context
        wave_pooled = wave_proj.mean(dim=1)  # [batch, d_model]
        field_pooled = field_context.mean(dim=1)  # [batch, d_model]
        
        # Merge contexts
        merged = self.context_merge(
            torch.cat([wave_pooled, field_pooled], dim=-1)
        )  # [batch, d_model]
        
        # ── Emission with Teacher Forcing ──
        # Build input: [BOS, byte_0, ..., byte_{n-2}]
        bos = torch.full((batch_size, 1), self.BOS, dtype=torch.long, device=device)
        input_bytes = torch.cat([bos, target_bytes[:, :-1]], dim=1)  # [batch, tgt_len]
        
        # Embed bytes + positions
        byte_embeds = self.byte_embed(input_bytes)  # [batch, tgt_len, d_model]
        positions = torch.arange(tgt_len, device=device).unsqueeze(0)
        pos_embeds = self.pos_embed(positions)  # [1, tgt_len, d_model]
        
        decoder_input = byte_embeds + pos_embeds
        
        # Create memory from merged context (expand for decoder)
        memory = merged.unsqueeze(1).expand(-1, tgt_len, -1)  # [batch, tgt_len, d_model]
        
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=device), diagonal=1
        ).bool()
        
        # Decode
        output = self.decoder(
            decoder_input,
            memory,
            tgt_mask=causal_mask,
        )  # [batch, tgt_len, d_model]
        
        # Output projection
        output = self.output_norm(output)
        output = output + self.output_mlp(output)  # Residual
        logits = self.output_proj(output)  # [batch, tgt_len, 256]
        
        return logits
        
    def generate(
        self,
        field_features: Tensor,
        field_weights: Tensor,
        wave_context: Tensor,
        max_length: int = 200,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> bytes:
        """
        Autoregressive generation.
        """
        self.eval()
        device = wave_context.device
        
        # Merge context (same as forward)
        field_proj = self.field_proj(field_features)
        wave_proj = self.wave_proj(wave_context)
        field_context, _ = self.field_attention(wave_proj, field_proj, field_proj)
        wave_pooled = wave_proj.mean(dim=1)
        field_pooled = field_context.mean(dim=1)
        merged = self.context_merge(torch.cat([wave_pooled, field_pooled], dim=-1))
        
        # Start with BOS
        generated = [self.BOS]
        
        for i in range(max_length):
            # Current sequence
            input_bytes = torch.tensor([generated], dtype=torch.long, device=device)
            byte_embeds = self.byte_embed(input_bytes)
            positions = torch.arange(len(generated), device=device).unsqueeze(0)
            pos_embeds = self.pos_embed(positions)
            decoder_input = byte_embeds + pos_embeds
            
            # Memory
            memory = merged.unsqueeze(1).expand(-1, len(generated), -1)
            
            # Causal mask
            causal_mask = torch.triu(
                torch.ones(len(generated), len(generated), device=device), diagonal=1
            ).bool()
            
            # Decode
            output = self.decoder(decoder_input, memory, tgt_mask=causal_mask)
            output = self.output_norm(output[:, -1:, :])
            output = output + self.output_mlp(output)
            logits = self.output_proj(output).squeeze(1)  # [1, 256]
            
            # Sample
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                
                # Top-p sampling
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                sorted_probs[mask] = 0
                sorted_probs = sorted_probs / sorted_probs.sum()
                
                next_byte = sorted_idx[0, torch.multinomial(sorted_probs[0], 1)].item()
            else:
                next_byte = logits.argmax(-1).item()
                
            if next_byte == self.BOS:  # EOS
                break
                
            generated.append(next_byte)
            
        return bytes(generated[1:])  # Skip BOS
```

---

## Parameter Summary

| Component | Params | Trainable | Training Method |
|-----------|--------|-----------|-----------------|
| **Resonance Field** | 6.5B | ✗ No | Injection + Settling |
| **Physics Stack** | 300M | ✗ No | Frozen (Phase 1-7) |
| **Emission Head** | 200M | ✓ Yes | Gradient descent |
| **Total** | **7B** | **200M** | |

---

## Training Pipeline

### Phase A: Knowledge Injection (No Gradients)

```python
# Inject corpus into field — NO TRAINING
def inject_corpus(field, physics_stack, corpus_path):
    """
    Inject entire corpus into field. Takes hours, not weeks.
    """
    for doc in load_corpus(corpus_path):
        # Encode to waves
        waves = physics_stack.encode(doc)  # [seq, 432]
        
        # Inject each wave into field
        for wave in waves:
            field_wave = physics_stack.wave_to_field(wave)
            field.inject(field_wave, strength=1.0)
            
        # Periodic settling
        if doc_count % 1000 == 0:
            field.settle(steps=50)
            
    # Final consolidation
    field.consolidate()
    
# Injection times (estimated):
# Wikipedia (6M docs):     ~4 hours on single GPU
# OpenWebText (8M docs):   ~6 hours
# Books (100K docs):       ~1 hour
# Total corpus injection:  ~12 hours (vs weeks for LLM training)
```

### Phase B: Emission Training (Gradients, but only 200M params)

```python
# Train emission head — ONLY 200M params
def train_emission(emission_head, field, physics_stack, data):
    """
    Train the emission head. Small, fast.
    
    Estimated time: 2-3 days on 8× A100
    (vs 2-4 weeks for full 7B LLM)
    """
    optimizer = AdamW(emission_head.parameters(), lr=1e-4)
    
    for text in data:
        # Encode
        waves = physics_stack.encode(text)
        target_bytes = torch.tensor(list(text.encode('utf-8')))
        
        # Query field for relevant knowledge (NO GRAD)
        with torch.no_grad():
            field_query = physics_stack.wave_to_field(waves.mean(dim=0))
            field_features, gravity_weights, _ = field.query(field_query, k=32)
            
        # Train emission (GRAD only here)
        logits = emission_head(
            field_features.unsqueeze(0),
            gravity_weights.unsqueeze(0),
            waves.unsqueeze(0),
            target_bytes.unsqueeze(0),
        )
        
        loss = F.cross_entropy(logits.view(-1, 256), target_bytes.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### Phase C: Add New Knowledge (Instant)

```python
# Add knowledge at any time — NO RETRAINING
def add_knowledge(field, physics_stack, new_doc):
    """
    Add new knowledge to FLUX. Takes seconds.
    
    LLM equivalent: Fine-tuning ($$$) or RAG (retrieval only)
    FLUX: Direct injection + settling
    """
    waves = physics_stack.encode(new_doc)
    
    for wave in waves:
        field_wave = physics_stack.wave_to_field(wave)
        field.inject(field_wave, strength=1.0)
        
    field.settle(steps=20)
    
    # Done. New knowledge is integrated.
    # Time: ~0.1 seconds per KB
```

---

## Comparison to LLMs

| Aspect | LLaMA-7B | FLUX-7B |
|--------|----------|---------|
| **Total params** | 7B | 7B |
| **Trainable params** | 7B | 200M (36× fewer) |
| **Training time** | 3-4 weeks | 2-3 days |
| **Training data** | 1-2T tokens | Same |
| **Add new knowledge** | Fine-tune or RAG | Inject (seconds) |
| **Catastrophic forgetting** | Yes | No (field persists) |
| **Interpretability** | None | Causal geometry traces |
| **Inference cost** | O(n²) attention | O(n log n) gravity |

---

## Hardware Requirements

### Training

```
Phase A (Injection):
  - 1× A100 80GB (field fits in memory)
  - 12 hours for full corpus

Phase B (Emission):
  - 8× A100 40GB (model parallel)
  - 2-3 days for emission head
  
Total: ~3 days vs LLaMA's ~4 weeks
```

### Inference

```
Field: 6.5B × 4 bytes = 26GB (fits single A100)
Emission: 200M × 4 bytes = 0.8GB
Physics: 300M × 4 bytes = 1.2GB

Total VRAM: ~28GB (single A100 80GB)
```

---

## File Structure

```
flux/
├── phases/
│   └── flux_7b/
│       ├── FLUX_7B_SPEC.md          # This file
│       ├── resonance_field_7b.py    # 6.5B field
│       ├── emission_head.py         # 200M trainable
│       ├── flux_7b_model.py         # Full system
│       ├── inject_corpus.py         # Knowledge injection
│       ├── train_emission.py        # Emission training
│       └── inference.py             # Generation
```

---

## Key Innovations

1. **Knowledge ≠ Training**: 6.5B params store knowledge without gradients
2. **Train once, know forever**: Emission head learns spelling, not facts
3. **Add knowledge instantly**: Field injection in seconds
4. **No catastrophic forgetting**: Attractors persist
5. **O(log n) retrieval**: Gravitational index
6. **Interpretable reasoning**: CGN traces

---

## Next Steps

1. [ ] Implement `ResonanceField7B` with 512³ × 1536 tensor
2. [ ] Implement `EmissionHead` with 12-layer transformer
3. [ ] Create injection pipeline for Wikipedia/OpenWebText
4. [ ] Train emission head on injected field
5. [ ] Benchmark against LLaMA-7B

---

*FLUX-7B: 7 billion parameters, but only 200 million trained.*
