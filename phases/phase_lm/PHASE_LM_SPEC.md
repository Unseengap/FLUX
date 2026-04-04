# Phase LM Specification: FLUX Language Model

> **The world's first vocabulary-free, wave-based autoregressive language model.**

---

## Executive Summary

FLUX-LM combines Phase 1 (CSE) and Phase 1.5 (CWC) with a new WavePredictor to create a standalone language model that generates text **without tokens, without vocabulary, operating purely on semantic waves**.

**Target Size:** ~124M parameters (GPT-2 small equivalent)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FLUX-LM Architecture                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Input: "The quick brown"                                              │
│              │                                                          │
│              ▼                                                          │
│   ┌─────────────────────┐                                               │
│   │   CSE (Encoder)     │  Bytes → 432D Semantic Waves                  │
│   │   ~10M params       │  [18 bytes] → [18, 432]                       │
│   └─────────────────────┘                                               │
│              │                                                          │
│              ▼                                                          │
│   ┌─────────────────────┐                                               │
│   │   CWC (Causal)      │  Add causal direction + temporal              │
│   │   ~5M params        │  [18, 432] → [18, 608]                        │
│   └─────────────────────┘                                               │
│              │                                                          │
│              ▼                                                          │
│   ┌─────────────────────┐                                               │
│   │  WavePredictor      │  Predict NEXT wave from context               │
│   │  (Transformer)      │  [18, 608] → [1, 608] (next wave)             │
│   │  ~100M params       │  12 layers, 12 heads, 768 hidden              │
│   └─────────────────────┘                                               │
│              │                                                          │
│              ▼                                                          │
│   ┌─────────────────────┐                                               │
│   │   WaveDecoder       │  Wave → byte probability distribution         │
│   │   ~5M params        │  [1, 608] → [1, 256] (byte logits)            │
│   └─────────────────────┘                                               │
│              │                                                          │
│              ▼                                                          │
│   Output: byte 102 = 'f' → "The quick brown f..."                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Parameter Budget (124M Target)

| Component | Current | Scaled | Purpose |
|-----------|---------|--------|---------|
| **CSE** | 1.3M | **10M** | Byte → wave encoding |
| **CWC** | 570K | **5M** | Causal direction + temporal |
| **WavePredictor** | NEW | **100M** | Autoregressive prediction |
| **WaveDecoder** | 418K | **9M** | Wave → byte decoding |
| **Total** | ~2.3M | **~124M** | GPT-2 small equivalent |

---

## Component Specifications

### 1. CSE-L (Continuous Semantic Encoder - Large)

Scaled version of Phase 1 CSE for better capacity.

```python
CSE_L_CONFIG = {
    'byte_window': 8,
    'byte_stride': 1,
    'byte_embed_dim': 64,           # Up from 32
    'conv_channels': [256, 512, 512, 512],  # Wider conv bank
    'hidden_dim': 1024,             # Up from 512
    'wave_dims': {
        'phonetic':  64,
        'syntactic': 64,
        'semantic':  256,
        'temporal':  32,
        'intensity': 16,
    },  # Total: 432
    'interference_radius': 4,
    'dropout': 0.1,
}
# ~10M parameters
```

### 2. CWC-L (Causal Wave Chainer - Large)

Scaled version of Phase 1.5 CWC with better order modeling.

```python
CWC_L_CONFIG = {
    'wave_dim': 432,
    'causal_dim': 176,              # Total output: 608
    'hidden_dim': 1024,
    'n_heads': 8,
    'n_layers': 2,
    'dropout': 0.1,
    
    # Improved order modeling (fixing the coherence gap = 0 problem)
    'position_encoding': 'rotary',  # RoPE for better position awareness
    'temporal_mlp_layers': 3,
    'order_aware': True,            # Explicit order-sensitive features
}
# ~5M parameters
```

### 3. WavePredictor (NEW - Main Backbone)

The core autoregressive transformer operating on waves.

```python
WAVE_PREDICTOR_CONFIG = {
    'input_dim': 608,               # From CWC output
    'hidden_dim': 768,              # GPT-2 small hidden size
    'n_heads': 12,                  # GPT-2 small heads
    'n_layers': 12,                 # GPT-2 small layers
    'ff_dim': 3072,                 # 4x hidden
    'max_seq_len': 1024,
    'dropout': 0.1,
    
    # Wave-specific modifications
    'wave_attention': True,         # Interference-aware attention
    'position_encoding': 'rotary',  # RoPE
    'tie_weights': False,           # No embedding tying (not token-based)
}
# ~100M parameters
```

### 4. WaveDecoder-L (Wave Decoder - Large)

Scaled decoder for next-byte prediction.

```python
WAVE_DECODER_L_CONFIG = {
    'input_dim': 608,               # From predictor output
    'hidden_dim': 2048,
    'n_layers': 4,
    'output_dim': 256,              # Byte vocabulary
    'dropout': 0.1,
    
    # Multi-scale decoding for better byte prediction
    'use_multi_scale': True,
    'scales': [1, 2, 4],            # Different receptive fields
}
# ~9M parameters
```

---

## Training Objectives

### Primary: Next-Byte Prediction Loss

```python
def next_byte_loss(model, text):
    """Autoregressive language modeling loss."""
    bytes_tensor = text.encode('utf-8')
    
    total_loss = 0
    for i in range(1, len(bytes_tensor)):
        context = bytes_tensor[:i]
        target = bytes_tensor[i]
        
        # Encode context → waves
        waves = model.cse.encode(context)           # [i, 432]
        causal_waves = model.cwc(waves)             # [i, 608]
        
        # Predict next wave
        next_wave = model.predictor(causal_waves)   # [1, 608]
        
        # Decode to byte distribution
        logits = model.decoder(next_wave)           # [1, 256]
        
        loss = F.cross_entropy(logits, target)
        total_loss += loss
    
    return total_loss / (len(bytes_tensor) - 1)
```

### Secondary Objectives (Multi-Task)

| Loss | Weight | Purpose |
|------|--------|---------|
| Next-byte CE | 1.0 | Primary LM objective |
| Wave reconstruction | 0.1 | Stability |
| Semantic contrastive | 0.1 | Meaning preservation |
| Order discrimination | 0.2 | **Fix coherence gap** |
| Contradiction detection | 0.1 | Reasoning |

### Order Discrimination Loss (Critical)

The Phase 1.5 results show coherence_gap = 0.0. We need explicit order supervision:

```python
def order_discrimination_loss(model, text):
    """
    Explicitly train model to distinguish ordered vs shuffled.
    This fixes the coherence_gap = 0.0 problem from Phase 1.5.
    """
    words = text.split()
    if len(words) < 4:
        return 0
    
    # Original order
    original_waves = model.encode(text)
    
    # Shuffled order
    shuffled = ' '.join(random.sample(words, len(words)))
    shuffled_waves = model.encode(shuffled)
    
    # Binary classification: which is ordered?
    orig_score = model.order_classifier(original_waves)
    shuf_score = model.order_classifier(shuffled_waves)
    
    # Margin loss: original should score higher
    margin = 0.5
    loss = F.relu(margin - orig_score + shuf_score)
    
    return loss
```

---

## Data Pipeline

### Training Data

| Dataset | Size | Purpose |
|---------|------|---------|
| WikiText-103 | 100M tokens | Primary training |
| OpenWebText | 8B tokens | Scale up |
| Books3 subset | 10B tokens | Long-range coherence |
| Code (The Stack sample) | 1B tokens | Multi-domain |

### Data Processing

```python
class FluxLMDataset:
    """Byte-level dataset for FLUX-LM."""
    
    def __init__(self, texts, max_len=512):
        self.texts = texts
        self.max_len = max_len
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Convert to bytes
        bytes_seq = list(text.encode('utf-8'))[:self.max_len]
        
        # Input: bytes[:-1], Target: bytes[1:]
        input_bytes = bytes_seq[:-1]
        target_bytes = bytes_seq[1:]
        
        return {
            'input': torch.tensor(input_bytes, dtype=torch.long),
            'target': torch.tensor(target_bytes, dtype=torch.long),
        }
```

---

## Training Configuration

### Kaggle T4 (16GB VRAM)

```python
TRAIN_CONFIG_T4 = {
    'batch_size': 8,
    'grad_accum': 8,                # Effective batch = 64
    'max_seq_len': 256,             # Limited for memory
    'learning_rate': 3e-4,
    'warmup_steps': 1000,
    'total_steps': 50000,
    'weight_decay': 0.01,
    'grad_clip': 1.0,
    
    # Memory optimization
    'mixed_precision': True,        # FP16
    'gradient_checkpointing': True,
    'compile': False,               # torch.compile doesn't work on T4
}
```

### A100 (40GB VRAM)

```python
TRAIN_CONFIG_A100 = {
    'batch_size': 32,
    'grad_accum': 4,                # Effective batch = 128
    'max_seq_len': 512,
    'learning_rate': 3e-4,
    'warmup_steps': 2000,
    'total_steps': 100000,
    'weight_decay': 0.01,
    'grad_clip': 1.0,
    
    # Full precision OK
    'mixed_precision': False,
    'gradient_checkpointing': False,
    'compile': True,                # Faster with compile
}
```

---

## Evaluation Metrics

### Perplexity (Primary)

```python
def compute_perplexity(model, test_texts):
    """Byte-level perplexity."""
    total_loss = 0
    total_bytes = 0
    
    for text in test_texts:
        bytes_seq = list(text.encode('utf-8'))
        
        for i in range(1, len(bytes_seq)):
            logits = model.forward(bytes_seq[:i])
            loss = F.cross_entropy(logits, bytes_seq[i])
            total_loss += loss.item()
            total_bytes += 1
    
    avg_loss = total_loss / total_bytes
    return math.exp(avg_loss)
```

### Target Metrics

| Metric | Target | GPT-2 Small Ref |
|--------|--------|-----------------|
| Byte perplexity | < 2.5 | ~1.8 (token) |
| Generation coherence | > 0.7 | N/A |
| Order discrimination | > 90% | N/A |
| Contradiction detection | > 90% | N/A |

---

## Generation

### Autoregressive Sampling

```python
def generate(model, prompt: str, max_new_bytes: int = 100, temperature: float = 0.8):
    """Generate text byte-by-byte."""
    
    output = list(prompt.encode('utf-8'))
    
    for _ in range(max_new_bytes):
        # Encode current context
        waves = model.cse.encode(bytes(output))
        causal_waves = model.cwc(waves)
        
        # Predict next wave
        next_wave = model.predictor(causal_waves)
        
        # Decode to byte distribution
        logits = model.decoder(next_wave) / temperature
        probs = F.softmax(logits, dim=-1)
        
        # Sample
        next_byte = torch.multinomial(probs, 1).item()
        
        # Stop conditions
        if next_byte == 0:  # Null byte = EOS
            break
        if next_byte == 10 and len(output) > len(prompt) + 50:  # Newline after 50 chars
            output.append(next_byte)
            break
            
        output.append(next_byte)
    
    return bytes(output).decode('utf-8', errors='replace')
```

### Sampling Strategies

| Strategy | Use Case |
|----------|----------|
| Greedy | Factual, deterministic |
| Temperature (0.5-1.0) | Creative, varied |
| Top-k (k=50) | Balanced |
| Top-p (p=0.9) | Natural distribution |
| Beam search (n=4) | Quality, slower |

---

## File Structure

```
phases/phase_lm/
├── PHASE_LM_SPEC.md              ← This file (copy to phases/)
├── __init__.py
├── cse_large.py                  ← Scaled CSE (10M)
├── cwc_large.py                  ← Scaled CWC (5M)
├── wave_predictor.py             ← Main transformer (100M)
├── wave_decoder_large.py         ← Scaled decoder (9M)
├── flux_lm.py                    ← Combined model
├── train_flux_lm.py              ← Training script
├── dataset.py                    ← Data pipeline
├── generate.py                   ← Generation utilities
├── test_flux_lm_test1.py         ← Perplexity test
├── test_flux_lm_test2.py         ← Order discrimination test
├── test_flux_lm_test3.py         ← Generation quality test
├── demo_flux_lm_demo1.py         ← Interactive demo
└── RESULTS_PHASE_LM.md           ← Auto-generated results

notebooks/
├── flux_lm_train.ipynb           ← Kaggle training notebook
```

---

## Acceptance Criteria

| Test | Threshold | Status |
|------|-----------|--------|
| Byte perplexity | < 2.5 | [ ] |
| Order discrimination | > 90% | [ ] |
| Contradiction detection | > 85% | [ ] |
| Generation coherence | > 0.7 | [ ] |
| Speed (T4) | > 10 bytes/sec | [ ] |
| No vocab needed | ✓ | [ ] |

---

## Training Schedule (Kaggle T4)

| Day | Phase | Steps | Hours |
|-----|-------|-------|-------|
| 1 | CSE-L warmup | 5,000 | 2 |
| 1 | CWC-L warmup | 5,000 | 2 |
| 1-2 | Full model joint | 20,000 | 8 |
| 2-3 | Scale to WikiText-103 | 30,000 | 8 |
| 3 | Fine-tune + eval | 10,000 | 4 |
| **Total** | | 70,000 | **~24h** |

---

## Why This Works

### Fixing the Phase 1.5 Problems

1. **Coherence gap = 0.0** → Add explicit order discrimination loss
2. **Order accuracy 66%** → Scale CWC with RoPE positions
3. **No prediction** → Add WavePredictor transformer

### Advantages Over Transformers

| Aspect | Transformer | FLUX-LM |
|--------|------------|---------|
| Vocabulary | 50K fixed tokens | 256 bytes (any UTF-8) |
| Tokenization | Required | None |
| New languages | Retrain tokenizer | Works immediately |
| Subword splits | Arbitrary | Natural byte boundaries |
| Semantic space | Discrete | Continuous waves |

---

## Next Steps

1. **Implement phase_lm modules** (this spec)
2. **Train on Kaggle** (24h on T4)
3. **Evaluate perplexity** (target < 2.5)
4. **Demo generation** (coherent text)
5. **Save as Flux-LM-124M.flx**

---

*This specification enables the world's first production-ready vocabulary-free language model.*
