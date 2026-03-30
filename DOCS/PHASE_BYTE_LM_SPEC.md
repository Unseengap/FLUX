# Phase ByteLM: Native Byte-Level Language Model

## The Missing Piece That Replaces Tokens

> **Problem:** FLUX has rich semantic understanding (field, memory, waves) but cannot generate coherent text natively. Currently depends on external LLMs (Qwen) for fluency.
>
> **Solution:** ByteLanguageModel — self-attention over byte history + cross-attention to waves.

---

## Why Current WaveDecoder Fails

```
Input:  "The capital of France is"
CSE:    [wave₀, wave₁, wave₂, ...]     ← semantic meaning encoded
        
WaveDecoder generates "Paris" as: P-a-r-i-s
        
Step 1: Generate 'P' — looks at waves, no byte history
Step 2: Generate 'a' — looks at waves + GRU remembers 'P'  
Step 3: Generate 'r' — looks at waves + GRU remembers 'a' (forgot 'P')
Step 4: Generate 'i' — looks at waves + GRU remembers 'r' (forgot 'P','a')
...

Result: "dv g rr orhsrtiurl esgntshrllso" (actual Phase 8 output)
```

**The GRU has ~256 hidden units.** It cannot hold the context of "The capital of France is" while also tracking what it has generated. It collapses.

---

## Architecture: ByteLanguageModel

```
┌────────────────────────────────────────────────────────────────────────┐
│                        ByteLanguageModel                                │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Wave Sequence [seq, 432] ──────────────────────────┐                 │
│        (from CSE)                                     │                 │
│                                                       ▼                 │
│   Byte History [gen_len]                        ┌─────────────┐        │
│        │                                        │ Wave Proj   │        │
│        ▼                                        │ 432 → 512   │        │
│   ┌─────────────┐                               └──────┬──────┘        │
│   │ Byte Embed  │                                      │               │
│   │ 256 → 512   │                                      │               │
│   └──────┬──────┘                                      │               │
│          │                                             │               │
│          ▼                                             │               │
│   ┌─────────────────────────────────────────────────────────────┐     │
│   │                    Transformer Layers (×N)                   │     │
│   │                                                              │     │
│   │   ┌─────────────────────────────────────────────────────┐   │     │
│   │   │  Causal Self-Attention (over byte history)          │   │     │
│   │   │  • Looks at ALL previous bytes generated            │   │     │
│   │   │  • Learns byte-level language patterns              │   │     │
│   │   │  • "After 'Pari' usually comes 's'"                 │   │     │
│   │   └─────────────────────────────────────────────────────┘   │     │
│   │                          │                                   │     │
│   │                          ▼                                   │     │
│   │   ┌─────────────────────────────────────────────────────┐   │     │
│   │   │  Cross-Attention (to wave sequence)           ◄─────┼───┼─────┘
│   │   │  • Semantic guidance: WHAT to say              │     │   │
│   │   │  • Focuses on relevant input positions         │     │   │
│   │   │  • "France" wave → high attention when         │     │   │
│   │   │    generating "Paris"                          │     │   │
│   │   └─────────────────────────────────────────────────────┘   │
│   │                          │                                   │
│   │                          ▼                                   │
│   │   ┌─────────────────────────────────────────────────────┐   │
│   │   │  FFN (Feed-Forward Network)                         │   │
│   │   └─────────────────────────────────────────────────────┘   │
│   │                                                              │
│   └──────────────────────────────────────────────────────────────┘     │
│                          │                                              │
│                          ▼                                              │
│   ┌─────────────┐                                                      │
│   │ Output Head │ → 256 logits (next byte)                             │
│   │ 512 → 256   │                                                      │
│   └─────────────┘                                                      │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Key Differences from WaveDecoder

| Feature | WaveDecoder (Current) | ByteLM (New) |
|---------|----------------------|--------------|
| **Self-attention over bytes** | ✗ None (GRU only) | ✓ Full causal mask |
| **Context window** | ~5 bytes (GRU limit) | 2048+ bytes |
| **Byte history access** | Sequential decay | Direct attention |
| **Cross-attention to waves** | ✓ Yes | ✓ Yes (preserved) |
| **Parameters** | ~38M | ~100-200M |
| **Training needed** | Minimal | Significant |

---

## Implementation

```python
"""
ByteLanguageModel: Self-attention over bytes + cross-attention to waves.

This is FLUX's native language generation head. It replaces tokens with
bytes while maintaining coherent generation through self-attention over
the full generation history.

Physics Analogy:
    Self-attention = local wave interference (bytes affect each other)
    Cross-attention = field guidance (semantic meaning shapes output)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ByteLanguageModel(nn.Module):
    """
    Native byte-level language model for FLUX.
    
    Combines:
    1. Self-attention over byte history (coherent generation)
    2. Cross-attention to wave sequence (semantic guidance)
    
    This replaces tokens in traditional LLMs while staying vocabulary-free.
    
    Args:
        wave_dim: CSE wave dimension (432)
        embed_dim: Internal embedding dimension (512)
        num_layers: Number of transformer layers (8-12)
        num_heads: Number of attention heads (8-16)
        max_seq_len: Maximum generation length (2048)
        dropout: Dropout rate (0.1)
    """
    
    def __init__(
        self,
        wave_dim: int = 432,
        embed_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.wave_dim = wave_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Byte embedding (256 bytes + BOS + EOS)
        self.byte_embed = nn.Embedding(258, embed_dim)
        self.BOS = 256
        self.EOS = 257
        
        # Positional encoding (learnable)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        
        # Wave projection (for cross-attention K/V)
        self.wave_proj = nn.Linear(wave_dim, embed_dim)
        
        # Transformer decoder layers
        # Each layer has: self-attn → cross-attn → FFN
        self.layers = nn.ModuleList([
            ByteLMLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, 256)  # 256 byte values
        
        # Causal mask (cached)
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        )
    
    def forward(
        self,
        target_bytes: torch.Tensor,
        wave_sequence: torch.Tensor,
        return_loss: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Teacher-forced training.
        
        Args:
            target_bytes: [seq_len] byte values (0-255)
            wave_sequence: [src_len, wave_dim] CSE output
            return_loss: Whether to compute and return loss
            
        Returns:
            (logits [seq_len, 256], loss)
        """
        device = target_bytes.device
        seq_len = target_bytes.shape[0]
        
        # Build input: BOS + target[:-1]
        bos = torch.full((1,), self.BOS, dtype=torch.long, device=device)
        input_bytes = torch.cat([bos, target_bytes[:-1]])  # [seq_len]
        
        # Byte + position embeddings
        byte_emb = self.byte_embed(input_bytes)  # [seq_len, embed_dim]
        positions = torch.arange(seq_len, device=device)
        pos_emb = self.pos_embed(positions)  # [seq_len, embed_dim]
        x = byte_emb + pos_emb  # [seq_len, embed_dim]
        
        # Project waves for cross-attention
        wave_kv = self.wave_proj(wave_sequence)  # [src_len, embed_dim]
        
        # Causal mask for self-attention
        mask = self.causal_mask[:seq_len, :seq_len]
        
        # Pass through layers
        for layer in self.layers:
            x = layer(x, wave_kv, attn_mask=mask)
        
        # Output projection
        x = self.output_norm(x)
        logits = self.output_proj(x)  # [seq_len, 256]
        
        # Compute loss if requested
        loss = None
        if return_loss:
            loss = F.cross_entropy(logits, target_bytes)
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        wave_sequence: torch.Tensor,
        max_length: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> bytes:
        """
        Autoregressive generation with full self-attention.
        
        Args:
            wave_sequence: [src_len, wave_dim] CSE output
            max_length: Maximum bytes to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            
        Returns:
            Generated bytes
        """
        device = wave_sequence.device
        
        # Project waves (done once, reused for all steps)
        wave_kv = self.wave_proj(wave_sequence)  # [src_len, embed_dim]
        
        # Start with BOS
        generated = [self.BOS]
        
        for step in range(max_length):
            seq_len = len(generated)
            
            # Embed all generated bytes so far
            input_ids = torch.tensor(generated, device=device)
            byte_emb = self.byte_embed(input_ids)
            positions = torch.arange(seq_len, device=device)
            pos_emb = self.pos_embed(positions)
            x = byte_emb + pos_emb  # [seq_len, embed_dim]
            
            # Causal mask
            mask = self.causal_mask[:seq_len, :seq_len]
            
            # Forward through all layers
            for layer in self.layers:
                x = layer(x, wave_kv, attn_mask=mask)
            
            # Get logits for last position only
            x = self.output_norm(x[-1:])  # [1, embed_dim]
            logits = self.output_proj(x).squeeze(0)  # [256]
            
            # Temperature + nucleus sampling
            scaled = logits / max(temperature, 1e-8)
            probs = F.softmax(scaled, dim=-1)
            
            # Top-p filtering
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=0)
            mask_nucleus = cumsum > top_p
            mask_nucleus[0] = False  # Keep at least one
            sorted_probs[mask_nucleus] = 0
            sorted_probs = sorted_probs / sorted_probs.sum()
            
            # Sample
            sample_idx = torch.multinomial(sorted_probs, 1)
            next_byte = sorted_idx[sample_idx].item()
            
            # Check for EOS or null
            if next_byte >= 256 or next_byte == 0:
                break
            
            generated.append(next_byte)
        
        # Skip BOS token, return bytes
        return bytes(generated[1:])


class ByteLMLayer(nn.Module):
    """
    Single ByteLM layer: Self-Attn → Cross-Attn → FFN
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        
        # Self-attention over byte history
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(embed_dim)
        
        # Cross-attention to wave sequence
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        wave_kv: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [seq_len, embed_dim] byte embeddings
            wave_kv: [src_len, embed_dim] wave projections
            attn_mask: [seq_len, seq_len] causal mask
        """
        # Add batch dimension for attention
        x = x.unsqueeze(0)  # [1, seq_len, embed_dim]
        wave_kv = wave_kv.unsqueeze(0)  # [1, src_len, embed_dim]
        
        # Self-attention (causal, over byte history)
        residual = x
        x_norm = self.self_attn_norm(x)
        self_out, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            attn_mask=attn_mask,
            is_causal=False,  # We provide explicit mask
        )
        x = residual + self_out
        
        # Cross-attention (to wave sequence)
        residual = x
        x_norm = self.cross_attn_norm(x)
        cross_out, _ = self.cross_attn(x_norm, wave_kv, wave_kv)
        x = residual + cross_out
        
        # FFN
        residual = x
        x_norm = self.ffn_norm(x)
        ffn_out = self.ffn(x_norm)
        x = residual + ffn_out
        
        return x.squeeze(0)  # [seq_len, embed_dim]
```

---

## Parameter Count

| Configuration | Layers | Heads | Embed | Parameters |
|--------------|--------|-------|-------|------------|
| ByteLM-Small | 6 | 8 | 512 | ~50M |
| ByteLM-Base | 8 | 8 | 512 | ~70M |
| ByteLM-Medium | 12 | 12 | 768 | ~150M |
| ByteLM-Large | 16 | 16 | 1024 | ~300M |

**Recommended:** ByteLM-Base (8 layers, 512 dim) — comparable to WaveDecoder but with proper self-attention.

---

## Training Requirements

### Data
- **Minimum:** 10M bytes (~10MB of text)
- **Recommended:** 1B bytes (~1GB of text)  
- **Ideal:** 10B bytes (~10GB of text)

### Training Strategy
```python
# Phase A: Basic byte-level language modeling
# Train on raw text, byte prediction loss only
for text in dataset:
    bytes_tensor = torch.tensor(list(text.encode('utf-8')))
    wave = cse.encode(text)
    _, loss = byte_lm(bytes_tensor, wave.full)
    loss.backward()

# Phase B: Fine-tune with FLUX field guidance
# Inject field context alongside waves
for text in dataset:
    wave = cse.encode(text)
    field_context = field.query(wave.mean(dim=0))
    # Cross-attend to both wave sequence AND field context
```

### Hardware
- **Minimum:** 1x T4 (16GB) — batch size 4, gradient accumulation
- **Recommended:** 1x A100 (40GB) — batch size 32
- **Training time:** ~24-48 hours for ByteLM-Base on 1GB data

---

## Integration with FLUXModel

```python
class FLUXModel(nn.Module):
    def __init__(self, ...):
        # Existing components
        self.cse = ContinuousSemanticEncoder(...)
        self.field = ResonanceField(...)
        self.memory = MemoryRouter(...)
        
        # NEW: Replace WaveDecoder with ByteLM
        self.byte_lm = ByteLanguageModel(
            wave_dim=432,
            embed_dim=512,
            num_layers=8,
        )
    
    def generate(self, prompt: str, max_length: int = 256) -> str:
        # Encode prompt
        wave = self.cse.encode(prompt)
        wave_sequence = wave.full  # [seq, 432]
        
        # Generate with ByteLM (native, no external LLM!)
        generated_bytes = self.byte_lm.generate(
            wave_sequence=wave_sequence,
            max_length=max_length,
        )
        
        return prompt + generated_bytes.decode('utf-8', errors='replace')
```

---

## Expected Results

### Before (WaveDecoder)
```
Prompt: "The capital of France is"
Output: "dv g rr orhsrtiurl esgntshrllso"
```

### After (ByteLM, properly trained)
```
Prompt: "The capital of France is"
Output: "Paris, which is known for the Eiffel Tower"
```

---

## Why This Works

1. **Self-attention sees the full context:**
   - "The capital of France is P" → attention focuses on "France", "capital"
   - Knows "Paris" should follow because it learned this pattern

2. **Cross-attention provides semantic guidance:**
   - Wave for "France" strongly influences the generation
   - Even if the model hasn't seen "France → Paris" literally, the wave meaning guides it

3. **Byte-level = vocabulary-free:**
   - Works on UTF-8 directly
   - No OOV errors
   - Handles any language, emoji, code

---

## Migration Path

1. **Phase A:** Implement ByteLM module
2. **Phase B:** Train on OpenWebText (1GB subset)
3. **Phase C:** Fine-tune with FLUX field guidance
4. **Phase D:** Replace WaveDecoder in FLUXModel
5. **Phase E:** Remove external LLM dependency

---

## Summary

| Current | Proposed |
|---------|----------|
| WaveDecoder (GRU) | ByteLanguageModel (Transformer) |
| ~5 byte context | 2048+ byte context |
| Cross-attention only | Self + Cross attention |
| Gibberish output | Coherent generation |
| External LLM needed | Fully native |

**This is what makes FLUX a complete, self-sufficient language model.**
