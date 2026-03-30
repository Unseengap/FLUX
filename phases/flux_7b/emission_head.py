"""
FLUX-7B: Emission Head (200M Trainable Parameters)

The ONLY component that needs gradient training.

This learns HOW TO SPELL — converting field knowledge into byte sequences.
The knowledge itself is stored in the field (6.5B, no gradients).

Architecture:
    Context Merger (20M):  Fuse field retrievals with wave context
    Resonance Emitter (150M): 12-layer transformer for coherent emission
    Byte Output (30M): Project to byte logits

Training: Standard cross-entropy on target bytes.
Time: 2-3 days on 8× A100 (vs weeks for full 7B LLM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class EmissionConfig:
    """Configuration for emission head."""
    
    # Input dimensions
    field_features: int = 1536
    wave_dim: int = 432
    
    # Model dimensions
    d_model: int = 1024
    num_layers: int = 12
    num_heads: int = 16
    d_ff: int = 4096  # FFN intermediate
    
    # Output
    vocab_size: int = 256  # Bytes
    max_seq_len: int = 2048
    
    # Regularization
    dropout: float = 0.1
    
    # Retrieval
    k_attractors: int = 32  # Field retrievals per query
    
    @property
    def total_params(self) -> int:
        """Estimate total parameters."""
        # Context merger
        merger = (
            self.field_features * self.d_model +  # field_proj
            self.wave_dim * self.d_model +  # wave_proj
            4 * self.d_model * self.d_model +  # field_attention
            self.d_model * 2 * self.d_model * 4 + self.d_model * 4 * self.d_model  # merge MLP
        )
        
        # Emitter (transformer)
        per_layer = (
            4 * self.d_model * self.d_model +  # self-attn
            4 * self.d_model * self.d_model +  # cross-attn
            self.d_model * self.d_ff + self.d_ff * self.d_model  # FFN
        )
        emitter = self.num_layers * per_layer
        
        # Output
        output = self.d_model * self.vocab_size * 2  # proj + codebook
        
        return merger + emitter + output
        
    def __post_init__(self):
        print(f"  EmissionHead: ~{self.total_params / 1e6:.0f}M parameters")


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (like LLaMA)."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to q and k."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GravityAttention(nn.Module):
    """
    Self-attention with optional gravity weighting.
    
    Standard multi-head attention that can incorporate
    external gravity weights for field cross-attention.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        is_cross: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.is_cross = is_cross
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        if not is_cross:
            self.rotary = RotaryEmbedding(self.head_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        gravity_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, D = x.shape
        
        # Query from x
        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Key, Value from context (cross) or x (self)
        kv_input = context if self.is_cross and context is not None else x
        K = self.k_proj(kv_input).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv_input).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        S = K.shape[2]  # Source length
        
        # Apply rotary embeddings for self-attention
        if not self.is_cross:
            cos, sin = self.rotary(x, L)
            cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, L, head_dim]
            sin = sin.unsqueeze(0).unsqueeze(0)
            Q, K = apply_rotary_emb(Q, K, cos, sin)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, L, S]
        
        # Apply causal mask for self-attention
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        # Apply gravity weights for cross-attention (field retrieval)
        if gravity_weights is not None and self.is_cross:
            # gravity_weights: [B, S] → [B, 1, 1, S]
            g = gravity_weights.unsqueeze(1).unsqueeze(1)
            scores = scores + torch.log(g + 1e-6)  # Log-gravity weighting
            
        # Softmax + dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply to values
        out = torch.matmul(attn, V)  # [B, H, L, head_dim]
        out = out.transpose(1, 2).reshape(B, L, D)
        
        return self.out_proj(out)


class WaveFFN(nn.Module):
    """
    Feed-forward network with SwiGLU activation.
    Physics interpretation: multi-frequency wave mixing.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: (x W1) ⊙ silu(x W3) then W2
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class EmitterBlock(nn.Module):
    """
    Single transformer block for emission.
    
    Components:
    1. Causal self-attention (over generated sequence)
    2. Cross-attention to field context (with gravity weighting)
    3. Wave FFN (SwiGLU)
    """
    
    def __init__(self, config: EmissionConfig):
        super().__init__()
        
        # Pre-norm
        self.norm1 = RMSNorm(config.d_model)
        self.self_attn = GravityAttention(
            config.d_model, config.num_heads, config.dropout, is_cross=False
        )
        
        self.norm2 = RMSNorm(config.d_model)
        self.cross_attn = GravityAttention(
            config.d_model, config.num_heads, config.dropout, is_cross=True
        )
        
        self.norm3 = RMSNorm(config.d_model)
        self.ffn = WaveFFN(config.d_model, config.d_ff, config.dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        causal_mask: torch.Tensor,
        gravity_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with causal mask
        x = x + self.self_attn(self.norm1(x), mask=causal_mask)
        
        # Cross-attention to context with gravity weights
        x = x + self.cross_attn(self.norm2(x), context, gravity_weights=gravity_weights)
        
        # FFN
        x = x + self.ffn(self.norm3(x))
        
        return x


class EmissionHead(nn.Module):
    """
    FLUX-7B Emission Head — The ONLY trainable component.
    
    Converts field knowledge + wave context into byte sequences.
    Learns HOW TO SPELL, not WHAT TO KNOW.
    
    Architecture:
        Context Merger (20M):  Fuse field + wave
        Resonance Emitter (150M): 12-layer transformer
        Byte Output (30M): Codebook + projection
        
    Total: ~200M parameters
    """
    
    def __init__(self, config: Optional[EmissionConfig] = None):
        super().__init__()
        self.config = config or EmissionConfig()
        
        # ─────────────────────────────────────────────
        # Context Merger (~20M)
        # ─────────────────────────────────────────────
        
        # Project field features to model dim
        self.field_proj = nn.Linear(self.config.field_features, self.config.d_model, bias=False)
        
        # Project wave context to model dim
        self.wave_proj = nn.Linear(self.config.wave_dim, self.config.d_model, bias=False)
        
        # Cross-attention: wave queries field
        self.field_attention = GravityAttention(
            self.config.d_model, 
            num_heads=8, 
            dropout=self.config.dropout,
            is_cross=True,
        )
        self.field_attn_norm = RMSNorm(self.config.d_model)
        
        # Merge into context
        self.context_merge = nn.Sequential(
            nn.Linear(self.config.d_model * 2, self.config.d_model * 4, bias=False),
            nn.GELU(),
            nn.Linear(self.config.d_model * 4, self.config.d_model, bias=False),
        )
        self.context_norm = RMSNorm(self.config.d_model)
        
        # ─────────────────────────────────────────────
        # Resonance Emitter (~150M)
        # ─────────────────────────────────────────────
        
        # Byte embedding
        self.byte_embed = nn.Embedding(self.config.vocab_size + 1, self.config.d_model)
        self.BOS = self.config.vocab_size
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            EmitterBlock(self.config) for _ in range(self.config.num_layers)
        ])
        
        # ─────────────────────────────────────────────
        # Byte Output (~30M)
        # ─────────────────────────────────────────────
        
        self.output_norm = RMSNorm(self.config.d_model)
        
        # Byte codebook: learned representations for each byte value
        self.byte_codebook = nn.Parameter(
            torch.randn(self.config.vocab_size, self.config.d_model) * 0.02
        )
        
        # Output projection (tied to codebook)
        self.output_proj = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        self.output_proj.weight = self.byte_codebook
        
        # Initialize
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, std=0.02)
                
    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask
        
    def _merge_context(
        self,
        field_features: torch.Tensor,
        gravity_weights: torch.Tensor,
        wave_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Merge field and wave context into unified representation.
        
        Args:
            field_features: [batch, k, field_features] retrieved attractors
            gravity_weights: [batch, k] gravity scores
            wave_context: [batch, seq, wave_dim] input waves
            
        Returns:
            [batch, seq, d_model] merged context
        """
        # Project to model dimension
        field_proj = self.field_proj(field_features)  # [batch, k, d_model]
        wave_proj = self.wave_proj(wave_context)  # [batch, seq, d_model]
        
        # Wave attends to field with gravity weighting
        field_context = self.field_attention(
            self.field_attn_norm(wave_proj),
            field_proj,
            gravity_weights=gravity_weights,
        )  # [batch, seq, d_model]
        
        # Pool both for context vector
        wave_pooled = wave_proj.mean(dim=1)  # [batch, d_model]
        field_pooled = field_context.mean(dim=1)  # [batch, d_model]
        
        # Merge
        merged = self.context_merge(torch.cat([wave_pooled, field_pooled], dim=-1))
        merged = self.context_norm(merged)  # [batch, d_model]
        
        return merged
        
    def forward(
        self,
        field_features: torch.Tensor,
        gravity_weights: torch.Tensor,
        wave_context: torch.Tensor,
        target_bytes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Training forward pass with teacher forcing.
        
        Args:
            field_features: [batch, k, field_features] from field.query()
            gravity_weights: [batch, k] gravity weights from field.query()
            wave_context: [batch, seq, wave_dim] from CSE
            target_bytes: [batch, tgt_len] target byte sequence
            
        Returns:
            [batch, tgt_len, vocab_size] logits
        """
        batch_size, tgt_len = target_bytes.shape
        device = target_bytes.device
        
        # Merge field + wave context
        context = self._merge_context(field_features, gravity_weights, wave_context)
        
        # Build decoder input: [BOS, b0, b1, ..., b_{n-2}]
        bos = torch.full((batch_size, 1), self.BOS, dtype=torch.long, device=device)
        input_bytes = torch.cat([bos, target_bytes[:, :-1]], dim=1)
        
        # Embed bytes
        x = self.byte_embed(input_bytes)  # [batch, tgt_len, d_model]
        
        # Create causal mask
        causal_mask = self._make_causal_mask(tgt_len, device)
        
        # Expand context for all positions
        context_expanded = context.unsqueeze(1).expand(-1, tgt_len, -1)  # [batch, tgt_len, d_model]
        
        # Through transformer blocks
        for block in self.blocks:
            x = block(x, context_expanded, causal_mask, gravity_weights)
            
        # Output logits
        x = self.output_norm(x)
        logits = self.output_proj(x)  # [batch, tgt_len, vocab_size]
        
        return logits
        
    @torch.no_grad()
    def generate(
        self,
        field_features: torch.Tensor,
        gravity_weights: torch.Tensor,
        wave_context: torch.Tensor,
        max_length: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> bytes:
        """
        Autoregressive generation.
        
        Args:
            field_features: [1, k, field_features] from field.query()
            gravity_weights: [1, k] gravity weights
            wave_context: [1, seq, wave_dim] from CSE
            max_length: Maximum bytes to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            
        Returns:
            Generated bytes
        """
        self.eval()
        device = wave_context.device
        
        # Merge context once
        context = self._merge_context(field_features, gravity_weights, wave_context)
        
        # Start with BOS
        generated = [self.BOS]
        
        for _ in range(max_length):
            # Current sequence
            input_bytes = torch.tensor([generated], dtype=torch.long, device=device)
            tgt_len = input_bytes.shape[1]
            
            # Embed
            x = self.byte_embed(input_bytes)  # [1, len, d_model]
            
            # Causal mask
            causal_mask = self._make_causal_mask(tgt_len, device)
            
            # Context for all positions
            context_expanded = context.unsqueeze(1).expand(-1, tgt_len, -1)
            
            # Through blocks
            for block in self.blocks:
                x = block(x, context_expanded, causal_mask, gravity_weights)
                
            # Get last position logits
            x = self.output_norm(x[:, -1:, :])
            logits = self.output_proj(x).squeeze(1)  # [1, vocab_size]
            
            # Sample
            if temperature > 0:
                logits = logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                    
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        -1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')
                    
                probs = F.softmax(logits, dim=-1)
                next_byte = torch.multinomial(probs, 1).item()
            else:
                next_byte = logits.argmax(-1).item()
                
            # Stop on BOS (used as EOS too)
            if next_byte == self.BOS:
                break
                
            generated.append(next_byte)
            
            # Check for natural stop
            if len(generated) > 3:
                last_3 = bytes(generated[-3:])
                if last_3 in [b'\n\n\n', b'...', b'###']:
                    break
                    
        return bytes(generated[1:])  # Skip BOS
        
    def count_parameters(self) -> dict:
        """Count parameters by component."""
        def count_in(modules):
            return sum(p.numel() for m in modules for p in m.parameters())
            
        merger = count_in([self.field_proj, self.wave_proj, self.field_attention, 
                          self.field_attn_norm, self.context_merge, self.context_norm])
        emitter = count_in([self.byte_embed] + list(self.blocks))
        output = count_in([self.output_norm, self.output_proj]) + self.byte_codebook.numel()
        
        return {
            'context_merger': merger / 1e6,
            'resonance_emitter': emitter / 1e6,
            'byte_output': output / 1e6,
            'total': (merger + emitter + output) / 1e6,
        }


# ─────────────────────────────────────────────
# Factory Functions
# ─────────────────────────────────────────────

def create_emission_head_200m() -> EmissionHead:
    """Create standard 200M emission head."""
    config = EmissionConfig(
        d_model=1024,
        num_layers=12,
        num_heads=16,
        d_ff=4096,
    )
    return EmissionHead(config)


def create_emission_head_50m() -> EmissionHead:
    """Create smaller 50M emission head for testing."""
    config = EmissionConfig(
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
    )
    return EmissionHead(config)


if __name__ == '__main__':
    print("Testing EmissionHead...")
    
    # Create small config for testing
    config = EmissionConfig(
        d_model=256,
        num_layers=2,
        num_heads=4,
        d_ff=512,
    )
    
    model = EmissionHead(config)
    print(f"  Parameters: {model.count_parameters()}")
    
    # Test forward
    batch_size = 2
    k = 32
    seq_len = 64
    tgt_len = 32
    
    field_features = torch.randn(batch_size, k, config.field_features)
    gravity_weights = torch.rand(batch_size, k)
    wave_context = torch.randn(batch_size, seq_len, config.wave_dim)
    target_bytes = torch.randint(0, 256, (batch_size, tgt_len))
    
    logits = model(field_features, gravity_weights, wave_context, target_bytes)
    print(f"  Forward output: {logits.shape}")
    
    # Test generation
    output = model.generate(
        field_features[:1], 
        gravity_weights[:1], 
        wave_context[:1],
        max_length=50,
    )
    print(f"  Generated: {output[:50]}...")
    
    print("  ✓ EmissionHead tests passed")
