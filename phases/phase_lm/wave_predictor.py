"""
WavePredictor: Main Transformer Backbone for FLUX-LM.

This is the NEW component that enables autoregressive generation.
Operates on causal waves (608D) and predicts the next wave.

Architecture:
- 12 transformer layers
- 12 attention heads
- 768 hidden dimension
- ~100M parameters (GPT-2 small equivalent)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Any, Optional, Tuple


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

WAVE_PREDICTOR_CONFIG = {
    'input_dim': 608,               # From CWC output
    'hidden_dim': 768,              # GPT-2 small hidden size
    'n_heads': 12,                  # GPT-2 small heads
    'n_layers': 12,                 # GPT-2 small layers
    'ff_dim': 3072,                 # 4x hidden
    'max_seq_len': 1024,
    'dropout': 0.1,
    'layer_norm_eps': 1e-5,
}


# ─────────────────────────────────────────────
# Rotary Position Encoding
# ─────────────────────────────────────────────

class RotaryPositionEncoding(nn.Module):
    """RoPE for transformer attention."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # [seq_len, dim/2]
        # Keep as [seq_len, dim/2] - don't double
        self.register_buffer('cos_cached', freqs.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer('sin_cached', freqs.sin().unsqueeze(0).unsqueeze(0), persistent=False)
    
    def forward(self, x: Tensor, seq_len: int) -> Tuple[Tensor, Tensor]:
        """Return cos and sin for position encoding."""
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        return self.cos_cached[:, :, :seq_len], self.sin_cached[:, :, :seq_len]
    
    @staticmethod
    def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Apply rotary embedding to tensor."""
        dim = x.shape[-1] // 2
        x1, x2 = x[..., :dim], x[..., dim:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ─────────────────────────────────────────────
# Multi-Head Self-Attention
# ─────────────────────────────────────────────

class WaveAttention(nn.Module):
    """
    Multi-head self-attention with RoPE and optional KV cache.
    """
    
    def __init__(
        self,
        dim: int,
        n_heads: int = 12,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionEncoding(self.head_dim, max_seq_len)
    
    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        past_kv: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Args:
            x: [batch, seq_len, dim]
            mask: Optional attention mask
            past_kv: Optional cached (key, value) for generation
            use_cache: Whether to return updated cache
        
        Returns:
            output: [batch, seq_len, dim]
            present_kv: Optional (key, value) cache
        """
        batch, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        start_pos = past_kv[0].shape[2] if past_kv is not None else 0
        cos, sin = self.rope(q, start_pos + seq_len)
        cos = cos[:, :, start_pos:start_pos + seq_len]
        sin = sin[:, :, start_pos:start_pos + seq_len]
        
        q = self.rope.apply_rotary(q, cos, sin)
        k = self.rope.apply_rotary(k, cos, sin)
        
        # Handle KV cache
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        
        present_kv = (k, v) if use_cache else None
        
        # Compute attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        out = self.out_proj(out)
        
        return out, present_kv


# ─────────────────────────────────────────────
# Feedforward Network
# ─────────────────────────────────────────────

class WaveFFN(nn.Module):
    """Feedforward network with GELU activation."""
    
    def __init__(self, dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# ─────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────

class WaveTransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture."""
    
    def __init__(
        self,
        dim: int,
        n_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.attn = WaveAttention(dim, n_heads, dropout, max_seq_len)
        
        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.ffn = WaveFFN(dim, ff_dim, dropout)
    
    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        past_kv: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        # Pre-norm attention
        residual = x
        x = self.norm1(x)
        x, present_kv = self.attn(x, mask, past_kv, use_cache)
        x = residual + x
        
        # Pre-norm FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x, present_kv


# ─────────────────────────────────────────────
# WavePredictor Main Module
# ─────────────────────────────────────────────

class WavePredictor(nn.Module):
    """
    Main transformer backbone for next-wave prediction.
    
    Input:  Causal waves [batch, seq_len, 608] from CWC
    Output: Predicted next wave [batch, 1, 608]
    
    ~100M parameters (GPT-2 small equivalent)
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        config = config or WAVE_PREDICTOR_CONFIG
        
        self.input_dim = config['input_dim']  # 608
        self.hidden_dim = config['hidden_dim']  # 768
        self.n_heads = config['n_heads']  # 12
        self.n_layers = config['n_layers']  # 12
        self.ff_dim = config['ff_dim']  # 3072
        self.max_seq_len = config['max_seq_len']  # 1024
        self.dropout = config['dropout']
        self.layer_norm_eps = config.get('layer_norm_eps', 1e-5)
        
        # Input projection: wave space → hidden space
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim, eps=self.layer_norm_eps),
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            WaveTransformerBlock(
                dim=self.hidden_dim,
                n_heads=self.n_heads,
                ff_dim=self.ff_dim,
                dropout=self.dropout,
                max_seq_len=self.max_seq_len,
                layer_norm_eps=self.layer_norm_eps,
            )
            for _ in range(self.n_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(self.hidden_dim, eps=self.layer_norm_eps)
        
        # Output projection: hidden space → wave space
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.input_dim),
        )
        
        # Dropout
        self.drop = nn.Dropout(self.dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using GPT-2 style initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Scale output projections by layer depth
        for i, block in enumerate(self.blocks):
            block.attn.out_proj.weight.data *= (2 * self.n_layers) ** -0.5
            block.ffn.fc2.weight.data *= (2 * self.n_layers) ** -0.5
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """Create causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(
        self,
        x: Tensor,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[list]]:
        """
        Predict next wave from context.
        
        Args:
            x: [batch, seq_len, 608] causal waves
            past_key_values: Optional list of (k, v) caches per layer
            use_cache: Whether to return updated caches
        
        Returns:
            next_wave: [batch, 1, 608] predicted next wave (last position)
            present_key_values: Optional list of (k, v) caches
        """
        batch, seq_len, _ = x.shape
        
        # Project to hidden space
        x = self.input_proj(x)
        x = self.drop(x)
        
        # Create causal mask
        if past_key_values is None:
            mask = self._create_causal_mask(seq_len, x.device)
        else:
            # For generation with cache, only attend to new position
            total_len = past_key_values[0][0].shape[2] + seq_len
            mask = self._create_causal_mask(total_len, x.device)
            mask = mask[-seq_len:]
        
        # Apply transformer blocks
        present_key_values = [] if use_cache else None
        
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = block(x, mask, past_kv, use_cache)
            
            if use_cache:
                present_key_values.append(present_kv)
        
        # Final norm
        x = self.final_norm(x)
        
        # Project back to wave space
        x = self.output_proj(x)
        
        return x, present_key_values
    
    def predict_next(
        self,
        context_waves: Tensor,
        past_key_values: Optional[list] = None,
    ) -> Tuple[Tensor, Optional[list]]:
        """
        Predict the next wave given context.
        
        Args:
            context_waves: [batch, seq_len, 608] context waves
            past_key_values: Optional KV cache
        
        Returns:
            next_wave: [batch, 608] predicted next wave
            present_key_values: Updated KV cache
        """
        output, present_kv = self.forward(context_waves, past_key_values, use_cache=True)
        
        # Return only the last position
        next_wave = output[:, -1]  # [batch, 608]
        
        return next_wave, present_kv
    
    def generate_wave_sequence(
        self,
        initial_waves: Tensor,
        num_waves: int,
    ) -> Tensor:
        """
        Autoregressively generate a sequence of waves.
        
        Args:
            initial_waves: [batch, seq_len, 608] initial context
            num_waves: Number of new waves to generate
        
        Returns:
            [batch, seq_len + num_waves, 608] full wave sequence
        """
        waves = initial_waves
        past_kv = None
        
        for _ in range(num_waves):
            if past_kv is None:
                input_waves = waves
            else:
                input_waves = waves[:, -1:]  # Only the last wave
            
            next_wave, past_kv = self.predict_next(input_waves, past_kv)
            waves = torch.cat([waves, next_wave.unsqueeze(1)], dim=1)
        
        return waves


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test
    model = WavePredictor()
    print(f"WavePredictor parameters: {count_parameters(model):,}")
    
    # Test forward
    x = torch.randn(2, 10, 608)  # [batch, seq_len, 608]
    output, _ = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test predict next
    next_wave, cache = model.predict_next(x)
    print(f"Next wave shape: {next_wave.shape}")
    
    # Test generation
    generated = model.generate_wave_sequence(x, num_waves=5)
    print(f"Generated shape: {generated.shape}")
