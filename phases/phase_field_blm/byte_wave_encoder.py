"""
Byte Wave Encoder: Minimal parameter encoding from bytes to wave space.

This is the ONLY learned component in Field-Based BLM.
Total parameters: ~108K (256 × 432 byte embedding)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from dataclasses import dataclass


@dataclass
class ByteWaveConfig:
    """Configuration for ByteWaveEncoder."""
    wave_dim: int = 432          # Wave dimension (matches FLUX standard)
    max_seq_len: int = 2048      # Maximum sequence length
    dropout: float = 0.0         # No dropout by default (minimal params)
    use_learned_pos: bool = False  # Use sinusoidal (no params) by default


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding.
    NO PARAMETERS - pure math.
    """
    
    def __init__(self, dim: int, max_len: int = 2048):
        super().__init__()
        self.dim = dim
        
        # Precompute positional encodings
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, dim]
        Returns:
            [batch, seq_len, dim] with positional encoding added
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0)


class CausalContextAggregator(nn.Module):
    """
    Aggregate context bytes into a single query wave.
    Uses exponential decay weighting (no parameters).
    """
    
    def __init__(self, wave_dim: int, decay: float = 0.9):
        super().__init__()
        self.wave_dim = wave_dim
        self.decay = decay
    
    def forward(self, waves: torch.Tensor) -> torch.Tensor:
        """
        Aggregate sequence of waves into context wave.
        
        Args:
            waves: [batch, seq_len, wave_dim]
        Returns:
            [batch, wave_dim] context wave
        """
        batch, seq_len, dim = waves.shape
        
        # Exponential decay weights (recent positions matter more)
        positions = torch.arange(seq_len, device=waves.device, dtype=waves.dtype)
        weights = self.decay ** (seq_len - 1 - positions)
        weights = weights / weights.sum()  # Normalize
        
        # Weighted sum
        context = torch.einsum('bsd,s->bd', waves, weights)
        
        return context


class ByteWaveEncoder(nn.Module):
    """
    Minimal parameter encoder: bytes → wave space.
    
    This is the ONLY learned component in Field-Based BLM.
    
    Parameters:
        - Byte embedding: 256 × wave_dim = 108K params (for wave_dim=432)
        - Total: ~100K params
    
    Compare to:
        - FLUX-LM CSE: 12.7M params
        - FLUX-LM total: 141M params
    """
    
    def __init__(self, config: ByteWaveConfig = None):
        super().__init__()
        config = config or ByteWaveConfig()
        self.config = config
        self.wave_dim = config.wave_dim
        
        # THE ONLY PARAMETERS: byte embedding
        # 256 bytes × 432 dimensions = 110,592 params
        self.byte_embed = nn.Embedding(256, config.wave_dim)
        
        # Positional encoding (NO PARAMETERS - fixed sinusoidal)
        self.pos_enc = SinusoidalPositionalEncoding(
            config.wave_dim, 
            config.max_seq_len
        )
        
        # Context aggregation (NO PARAMETERS - exponential decay)
        self.aggregator = CausalContextAggregator(config.wave_dim)
        
        # Optional normalization (1 param for scale)
        self.norm = nn.LayerNorm(config.wave_dim, elementwise_affine=False)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize byte embeddings for good wave properties."""
        # Initialize with small random values
        nn.init.normal_(self.byte_embed.weight, mean=0, std=0.02)
        
        # Make common bytes (ASCII letters/numbers) more distinct
        with torch.no_grad():
            # Boost ASCII printable range
            self.byte_embed.weight[32:127] *= 1.5
    
    @property
    def device(self):
        return self.byte_embed.weight.device
    
    @property
    def num_params(self) -> int:
        """Count learnable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def encode_bytes(self, bytes_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode byte sequence to wave sequence.
        
        Args:
            bytes_tensor: [batch, seq_len] or [seq_len] byte values
        Returns:
            [batch, seq_len, wave_dim] wave sequence
        """
        # Handle unbatched input
        if bytes_tensor.dim() == 1:
            bytes_tensor = bytes_tensor.unsqueeze(0)
        
        # Embed bytes
        waves = self.byte_embed(bytes_tensor)  # [batch, seq_len, wave_dim]
        
        # Add positional encoding
        waves = self.pos_enc(waves)
        
        # Normalize
        waves = self.norm(waves)
        
        return waves
    
    def encode_context(self, bytes_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode byte sequence to single context wave.
        
        Args:
            bytes_tensor: [batch, seq_len] or [seq_len] byte values
        Returns:
            [batch, wave_dim] context wave for field query
        """
        waves = self.encode_bytes(bytes_tensor)
        context = self.aggregator(waves)
        return context
    
    def forward(self, bytes_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode bytes to context wave.
        
        Args:
            bytes_tensor: [batch, seq_len] byte values
        Returns:
            [batch, wave_dim] context wave
        """
        return self.encode_context(bytes_tensor)
    
    def text_to_bytes(self, text: str) -> torch.Tensor:
        """Convert text string to byte tensor."""
        return torch.tensor(
            list(text.encode('utf-8')), 
            dtype=torch.long, 
            device=self.device
        )
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text string to context wave."""
        bytes_tensor = self.text_to_bytes(text)
        return self.encode_context(bytes_tensor)


# ─────────────────────────────────────────────
# Parameter count verification
# ─────────────────────────────────────────────

def verify_params():
    """Verify we have minimal parameters."""
    encoder = ByteWaveEncoder()
    
    total = encoder.num_params
    embed_params = encoder.byte_embed.weight.numel()
    
    print(f"ByteWaveEncoder Parameters:")
    print(f"  Byte embedding: {embed_params:,} ({embed_params/1e6:.3f}M)")
    print(f"  Total:          {total:,} ({total/1e6:.3f}M)")
    print(f"")
    print(f"Comparison:")
    print(f"  FLUX-LM CSE:    12,700,000 (12.7M)")
    print(f"  Reduction:      {12_700_000 / total:.0f}x smaller")


if __name__ == '__main__':
    verify_params()
    
    # Test encoding
    encoder = ByteWaveEncoder()
    
    text = "The scientist discovered"
    context_wave = encoder.encode_text(text)
    
    print(f"\nTest encoding:")
    print(f"  Input: '{text}'")
    print(f"  Bytes: {len(text.encode('utf-8'))}")
    print(f"  Context wave shape: {context_wave.shape}")
    print(f"  Context wave norm: {context_wave.norm().item():.4f}")
