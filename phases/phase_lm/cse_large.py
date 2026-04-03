"""
CSE-Large: Scaled Continuous Semantic Encoder for FLUX-LM.

Improvements over Phase 1 CSE:
- Larger byte embeddings (64 vs 32)
- Wider conv bank (256-512 channels)
- Bigger hidden dimension (1024 vs 512)
- Better position encoding
- ~10M parameters (vs 1.3M)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass
from typing import Dict, Any, Optional


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

CSE_L_CONFIG = {
    'byte_window': 8,
    'byte_stride': 1,
    'byte_embed_dim': 64,
    'conv_channels': [256, 512, 512, 512],
    'hidden_dim': 1024,
    'wave_dims': {
        'phonetic':  64,
        'syntactic': 64,
        'semantic':  256,
        'temporal':  32,
        'intensity': 16,
    },
    'interference_radius': 4,
    'dropout': 0.1,
}


# ─────────────────────────────────────────────
# Wave Data Structure
# ─────────────────────────────────────────────

@dataclass
class SemanticWave:
    """Semantic wave with 432 dimensions."""
    phonetic:  Tensor    # [seq_len, 64]
    syntactic: Tensor    # [seq_len, 64]
    semantic:  Tensor    # [seq_len, 256]
    temporal:  Tensor    # [seq_len, 32]
    intensity: Tensor    # [seq_len, 16]

    @property
    def full(self) -> Tensor:
        """Concatenate all dimensions → [seq_len, 432]"""
        return torch.cat([
            self.phonetic, self.syntactic, self.semantic,
            self.temporal, self.intensity
        ], dim=-1)

    @property
    def seq_len(self) -> int:
        return self.semantic.shape[0]

    def to(self, device) -> 'SemanticWave':
        return SemanticWave(
            phonetic=self.phonetic.to(device),
            syntactic=self.syntactic.to(device),
            semantic=self.semantic.to(device),
            temporal=self.temporal.to(device),
            intensity=self.intensity.to(device),
        )


# ─────────────────────────────────────────────
# Causal Conv1d (Left-padded only)
# ─────────────────────────────────────────────

class CausalConv1d(nn.Module):
    """
    Causal convolution that only sees past context.
    Critical for autoregressive generation - ensures waves don't change
    when new bytes are appended.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1  # Left padding only
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [batch, in_channels, seq_len]
        Returns:
            [batch, out_channels, seq_len]
        """
        # Pad only on the left (past)
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


# ─────────────────────────────────────────────
# Causal Multi-Scale Conv Bank
# ─────────────────────────────────────────────

class CSEConvBankLarge(nn.Module):
    """
    Causal multi-scale convolutional bank.
    Uses left-only padding so waves remain stable during generation.
    
    Key difference from bidirectional: adding a new byte only affects
    the NEW position's wave, not previous positions.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: list = [256, 512, 512, 512],
        out_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Multi-kernel CAUSAL convolutions: 1, 3, 5, 7, 9
        kernels = [1, 3, 5, 7, 9]
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i, k in enumerate(kernels):
            ch = hidden_channels[min(i, len(hidden_channels)-1)]
            # Use CausalConv1d instead of regular Conv1d
            self.convs.append(CausalConv1d(in_channels, ch, kernel_size=k))
            self.norms.append(nn.LayerNorm(ch))
        
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        total_channels = sum([
            hidden_channels[min(i, len(hidden_channels)-1)] 
            for i in range(len(kernels))
        ])
        
        self.project = nn.Sequential(
            nn.Linear(total_channels, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [batch, seq_len, in_channels]
        Returns:
            [batch, seq_len, out_dim]
        """
        x_t = x.transpose(1, 2)  # [batch, in_channels, seq_len]

        conv_outs = []
        for conv, norm in zip(self.convs, self.norms):
            c = conv(x_t)  # Causal conv: [batch, channels, seq_len]
            c = c.transpose(1, 2)  # [batch, seq_len, channels]
            c = norm(c)
            c = self.act(c)
            c = self.dropout(c)
            conv_outs.append(c)

        # Concatenate all scales
        out = torch.cat(conv_outs, dim=-1)  # [batch, seq_len, total_channels]
        out = self.project(out)  # [batch, seq_len, out_dim]
        return out


# ─────────────────────────────────────────────
# Position Encoding (Rotary)
# ─────────────────────────────────────────────

class RotaryPositionEncoding(nn.Module):
    """Rotary Position Embedding (RoPE) for better position awareness."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute position embeddings
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos', emb.cos())
        self.register_buffer('sin', emb.sin())
    
    def forward(self, x: Tensor, seq_len: int) -> Tensor:
        """Apply rotary position encoding."""
        return x * self.cos[:seq_len] + self._rotate_half(x) * self.sin[:seq_len]
    
    def _rotate_half(self, x: Tensor) -> Tensor:
        x1, x2 = x[..., :self.dim//2], x[..., self.dim//2:]
        return torch.cat([-x2, x1], dim=-1)


# ─────────────────────────────────────────────
# Causal Interference Layer
# ─────────────────────────────────────────────

class InterferenceLayer(nn.Module):
    """
    Causal wave interference: only PAST waves affect current position.
    This ensures waves remain stable when new bytes are appended.
    
    Constructive (similar) → amplify
    Destructive (opposite) → cancel
    """
    
    def __init__(self, dim: int, radius: int = 4, scale: float = 0.1):
        super().__init__()
        self.radius = radius
        self.scale = scale
        
        # Learnable decay rates per distance
        self.decay = nn.Parameter(torch.ones(radius) * 0.5)
    
    def forward(self, waves: Tensor) -> Tensor:
        """
        Args:
            waves: [batch, seq_len, dim]
        Returns:
            [batch, seq_len, dim] with causal interference applied
        """
        batch, seq_len, dim = waves.shape
        interference = torch.zeros_like(waves)
        
        for offset in range(1, min(self.radius + 1, seq_len)):
            decay = torch.sigmoid(self.decay[offset - 1])
            
            # CAUSAL ONLY: past positions (w1) affect current positions (w2)
            # Position i is affected by position i-offset (past)
            w1 = waves[:, :-offset]  # Past waves
            w2 = waves[:, offset:]   # Current waves
            
            # How similar is current wave to past wave?
            cos_sim = F.cosine_similarity(w2, w1, dim=-1).unsqueeze(-1)
            
            # Add interference from past to current (causal direction only)
            interference[:, offset:] += cos_sim * decay * w1
            
            # REMOVED: Non-causal interference that would affect past positions
            # interference[:, :-offset] += ... (this was the bug!)
        
        return waves + self.scale * interference


# ─────────────────────────────────────────────
# CSE-Large Main Module
# ─────────────────────────────────────────────

class CSELarge(nn.Module):
    """
    Scaled Continuous Semantic Encoder.
    
    Input:  Raw UTF-8 bytes or text string
    Output: SemanticWave with shape [seq_len, 432]
    
    ~10M parameters
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        config = config or CSE_L_CONFIG
        
        self.byte_window = config['byte_window']
        self.byte_stride = config['byte_stride']
        self.byte_embed_dim = config['byte_embed_dim']
        self.hidden_dim = config['hidden_dim']
        self.wave_dims = config['wave_dims']
        self.interference_radius = config['interference_radius']
        self.dropout = config['dropout']
        
        self.total_wave_dim = sum(self.wave_dims.values())  # 432
        
        # Byte embedding
        self.byte_embed = nn.Embedding(256, self.byte_embed_dim)
        
        # Position encoding for bytes within window
        self.window_pos = nn.Parameter(
            torch.randn(1, self.byte_window, self.byte_embed_dim) * 0.02
        )
        
        # Conv bank
        in_channels = self.byte_window * self.byte_embed_dim
        self.conv_bank = CSEConvBankLarge(
            in_channels=in_channels,
            hidden_channels=config['conv_channels'],
            out_dim=self.hidden_dim,
            dropout=self.dropout,
        )
        
        # Wave projections (separate for each dimension)
        self.wave_projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim // 2, dim),
            )
            for name, dim in self.wave_dims.items()
        })
        
        # Temporal position encoding (rotary)
        self.rope = RotaryPositionEncoding(
            dim=self.wave_dims['temporal'],
            max_seq_len=2048,
        )
        
        # Interference layer
        self.interference = InterferenceLayer(
            dim=self.total_wave_dim,
            radius=self.interference_radius,
        )
        
        # Final normalization
        self.final_norm = nn.LayerNorm(self.total_wave_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    @property
    def device(self):
        return self.byte_embed.weight.device
    
    def text_to_bytes(self, text: str) -> Tensor:
        """Convert text to byte tensor."""
        byte_vals = list(text.encode('utf-8'))
        return torch.tensor(byte_vals, dtype=torch.long, device=self.device)
    
    def bytes_to_windows(self, bytes_tensor: Tensor) -> Tensor:
        """
        Causal sliding window over bytes.
        Each position i only sees bytes [i-window+1, ..., i] (past and current).
        This ensures adding new bytes doesn't change existing position waves.
        """
        seq_len = bytes_tensor.shape[-1]
        
        # CAUSAL: All padding on left (past context), none on right (no future)
        pad_left = self.byte_window - 1
        pad_right = 0
        
        # Handle batched or unbatched input
        if bytes_tensor.dim() == 1:
            padded = F.pad(bytes_tensor, (pad_left, pad_right), value=0)
            windows = padded.unfold(0, self.byte_window, self.byte_stride)
            return windows[:seq_len]
        else:
            padded = F.pad(bytes_tensor, (pad_left, pad_right), value=0)
            windows = padded.unfold(-1, self.byte_window, self.byte_stride)
            return windows[..., :seq_len, :]
    
    def encode_bytes(self, bytes_tensor: Tensor) -> SemanticWave:
        """
        Encode byte tensor to semantic wave.
        
        Args:
            bytes_tensor: [seq_len] or [batch, seq_len] byte values
        Returns:
            SemanticWave
        """
        # Handle unbatched input
        unbatched = bytes_tensor.dim() == 1
        if unbatched:
            bytes_tensor = bytes_tensor.unsqueeze(0)
        
        batch_size, seq_len = bytes_tensor.shape
        
        if seq_len == 0:
            return self._empty_wave(batch_size)
        
        # Bytes → windows
        windows = self.bytes_to_windows(bytes_tensor)  # [batch, seq_len, window]
        
        # Embed bytes
        embedded = self.byte_embed(windows)  # [batch, seq_len, window, embed_dim]
        embedded = embedded + self.window_pos
        
        # Flatten window dimension
        embedded_flat = embedded.reshape(batch_size, seq_len, -1)
        
        # Extract patterns
        features = self.conv_bank(embedded_flat)  # [batch, seq_len, hidden_dim]
        
        # Project to wave dimensions
        phonetic = torch.tanh(self.wave_projections['phonetic'](features))
        syntactic = torch.tanh(self.wave_projections['syntactic'](features))
        semantic = torch.tanh(self.wave_projections['semantic'](features))
        temporal_base = torch.tanh(self.wave_projections['temporal'](features))
        intensity = torch.sigmoid(self.wave_projections['intensity'](features))
        
        # Apply rotary position encoding to temporal
        temporal = self.rope(temporal_base, seq_len)
        
        # Build wave
        wave_full = torch.cat([phonetic, syntactic, semantic, temporal, intensity], dim=-1)
        
        # Apply interference
        wave_full = self.interference(wave_full)
        
        # Normalize
        wave_full = self.final_norm(wave_full)
        
        # Split back to components
        if unbatched:
            wave_full = wave_full.squeeze(0)
        
        return self._split_wave(wave_full)
    
    def encode(self, text: str) -> SemanticWave:
        """Encode text string to semantic wave."""
        bytes_tensor = self.text_to_bytes(text)
        return self.encode_bytes(bytes_tensor)
    
    def forward(self, x) -> SemanticWave:
        """
        Forward pass.
        
        Args:
            x: str, bytes, or Tensor of byte values
        Returns:
            SemanticWave
        """
        if isinstance(x, str):
            return self.encode(x)
        elif isinstance(x, bytes):
            return self.encode(x.decode('utf-8'))
        else:
            return self.encode_bytes(x)
    
    def _split_wave(self, full_wave: Tensor) -> SemanticWave:
        """Split full wave tensor back to SemanticWave."""
        dims = self.wave_dims
        splits = [dims['phonetic'], dims['syntactic'], dims['semantic'],
                  dims['temporal'], dims['intensity']]
        parts = torch.split(full_wave, splits, dim=-1)
        return SemanticWave(*parts)
    
    def _empty_wave(self, batch_size: int) -> SemanticWave:
        """Return empty wave for zero-length input."""
        dev = self.device
        return SemanticWave(
            phonetic=torch.zeros(batch_size, 0, self.wave_dims['phonetic'], device=dev),
            syntactic=torch.zeros(batch_size, 0, self.wave_dims['syntactic'], device=dev),
            semantic=torch.zeros(batch_size, 0, self.wave_dims['semantic'], device=dev),
            temporal=torch.zeros(batch_size, 0, self.wave_dims['temporal'], device=dev),
            intensity=torch.zeros(batch_size, 0, self.wave_dims['intensity'], device=dev),
        )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test
    model = CSELarge()
    print(f"CSE-Large parameters: {count_parameters(model):,}")
    
    wave = model.encode("Hello, world!")
    print(f"Wave shape: [{wave.seq_len}, {wave.full.shape[-1]}]")
