"""
FluxLM Scaling: Upscale 141M model to 1B parameters.

Scaling strategy:
- CSE-Large → CSE-XL: 10M → 25M (wider convs, deeper)
- CWC-Large → CWC-XL: 5M → 15M (more layers, wider)
- WavePredictor → WavePredictor-XL: 100M → 900M (24 layers, 2048 hidden)
- WaveDecoder-Large → WaveDecoder-XL: 9M → 60M (deeper, wider)

Total: ~1B parameters

Weight transfer:
- Smaller weights are copied to corresponding positions in larger tensors
- New weights are initialized with scaled initialization
"""

import os
import shutil
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# Import base components
from cse_large import CSELarge, CSE_L_CONFIG, SemanticWave
from cwc_large import CWCLarge, CWC_L_CONFIG
from wave_predictor import WavePredictor, WAVE_PREDICTOR_CONFIG
from wave_decoder_large import WaveDecoderLarge, WAVE_DECODER_L_CONFIG
from flux_lm import FluxLM, GenerationConfig, FLUX_LM_CONFIG


# ─────────────────────────────────────────────
# 1B Configuration
# ─────────────────────────────────────────────

CSE_XL_CONFIG = {
    'byte_window': 8,
    'byte_stride': 1,
    'byte_embed_dim': 128,              # 64 → 128
    'conv_channels': [512, 1024, 1024, 1024],  # Doubled
    'hidden_dim': 2048,                 # 1024 → 2048
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

CWC_XL_CONFIG = {
    'wave_dim': 432,
    'causal_dim': 176,
    'hidden_dim': 2048,                 # 1024 → 2048
    'n_heads': 16,                      # 8 → 16
    'n_layers': 4,                      # 2 → 4
    'dropout': 0.1,
    'max_seq_len': 2048,
    'order_aware': True,
    'order_hidden_dim': 1024,           # 512 → 1024
    'temporal_mlp_layers': 4,           # 3 → 4
}

WAVE_PREDICTOR_XL_CONFIG = {
    'input_dim': 608,
    'hidden_dim': 2048,                 # 768 → 2048
    'n_heads': 16,                      # 12 → 16
    'n_layers': 24,                     # 12 → 24
    'ff_dim': 8192,                     # 3072 → 8192 (4x hidden)
    'max_seq_len': 2048,
    'dropout': 0.1,
    'layer_norm_eps': 1e-5,
}

WAVE_DECODER_XL_CONFIG = {
    'input_dim': 608,
    'hidden_dim': 4096,                 # 2048 → 4096
    'n_layers': 6,                      # 4 → 6
    'output_dim': 256,
    'dropout': 0.1,
    'use_multi_scale': True,
    'scales': [1, 2, 4],
}

FLUX_LM_1B_CONFIG = {
    'cse': CSE_XL_CONFIG,
    'cwc': CWC_XL_CONFIG,
    'predictor': WAVE_PREDICTOR_XL_CONFIG,
    'decoder': WAVE_DECODER_XL_CONFIG,
    'max_seq_len': 2048,
    'gradient_checkpointing': True,     # Enable for memory efficiency
}


# ─────────────────────────────────────────────
# CSE-XL (Scaled CSE)
# ─────────────────────────────────────────────

class CausalConv1d(nn.Module):
    """Causal convolution that only sees past context."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0)
    
    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class CSE_XL(nn.Module):
    """
    CSE-XL: Scaled Continuous Semantic Encoder (~25M params).
    Wider and deeper than CSE-Large.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        config = config or CSE_XL_CONFIG
        
        self.byte_window = config['byte_window']
        self.byte_stride = config['byte_stride']
        self.byte_embed_dim = config['byte_embed_dim']
        self.hidden_dim = config['hidden_dim']
        self.wave_dims = config['wave_dims']
        self.total_wave_dim = sum(self.wave_dims.values())
        self.interference_radius = config['interference_radius']
        
        # Byte embedding
        self.byte_embedding = nn.Embedding(256, self.byte_embed_dim)
        
        # Position encoding
        self.position_encoding = nn.Parameter(
            torch.randn(1, 512, self.byte_embed_dim * self.byte_window) * 0.02
        )
        
        # Causal conv bank (wider)
        conv_channels = config['conv_channels']
        self.conv_bank = nn.ModuleList()
        in_ch = self.byte_embed_dim * self.byte_window
        
        for i, out_ch in enumerate(conv_channels):
            kernel_size = 3 + i * 2
            self.conv_bank.append(nn.Sequential(
                CausalConv1d(in_ch, out_ch, kernel_size),
                nn.BatchNorm1d(out_ch),  # Use BatchNorm1d for [batch, channels, seq] format
                nn.GELU(),
                nn.Dropout(config['dropout']),
            ))
            in_ch = out_ch
        
        # Extra projection layers for XL
        self.deep_proj = nn.Sequential(
            nn.Linear(conv_channels[-1], self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        
        # Wave projections
        self.wave_projections = nn.ModuleDict()
        for name, dim in self.wave_dims.items():
            self.wave_projections[name] = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.GELU(),
                nn.Linear(self.hidden_dim // 2, dim),
            )
        
        # Interference network
        self.interference_net = nn.Sequential(
            nn.Linear(self.total_wave_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.total_wave_dim),
            nn.Tanh(),
        )
    
    def encode(self, text: str) -> SemanticWave:
        """Encode text to semantic wave."""
        bytes_list = list(text.encode('utf-8'))
        bytes_tensor = torch.tensor(bytes_list, dtype=torch.long, device=self.byte_embedding.weight.device)
        return self.encode_bytes(bytes_tensor)
    
    def encode_bytes(self, bytes_tensor: Tensor) -> SemanticWave:
        """Encode byte tensor to semantic wave."""
        if bytes_tensor.dim() == 1:
            bytes_tensor = bytes_tensor.unsqueeze(0)
        
        batch_size, seq_len = bytes_tensor.shape
        device = bytes_tensor.device
        
        # Embed bytes
        embedded = self.byte_embedding(bytes_tensor)  # [batch, seq, embed_dim]
        
        # Create sliding windows
        if seq_len < self.byte_window:
            pad_size = self.byte_window - seq_len
            embedded = F.pad(embedded, (0, 0, pad_size, 0))
            seq_len = self.byte_window
        
        windows = []
        for i in range(seq_len - self.byte_window + 1):
            window = embedded[:, i:i+self.byte_window].reshape(batch_size, -1)
            windows.append(window)
        
        if not windows:
            windows = [embedded.reshape(batch_size, -1)]
        
        windowed = torch.stack(windows, dim=1)  # [batch, num_windows, window_features]
        
        # Add position encoding
        pos_len = min(windowed.shape[1], self.position_encoding.shape[1])
        windowed[:, :pos_len] = windowed[:, :pos_len] + self.position_encoding[:, :pos_len]
        
        # Conv bank
        x = windowed.transpose(1, 2)  # [batch, features, seq]
        for conv_block in self.conv_bank:
            x = conv_block(x)
        x = x.transpose(1, 2)  # [batch, seq, features]
        
        # Deep projection
        x = self.deep_proj(x)
        
        # Project to wave dimensions
        waves = {}
        for name, proj in self.wave_projections.items():
            waves[name] = proj(x)
        
        # Apply interference
        full_wave = torch.cat([waves[k] for k in self.wave_dims.keys()], dim=-1)
        full_wave = self._apply_interference(full_wave)
        
        # Split back
        idx = 0
        for name, dim in self.wave_dims.items():
            waves[name] = full_wave[..., idx:idx+dim]
            idx += dim
        
        # Remove batch dim if input was unbatched
        if batch_size == 1:
            return SemanticWave(
                phonetic=waves['phonetic'].squeeze(0),
                syntactic=waves['syntactic'].squeeze(0),
                semantic=waves['semantic'].squeeze(0),
                temporal=waves['temporal'].squeeze(0),
                intensity=waves['intensity'].squeeze(0),
            )
        
        return SemanticWave(**waves)
    
    def _apply_interference(self, waves: Tensor) -> Tensor:
        """Apply wave interference between nearby positions."""
        batch, seq_len, dim = waves.shape
        
        if seq_len <= 1:
            return waves
        
        output = waves.clone()
        for i in range(seq_len):
            start = max(0, i - self.interference_radius)
            end = min(seq_len, i + self.interference_radius + 1)
            
            if end - start > 1:
                neighbors = waves[:, start:end].mean(dim=1)
                combined = torch.cat([waves[:, i], neighbors], dim=-1)
                interference = self.interference_net(combined)
                output[:, i] = waves[:, i] + 0.1 * interference
        
        return output


# ─────────────────────────────────────────────
# CWC-XL (Scaled CWC)
# ─────────────────────────────────────────────

class RotaryPositionEncoding(nn.Module):
    """RoPE for position-aware causal processing."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0))
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0))
    
    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[1]
        cos = self.cos_cached[:, :seq_len]
        sin = self.sin_cached[:, :seq_len]
        return x * cos + self._rotate_half(x) * sin
    
    def _rotate_half(self, x: Tensor) -> Tensor:
        x1, x2 = x[..., :self.dim//2], x[..., self.dim//2:]
        return torch.cat([-x2, x1], dim=-1)


class CWC_XL(nn.Module):
    """
    CWC-XL: Scaled Causal Wave Chainer (~15M params).
    More layers and wider hidden dimensions.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        config = config or CWC_XL_CONFIG
        
        self.wave_dim = config['wave_dim']
        self.causal_dim = config['causal_dim']
        self.hidden_dim = config['hidden_dim']
        self.output_dim = self.wave_dim + self.causal_dim  # 608
        
        # Input projection
        self.input_proj = nn.Linear(self.wave_dim, self.hidden_dim)
        
        # Temporal MLP (deeper)
        temporal_layers = []
        for i in range(config['temporal_mlp_layers']):
            if i == 0:
                temporal_layers.append(nn.Linear(self.hidden_dim, config['order_hidden_dim']))
            else:
                temporal_layers.append(nn.Linear(config['order_hidden_dim'], config['order_hidden_dim']))
            temporal_layers.append(nn.LayerNorm(config['order_hidden_dim']))
            temporal_layers.append(nn.GELU())
            temporal_layers.append(nn.Dropout(config['dropout']))
        
        self.temporal_mlp = nn.Sequential(*temporal_layers)
        
        # RoPE
        self.rope = RotaryPositionEncoding(config['order_hidden_dim'], config['max_seq_len'])
        
        # Self-attention layers (more layers)
        self.attn_layers = nn.ModuleList()
        self.ff_layers = nn.ModuleList()
        self.norms1 = nn.ModuleList()
        self.norms2 = nn.ModuleList()
        
        for _ in range(config['n_layers']):
            self.attn_layers.append(
                nn.MultiheadAttention(
                    config['order_hidden_dim'],
                    config['n_heads'],
                    dropout=config['dropout'],
                    batch_first=True
                )
            )
            self.ff_layers.append(nn.Sequential(
                nn.Linear(config['order_hidden_dim'], config['order_hidden_dim'] * 4),
                nn.GELU(),
                nn.Dropout(config['dropout']),
                nn.Linear(config['order_hidden_dim'] * 4, config['order_hidden_dim']),
                nn.Dropout(config['dropout']),
            ))
            self.norms1.append(nn.LayerNorm(config['order_hidden_dim']))
            self.norms2.append(nn.LayerNorm(config['order_hidden_dim']))
        
        # Output projections
        self.causal_proj = nn.Linear(config['order_hidden_dim'], self.causal_dim)
        self.order_head = nn.Linear(config['order_hidden_dim'], 1)
    
    def forward(self, waves: Tensor, return_order_score: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        """Add causal direction to waves."""
        if waves.dim() == 2:
            waves = waves.unsqueeze(0)
        
        batch, seq_len, _ = waves.shape
        
        # Project and add temporal info
        x = self.input_proj(waves)
        x = self.temporal_mlp(x)
        x = self.rope(x)
        
        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=waves.device), diagonal=1).bool()
        
        # Self-attention layers
        for attn, ff, norm1, norm2 in zip(self.attn_layers, self.ff_layers, self.norms1, self.norms2):
            # Pre-norm attention
            x_norm = norm1(x)
            attn_out, _ = attn(x_norm, x_norm, x_norm, attn_mask=mask)
            x = x + attn_out
            
            # Pre-norm FFN
            x = x + ff(norm2(x))
        
        # Project to causal dimensions
        causal_info = self.causal_proj(x)
        
        # Concatenate with original waves
        output = torch.cat([waves, causal_info], dim=-1)
        
        order_score = None
        if return_order_score:
            order_score = self.order_head(x.mean(dim=1)).squeeze(-1)
        
        # Keep batch dimension for downstream compatibility
        return (output, order_score) if return_order_score else output
    
    def get_order_score(self, causal_waves: Tensor) -> Tensor:
        """Get order coherence score."""
        if causal_waves.dim() == 2:
            causal_waves = causal_waves.unsqueeze(0)
        
        waves = causal_waves[..., :self.wave_dim]
        x = self.input_proj(waves)
        x = self.temporal_mlp(x)
        
        return self.order_head(x.mean(dim=1)).squeeze(-1)
    
    def compute_tension(self, wave1: Tensor, wave2: Tensor) -> Tensor:
        """Compute contradiction tension."""
        if wave1.dim() == 2:
            wave1 = wave1.unsqueeze(0)
        if wave2.dim() == 2:
            wave2 = wave2.unsqueeze(0)
        
        mean1 = wave1.mean(dim=1)
        mean2 = wave2.mean(dim=1)
        
        cos_sim = F.cosine_similarity(mean1, mean2, dim=-1)
        tension = (1 - cos_sim) / 2
        
        return tension


# ─────────────────────────────────────────────
# WavePredictor-XL (Main Backbone)
# ─────────────────────────────────────────────

class WaveAttentionXL(nn.Module):
    """Scaled attention with RoPE."""
    
    def __init__(self, dim: int, n_heads: int = 16, dropout: float = 0.1, max_seq_len: int = 2048):
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
        
        # RoPE
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def _get_rotary(self, seq_len: int, device):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq.to(device))
        return freqs.cos().unsqueeze(0).unsqueeze(0), freqs.sin().unsqueeze(0).unsqueeze(0)
    
    def _apply_rotary(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        dim = x.shape[-1] // 2
        x1, x2 = x[..., :dim], x[..., dim:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None, 
                past_kv: Optional[Tuple[Tensor, Tensor]] = None,
                use_cache: bool = False) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        batch, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        start_pos = past_kv[0].shape[2] if past_kv is not None else 0
        cos, sin = self._get_rotary(start_pos + seq_len, x.device)
        cos = cos[:, :, start_pos:start_pos + seq_len]
        sin = sin[:, :, start_pos:start_pos + seq_len]
        
        q = self._apply_rotary(q, cos, sin)
        k = self._apply_rotary(k, cos, sin)
        
        # KV cache
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        
        present_kv = (k, v) if use_cache else None
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.dim)
        out = self.out_proj(out)
        
        return out, present_kv


class WavePredictorXL(nn.Module):
    """
    WavePredictor-XL: Main transformer backbone (~900M params).
    24 layers, 2048 hidden, 16 heads.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        config = config or WAVE_PREDICTOR_XL_CONFIG
        
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.n_heads = config['n_heads']
        self.n_layers = config['n_layers']
        
        # Input projection
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(nn.ModuleDict({
                'attn': WaveAttentionXL(self.hidden_dim, self.n_heads, config['dropout'], config['max_seq_len']),
                'norm1': nn.LayerNorm(self.hidden_dim, eps=config['layer_norm_eps']),
                'ff': nn.Sequential(
                    nn.Linear(self.hidden_dim, config['ff_dim']),
                    nn.GELU(),
                    nn.Dropout(config['dropout']),
                    nn.Linear(config['ff_dim'], self.hidden_dim),
                    nn.Dropout(config['dropout']),
                ),
                'norm2': nn.LayerNorm(self.hidden_dim, eps=config['layer_norm_eps']),
            }))
        
        # Output projection
        self.output_norm = nn.LayerNorm(self.hidden_dim, eps=config['layer_norm_eps'])
        self.output_proj = nn.Linear(self.hidden_dim, self.input_dim)
    
    def forward(self, x: Tensor, past_kvs: Optional[list] = None, 
                use_cache: bool = False) -> Tuple[Tensor, Optional[list]]:
        batch, seq_len, _ = x.shape
        
        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        # Input projection
        x = self.input_proj(x)
        
        present_kvs = [] if use_cache else None
        
        # Transformer layers
        for i, layer in enumerate(self.layers):
            past_kv = past_kvs[i] if past_kvs is not None else None
            
            # Pre-norm attention
            normed = layer['norm1'](x)
            attn_out, present_kv = layer['attn'](normed, mask, past_kv, use_cache)
            x = x + attn_out
            
            if use_cache:
                present_kvs.append(present_kv)
            
            # Pre-norm FFN
            x = x + layer['ff'](layer['norm2'](x))
        
        # Output
        x = self.output_norm(x)
        x = self.output_proj(x)
        
        return x, present_kvs


# ─────────────────────────────────────────────
# WaveDecoder-XL
# ─────────────────────────────────────────────

class WaveDecoderXL(nn.Module):
    """
    WaveDecoder-XL: Scaled decoder (~60M params).
    Deeper and wider than WaveDecoder-Large.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        config = config or WAVE_DECODER_XL_CONFIG
        
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['output_dim']
        self.scales = config.get('scales', [1, 2, 4])
        
        # Multi-scale decoders
        self.scale_decoders = nn.ModuleList()
        for scale in self.scales:
            layers = []
            in_dim = self.input_dim * scale
            
            # Deeper network for each scale
            for i in range(config['n_layers']):
                out_dim = self.hidden_dim if i < config['n_layers'] - 1 else self.hidden_dim // 2
                layers.extend([
                    nn.Linear(in_dim if i == 0 else self.hidden_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.GELU(),
                    nn.Dropout(config['dropout']),
                ])
            
            layers.append(nn.Linear(self.hidden_dim // 2, self.output_dim))
            self.scale_decoders.append(nn.Sequential(*layers))
        
        # Combine scales
        self.combine = nn.Sequential(
            nn.Linear(self.output_dim * len(self.scales), self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.output_dim),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        single = x.dim() == 2
        if single:
            x = x.unsqueeze(1)
        
        batch, seq_len, dim = x.shape
        outputs = []
        
        for scale, decoder in zip(self.scales, self.scale_decoders):
            if scale == 1:
                scale_out = decoder(x)
            else:
                if seq_len >= scale:
                    # Get context from last `scale` positions
                    context = x[:, -scale:].mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
                    scale_input = torch.cat([x] + [context] * (scale - 1), dim=-1)
                else:
                    scale_input = F.pad(x, (0, dim * (scale - 1)))
                scale_out = decoder(scale_input)
            outputs.append(scale_out)
        
        combined = torch.cat(outputs, dim=-1)
        output = self.combine(combined)
        
        if single:
            output = output.squeeze(1)
        
        return output


# ─────────────────────────────────────────────
# FluxLM-1B (Complete Model)
# ─────────────────────────────────────────────

class FluxLM_1B(nn.Module):
    """
    FluxLM-1B: 1 Billion parameter vocabulary-free language model.
    
    Architecture:
    - CSE-XL: ~25M params
    - CWC-XL: ~15M params
    - WavePredictor-XL: ~900M params
    - WaveDecoder-XL: ~60M params
    
    Total: ~1B parameters
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        config = config or FLUX_LM_1B_CONFIG
        
        self.config = config
        self.max_seq_len = config.get('max_seq_len', 2048)
        
        # Initialize components
        self.cse = CSE_XL(config.get('cse', CSE_XL_CONFIG))
        self.cwc = CWC_XL(config.get('cwc', CWC_XL_CONFIG))
        self.predictor = WavePredictorXL(config.get('predictor', WAVE_PREDICTOR_XL_CONFIG))
        self.decoder = WaveDecoderXL(config.get('decoder', WAVE_DECODER_XL_CONFIG))
        
        self.wave_dim = 432
        self.causal_dim = 608
        
        self.gradient_checkpointing = config.get('gradient_checkpointing', True)
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def count_parameters(self) -> Dict[str, int]:
        def count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        return {
            'cse': count(self.cse),
            'cwc': count(self.cwc),
            'predictor': count(self.predictor),
            'decoder': count(self.decoder),
            'total': count(self),
        }
    
    def encode_text(self, text: str) -> Tensor:
        semantic_wave = self.cse.encode(text)
        wave = semantic_wave.full
        causal_wave = self.cwc(wave)
        return causal_wave
    
    def encode_bytes(self, bytes_tensor: Tensor) -> Tensor:
        semantic_wave = self.cse.encode_bytes(bytes_tensor)
        wave = semantic_wave.full
        causal_wave = self.cwc(wave)
        return causal_wave
    
    def forward(
        self,
        input_bytes: Tensor,
        target_bytes: Optional[Tensor] = None,
        return_loss: bool = True,
    ) -> Dict[str, Tensor]:
        batch_size, seq_len = input_bytes.shape
        
        # Encode
        semantic_wave = self.cse.encode_bytes(input_bytes)
        wave = semantic_wave.full
        causal_wave, order_score = self.cwc(wave, return_order_score=True)
        
        # Predict
        predicted_waves, _ = self.predictor(causal_wave)
        
        # Decode
        logits = self.decoder(predicted_waves)
        
        result = {'logits': logits, 'order_score': order_score}
        
        if return_loss and target_bytes is not None:
            loss = F.cross_entropy(
                logits.view(-1, 256),
                target_bytes.view(-1),
                ignore_index=-100,
            )
            result['loss'] = loss
        
        return result
    
    @torch.no_grad()
    def generate(self, prompt: str, config: GenerationConfig = None) -> str:
        was_training = self.training
        self.eval()
        
        config = config or GenerationConfig()
        output_bytes = list(prompt.encode('utf-8'))
        bytes_tensor = torch.tensor(output_bytes, dtype=torch.long, device=self.device)
        
        for _ in range(config.max_new_bytes):
            input_tensor = bytes_tensor.unsqueeze(0)
            
            semantic_wave = self.cse.encode_bytes(input_tensor)
            wave = semantic_wave.full
            causal_waves = self.cwc(wave)
            predicted_waves, _ = self.predictor(causal_waves)
            logits = self.decoder(predicted_waves)
            logits = logits[:, -1]
            
            # Repetition penalty
            if config.repetition_penalty != 1.0:
                for prev_byte in set(output_bytes[-50:]):
                    logits[0, prev_byte] /= config.repetition_penalty
            
            # Sample
            next_byte = self._sample_byte(logits[0], config.temperature, config.top_k, config.top_p)
            
            if next_byte in config.stop_bytes:
                break
            
            output_bytes.append(next_byte)
            bytes_tensor = torch.cat([
                bytes_tensor,
                torch.tensor([next_byte], dtype=torch.long, device=self.device)
            ])
        
        if was_training:
            self.train()
        
        return bytes(output_bytes).decode('utf-8', errors='replace')
    
    def _sample_byte(self, logits: Tensor, temperature: float, top_k: int, top_p: float) -> int:
        if temperature == 0.0:
            return logits.argmax().item()
        
        if temperature != 1.0:
            logits = logits / temperature
        
        if top_k > 0:
            top_k = min(top_k, 256)
            indices_to_remove = logits < torch.topk(logits, top_k)[0][-1]
            logits[indices_to_remove] = float('-inf')
        
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')
        
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).item()
    
    def save(self, path: str):
        checkpoint = {
            'config': self.config,
            'state_dict': self.state_dict(),
            'param_counts': self.count_parameters(),
            'model_class': 'FluxLM_1B',
        }
        torch.save(checkpoint, path)
        print(f"✓ FluxLM-1B saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'FluxLM_1B':
        checkpoint = torch.load(path, map_location=device)
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        print(f"✓ FluxLM-1B loaded from {path}")
        return model


# ─────────────────────────────────────────────
# Weight Transfer Utilities
# ─────────────────────────────────────────────

def transfer_linear_weights(small: nn.Linear, large: nn.Linear, scale_init: float = 0.02):
    """
    Transfer weights from smaller linear layer to larger one.
    Extra dimensions are initialized with scaled random values.
    """
    with torch.no_grad():
        # Get shapes
        small_out, small_in = small.weight.shape
        large_out, large_in = large.weight.shape
        
        # Initialize large with scaled random
        nn.init.normal_(large.weight, mean=0.0, std=scale_init)
        if large.bias is not None:
            nn.init.zeros_(large.bias)
        
        # Copy small weights to top-left corner
        out_copy = min(small_out, large_out)
        in_copy = min(small_in, large_in)
        large.weight[:out_copy, :in_copy] = small.weight[:out_copy, :in_copy]
        
        if small.bias is not None and large.bias is not None:
            large.bias[:out_copy] = small.bias[:out_copy]


def transfer_embedding_weights(small: nn.Embedding, large: nn.Embedding, scale_init: float = 0.02):
    """Transfer embedding weights."""
    with torch.no_grad():
        nn.init.normal_(large.weight, mean=0.0, std=scale_init)
        
        vocab_copy = min(small.num_embeddings, large.num_embeddings)
        dim_copy = min(small.embedding_dim, large.embedding_dim)
        large.weight[:vocab_copy, :dim_copy] = small.weight[:vocab_copy, :dim_copy]


def transfer_layernorm_weights(small: nn.LayerNorm, large: nn.LayerNorm):
    """Transfer LayerNorm weights."""
    with torch.no_grad():
        dim_copy = min(small.normalized_shape[0], large.normalized_shape[0])
        large.weight[:dim_copy] = small.weight[:dim_copy]
        large.bias[:dim_copy] = small.bias[:dim_copy]


# ─────────────────────────────────────────────
# Main Scaling Function
# ─────────────────────────────────────────────

def scale_141m_to_1b(
    source_path: str = 'checkpoints/Flux-LM-141M.pt',
    target_path: str = 'checkpoints/Flux-LM-1B.pt',
    backup_path: str = 'checkpoints/Flux-LM-141M-original.pt',
    device: str = 'cpu',
) -> FluxLM_1B:
    """
    Scale the 141M model to 1B parameters.
    
    1. Creates backup of original 141M model
    2. Loads 141M model
    3. Creates new 1B model
    4. Transfers weights where possible
    5. Saves new 1B model
    
    Args:
        source_path: Path to 141M checkpoint
        target_path: Path to save 1B checkpoint
        backup_path: Path to save backup of original
        device: Device to use
    
    Returns:
        FluxLM_1B: The scaled model
    """
    print("=" * 60)
    print("FluxLM Scaling: 141M → 1B")
    print("=" * 60)
    
    # Step 1: Backup original
    if os.path.exists(source_path) and not os.path.exists(backup_path):
        print(f"\n1. Creating backup: {backup_path}")
        shutil.copy2(source_path, backup_path)
        print(f"   ✓ Backup created")
    else:
        print(f"\n1. Backup already exists or source missing")
    
    # Step 2: Load 141M model
    print(f"\n2. Loading 141M model from {source_path}")
    small_model = FluxLM.load(source_path, device=device)
    small_params = small_model.count_parameters()
    print(f"   Source model: {small_params['total']:,} parameters")
    
    # Step 3: Create 1B model
    print(f"\n3. Creating 1B model architecture")
    large_model = FluxLM_1B(FLUX_LM_1B_CONFIG)
    large_model.to(device)
    large_params = large_model.count_parameters()
    print(f"   Target model: {large_params['total']:,} parameters")
    
    # Step 4: Transfer weights
    print(f"\n4. Transferring weights...")
    
    # Transfer what we can (this is partial - full transfer would need component-by-component)
    transferred = 0
    
    # CSE byte embedding
    try:
        transfer_embedding_weights(small_model.cse.byte_embedding, large_model.cse.byte_embedding)
        transferred += small_model.cse.byte_embedding.weight.numel()
        print(f"   ✓ CSE byte embedding transferred")
    except Exception as e:
        print(f"   ⚠ CSE byte embedding: {e}")
    
    # Note: Full weight transfer would require matching layer structures
    # For now, we initialize the larger model fresh with some transferred embeddings
    
    print(f"\n   Total transferred: {transferred:,} parameters")
    print(f"   Newly initialized: {large_params['total'] - transferred:,} parameters")
    
    # Step 5: Save 1B model
    print(f"\n5. Saving 1B model to {target_path}")
    large_model.save(target_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("Scaling Complete!")
    print("=" * 60)
    print(f"\nParameter breakdown:")
    for name, count in large_params.items():
        if name != 'total':
            print(f"  {name}: {count:,}")
    print(f"\n  TOTAL: {large_params['total']:,} ({large_params['total']/1e9:.2f}B)")
    
    print(f"\nFiles:")
    print(f"  Original (backup): {backup_path}")
    print(f"  1B model: {target_path}")
    
    return large_model


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Scale FluxLM 141M to 1B')
    parser.add_argument('--source', default='checkpoints/Flux-LM-141M.pt', help='Source 141M checkpoint')
    parser.add_argument('--target', default='checkpoints/Flux-LM-1B.pt', help='Target 1B checkpoint')
    parser.add_argument('--backup', default='checkpoints/Flux-LM-141M-original.pt', help='Backup path')
    parser.add_argument('--device', default='cpu', help='Device to use')
    parser.add_argument('--test', action='store_true', help='Run quick test after scaling')
    
    args = parser.parse_args()
    
    # Scale
    model = scale_141m_to_1b(
        source_path=args.source,
        target_path=args.target,
        backup_path=args.backup,
        device=args.device,
    )
    
    # Optional test
    if args.test:
        print("\n" + "=" * 60)
        print("Quick Test")
        print("=" * 60)
        
        test_prompt = "The future of AI is"
        print(f"\nPrompt: {repr(test_prompt)}")
        
        try:
            output = model.generate(
                test_prompt,
                GenerationConfig(
                    max_new_bytes=50,
                    temperature=0.8,
                )
            )
            print(f"Output: {repr(output)}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
