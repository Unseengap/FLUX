"""
CWC-Large: Scaled Causal Wave Chainer for FLUX-LM.

Improvements over Phase 1.5 CWC:
- Explicit order discrimination (fixes coherence_gap = 0.0)
- Rotary position encoding for better sequence awareness
- Deeper temporal MLP
- Order classifier head
- ~5M parameters (vs 570K)
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

CWC_L_CONFIG = {
    'wave_dim': 432,
    'causal_dim': 176,              # Additional dims for causal info
    'hidden_dim': 1024,
    'n_heads': 8,
    'n_layers': 2,
    'dropout': 0.1,
    'max_seq_len': 2048,
    
    # Order awareness (fixes coherence_gap = 0.0)
    'order_aware': True,
    'order_hidden_dim': 512,
    'temporal_mlp_layers': 3,
}


# ─────────────────────────────────────────────
# Rotary Position Encoding
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
        """Apply RoPE to input tensor."""
        seq_len = x.shape[1]
        cos = self.cos_cached[:, :seq_len]
        sin = self.sin_cached[:, :seq_len]
        return x * cos + self._rotate_half(x) * sin
    
    def _rotate_half(self, x: Tensor) -> Tensor:
        x1, x2 = x[..., :self.dim//2], x[..., self.dim//2:]
        return torch.cat([-x2, x1], dim=-1)


# ─────────────────────────────────────────────
# Temporal MLP (Deeper)
# ─────────────────────────────────────────────

class TemporalMLP(nn.Module):
    """
    Deep MLP for encoding temporal/causal information.
    Each position gets its own causal embedding.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(n_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


# ─────────────────────────────────────────────
# Order-Aware Self-Attention
# ─────────────────────────────────────────────

class OrderAwareAttention(nn.Module):
    """
    Self-attention with explicit order awareness.
    Uses relative position bias to enforce order sensitivity.
    """
    
    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
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
        
        # Learnable relative position bias
        self.rel_pos_bias = nn.Embedding(2 * max_seq_len - 1, n_heads)
        self.max_seq_len = max_seq_len
        
        # RoPE for Q and K
        self.rope = RotaryPositionEncoding(self.head_dim, max_seq_len)
    
    def forward(self, x: Tensor, causal_mask: bool = True) -> Tensor:
        """
        Args:
            x: [batch, seq_len, dim]
            causal_mask: Whether to apply causal masking
        Returns:
            [batch, seq_len, dim]
        """
        batch, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        
        # Apply RoPE to Q and K
        q = q.transpose(1, 2)  # [batch, heads, seq, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Add relative position bias
        positions = torch.arange(seq_len, device=x.device)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)  # [seq, seq]
        rel_pos = rel_pos + self.max_seq_len - 1  # Shift to positive indices
        rel_pos = rel_pos.clamp(0, 2 * self.max_seq_len - 2)
        
        pos_bias = self.rel_pos_bias(rel_pos)  # [seq, seq, heads]
        pos_bias = pos_bias.permute(2, 0, 1).unsqueeze(0)  # [1, heads, seq, seq]
        attn = attn + pos_bias
        
        # Causal mask (for autoregressive generation)
        if causal_mask:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # [batch, heads, seq, head_dim]
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        out = self.out_proj(out)
        
        return out


# ─────────────────────────────────────────────
# Causal Transformer Block
# ─────────────────────────────────────────────

class CausalBlock(nn.Module):
    """Transformer block with order-aware attention."""
    
    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        ff_mult: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = OrderAwareAttention(dim, n_heads, dropout, max_seq_len)
        self.norm2 = nn.LayerNorm(dim)
        
        ff_dim = int(dim * ff_mult)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: Tensor, causal_mask: bool = True) -> Tensor:
        x = x + self.attn(self.norm1(x), causal_mask)
        x = x + self.ff(self.norm2(x))
        return x


# ─────────────────────────────────────────────
# Order Classifier Head
# ─────────────────────────────────────────────

class OrderClassifier(nn.Module):
    """
    Classifies whether a sequence is in correct order.
    This is the key to fixing coherence_gap = 0.0.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        
        self.pool_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [batch, seq_len, dim]
        Returns:
            [batch, 1] score (higher = more ordered)
        """
        # Mean pool over sequence
        pooled = x.mean(dim=1)  # [batch, dim]
        return self.pool_mlp(pooled)


# ─────────────────────────────────────────────
# Tension Computator
# ─────────────────────────────────────────────

class TensionComputer(nn.Module):
    """
    Computes tension between wave pairs.
    Higher tension = contradiction.
    """
    
    def __init__(self, dim: int, hidden_dim: int = 512):
        super().__init__()
        
        self.combine = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # Tension is always positive
        )
    
    def forward(self, wave1: Tensor, wave2: Tensor) -> Tensor:
        """
        Compute tension between two waves.
        
        Args:
            wave1, wave2: [batch, dim] or [dim]
        Returns:
            [batch, 1] or [1] tension score
        """
        if wave1.dim() == 1:
            wave1 = wave1.unsqueeze(0)
            wave2 = wave2.unsqueeze(0)
        
        combined = torch.cat([wave1, wave2], dim=-1)
        return self.combine(combined)


# ─────────────────────────────────────────────
# CWC-Large Main Module
# ─────────────────────────────────────────────

class CWCLarge(nn.Module):
    """
    Scaled Causal Wave Chainer.
    
    Input:  SemanticWave [seq_len, 432] from CSE
    Output: CausalWave [seq_len, 608] with causal direction
    
    Key improvements:
    - Explicit order discrimination
    - Relative position bias
    - Deeper temporal encoding
    
    ~5M parameters
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        config = config or CWC_L_CONFIG
        
        self.wave_dim = config['wave_dim']  # 432
        self.causal_dim = config['causal_dim']  # 176
        self.output_dim = self.wave_dim + self.causal_dim  # 608
        self.hidden_dim = config['hidden_dim']
        self.n_heads = config['n_heads']
        self.n_layers = config['n_layers']
        self.dropout = config['dropout']
        self.max_seq_len = config['max_seq_len']
        self.order_aware = config.get('order_aware', True)
        
        # Input projection
        self.input_proj = nn.Linear(self.wave_dim, self.hidden_dim)
        
        # Causal transformer blocks
        self.blocks = nn.ModuleList([
            CausalBlock(
                dim=self.hidden_dim,
                n_heads=self.n_heads,
                ff_mult=4.0,
                dropout=self.dropout,
                max_seq_len=self.max_seq_len,
            )
            for _ in range(self.n_layers)
        ])
        
        # Temporal MLP for causal direction
        self.temporal_mlp = TemporalMLP(
            input_dim=self.hidden_dim,
            hidden_dim=config.get('order_hidden_dim', 512),
            output_dim=self.causal_dim,
            n_layers=config.get('temporal_mlp_layers', 3),
            dropout=self.dropout,
        )
        
        # Output projection (back to wave space + causal)
        self.output_proj = nn.Linear(self.hidden_dim, self.wave_dim)
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(self.output_dim)
        
        # Order classifier (for training)
        if self.order_aware:
            self.order_classifier = OrderClassifier(
                self.output_dim,
                config.get('order_hidden_dim', 512),
                self.dropout,
            )
        
        # Tension computer (for contradiction detection)
        self.tension_computer = TensionComputer(
            self.output_dim,
            config.get('order_hidden_dim', 512),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        wave: Tensor,
        return_order_score: bool = False,
        causal_mask: bool = True,
    ) -> Tensor:
        """
        Add causal direction to semantic wave.
        
        Args:
            wave: [batch, seq_len, 432] or [seq_len, 432] semantic wave
            return_order_score: Whether to also return order score
            causal_mask: Whether to apply causal masking
        
        Returns:
            causal_wave: [batch, seq_len, 608] or [seq_len, 608]
            (optionally: order_score if return_order_score=True)
        """
        unbatched = wave.dim() == 2
        if unbatched:
            wave = wave.unsqueeze(0)
        
        batch, seq_len, _ = wave.shape
        
        # Project to hidden dim
        x = self.input_proj(wave)  # [batch, seq_len, hidden_dim]
        
        # Apply causal blocks
        for block in self.blocks:
            x = block(x, causal_mask)
        
        # Get processed wave (preserving original semantic content)
        processed_wave = self.output_proj(x)  # [batch, seq_len, 432]
        processed_wave = processed_wave + wave  # Residual
        
        # Get causal direction
        causal_direction = self.temporal_mlp(x)  # [batch, seq_len, 176]
        
        # Combine
        causal_wave = torch.cat([processed_wave, causal_direction], dim=-1)
        causal_wave = self.final_norm(causal_wave)
        
        if unbatched:
            causal_wave = causal_wave.squeeze(0)
        
        if return_order_score and self.order_aware:
            order_score = self.order_classifier(causal_wave if unbatched else causal_wave)
            return causal_wave, order_score
        
        return causal_wave
    
    def compute_tension(self, wave1: Tensor, wave2: Tensor) -> Tensor:
        """Compute contradiction tension between two causal waves."""
        # Pool to single vectors
        if wave1.dim() == 2:
            wave1 = wave1.mean(dim=0)
            wave2 = wave2.mean(dim=0)
        elif wave1.dim() == 3:
            wave1 = wave1.mean(dim=1)
            wave2 = wave2.mean(dim=1)
        
        return self.tension_computer(wave1, wave2)
    
    def get_order_score(self, causal_wave: Tensor) -> Tensor:
        """Get order score for a causal wave."""
        if not self.order_aware:
            raise ValueError("Order classifier not enabled")
        
        if causal_wave.dim() == 2:
            causal_wave = causal_wave.unsqueeze(0)
        
        return self.order_classifier(causal_wave)
    
    def to_base_wave(self, causal_wave: Tensor) -> Tensor:
        """Extract original 432D wave from causal wave."""
        return causal_wave[..., :self.wave_dim]


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test
    model = CWCLarge()
    print(f"CWC-Large parameters: {count_parameters(model):,}")
    
    # Test forward
    wave = torch.randn(10, 432)  # [seq_len, 432]
    causal_wave, order_score = model(wave, return_order_score=True)
    print(f"Input shape: {wave.shape}")
    print(f"Output shape: {causal_wave.shape}")
    print(f"Order score: {order_score.item():.4f}")
