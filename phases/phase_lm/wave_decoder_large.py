"""
WaveDecoder-Large: Scaled Wave-to-Byte Decoder for FLUX-LM.

Converts predicted waves to byte probability distributions.

Improvements over Phase 1 decoder:
- Larger hidden dimension (2048 vs 256)
- More layers (4 vs 3)
- Multi-scale decoding
- ~9M parameters (vs 418K)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Any, Optional


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

WAVE_DECODER_L_CONFIG = {
    'input_dim': 608,               # From WavePredictor output
    'hidden_dim': 2048,
    'n_layers': 4,
    'output_dim': 256,              # Byte vocabulary
    'dropout': 0.1,
    'use_multi_scale': True,
    'scales': [1, 2, 4],
}


# ─────────────────────────────────────────────
# Multi-Scale Decoder
# ─────────────────────────────────────────────

class MultiScaleDecoder(nn.Module):
    """
    Decode at multiple scales for better byte prediction.
    Different scales capture different patterns.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        scales: list = [1, 2, 4],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.scales = scales
        
        # One decoder head per scale
        self.scale_decoders = nn.ModuleList()
        for scale in scales:
            self.scale_decoders.append(
                nn.Sequential(
                    nn.Linear(input_dim * scale, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, output_dim),
                )
            )
        
        # Combine scale outputs
        self.combine = nn.Linear(output_dim * len(scales), output_dim)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim] or [batch, input_dim]
        Returns:
            [batch, seq_len, output_dim] or [batch, output_dim]
        """
        single_vector = x.dim() == 2
        if single_vector:
            x = x.unsqueeze(1)  # [batch, 1, input_dim]
        
        batch, seq_len, dim = x.shape
        outputs = []
        
        for scale, decoder in zip(self.scales, self.scale_decoders):
            if scale == 1:
                scale_input = x
            else:
                # Pool over scale positions (use last `scale` positions)
                if seq_len >= scale:
                    # Average pool over last `scale` positions
                    pooled = x[:, -scale:].mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
                    scale_input = torch.cat([x, pooled.expand(-1, -1, -1)], dim=-1)
                    scale_input = scale_input[..., :dim * scale]  # Ensure correct size
                else:
                    # Pad with zeros
                    pad = torch.zeros(batch, seq_len, dim * (scale - 1), device=x.device)
                    scale_input = torch.cat([x, pad], dim=-1)
            
            # Ensure dimension match
            expected_dim = dim * scale
            if scale_input.shape[-1] < expected_dim:
                pad_size = expected_dim - scale_input.shape[-1]
                scale_input = F.pad(scale_input, (0, pad_size))
            elif scale_input.shape[-1] > expected_dim:
                scale_input = scale_input[..., :expected_dim]
            
            outputs.append(decoder(scale_input))
        
        # Combine all scale outputs
        combined = torch.cat(outputs, dim=-1)
        result = self.combine(combined)
        
        if single_vector:
            result = result.squeeze(1)
        
        return result


# ─────────────────────────────────────────────
# WaveDecoder-Large Main Module
# ─────────────────────────────────────────────

class WaveDecoderLarge(nn.Module):
    """
    Scaled Wave-to-Byte Decoder.
    
    Input:  Predicted wave [batch, 608] or [batch, seq_len, 608]
    Output: Byte logits [batch, 256] or [batch, seq_len, 256]
    
    ~9M parameters
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        config = config or WAVE_DECODER_L_CONFIG
        
        self.input_dim = config['input_dim']  # 608
        self.hidden_dim = config['hidden_dim']  # 2048
        self.n_layers = config['n_layers']  # 4
        self.output_dim = config['output_dim']  # 256
        self.dropout = config['dropout']
        self.use_multi_scale = config.get('use_multi_scale', True)
        self.scales = config.get('scales', [1, 2, 4])
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        
        # Deep decoder layers
        layers = []
        for i in range(self.n_layers - 1):
            in_dim = self.hidden_dim
            out_dim = self.hidden_dim // (2 ** min(i, 2))  # Gradually reduce
            
            layers.extend([
                nn.Linear(in_dim if i == 0 else self.hidden_dim // (2 ** min(i-1, 2)), out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
            ])
        
        self.decoder_layers = nn.Sequential(*layers)
        
        # Final layer size depends on last hidden
        final_hidden = self.hidden_dim // (2 ** min(self.n_layers - 2, 2))
        
        # Multi-scale decoding (optional)
        if self.use_multi_scale:
            self.multi_scale = MultiScaleDecoder(
                input_dim=final_hidden,
                hidden_dim=self.hidden_dim // 2,
                output_dim=self.output_dim,
                scales=self.scales,
                dropout=self.dropout,
            )
            self.output_head = None
        else:
            self.multi_scale = None
            self.output_head = nn.Linear(final_hidden, self.output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Decode wave to byte logits.
        
        Args:
            x: [batch, 608] or [batch, seq_len, 608] wave(s)
        Returns:
            [batch, 256] or [batch, seq_len, 256] byte logits
        """
        single_vector = x.dim() == 2 and x.shape[0] != 1
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single_vector = True
        
        # Input projection
        x = self.input_proj(x)
        
        # Decoder layers
        x = self.decoder_layers(x)
        
        # Output head
        if self.use_multi_scale and self.multi_scale is not None:
            logits = self.multi_scale(x)
        else:
            logits = self.output_head(x)
        
        if single_vector and logits.dim() == 3:
            logits = logits.squeeze(1) if logits.shape[1] == 1 else logits
        
        return logits
    
    def decode_to_bytes(self, x: Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0) -> Tensor:
        """
        Decode wave to actual bytes with sampling.
        
        Args:
            x: [batch, 608] wave
            temperature: Sampling temperature
            top_k: Top-k filtering (0 = disabled)
            top_p: Nucleus sampling threshold (0 = disabled)
        
        Returns:
            [batch] sampled byte values
        """
        logits = self.forward(x)  # [batch, 256]
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Top-p (nucleus) filtering
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        next_byte = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        return next_byte
    
    def decode_greedy(self, x: Tensor) -> Tensor:
        """Greedy decoding (argmax)."""
        logits = self.forward(x)
        return logits.argmax(dim=-1)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test
    model = WaveDecoderLarge()
    print(f"WaveDecoder-Large parameters: {count_parameters(model):,}")
    
    # Test forward (single vector)
    x = torch.randn(2, 608)  # [batch, 608]
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Test forward (sequence)
    x_seq = torch.randn(2, 10, 608)  # [batch, seq_len, 608]
    logits_seq = model(x_seq)
    print(f"Sequence input shape: {x_seq.shape}")
    print(f"Sequence output shape: {logits_seq.shape}")
    
    # Test sampling
    bytes_sampled = model.decode_to_bytes(x, temperature=0.8, top_k=50)
    print(f"Sampled bytes: {bytes_sampled.tolist()}")
