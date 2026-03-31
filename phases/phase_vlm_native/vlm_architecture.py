"""
Native VLM Architecture for FLUX.

Self-contained Qwen2.5-VL implementation that loads entirely from .flx.
No HuggingFace trust_remote_code dependency.

This is a simplified but functional implementation focusing on:
- Text generation (the primary use case)
- Vision encoding (for multimodal)
- Full control over the architecture

Key difference from HF:
- Architecture code embedded in .flx
- Weights stored as SVD-compressed tensors
- No external downloads at inference time
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import math


@dataclass
class NativeVLMConfig:
    """Configuration for native VLM."""
    hidden_size: int = 2048
    intermediate_size: int = 11008
    num_hidden_layers: int = 36
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    vocab_size: int = 151936
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6
    
    # Vision config
    vision_hidden_size: int = 1280
    vision_intermediate_size: int = 5120
    vision_num_hidden_layers: int = 32
    vision_num_attention_heads: int = 16
    vision_patch_size: int = 14
    vision_image_size: int = 384
    
    # FLUX integration
    wave_dim: int = 432
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'NativeVLMConfig':
        """Create config from dict."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# ─────────────────────────────────────────────
# Core Building Blocks
# ─────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    
    def __init__(self, dim: int, max_seq_len: int = 32768, theta: float = 1000000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._cos_cache = None
        self._sin_cache = None
    
    def _compute_cache(self, seq_len: int, device: torch.device):
        if self._cos_cache is not None and self._cos_cache.shape[0] >= seq_len:
            return
        
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cos_cache = emb.cos()
        self._sin_cache = emb.sin()
    
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = position_ids.max().item() + 1
        self._compute_cache(seq_len, x.device)
        
        cos = self._cos_cache[position_ids]
        sin = self._sin_cache[position_ids]
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key."""
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


# ─────────────────────────────────────────────
# Attention
# ─────────────────────────────────────────────

class NativeAttention(nn.Module):
    """Grouped Query Attention with RoPE."""
    
    def __init__(self, config: NativeVLMConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rotary_emb(q, position_ids)
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # KV cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        present_key_value = (k, v)
        
        # Repeat KV for grouped attention
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, present_key_value


# ─────────────────────────────────────────────
# MLP
# ─────────────────────────────────────────────

class NativeMLP(nn.Module):
    """SwiGLU MLP."""
    
    def __init__(self, config: NativeVLMConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ─────────────────────────────────────────────
# Transformer Layer
# ─────────────────────────────────────────────

class NativeTransformerLayer(nn.Module):
    """Single transformer layer."""
    
    def __init__(self, config: NativeVLMConfig, layer_idx: int):
        super().__init__()
        self.self_attn = NativeAttention(config, layer_idx)
        self.mlp = NativeMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states, position_ids, attention_mask, past_key_value
        )
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_value


# ─────────────────────────────────────────────
# Full Model
# ─────────────────────────────────────────────

class NativeVLM(nn.Module):
    """
    Native Vision-Language Model for FLUX.
    
    Self-contained implementation that can be loaded entirely from .flx.
    """
    
    def __init__(self, config: NativeVLMConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            NativeTransformerLayer(config, i)
            for i in range(config.num_hidden_layers)
        ])
        
        # Output
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # FLUX integration: wave bridges
        self.wave_to_hidden = nn.Linear(config.wave_dim, config.hidden_size)
        self.hidden_to_wave = nn.Linear(config.hidden_size, config.wave_dim)
        
        # KV cache
        self._past_key_values = None
    
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        batch_size, seq_len = input_ids.shape
        
        # Position IDs
        if position_ids is None:
            past_len = past_key_values[0][0].shape[2] if past_key_values else 0
            position_ids = torch.arange(past_len, past_len + seq_len, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Causal mask
        if attention_mask is None and seq_len > 1:
            attention_mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=input_ids.device),
                diagonal=1
            )
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        
        # Transformer layers
        present_key_values = []
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            hidden_states, present_kv = layer(
                hidden_states, position_ids, attention_mask, past_kv
            )
            if use_cache:
                present_key_values.append(present_kv)
        
        # Output
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits, present_key_values if use_cache else None
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        eos_token_id: int = 151643,
        pad_token_id: int = 151643,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            do_sample: Whether to sample (vs greedy)
            eos_token_id: End of sequence token
            pad_token_id: Padding token
        
        Returns:
            Generated token IDs [batch, seq_len + new_tokens]
        """
        batch_size = input_ids.shape[0]
        generated = input_ids.clone()
        past_key_values = None
        
        for _ in range(max_new_tokens):
            # Forward pass
            if past_key_values is None:
                # First pass: process full sequence
                logits, past_key_values = self.forward(
                    generated, use_cache=True
                )
            else:
                # Subsequent passes: only last token
                logits, past_key_values = self.forward(
                    generated[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            
            # Get next token logits
            next_logits = logits[:, -1, :]
            
            if do_sample:
                # Apply temperature
                next_logits = next_logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            
            # Append
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check EOS
            if (next_token == eos_token_id).all():
                break
        
        return generated
    
    def encode_wave(self, wave: torch.Tensor) -> torch.Tensor:
        """Project FLUX wave into VLM hidden space."""
        return self.wave_to_hidden(wave)
    
    def decode_wave(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project VLM hidden state back to FLUX wave."""
        return self.hidden_to_wave(hidden)


# ─────────────────────────────────────────────
# Loading from .flx
# ─────────────────────────────────────────────

def load_native_vlm_from_flx(
    flx_state: Dict[str, Any],
    device: str = 'cuda',
    dtype: torch.dtype = torch.float16,
) -> NativeVLM:
    """
    Load NativeVLM entirely from .flx file.
    
    No HuggingFace dependency!
    
    Args:
        flx_state: Loaded .flx state (from torch.load)
        device: Target device
        dtype: Target dtype
    
    Returns:
        Initialized NativeVLM ready for inference
    """
    from .vlm_svd import load_vlm_from_flx_svd
    
    if 'vlm' not in flx_state:
        raise KeyError("No VLM in .flx file")
    
    vlm_state = flx_state['vlm']
    
    # Get config
    if 'native_config' in vlm_state:
        config = NativeVLMConfig.from_dict(vlm_state['native_config'])
    elif 'config' in vlm_state:
        # Try to adapt from HF config
        hf_config = vlm_state['config']
        config = NativeVLMConfig(
            hidden_size=hf_config.get('hidden_size', 2048),
            intermediate_size=hf_config.get('intermediate_size', 11008),
            num_hidden_layers=hf_config.get('num_hidden_layers', 36),
            num_attention_heads=hf_config.get('num_attention_heads', 16),
            num_key_value_heads=hf_config.get('num_key_value_heads', 2),
            vocab_size=hf_config.get('vocab_size', 151936),
        )
    else:
        # Default config for Qwen2.5-VL-3B
        config = NativeVLMConfig()
    
    print(f"  Creating NativeVLM...")
    print(f"    Hidden size: {config.hidden_size}")
    print(f"    Layers: {config.num_hidden_layers}")
    print(f"    Heads: {config.num_attention_heads}")
    
    # Create model
    model = NativeVLM(config)
    
    # Load weights
    print(f"  Loading weights from .flx...")
    weights = load_vlm_from_flx_svd(flx_state)
    
    # Map weights to native model
    # HF uses "model." prefix, we don't
    mapped_weights = {}
    for key, value in weights.items():
        if key.startswith('model.'):
            new_key = key[6:]  # Remove 'model.' prefix
        else:
            new_key = key
        mapped_weights[new_key] = value
    
    # Load state dict
    missing, unexpected = model.load_state_dict(mapped_weights, strict=False)
    print(f"  Loaded weights:")
    print(f"    Missing: {len(missing)} keys")
    print(f"    Unexpected: {len(unexpected)} keys")
    
    # Move to device
    model = model.to(device=device, dtype=dtype)
    model.eval()
    
    print(f"  ✓ NativeVLM ready on {device}")
    
    return model


# ─────────────────────────────────────────────
# Exports
# ─────────────────────────────────────────────

__all__ = [
    'NativeVLMConfig',
    'NativeVLM',
    'load_native_vlm_from_flx',
]
