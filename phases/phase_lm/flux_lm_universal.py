"""
FluxLM-Universal: Multi-Domain Vocabulary-Free Language Model.

Extends FluxLM to support multiple domains with:
- Extended vocabulary (256 bytes + 64 special tokens = 320)
- Domain embeddings for domain-aware generation
- Scaled architecture for 3B parameters

Domain Categories:
- TEXT: Assistant, Reasoning, Tool Use, Multilingual
- CODE: Python, JavaScript, etc.
- DOCS: Markdown, HTML, LaTeX
- DATA: JSON, CSV, YAML
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

# Import base components
try:
    from .cse_large import CSELarge, CSE_L_CONFIG, SemanticWave
    from .cwc_large import CWCLarge, CWC_L_CONFIG
    from .wave_predictor import WavePredictor, WAVE_PREDICTOR_CONFIG
    from .wave_decoder_large import WaveDecoderLarge, WAVE_DECODER_L_CONFIG
    from .multi_domain_data import SPECIAL_TOKENS, VOCAB_SIZE, encode_special, decode_special
except ImportError:
    from cse_large import CSELarge, CSE_L_CONFIG, SemanticWave
    from cwc_large import CWCLarge, CWC_L_CONFIG
    from wave_predictor import WavePredictor, WAVE_PREDICTOR_CONFIG
    from wave_decoder_large import WaveDecoderLarge, WAVE_DECODER_L_CONFIG
    from multi_domain_data import SPECIAL_TOKENS, VOCAB_SIZE, encode_special, decode_special


# ─────────────────────────────────────────────
# Scale Configurations
# ─────────────────────────────────────────────

# 500M Configuration
FLUX_LM_500M_CONFIG = {
    'name': '500M',
    'vocab_size': VOCAB_SIZE,  # 320 (256 bytes + 64 special)
    
    'cse': {
        **CSE_L_CONFIG,
        'byte_embed_dim': 128,
        'conv_channels': [512, 768, 768, 768],
        'hidden_dim': 1536,
    },
    
    'cwc': {
        **CWC_L_CONFIG,
        'hidden_dim': 1024,
    },
    
    'predictor': {
        'input_dim': 608,
        'hidden_dim': 1536,
        'n_heads': 16,
        'n_layers': 16,
        'ff_dim': 6144,
        'max_seq_len': 2048,
        'dropout': 0.1,
    },
    
    'decoder': {
        **WAVE_DECODER_L_CONFIG,
        'hidden_dim': 1536,
        'num_bytes': VOCAB_SIZE,  # Extended for special tokens
    },
    
    'max_seq_len': 1024,
    'gradient_checkpointing': True,
}

# 1B Configuration
FLUX_LM_1B_CONFIG = {
    'name': '1B',
    'vocab_size': VOCAB_SIZE,
    
    'cse': {
        **CSE_L_CONFIG,
        'byte_embed_dim': 256,
        'conv_channels': [512, 1024, 1024, 1024],
        'hidden_dim': 2048,
    },
    
    'cwc': {
        **CWC_L_CONFIG,
        'hidden_dim': 1536,
    },
    
    'predictor': {
        'input_dim': 608,
        'hidden_dim': 2048,
        'n_heads': 16,
        'n_layers': 24,
        'ff_dim': 8192,
        'max_seq_len': 4096,
        'dropout': 0.1,
    },
    
    'decoder': {
        **WAVE_DECODER_L_CONFIG,
        'hidden_dim': 2048,
        'num_bytes': VOCAB_SIZE,
    },
    
    'max_seq_len': 2048,
    'gradient_checkpointing': True,
}

# 3B Configuration (Target)
FLUX_LM_3B_CONFIG = {
    'name': '3B',
    'vocab_size': VOCAB_SIZE,
    
    'cse': {
        **CSE_L_CONFIG,
        'byte_embed_dim': 256,
        'conv_channels': [768, 1024, 1536, 1536],
        'hidden_dim': 2048,
    },
    
    'cwc': {
        **CWC_L_CONFIG,
        'hidden_dim': 2048,
    },
    
    'predictor': {
        'input_dim': 608,
        'hidden_dim': 3072,
        'n_heads': 24,
        'n_layers': 32,
        'ff_dim': 12288,
        'max_seq_len': 4096,
        'dropout': 0.1,
    },
    
    'decoder': {
        **WAVE_DECODER_L_CONFIG,
        'hidden_dim': 3072,
        'num_bytes': VOCAB_SIZE,
    },
    
    'max_seq_len': 4096,
    'gradient_checkpointing': True,
}

# Default to previous 141M for backward compatibility
FLUX_LM_UNIVERSAL_CONFIG = {
    'name': '141M-multi',
    'vocab_size': VOCAB_SIZE,
    
    'cse': CSE_L_CONFIG,
    'cwc': CWC_L_CONFIG,
    'predictor': WAVE_PREDICTOR_CONFIG,
    'decoder': {
        **WAVE_DECODER_L_CONFIG,
        'num_bytes': VOCAB_SIZE,  # Extended vocabulary
    },
    
    'max_seq_len': 512,
    'gradient_checkpointing': False,
}


# ─────────────────────────────────────────────
# Domain Embedding Layer
# ─────────────────────────────────────────────

class DomainEmbedding(nn.Module):
    """Learnable domain embeddings added to wave representations."""
    
    DOMAINS = {
        'text': 0,
        'assistant': 1,
        'reasoning': 2,
        'tool_use': 3,
        'code': 4,
        'docs': 5,
        'data': 6,
        'multilingual': 7,
        'media': 8,
        'protocol': 9,
    }
    
    def __init__(self, wave_dim: int = 432, num_domains: int = 16):
        super().__init__()
        self.wave_dim = wave_dim
        self.num_domains = num_domains
        
        self.embed = nn.Embedding(num_domains, wave_dim)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)  # Small initial scale
    
    def forward(self, waves: Tensor, domain_ids: Optional[Tensor] = None) -> Tensor:
        """
        Add domain embedding to waves.
        
        Args:
            waves: [batch, seq_len, wave_dim]
            domain_ids: [batch] domain indices
        
        Returns:
            waves with domain signal added
        """
        if domain_ids is None:
            return waves
        
        # Get domain embeddings
        domain_emb = self.embed(domain_ids)  # [batch, wave_dim]
        domain_emb = domain_emb.unsqueeze(1)  # [batch, 1, wave_dim]
        
        # Add scaled domain signal
        return waves + self.scale * domain_emb
    
    @classmethod
    def domain_to_id(cls, domain: str) -> int:
        """Convert domain name to ID."""
        return cls.DOMAINS.get(domain, 0)


# ─────────────────────────────────────────────
# Extended Wave Decoder for Special Tokens
# ─────────────────────────────────────────────

class WaveDecoderUniversal(nn.Module):
    """
    Wave decoder with extended vocabulary for special tokens.
    
    Outputs logits over 320 classes (256 bytes + 64 special tokens).
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.input_dim = config.get('input_dim', 608)
        self.hidden_dim = config.get('hidden_dim', 1024)
        self.num_bytes = config.get('num_bytes', VOCAB_SIZE)
        self.dropout = config.get('dropout', 0.1)
        
        # Multi-layer decoder
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            
            nn.Linear(self.hidden_dim // 2, self.num_bytes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, waves: Tensor) -> Tensor:
        """
        Decode waves to byte/token logits.
        
        Args:
            waves: [batch, seq_len, input_dim]
        
        Returns:
            logits: [batch, seq_len, num_bytes]
        """
        return self.layers(waves)


# ─────────────────────────────────────────────
# Generation Config (Extended)
# ─────────────────────────────────────────────

@dataclass
class GenerationConfigUniversal:
    """Configuration for multi-domain text generation."""
    max_new_bytes: int = 200
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    stop_bytes: List[int] = None
    domain: Optional[str] = None  # Domain hint for generation
    
    def __post_init__(self):
        if self.stop_bytes is None:
            # Stop on null byte or end token
            self.stop_bytes = [0, SPECIAL_TOKENS.get('<|end|>', 259)]


# ─────────────────────────────────────────────
# FluxLM Universal Model
# ─────────────────────────────────────────────

class FluxLMUniversal(nn.Module):
    """
    FLUX Language Model Universal: Multi-domain vocabulary-free generation.
    
    Extends base FluxLM with:
    - Extended vocabulary (320 = 256 bytes + 64 special tokens)
    - Domain embeddings for domain-aware generation
    - Scalable architecture (141M → 500M → 1B → 3B)
    
    Pipeline:
        text → bytes/special → CSE → waves [432] → (+domain) → CWC → causal_waves [608]
        → WavePredictor → next_wave [608] → Decoder → logits [320]
    """

    def __init__(self, config: Dict[str, Any] = None, scale: str = '141M'):
        super().__init__()
        
        # Select config by scale
        if config is None:
            if scale == '500M':
                config = FLUX_LM_500M_CONFIG
            elif scale == '1B':
                config = FLUX_LM_1B_CONFIG
            elif scale == '3B':
                config = FLUX_LM_3B_CONFIG
            else:
                config = FLUX_LM_UNIVERSAL_CONFIG
        
        self.config = config
        self.scale = config.get('name', scale)
        self.max_seq_len = config.get('max_seq_len', 512)
        self.vocab_size = config.get('vocab_size', VOCAB_SIZE)
        
        # Core components
        self.cse = CSELarge(config.get('cse', CSE_L_CONFIG))
        self.cwc = CWCLarge(config.get('cwc', CWC_L_CONFIG))
        self.predictor = WavePredictor(config.get('predictor', WAVE_PREDICTOR_CONFIG))
        
        # Extended decoder for special tokens
        decoder_config = config.get('decoder', WAVE_DECODER_L_CONFIG)
        decoder_config = {**decoder_config, 'num_bytes': self.vocab_size}
        self.decoder = WaveDecoderUniversal(decoder_config)
        
        # Domain embeddings
        self.domain_embed = DomainEmbedding(wave_dim=432, num_domains=16)
        
        # ==== SPECIAL TOKEN EMBEDDINGS (FIX for tokens 256-319) ====
        # CSE only knows bytes 0-255, so we need separate embeddings for special tokens
        self.num_special_tokens = self.vocab_size - 256  # 64 special tokens
        self.special_token_embed = nn.Embedding(self.num_special_tokens, 432)
        nn.init.normal_(self.special_token_embed.weight, std=0.02)
        
        # Dimensions
        self.wave_dim = 432
        self.causal_dim = 608
        
        self.gradient_checkpointing = config.get('gradient_checkpointing', False)
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters per component."""
        def count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        return {
            'cse': count(self.cse),
            'cwc': count(self.cwc),
            'predictor': count(self.predictor),
            'decoder': count(self.decoder),
            'domain_embed': count(self.domain_embed),
            'special_token_embed': count(self.special_token_embed),
            'total': count(self),
        }
    
    # ─────────────────────────────────────────────
    # Encoding
    # ─────────────────────────────────────────────
    
    def encode_text(self, text: str, domain: Optional[str] = None) -> Tensor:
        """
        Encode text with special tokens to causal waves.
        
        Args:
            text: Input text (may contain special tokens)
            domain: Optional domain hint
        
        Returns:
            [seq_len, 608] causal waves
        """
        # Encode to bytes (handles special tokens)
        byte_seq = encode_special(text)
        bytes_tensor = torch.tensor(byte_seq, dtype=torch.long, device=self.device)
        
        return self.encode_bytes(bytes_tensor.unsqueeze(0), domain).squeeze(0)
    
    def encode_bytes(self, bytes_tensor: Tensor, domain: Optional[str] = None) -> Tensor:
        """
        Encode byte tensor to causal waves.
        
        Args:
            bytes_tensor: [batch, seq_len] byte values (0-319)
            domain: Optional domain name
        
        Returns:
            [batch, seq_len, 608] causal waves
        """
        batch_size, seq_len = bytes_tensor.shape
        
        # ==== SEPARATE BYTES (0-255) AND SPECIAL TOKENS (256+) ====
        is_special = bytes_tensor >= 256
        is_byte = ~is_special
        
        # Create CSE input: bytes stay as-is, special tokens become 0 (placeholder)
        cse_input = bytes_tensor.clone()
        cse_input[is_special] = 0  # Placeholder byte for special token positions
        
        # Bytes → semantic waves via CSE
        semantic_wave = self.cse.encode_bytes(cse_input)
        wave = semantic_wave.full  # [batch, seq_len, 432]
        
        # ==== INJECT SPECIAL TOKEN EMBEDDINGS ====
        # For positions with special tokens, replace CSE output with learned embeddings
        if is_special.any():
            # Get special token indices (0-63 for tokens 256-319)
            special_indices = (bytes_tensor - 256).clamp(0, self.num_special_tokens - 1)
            special_embeds = self.special_token_embed(special_indices)  # [batch, seq_len, 432]
            
            # Replace CSE output with special embeddings at special positions
            wave = torch.where(
                is_special.unsqueeze(-1).expand_as(wave),
                special_embeds,
                wave
            )
        
        # Add domain embedding
        if domain:
            domain_id = DomainEmbedding.domain_to_id(domain)
            domain_ids = torch.tensor([domain_id] * wave.shape[0], device=wave.device)
            wave = self.domain_embed(wave, domain_ids)
        
        # Add causal direction
        causal_wave = self.cwc(wave)
        
        return causal_wave
    
    # ─────────────────────────────────────────────
    # Forward Pass (Training)
    # ─────────────────────────────────────────────
    
    def forward(
        self,
        input_bytes: Tensor,
        target_bytes: Optional[Tensor] = None,
        domain: Optional[str] = None,
        return_loss: bool = True,
    ) -> Dict[str, Tensor]:
        """
        Forward pass for training.
        
        Args:
            input_bytes: [batch, seq_len] input byte values (0-319)
            target_bytes: [batch, seq_len] target byte values
            domain: Optional domain hint
            return_loss: Whether to compute loss
        
        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        batch_size, seq_len = input_bytes.shape
        
        # Encode to causal waves
        causal_wave = self.encode_bytes(input_bytes, domain)
        
        # Predict next waves
        predicted_waves, _ = self.predictor(causal_wave)
        
        # Decode to byte logits (extended vocabulary)
        logits = self.decoder(predicted_waves)
        
        result = {'logits': logits}
        
        if return_loss and target_bytes is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                target_bytes.view(-1),
                ignore_index=-100,
            )
            result['loss'] = loss
        
        return result
    
    # ─────────────────────────────────────────────
    # Generation
    # ─────────────────────────────────────────────
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        config: GenerationConfigUniversal = None,
    ) -> str:
        """
        Generate text autoregressively with special token support.
        
        Args:
            prompt: Initial text (may contain special tokens)
            config: Generation configuration
        
        Returns:
            Generated text (including prompt)
        """
        was_training = self.training
        self.eval()
        
        config = config or GenerationConfigUniversal()
        
        # Encode prompt with special tokens
        output_tokens = encode_special(prompt)
        tokens_tensor = torch.tensor(output_tokens, dtype=torch.long, device=self.device)
        
        for _ in range(config.max_new_bytes):
            input_tensor = tokens_tensor.unsqueeze(0)
            
            # Encode to causal waves
            causal_waves = self.encode_bytes(input_tensor, config.domain)
            
            # Predict next wave
            predicted_waves, _ = self.predictor(causal_waves)
            
            # Decode to logits
            logits = self.decoder(predicted_waves)
            logits = logits[:, -1]  # Last position
            
            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                for prev_token in set(output_tokens[-50:]):
                    logits[0, prev_token] /= config.repetition_penalty
            
            # Sample next token
            next_token = self._sample_token(
                logits[0],
                config.temperature,
                config.top_k,
                config.top_p,
            )
            
            # Check stop conditions
            if next_token in config.stop_bytes:
                break
            
            # Append
            output_tokens.append(next_token)
            tokens_tensor = torch.cat([
                tokens_tensor,
                torch.tensor([next_token], dtype=torch.long, device=self.device)
            ])
        
        if was_training:
            self.train()
        
        # Decode with special tokens
        return decode_special(output_tokens)
    
    def _sample_token(
        self,
        logits: Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> int:
        """Sample next token from logits."""
        if temperature == 0:
            return logits.argmax().item()
        
        logits = logits / temperature
        
        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Top-p filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()
    
    # ─────────────────────────────────────────────
    # Save/Load
    # ─────────────────────────────────────────────
    
    def save(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'config': self.config,
            'scale': self.scale,
            'vocab_size': self.vocab_size,
            'state_dict': self.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"✓ FluxLM-Universal saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'FluxLMUniversal':
        """Load model from checkpoint (backwards compatible)."""
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(checkpoint['config'])
        
        # Load state dict with backwards compatibility
        # Use strict=False to allow missing keys (e.g., special_token_embed from older checkpoints)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=False)
        
        if missing_keys:
            print(f"  ⚠ Missing keys (will use random init): {missing_keys}")
            # special_token_embed is expected to be missing from old checkpoints
            # It's already randomly initialized in __init__, so this is fine
        
        if unexpected_keys:
            print(f"  ⚠ Unexpected keys (ignored): {unexpected_keys}")
        
        model.to(device)
        
        print(f"✓ FluxLM-Universal loaded from {path} (scale: {checkpoint.get('scale', 'unknown')})")
        return model
    
    def save_to_flx(self, path: str):
        """Save as .flx format for FLUX ecosystem."""
        flx_data = {
            'format': 'FLUX-LM-Universal',
            'version': '2.0.0',
            'config': self.config,
            'scale': self.scale,
            'vocab_size': self.vocab_size,
            'special_tokens': SPECIAL_TOKENS,
            'components': {
                'cse': True,
                'cwc': True,
                'predictor': True,
                'decoder': True,
                'domain_embed': True,
            },
            'cse': {'state_dict': self.cse.state_dict()},
            'cwc': {'state_dict': self.cwc.state_dict()},
            'predictor': {'state_dict': self.predictor.state_dict()},
            'decoder': {'state_dict': self.decoder.state_dict()},
            'domain_embed': {'state_dict': self.domain_embed.state_dict()},
            'metadata': {
                'param_breakdown': self.count_parameters(),
                'domains': list(DomainEmbedding.DOMAINS.keys()),
            },
        }
        torch.save(flx_data, path)
        print(f"✓ FluxLM-Universal saved to {path} (.flx format)")


# ─────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────

def format_params(n: int) -> str:
    """Format parameter count."""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)


def get_model_by_scale(scale: str, device: str = 'cpu') -> FluxLMUniversal:
    """Create model by scale name."""
    model = FluxLMUniversal(scale=scale)
    model.to(device)
    
    params = model.count_parameters()
    print(f"\n✓ FluxLM-Universal-{scale} created:")
    for name, count in params.items():
        print(f"  {name:12s}: {format_params(count):>10s}")
    
    return model


if __name__ == '__main__':
    print("=" * 60)
    print("FluxLM-Universal Test")
    print("=" * 60)
    
    # Test 141M model
    model = FluxLMUniversal(scale='141M')
    params = model.count_parameters()
    
    print("\nParameter counts:")
    for name, count in params.items():
        print(f"  {name}: {format_params(count)}")
    
    # Test encoding with special tokens
    print("\nTesting special token encoding...")
    test_text = "<|user|>What is 2+2?<|end|><|assistant|>The answer is 4.<|end|>"
    
    # Manual encoding test
    encoded = encode_special(test_text)
    print(f"  Original: {test_text[:50]}...")
    print(f"  Encoded first 20: {encoded[:20]}")
    
    # Model forward test
    print("\nTesting forward pass...")
    input_bytes = torch.tensor([encoded[:-1]], dtype=torch.long)
    target_bytes = torch.tensor([encoded[1:]], dtype=torch.long)
    
    output = model(input_bytes, target_bytes, domain='assistant')
    print(f"  Logits shape: {output['logits'].shape}")  # Should be [1, seq_len, 320]
    print(f"  Loss: {output['loss'].item():.4f}")
    
    # Generation test
    print("\nTesting generation...")
    generated = model.generate(
        "<|user|>Hello!<|end|><|assistant|>",
        GenerationConfigUniversal(max_new_bytes=50, temperature=0.8)
    )
    print(f"  Generated: {generated[:100]}...")
    
    print("\n✓ FluxLM-Universal ready!")
