"""
FluxLM: Complete Vocabulary-Free Wave-Based Language Model.

Combines:
- CSE-Large (10M): Byte → 432D wave encoding
- CWC-Large (5M): Causal direction + order awareness
- WavePredictor (100M): Autoregressive next-wave prediction
- WaveDecoder-Large (9M): Wave → byte decoding

Total: ~124M parameters (GPT-2 small equivalent)

Key features:
- No tokenizer, no vocabulary
- Works on any UTF-8 byte sequence
- Language-agnostic
- Continuous semantic space
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass

# Support both package and direct imports
try:
    from .cse_large import CSELarge, CSE_L_CONFIG, SemanticWave
    from .cwc_large import CWCLarge, CWC_L_CONFIG
    from .wave_predictor import WavePredictor, WAVE_PREDICTOR_CONFIG
    from .wave_decoder_large import WaveDecoderLarge, WAVE_DECODER_L_CONFIG
except ImportError:
    from cse_large import CSELarge, CSE_L_CONFIG, SemanticWave
    from cwc_large import CWCLarge, CWC_L_CONFIG
    from wave_predictor import WavePredictor, WAVE_PREDICTOR_CONFIG
    from wave_decoder_large import WaveDecoderLarge, WAVE_DECODER_L_CONFIG


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

FLUX_LM_CONFIG = {
    'cse': CSE_L_CONFIG,
    'cwc': CWC_L_CONFIG,
    'predictor': WAVE_PREDICTOR_CONFIG,
    'decoder': WAVE_DECODER_L_CONFIG,
    
    # Training config
    'max_seq_len': 512,
    'gradient_checkpointing': False,
}


# ─────────────────────────────────────────────
# Generation Config
# ─────────────────────────────────────────────

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_bytes: int = 100
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    stop_bytes: List[int] = None  # Bytes that stop generation
    
    def __post_init__(self):
        if self.stop_bytes is None:
            self.stop_bytes = [0]  # Null byte


# ─────────────────────────────────────────────
# FluxLM Main Model
# ─────────────────────────────────────────────

class FluxLM(nn.Module):
    """
    FLUX Language Model: Vocabulary-free autoregressive generation.
    
    Pipeline:
        text → bytes → CSE → waves [432] → CWC → causal_waves [608]
        → WavePredictor → next_wave [608] → Decoder → byte logits [256]
    
    ~124M parameters total
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        config = config or FLUX_LM_CONFIG
        
        self.config = config
        self.max_seq_len = config.get('max_seq_len', 512)
        
        # Initialize components
        self.cse = CSELarge(config.get('cse', CSE_L_CONFIG))
        self.cwc = CWCLarge(config.get('cwc', CWC_L_CONFIG))
        self.predictor = WavePredictor(config.get('predictor', WAVE_PREDICTOR_CONFIG))
        self.decoder = WaveDecoderLarge(config.get('decoder', WAVE_DECODER_L_CONFIG))
        
        # Dimension checks
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
            'total': count(self),
        }
    
    # ─────────────────────────────────────────────
    # Encoding
    # ─────────────────────────────────────────────
    
    def encode_text(self, text: str) -> Tensor:
        """
        Encode text to causal waves.
        
        Args:
            text: Input text string
        Returns:
            [seq_len, 608] causal waves
        """
        # Text → bytes → semantic waves
        semantic_wave = self.cse.encode(text)  # SemanticWave
        wave = semantic_wave.full  # [seq_len, 432]
        
        # Add causal direction
        causal_wave = self.cwc(wave)  # [seq_len, 608]
        
        return causal_wave
    
    def encode_bytes(self, bytes_tensor: Tensor) -> Tensor:
        """
        Encode byte tensor to causal waves.
        
        Args:
            bytes_tensor: [seq_len] or [batch, seq_len] byte values
        Returns:
            [seq_len, 608] or [batch, seq_len, 608] causal waves
        """
        # Bytes → semantic waves
        semantic_wave = self.cse.encode_bytes(bytes_tensor)
        wave = semantic_wave.full
        
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
        return_loss: bool = True,
    ) -> Dict[str, Tensor]:
        """
        Forward pass for training.
        
        Args:
            input_bytes: [batch, seq_len] input byte values
            target_bytes: [batch, seq_len] target byte values (shifted by 1)
            return_loss: Whether to compute and return loss
        
        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        batch_size, seq_len = input_bytes.shape
        
        # Encode to waves
        semantic_wave = self.cse.encode_bytes(input_bytes)
        wave = semantic_wave.full  # [batch, seq_len, 432]
        
        # Add causal direction with order score
        causal_wave, order_score = self.cwc(wave, return_order_score=True)
        # causal_wave: [batch, seq_len, 608]
        
        # Predict next waves
        predicted_waves, _ = self.predictor(causal_wave)
        # predicted_waves: [batch, seq_len, 608]
        
        # Decode to byte logits
        logits = self.decoder(predicted_waves)
        # logits: [batch, seq_len, 256]
        
        result = {'logits': logits, 'order_score': order_score}
        
        if return_loss and target_bytes is not None:
            # Next-byte prediction loss
            loss = F.cross_entropy(
                logits.view(-1, 256),
                target_bytes.view(-1),
                ignore_index=-100,  # Ignore padding
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
        config: GenerationConfig = None,
    ) -> str:
        """
        Generate text autoregressively.
        
        Args:
            prompt: Initial text
            config: Generation configuration
        
        Returns:
            Generated text (including prompt)
        """
        config = config or GenerationConfig()
        
        # Convert prompt to bytes
        output_bytes = list(prompt.encode('utf-8'))
        
        # Encode initial context
        bytes_tensor = torch.tensor(output_bytes, dtype=torch.long, device=self.device)
        
        # KV cache for efficiency
        past_kv = None
        
        for _ in range(config.max_new_bytes):
            # Always encode full context through CSE and CWC
            # (they don't support incremental encoding)
            input_tensor = bytes_tensor.unsqueeze(0)  # [1, seq_len]
            
            # Encode to causal waves
            semantic_wave = self.cse.encode_bytes(input_tensor)
            wave = semantic_wave.full  # [1, seq_len, 432]
            causal_waves = self.cwc(wave)  # [1, seq_len, 608]
            
            # Predict next wave (KV cache only helps predictor)
            if past_kv is None:
                next_wave, past_kv = self.predictor.predict_next(causal_waves, None)
            else:
                # With cache, only pass the new position to predictor
                next_wave, past_kv = self.predictor.predict_next(causal_waves[:, -1:], past_kv)
            
            # Decode to byte logits
            logits = self.decoder(next_wave.unsqueeze(1) if next_wave.dim() == 2 else next_wave)
            logits = logits.squeeze(1)  # [1, 256]
            
            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                for prev_byte in set(output_bytes[-50:]):  # Last 50 bytes
                    logits[0, prev_byte] /= config.repetition_penalty
            
            # Sample next byte
            next_byte = self._sample_byte(
                logits[0],
                config.temperature,
                config.top_k,
                config.top_p,
            )
            
            # Check stop conditions
            if next_byte in config.stop_bytes:
                break
            
            # Append to output
            output_bytes.append(next_byte)
            bytes_tensor = torch.cat([
                bytes_tensor,
                torch.tensor([next_byte], dtype=torch.long, device=self.device)
            ])
        
        # Decode bytes to string
        return bytes(output_bytes).decode('utf-8', errors='replace')
    
    def _sample_byte(
        self,
        logits: Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> int:
        """Sample a byte from logits with temperature, top-k, and top-p."""
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, 256)
            indices_to_remove = logits < torch.topk(logits, top_k)[0][-1]
            logits[indices_to_remove] = float('-inf')
        
        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        next_byte = torch.multinomial(probs, 1).item()
        
        return next_byte
    
    # ─────────────────────────────────────────────
    # Order Discrimination (Training)
    # ─────────────────────────────────────────────
    
    def compute_order_loss(self, text: str) -> Tensor:
        """
        Compute order discrimination loss.
        Original text should score higher than shuffled.
        
        This fixes the coherence_gap = 0.0 problem from Phase 1.5.
        """
        words = text.split()
        if len(words) < 4:
            return torch.tensor(0.0, device=self.device)
        
        # Original
        original_wave = self.encode_text(text)
        original_score = self.cwc.get_order_score(original_wave)
        
        # Shuffled
        shuffled_words = words.copy()
        random.shuffle(shuffled_words)
        shuffled_text = ' '.join(shuffled_words)
        shuffled_wave = self.encode_text(shuffled_text)
        shuffled_score = self.cwc.get_order_score(shuffled_wave)
        
        # Margin ranking loss: original should be higher
        margin = 0.5
        loss = F.relu(margin - original_score + shuffled_score)
        
        return loss.mean()
    
    # ─────────────────────────────────────────────
    # Contradiction Detection (Inherited from CWC)
    # ─────────────────────────────────────────────
    
    def compute_tension(self, text1: str, text2: str) -> float:
        """Compute contradiction tension between two texts."""
        wave1 = self.encode_text(text1)
        wave2 = self.encode_text(text2)
        
        tension = self.cwc.compute_tension(wave1, wave2)
        return tension.item()
    
    # ─────────────────────────────────────────────
    # Save / Load
    # ─────────────────────────────────────────────
    
    def save(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'config': self.config,
            'state_dict': self.state_dict(),
            'param_counts': self.count_parameters(),
        }
        torch.save(checkpoint, path)
        print(f"✓ FluxLM saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'FluxLM':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        
        print(f"✓ FluxLM loaded from {path}")
        return model
    
    def save_to_flx(self, path: str):
        """Save as .flx format for integration with FLUX ecosystem."""
        flx_data = {
            'format': 'FLUX-LM',
            'version': '1.0.0',
            'config': self.config,
            'components': {
                'cse': True,
                'cwc': True,
                'predictor': True,
                'decoder': True,
            },
            'cse': {
                'state_dict': self.cse.state_dict(),
                'config': self.config.get('cse', CSE_L_CONFIG),
            },
            'cwc': {
                'state_dict': self.cwc.state_dict(),
                'config': self.config.get('cwc', CWC_L_CONFIG),
            },
            'predictor': {
                'state_dict': self.predictor.state_dict(),
                'config': self.config.get('predictor', WAVE_PREDICTOR_CONFIG),
            },
            'decoder': {
                'state_dict': self.decoder.state_dict(),
                'config': self.config.get('decoder', WAVE_DECODER_L_CONFIG),
            },
            'metadata': {
                'total_params': sum(self.count_parameters().values()) - self.count_parameters()['total'],
                'param_breakdown': self.count_parameters(),
            },
        }
        torch.save(flx_data, path)
        print(f"✓ FluxLM saved to {path} (.flx format)")


# ─────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_params(n: int) -> str:
    """Format parameter count nicely."""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)


if __name__ == '__main__':
    # Test model creation
    print("Creating FluxLM...")
    model = FluxLM()
    
    # Parameter counts
    params = model.count_parameters()
    print("\nParameter counts:")
    for name, count in params.items():
        print(f"  {name}: {format_params(count)}")
    
    # Test encoding
    print("\nTesting encoding...")
    text = "Hello, world!"
    waves = model.encode_text(text)
    print(f"  Input: '{text}'")
    print(f"  Wave shape: {waves.shape}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    input_bytes = torch.tensor([[72, 101, 108, 108, 111]], dtype=torch.long)  # "Hello"
    target_bytes = torch.tensor([[101, 108, 108, 111, 33]], dtype=torch.long)  # "ello!"
    
    output = model(input_bytes, target_bytes)
    print(f"  Logits shape: {output['logits'].shape}")
    print(f"  Loss: {output['loss'].item():.4f}")
    
    print("\n✓ FluxLM ready for training!")
