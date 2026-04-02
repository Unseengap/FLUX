"""
Phase 8.8: Text ↔ Wave Adapters

TextToWave wraps the Phase 1 CSE (already does byte→wave perfectly).
WaveToText is a simple last-mile byte decoder for single chunks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
import sys
from pathlib import Path

# Add project paths
_PHASES_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PHASES_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PHASES_DIR / 'phase1') not in sys.path:
    sys.path.insert(0, str(_PHASES_DIR / 'phase1'))

from wave_to_x import XToWave, WaveToX, register_input_adapter, register_output_adapter


@register_input_adapter('text')
class TextToWave(XToWave):
    """
    Text → Wave adapter.
    
    Wraps the Phase 1 CSE. This is the original FLUX encoder.
    No new architecture needed — CSE already does this perfectly.
    """
    
    def __init__(
        self,
        wave_dim: int = 432,
        cse: Optional[nn.Module] = None,
        device: str = 'cpu',
    ):
        super().__init__('text', wave_dim)
        self.device = device
        self._cse = cse
        self._cse_loaded = cse is not None
    
    def _ensure_cse(self):
        """Lazy load CSE from checkpoint if not provided."""
        if self._cse_loaded:
            return
        
        try:
            from flux_utils import load_checkpoint
            from cse import ContinuousSemanticEncoder
            
            ckpt = load_checkpoint(1)
            self._cse = ContinuousSemanticEncoder(**ckpt['config'])
            self._cse.load_state_dict(ckpt['state_dict'])
            self._cse.eval()
            self._cse = self._cse.to(self.device)
            self._cse_loaded = True
        except Exception as e:
            raise RuntimeError(f"Could not load CSE: {e}")
    
    @property
    def cse(self) -> nn.Module:
        self._ensure_cse()
        return self._cse
    
    @cse.setter  
    def cse(self, value: nn.Module):
        self._cse = value
        self._cse_loaded = value is not None
    
    def encode(self, text: str) -> Tensor:
        """
        Encode text to wave sequence.
        
        Args:
            text: UTF-8 string
        
        Returns:
            [seq_len, 432] wave tensor
        """
        with torch.no_grad():
            wave = self.cse.encode(text)
        return wave.full.to(self.device)
    
    def encode_pooled(self, text: str) -> Tensor:
        """
        Encode text to single wave vector (mean pooled).
        
        Returns:
            [432] wave tensor
        """
        return self.encode(text).mean(dim=0)


@register_output_adapter('text')
class WaveToText(WaveToX):
    """
    ############################################################################
    # DEPRECATED as of v6.0-autonomous (April 2026)
    #
    # This wave-to-text decoder is SUPERSEDED by embedded models.instruct.
    # Use the embedded Qwen2.5-1.5B-Instruct model for text generation instead.
    #
    # This class is preserved for HISTORICAL REFERENCE ONLY.
    # See: DOCS/FLUX_CONSOLIDATION_ROADMAP.md
    ############################################################################
    
    Wave → Text adapter.
    
    Decodes a single wave (representing ~1 word/chunk) to UTF-8 bytes.
    Simpler than Phase 8's full decoder — this handles one chunk at a time.
    
    Architecture:
        wave [432] → hidden [256] → logits [256 × max_bytes]
        Autoregressive byte-by-byte with learned stop token.
    """
    
    def __init__(
        self,
        wave_dim: int = 432,
        hidden_dim: int = 256,
        max_bytes: int = 24,
        device: str = 'cpu',
    ):
        super().__init__('text', wave_dim)
        self.hidden_dim = hidden_dim
        self.max_bytes = max_bytes
        self.device = device
        
        # Wave → hidden
        self.wave_proj = nn.Linear(wave_dim, hidden_dim)
        
        # Byte embedding (256 possible bytes + stop token)
        self.byte_embed = nn.Embedding(257, hidden_dim)
        
        # GRU for autoregressive decoding
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, 257)
        
        self.to(device)
    
    def decode(
        self,
        wave: Tensor,
        temperature: float = 0.8,
        max_bytes: Optional[int] = None,
    ) -> str:
        """
        Decode wave to text string.
        
        Args:
            wave: [432] wave tensor
            temperature: Sampling temperature
            max_bytes: Override max bytes
        
        Returns:
            Decoded UTF-8 string
        """
        max_len = max_bytes or self.max_bytes
        
        # Project wave to hidden
        wave = wave.to(self.device)
        if wave.dim() == 1:
            wave = wave.unsqueeze(0)
        
        hidden = self.wave_proj(wave).unsqueeze(0)  # [1, 1, hidden]
        
        # Start token (space = 32)
        current = torch.tensor([[32]], device=self.device)
        
        bytes_out = []
        for _ in range(max_len):
            # Embed current byte
            emb = self.byte_embed(current)  # [1, 1, hidden]
            
            # GRU step
            out, hidden = self.gru(emb, hidden)
            
            # Predict next byte
            logits = self.out_proj(out.squeeze(1))  # [1, 257]
            
            # Sample
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_byte = torch.multinomial(probs, 1)
            else:
                next_byte = logits.argmax(dim=-1, keepdim=True)
            
            byte_val = next_byte.item()
            
            # Stop token (256) or null byte
            if byte_val >= 256 or byte_val == 0:
                break
            
            bytes_out.append(byte_val)
            current = next_byte
        
        # Decode bytes to string
        return bytes(bytes_out).decode('utf-8', errors='replace')
    
    def forward_train(
        self,
        wave: Tensor,
        target_bytes: Tensor,
    ) -> Tensor:
        """
        Training forward pass.
        
        Args:
            wave: [batch, 432] wave tensors
            target_bytes: [batch, max_bytes] target byte sequences
        
        Returns:
            [batch, max_bytes, 257] logits
        """
        # Project waves
        hidden = self.wave_proj(wave).unsqueeze(0)  # [1, batch, hidden]
        
        # Embed target (shifted by 1 for teacher forcing)
        emb = self.byte_embed(target_bytes)  # [batch, seq, hidden]
        
        # GRU
        out, _ = self.gru(emb, hidden)
        
        # Project to logits
        logits = self.out_proj(out)  # [batch, seq, 257]
        
        return logits


# ─────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("Text Adapters — Module Check")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test WaveToText (doesn't need CSE)
    decoder = WaveToText(device=device)
    test_wave = torch.randn(432, device=device)
    text = decoder.decode(test_wave)
    print(f"  WaveToText output: '{text[:30]}...'")
    
    print("  ✓ Text adapters ready (WaveToText)")
    print("  ℹ TextToWave requires Phase 1 CSE checkpoint")
