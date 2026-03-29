"""
WordLevelCSE: Word-level Continuous Semantic Encoder.

Upgrades byte-level encoding to word-level by:
1. Using Phase 1 CSE to encode bytes
2. Detecting word boundaries algorithmically
3. Pooling byte waves into word waves via attention
4. Applying word-to-word interference

Still vocabulary-free: any word in any language works.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Import Phase 1 CSE
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1'))
from cse import ContinuousSemanticEncoder
from wave_types import WAVE_DIMS, TOTAL_WAVE_DIM, SemanticWave
from interference import apply_neighborhood_interference


# ─────────────────────────────────────────────
# WordWave Dataclass
# ─────────────────────────────────────────────

@dataclass
class WordWave:
    """
    Word-level continuous representation.
    Shape: [num_words, 432]
    
    Unlike byte-level SemanticWave:
    - Each position = complete word concept
    - Much shorter sequences
    - Direct semantic units
    """
    words: List[str]                        # Original words (for debugging)
    waves: Tensor                           # [num_words, 432] wave representations
    word_boundaries: List[Tuple[int, int]]  # Byte spans for each word
    
    @property
    def num_words(self) -> int:
        return len(self.words)
    
    @property
    def seq_len(self) -> int:
        """Alias for num_words for compatibility."""
        return self.num_words
    
    @property
    def full(self) -> Tensor:
        """The wave tensor itself."""
        return self.waves
    
    def to_retrieval_vector(self) -> Tensor:
        """Mean pool all word waves for document-level retrieval."""
        return self.waves.mean(dim=0)
    
    def cosine_similarity(self, other: 'WordWave') -> float:
        """Semantic similarity between two word waves."""
        v1 = self.to_retrieval_vector()
        v2 = other.to_retrieval_vector()
        return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
    
    def to(self, device: str) -> 'WordWave':
        """Move tensors to device."""
        return WordWave(
            words=self.words,
            waves=self.waves.to(device),
            word_boundaries=self.word_boundaries,
        )
    
    def detach(self) -> 'WordWave':
        """Detach from computation graph."""
        return WordWave(
            words=self.words,
            waves=self.waves.detach(),
            word_boundaries=self.word_boundaries,
        )


# ─────────────────────────────────────────────
# Word Boundary Detection
# ─────────────────────────────────────────────

class WordBoundaryDetector:
    """
    Detects word boundaries in UTF-8 text.
    Returns byte index spans for each word.
    
    Handles:
    - Whitespace-delimited languages (English, etc.)
    - CJK (each character = word)  
    - Punctuation (separate tokens)
    - Mixed content ("I love 寿司" → ["I", "love", "寿司"])
    """
    
    # Delimiters that separate words
    DELIMITERS = set(' \t\n\r')
    
    # Punctuation that becomes its own token
    PUNCTUATION = set('.,;:!?()[]{}"\'-/\\@#$%^&*+=<>|~`')
    
    # CJK Unicode ranges (Chinese, Japanese, Korean)
    CJK_RANGES = [
        (0x4E00, 0x9FFF),    # CJK Unified Ideographs
        (0x3400, 0x4DBF),    # CJK Extension A
        (0x3000, 0x303F),    # CJK Punctuation
        (0x3040, 0x309F),    # Hiragana
        (0x30A0, 0x30FF),    # Katakana
        (0xAC00, 0xD7AF),    # Hangul Syllables
    ]
    
    def _is_cjk(self, char: str) -> bool:
        """Check if character is CJK (each char = separate word)."""
        code = ord(char)
        return any(start <= code <= end for start, end in self.CJK_RANGES)
    
    def detect(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        Detect word boundaries in text.
        
        Args:
            text: Input string (any UTF-8)
            
        Returns:
            Tuple of:
            - words: List of word strings
            - spans: List of (start_byte_idx, end_byte_idx) tuples
        """
        if not text:
            return [], []
        
        words = []
        spans = []
        
        # Convert to bytes for accurate byte indexing
        text_bytes = text.encode('utf-8')
        
        current_word = []
        current_start = 0
        byte_idx = 0
        
        i = 0
        while i < len(text):
            char = text[i]
            char_bytes = char.encode('utf-8')
            char_byte_len = len(char_bytes)
            
            if char in self.DELIMITERS:
                # Flush current word
                if current_word:
                    word = ''.join(current_word)
                    words.append(word)
                    spans.append((current_start, byte_idx))
                    current_word = []
                # Skip delimiter
                byte_idx += char_byte_len
                current_start = byte_idx
                
            elif char in self.PUNCTUATION:
                # Flush current word
                if current_word:
                    word = ''.join(current_word)
                    words.append(word)
                    spans.append((current_start, byte_idx))
                    current_word = []
                # Punctuation as its own word
                words.append(char)
                spans.append((byte_idx, byte_idx + char_byte_len))
                byte_idx += char_byte_len
                current_start = byte_idx
                
            elif self._is_cjk(char):
                # Flush current word
                if current_word:
                    word = ''.join(current_word)
                    words.append(word)
                    spans.append((current_start, byte_idx))
                    current_word = []
                # CJK character as its own word
                words.append(char)
                spans.append((byte_idx, byte_idx + char_byte_len))
                byte_idx += char_byte_len
                current_start = byte_idx
                
            else:
                # Regular character - add to current word
                if not current_word:
                    current_start = byte_idx
                current_word.append(char)
                byte_idx += char_byte_len
            
            i += 1
        
        # Flush final word
        if current_word:
            word = ''.join(current_word)
            words.append(word)
            spans.append((current_start, byte_idx))
        
        return words, spans


# ─────────────────────────────────────────────
# Word Pooling Attention
# ─────────────────────────────────────────────

class WordPoolingAttention(nn.Module):
    """
    Learns to aggregate byte waves into word waves using attention.
    
    For each word span, computes attention weights over byte positions
    and produces a weighted sum as the word representation.
    
    Example: In "running"
    - 'r' might get high attention (word start)
    - 'ing' suffix gets attention (grammatical marker)
    - Middle letters get less attention
    """
    
    def __init__(self, wave_dim: int = TOTAL_WAVE_DIM):
        super().__init__()
        self.wave_dim = wave_dim
        
        # Attention scoring network
        self.attention = nn.Sequential(
            nn.Linear(wave_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        
        # Optional: position encoding within word
        self.position_weight = nn.Parameter(torch.ones(1))
        
        # Output projection (refine after pooling)
        self.output_proj = nn.Sequential(
            nn.Linear(wave_dim, wave_dim),
            nn.GELU(),
            nn.Linear(wave_dim, wave_dim),
        )
    
    def forward(
        self,
        byte_waves: Tensor,  # [seq_len, wave_dim]
        word_spans: List[Tuple[int, int]],
    ) -> Tensor:  # [num_words, wave_dim]
        """
        Pool byte waves into word waves.
        
        Args:
            byte_waves: [seq_len, wave_dim] byte-level waves
            word_spans: List of (start_idx, end_idx) for each word
            
        Returns:
            [num_words, wave_dim] word-level waves
        """
        if len(word_spans) == 0:
            return torch.zeros(0, self.wave_dim, device=byte_waves.device)
        
        word_waves = []
        
        for start, end in word_spans:
            if start >= byte_waves.shape[0]:
                # Handle edge case: span beyond sequence
                word_waves.append(torch.zeros(self.wave_dim, device=byte_waves.device))
                continue
                
            # Clamp end to sequence length
            end = min(end, byte_waves.shape[0])
            
            if start >= end:
                # Empty span
                word_waves.append(torch.zeros(self.wave_dim, device=byte_waves.device))
                continue
            
            # Get byte waves for this word
            word_bytes = byte_waves[start:end]  # [word_len, wave_dim]
            
            if word_bytes.shape[0] == 1:
                # Single byte word - no attention needed
                word_wave = word_bytes.squeeze(0)
            else:
                # Compute attention weights
                attn_scores = self.attention(word_bytes)  # [word_len, 1]
                attn_weights = F.softmax(attn_scores, dim=0)  # [word_len, 1]
                
                # Weighted sum
                word_wave = (attn_weights * word_bytes).sum(dim=0)  # [wave_dim]
            
            word_waves.append(word_wave)
        
        # Stack into tensor
        stacked = torch.stack(word_waves, dim=0)  # [num_words, wave_dim]
        
        # Refine through output projection
        return self.output_proj(stacked)


# ─────────────────────────────────────────────
# Word-Level Interference
# ─────────────────────────────────────────────

def apply_word_interference(
    word_waves: Tensor,
    radius: int = 3,
    scale: float = 0.1,
) -> Tensor:
    """
    Apply interference between neighboring word waves.
    Similar to byte-level but operates on word positions.
    
    Args:
        word_waves: [num_words, wave_dim] word-level waves
        radius: How many words can affect each other
        scale: Interference strength scaling
        
    Returns:
        [num_words, wave_dim] waves after interference
    """
    # Reuse the byte-level interference function
    return apply_neighborhood_interference(word_waves, radius=radius, scale=scale)


# ─────────────────────────────────────────────
# WordLevelCSE Main Module
# ─────────────────────────────────────────────

class WordLevelCSE(nn.Module):
    """
    Complete word-level encoder.
    
    Pipeline:
    1. Text → bytes (UTF-8)
    2. Bytes → byte waves (Phase 1 CSE)
    3. Detect word boundaries
    4. Pool byte waves → word waves (attention)
    5. Apply word-to-word interference
    6. Return WordWave
    
    Still vocabulary-free: any word in any language works.
    """
    
    def __init__(
        self,
        byte_cse: Optional[ContinuousSemanticEncoder] = None,
        wave_dim: int = TOTAL_WAVE_DIM,
        interference_radius: int = 3,
        device: str = 'cpu',
    ):
        super().__init__()
        
        # Byte-level encoder (from Phase 1)
        if byte_cse is None:
            byte_cse = ContinuousSemanticEncoder(device=device)
        self.byte_cse = byte_cse
        
        # Word boundary detection
        self.boundary_detector = WordBoundaryDetector()
        
        # Word pooling attention
        self.pooling = WordPoolingAttention(wave_dim=wave_dim)
        
        # Configuration
        self.wave_dim = wave_dim
        self.interference_radius = interference_radius
        self._device_str = device
    
    @property
    def device(self):
        """Get device from pooling layer."""
        return next(self.pooling.parameters()).device
    
    def encode(self, text: str) -> WordWave:
        """
        Encode text to word-level waves.
        
        Args:
            text: Any UTF-8 string
            
        Returns:
            WordWave with one position per word
        """
        if not text or not text.strip():
            return WordWave(
                words=[],
                waves=torch.zeros(0, self.wave_dim, device=self.device),
                word_boundaries=[],
            )
        
        # Step 1: Encode text to byte-level waves
        byte_wave = self.byte_cse.encode(text)
        byte_waves = byte_wave.full  # [seq_len, 432]
        
        # Step 2: Detect word boundaries
        words, byte_spans = self.boundary_detector.detect(text)
        
        if not words:
            return WordWave(
                words=[],
                waves=torch.zeros(0, self.wave_dim, device=self.device),
                word_boundaries=[],
            )
        
        # Step 3: Pool byte waves into word waves
        word_waves = self.pooling(byte_waves, byte_spans)  # [num_words, 432]
        
        # Step 4: Apply word-to-word interference
        word_waves = apply_word_interference(
            word_waves,
            radius=self.interference_radius,
            scale=0.1,
        )
        
        return WordWave(
            words=words,
            waves=word_waves,
            word_boundaries=byte_spans,
        )
    
    def encode_batch(self, texts: List[str]) -> List[WordWave]:
        """Encode a batch of texts."""
        return [self.encode(text) for text in texts]
    
    def forward(self, texts: List[str]) -> List[WordWave]:
        """nn.Module forward pass."""
        return self.encode_batch(texts)
    
    def get_word_embedding(self, word: str) -> Tensor:
        """
        Get embedding for a single word.
        Useful for similarity comparisons.
        
        Args:
            word: Single word string
            
        Returns:
            [432] wave vector for the word
        """
        wave = self.encode(word)
        if wave.num_words == 0:
            return torch.zeros(self.wave_dim, device=self.device)
        return wave.waves[0]
    
    def word_similarity(self, word1: str, word2: str) -> float:
        """
        Compute cosine similarity between two words.
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            Cosine similarity in [-1, 1]
        """
        v1 = self.get_word_embedding(word1)
        v2 = self.get_word_embedding(word2)
        return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()


# ─────────────────────────────────────────────
# Word Decoder (Training Only)
# ─────────────────────────────────────────────

class WordDecoder(nn.Module):
    """
    Decodes a word wave back to byte predictions.
    Used for reconstruction training only.
    
    For each word wave, predicts the sequence of bytes
    that make up that word (variable length).
    """
    
    def __init__(
        self,
        wave_dim: int = TOTAL_WAVE_DIM,
        max_word_len: int = 32,
        num_bytes: int = 256,
    ):
        super().__init__()
        self.wave_dim = wave_dim
        self.max_word_len = max_word_len
        self.num_bytes = num_bytes
        
        # Decoder RNN: generate bytes autoregressively
        self.hidden_proj = nn.Linear(wave_dim, 256)
        self.rnn = nn.GRU(
            input_size=num_bytes + wave_dim,  # prev byte embedding + word wave
            hidden_size=256,
            num_layers=2,
            batch_first=True,
        )
        self.output = nn.Linear(256, num_bytes)
        
        # Byte embedding for autoregressive decoding
        self.byte_embed = nn.Embedding(num_bytes, num_bytes)
        nn.init.eye_(self.byte_embed.weight)  # Start with one-hot
    
    def forward(
        self,
        word_wave: Tensor,  # [wave_dim] or [batch, wave_dim]
        target_bytes: Optional[Tensor] = None,  # [batch, max_len] for teacher forcing
        max_len: int = 32,
    ) -> Tensor:
        """
        Decode word wave to byte sequence.
        
        Args:
            word_wave: [wave_dim] or [batch, wave_dim] word representation
            target_bytes: Optional target for teacher forcing
            max_len: Maximum bytes to generate
            
        Returns:
            [batch, max_len, num_bytes] logits
        """
        # Handle unbatched input
        if word_wave.dim() == 1:
            word_wave = word_wave.unsqueeze(0)
        
        batch_size = word_wave.shape[0]
        device = word_wave.device
        
        # Initialize hidden state from word wave
        h0 = self.hidden_proj(word_wave)  # [batch, 256]
        h0 = h0.unsqueeze(0).repeat(2, 1, 1)  # [2, batch, 256] for 2-layer GRU
        
        # Start token (zero byte)
        current_byte = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        outputs = []
        hidden = h0
        
        for t in range(max_len):
            # Embed current byte
            byte_emb = self.byte_embed(current_byte)  # [batch, num_bytes]
            
            # Combine with word wave
            rnn_input = torch.cat([byte_emb, word_wave], dim=-1)  # [batch, num_bytes + wave_dim]
            rnn_input = rnn_input.unsqueeze(1)  # [batch, 1, input_dim]
            
            # RNN step
            rnn_out, hidden = self.rnn(rnn_input, hidden)  # [batch, 1, 256]
            
            # Output byte logits
            logits = self.output(rnn_out.squeeze(1))  # [batch, num_bytes]
            outputs.append(logits)
            
            # Next input: teacher forcing or argmax
            if target_bytes is not None and t < target_bytes.shape[1]:
                current_byte = target_bytes[:, t]
            else:
                current_byte = logits.argmax(dim=-1)
        
        return torch.stack(outputs, dim=1)  # [batch, max_len, num_bytes]


# ─────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────

def compare_byte_vs_word(text: str, byte_cse: ContinuousSemanticEncoder, word_cse: WordLevelCSE):
    """
    Compare byte-level vs word-level encoding for visualization.
    
    Returns dict with comparison metrics.
    """
    # Byte-level encoding
    byte_wave = byte_cse.encode(text)
    byte_len = byte_wave.seq_len
    
    # Word-level encoding
    word_wave = word_cse.encode(text)
    word_len = word_wave.num_words
    
    compression_ratio = byte_len / max(word_len, 1)
    
    return {
        'text': text,
        'byte_positions': byte_len,
        'word_positions': word_len,
        'compression_ratio': compression_ratio,
        'words': word_wave.words,
        'byte_wave_shape': tuple(byte_wave.full.shape),
        'word_wave_shape': tuple(word_wave.waves.shape),
    }
