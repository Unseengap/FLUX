"""
Phase 10: WaveGenerator — Concept-Level Generation

Generates semantic waves from field context. Unlike byte-level
generation (~100 steps/sentence), wave generation operates at
word/concept level (~15 steps/sentence → 6x faster).

The WaveGenerator produces waves that encode semantic meaning.
The WaveToText module handles the "last mile" spelling.

This is INFERENCE from the trained field — the field has learned
semantic structure through physics-based injection. WaveGenerator
extracts and generates from that knowledge.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class WaveGenerationResult:
    """Result from wave generation."""
    waves: torch.Tensor          # [N, wave_dim] generated waves
    confidences: torch.Tensor    # [N] confidence per wave
    stop_reason: str             # Why generation stopped
    n_steps: int                 # Number of generation steps


class WaveGenerator(nn.Module):
    """
    Generate semantic waves from field context.
    
    Architecture:
    ```
    field_context → [project] → hidden
    hidden + prev_wave → [recurrent] → next_wave, confidence
    ```
    
    Each generated wave represents a concept/word.
    The WaveToText decoder converts waves to bytes.
    
    Args:
        wave_dim: Semantic wave dimension (432)
        field_features: Field feature dimension (512)
        hidden_dim: Hidden state dimension
        n_layers: Number of recurrent layers
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        wave_dim: int = 432,
        field_features: int = 512,
        hidden_dim: int = 512,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.wave_dim = wave_dim
        self.field_features = field_features
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Project field context to hidden dim
        self.field_to_hidden = nn.Linear(field_features, hidden_dim)
        
        # Project wave to hidden dim
        self.wave_to_hidden = nn.Linear(wave_dim, hidden_dim)
        
        # Recurrent core (GRU for efficiency)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )
        
        # Wave prediction head
        self.to_wave = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, wave_dim),
        )
        
        # Confidence head (probability of continuing)
        self.to_confidence = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        # Start token (learnable)
        self.start_wave = nn.Parameter(torch.randn(1, wave_dim) * 0.02)
        
        # config for saving
        self.config = {
            'wave_dim': wave_dim,
            'field_features': field_features,
            'hidden_dim': hidden_dim,
            'n_layers': n_layers,
            'dropout': dropout,
        }
    
    def forward(
        self,
        field_context: torch.Tensor,
        target_waves: Optional[torch.Tensor] = None,
        max_waves: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate waves from field context.
        
        Args:
            field_context: [field_features] or [batch, field_features]
            target_waves: [N, wave_dim] target waves for teacher forcing
            max_waves: Maximum waves to generate
        
        Returns:
            waves: [N, wave_dim] generated waves
            confidences: [N] confidence scores
        """
        if field_context.dim() == 1:
            field_context = field_context.unsqueeze(0)
        
        batch_size = field_context.shape[0]
        device = field_context.device
        
        # Initialize hidden state from field context
        h_init = self.field_to_hidden(field_context)  # [batch, hidden]
        hidden = h_init.unsqueeze(0).repeat(self.n_layers, 1, 1)  # [layers, batch, hidden]
        
        # Start with learned start wave
        prev_wave = self.start_wave.expand(batch_size, -1)  # [batch, wave_dim]
        
        waves = []
        confidences = []
        
        # Teacher forcing or autoregressive
        if target_waves is not None:
            n_waves = target_waves.shape[0]
            for i in range(n_waves - 1):
                wave_hidden = self.wave_to_hidden(target_waves[i:i+1].expand(batch_size, -1))
                gru_input = wave_hidden.unsqueeze(1)  # [batch, 1, hidden]
                
                output, hidden = self.gru(gru_input, hidden)
                
                next_wave = self.to_wave(output.squeeze(1))
                conf = self.to_confidence(output.squeeze(1))
                
                waves.append(next_wave)
                confidences.append(conf)
        else:
            # Autoregressive generation
            for _ in range(max_waves):
                wave_hidden = self.wave_to_hidden(prev_wave)
                gru_input = wave_hidden.unsqueeze(1)
                
                output, hidden = self.gru(gru_input, hidden)
                
                next_wave = self.to_wave(output.squeeze(1))
                conf = self.to_confidence(output.squeeze(1))
                
                waves.append(next_wave)
                confidences.append(conf)
                
                prev_wave = next_wave
                
                # Stop if confidence drops too low
                if conf.mean() < 0.1:
                    break
        
        if not waves:
            return torch.zeros(0, self.wave_dim, device=device), torch.zeros(0, device=device)
        
        waves = torch.stack(waves, dim=0).squeeze(1)  # [N, wave_dim]
        confidences = torch.stack(confidences, dim=0).squeeze()  # [N]
        
        return waves, confidences
    
    def generate(
        self,
        field_context: torch.Tensor,
        max_waves: int = 50,
        temperature: float = 1.0,
        min_confidence: float = 0.05,
    ) -> WaveGenerationResult:
        """
        Generate waves with stopping criteria.
        
        Args:
            field_context: [field_features] field context
            max_waves: Maximum waves to generate
            temperature: Sampling temperature (scales confidence)
            min_confidence: Minimum confidence to continue
        
        Returns:
            WaveGenerationResult with waves, confidences, etc.
        """
        self.eval()
        device = field_context.device
        
        if field_context.dim() == 1:
            field_context = field_context.unsqueeze(0)
        
        # Initialize
        h_init = self.field_to_hidden(field_context)
        hidden = h_init.unsqueeze(0).repeat(self.n_layers, 1, 1)
        prev_wave = self.start_wave.to(device)
        
        waves = []
        confidences = []
        stop_reason = "max_waves"
        
        with torch.no_grad():
            for step in range(max_waves):
                wave_hidden = self.wave_to_hidden(prev_wave)
                gru_input = wave_hidden.unsqueeze(1)
                
                output, hidden = self.gru(gru_input, hidden)
                
                next_wave = self.to_wave(output.squeeze(1))
                conf = self.to_confidence(output.squeeze(1)).squeeze()
                
                # Temperature scaling
                if temperature != 1.0:
                    # Add small noise based on temperature
                    noise = torch.randn_like(next_wave) * temperature * 0.1
                    next_wave = next_wave + noise
                    conf = (conf - 0.5) / temperature + 0.5
                    conf = conf.clamp(0, 1)
                
                waves.append(next_wave.squeeze(0))
                confidences.append(conf.item() if conf.dim() == 0 else conf.mean().item())
                
                prev_wave = next_wave
                
                # Stopping criteria
                if conf.mean() < min_confidence:
                    stop_reason = "low_confidence"
                    break
        
        if not waves:
            return WaveGenerationResult(
                waves=torch.zeros(0, self.wave_dim, device=device),
                confidences=torch.zeros(0, device=device),
                stop_reason="no_output",
                n_steps=0,
            )
        
        return WaveGenerationResult(
            waves=torch.stack(waves),
            confidences=torch.tensor(confidences, device=device),
            stop_reason=stop_reason,
            n_steps=len(waves),
        )
    
    def forward_teacher(
        self,
        field_context: torch.Tensor,
        target_waves: torch.Tensor,
    ) -> torch.Tensor:
        """
        Teacher-forced forward pass for training.
        
        Args:
            field_context: [field_features] field context
            target_waves: [N, wave_dim] target wave sequence
        
        Returns:
            predicted_waves: [N-1, wave_dim] predicted waves (shifted by 1)
        """
        waves, _ = self.forward(field_context, target_waves, max_waves=0)
        return waves


class WaveChunker(nn.Module):
    """
    Detect word/concept boundaries in wave sequences.
    
    Given a continuous wave sequence from CSE, identifies where
    one word ends and another begins. This enables:
    - Chunk-level wave generation
    - Proper alignment for WaveToText
    
    Architecture:
    Uses learned boundary detection via 1D convolution
    across the wave sequence.
    """
    
    def __init__(
        self,
        wave_dim: int = 432,
        hidden_dim: int = 128,
        kernel_size: int = 5,
    ):
        super().__init__()
        
        self.wave_dim = wave_dim
        self.hidden_dim = hidden_dim
        
        # Convolution to detect boundaries
        self.conv = nn.Sequential(
            nn.Conv1d(wave_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 1, 1),  # Boundary probability per position
            nn.Sigmoid(),
        )
        
        # Chunk aggregation
        self.chunk_aggregate = nn.Sequential(
            nn.Linear(wave_dim, wave_dim),
            nn.Tanh(),
        )
        
        self.config = {
            'wave_dim': wave_dim,
            'hidden_dim': hidden_dim,
            'kernel_size': kernel_size,
        }
    
    def forward(
        self,
        wave_seq: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        Chunk wave sequence into word-level waves.
        
        Args:
            wave_seq: [seq_len, wave_dim] continuous wave sequence
            threshold: Boundary detection threshold
        
        Returns:
            chunk_waves: [n_chunks, wave_dim] word-level waves
            spans: List of (start, end) indices
        """
        seq_len = wave_seq.shape[0]
        
        # Detect boundaries
        x = wave_seq.T.unsqueeze(0)  # [1, wave_dim, seq_len]
        boundary_probs = self.conv(x).squeeze()  # [seq_len]
        
        # Find boundary positions
        boundaries = [0]  # Start
        for i in range(1, seq_len - 1):
            if boundary_probs[i] > threshold:
                boundaries.append(i)
        boundaries.append(seq_len)  # End
        
        # Aggregate waves within chunks
        chunk_waves = []
        spans = []
        
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            if end > start:
                chunk = wave_seq[start:end].mean(dim=0)  # Average
                chunk = self.chunk_aggregate(chunk)
                chunk_waves.append(chunk)
                spans.append((start, end))
        
        if not chunk_waves:
            # Fallback: entire sequence as one chunk
            chunk = wave_seq.mean(dim=0)
            chunk = self.chunk_aggregate(chunk)
            return chunk.unsqueeze(0), [(0, seq_len)]
        
        return torch.stack(chunk_waves), spans
    
    def chunk_text(
        self,
        text: str,
        cse,
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Convenience method: chunk text → wave chunks + words.
        
        Args:
            text: Input text
            cse: ContinuousSemanticEncoder instance
        
        Returns:
            chunk_waves: [n_chunks, wave_dim] 
            words: List of word strings
        """
        # Encode text
        wave_result = cse.encode(text)
        wave_seq = wave_result.full  # [seq_len, wave_dim]
        
        # Get chunks
        chunk_waves, spans = self.forward(wave_seq)
        
        # Extract words from spans
        text_bytes = text.encode('utf-8', errors='replace')
        words = []
        for start, end in spans:
            # Map positions to original text (approximate)
            char_start = min(start, len(text))
            char_end = min(end, len(text))
            word = text[char_start:char_end].strip()
            if word:
                words.append(word)
            else:
                words.append('<unk>')
        
        return chunk_waves, words


class WaveToText(nn.Module):
    """
    Convert semantic waves to UTF-8 bytes.
    
    This is the "last mile" decoder that spells out what
    the wave semantically represents.
    
    Architecture:
    wave → [project] → [autoregressive LSTM] → byte logits
    
    Each wave can produce up to max_bytes characters.
    Uses EOS token (byte 0) to stop.
    """
    
    def __init__(
        self,
        wave_dim: int = 432,
        hidden_dim: int = 256,
        max_bytes: int = 20,
        n_layers: int = 1,
    ):
        super().__init__()
        
        self.wave_dim = wave_dim
        self.hidden_dim = hidden_dim
        self.max_bytes = max_bytes
        self.n_layers = n_layers
        
        # Project wave to hidden
        self.wave_project = nn.Linear(wave_dim, hidden_dim)
        
        # Byte embedding (256 bytes + special tokens)
        self.byte_embed = nn.Embedding(256, hidden_dim)
        
        # Autoregressive LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        
        # Output head
        self.to_logits = nn.Linear(hidden_dim, 256)
        
        # Special tokens
        self.EOS = 0  # End of string
        self.START = 1  # Start token
        
        self.config = {
            'wave_dim': wave_dim,
            'hidden_dim': hidden_dim,
            'max_bytes': max_bytes,
            'n_layers': n_layers,
        }
    
    def forward(
        self,
        wave: torch.Tensor,
        target_bytes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode wave to byte logits.
        
        Args:
            wave: [wave_dim] single wave
            target_bytes: [N] target byte sequence for teacher forcing
        
        Returns:
            logits: [max_bytes, 256] byte logits
        """
        device = wave.device
        
        if wave.dim() == 1:
            wave = wave.unsqueeze(0)
        
        # Initialize hidden from wave
        h_init = self.wave_project(wave)  # [1, hidden]
        hidden = (
            h_init.unsqueeze(0).repeat(self.n_layers, 1, 1),  # h
            torch.zeros_like(h_init).unsqueeze(0).repeat(self.n_layers, 1, 1),  # c
        )
        
        # Start token
        prev_byte = torch.tensor([[self.START]], device=device)
        
        all_logits = []
        
        n_steps = target_bytes.shape[0] if target_bytes is not None else self.max_bytes
        
        for i in range(n_steps):
            byte_emb = self.byte_embed(prev_byte)  # [1, 1, hidden]
            output, hidden = self.lstm(byte_emb, hidden)
            logits = self.to_logits(output.squeeze(1))  # [1, 256]
            all_logits.append(logits)
            
            if target_bytes is not None:
                # Teacher forcing
                prev_byte = target_bytes[i:i+1].unsqueeze(0)
            else:
                # Autoregressive
                prev_byte = logits.argmax(dim=-1, keepdim=True)
        
        return torch.stack(all_logits, dim=1).squeeze(0)  # [N, 256]
    
    def decode(
        self,
        wave: torch.Tensor,
        temperature: float = 1.0,
    ) -> bytes:
        """
        Decode wave to bytes.
        
        Args:
            wave: [wave_dim] single wave
            temperature: Sampling temperature
        
        Returns:
            Decoded bytes
        """
        self.eval()
        device = wave.device
        
        if wave.dim() == 1:
            wave = wave.unsqueeze(0)
        
        # Initialize hidden
        h_init = self.wave_project(wave)
        hidden = (
            h_init.unsqueeze(0).repeat(self.n_layers, 1, 1),
            torch.zeros_like(h_init).unsqueeze(0).repeat(self.n_layers, 1, 1),
        )
        
        prev_byte = torch.tensor([[self.START]], device=device)
        output_bytes = []
        
        with torch.no_grad():
            for _ in range(self.max_bytes):
                byte_emb = self.byte_embed(prev_byte)
                output, hidden = self.lstm(byte_emb, hidden)
                logits = self.to_logits(output.squeeze(1))
                
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Sample or argmax
                if temperature > 0:
                    probs = F.softmax(logits, dim=-1)
                    next_byte = torch.multinomial(probs, 1)
                else:
                    next_byte = logits.argmax(dim=-1, keepdim=True)
                
                byte_val = next_byte.item()
                
                # Stop on EOS
                if byte_val == self.EOS:
                    break
                
                output_bytes.append(byte_val)
                prev_byte = next_byte.unsqueeze(0)
        
        return bytes(output_bytes)
    
    def decode_sequence(
        self,
        waves: torch.Tensor,
        temperature: float = 1.0,
        separator: bytes = b' ',
    ) -> str:
        """
        Decode multiple waves to text.
        
        Args:
            waves: [N, wave_dim] wave sequence
            temperature: Sampling temperature
            separator: Separator between decoded chunks
        
        Returns:
            Decoded text string
        """
        parts = []
        for wave in waves:
            decoded = self.decode(wave, temperature)
            if decoded:
                parts.append(decoded)
        
        combined = separator.join(parts)
        return combined.decode('utf-8', errors='replace')


# ─────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("Wave Modules (Phase 10) — Testing")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test WaveGenerator
    print("\n1. WaveGenerator")
    gen = WaveGenerator(wave_dim=432, field_features=512).to(device)
    field_ctx = torch.randn(512, device=device)
    
    result = gen.generate(field_ctx, max_waves=10)
    print(f"   Generated {result.n_steps} waves")
    print(f"   Shape: {result.waves.shape}")
    print(f"   Stop reason: {result.stop_reason}")
    print(f"   ✓ WaveGenerator works")
    
    # Test WaveChunker
    print("\n2. WaveChunker")
    chunker = WaveChunker(wave_dim=432).to(device)
    wave_seq = torch.randn(50, 432, device=device)
    
    chunks, spans = chunker(wave_seq)
    print(f"   Input: {wave_seq.shape[0]} positions")
    print(f"   Chunks: {chunks.shape[0]} word-level waves")
    print(f"   Spans: {spans[:3]}...")
    print(f"   ✓ WaveChunker works")
    
    # Test WaveToText
    print("\n3. WaveToText")
    decoder = WaveToText(wave_dim=432).to(device)
    wave = torch.randn(432, device=device)
    
    decoded = decoder.decode(wave, temperature=0.8)
    print(f"   Decoded ({len(decoded)} bytes): {decoded[:30]}")
    print(f"   ✓ WaveToText works")
    
    # Count parameters
    total = sum(p.numel() for p in gen.parameters())
    total += sum(p.numel() for p in chunker.parameters())
    total += sum(p.numel() for p in decoder.parameters())
    print(f"\n   Total params (all modules): {total:,}")
    print("=" * 60)
