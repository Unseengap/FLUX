"""
Phase 9 — WaveToText: Last-Mile Wave→Bytes Decoder

Convert a single 432-dim wave into its byte representation.
This is the text-specific last mile — the only part that is
NOT reusable across modalities.

Architecture:
    wave [432] → project to hidden → unroll bytes autoregressively
    Simple 1-layer GRU (tiny) — only needs to spell one word,
    not generate coherent prose. The coherence lives in WaveGenerator.

For other modalities, replace this with:
    WaveToImage: wave → pixels
    WaveToAudio: wave → waveform
    WaveToMol:   wave → molecular graph
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class WaveToText(nn.Module):
    """
    Convert a single 432-dim wave into its byte representation.

    Args:
        wave_dim: Input wave dimension (432)
        hidden_dim: Small GRU hidden (256 — only spelling one word)
        max_bytes: Maximum bytes per chunk (20)
        vocab_size: Byte vocabulary (256)
    """

    def __init__(
        self,
        wave_dim: int = 432,
        hidden_dim: int = 256,
        max_bytes: int = 20,
        vocab_size: int = 256,
    ):
        super().__init__()
        self.wave_dim = wave_dim
        self.hidden_dim = hidden_dim
        self.max_bytes = max_bytes
        self.vocab_size = vocab_size

        # Special token indices
        self.BOS = vocab_size       # 256
        self.EOS = vocab_size + 1   # 257

        # Wave → GRU initial hidden state
        self.wave_to_hidden = nn.Linear(wave_dim, hidden_dim)

        # Byte embedding (256 normal bytes + BOS + EOS)
        self.byte_embed = nn.Embedding(vocab_size + 2, 64)

        # Tiny GRU — only needs to spell one word
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Output projection (256 bytes + EOS)
        self.output_proj = nn.Linear(hidden_dim, vocab_size + 1)

    def forward(
        self,
        wave: torch.Tensor,
        target_bytes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Teacher-forced: predict bytes + EOS for one chunk given its wave.

        Args:
            wave: [432] the wave to decode
            target_bytes: [chunk_len] byte values for this chunk (0..255)

        Returns:
            [chunk_len+1, 257] logits (256 bytes + EOS)
            Targets should be: byte_0, byte_1, ..., byte_{n-1}, EOS
        """
        # Project wave → GRU initial hidden [1, 1, hidden_dim]
        hidden = self.wave_to_hidden(wave).unsqueeze(0).unsqueeze(0)

        # Build input sequence: BOS followed by all target bytes
        # Input:  [BOS, byte_0, byte_1, ..., byte_{n-1}]  (length = chunk_len + 1)
        # Target: [byte_0, byte_1, ..., byte_{n-1}, EOS]  (length = chunk_len + 1)
        bos = torch.full(
            (1,), self.BOS, dtype=torch.long, device=wave.device
        )
        input_seq = torch.cat([bos, target_bytes])  # [chunk_len + 1]
        embedded = self.byte_embed(input_seq).unsqueeze(0)  # [1, chunk_len+1, 64]

        # Run GRU
        output, _ = self.gru(embedded, hidden)  # [1, chunk_len+1, hidden_dim]
        logits = self.output_proj(output.squeeze(0))  # [chunk_len+1, 257]

        return logits

    def forward_batch(
        self,
        waves: torch.Tensor,
        target_list: list,
    ) -> torch.Tensor:
        """
        Batched teacher-forced training over multiple chunks.

        Pads all targets to max length, runs in parallel.

        Args:
            waves: [B, 432] batch of chunk waves
            target_list: List of [chunk_len_i] tensors (byte values)

        Returns:
            total cross-entropy loss (averaged across valid tokens)
        """
        device = waves.device
        batch_size = waves.shape[0]
        max_len = max(t.shape[0] for t in target_list)

        # +1 so there is always room for the EOS target after the last byte
        seq_len = max_len + 1

        # Pad targets to seq_len  (pad value = -100 = CE ignore)
        padded_targets = torch.full(
            (batch_size, seq_len), -100, dtype=torch.long, device=device
        )
        padded_inputs = torch.full(
            (batch_size, seq_len), self.BOS, dtype=torch.long, device=device
        )

        for i, t in enumerate(target_list):
            length = t.shape[0]
            # Target: byte_0, byte_1, ..., byte_{n-1}, EOS
            padded_targets[i, :length] = t
            padded_targets[i, length] = self.vocab_size  # EOS always present

            # Input: BOS, byte_0, byte_1, ..., byte_{n-1}
            # (shifted right — last input is the last real byte, predicting EOS)
            if length > 0:
                padded_inputs[i, 1:length + 1] = t
            # Position 0 is already BOS

        # Wave → hidden [1, B, hidden_dim]
        hidden = self.wave_to_hidden(waves).unsqueeze(0)  # [1, B, hidden_dim]

        # Embed inputs [B, seq_len, 64]
        embedded = self.byte_embed(padded_inputs)

        # GRU forward [B, max_len, hidden_dim]
        output, _ = self.gru(embedded, hidden)

        # Logits [B, max_len, 257]
        logits = self.output_proj(output)

        # Cross-entropy with ignore_index=-100
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size + 1),
            padded_targets.view(-1),
            ignore_index=-100,
        )

        return loss

    @torch.no_grad()
    def decode(
        self,
        wave: torch.Tensor,
        temperature: float = 0.8,
        max_bytes: Optional[int] = None,
    ) -> bytes:
        """
        Autoregressive: generate bytes from a wave.

        Args:
            wave: [432] the wave to decode
            temperature: Sampling temperature
            max_bytes: Override max bytes per chunk

        Returns:
            Decoded bytes
        """
        max_b = max_bytes or self.max_bytes
        hidden = self.wave_to_hidden(wave).unsqueeze(0).unsqueeze(0)  # [1, 1, H]
        current = torch.full(
            (1,), self.BOS, dtype=torch.long, device=wave.device
        )
        result = []

        for _ in range(max_b):
            embedded = self.byte_embed(current).unsqueeze(0)  # [1, 1, 64]
            output, hidden = self.gru(embedded, hidden)  # [1, 1, H]
            logits = self.output_proj(output.squeeze(0).squeeze(0))  # [257]

            # Temperature sampling
            scaled = logits / max(temperature, 1e-8)
            probs = F.softmax(scaled, dim=-1)
            next_byte = torch.multinomial(probs, 1).item()

            if next_byte == self.vocab_size:  # EOS
                break
            if next_byte < 256:
                result.append(next_byte)
                current = torch.tensor([next_byte], device=wave.device)

        return bytes(result)

    @torch.no_grad()
    def decode_batch(
        self,
        waves: torch.Tensor,
        temperature: float = 0.8,
    ) -> list:
        """
        Decode multiple chunks in parallel.

        Args:
            waves: [B, 432] batch of chunk waves
            temperature: Sampling temperature

        Returns:
            List of bytes objects
        """
        return [self.decode(w, temperature) for w in waves]
