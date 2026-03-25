"""
WaveChunker: Segment CSE Wave Sequences into Word-Level Chunks

Uses wave interference physics: word boundaries are detected where
consecutive waves have low cosine similarity (coherence drop).
Each chunk is compressed to a single 432-dim wave via learned pooling.

This is the INVERSE of WaveToText. CSE produces byte-level waves.
WaveChunker compresses them into word-level waves.
WaveToText expands word-level waves back into bytes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class WaveChunker(nn.Module):
    """
    Segment CSE wave sequences into word-level wave chunks.

    Uses wave interference physics: word boundaries are detected where
    consecutive waves have low cosine similarity (coherence drop).
    Each chunk is compressed to a single 432-dim wave via learned pooling.

    Args:
        wave_dim: CSE wave dimension (432)
        min_chunk_size: Minimum bytes per chunk (2)
        max_chunk_size: Maximum bytes per chunk (20)
        coherence_threshold: Cosine similarity below which = boundary
    """

    def __init__(
        self,
        wave_dim: int = 432,
        min_chunk_size: int = 2,
        max_chunk_size: int = 20,
        coherence_threshold: float = 0.5,
    ):
        super().__init__()
        self.wave_dim = wave_dim
        self.min_chunk = min_chunk_size
        self.max_chunk = max_chunk_size
        self.coherence_threshold = coherence_threshold

        # Learned chunk compression: variable-length byte waves → single wave
        self.chunk_compress = nn.Sequential(
            nn.Linear(wave_dim, wave_dim),
            nn.GELU(),
            nn.Linear(wave_dim, wave_dim),
        )

    def find_boundaries(self, waves: torch.Tensor) -> List[int]:
        """
        Detect word boundaries using wave coherence drops.

        Args:
            waves: [seq_len, 432] CSE wave output
        Returns:
            List of boundary indices (start of each new chunk)
        """
        if waves.shape[0] <= 1:
            return [0]

        cos_sim = F.cosine_similarity(waves[:-1], waves[1:], dim=-1)

        boundaries = [0]
        pos = 0
        for i in range(len(cos_sim)):
            chunk_len = i - pos + 1
            if chunk_len >= self.min_chunk:
                if cos_sim[i] < self.coherence_threshold or chunk_len >= self.max_chunk:
                    boundaries.append(i + 1)
                    pos = i + 1

        return boundaries

    def forward(
        self, waves: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        Segment wave sequence into word-level chunks.

        Args:
            waves: [seq_len, 432] CSE wave output
        Returns:
            (chunk_waves [N, 432], spans [(start, end), ...])
        """
        boundaries = self.find_boundaries(waves)
        chunks = []
        spans = []

        for i in range(len(boundaries)):
            start = boundaries[i]
            end = boundaries[i + 1] if i + 1 < len(boundaries) else waves.shape[0]
            if start >= waves.shape[0]:
                break
            chunk = waves[start:end]  # [chunk_len, 432]

            compressed = self.chunk_compress(chunk.mean(dim=0))  # [432]
            chunks.append(compressed)
            spans.append((start, end))

        if len(chunks) == 0:
            compressed = self.chunk_compress(waves.mean(dim=0))
            chunks.append(compressed)
            spans.append((0, waves.shape[0]))

        chunk_waves = torch.stack(chunks)  # [N, 432]
        return chunk_waves, spans

    def chunk_with_bytes(
        self,
        waves: torch.Tensor,
        text_bytes: bytes,
    ) -> List[Tuple[torch.Tensor, bytes]]:
        """
        Chunk waves and return paired (chunk_wave, byte_span) for training.

        Args:
            waves: [seq_len, 432] CSE wave output
            text_bytes: Raw UTF-8 bytes of the original text
        Returns:
            List of (chunk_wave [432], byte_span) pairs
        """
        chunk_waves, spans = self.forward(waves)
        pairs = []

        for i, (start, end) in enumerate(spans):
            byte_start = min(start, len(text_bytes))
            byte_end = min(end, len(text_bytes))
            chunk_bytes = text_bytes[byte_start:byte_end]
            if len(chunk_bytes) > 0:
                pairs.append((chunk_waves[i], chunk_bytes))

        return pairs
