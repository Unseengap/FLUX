"""
ContinuousSemanticEncoder: The complete CSE module.

Architecture:
    Raw bytes
        → Sliding window (window=8, stride=1)
        → Per-dimension convolutional filters
        → Projection to wave space (432-dim)
        → Neighborhood interference pass
        → Normalized output SemanticWave

No vocabulary. No tokenizer. Any bytes accepted.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Local imports — same directory
sys.path.insert(0, str(Path(__file__).parent))
from wave_types import SemanticWave, WAVE_DIMS, TOTAL_WAVE_DIM
from interference import apply_neighborhood_interference


# ─────────────────────────────────────────────
# Convolutional Filter Bank
# ─────────────────────────────────────────────

class CSEConvBank(nn.Module):
    """
    Multi-scale convolutional filter bank for extracting byte patterns.
    Different kernel sizes capture different n-gram patterns:
        k=1  → unigram (single byte features)
        k=3  → trigram (3-byte patterns)
        k=5  → 5-gram (word fragments)
        k=7  → 7-gram (short word patterns)
    """

    def __init__(self, in_channels: int = 256, out_dim: int = 512):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, 128, kernel_size=k, padding=k // 2),
                nn.GELU(),
            )
            for k in [1, 3, 5, 7]
        ])
        self.project = nn.Sequential(
            nn.Linear(128 * 4, out_dim),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [batch, seq_len, in_channels] flattened byte embeddings
        Returns:
            [batch, seq_len, out_dim] pattern features
        """
        seq_len = x.shape[1]
        x_t = x.transpose(1, 2)  # [batch, in_channels, seq_len]

        conv_outs = []
        for conv in self.convs:
            c = conv(x_t)  # [batch, 128, seq_len']
            c = c[:, :, :seq_len]
            conv_outs.append(c)

        out = torch.cat(conv_outs, dim=1)  # [batch, 512, seq_len]
        out = out.transpose(1, 2)           # [batch, seq_len, 512]
        out = self.project(out)             # [batch, seq_len, out_dim]
        return out


# ─────────────────────────────────────────────
# Continuous Semantic Encoder
# ─────────────────────────────────────────────

class ContinuousSemanticEncoder(nn.Module):
    """
    Main CSE module. Replaces tokenizer + embedding + positional encoding.

    Input:  Raw string or bytes
    Output: SemanticWave with shape [seq_len, 432]

    Key property: DETERMINISTIC — same input always produces same wave.
    Key property: VOCABULARY-FREE — no lookup tables, pure computation.
    """

    def __init__(
        self,
        wave_dims: dict = None,
        byte_window: int = 8,
        byte_stride: int = 1,
        interference_radius: int = 4,
        device: str = 'cpu',
    ):
        super().__init__()
        if wave_dims is None:
            wave_dims = dict(WAVE_DIMS)
        self.wave_dims = wave_dims
        self.byte_window = byte_window
        self.byte_stride = byte_stride
        self.interference_radius = interference_radius
        self._device_str = device

        total_wave_dim = sum(wave_dims.values())

        # Byte embedding: maps each byte value (0–255) to a 32-dim vector
        self.byte_embed = nn.Embedding(256, 32)

        # Conv bank: multi-scale pattern extraction over sequence
        self.conv_bank = CSEConvBank(
            in_channels=byte_window * 32,
            out_dim=512,
        )

        # Separate projections for each wave dimension
        self.phonetic_proj  = nn.Linear(512, wave_dims['phonetic'])
        self.syntactic_proj = nn.Linear(512, wave_dims['syntactic'])
        self.semantic_proj  = nn.Linear(512, wave_dims['semantic'])
        self.temporal_proj  = nn.Linear(512, wave_dims['temporal'])
        self.intensity_proj = nn.Linear(512, wave_dims['intensity'])

        # Learned temporal position encoding (continuous, not discrete)
        self.temporal_encoding = nn.Parameter(
            torch.randn(1, 10000, wave_dims['temporal']) * 0.01
        )

    @property
    def device(self):
        """Get actual device from parameters."""
        return self.byte_embed.weight.device

    def text_to_bytes(self, text: str) -> Tensor:
        """Convert text to byte tensor. Any UTF-8 string accepted."""
        byte_vals = list(text.encode('utf-8'))
        return torch.tensor(byte_vals, dtype=torch.long, device=self.device)

    def bytes_to_windows(self, bytes_tensor: Tensor) -> Tensor:
        """
        Sliding window over bytes with center-aligned padding.

        Args:
            bytes_tensor: [seq_len] byte values (long tensor, 0–255)
        Returns:
            [seq_len, window_size] byte windows
        """
        seq_len = bytes_tensor.shape[0]
        pad_left = self.byte_window // 2
        pad_right = self.byte_window - 1 - pad_left
        padded = F.pad(bytes_tensor, (pad_left, pad_right), value=0)
        windows = padded.unfold(0, self.byte_window, self.byte_stride)
        return windows[:seq_len]

    def encode(self, text: str) -> SemanticWave:
        """
        Main encoding function. No tokenization. Pure continuous encoding.

        Args:
            text: Any string, any language, any content
        Returns:
            SemanticWave: continuous wave representation
        """
        bytes_tensor = self.text_to_bytes(text)
        seq_len = len(bytes_tensor)

        if seq_len == 0:
            dev = self.device
            return SemanticWave(
                phonetic=torch.zeros(0, self.wave_dims['phonetic'], device=dev),
                syntactic=torch.zeros(0, self.wave_dims['syntactic'], device=dev),
                semantic=torch.zeros(0, self.wave_dims['semantic'], device=dev),
                temporal=torch.zeros(0, self.wave_dims['temporal'], device=dev),
                intensity=torch.zeros(0, self.wave_dims['intensity'], device=dev),
            )

        windows = self.bytes_to_windows(bytes_tensor)
        embedded = self.byte_embed(windows)
        embedded_flat = embedded.reshape(1, seq_len, -1)

        features = self.conv_bank(embedded_flat).squeeze(0)

        phonetic  = torch.tanh(self.phonetic_proj(features))
        syntactic = torch.tanh(self.syntactic_proj(features))
        semantic  = torch.tanh(self.semantic_proj(features))
        intensity = torch.sigmoid(self.intensity_proj(features))

        temporal_base = torch.tanh(self.temporal_proj(features))
        if seq_len <= self.temporal_encoding.shape[1]:
            temporal_enc = self.temporal_encoding[0, :seq_len, :]
        else:
            reps = (seq_len // self.temporal_encoding.shape[1]) + 1
            temporal_enc = self.temporal_encoding[0].repeat(reps, 1)[:seq_len]
        temporal = temporal_base + temporal_enc

        wave = SemanticWave(
            phonetic=phonetic,
            syntactic=syntactic,
            semantic=semantic,
            temporal=temporal,
            intensity=intensity,
        )

        wave_full = wave.full
        wave_full = apply_neighborhood_interference(
            wave_full, radius=self.interference_radius
        )

        return self._split_wave(wave_full)

    def _split_wave(self, full_wave: Tensor) -> SemanticWave:
        """Split a full [seq_len, 432] tensor back into SemanticWave."""
        dims = self.wave_dims
        splits = [
            dims['phonetic'], dims['syntactic'], dims['semantic'],
            dims['temporal'], dims['intensity'],
        ]
        parts = torch.split(full_wave, splits, dim=-1)
        return SemanticWave(*parts)

    def forward(self, texts: list) -> list:
        """Batch encode list of strings."""
        return [self.encode(text) for text in texts]
