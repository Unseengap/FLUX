import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional

# ─────────────────────────────────────────────
# SanityDecoder — maps GR output features back
# to a byte sequence for pipeline sanity checks.
#
# Architecture:
#   1. Pool [k, feature_dim] → [feature_dim] via attention-weighted mean
#   2. Decode [feature_dim] → [max_len, 256] byte logits
#   3. argmax → printable ASCII string
#
# Must be TRAINED with byte-level cross-entropy
# against original text before it produces output.
# ─────────────────────────────────────────────

MAX_DECODE_LEN = 128  # maximum characters the decoder can output


class SanityDecoder(nn.Module):
    """Lightweight decoder that reconstructs text from GR output features."""

    def __init__(self, feature_dim: int = 512, max_len: int = MAX_DECODE_LEN,
                 device: str = 'cuda'):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_len = max_len
        self.device = device

        # Attention pooling: [k, D] → [D]
        self.pool_attn = nn.Linear(feature_dim, 1)

        # Decoder: [D] → [max_len * 256] → [max_len, 256]
        self.decode_net = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, max_len * 256),
        )

    def _pool(self, features: Tensor) -> Tensor:
        """Attention-weighted mean pooling.

        Args:
            features: [k, D] or [D] tensor from GR output.

        Returns:
            [D] pooled representation.
        """
        if features.dim() == 1:
            return features
        # features: [k, D]
        weights = self.pool_attn(features).squeeze(-1)   # [k]
        weights = F.softmax(weights, dim=0)                # [k]
        pooled = (weights.unsqueeze(-1) * features).sum(dim=0)  # [D]
        return pooled

    def forward(self, features: Tensor) -> Tensor:
        """Produce byte logits from pooled GR output.

        Args:
            features: [k, D] or [D] from GR output.

        Returns:
            [max_len, 256] byte logits.
        """
        pooled = self._pool(features)                       # [D]
        flat = self.decode_net(pooled)                      # [max_len * 256]
        logits = flat.view(self.max_len, 256)               # [max_len, 256]
        return logits

    # ── Text helpers ───────────────────────────────────────

    @staticmethod
    def text_to_target(text: str, max_len: int = MAX_DECODE_LEN) -> Tensor:
        """Convert text to a padded byte-ID target tensor.

        Bytes outside 32-126 printable range are clamped.
        Padding uses 0 (will be masked in loss).

        Returns:
            [max_len] LongTensor of byte IDs.
        """
        raw = list(text.encode('utf-8', errors='replace'))[:max_len]
        # clamp to valid range; 0 = padding
        ids = [min(max(b, 32), 126) for b in raw]
        ids += [0] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, features: Tensor) -> str:
        """Greedy-decode features → printable ASCII string."""
        with torch.no_grad():
            logits = self.forward(features)              # [max_len, 256]
            byte_ids = logits.argmax(dim=-1).cpu().tolist()
            # Stop at first padding (0) or non-printable
            chars = []
            for b in byte_ids:
                if b == 0:
                    break
                if 32 <= b <= 126:
                    chars.append(b)
            try:
                return bytes(chars).decode('ascii', errors='replace')
            except Exception:
                return '[decode error]'

    def reconstruct_and_compare(self, original_text: str, features: Tensor) -> dict:
        """Decode features and compare against original."""
        reconstructed = self.decode(features)
        min_len = min(len(original_text), len(reconstructed))
        if min_len == 0:
            char_acc = 0.0
        else:
            char_acc = sum(
                a == b for a, b in zip(original_text[:min_len], reconstructed[:min_len])
            ) / max(len(original_text), 1)

        original_chars = set(original_text.lower())
        output_chars = set(reconstructed.lower())
        overlap = len(original_chars & output_chars) / max(len(original_chars), 1)

        return {
            'original': original_text,
            'reconstructed': reconstructed,
            'char_accuracy': char_acc,
            'char_overlap': overlap,
            'recognizable': overlap > 0.2,
        }

    def compute_loss(self, features: Tensor, target_text: str) -> Tensor:
        """Compute cross-entropy loss for training.

        Args:
            features: [k, D] GR output for one example.
            target_text: original text string.

        Returns:
            Scalar loss.
        """
        logits = self.forward(features)                  # [max_len, 256]
        target = self.text_to_target(target_text, self.max_len).to(logits.device)

        # Mask out padding positions (target == 0)
        mask = target > 0
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        loss = F.cross_entropy(logits[mask], target[mask])
        return loss


# ─────────────────────────────────────────────
# Pipeline check utility
# ─────────────────────────────────────────────

def pipeline_check(cse, field, gr, decoder, test_input, verbose=True):
    """Run full CSE → Field → GR → Decoder pipeline on one text."""
    with torch.no_grad():
        wave = cse.encode(test_input)
        wave_full_mean = wave.full.mean(dim=0)
        field_out, _, _ = field.query(wave_full_mean, k=8)
        gr_output = gr(field_out.unsqueeze(0)).squeeze(0)  # [k, D]
        result = decoder.reconstruct_and_compare(test_input, gr_output)

    if verbose:
        print(f"\nInput: '{test_input}'"
              f"\nReconstructed: '{result['reconstructed']}'"
              f"\nRecognizable: {'YES' if result['recognizable'] else 'NO'}")
    return result
