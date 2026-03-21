import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional

# ─────────────────────────────────────────────
# SanityDecoder v2 — wave-aware text decoder
#
# Key fix: the decoder now receives the CSE wave
# vector (the text's own fingerprint) alongside
# the GR field features. Without this, different
# inputs produce identical GR outputs (same field
# neighbors) and the decoder can only memorize
# one average pattern.
#
# Architecture:
#   1. Project CSE wave [432] → [feature_dim] via wave_proj
#   2. Prepend projected wave as "identity token" to GR features
#      → [k+1, D] (first row = input signal, rest = field context)
#   3. Attention-pool [k+1, D] → [D]
#   4. Decode [D] → [max_len, 256] byte logits
#   5. argmax → printable ASCII string
#
# Must be TRAINED with byte-level cross-entropy
# against original text before it produces output.
# ─────────────────────────────────────────────

MAX_DECODE_LEN = 128  # maximum characters the decoder can output
WAVE_DIM = 432        # CSE total wave dimension


class SanityDecoder(nn.Module):
    """Wave-aware decoder that reconstructs text from GR + CSE features."""

    def __init__(self, feature_dim: int = 512, wave_dim: int = WAVE_DIM,
                 max_len: int = MAX_DECODE_LEN, device: str = 'cuda'):
        super().__init__()
        self.feature_dim = feature_dim
        self.wave_dim = wave_dim
        self.max_len = max_len
        self.device = device

        # Project CSE wave → same dim as GR features
        self.wave_proj = nn.Sequential(
            nn.Linear(wave_dim, feature_dim),
            nn.GELU(),
        )

        # Attention pooling: [k+1, D] → [D]
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

    def _prepare_input(self, features: Tensor, wave_vec: Optional[Tensor] = None) -> Tensor:
        """Prepend projected wave vector as identity token.

        Args:
            features: [k, D] or [D] from GR output.
            wave_vec: [wave_dim] CSE mean wave (optional for backward compat).

        Returns:
            [k+1, D] if wave_vec provided, [k, D] otherwise.
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)  # [1, D]
        if wave_vec is not None:
            # Project wave → [D] and prepend as first row
            identity = self.wave_proj(wave_vec.to(features.device))  # [D]
            features = torch.cat([identity.unsqueeze(0), features], dim=0)  # [k+1, D]
        return features

    def _pool(self, features: Tensor) -> Tensor:
        """Attention-weighted mean pooling.

        Args:
            features: [k, D] tensor (possibly with wave identity prepended).

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

    def forward(self, features: Tensor, wave_vec: Optional[Tensor] = None) -> Tensor:
        """Produce byte logits from pooled GR output + wave identity.

        Args:
            features: [k, D] or [D] from GR output.
            wave_vec: [432] CSE mean wave (the text's own fingerprint).

        Returns:
            [max_len, 256] byte logits.
        """
        combined = self._prepare_input(features, wave_vec)  # [k+1, D]
        pooled = self._pool(combined)                        # [D]
        flat = self.decode_net(pooled)                       # [max_len * 256]
        logits = flat.view(self.max_len, 256)                # [max_len, 256]
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

    def decode(self, features: Tensor, wave_vec: Optional[Tensor] = None) -> str:
        """Greedy-decode features → printable ASCII string.

        Args:
            features: [k, D] GR output.
            wave_vec: [432] CSE wave mean (strongly recommended).
        """
        with torch.no_grad():
            logits = self.forward(features, wave_vec)    # [max_len, 256]
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

    def reconstruct_and_compare(self, original_text: str, features: Tensor,
                                wave_vec: Optional[Tensor] = None) -> dict:
        """Decode features and compare against original with honest metrics.

        Metrics:
            char_accuracy:  character-level exact match (position aligned)
            lcs_ratio:      longest common subsequence / max(len_orig, len_recon)
            trigram_overlap: fraction of input trigrams found in output
            recognizable:   lcs_ratio > 0.15 OR char_accuracy > 0.08
        """
        reconstructed = self.decode(features, wave_vec)
        min_len = min(len(original_text), len(reconstructed))

        # ── Char accuracy (positional) ──
        if min_len == 0:
            char_acc = 0.0
        else:
            char_acc = sum(
                a == b for a, b in zip(original_text[:min_len], reconstructed[:min_len])
            ) / max(len(original_text), 1)

        # ── Longest Common Subsequence ratio ──
        lcs_len = _lcs_length(original_text.lower(), reconstructed.lower())
        lcs_ratio = lcs_len / max(len(original_text), 1)

        # ── Trigram overlap (n-gram based) ──
        orig_trigrams = _get_trigrams(original_text.lower())
        recon_trigrams = _get_trigrams(reconstructed.lower())
        if len(orig_trigrams) == 0:
            trigram_overlap = 0.0
        else:
            trigram_overlap = len(orig_trigrams & recon_trigrams) / len(orig_trigrams)

        # ── Recognizable: requires actual content match, not just
        #    "common ASCII letters happen to appear" ──
        recognizable = (lcs_ratio > 0.15) or (char_acc > 0.08)

        return {
            'original': original_text,
            'reconstructed': reconstructed,
            'char_accuracy': char_acc,
            'lcs_ratio': lcs_ratio,
            'trigram_overlap': trigram_overlap,
            'recognizable': recognizable,
        }

    def compute_loss(self, features: Tensor, target_text: str,
                     wave_vec: Optional[Tensor] = None) -> Tensor:
        """Compute cross-entropy loss for training.

        Args:
            features: [k, D] GR output for one example.
            target_text: original text string.
            wave_vec: [432] CSE mean wave (pass this for best results).

        Returns:
            Scalar loss.
        """
        logits = self.forward(features, wave_vec)        # [max_len, 256]
        target = self.text_to_target(target_text, self.max_len).to(logits.device)

        # Mask out padding positions (target == 0)
        mask = target > 0
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        loss = F.cross_entropy(logits[mask], target[mask])
        return loss


# ─────────────────────────────────────────────
# Honest metric helpers
# ─────────────────────────────────────────────

def _lcs_length(a: str, b: str) -> int:
    """Longest common subsequence length (O(n*m) DP)."""
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    # Space-optimized: only keep two rows
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def _get_trigrams(text: str) -> set:
    """Extract character trigrams from text."""
    if len(text) < 3:
        return set()
    return {text[i:i + 3] for i in range(len(text) - 2)}


# ─────────────────────────────────────────────
# Pipeline check utility
# ─────────────────────────────────────────────

def pipeline_check(cse, field, gr, decoder, test_input, verbose=True):
    """Run full CSE → Field → GR → Decoder pipeline on one text.

    The CSE wave vector is now passed to the decoder as the
    text-specific identity signal alongside the field features.
    """
    with torch.no_grad():
        wave = cse.encode(test_input)
        wave_mean = wave.full.mean(dim=0)                      # [432]
        field_out, _, _ = field.query(wave_mean, k=8)
        gr_output = gr(field_out.unsqueeze(0)).squeeze(0)       # [k, D]
        result = decoder.reconstruct_and_compare(
            test_input, gr_output, wave_vec=wave_mean
        )

    if verbose:
        print(f"\nInput: '{test_input}'"
              f"\nReconstructed: '{result['reconstructed']}'"
              f"\nChar accuracy: {result['char_accuracy']:.1%}"
              f"  LCS ratio: {result['lcs_ratio']:.1%}"
              f"  Trigrams: {result['trigram_overlap']:.1%}"
              f"\nRecognizable: {'YES' if result['recognizable'] else 'NO'}")
    return result
