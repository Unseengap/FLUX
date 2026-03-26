"""
Phase 8: WaveDecoder — Autoregressive Byte-Level Decoder

The FLUX pipeline provides WHAT to say (semantic meaning via field features
and wave context). The WaveDecoder handles HOW to spell it — generating
coherent bytes one at a time using a GRU conditioned on the semantic state.

Architecture:
    FLUX context:
        field_features → context_proj → GRU initial hidden state
        wave_sequence [seq, 432] → wave_proj → cross-attention K/V
    Decoding:
        GRU generates bytes autoregressively
        At each step, GRU output cross-attends to the wave sequence
        This lets the decoder focus on different input positions
        as it generates each byte — gravitational focus.

Training:
    Teacher forcing — each step receives the ACTUAL previous byte,
    predicts the next byte. Standard cross-entropy loss.

The decoder is ~5M params — the "intelligence" lives in the FLUX
field. Cross-attention gives position-aware access to input meaning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class WaveDecoder(nn.Module):
    """
    Autoregressive byte-level decoder guided by FLUX semantic waves.

    The FLUX pipeline (CSE→Field→GR→CGN→TL→Memory) provides rich
    semantic context. This decoder converts that context into a
    sequence of coherent bytes.

    Think of it as: the field knows WHAT to say, the decoder knows
    HOW to spell it. Cross-attention lets the decoder "look at"
    specific parts of the input at each generation step.

    Args:
        wave_dim: CSE wave dimension (432)
        field_features: Field feature dimension (512 for Phase 8)
        embed_dim: Byte embedding dimension
        hidden_dim: GRU hidden dimension
        num_layers: Number of GRU layers
        num_heads: Number of cross-attention heads
        vocab_size: Output vocabulary (256 for bytes)
        dropout: Dropout rate
    """

    def __init__(
        self,
        wave_dim: int = 432,
        field_features: int = 512,
        embed_dim: int = 256,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        num_heads: int = 16,
        vocab_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.wave_dim = wave_dim

        # ── Byte embedding (input at each decode step) ──
        self.byte_embed = nn.Embedding(vocab_size + 1, embed_dim)  # +1 for BOS token
        self.BOS_TOKEN = vocab_size  # 256 = start-of-sequence

        # ── Project field context → GRU initial hidden state ──
        # field_features [field_features] → hidden × layers
        self.context_proj = nn.Sequential(
            nn.Linear(field_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * num_layers),
            nn.Tanh(),
        )

        # ── Autoregressive GRU ──
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ── Cross-attention: decoder attends to full wave sequence ──
        # At each step, GRU output "looks at" specific positions in
        # the input wave — gravitational focus.
        self.wave_proj = nn.Linear(wave_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # ── Output projection → byte logits ──
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def _init_hidden(self, field_features: torch.Tensor) -> torch.Tensor:
        """
        Convert field context into GRU initial hidden state.

        Args:
            field_features: [field_features] merged field + CGN context

        Returns:
            [num_layers, 1, hidden_dim] initial hidden state
        """
        h = self.context_proj(field_features)  # [hidden_dim * num_layers]
        h = h.view(self.num_layers, 1, self.hidden_dim)  # [layers, batch=1, hidden]
        return h

    def forward(
        self,
        target_bytes: torch.Tensor,
        wave_sequence: torch.Tensor,
        field_features: torch.Tensor,
        max_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Teacher-forced training: predict each byte given previous bytes + context.

        The decoder attends to the full wave sequence at each position,
        allowing it to focus on different parts of the input meaning
        as it generates different bytes.

        Args:
            target_bytes: [seq_len] LongTensor of byte values (0-255)
            wave_sequence: [src_seq, wave_dim] full CSE wave output
            field_features: [field_features] merged field + CGN context

        Returns:
            [seq_len, 256] logits for each position
        """
        seq_len = target_bytes.shape[0]
        if max_len:
            seq_len = min(seq_len, max_len)
            target_bytes = target_bytes[:seq_len]

        device = target_bytes.device

        # Initialize GRU hidden state from field context
        hidden = self._init_hidden(field_features)

        # Build input sequence: [BOS, byte_0, byte_1, ..., byte_{n-2}]
        bos = torch.full((1,), self.BOS_TOKEN, dtype=torch.long, device=device)
        input_bytes = torch.cat([bos, target_bytes[:-1]])  # [seq_len]

        # Embed bytes
        embedded = self.byte_embed(input_bytes).unsqueeze(0)  # [1, seq, embed]

        # Run GRU
        gru_out, _ = self.gru(embedded, hidden)  # [1, seq, hidden]

        # Cross-attention: GRU output attends to full wave sequence
        wave_kv = self.wave_proj(wave_sequence).unsqueeze(0)  # [1, src_seq, hidden]
        attn_out, _ = self.cross_attn(
            query=gru_out, key=wave_kv, value=wave_kv,
        )  # [1, seq, hidden]

        # Residual connection + norm
        combined = self.attn_norm(gru_out + attn_out)  # [1, seq, hidden]
        combined = combined.squeeze(0)  # [seq, hidden]

        # Project to byte logits
        logits = self.output_proj(self.output_norm(combined))  # [seq, 256]

        return logits

    def generate_init(self, wave_sequence: torch.Tensor,
                      field_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initialize decoder state for autoregressive generation.

        Pre-computes wave key/value projections (cached for all steps).

        Args:
            wave_sequence: [src_seq, wave_dim] full wave output
            field_features: [field_features] field context

        Returns:
            (bos_logits, hidden_state, wave_kv) — logits + GRU state + cached K/V
        """
        hidden = self._init_hidden(field_features)

        # Pre-compute wave K/V (reused at every decode step)
        wave_kv = self.wave_proj(wave_sequence).unsqueeze(0)  # [1, src_seq, hidden]

        # Run BOS token through
        bos = torch.full((1,), self.BOS_TOKEN, dtype=torch.long,
                         device=wave_sequence.device)
        embedded = self.byte_embed(bos).unsqueeze(0)  # [1, 1, embed]

        gru_out, hidden = self.gru(embedded, hidden)  # [1, 1, hidden]

        # Cross-attend to wave
        attn_out, _ = self.cross_attn(query=gru_out, key=wave_kv, value=wave_kv)
        combined = self.attn_norm(gru_out + attn_out).squeeze(0).squeeze(0)
        logits = self.output_proj(self.output_norm(combined))  # [256]

        return logits, hidden, wave_kv

    def generate_step(self, byte_input: torch.Tensor,
                      hidden: torch.Tensor,
                      wave_kv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single autoregressive decode step with cross-attention.

        Args:
            byte_input: scalar or [1] byte value (0-255)
            hidden: [num_layers, 1, hidden_dim] GRU state
            wave_kv: [1, src_seq, hidden_dim] pre-computed wave K/V

        Returns:
            (logits [256], new_hidden)
        """
        embedded = self.byte_embed(byte_input.view(1, 1))  # [1, 1, embed]

        gru_out, new_hidden = self.gru(embedded, hidden)  # [1, 1, hidden]

        # Cross-attend to cached wave K/V
        attn_out, _ = self.cross_attn(query=gru_out, key=wave_kv, value=wave_kv)
        combined = self.attn_norm(gru_out + attn_out).squeeze(0).squeeze(0)
        logits = self.output_proj(self.output_norm(combined))  # [256]

        return logits, new_hidden

    def generate(
        self,
        wave_sequence: torch.Tensor,
        field_features: torch.Tensor,
        max_length: int = 100,
        temperature: float = 0.8,
        prompt_bytes: Optional[torch.Tensor] = None,
    ) -> bytes:
        """
        Full autoregressive generation with cross-attention.

        At each step, the decoder attends to the full wave sequence,
        focusing on different input positions as it generates.

        Args:
            wave_sequence: [src_seq, wave_dim] full CSE wave output
            field_features: [field_features] field context
            max_length: Max bytes to generate
            temperature: Sampling temperature
            prompt_bytes: Optional [seq] prompt bytes to seed the decoder

        Returns:
            Generated bytes
        """
        device = wave_sequence.device
        generated = []

        # Initialize (pre-computes wave K/V for all steps)
        first_logits, hidden, wave_kv = self.generate_init(wave_sequence, field_features)

        # If we have prompt bytes, feed them through first (teacher-forced prefix)
        if prompt_bytes is not None and len(prompt_bytes) > 0:
            for byte_val in prompt_bytes:
                byte_t = torch.tensor([byte_val], dtype=torch.long, device=device)
                _, hidden = self.generate_step(byte_t, hidden, wave_kv)
                generated.append(byte_val.item() if isinstance(byte_val, torch.Tensor)
                                 else byte_val)
            # Get logits for first generated byte
            last_byte = torch.tensor([prompt_bytes[-1]], dtype=torch.long, device=device)
            logits, hidden = self.generate_step(last_byte, hidden, wave_kv)
        else:
            logits = first_logits

        # Autoregressive generation
        for _ in range(max_length):
            # Temperature-scaled sampling
            scaled = logits / max(temperature, 1e-8)
            probs = F.softmax(scaled, dim=-1)
            next_byte = torch.multinomial(probs, 1).item()

            # Stop on null byte
            if next_byte == 0:
                break

            generated.append(next_byte)

            # Feed back
            byte_t = torch.tensor([next_byte], dtype=torch.long, device=device)
            logits, hidden = self.generate_step(byte_t, hidden, wave_kv)

        return bytes(generated)


# ─────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("WaveDecoder self-test (cross-attention)")
    decoder = WaveDecoder(wave_dim=432, field_features=512)
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"  Parameters: {n_params:,}")

    # Dummy inputs — wave_sequence is [src_seq, 432], NOT collapsed
    wave_seq = torch.randn(10, 432)  # 10-position wave sequence
    field_feat = torch.randn(512)
    target = torch.randint(0, 256, (50,))

    # Teacher-forced forward
    logits = decoder(target, wave_seq, field_feat)
    print(f"  Teacher-forced: target={target.shape} → logits={logits.shape}")

    # Generation
    output = decoder.generate(wave_seq, field_feat, max_length=20)
    print(f"  Generated: {len(output)} bytes → {output.decode('utf-8', errors='replace')!r}")

    print("  ✓ WaveDecoder OK (cross-attention)")
