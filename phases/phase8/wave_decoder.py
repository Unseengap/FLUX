"""
Phase 8: WaveDecoder — Autoregressive Byte-Level Decoder

The FLUX pipeline provides WHAT to say (semantic meaning via field features
and wave context). The WaveDecoder handles HOW to spell it — generating
coherent bytes one at a time using a GRU conditioned on the semantic state.

Architecture:
    FLUX context (wave_vec + field_features)
        → context_proj → GRU initial hidden state
        → GRU generates bytes autoregressively:
            input: previous byte embedding
            output: next byte logits [256]

Training:
    Teacher forcing — each step receives the ACTUAL previous byte,
    predicts the next byte. Standard cross-entropy loss on full
    byte sequences (not histograms).

The decoder is small (~4M params) — the "intelligence" lives in
the FLUX field. The decoder just needs to articulate it coherently.
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
    HOW to spell it.

    Args:
        wave_dim: CSE wave dimension (432)
        field_features: Field feature dimension (768 for FLUXLarge)
        embed_dim: Byte embedding dimension
        hidden_dim: GRU hidden dimension
        num_layers: Number of GRU layers
        vocab_size: Output vocabulary (256 for bytes)
        dropout: Dropout rate between GRU layers
    """

    def __init__(
        self,
        wave_dim: int = 432,
        field_features: int = 768,
        embed_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 2,
        vocab_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embed_dim = embed_dim

        # ── Byte embedding (input at each decode step) ──
        self.byte_embed = nn.Embedding(vocab_size + 1, embed_dim)  # +1 for BOS token
        self.BOS_TOKEN = vocab_size  # 256 = start-of-sequence

        # ── Project FLUX context → GRU initial hidden state ──
        # wave_vec [wave_dim] + field_features [field_features] → hidden × layers
        self.context_proj = nn.Sequential(
            nn.Linear(wave_dim + field_features, hidden_dim),
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

        # ── Output projection → byte logits ──
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

        # ── Optional: wave injection at every step ──
        # Allows the decoder to "look at" the semantic context at each step
        self.wave_inject = nn.Linear(wave_dim, embed_dim)

    def _init_hidden(self, wave_vec: torch.Tensor,
                     field_features: torch.Tensor) -> torch.Tensor:
        """
        Convert FLUX semantic context into GRU initial hidden state.

        Args:
            wave_vec: [wave_dim] mean semantic wave from CSE
            field_features: [field_features] from field query

        Returns:
            [num_layers, 1, hidden_dim] initial hidden state
        """
        context = torch.cat([wave_vec, field_features], dim=-1)
        h = self.context_proj(context)  # [hidden_dim * num_layers]
        h = h.view(self.num_layers, 1, self.hidden_dim)  # [layers, batch=1, hidden]
        return h

    def forward(
        self,
        target_bytes: torch.Tensor,
        wave_vec: torch.Tensor,
        field_features: torch.Tensor,
        max_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Teacher-forced training: predict each byte given previous bytes + context.

        Args:
            target_bytes: [seq_len] LongTensor of byte values (0-255)
            wave_vec: [wave_dim] semantic wave from CSE
            field_features: [field_features] from field query

        Returns:
            [seq_len, 256] logits for each position
        """
        seq_len = target_bytes.shape[0]
        if max_len:
            seq_len = min(seq_len, max_len)
            target_bytes = target_bytes[:seq_len]

        device = target_bytes.device

        # Initialize GRU hidden state from FLUX context
        hidden = self._init_hidden(wave_vec, field_features)

        # Build input sequence: [BOS, byte_0, byte_1, ..., byte_{n-2}]
        # (shifted right — predict byte_i from bytes before it)
        bos = torch.full((1,), self.BOS_TOKEN, dtype=torch.long, device=device)
        input_bytes = torch.cat([bos, target_bytes[:-1]])  # [seq_len]

        # Embed bytes
        embedded = self.byte_embed(input_bytes).unsqueeze(0)  # [1, seq, embed]

        # Add wave injection (semantic context at every step)
        wave_signal = self.wave_inject(wave_vec).unsqueeze(0).unsqueeze(0)  # [1, 1, embed]
        embedded = embedded + wave_signal  # broadcast over seq dim

        # Run GRU
        output, _ = self.gru(embedded, hidden)  # [1, seq, hidden]
        output = output.squeeze(0)  # [seq, hidden]

        # Project to byte logits
        output = self.output_norm(output)
        logits = self.output_proj(output)  # [seq, 256]

        return logits

    def generate_init(self, wave_vec: torch.Tensor,
                      field_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize decoder state for autoregressive generation.

        Args:
            wave_vec: [wave_dim] semantic wave
            field_features: [field_features] field context

        Returns:
            (bos_logits, hidden_state) — first byte logits + GRU state
        """
        hidden = self._init_hidden(wave_vec, field_features)

        # Run BOS token through
        bos = torch.full((1,), self.BOS_TOKEN, dtype=torch.long,
                         device=wave_vec.device)
        embedded = self.byte_embed(bos).unsqueeze(0)  # [1, 1, embed]

        # Add wave injection
        wave_signal = self.wave_inject(wave_vec).unsqueeze(0).unsqueeze(0)
        embedded = embedded + wave_signal

        output, hidden = self.gru(embedded, hidden)  # [1, 1, hidden]
        output = self.output_norm(output.squeeze(0).squeeze(0))
        logits = self.output_proj(output)  # [256]

        return logits, hidden

    def generate_step(self, byte_input: torch.Tensor,
                      hidden: torch.Tensor,
                      wave_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single autoregressive decode step.

        Args:
            byte_input: scalar or [1] byte value (0-255)
            hidden: [num_layers, 1, hidden_dim] GRU state
            wave_vec: [wave_dim] for wave injection

        Returns:
            (logits [256], new_hidden)
        """
        embedded = self.byte_embed(byte_input.view(1, 1))  # [1, 1, embed]

        # Wave injection
        wave_signal = self.wave_inject(wave_vec).unsqueeze(0).unsqueeze(0)
        embedded = embedded + wave_signal

        output, new_hidden = self.gru(embedded, hidden)
        output = self.output_norm(output.squeeze(0).squeeze(0))
        logits = self.output_proj(output)  # [256]

        return logits, new_hidden

    def generate(
        self,
        wave_vec: torch.Tensor,
        field_features: torch.Tensor,
        max_length: int = 100,
        temperature: float = 0.8,
        prompt_bytes: Optional[torch.Tensor] = None,
    ) -> bytes:
        """
        Full autoregressive generation.

        Args:
            wave_vec: [wave_dim] semantic context
            field_features: [field_features] field context
            max_length: Max bytes to generate
            temperature: Sampling temperature
            prompt_bytes: Optional [seq] prompt bytes to seed the decoder

        Returns:
            Generated bytes
        """
        device = wave_vec.device
        generated = []

        # Initialize from context
        first_logits, hidden = self.generate_init(wave_vec, field_features)

        # If we have prompt bytes, feed them through first (teacher-forced prefix)
        if prompt_bytes is not None and len(prompt_bytes) > 0:
            for byte_val in prompt_bytes:
                byte_t = torch.tensor([byte_val], dtype=torch.long, device=device)
                _, hidden = self.generate_step(byte_t, hidden, wave_vec)
                generated.append(byte_val.item() if isinstance(byte_val, torch.Tensor)
                                 else byte_val)
            # Get logits for first generated byte
            last_byte = torch.tensor([prompt_bytes[-1]], dtype=torch.long, device=device)
            logits, hidden = self.generate_step(last_byte, hidden, wave_vec)
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
            logits, hidden = self.generate_step(byte_t, hidden, wave_vec)

        return bytes(generated)


# ─────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("WaveDecoder self-test")
    decoder = WaveDecoder(wave_dim=432, field_features=768)
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"  Parameters: {n_params:,}")

    # Dummy inputs
    wave = torch.randn(432)
    field_feat = torch.randn(768)
    target = torch.randint(0, 256, (50,))

    # Teacher-forced forward
    logits = decoder(target, wave, field_feat)
    print(f"  Teacher-forced: target={target.shape} → logits={logits.shape}")

    # Generation
    output = decoder.generate(wave, field_feat, max_length=20)
    print(f"  Generated: {len(output)} bytes → {output.decode('utf-8', errors='replace')!r}")

    print("  ✓ WaveDecoder OK")
