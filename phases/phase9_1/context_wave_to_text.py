"""
Phase 9.1 — ContextWaveToText: Context-Aware Wave→Bytes Decoder

Enhanced WaveToText that uses left-context chunks to resolve
boundary ambiguity and improve spelling accuracy.

Key improvements over Phase 9 WaveToText:
    1. Left-context awareness: previous 2 chunk waves → cross-attention
    2. Larger hidden dim: 512 (up from 256)
    3. Deeper GRU: 2 layers with dropout (up from 1 layer)
    4. Larger byte embedding: 128-dim (up from 64-dim)
    5. Context dropout: randomly drops context during training (regularization)

Architecture:
    [current_wave, left_context] → context_fusion → enriched [512]
    enriched → GRU hidden init → autoregressive byte generation

Backward compatible: decode() API matches Phase 9 WaveToText.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class ContextWaveToText(nn.Module):
    """
    Context-aware wave-to-bytes decoder.

    Uses cross-attention over left-context chunk waves to resolve
    boundary ambiguity. A word like "energy" split as ["en", "ergy"]
    can only be spelled correctly if the decoder knows the previous
    chunk ended with "en".

    Args:
        wave_dim: Input wave dimension (432)
        hidden_dim: GRU hidden dimension (512)
        byte_embed_dim: Byte embedding dimension (128)
        gru_layers: Number of GRU layers (2)
        dropout: GRU and attention dropout (0.1)
        max_bytes: Maximum bytes per chunk (20)
        vocab_size: Byte vocabulary (256)
        num_context: Number of left-context chunks (2)
        context_drop_prob: Probability of dropping each context chunk during training (0.25)
    """

    def __init__(
        self,
        wave_dim: int = 432,
        hidden_dim: int = 512,
        byte_embed_dim: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.1,
        max_bytes: int = 20,
        vocab_size: int = 256,
        num_context: int = 2,
        context_drop_prob: float = 0.25,
    ):
        super().__init__()
        self.wave_dim = wave_dim
        self.hidden_dim = hidden_dim
        self.byte_embed_dim = byte_embed_dim
        self.gru_layers = gru_layers
        self.dropout_rate = dropout
        self.max_bytes = max_bytes
        self.vocab_size = vocab_size
        self.num_context = num_context
        self.context_drop_prob = context_drop_prob

        # Special token indices
        self.BOS = vocab_size       # 256
        self.EOS = vocab_size + 1   # 257

        # ─────────────────────────────────────────────
        # Context Fusion: cross-attention over left chunks
        # ─────────────────────────────────────────────

        # Project all waves (current + context) to fusion space
        self.wave_project = nn.Linear(wave_dim, hidden_dim)

        # Cross-attention: current chunk queries left-context chunks
        # Q = current wave projected, K/V = context waves projected
        self.attn_query = nn.Linear(hidden_dim, hidden_dim)
        self.attn_key = nn.Linear(hidden_dim, hidden_dim)
        self.attn_value = nn.Linear(hidden_dim, hidden_dim)
        self.attn_scale = hidden_dim ** 0.5
        self.attn_dropout = nn.Dropout(dropout)

        # Combine attention output with current wave
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.fusion_combine = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fusion_norm = nn.LayerNorm(hidden_dim)

        # ─────────────────────────────────────────────
        # GRU Decoder
        # ─────────────────────────────────────────────

        # Enriched wave → GRU initial hidden state
        self.wave_to_hidden = nn.Linear(hidden_dim, hidden_dim * gru_layers)

        # Byte embedding (256 normal bytes + BOS + EOS)
        self.byte_embed = nn.Embedding(vocab_size + 2, byte_embed_dim)

        # 2-layer GRU with dropout
        self.gru = nn.GRU(
            input_size=byte_embed_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # Output projection (256 bytes + EOS)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size + 1),
        )

        # Learnable "no context" embedding for when context is missing
        self.no_context_embed = nn.Parameter(torch.randn(1, hidden_dim) * 0.01)

    # ─────────────────────────────────────────────
    # Context Fusion
    # ─────────────────────────────────────────────

    def _fuse_context(
        self,
        current_wave: torch.Tensor,
        context_waves: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fuse current wave with left-context chunks via cross-attention.

        Args:
            current_wave: [wave_dim] or [B, wave_dim] current chunk wave
            context_waves: [N, wave_dim] or [B, N, wave_dim] left-context chunk waves
                           N can be 0, 1, or 2. None means no context.

        Returns:
            [hidden_dim] or [B, hidden_dim] context-enriched representation
        """
        # Handle both batched and unbatched input
        batched = current_wave.dim() == 2
        if not batched:
            current_wave = current_wave.unsqueeze(0)  # [1, wave_dim]
            if context_waves is not None:
                context_waves = context_waves.unsqueeze(0)  # [1, N, wave_dim]

        B = current_wave.shape[0]
        device = current_wave.device

        # Project current wave
        current_proj = self.wave_project(current_wave)  # [B, hidden_dim]

        # If no context, use learned no-context embedding
        if context_waves is None or context_waves.shape[-2] == 0:
            # Gate between current and no-context
            no_ctx = self.no_context_embed.expand(B, -1)  # [B, hidden_dim]
            gate = self.fusion_gate(torch.cat([current_proj, no_ctx], dim=-1))
            combined = gate * current_proj + (1 - gate) * no_ctx
            enriched = self.fusion_norm(self.fusion_combine(
                torch.cat([current_proj, combined], dim=-1)
            ))
            if not batched:
                enriched = enriched.squeeze(0)
            return enriched

        # Project context waves
        ctx_proj = self.wave_project(context_waves)  # [B, N, hidden_dim]
        N = ctx_proj.shape[1]

        # Cross-attention: current queries context
        Q = self.attn_query(current_proj).unsqueeze(1)   # [B, 1, hidden_dim]
        K = self.attn_key(ctx_proj)                       # [B, N, hidden_dim]
        V = self.attn_value(ctx_proj)                     # [B, N, hidden_dim]

        # Scaled dot-product attention
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / self.attn_scale  # [B, 1, N]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_out = torch.bmm(attn_weights, V).squeeze(1)  # [B, hidden_dim]

        # Gated fusion: learn how much to trust context vs current wave
        gate = self.fusion_gate(torch.cat([current_proj, attn_out], dim=-1))
        combined = gate * current_proj + (1 - gate) * attn_out
        enriched = self.fusion_norm(self.fusion_combine(
            torch.cat([current_proj, combined], dim=-1)
        ))

        if not batched:
            enriched = enriched.squeeze(0)
        return enriched

    def _maybe_drop_context(
        self,
        context_waves: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Randomly drop context chunks during training for regularization.

        Each context chunk is independently dropped with probability
        context_drop_prob. This forces the model to not rely too heavily
        on context — it must still spell reasonably without it.

        Args:
            context_waves: [N, wave_dim] or None

        Returns:
            [M, wave_dim] where M <= N, or None if all dropped
        """
        if context_waves is None or not self.training:
            return context_waves

        if context_waves.dim() == 1:
            context_waves = context_waves.unsqueeze(0)

        N = context_waves.shape[0]
        keep_mask = torch.rand(N) > self.context_drop_prob
        if not keep_mask.any():
            return None

        return context_waves[keep_mask]

    # ─────────────────────────────────────────────
    # Forward (Teacher-Forced Training)
    # ─────────────────────────────────────────────

    def forward(
        self,
        wave: torch.Tensor,
        target_bytes: torch.Tensor,
        context_waves: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Teacher-forced: predict bytes + EOS for one chunk given its wave + context.

        Args:
            wave: [432] the current chunk wave to decode
            target_bytes: [chunk_len] byte values for this chunk (0..255)
            context_waves: [N, 432] left-context chunk waves (N=0..2), or None

        Returns:
            [chunk_len+1, 257] logits (256 bytes + EOS)
        """
        # Drop context randomly during training
        ctx = self._maybe_drop_context(context_waves)

        # Fuse current wave with context
        enriched = self._fuse_context(wave, ctx)  # [hidden_dim]

        # Project to GRU initial hidden [gru_layers, 1, hidden_dim]
        h_flat = self.wave_to_hidden(enriched)  # [hidden_dim * gru_layers]
        hidden = h_flat.view(self.gru_layers, 1, self.hidden_dim)

        # Build input sequence: BOS followed by all target bytes
        bos = torch.full(
            (1,), self.BOS, dtype=torch.long, device=wave.device
        )
        input_seq = torch.cat([bos, target_bytes])  # [chunk_len + 1]
        embedded = self.byte_embed(input_seq).unsqueeze(0)  # [1, chunk_len+1, byte_embed_dim]

        # Run GRU
        output, _ = self.gru(embedded, hidden)  # [1, chunk_len+1, hidden_dim]
        logits = self.output_proj(output.squeeze(0))  # [chunk_len+1, 257]

        return logits

    def forward_batch(
        self,
        waves: torch.Tensor,
        target_list: list,
        context_list: Optional[list] = None,
    ) -> torch.Tensor:
        """
        Batched teacher-forced training over multiple chunks.

        Args:
            waves: [B, 432] batch of current chunk waves
            target_list: List of [chunk_len_i] tensors (byte values)
            context_list: List of ([N_i, 432] or None) context wave tensors.
                          If None, no context for any chunk.

        Returns:
            total cross-entropy loss (averaged across valid tokens)
        """
        device = waves.device
        batch_size = waves.shape[0]
        max_len = max(t.shape[0] for t in target_list)
        seq_len = max_len + 1  # Room for EOS

        # ─── Context fusion for each sample ───
        enriched_list = []
        for i in range(batch_size):
            ctx = None
            if context_list is not None and context_list[i] is not None:
                ctx = self._maybe_drop_context(context_list[i])
            enriched = self._fuse_context(waves[i], ctx)  # [hidden_dim]
            enriched_list.append(enriched)
        enriched_batch = torch.stack(enriched_list)  # [B, hidden_dim]

        # ─── GRU hidden init ───
        h_flat = self.wave_to_hidden(enriched_batch)  # [B, hidden_dim * gru_layers]
        # Reshape to [gru_layers, B, hidden_dim]
        hidden = h_flat.view(batch_size, self.gru_layers, self.hidden_dim)
        hidden = hidden.permute(1, 0, 2).contiguous()  # [gru_layers, B, hidden_dim]

        # ─── Pad targets and build inputs ───
        padded_targets = torch.full(
            (batch_size, seq_len), -100, dtype=torch.long, device=device
        )
        padded_inputs = torch.full(
            (batch_size, seq_len), self.BOS, dtype=torch.long, device=device
        )

        for i, t in enumerate(target_list):
            length = t.shape[0]
            padded_targets[i, :length] = t
            padded_targets[i, length] = self.vocab_size  # EOS
            if length > 0:
                padded_inputs[i, 1:length + 1] = t

        # ─── Embed and run GRU ───
        embedded = self.byte_embed(padded_inputs)  # [B, seq_len, byte_embed_dim]
        output, _ = self.gru(embedded, hidden)    # [B, seq_len, hidden_dim]
        logits = self.output_proj(output)         # [B, seq_len, 257]

        # ─── Cross-entropy loss ───
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size + 1),
            padded_targets.view(-1),
            ignore_index=-100,
        )

        return loss

    # ─────────────────────────────────────────────
    # Autoregressive Decoding
    # ─────────────────────────────────────────────

    @torch.no_grad()
    def decode(
        self,
        wave: torch.Tensor,
        temperature: float = 0.8,
        max_bytes: Optional[int] = None,
        context_waves: Optional[torch.Tensor] = None,
    ) -> bytes:
        """
        Autoregressive: generate bytes from a wave + optional context.

        Args:
            wave: [432] the wave to decode
            temperature: Sampling temperature
            max_bytes: Override max bytes per chunk
            context_waves: [N, 432] left-context chunk waves (optional)

        Returns:
            Decoded bytes
        """
        max_b = max_bytes or self.max_bytes

        # Fuse context
        enriched = self._fuse_context(wave, context_waves)  # [hidden_dim]

        # GRU hidden init
        h_flat = self.wave_to_hidden(enriched)
        hidden = h_flat.view(self.gru_layers, 1, self.hidden_dim)

        current = torch.full(
            (1,), self.BOS, dtype=torch.long, device=wave.device
        )
        result = []

        for _ in range(max_b):
            embedded = self.byte_embed(current).unsqueeze(0)  # [1, 1, byte_embed_dim]
            output, hidden = self.gru(embedded, hidden)
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
    def decode_sequence(
        self,
        chunk_waves: torch.Tensor,
        temperature: float = 0.8,
    ) -> List[bytes]:
        """
        Decode a full sequence of chunks, using previous chunks as context.

        This is the main inference entry point. Each chunk automatically
        gets its left-context from the previous chunks in the sequence.

        Args:
            chunk_waves: [N, 432] sequence of chunk waves
            temperature: Sampling temperature

        Returns:
            List of decoded byte sequences, one per chunk
        """
        results = []
        N = chunk_waves.shape[0]

        for i in range(N):
            # Build left-context from previous chunks
            ctx_start = max(0, i - self.num_context)
            if i > 0:
                context = chunk_waves[ctx_start:i]  # [0..2, 432]
            else:
                context = None

            decoded = self.decode(
                chunk_waves[i],
                temperature=temperature,
                context_waves=context,
            )
            results.append(decoded)

        return results

    @torch.no_grad()
    def decode_batch(
        self,
        waves: torch.Tensor,
        temperature: float = 0.8,
        context_waves: Optional[torch.Tensor] = None,
    ) -> list:
        """
        Decode multiple independent chunks (no inter-chunk context).

        For sequential decoding with context, use decode_sequence() instead.

        Args:
            waves: [B, 432] batch of chunk waves
            temperature: Sampling temperature
            context_waves: Optional [B, N, 432] context for each chunk

        Returns:
            List of bytes objects
        """
        results = []
        for i in range(waves.shape[0]):
            ctx = None
            if context_waves is not None and i < context_waves.shape[0]:
                ctx = context_waves[i]
            results.append(self.decode(waves[i], temperature, context_waves=ctx))
        return results
