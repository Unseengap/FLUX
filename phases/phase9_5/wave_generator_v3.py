"""
Phase 9.5 — WaveGeneratorV3: Batched GRU-Based Wave Sequence Generator

Fixes from Phase 9 WaveGenerator (v2):
    - batch_first=True GRU for efficient [batch, seq, dim] processing
    - context_projection: LayerNorm + Linear decorrelation layer
    - Dropout on GRU output (0.15) — prevents memorization
    - forward_batch(): processes entire batches in one GRU call
    - init_hidden_batch(): batched hidden state initialization
    - No more Python for-loops over sequence positions during training

Architecture (unchanged GRU core):
    context [512] → LayerNorm → projection → context_to_hidden → GRU h0
                                            → context_to_wave → context_wave [432]
    GRU input: [prev_wave (432) + context_wave (432)] = [864]
    GRU: hidden=512, layers=1, batch_first=True
    GRU output → Dropout → gru_to_wave → next_wave [432]
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# ─────────────────────────────────────────────
# Path setup for cross-phase imports
# ─────────────────────────────────────────────
_PHASES_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PHASES_DIR.parent

for _phase in ['phase1', 'phase2', 'phase3', 'phase4',
               'phase5', 'phase6', 'phase7', 'phase8', 'phase9']:
    _p = str(_PHASES_DIR / _phase)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from interference import apply_neighborhood_interference


class WaveGeneratorV3(nn.Module):
    """
    GRU-based wave sequence generator with batched processing.

    Replaces Phase 9 WaveGenerator (v2) with fixes for:
    - Context collapse (LayerNorm + decorrelation projection)
    - Batch processing ([batch, seq, 864] in one GRU call)
    - Dropout for regularization

    Args:
        wave_dim: Wave dimension (432)
        field_features: Field feature dimension (512)
        max_waves: Maximum waves to generate (50)
        k_neighbors: Gravitational readout neighbors (16)
        interference_radius: How many previous waves influence the next (4)
        gru_hidden: GRU hidden dimension (512)
        gru_layers: Number of GRU layers (1)
        dropout: Dropout rate on GRU output (0.15)
    """

    def __init__(
        self,
        wave_dim: int = 432,
        field_features: int = 512,
        max_waves: int = 50,
        k_neighbors: int = 16,
        interference_radius: int = 4,
        gru_hidden: int = 512,
        gru_layers: int = 1,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.wave_dim = wave_dim
        self.field_features = field_features
        self.max_waves = max_waves
        self.k_neighbors = k_neighbors
        self.interference_radius = interference_radius
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers

        # ── Context decorrelation: prevents context collapse ──
        # Phase 9 fed raw merged vectors directly. Avg cosine was 0.980.
        # LayerNorm + learned projection decorrelates the representations.
        self.context_norm = nn.LayerNorm(field_features)
        self.context_projection = nn.Sequential(
            nn.Linear(field_features, field_features),
            nn.GELU(),
        )

        # ── Context bridge: field features → wave space ──
        self.context_to_wave = nn.Sequential(
            nn.Linear(field_features, wave_dim),
            nn.GELU(),
            nn.Linear(wave_dim, wave_dim),
        )

        # ── GRU core: batch_first=True for [batch, seq, dim] efficiency ──
        # Input: [prev_wave (432) + context_wave (432)] = 864
        self.wave_gru = nn.GRU(
            input_size=wave_dim * 2,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # ── Dropout on GRU output ──
        self.gru_dropout = nn.Dropout(dropout)

        # ── GRU output → next wave prediction ──
        self.gru_to_wave = nn.Sequential(
            nn.Linear(gru_hidden, wave_dim),
            nn.Tanh(),
        )

        # ── Attractor selector: picks ONE attractor per step (inference) ──
        self.attractor_scorer = nn.Sequential(
            nn.Linear(wave_dim * 2, wave_dim),
            nn.GELU(),
            nn.Linear(wave_dim, 1),
        )

        # ── Confidence head ──
        self.confidence_head = nn.Sequential(
            nn.Linear(gru_hidden, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # ── Context → GRU hidden state (CRITICAL for context sensitivity) ──
        self.context_to_hidden = nn.Sequential(
            nn.Linear(field_features, gru_hidden),
            nn.Tanh(),
        )

        # ── Learned start-of-sequence wave ──
        self.bos_wave = nn.Parameter(torch.randn(wave_dim) * 0.01)

    # ─────────────────────────────────────────────
    # Context Processing (with decorrelation)
    # ─────────────────────────────────────────────

    def process_context(self, field_context: torch.Tensor) -> torch.Tensor:
        """
        Decorrelate and normalize field context.

        L2-normalizes, then applies LayerNorm + learned projection.
        This prevents context collapse (avg cosine 0.980 → <0.85).

        Args:
            field_context: [field_features] or [batch, field_features]

        Returns:
            Processed context, same shape as input
        """
        # L2 normalize first — pushes all contexts onto unit sphere
        ctx = F.normalize(field_context, p=2, dim=-1)
        # LayerNorm + projection — decorrelates dimensions
        ctx = self.context_norm(ctx)
        ctx = self.context_projection(ctx)
        return ctx

    # ─────────────────────────────────────────────
    # Hidden State Management
    # ─────────────────────────────────────────────

    def init_hidden(
        self,
        device: Optional[str] = None,
        field_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Initialize GRU hidden state from field context (single sample).

        Args:
            device: Target device
            field_context: [field_features] context vector

        Returns:
            [gru_layers, 1, gru_hidden] hidden state
        """
        if device is None:
            device = self.bos_wave.device
        if field_context is not None:
            ctx = self.process_context(field_context)
            h = self.context_to_hidden(ctx)  # [gru_hidden]
            return h.unsqueeze(0).unsqueeze(0).expand(
                self.gru_layers, 1, self.gru_hidden
            ).contiguous()
        return torch.zeros(
            self.gru_layers, 1, self.gru_hidden, device=device
        )

    def init_hidden_batch(
        self,
        field_contexts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Initialize GRU hidden state for a batch.

        Args:
            field_contexts: [batch, field_features] context vectors

        Returns:
            [gru_layers, batch, gru_hidden] hidden state
        """
        ctx = self.process_context(field_contexts)               # [batch, field_features]
        h = self.context_to_hidden(ctx)                          # [batch, gru_hidden]
        # GRU expects [layers, batch, hidden]
        return h.unsqueeze(0).expand(
            self.gru_layers, -1, self.gru_hidden
        ).contiguous()

    # ─────────────────────────────────────────────
    # Single Step Prediction (inference)
    # ─────────────────────────────────────────────

    def forward_step(
        self,
        prev_wave: torch.Tensor,
        context_wave: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:
        """
        Predict the next wave (single step, single sample — for inference).

        Args:
            prev_wave: [432] the most recent generated wave
            context_wave: [432] field context projected to wave space
            hidden: [gru_layers, 1, gru_hidden] GRU hidden state

        Returns:
            (next_wave [432], confidence float, new_hidden [layers, 1, hidden])
        """
        gru_input = torch.cat([prev_wave, context_wave]).unsqueeze(0).unsqueeze(0)
        gru_out, new_hidden = self.wave_gru(gru_input, hidden)
        gru_feat = gru_out.squeeze(0).squeeze(0)  # [gru_hidden]
        gru_feat = self.gru_dropout(gru_feat)
        next_wave = self.gru_to_wave(gru_feat)
        confidence = self.confidence_head(gru_feat).item()
        return next_wave, confidence, new_hidden

    # ─────────────────────────────────────────────
    # Batched Forward (training — the main fix)
    # ─────────────────────────────────────────────

    def forward_batch(
        self,
        field_contexts: torch.Tensor,
        target_waves: torch.Tensor,
        lengths: torch.Tensor,
        scheduled_sampling_p: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched teacher-forced forward pass. Processes entire batch in
        ONE GRU call instead of Python for-loops.

        Two modes controlled by scheduled_sampling_p:
        - Pure teacher forced (ss_p=0): Build input from ground truth,
          run entire sequence through GRU at once. Maximum efficiency.
        - With scheduled sampling (ss_p>0): Step-by-step with random
          substitution of model predictions for ground truth inputs.

        Args:
            field_contexts: [batch, field_features] merged contexts
            target_waves: [batch, max_seq_len, wave_dim] padded targets
            lengths: [batch] actual sequence lengths (unpadded)
            scheduled_sampling_p: Probability of using own prediction

        Returns:
            (predicted_waves [batch, max_seq_len, wave_dim],
             confidences [batch, max_seq_len])
        """
        batch_size = field_contexts.shape[0]
        max_seq = target_waves.shape[1]
        device = field_contexts.device

        # Process context through decorrelation layer
        ctx = self.process_context(field_contexts)              # [batch, field_features]
        context_waves = self.context_to_wave(ctx)               # [batch, wave_dim]
        hidden = self.init_hidden_batch(field_contexts)         # [layers, batch, hidden]

        # Expand context_waves to match sequence length
        # [batch, wave_dim] → [batch, max_seq, wave_dim]
        ctx_expanded = context_waves.unsqueeze(1).expand(-1, max_seq, -1)

        if scheduled_sampling_p < 0.01:
            # ── FAST PATH: pure teacher forcing ──
            # Build entire input sequence at once: [BOS, target_0, ..., target_{N-2}]
            bos = self.bos_wave.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
            prev_waves = torch.cat([bos, target_waves[:, :-1, :]], dim=1)  # [batch, max_seq, wave_dim]

            # Concatenate prev_waves + context → GRU input [batch, max_seq, 864]
            gru_input = torch.cat([prev_waves, ctx_expanded], dim=-1)

            # Single GRU call for entire batch and sequence
            gru_out, _ = self.wave_gru(gru_input, hidden)       # [batch, max_seq, gru_hidden]
            gru_out = self.gru_dropout(gru_out)

            predicted = self.gru_to_wave(gru_out)                # [batch, max_seq, wave_dim]
            confidences = self.confidence_head(gru_out).squeeze(-1)  # [batch, max_seq]

        else:
            # ── SCHEDULED SAMPLING PATH: step-by-step with random substitution ──
            predicted_list = []
            conf_list = []
            prev_wave = self.bos_wave.unsqueeze(0).expand(batch_size, -1)  # [batch, wave_dim]

            for t in range(max_seq):
                gru_input = torch.cat(
                    [prev_wave, context_waves], dim=-1
                ).unsqueeze(1)  # [batch, 1, 864]

                gru_out, hidden = self.wave_gru(gru_input, hidden)
                gru_feat = gru_out.squeeze(1)       # [batch, gru_hidden]
                gru_feat = self.gru_dropout(gru_feat)

                next_wave = self.gru_to_wave(gru_feat)   # [batch, wave_dim]
                conf = self.confidence_head(gru_feat).squeeze(-1)  # [batch]

                predicted_list.append(next_wave)
                conf_list.append(conf)

                # Scheduled sampling: randomly pick own prediction vs ground truth
                if t < max_seq - 1:
                    use_own = torch.rand(batch_size, device=device) < scheduled_sampling_p
                    use_own = use_own.unsqueeze(-1).expand(-1, self.wave_dim)
                    prev_wave = torch.where(
                        use_own, next_wave.detach(), target_waves[:, t, :]
                    )

            predicted = torch.stack(predicted_list, dim=1)     # [batch, max_seq, wave_dim]
            confidences = torch.stack(conf_list, dim=1)        # [batch, max_seq]

        return predicted, confidences

    # ─────────────────────────────────────────────
    # Dynamic Field Re-Query (Inference Only)
    # ─────────────────────────────────────────────

    def query_field_attractors(
        self,
        query_wave: torch.Tensor,
        flux_model,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Re-query the field and SAMPLE one attractor.

        Args:
            query_wave: [432] the most recent wave
            flux_model: FLUXModel instance
            temperature: Sampling temperature

        Returns:
            [432] one sampled attractor's wave
        """
        device = query_wave.device

        try:
            field_feats, sims, locs = flux_model.field.query(
                query_wave, k=self.k_neighbors
            )
        except Exception:
            fallback_feat = flux_model.wave_to_field(query_wave.unsqueeze(0)).squeeze(0)
            return self.context_to_wave(self.process_context(fallback_feat))

        if field_feats is None or field_feats.shape[0] == 0:
            fallback_feat = flux_model.wave_to_field(query_wave.unsqueeze(0)).squeeze(0)
            return self.context_to_wave(self.process_context(fallback_feat))

        attractor_waves = []
        logits = []

        for j in range(field_feats.shape[0]):
            feat = field_feats[j]
            nw = self.context_to_wave(self.process_context(feat))
            attractor_waves.append(nw)

            scorer_input = torch.cat([query_wave, nw])
            learned_score = self.attractor_scorer(scorer_input)

            sim_val = sims[j].item() if sims is not None and j < len(sims) else 0.5
            physics_prior = torch.tensor(
                max(sim_val, 1e-6), device=device
            ).log()

            logits.append(learned_score.squeeze() + physics_prior)

        logits_t = torch.stack(logits)
        probs = F.softmax(logits_t / max(temperature, 1e-8), dim=-1)
        idx = torch.multinomial(probs, 1).item()

        return attractor_waves[idx]

    # ─────────────────────────────────────────────
    # Full Generation Loop (Inference)
    # ─────────────────────────────────────────────

    def generate(
        self,
        field_context: torch.Tensor,
        max_waves: Optional[int] = None,
        min_confidence: float = 0.1,
        flux_model=None,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Generate a sequence of waves from field context (inference).

        Args:
            field_context: [512] merged field + CGN context
            max_waves: Maximum waves to generate
            min_confidence: Stop if confidence drops below this
            flux_model: FLUXModel for dynamic re-query (optional)
            temperature: Sampling temperature

        Returns:
            (generated_waves [N, 432], confidences [N])
        """
        max_waves = max_waves or self.max_waves
        device = self.bos_wave.device

        ctx = self.process_context(field_context)
        context_wave = self.context_to_wave(ctx)
        hidden = self.init_hidden(device, field_context=field_context)

        generated = []
        confidences = []
        prev_wave = self.bos_wave

        for i in range(max_waves):
            if flux_model is not None and not self.training:
                context_wave = self.query_field_attractors(
                    prev_wave, flux_model, temperature=temperature
                )

            next_wave, conf, hidden = self.forward_step(
                prev_wave, context_wave, hidden
            )

            generated.append(next_wave)
            confidences.append(conf)
            prev_wave = next_wave.detach()

            if conf < min_confidence:
                break

        if len(generated) == 0:
            next_wave, conf, hidden = self.forward_step(
                self.bos_wave, context_wave, hidden
            )
            generated.append(next_wave)
            confidences.append(conf)

        return torch.stack(generated), confidences

    # ─────────────────────────────────────────────
    # Training Forward Pass (compat with v2 API)
    # ─────────────────────────────────────────────

    def forward(
        self,
        field_context: torch.Tensor,
        target_waves: torch.Tensor,
        scheduled_sampling_p: float = 0.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Single-sample teacher-forced forward (backward compat with v2).

        For batched training, use forward_batch() instead.

        Args:
            field_context: [512] merged context
            target_waves: [N, 432] ground truth wave sequence
            scheduled_sampling_p: Probability of using own prediction

        Returns:
            (predicted_waves [N, 432], confidences [N])
        """
        # Wrap as batch of 1
        field_contexts = field_context.unsqueeze(0)
        targets = target_waves.unsqueeze(0)
        lengths = torch.tensor([target_waves.shape[0]], device=field_context.device)

        predicted, confs = self.forward_batch(
            field_contexts, targets, lengths,
            scheduled_sampling_p=scheduled_sampling_p,
        )

        return predicted.squeeze(0), confs.squeeze(0).tolist()
