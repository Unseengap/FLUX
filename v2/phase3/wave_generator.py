"""
FLUX v2 Phase 3 — WaveGenerator: GRU-Based Next-Wave Predictor

Ported from phases/phase9_5/wave_generator_v3.py.

Key changes from legacy:
    - Imports from v2/phase1 and v2/phase2 (not legacy phases/)
    - Checkpoint names use _v2 suffix
    - No dependency on FLUXModel (v2 has no Phase 7 yet)
    - decode_loss is applied in train_generator.py, not here

Architecture (identical to WaveGeneratorV3):
    context [512] → LayerNorm → projection → context_to_hidden → GRU h0
                                            → context_to_wave → context_wave [432]
    GRU input: [prev_wave (432) + context_wave (432)] = [864]
    GRU: hidden=512, layers=1, batch_first=True
    GRU output → Dropout(0.15) → gru_to_wave → next_wave [432]
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# ─────────────────────────────────────────────
# Path setup — v2 imports only
# ─────────────────────────────────────────────
_V2_DIR = Path(__file__).parent.parent           # v2/
_PHASE1_DIR = _V2_DIR / 'phase1'
_PHASE2_DIR = _V2_DIR / 'phase2'
_PROJECT_ROOT = _V2_DIR.parent

for _p in [str(_PHASE1_DIR), str(_PHASE2_DIR), str(_PROJECT_ROOT), str(Path(__file__).parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from interference import apply_neighborhood_interference


class WaveGenerator(nn.Module):
    """
    GRU-based wave sequence generator with batched processing.

    Given a field context vector [512], predicts the next N waves [N, 432].
    Every predicted wave is decodable to text via WaveToText (enforced
    during training through decode_loss in train_generator.py).

    Architecture:
        context [512] → LayerNorm + projection (decorrelation)
                      → context_to_hidden → GRU h₀
                      → context_to_wave → context_wave [432]
        GRU input: cat[prev_wave [432], context_wave [432]] = [864]
        GRU → Dropout(0.15) → gru_to_wave → next_wave [432]

    Args:
        wave_dim:            Wave feature dimension (432)
        field_features:      Field feature dimension (512)
        max_waves:           Max waves to generate at inference (50)
        interference_radius: How many previous waves influence each new one (4)
        gru_hidden:          GRU hidden state size (512)
        gru_layers:          GRU layer count (1)
        dropout:             Dropout rate on GRU output (0.15)
    """

    def __init__(
        self,
        wave_dim:            int   = 432,
        field_features:      int   = 512,
        max_waves:           int   = 50,
        interference_radius: int   = 4,
        gru_hidden:          int   = 512,
        gru_layers:          int   = 1,
        dropout:             float = 0.15,
    ):
        super().__init__()
        self.wave_dim            = wave_dim
        self.field_features      = field_features
        self.max_waves           = max_waves
        self.interference_radius = interference_radius
        self.gru_hidden          = gru_hidden
        self.gru_layers          = gru_layers

        # ── Context decorrelation: prevents context collapse ──
        # Raw merged vectors have avg cosine ~0.98 — all contexts look the same.
        # LayerNorm + learned projection pushes them apart.
        self.context_norm       = nn.LayerNorm(field_features)
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

        # ── GRU core: batch_first=True ──
        # Input: cat[prev_wave (432), context_wave (432)] = 864
        self.wave_gru = nn.GRU(
            input_size=wave_dim * 2,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        self.gru_dropout = nn.Dropout(dropout)

        # ── GRU output → wave prediction ──
        self.gru_to_wave = nn.Sequential(
            nn.Linear(gru_hidden, wave_dim),
            nn.Tanh(),
        )

        # ── Confidence head ──
        self.confidence_head = nn.Sequential(
            nn.Linear(gru_hidden, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # ── Context → GRU initial hidden state ──
        self.context_to_hidden = nn.Sequential(
            nn.Linear(field_features, gru_hidden),
            nn.Tanh(),
        )

        # ── Learned start-of-sequence wave ──
        self.bos_wave = nn.Parameter(torch.randn(wave_dim) * 0.01)

    # ─────────────────────────────────────────────
    # Context Processing
    # ─────────────────────────────────────────────

    def process_context(self, field_context: torch.Tensor) -> torch.Tensor:
        """
        L2-normalize then decorrelate field context.

        Prevents context collapse: all prompts producing identical waves.

        Args:
            field_context: [field_features] or [batch, field_features]

        Returns:
            Processed context, same shape
        """
        ctx = F.normalize(field_context, p=2, dim=-1)
        ctx = self.context_norm(ctx)
        ctx = self.context_projection(ctx)
        return ctx

    # ─────────────────────────────────────────────
    # Hidden State Init
    # ─────────────────────────────────────────────

    def init_hidden(
        self,
        device: Optional[str] = None,
        field_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Initialize GRU hidden state from field context (single sample).

        Args:
            device: Target device string
            field_context: [field_features] optional context

        Returns:
            [gru_layers, 1, gru_hidden]
        """
        if device is None:
            device = self.bos_wave.device
        if field_context is not None:
            ctx = self.process_context(field_context)
            h   = self.context_to_hidden(ctx)
            return h.unsqueeze(0).unsqueeze(0).expand(
                self.gru_layers, 1, self.gru_hidden
            ).contiguous()
        return torch.zeros(self.gru_layers, 1, self.gru_hidden, device=device)

    def init_hidden_batch(self, field_contexts: torch.Tensor) -> torch.Tensor:
        """
        Initialize GRU hidden state for a batch.

        Args:
            field_contexts: [batch, field_features]

        Returns:
            [gru_layers, batch, gru_hidden]
        """
        ctx = self.process_context(field_contexts)
        h   = self.context_to_hidden(ctx)
        return h.unsqueeze(0).expand(self.gru_layers, -1, self.gru_hidden).contiguous()

    # ─────────────────────────────────────────────
    # Single Step (inference)
    # ─────────────────────────────────────────────

    def forward_step(
        self,
        prev_wave:     torch.Tensor,
        context_wave:  torch.Tensor,
        hidden:        torch.Tensor,
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:
        """
        Predict the next wave (single step, single sample).

        Args:
            prev_wave:    [wave_dim] most recent generated wave
            context_wave: [wave_dim] field context projected to wave space
            hidden:       [gru_layers, 1, gru_hidden]

        Returns:
            (next_wave [wave_dim], confidence float, new_hidden)
        """
        gru_input   = torch.cat([prev_wave, context_wave]).unsqueeze(0).unsqueeze(0)
        gru_out, new_hidden = self.wave_gru(gru_input, hidden)
        gru_feat    = self.gru_dropout(gru_out.squeeze(0).squeeze(0))
        next_wave   = self.gru_to_wave(gru_feat)
        confidence  = self.confidence_head(gru_feat).item()
        return next_wave, confidence, new_hidden

    # ─────────────────────────────────────────────
    # Batched Forward (training)
    # ─────────────────────────────────────────────

    def forward_batch(
        self,
        field_contexts:       torch.Tensor,
        target_waves:         torch.Tensor,
        lengths:              torch.Tensor,
        scheduled_sampling_p: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched teacher-forced forward pass.

        Processes entire batch in ONE GRU call (fast path) or step-by-step
        with scheduled sampling (diversity path).

        Args:
            field_contexts:       [batch, field_features]
            target_waves:         [batch, max_seq_len, wave_dim] padded targets
            lengths:              [batch] actual sequence lengths
            scheduled_sampling_p: Probability of feeding own prediction back

        Returns:
            (predicted_waves [batch, max_seq_len, wave_dim],
             confidences     [batch, max_seq_len])
        """
        batch_size = field_contexts.shape[0]
        max_seq    = target_waves.shape[1]
        device     = field_contexts.device

        ctx           = self.process_context(field_contexts)      # [batch, field_features]
        context_waves = self.context_to_wave(ctx)                 # [batch, wave_dim]
        hidden        = self.init_hidden_batch(field_contexts)    # [layers, batch, hidden]
        ctx_expanded  = context_waves.unsqueeze(1).expand(-1, max_seq, -1)

        if scheduled_sampling_p < 0.01:
            # ── Fast path: pure teacher forcing ──
            bos        = self.bos_wave.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
            prev_waves = torch.cat([bos, target_waves[:, :-1, :]], dim=1)  # [batch, max_seq, wave_dim]
            gru_input  = torch.cat([prev_waves, ctx_expanded], dim=-1)     # [batch, max_seq, 864]

            gru_out, _ = self.wave_gru(gru_input, hidden)                  # [batch, max_seq, gru_hidden]
            gru_out    = self.gru_dropout(gru_out)

            predicted   = self.gru_to_wave(gru_out)                        # [batch, max_seq, wave_dim]
            confidences = self.confidence_head(gru_out).squeeze(-1)        # [batch, max_seq]

        else:
            # ── Scheduled sampling path ──
            predicted_list, conf_list = [], []
            prev_wave = self.bos_wave.unsqueeze(0).expand(batch_size, -1)  # [batch, wave_dim]

            for t in range(max_seq):
                gru_input = torch.cat([prev_wave, context_waves], dim=-1).unsqueeze(1)
                gru_out, hidden = self.wave_gru(gru_input, hidden)
                gru_feat  = self.gru_dropout(gru_out.squeeze(1))

                next_wave = self.gru_to_wave(gru_feat)
                conf      = self.confidence_head(gru_feat).squeeze(-1)

                predicted_list.append(next_wave)
                conf_list.append(conf)

                if t < max_seq - 1:
                    use_own   = (torch.rand(batch_size, device=device) < scheduled_sampling_p).unsqueeze(-1).expand(-1, self.wave_dim)
                    prev_wave = torch.where(use_own, next_wave.detach(), target_waves[:, t, :])

            predicted   = torch.stack(predicted_list, dim=1)
            confidences = torch.stack(conf_list, dim=1)

        return predicted, confidences

    # ─────────────────────────────────────────────
    # Full Generation (inference)
    # ─────────────────────────────────────────────

    def generate(
        self,
        field_context:  torch.Tensor,
        max_waves:      Optional[int]  = None,
        min_confidence: float          = 0.1,
        temperature:    float          = 1.0,
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Generate a wave sequence from a field context vector.

        Args:
            field_context:  [field_features] context from wave_to_field(prompt)
            max_waves:      Maximum waves to generate (defaults to self.max_waves)
            min_confidence: Stop when confidence drops below this
            temperature:    Not used in GRU path (reserved for future sampling)

        Returns:
            (generated_waves [N, wave_dim], confidences [N float])
        """
        max_waves = max_waves or self.max_waves
        device    = self.bos_wave.device

        ctx          = self.process_context(field_context)
        context_wave = self.context_to_wave(ctx)
        hidden       = self.init_hidden(device, field_context=field_context)

        generated, confidences = [], []
        prev_wave = self.bos_wave

        for _ in range(max_waves):
            next_wave, conf, hidden = self.forward_step(prev_wave, context_wave, hidden)
            generated.append(next_wave)
            confidences.append(conf)
            prev_wave = next_wave.detach()
            if conf < min_confidence:
                break

        if not generated:
            next_wave, conf, hidden = self.forward_step(self.bos_wave, context_wave, hidden)
            generated.append(next_wave)
            confidences.append(conf)

        return torch.stack(generated), confidences

    # ─────────────────────────────────────────────
    # Single-sample forward (compat shim)
    # ─────────────────────────────────────────────

    def forward(
        self,
        field_context:        torch.Tensor,
        target_waves:         torch.Tensor,
        scheduled_sampling_p: float = 0.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Single-sample teacher-forced forward. Wraps forward_batch.

        Args:
            field_context: [field_features]
            target_waves:  [N, wave_dim]

        Returns:
            (predicted_waves [N, wave_dim], confidences [N])
        """
        lengths   = torch.tensor([target_waves.shape[0]], device=field_context.device)
        predicted, confs = self.forward_batch(
            field_context.unsqueeze(0),
            target_waves.unsqueeze(0),
            lengths,
            scheduled_sampling_p=scheduled_sampling_p,
        )
        return predicted.squeeze(0), confs.squeeze(0).tolist()

    # ─────────────────────────────────────────────
    # Checkpoint
    # ─────────────────────────────────────────────

    def get_config(self) -> Dict[str, Any]:
        """Return config dict for checkpoint saving."""
        return {
            'wave_dim':            self.wave_dim,
            'field_features':      self.field_features,
            'max_waves':           self.max_waves,
            'interference_radius': self.interference_radius,
            'gru_hidden':          self.gru_hidden,
            'gru_layers':          self.gru_layers,
        }
