"""
Phase 9 — WaveGenerator v2: GRU-Based Wave Sequence Generator

Predicts the next 432-dim wave from context and previous waves using
a GRU recurrent core that carries temporal state across steps.

Key changes from v1 (MLP):
    - GRU replaces stateless MLP → carries hidden state (no more mode collapse)
    - Interference signal replaced by GRU hidden state (implicit temporal memory)
    - Dynamic field re-query at inference for semantic diversity
    - Confidence derived from GRU features (richer than wave features alone)

Generation mechanism:
    1. Project field context (512) → wave space (432)
    2. GRU processes [prev_wave + context_wave] → hidden state → next_wave
    3. Hidden state carries temporal coherence between steps
    4. At inference: RE-QUERY field at each step → dynamic context
    5. Thermodynamic confidence → when to stop generating
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
               'phase5', 'phase6', 'phase7', 'phase8']:
    _p = str(_PHASES_DIR / _phase)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from interference import apply_neighborhood_interference


class WaveGenerator(nn.Module):
    """
    GRU-based universal wave sequence generator. Predicts the next
    432-dim wave from context and previous waves using a recurrent
    core that maintains temporal state across steps.

    This module is MODALITY-AGNOSTIC. It generates waves. What those
    waves represent (text, image, audio, molecules) depends on what
    encoder created the input waves and what decoder converts the
    output waves.

    Key architectural choice: GRU hidden state replaces the explicit
    interference signal from v1. The hidden state naturally accumulates
    sequential context (what concepts were already generated, what
    themes are developing), eliminating the stateless MLP problem that
    caused mode collapse to repetitive outputs.

    Args:
        wave_dim: Wave dimension (432)
        field_features: Field feature dimension (512 for FLUXModel)
        max_waves: Maximum waves to generate (50)
        k_neighbors: Gravitational readout neighbors (16)
        interference_radius: How many previous waves influence the next (4)
        gru_hidden: GRU hidden dimension (512)
        gru_layers: Number of GRU layers (1)
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
    ):
        super().__init__()
        self.wave_dim = wave_dim
        self.field_features = field_features
        self.max_waves = max_waves
        self.k_neighbors = k_neighbors
        self.interference_radius = interference_radius
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers

        # ── Context bridge: field features → wave space ──
        self.context_to_wave = nn.Sequential(
            nn.Linear(field_features, wave_dim),
            nn.GELU(),
            nn.Linear(wave_dim, wave_dim),
        )

        # ── GRU core: carries temporal state between wave predictions ──
        # Input: [prev_wave (432) + context_wave (432)] = 864
        # Hidden state naturally replaces interference signal — accumulates
        # sequential context (themes, coherence, what was already said).
        self.wave_gru = nn.GRU(
            input_size=wave_dim * 2,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=False,
        )

        # ── GRU output → next wave prediction ──
        self.gru_to_wave = nn.Sequential(
            nn.Linear(gru_hidden, wave_dim),
            nn.Tanh(),  # Bounded output — waves are normalized
        )

        # ── Attractor selector: picks ONE attractor per step ──
        # GR returns K neighbors. This scores them. Then we SAMPLE one.
        # LLM:  softmax over vocab → sample one token
        # FLUX: gravity over attractors → sample one wave
        self.attractor_scorer = nn.Sequential(
            nn.Linear(wave_dim * 2, wave_dim),  # [current_wave + attractor] → score
            nn.GELU(),
            nn.Linear(wave_dim, 1),              # → scalar logit
        )

        # ── Confidence head: how sure is this prediction? ──
        # Fed from GRU hidden features (richer than wave alone)
        self.confidence_head = nn.Sequential(
            nn.Linear(gru_hidden, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # ── Learned start-of-sequence wave ──
        self.bos_wave = nn.Parameter(torch.randn(wave_dim) * 0.01)

    # ─────────────────────────────────────────────
    # Hidden State Management
    # ─────────────────────────────────────────────

    def init_hidden(self, device: Optional[str] = None) -> torch.Tensor:
        """
        Initialize GRU hidden state to zeros.

        Args:
            device: Target device (defaults to bos_wave's device)

        Returns:
            [gru_layers, 1, gru_hidden] zero tensor
        """
        if device is None:
            device = self.bos_wave.device
        return torch.zeros(
            self.gru_layers, 1, self.gru_hidden, device=device
        )

    # ─────────────────────────────────────────────
    # Interference Signal (Utility — for inference enrichment)
    # ─────────────────────────────────────────────

    def compute_interference_signal(
        self, generated_waves: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute interference delta from recently generated waves.

        Returns the DELTA (difference) caused by wave interference,
        not the full interfered wave. This isolates the coherence
        signal from the wave identity.

        Note: In v2, this is a utility method. The GRU hidden state
        is the primary mechanism for temporal coherence. Interference
        can optionally enrich inference but is not used during training.

        Args:
            generated_waves: List of [432] previously generated waves

        Returns:
            [432] interference delta for the next position
        """
        if len(generated_waves) == 0:
            return torch.zeros(self.wave_dim, device=self.bos_wave.device)

        # Stack recent waves (up to interference_radius)
        recent = generated_waves[-self.interference_radius:]
        stacked = torch.stack(recent)  # [R, 432]

        # Apply neighborhood interference
        interfered = apply_neighborhood_interference(
            stacked, radius=max(len(recent) - 1, 1), scale=0.1
        )

        # Return DELTA: how interference changed the last wave
        # (v1 returned interfered[-1] which contaminated the signal)
        return interfered[-1] - stacked[-1]

    # ─────────────────────────────────────────────
    # Single Step Prediction
    # ─────────────────────────────────────────────

    def forward_step(
        self,
        prev_wave: torch.Tensor,
        context_wave: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:
        """
        Predict the next wave using GRU.

        The GRU hidden state carries temporal context from all
        previous steps — this is what prevents mode collapse.

        Args:
            prev_wave: [432] the most recent generated wave
            context_wave: [432] field context projected to wave space
            hidden: [gru_layers, 1, gru_hidden] GRU hidden state

        Returns:
            (next_wave [432], confidence float, new_hidden [layers, 1, hidden])
        """
        # GRU input: [prev_wave + context_wave] → [1, 1, 864]
        gru_input = torch.cat([prev_wave, context_wave]).unsqueeze(0).unsqueeze(0)

        # GRU step — hidden state carries temporal memory
        gru_out, new_hidden = self.wave_gru(gru_input, hidden)
        gru_feat = gru_out.squeeze(0).squeeze(0)  # [gru_hidden]

        # Project GRU features → next wave
        next_wave = self.gru_to_wave(gru_feat)  # [432]

        # Confidence from GRU features (richer than wave alone)
        confidence = self.confidence_head(gru_feat).item()

        return next_wave, confidence, new_hidden

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
        Re-query the field and SAMPLE one attractor (not average them).

        This is the core of true generative behavior:
            LLM:  softmax(logits / T) → multinomial → one token
            FLUX: softmax(scores / T) → multinomial → one attractor wave

        Same prompt, different attractor sampled each time → different
        words, different phrasings, different ideas.

        Args:
            query_wave: [432] the most recent wave (used as field query)
            flux_model: FLUXModel instance (for field + GR access)
            temperature: Sampling temperature (higher = more diverse)

        Returns:
            [432] one sampled attractor's wave (not an average)
        """
        device = query_wave.device

        # Query the field directly with the 432-dim wave
        try:
            field_feats, sims, locs = flux_model.field.query(
                query_wave, k=self.k_neighbors
            )
        except Exception:
            # Fallback: use static context projection
            fallback_feat = flux_model.wave_to_field(query_wave.unsqueeze(0)).squeeze(0)
            return self.context_to_wave(fallback_feat)

        if field_feats is None or field_feats.shape[0] == 0:
            fallback_feat = flux_model.wave_to_field(query_wave.unsqueeze(0)).squeeze(0)
            return self.context_to_wave(fallback_feat)

        # Convert each field neighbor to wave space and score it
        attractor_waves = []
        logits = []

        for j in range(field_feats.shape[0]):
            feat = field_feats[j]  # [512]
            nw = self.context_to_wave(feat)  # [432]
            attractor_waves.append(nw)

            # Score: learned relevance + physics prior
            scorer_input = torch.cat([query_wave, nw])
            learned_score = self.attractor_scorer(scorer_input)  # [1]

            # Physics prior: similarity as proxy for mass / distance
            sim_val = sims[j].item() if sims is not None and j < len(sims) else 0.5
            physics_prior = torch.tensor(
                max(sim_val, 1e-6), device=device
            ).log()

            logits.append(learned_score.squeeze() + physics_prior)

        # Stack and sample — THIS is the generative step
        logits_t = torch.stack(logits)  # [K]
        probs = F.softmax(logits_t / max(temperature, 1e-8), dim=-1)
        idx = torch.multinomial(probs, 1).item()

        return attractor_waves[idx]

    # ─────────────────────────────────────────────
    # Full Generation Loop
    # ─────────────────────────────────────────────

    def generate(
        self,
        field_context: torch.Tensor,
        max_waves: Optional[int] = None,
        min_confidence: float = 0.1,
        target_waves: Optional[torch.Tensor] = None,
        flux_model=None,
        temperature: float = 1.0,
        scheduled_sampling_p: float = 0.0,
        skip_interference: bool = False,  # Deprecated, ignored (kept for compat)
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Generate a sequence of waves from field context.

        GRU hidden state carries temporal context across steps,
        replacing the explicit interference signal from v1.

        If flux_model is provided, re-queries the field at each step
        (dynamic context — enables semantic diversity). Otherwise falls
        back to static context (faster, used during training).

        Args:
            field_context: [512] merged field + CGN context (initial)
            max_waves: Maximum waves to generate (default: self.max_waves)
            min_confidence: Stop if confidence drops below this
            target_waves: [N, 432] teacher-forcing targets (training only)
            flux_model: FLUXModel instance for dynamic field re-query
            temperature: Sampling temperature for field re-query
            scheduled_sampling_p: Probability [0,1] of using the model's
                own prediction as prev_wave instead of ground truth.
                0.0 = pure teacher forcing, 1.0 = pure free generation.
            skip_interference: Deprecated, ignored (GRU replaces interference)

        Returns:
            (generated_waves [N, 432], confidences [N])
        """
        max_waves = max_waves or self.max_waves
        device = self.bos_wave.device
        context_wave = self.context_to_wave(field_context)  # Initial context [432]

        # Initialize GRU hidden state — this IS the temporal memory
        hidden = self.init_hidden(device)

        generated = []
        confidences = []
        prev_wave = self.bos_wave

        import random as _rand

        for i in range(max_waves):
            # Dynamic re-query: ask the field "what concepts live near here?"
            if flux_model is not None and not self.training:
                context_wave = self.query_field_attractors(
                    prev_wave, flux_model, temperature=temperature
                )

            if target_waves is not None and i < len(target_waves):
                # Teacher forcing: predict from prev, but feed ground truth
                next_wave, conf, hidden = self.forward_step(
                    prev_wave, context_wave, hidden
                )
                # Scheduled sampling: with probability p, use own prediction
                # instead of ground truth as the next prev_wave.
                # This forces the model to handle its own errors.
                if scheduled_sampling_p > 0 and _rand.random() < scheduled_sampling_p:
                    prev_wave = next_wave.detach()
                else:
                    prev_wave = target_waves[i].detach()
            else:
                next_wave, conf, hidden = self.forward_step(
                    prev_wave, context_wave, hidden
                )
                prev_wave = next_wave.detach()

            generated.append(next_wave)
            confidences.append(conf)

            # Stop if confidence drops (model is done generating)
            if target_waves is None and conf < min_confidence:
                break

        if len(generated) == 0:
            # Safety: always generate at least one wave
            next_wave, conf, hidden = self.forward_step(
                self.bos_wave,
                context_wave,
                hidden,
            )
            generated.append(next_wave)
            confidences.append(conf)

        return torch.stack(generated), confidences

    # ─────────────────────────────────────────────
    # Training Forward Pass
    # ─────────────────────────────────────────────

    def forward(
        self,
        field_context: torch.Tensor,
        target_waves: torch.Tensor,
        scheduled_sampling_p: float = 0.0,
        skip_interference: bool = True,  # Deprecated, ignored (kept for compat)
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Teacher-forced training forward pass (static context — no re-query).

        GRU hidden state carries temporal context — no need for
        explicit interference computation during training.

        During training we use static context for stable gradients.
        During inference, pass flux_model to generate() for dynamic re-query.

        Args:
            field_context: [512] merged context
            target_waves: [N, 432] ground truth wave sequence
            scheduled_sampling_p: Probability of using own prediction
                as prev_wave instead of ground truth (exposure bias fix)
            skip_interference: Deprecated, ignored (GRU handles temporal state)

        Returns:
            (predicted_waves [N, 432], confidences [N])
        """
        return self.generate(
            field_context,
            max_waves=len(target_waves),
            target_waves=target_waves,
            flux_model=None,  # Static context during training
            scheduled_sampling_p=scheduled_sampling_p,
        )
