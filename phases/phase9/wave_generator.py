"""
Phase 9 — WaveGenerator: Universal Wave Sequence Generator

Predicts the next 432-dim wave from context and previous waves using
FLUX physics. This module is MODALITY-AGNOSTIC — it generates waves.
What those waves represent (text, image, audio, molecules) depends on
what encoder created the input waves and what decoder converts them.

Generation mechanism:
    1. RE-QUERY the field at each step using the latest wave → dynamic context
    2. GR returns a NEIGHBORHOOD of attractors → multiple valid next concepts
    3. Apply interference between previous waves → coherence signal
    4. Combine attractor neighborhood + interference → predict next wave
    5. Thermodynamic confidence → how certain is this prediction

No GRU. No attention. No Transformer components.
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
    Universal wave sequence generator. Predicts the next 432-dim wave
    from context and previous waves using FLUX physics.

    This module is MODALITY-AGNOSTIC. It generates waves. What those
    waves represent (text, image, audio, molecules) depends on what
    encoder created the input waves and what decoder converts the
    output waves.

    Args:
        wave_dim: Wave dimension (432)
        field_features: Field feature dimension (768 for FLUXLarge)
        max_waves: Maximum waves to generate (50)
        k_neighbors: Gravitational readout neighbors (16)
        interference_radius: How many previous waves influence the next (4)
    """

    def __init__(
        self,
        wave_dim: int = 432,
        field_features: int = 768,
        max_waves: int = 50,
        k_neighbors: int = 16,
        interference_radius: int = 4,
    ):
        super().__init__()
        self.wave_dim = wave_dim
        self.field_features = field_features
        self.max_waves = max_waves
        self.k_neighbors = k_neighbors
        self.interference_radius = interference_radius

        # ── Context bridge: field features → wave space ──
        self.context_to_wave = nn.Sequential(
            nn.Linear(field_features, wave_dim),
            nn.GELU(),
            nn.Linear(wave_dim, wave_dim),
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

        # ── Wave predictor: previous_wave + interference + context → next_wave ──
        # Input: [prev_wave (432) + interference_signal (432) + context (432)] = 1296
        self.wave_predictor = nn.Sequential(
            nn.Linear(wave_dim * 3, wave_dim * 2),
            nn.GELU(),
            nn.LayerNorm(wave_dim * 2),
            nn.Linear(wave_dim * 2, wave_dim),
            nn.Tanh(),  # Bounded output — waves are normalized
        )

        # ── Confidence head: how sure is this prediction? ──
        # Used by thermodynamic sampler to decide when to stop
        self.confidence_head = nn.Sequential(
            nn.Linear(wave_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # ── Learned start-of-sequence wave ──
        self.bos_wave = nn.Parameter(torch.randn(wave_dim) * 0.01)

    # ─────────────────────────────────────────────
    # Interference Signal
    # ─────────────────────────────────────────────

    def compute_interference_signal(
        self, generated_waves: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute interference from recently generated waves.

        Nearby waves in the output sequence interfere constructively
        (reinforcing coherent themes) or destructively (reducing
        contradictory signals). This is Phase 1's interference
        physics applied to the output sequence.

        Args:
            generated_waves: List of [432] previously generated waves

        Returns:
            [432] interference signal for the next position
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

        # The interference signal is the last position after interference
        return interfered[-1]

    # ─────────────────────────────────────────────
    # Single Step Prediction
    # ─────────────────────────────────────────────

    def forward_step(
        self,
        prev_wave: torch.Tensor,
        interference_signal: torch.Tensor,
        context_wave: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Predict the next wave from previous wave + interference + context.

        Args:
            prev_wave: [432] the most recent generated wave
            interference_signal: [432] coherence signal from recent waves
            context_wave: [432] field context projected to wave space

        Returns:
            (next_wave [432], confidence [0, 1])
        """
        combined = torch.cat([prev_wave, interference_signal, context_wave])
        next_wave = self.wave_predictor(combined)
        confidence = self.confidence_head(next_wave).item()
        return next_wave, confidence

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
            flux_model: FLUXLarge instance (for field + GR access)
            temperature: Sampling temperature (higher = more diverse)

        Returns:
            [432] one sampled attractor's wave (not an average)
        """
        device = query_wave.device

        # Query the field directly with the 432-dim wave
        # field.query() expects wave_dim (432) — it uses its own
        # wave_to_field_coords (432→3) and wave_to_feature (432→768)
        # internally. Do NOT project to 768 first — that's a dimension mismatch.
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
            feat = field_feats[j]  # [768]
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
        skip_interference: bool = False,
        scheduled_sampling_p: float = 0.0,
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Generate a sequence of waves from field context.

        If flux_model is provided, re-queries the field at each step
        (dynamic context — enables semantic diversity). Otherwise falls
        back to static context (faster, but less diverse).

        Args:
            field_context: [768] merged field + CGN context (initial)
            max_waves: Maximum waves to generate (default: self.max_waves)
            min_confidence: Stop if confidence drops below this
            target_waves: [N, 432] teacher-forcing targets (training only)
            flux_model: FLUXLarge instance for dynamic field re-query
            temperature: Sampling temperature for field re-query
            skip_interference: If True, zero out interference signal
                (faster training — skips O(N^2) Python loop)
            scheduled_sampling_p: Probability [0,1] of using the model's
                own prediction as prev_wave instead of ground truth.
                0.0 = pure teacher forcing, 1.0 = pure free generation.
                Ramps up during training to cure exposure bias.

        Returns:
            (generated_waves [N, 432], confidences [N])
        """
        max_waves = max_waves or self.max_waves
        context_wave = self.context_to_wave(field_context)  # Initial context

        generated = []
        confidences = []
        prev_wave = self.bos_wave
        _zero_interference = torch.zeros(self.wave_dim, device=self.bos_wave.device)

        import random as _rand

        for i in range(max_waves):
            if skip_interference:
                interference = _zero_interference
            else:
                interference = self.compute_interference_signal(generated)

            # Dynamic re-query: ask the field "what concepts live near here?"
            if flux_model is not None and not self.training:
                context_wave = self.query_field_attractors(
                    prev_wave, flux_model, temperature=temperature
                )

            if target_waves is not None and i < len(target_waves):
                # Teacher forcing: predict from prev, but feed ground truth
                next_wave, conf = self.forward_step(
                    prev_wave, interference, context_wave
                )
                # Scheduled sampling: with probability p, use own prediction
                # instead of ground truth as the next prev_wave.
                # This forces the model to handle its own errors.
                if scheduled_sampling_p > 0 and _rand.random() < scheduled_sampling_p:
                    prev_wave = next_wave.detach()
                else:
                    prev_wave = target_waves[i].detach()
            else:
                next_wave, conf = self.forward_step(
                    prev_wave, interference, context_wave
                )
                prev_wave = next_wave.detach()

            generated.append(next_wave)
            confidences.append(conf)

            # Stop if confidence drops (model is done generating)
            if target_waves is None and conf < min_confidence:
                break

        if len(generated) == 0:
            # Safety: always generate at least one wave
            next_wave, conf = self.forward_step(
                self.bos_wave,
                torch.zeros(self.wave_dim, device=self.bos_wave.device),
                context_wave,
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
        skip_interference: bool = True,
        scheduled_sampling_p: float = 0.0,
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Teacher-forced training forward pass (static context — no re-query).

        During training we use static context for stable gradients.
        During inference, pass flux_model to generate() for dynamic re-query.

        By default, interference is SKIPPED during training to avoid
        the O(N^2) Python loop bottleneck (~10s/step → ~10ms/step).
        Interference is still used during inference via generate().

        Args:
            field_context: [768] merged context
            target_waves: [N, 432] ground truth wave sequence
            skip_interference: Skip interference computation for speed
                (default True during training)
            scheduled_sampling_p: Probability of using own prediction
                as prev_wave instead of ground truth (exposure bias fix)

        Returns:
            (predicted_waves [N, 432], confidences [N])
        """
        return self.generate(
            field_context,
            max_waves=len(target_waves),
            target_waves=target_waves,
            flux_model=None,  # Static context during training
            skip_interference=skip_interference,
            scheduled_sampling_p=scheduled_sampling_p,
        )
