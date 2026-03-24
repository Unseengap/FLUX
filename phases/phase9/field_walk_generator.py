"""
Phase 9 — FieldWalkGenerator: FLUX-Native Wave Sequence Generator

Generates waves by WALKING the resonance field — selecting and blending
existing attractors using the full FLUX physics stack. No GRU, no MLP,
no regression. Every output wave is a weighted combination of real
field attractors.

Architecture:
    For each output wave position:
    1. GR queries field for k attractors near current position (gravity)
    2. CGN scores causal compatibility for each candidate (causality)
    3. Interference coherence with recent output history (coherence)
    4. Transition scorer combines all signals → soft weights (tiny, learned)
    5. Output wave = weighted blend of attractor waves (always valid)
    6. Move current position to the blended result → walk continues

Why this works:
    - No regression in 432-dim space — selecting from existing attractors
    - No autoregressive error compounding — each step is a fresh field query
    - No training data bottleneck — the field has all the knowledge
    - Uses ALL 7 phases: CSE, Field, GR, CGN, Interference, Memory
    - Transition scorer is ~100 parameters, trains on classification not regression
    - Output is always in the span of real attractors → can't collapse to garbage

The ONLY learned component: TransitionScorer (~50 trainable params)
    score = w1*gravity + w2*causal_score + w3*interference_coherence + w4*memory_signal
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


# ─────────────────────────────────────────────
# Transition Scorer: The ONLY learned component
# ─────────────────────────────────────────────

class TransitionScorer(nn.Module):
    """
    Tiny learned component that combines physics signals into
    attractor selection probabilities.

    Combines 4 signals from existing FLUX components:
        1. Gravitational pull (from GR — mass / distance²)
        2. Causal compatibility (from CGN — does this follow logically?)
        3. Interference coherence (from Phase 1 — consistent with recent history?)
        4. Novelty bonus (prevents repeating the same attractor)

    This is ~50 trainable parameters. The physics does the heavy lifting.

    Args:
        temperature_init: Initial softmax temperature for selection
    """

    def __init__(self, temperature_init: float = 1.0):
        super().__init__()
        # Learned weights for combining physics signals
        # [gravity, causal, interference, novelty]
        self.signal_weights = nn.Parameter(torch.tensor([1.0, 0.8, 0.5, 0.3]))

        # Learned temperature for softmax selection sharpness
        self.temperature = nn.Parameter(torch.tensor(temperature_init))

        # Tiny bias per signal (allows shifting baselines)
        self.signal_bias = nn.Parameter(torch.zeros(4))

    def forward(
        self,
        gravity_scores: torch.Tensor,
        causal_scores: torch.Tensor,
        interference_scores: torch.Tensor,
        novelty_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine physics signals into attractor selection probabilities.

        All inputs are [k] tensors (one score per candidate attractor).

        Args:
            gravity_scores: [k] gravitational pull toward each attractor
            causal_scores: [k] causal compatibility with current context
            interference_scores: [k] coherence with recent output history
            novelty_scores: [k] penalty for recently-visited attractors

        Returns:
            [k] soft selection probabilities (sum to 1)
        """
        # Stack signals: [4, k]
        signals = torch.stack([
            gravity_scores,
            causal_scores,
            interference_scores,
            novelty_scores,
        ], dim=0)

        # Weight and bias each signal
        weights = F.softplus(self.signal_weights)  # Keep positive
        biased = signals + self.signal_bias.unsqueeze(1)
        combined = (weights.unsqueeze(1) * biased).sum(dim=0)  # [k]

        # Temperature-scaled softmax → selection probabilities
        temp = self.temperature.clamp(min=0.01)
        probs = F.softmax(combined / temp, dim=-1)

        return probs


# ─────────────────────────────────────────────
# Field Walk Generator
# ─────────────────────────────────────────────

class FieldWalkGenerator(nn.Module):
    """
    FLUX-native wave sequence generator. Generates waves by walking
    the resonance field, blending existing attractors using the full
    physics stack from Phases 1-7.

    No GRU. No MLP. No regression.
    Every output is a weighted blend of real field attractors.

    Args:
        wave_dim: Wave dimension (432)
        field_features: Field feature dimension (768 for FLUXLarge)
        k_neighbors: How many field attractors to consider per step (8)
        interference_radius: Recent wave history window for coherence (4)
        max_waves: Maximum waves to generate (50)
    """

    def __init__(
        self,
        wave_dim: int = 432,
        field_features: int = 768,
        k_neighbors: int = 8,
        interference_radius: int = 4,
        max_waves: int = 50,
    ):
        super().__init__()
        self.wave_dim = wave_dim
        self.field_features = field_features
        self.k_neighbors = k_neighbors
        self.interference_radius = interference_radius
        self.max_waves = max_waves

        # The ONLY learned component: transition scorer (~50 params)
        self.scorer = TransitionScorer(temperature_init=1.0)

        # Confidence head: determines when to stop generating
        # Uses interference energy as signal (low energy = stable = done)
        self.confidence_scale = nn.Parameter(torch.tensor(1.0))
        self.confidence_bias = nn.Parameter(torch.tensor(0.5))

        # Learned start-of-sequence wave
        self.bos_wave = nn.Parameter(torch.randn(wave_dim) * 0.01)

    # ─────────────────────────────────────────────
    # Physics Signal Computation
    # ─────────────────────────────────────────────

    def _compute_gravity_scores(
        self,
        similarities: torch.Tensor,
        masses: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Gravitational relevance scores: mass / distance².

        Attractors with high mass (lots of evidence) and high similarity
        (close in semantic space) exert the strongest pull.

        Args:
            similarities: [k] cosine similarities from field.query
            masses: [k] evidence masses (optional — use similarities as proxy)

        Returns:
            [k] gravity scores (higher = stronger pull)
        """
        # Distance = 1 - similarity (cosine distance)
        distances = (1.0 - similarities).clamp(min=1e-6)

        if masses is not None:
            gravity = masses / (distances ** 2)
        else:
            # Use similarity itself as mass proxy (more similar = more mass)
            gravity = similarities.clamp(min=0) / (distances ** 2)

        return gravity

    def _compute_causal_scores(
        self,
        candidate_features: torch.Tensor,
        context_feature: torch.Tensor,
        cgn,
    ) -> torch.Tensor:
        """
        Causal compatibility: which attractor FOLLOWS the current context?

        CGN bends the context signal through geometric curvature.
        Candidates that align with the bent signal are causally compatible.

        Args:
            candidate_features: [k, field_features] attractor feature vectors
            context_feature: [field_features] current context
            cgn: CausalGeometryNode instance

        Returns:
            [k] causal compatibility scores
        """
        # CGN bends the context through causal geometry
        bent_context = cgn(context_feature)  # [field_features]

        # Score: cosine similarity between bent context and each candidate
        scores = F.cosine_similarity(
            bent_context.unsqueeze(0),
            candidate_features,
            dim=-1,
        )  # [k]

        return scores

    def _compute_interference_scores(
        self,
        candidate_waves: torch.Tensor,
        recent_waves: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Interference coherence: which candidate is consistent with recent history?

        Uses Phase 1 wave interference physics. Candidates that interfere
        constructively with recent outputs score higher (coherent).
        Candidates that interfere destructively score lower (contradictory).

        Args:
            candidate_waves: [k, 432] candidate waves in wave space
            recent_waves: List of recent output waves (up to interference_radius)

        Returns:
            [k] coherence scores
        """
        if len(recent_waves) == 0:
            return torch.zeros(candidate_waves.shape[0], device=candidate_waves.device)

        # Stack recent waves (only keep last interference_radius)
        recent = torch.stack(recent_waves[-self.interference_radius:])  # [R, 432]

        # Batched cosine similarity: [k, R] in one GPU op
        # Normalize both sets of vectors, then matmul
        cand_norm = F.normalize(candidate_waves, dim=-1)   # [k, 432]
        rec_norm = F.normalize(recent, dim=-1)              # [R, 432]
        cos_matrix = cand_norm @ rec_norm.T                 # [k, R]

        # Recency weights: more recent waves have stronger influence
        weights = torch.linspace(0.5, 1.0, recent.shape[0], device=cos_matrix.device)  # [R]
        scores = (cos_matrix * weights.unsqueeze(0)).mean(dim=-1)  # [k]

        return scores

    def _compute_novelty_scores(
        self,
        candidate_waves: torch.Tensor,
        all_generated: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Novelty bonus: penalize candidates too similar to already-generated waves.

        Prevents the walk from looping back to the same attractor repeatedly.
        This is the FLUX-native solution to mode collapse.

        Args:
            candidate_waves: [k, 432] candidate waves
            all_generated: List of all previously generated waves

        Returns:
            [k] novelty scores (higher = more novel)
        """
        if len(all_generated) == 0:
            return torch.ones(candidate_waves.shape[0], device=candidate_waves.device)

        # Only check novelty against recent history (capped window)
        # Full history is O(N²) and unnecessary — local repetition is the real problem
        _NOVELTY_WINDOW = 15
        recent_gen = all_generated[-_NOVELTY_WINDOW:]
        generated = torch.stack(recent_gen)  # [W, 432] where W <= 15

        # Batched cosine similarity: [k, W] in one GPU op
        cand_norm = F.normalize(candidate_waves, dim=-1)  # [k, 432]
        gen_norm = F.normalize(generated, dim=-1)          # [W, 432]
        cos_matrix = cand_norm @ gen_norm.T                # [k, W]

        # Novelty = 1 - max similarity to any recent generated wave
        max_sims = cos_matrix.max(dim=-1).values            # [k]
        novelty = 1.0 - max_sims

        return novelty

    # ─────────────────────────────────────────────
    # Confidence Estimation
    # ─────────────────────────────────────────────

    def _compute_confidence(
        self,
        selection_probs: torch.Tensor,
        gravity_scores: torch.Tensor,
    ) -> float:
        """
        Estimate generation confidence from selection entropy and gravity.

        Low entropy (confident selection) + high gravity (strong attractors)
        = high confidence. When the field has no strong nearby attractors,
        confidence drops and generation should stop.

        Args:
            selection_probs: [k] selection probabilities
            gravity_scores: [k] gravitational pull scores

        Returns:
            Confidence in [0, 1]
        """
        # Entropy of selection (low = confident, high = uncertain)
        entropy = -(selection_probs * (selection_probs + 1e-8).log()).sum()
        max_entropy = torch.tensor(self.k_neighbors, device=entropy.device).float().log()
        normalized_entropy = entropy / max_entropy.clamp(min=1e-8)

        # Mean gravity (strong attractors = confident)
        mean_gravity = gravity_scores.mean()

        # Combine: high gravity, low entropy = high confidence
        raw = (1.0 - normalized_entropy) * torch.sigmoid(mean_gravity)
        confidence = torch.sigmoid(
            self.confidence_scale * raw + self.confidence_bias
        ).item()

        return confidence

    # ─────────────────────────────────────────────
    # Single Step: Walk One Position
    # ─────────────────────────────────────────────

    def walk_step(
        self,
        current_wave: torch.Tensor,
        recent_waves: List[torch.Tensor],
        all_generated: List[torch.Tensor],
        flux_model,
    ) -> Tuple[torch.Tensor, float]:
        """
        Walk one step through the field.

        1. Query field for nearby attractors (GR gravity)
        2. Score each by causal compatibility (CGN)
        3. Score each by interference coherence (Phase 1)
        4. Score each by novelty (anti-repetition)
        5. Blend signals → selection probabilities
        6. Output wave = weighted blend of attractor waves

        Args:
            current_wave: [432] current position in wave space
            recent_waves: Last few generated waves (interference window)
            all_generated: All generated waves so far (novelty tracking)
            flux_model: FLUXLarge instance with all Phase 1-7 components

        Returns:
            (next_wave [432], confidence float)
        """
        device = current_wave.device

        # Step 1: Query field for k nearby attractors
        field_feats, sims, locs = flux_model.field.query(
            current_wave, k=self.k_neighbors
        )  # field_feats: [k, 768], sims: [k]

        k_actual = field_feats.shape[0]
        if k_actual == 0:
            # No attractors found — return BOS as fallback
            return self.bos_wave.clone(), 0.0

        # Step 2: Convert field features to wave space for blending
        # Use the trained field_to_wave bridge from Phase 7
        candidate_waves = flux_model.field_to_wave(field_feats)  # [k, 432]

        # Step 3: Compute physics signals
        # 3a. Gravity: mass-weighted similarity
        masses = None
        if locs:
            masses = torch.tensor(
                [flux_model.field.get_mass_at(loc) for loc in locs],
                device=device,
            )
        gravity_scores = self._compute_gravity_scores(sims, masses)

        # 3b. Causal compatibility via CGN
        # Use current context through GR for richer signal
        try:
            field_input = flux_model.wave_to_field(current_wave)
            gr_context = flux_model.gr(field_input.unsqueeze(0)).squeeze(0)
            causal_scores = self._compute_causal_scores(
                field_feats, gr_context, flux_model.cgn
            )
        except Exception:
            causal_scores = torch.zeros(k_actual, device=device)

        # 3c. Interference coherence with recent history
        interference_scores = self._compute_interference_scores(
            candidate_waves, recent_waves
        )

        # 3d. Novelty bonus
        novelty_scores = self._compute_novelty_scores(
            candidate_waves, all_generated
        )

        # Step 4: Score and blend via learned transition scorer
        selection_probs = self.scorer(
            gravity_scores, causal_scores,
            interference_scores, novelty_scores,
        )  # [k]

        # Step 5: Blend attractor waves using selection probabilities
        # Output wave = Σ prob_i * candidate_wave_i
        next_wave = (selection_probs.unsqueeze(-1) * candidate_waves).sum(dim=0)  # [432]

        # Step 6: Confidence estimation
        confidence = self._compute_confidence(selection_probs, gravity_scores)

        return next_wave, confidence

    # ─────────────────────────────────────────────
    # Training Forward Pass
    # ─────────────────────────────────────────────

    def forward(
        self,
        field_context: torch.Tensor,
        target_waves: torch.Tensor,
        flux_model=None,
        scheduled_sampling_p: float = 0.0,
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Training forward pass. If flux_model is provided, uses real field walks.
        Otherwise falls back to a simplified scoring using stored context.

        During training with precomputed data (flux_model=None), we use a
        lightweight forward that still exercises the scorer on precomputed
        features. The full field walk happens during inference.

        Args:
            field_context: [768] merged field context
            target_waves: [N, 432] ground truth wave sequence
            flux_model: FLUXLarge for live field queries (None during precomputed training)
            scheduled_sampling_p: Unused (kept for interface compatibility)

        Returns:
            (predicted_waves [N, 432], confidences [N])
        """
        if flux_model is not None:
            return self._forward_live(field_context, target_waves, flux_model)
        else:
            return self._forward_precomputed(field_context, target_waves)

    def _forward_precomputed(
        self,
        field_context: torch.Tensor,
        target_waves: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Precomputed training forward. Uses target waves as pseudo-candidates
        to train the scorer's signal combination weights.

        The key insight: during precomputed training, we DON'T have live
        access to the field. But we can still train the scorer by creating
        synthetic candidate sets from the target waves themselves:
        - Current target = the "correct" candidate (should score highest)
        - Other targets = "distractors" (should score lower)
        - Physics signals computed from wave similarities

        Args:
            field_context: [768] merged context
            target_waves: [N, 432] ground truth chunks

        Returns:
            (predicted_waves [N, 432], confidences [N])
        """
        device = target_waves.device

        # Cap sequence length to avoid O(N²) blowup on long documents.
        # The scorer has ~50 params — 30 steps is plenty for learning.
        _MAX_TRAIN_CHUNKS = 30
        if target_waves.shape[0] > _MAX_TRAIN_CHUNKS:
            # Random window so all positions get covered across steps
            import random as _rand
            start = _rand.randint(0, target_waves.shape[0] - _MAX_TRAIN_CHUNKS)
            target_waves = target_waves[start:start + _MAX_TRAIN_CHUNKS]

        n_waves = target_waves.shape[0]

        predicted = []
        confidences = []
        recent_waves: List[torch.Tensor] = []
        all_generated: List[torch.Tensor] = []

        current_wave = self.bos_wave

        import random as _rand
        for i in range(n_waves):
            # Build candidate set: target[i] + k-1 distractors from other targets
            k = min(self.k_neighbors, n_waves)
            # Target is always the first candidate
            candidate_indices = [i]
            # Add random other targets as distractors
            other_indices = [j for j in range(n_waves) if j != i]
            _rand.shuffle(other_indices)
            candidate_indices.extend(other_indices[:k - 1])
            # Pad if not enough candidates
            while len(candidate_indices) < k:
                candidate_indices.append(i)

            candidates = target_waves[candidate_indices]  # [k, 432]

            # Compute physics signals on the candidate set
            # Gravity: similarity to current wave position
            cos_sims = F.cosine_similarity(
                current_wave.unsqueeze(0), candidates, dim=-1
            )
            gravity = cos_sims.clamp(min=0) / ((1 - cos_sims).clamp(min=1e-6) ** 2)

            # Causal: similarity to field context (projected to wave space)
            # Use a simple proxy since we don't have live CGN
            context_as_wave = candidates.mean(dim=0)  # rough proxy
            causal = F.cosine_similarity(
                context_as_wave.unsqueeze(0), candidates, dim=-1
            )

            # Interference: coherence with recent history
            interference = self._compute_interference_scores(
                candidates, recent_waves
            )

            # Novelty: penalize repeats
            novelty = self._compute_novelty_scores(candidates, all_generated)

            # Score and blend
            probs = self.scorer(gravity, causal, interference, novelty)
            blended = (probs.unsqueeze(-1) * candidates).sum(dim=0)

            # Confidence
            conf = self._compute_confidence(probs, gravity)

            predicted.append(blended)
            confidences.append(conf)

            # Advance position
            recent_waves.append(blended.detach())
            if len(recent_waves) > self.interference_radius:
                recent_waves.pop(0)
            all_generated.append(blended.detach())

            # Use target as next position (teacher forcing)
            current_wave = target_waves[i].detach()

        return torch.stack(predicted), confidences

    def _forward_live(
        self,
        field_context: torch.Tensor,
        target_waves: torch.Tensor,
        flux_model,
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Live field-walk training. Uses actual field queries.

        Args:
            field_context: [768] merged context
            target_waves: [N, 432] ground truth
            flux_model: FLUXLarge instance

        Returns:
            (predicted_waves [N, 432], confidences [N])
        """
        predicted = []
        confidences = []
        recent_waves: List[torch.Tensor] = []
        all_generated: List[torch.Tensor] = []

        current_wave = self.bos_wave

        for i in range(len(target_waves)):
            next_wave, conf = self.walk_step(
                current_wave, recent_waves, all_generated, flux_model
            )

            predicted.append(next_wave)
            confidences.append(conf)

            recent_waves.append(next_wave.detach())
            if len(recent_waves) > self.interference_radius:
                recent_waves.pop(0)
            all_generated.append(next_wave.detach())

            # Teacher forcing: move to target for next step
            current_wave = target_waves[i].detach()

        return torch.stack(predicted), confidences

    # ─────────────────────────────────────────────
    # Inference: Full Field Walk
    # ─────────────────────────────────────────────

    def generate(
        self,
        field_context: torch.Tensor,
        max_waves: Optional[int] = None,
        min_confidence: float = 0.1,
        flux_model=None,
        temperature: float = 1.0,
        target_waves: Optional[torch.Tensor] = None,
        **kwargs,  # Accept extra kwargs for interface compat
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Generate waves by walking the resonance field.

        At each step:
            1. Query field for nearby attractors
            2. Score by gravity + causality + interference + novelty
            3. Blend selected attractors → output wave
            4. Advance current position → next step

        Args:
            field_context: [768] initial context from FLUX pipeline
            max_waves: Maximum waves to generate
            min_confidence: Stop if confidence drops below this
            flux_model: FLUXLarge instance (required for field access)
            temperature: Controls selection sharpness (higher = more diverse)
            target_waves: If provided, teacher-forced (training mode)

        Returns:
            (generated_waves [N, 432], confidences [N])
        """
        if target_waves is not None:
            return self.forward(
                field_context, target_waves,
                flux_model=flux_model,
            )

        max_waves = max_waves or self.max_waves

        if flux_model is None:
            # Without a model, we can't walk the field — return BOS
            return self.bos_wave.unsqueeze(0), [0.5]

        generated = []
        confidences = []
        recent_waves: List[torch.Tensor] = []
        all_generated: List[torch.Tensor] = []

        # Start from initial context projected to wave space
        current_wave = flux_model.field_to_wave(field_context)

        # Temporarily adjust scorer temperature for diversity
        original_temp = self.scorer.temperature.item()
        with torch.no_grad():
            self.scorer.temperature.fill_(temperature)

        try:
            for i in range(max_waves):
                next_wave, conf = self.walk_step(
                    current_wave, recent_waves, all_generated, flux_model
                )

                generated.append(next_wave)
                confidences.append(conf)

                recent_waves.append(next_wave.detach())
                if len(recent_waves) > self.interference_radius:
                    recent_waves.pop(0)
                all_generated.append(next_wave.detach())

                # Advance position to the blended wave
                current_wave = next_wave.detach()

                # Stop if confidence drops (exhausted local field region)
                if conf < min_confidence:
                    break
        finally:
            # Restore original temperature
            with torch.no_grad():
                self.scorer.temperature.fill_(original_temp)

        if len(generated) == 0:
            return self.bos_wave.unsqueeze(0), [0.5]

        return torch.stack(generated), confidences
