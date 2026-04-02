"""
OnlineLearner: Real-time single-sample learning without training loop.

This wraps ThermodynamicLearner with a higher-level interface for
fact learning, fact retrieval, and continuous stream processing.

Usage:
    learner = OnlineLearner(cse, field, device='cuda')
    learner.learn_fact("The capital of France is Paris")
    result = learner.query_fact("What is the capital of France?")
    # result contains the retrieved field features closest to the query
"""

import sys
import torch
import torch.nn.functional as F
from torch import Tensor
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

try:
    from phases.phase4.thermodynamic import ThermodynamicLearner, SettleResult
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from thermodynamic import ThermodynamicLearner, SettleResult


@dataclass
class FactResult:
    """Result of querying a fact from the learned field."""
    query_text: str
    top_similarity: float
    retrieved_features: Tensor
    energy_at_location: float


class OnlineLearner:
    """
    High-level interface for one-shot and streaming learning.

    Wraps CSE (text → wave) + ThermodynamicLearner (wave → field settling).
    Provides text-level API: learn_fact(text), query_fact(text).

    Key properties:
    - No training loop, no epochs, no batches
    - Each fact learned in a single settle pass
    - Learned facts persist through subsequent updates (no forgetting)
    - Temperature adapts automatically to surprise level
    """

    def __init__(
        self,
        cse,
        tl: ThermodynamicLearner,
        device: str = 'cpu',
    ):
        """
        Args:
            cse:    ContinuousSemanticEncoder (Phase 1, frozen)
            tl:     ThermodynamicLearner wrapping a ResonanceField
            device: device for computation
        """
        self.cse = cse
        self.tl = tl
        self.device = device
        self._learned_facts: List[str] = []

    def _text_to_wave(self, text: str) -> Tensor:
        """Encode text to a mean-pooled wave vector [wave_dim]."""
        with torch.no_grad():
            wave = self.cse.encode(text)
            wave_vec = wave.full.mean(dim=0).to(self.device)
        return wave_vec

    def learn_fact(
        self,
        text: str,
        iterations: Optional[int] = None,
    ) -> SettleResult:
        """
        Learn a single fact in one settling pass.
        No training loop. No epochs. One example → one settle → done.

        Args:
            text:       the fact to learn (e.g., "Paris is in France")
            iterations: override number of settling iterations

        Returns:
            SettleResult with energy and convergence info
        """
        wave_vec = self._text_to_wave(text)
        result = self.tl.settle_once(wave_vec, iterations=iterations)
        self._learned_facts.append(text)
        return result

    def learn_facts(
        self,
        facts: List[str],
        verbose: bool = False,
    ) -> List[SettleResult]:
        """
        Learn multiple facts as a continuous stream.
        Each fact is processed once, in order. No epochs.

        Args:
            facts:   list of text strings to learn
            verbose: print progress

        Returns:
            List of SettleResults
        """
        pairs = [(self._text_to_wave(text), None) for text in facts]
        results = self.tl.learn_stream(pairs, verbose=verbose)
        self._learned_facts.extend(facts)
        return results

    def query_fact(self, query_text: str, k: int = 4) -> FactResult:
        """
        Retrieve knowledge from the field for a given query.

        Args:
            query_text: the question or probe text
            k:          number of neighbors to retrieve

        Returns:
            FactResult with similarity and retrieved features
        """
        wave_vec = self._text_to_wave(query_text)
        features, sims = self.tl.retrieve(wave_vec, k=k)

        # Get energy at query location
        with torch.no_grad():
            location = self.tl.field.wave_to_field_coords(wave_vec)
            energy_val = self.tl.field.energy[location.h, location.w, location.d, 0].item()

        return FactResult(
            query_text=query_text,
            top_similarity=sims[0].item(),
            retrieved_features=features,
            energy_at_location=energy_val,
        )

    def test_retention(
        self,
        fact_text: str,
        distractor_texts: List[str],
        n_distractors: int = 100,
    ) -> Dict[str, Any]:
        """
        Test if a learned fact survives N subsequent updates.
        This is the key acceptance criterion: retention after 100+ updates.

        Args:
            fact_text:         the fact to test retention of
            distractor_texts:  other texts to feed (noise)
            n_distractors:     how many distractors to feed after the fact

        Returns:
            Dict with retention metrics
        """
        # Learn the fact
        wave_before = self._text_to_wave(fact_text)
        with torch.no_grad():
            target_feature = self.tl.field.wave_to_feature(wave_before).detach()

        learn_result = self.learn_fact(fact_text)

        # Check retrieval immediately after learning
        features_after, sims_after = self.tl.retrieve(wave_before, k=1)
        sim_immediately = F.cosine_similarity(
            features_after[0].unsqueeze(0),
            target_feature.unsqueeze(0),
        ).item()

        # Feed N distractors
        for i in range(min(n_distractors, len(distractor_texts))):
            distractor_wave = self._text_to_wave(distractor_texts[i])
            self.tl.settle_once(distractor_wave)

        # Check retrieval after distractors
        features_final, sims_final = self.tl.retrieve(wave_before, k=1)
        sim_after_distractors = F.cosine_similarity(
            features_final[0].unsqueeze(0),
            target_feature.unsqueeze(0),
        ).item()

        retained = sim_after_distractors > 0.5  # Threshold for retention

        return {
            'fact': fact_text,
            'n_distractors': min(n_distractors, len(distractor_texts)),
            'sim_immediately': sim_immediately,
            'sim_after_distractors': sim_after_distractors,
            'similarity_drop': sim_immediately - sim_after_distractors,
            'retained': retained,
            'learn_energy': learn_result.final_energy,
            'temperature_at_learn': learn_result.temperature,
        }

    def compare_to_sgd(
        self,
        facts: List[str],
        sgd_steps: int = 100,
    ) -> Dict[str, Any]:
        """
        Compare thermodynamic learning to SGD on the same facts.
        SGD uses the field's wave_to_feature projection for equivalent task.

        Args:
            facts:     list of facts to learn
            sgd_steps: number of SGD optimization steps per fact

        Returns:
            Dict comparing convergence and speed
        """
        import time

        # ── Thermodynamic settling ──
        t0 = time.time()
        tl_results = self.learn_facts(facts)
        tl_time = time.time() - t0
        tl_energies = [r.final_energy for r in tl_results]

        # ── SGD baseline (for comparison only) ──
        t0 = time.time()
        sgd_energies = []
        # Create a temporary copy to not pollute the real field
        for text in facts:
            wave_vec = self._text_to_wave(text)
            with torch.no_grad():
                target = self.tl.field.wave_to_feature(wave_vec).detach()
                # Simulate SGD: start from random, optimize toward target
                param = torch.randn_like(target, requires_grad=True)
            opt = torch.optim.SGD([param], lr=0.01)
            for _ in range(sgd_steps):
                loss = F.mse_loss(param, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
            sgd_energies.append(loss.item())
        sgd_time = time.time() - t0

        return {
            'n_facts': len(facts),
            'tl_time': tl_time,
            'sgd_time': sgd_time,
            'sgd_steps_per_fact': sgd_steps,
            'tl_mean_energy': sum(tl_energies) / len(tl_energies),
            'sgd_mean_energy': sum(sgd_energies) / len(sgd_energies),
            'tl_faster': tl_time < sgd_time,
            'speedup': sgd_time / max(tl_time, 1e-6),
        }

    @property
    def learned_count(self) -> int:
        """Number of facts learned so far."""
        return len(self._learned_facts)

    def stats(self) -> Dict[str, Any]:
        """Summary statistics."""
        result = {
            'learned_facts': self.learned_count,
            **self.tl.stats(),
        }
        return result
