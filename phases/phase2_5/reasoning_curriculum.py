"""
ReasoningCurriculum: Teaching FLUX the physics of logic.

Two ingestion pipelines:

1. SNLI (Stanford Natural Language Inference)
   - entailment:    reinforce gravity between premise and hypothesis
   - contradiction: apply negative mass — physical repulsion
   - neutral:       create uncertainty tension in field

2. GSM8K (Grade School Math)
   - Each solution step becomes an attractor
   - Causal arrows forced between sequential steps
   - Builds the geometry of deduction into the field

We only use TRAINING splits. Test splits are untouched for evaluation.
"""

import sys
import re
import torch
import torch.nn.functional as F
from torch import Tensor
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1_5'))

from implication import ImplicationChainStore
from causal_types import CausalArrow


# ─────────────────────────────────────────────
# Fallback data for offline use
# ─────────────────────────────────────────────

FALLBACK_SNLI = [
    # (premise, hypothesis, label)
    ("A dog is running in a field.",
     "An animal is outside.",
     "entailment"),
    ("A man is sleeping on a bench.",
     "A person is awake and active.",
     "contradiction"),
    ("Children are playing in the park.",
     "Kids are outdoors.",
     "entailment"),
    ("The woman is cooking dinner.",
     "The woman is eating breakfast.",
     "contradiction"),
    ("A bird is sitting on a branch.",
     "An animal is near a tree.",
     "neutral"),
    ("The car is parked on the street.",
     "The vehicle is stationary.",
     "entailment"),
    ("It is raining heavily outside.",
     "The weather is sunny and clear.",
     "contradiction"),
    ("Two people are having a conversation.",
     "At least one person is present.",
     "entailment"),
    ("The cat is sleeping on the mat.",
     "The dog is playing outside.",
     "neutral"),
    ("A scientist is conducting an experiment.",
     "Someone is doing research.",
     "entailment"),
    ("The store is open for business.",
     "The store is closed.",
     "contradiction"),
    ("A student is studying in the library.",
     "Someone is in a building.",
     "entailment"),
    ("The chef is preparing a meal.",
     "The chef is eating the meal.",
     "neutral"),
    ("The sun is shining brightly.",
     "It is dark outside.",
     "contradiction"),
    ("A child is reading a book.",
     "A young person is engaged with text.",
     "entailment"),
    ("The train arrived at the station.",
     "The train departed from the city.",
     "neutral"),
    ("The athlete won the race.",
     "The athlete finished last.",
     "contradiction"),
    ("People are swimming in the ocean.",
     "Humans are near water.",
     "entailment"),
    ("The building is very tall.",
     "The structure has multiple floors.",
     "neutral"),
    ("The patient recovered from surgery.",
     "The patient is still in surgery.",
     "contradiction"),
] * 10   # Repeat to get 200 examples for stable training


FALLBACK_GSM8K = [
    {
        'question': "A store has 12 apples. They sell 5. How many remain?",
        'steps': [
            "Start with 12 apples.",
            "Subtract 5 sold apples: 12 - 5 = 7.",
            "The answer is 7 apples.",
        ]
    },
    {
        'question': "A train travels at 60 mph for 3 hours. How far does it go?",
        'steps': [
            "Speed is 60 mph, time is 3 hours.",
            "Distance = speed × time = 60 × 3 = 180.",
            "The train travels 180 miles.",
        ]
    },
    {
        'question': "There are 24 students in a class. Half are girls. How many boys?",
        'steps': [
            "Total students: 24.",
            "Girls: 24 / 2 = 12.",
            "Boys: 24 - 12 = 12.",
        ]
    },
    {
        'question': "A book costs $15. You buy 4 books. How much do you spend?",
        'steps': [
            "Cost per book: $15.",
            "Number of books: 4.",
            "Total: 15 × 4 = $60.",
        ]
    },
    {
        'question': "A box holds 8 oranges. You have 5 boxes. How many oranges total?",
        'steps': [
            "Oranges per box: 8.",
            "Number of boxes: 5.",
            "Total: 8 × 5 = 40 oranges.",
        ]
    },
    {
        'question': "A farmer has 30 cows and sells 12. How many remain?",
        'steps': [
            "Start with 30 cows.",
            "Sell 12: 30 - 12 = 18.",
            "The farmer has 18 cows.",
        ]
    },
    {
        'question': "A pool is 50 meters long. You swim 6 laps. How far do you swim?",
        'steps': [
            "Pool length: 50 meters.",
            "Laps: 6.",
            "Distance: 50 × 6 = 300 meters.",
        ]
    },
    {
        'question': "You earn $200 a week. How much in 4 weeks?",
        'steps': [
            "Weekly earnings: $200.",
            "Weeks: 4.",
            "Total: 200 × 4 = $800.",
        ]
    },
] * 5   # Repeat for 40 problems


class CurriculumRunner:
    """
    Ingests SNLI and GSM8K into the SparseResonanceField.

    SNLI builds contradiction/entailment geometry.
    GSM8K builds step-by-step logical chain geometry.
    Together they give the field the physics of reasoning.
    """

    def __init__(
        self,
        cse,
        cwc,
        field:      'SparseResonanceField',
        impl_store: ImplicationChainStore,
        device:     str = 'cuda',
    ):
        self.cse        = cse
        self.cwc        = cwc
        self.field      = field
        self.impl_store = impl_store
        self.device     = device

        self.snli_stats = {
            'entailment':    0,
            'contradiction': 0,
            'neutral':       0,
            'errors':        0,
        }
        self.gsm8k_stats = {
            'problems_ingested': 0,
            'steps_as_attractors': 0,
            'arrows_created': 0,
            'errors': 0,
        }

    def _encode(self, text: str) -> Tensor:
        """CSE → CWC → mean pool → [608]."""
        with torch.no_grad():
            wave   = self.cse.encode(text)
            causal = self.cwc.forward(wave)
        return causal.full.mean(dim=0)

    def _encode_base(self, text: str) -> Tensor:
        """CSE only → mean pool → [432] for field perturbation."""
        with torch.no_grad():
            wave = self.cse.encode(text)
        return wave.full.mean(dim=0)

    # ─────────────────────────────────────────
    # SNLI
    # ─────────────────────────────────────────

    def ingest_snli(
        self,
        premise:    str,
        hypothesis: str,
        label:      str,
    ):
        """
        Ingest one SNLI example.

        entailment:    reinforce gravity between premise and hypothesis
        contradiction: negative mass — physical repulsion
        neutral:       create uncertainty tension
        """
        try:
            wave_p = self._encode_base(premise)
            wave_h = self._encode_base(hypothesis)
            vec_p  = self._encode(premise)
            vec_h  = self._encode(hypothesis)

            if label == 'entailment':
                # Both attractors reinforced, strong arrow p → h
                self.field.perturb(wave_p, strength=0.8)
                self.field.perturb(wave_h, strength=0.8)
                self.impl_store.arrows.append(CausalArrow(
                    source_vector  = vec_p.cpu(),
                    target_vector  = vec_h.cpu(),
                    strength       = 0.85,
                    evidence_count = 1,
                    arrow_type     = 'entailment',
                ))
                self.snli_stats['entailment'] += 1

            elif label == 'contradiction':
                # Create attractors then repel them
                self.field.perturb(wave_p, strength=0.5)
                self.field.perturb(wave_h, strength=0.5)

                # Negative mass at hypothesis location
                h, w, d = self.field.wave_to_field_coords(wave_h)
                key = self.field.registry._key(h, w, d)
                current_mass = self.field.registry._mass.get(key, 0.0)
                self.field.registry._mass[key] = max(
                    -1.0, current_mass - 0.3
                )
                # Also negative mass at premise toward hypothesis
                hp, wp, dp = self.field.wave_to_field_coords(wave_p)
                # Register repulsion arrow
                self.impl_store.arrows.append(CausalArrow(
                    source_vector  = vec_p.cpu(),
                    target_vector  = vec_h.cpu(),
                    strength       = -0.9,
                    evidence_count = 1,
                    arrow_type     = 'contradiction',
                ))
                self.snli_stats['contradiction'] += 1

            else:  # neutral
                # Weak perturbation, no arrow
                self.field.perturb(wave_p, strength=0.3)
                self.field.perturb(wave_h, strength=0.3)
                self.snli_stats['neutral'] += 1

        except Exception:
            self.snli_stats['errors'] += 1

    def run_snli(
        self,
        examples:     Optional[List[Tuple[str,str,str]]] = None,
        max_examples: int = 500,
        log_every:    int = 50,
        check_growth: bool = True,
    ) -> Dict:
        """
        Run the full SNLI curriculum.

        Args:
            examples:     list of (premise, hypothesis, label), or None to load
            max_examples: cap on examples to process
            log_every:    log every N examples
            check_growth: check for field growth after each batch
        """
        if examples is None:
            examples = self._load_snli(max_examples)

        print(f"\n  SNLI Curriculum: {len(examples)} examples")

        for i, (premise, hypothesis, label) in enumerate(examples[:max_examples]):
            self.ingest_snli(premise, hypothesis, label)

            if (i + 1) % log_every == 0:
                self.field.growth_manager.log_capacity_status(self.field.registry)
                if check_growth:
                    self.field.check_and_grow({})

        print(f"  SNLI complete:")
        print(f"    Entailment:    {self.snli_stats['entailment']}")
        print(f"    Contradiction: {self.snli_stats['contradiction']}")
        print(f"    Neutral:       {self.snli_stats['neutral']}")
        print(f"    Active locs:   {self.field.registry.active_count():,}")

        return self.snli_stats

    def _load_snli(self, max_examples: int) -> List[Tuple[str,str,str]]:
        try:
            from datasets import load_dataset
            ds = load_dataset('snli', split='train')
            label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
            examples = []
            for item in ds:
                if len(examples) >= max_examples:
                    break
                if item['label'] in label_map:
                    examples.append((
                        item['premise'],
                        item['hypothesis'],
                        label_map[item['label']],
                    ))
            print(f"  ✓ Loaded {len(examples)} SNLI examples")
            return examples
        except Exception as e:
            print(f"  ⚠ SNLI download failed ({e}), using {len(FALLBACK_SNLI)} fallback examples")
            return FALLBACK_SNLI[:max_examples]

    # ─────────────────────────────────────────
    # GSM8K
    # ─────────────────────────────────────────

    def ingest_gsm8k(
        self,
        question: str,
        steps:    List[str],
    ):
        """
        Ingest one GSM8K problem.

        Each step becomes an attractor.
        Sequential step pairs get a causal arrow between them.
        This embeds the geometry of deduction into the field.
        """
        try:
            if not steps:
                return

            step_vecs = []
            for step in steps:
                wave = self._encode_base(step)
                vec  = self._encode(step)
                self.field.perturb(wave, strength=0.7)
                step_vecs.append((wave, vec))
                self.gsm8k_stats['steps_as_attractors'] += 1

            # Causal arrows between sequential steps
            for j in range(len(step_vecs) - 1):
                _, vec_a = step_vecs[j]
                _, vec_b = step_vecs[j + 1]
                # Strength decreases slightly for later steps
                strength = 0.9 - j * 0.05
                self.impl_store.arrows.append(CausalArrow(
                    source_vector  = vec_a.cpu(),
                    target_vector  = vec_b.cpu(),
                    strength       = max(0.5, strength),
                    evidence_count = 1,
                    arrow_type     = 'deduction',
                ))
                self.gsm8k_stats['arrows_created'] += 1

            self.gsm8k_stats['problems_ingested'] += 1

        except Exception:
            self.gsm8k_stats['errors'] += 1

    def run_gsm8k(
        self,
        problems:     Optional[List[Dict]] = None,
        max_problems: int = 200,
        log_every:    int = 20,
        check_growth: bool = True,
    ) -> Dict:
        """
        Run the full GSM8K curriculum.

        Args:
            problems:     list of {'question': str, 'steps': [str]}, or None to load
            max_problems: cap on problems to process
            log_every:    log every N problems
            check_growth: check for field growth
        """
        if problems is None:
            problems = self._load_gsm8k(max_problems)

        print(f"\n  GSM8K Curriculum: {len(problems)} problems")

        for i, problem in enumerate(problems[:max_problems]):
            self.ingest_gsm8k(
                problem['question'],
                problem.get('steps', [problem.get('answer', '')])
            )

            if (i + 1) % log_every == 0:
                self.field.growth_manager.log_capacity_status(self.field.registry)
                if check_growth:
                    self.field.check_and_grow({})

        print(f"  GSM8K complete:")
        print(f"    Problems ingested:    {self.gsm8k_stats['problems_ingested']}")
        print(f"    Steps as attractors:  {self.gsm8k_stats['steps_as_attractors']}")
        print(f"    Arrows created:       {self.gsm8k_stats['arrows_created']}")
        print(f"    Active locs:          {self.field.registry.active_count():,}")

        return self.gsm8k_stats

    def _load_gsm8k(self, max_problems: int) -> List[Dict]:
        try:
            from datasets import load_dataset
            ds = load_dataset('gsm8k', 'main', split='train')
            problems = []
            for item in ds:
                if len(problems) >= max_problems:
                    break
                # Parse steps from the annotated solution
                answer = item.get('answer', '')
                steps  = self._parse_gsm8k_steps(answer)
                if steps:
                    problems.append({
                        'question': item['question'],
                        'steps':    steps,
                    })
            print(f"  ✓ Loaded {len(problems)} GSM8K problems")
            return problems
        except Exception as e:
            print(f"  ⚠ GSM8K download failed ({e}), using {len(FALLBACK_GSM8K)} fallback problems")
            return FALLBACK_GSM8K[:max_problems]

    def _parse_gsm8k_steps(self, answer: str) -> List[str]:
        """
        Parse GSM8K annotated solution into step list.
        Solutions use '<<...>>' for calculations and '####' for final answer.
        """
        # Remove calculation annotations
        cleaned = re.sub(r'<<[^>]*>>', '', answer)
        # Split on newlines and sentence boundaries
        raw_steps = [s.strip() for s in cleaned.split('\n') if s.strip()]
        # Filter out very short steps
        steps = [s for s in raw_steps if len(s) > 10]
        return steps if steps else [answer[:200]]
