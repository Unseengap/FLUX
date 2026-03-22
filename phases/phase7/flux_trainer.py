"""
Phase 7: Unified FLUX Trainer

Trains the integrated model's output head and bridge projections
on text data using thermodynamic settling + episodic learning.

Unlike traditional training:
  - No epochs: Single-pass stream through data
  - No backprop for field: Thermodynamic settling IS learning
  - Output head uses supervised gradient for byte prediction
  - Episodic memory learns facts one-shot during streaming
"""

import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

_PHASES_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PHASES_DIR.parent
for _phase in ['phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'phase6']:
    _p = str(_PHASES_DIR / _phase)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from flux_model import FLUXModel


@dataclass
class TrainStep:
    """Result of a single training step."""
    step: int
    loss: float
    perplexity: float
    temperature: float
    energy_delta: float
    latency_ms: float


@dataclass
class TrainResult:
    """Result of a full training run."""
    total_steps: int
    final_loss: float
    final_perplexity: float
    avg_loss: float
    avg_perplexity: float
    min_loss: float
    total_time_seconds: float
    steps_per_second: float
    step_history: List[TrainStep] = field(default_factory=list)


class FLUXTrainer:
    """
    Unified training for the integrated FLUX model.

    Training combines two processes:
    1. **Thermodynamic settling:** The field learns structure from data
       implicitly — no backprop needed. Each sample perturbs the field
       and it settles to a new energy minimum.
    2. **Output head optimization:** Standard gradient descent trains only
       the output projection (bridge layers + output head) for byte prediction.

    This hybrid approach means the vast majority of parameters (field, CGN,
    memory) learn without gradients, while only the small output head uses them.
    """

    def __init__(self, model: FLUXModel, lr: float = 1e-3,
                 log_interval: int = 50):
        self.model = model
        self.lr = lr
        self.log_interval = log_interval

        # Only optimize output head + bridge projections
        # (field, CSE, GR, CGN, memory learn through their own physics)
        self.trainable_params = list(model.output_head.parameters()) + \
                                list(model.wave_to_field.parameters()) + \
                                list(model.field_to_wave.parameters())
        self.optimizer = torch.optim.Adam(self.trainable_params, lr=lr)

    def _text_to_targets(self, text: str, device: str) -> torch.Tensor:
        """
        Convert text to byte-level targets.

        Args:
            text: Input text

        Returns:
            [len] long tensor of byte values
        """
        return torch.tensor(list(text.encode('utf-8')), dtype=torch.long, device=device)

    def train_step(self, text: str) -> TrainStep:
        """
        Single training step on a text sample.

        1. Encode text with CSE (no grad — frozen)
        2. Pass through field + GR + CGN (thermodynamic — no grad)
        3. Compute byte-level prediction loss on output head (grad)
        4. Update output head + bridge projections only

        Args:
            text: Raw text sample

        Returns:
            TrainStep with loss and metrics
        """
        t0 = time.time()
        device = self.model._device_str

        # Get semantic wave
        with torch.no_grad():
            wave = self.model.cse.encode(text)
        wave_vec = wave.full.mean(dim=0).to(device)

        # Thermodynamic settle (field learning — no gradient needed)
        with torch.no_grad():
            settle_result = self.model.tl.settle_once(wave_vec)
            self.model._learning_steps += 1

            # Memory write (one-shot)
            compressed = self.model.working_memory.compress(
                wave_vec.unsqueeze(0)
            ).squeeze(0)
            self.model.episodic_memory.write(
                compressed, fact=text[:200], causal_source="training"
            )
            self.model.working_memory.add_perturbation(wave_vec)

        # Output head training (gradient-based)
        field_features, sims, locs = self.model.field.query(wave_vec.detach(), k=4)
        combined = field_features.mean(dim=0).detach()  # [field_features]

        # CGN processing (pass gradient through bridge only)
        cgn_out = self.model.cgn(combined.detach())
        merged = combined + cgn_out.detach()

        # Generate logits through output head
        logits = self.model.output_head(merged, wave_context=wave_vec.detach())
        # logits shape: [vocab_size=256]

        # Target: next bytes from the text
        targets = self._text_to_targets(text, device)
        if targets.numel() > 0:
            # Use the last byte as a prediction target
            target_dist = torch.zeros(256, device=device)
            for b in targets[:50]:  # use first 50 bytes for supervision
                target_dist[b.item()] += 1.0
            target_dist = target_dist / target_dist.sum().clamp(min=1e-8)

            loss = F.cross_entropy(logits.unsqueeze(0), target_dist.unsqueeze(0))
        else:
            loss = torch.tensor(0.0, device=device)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.trainable_params, 1.0)
        self.optimizer.step()

        elapsed = (time.time() - t0) * 1000
        ppl = math.exp(min(loss.item(), 20.0))

        return TrainStep(
            step=self.model._learning_steps,
            loss=loss.item(),
            perplexity=ppl,
            temperature=settle_result.temperature,
            energy_delta=settle_result.energy_delta,
            latency_ms=elapsed,
        )

    def train_on_texts(self, texts: List[str], max_steps: Optional[int] = None,
                       verbose: bool = True) -> TrainResult:
        """
        Train on a list of text samples (single pass, no epochs).

        Args:
            texts: List of text samples
            max_steps: Maximum training steps (None = all texts)
            verbose: Print progress

        Returns:
            TrainResult with full training metrics
        """
        t0 = time.time()
        steps_to_run = min(len(texts), max_steps) if max_steps else len(texts)

        losses = []
        perplexities = []
        history: List[TrainStep] = []

        for i, text in enumerate(texts[:steps_to_run]):
            if not text.strip():
                continue

            result = self.train_step(text)
            losses.append(result.loss)
            perplexities.append(result.perplexity)
            history.append(result)

            if verbose and (i + 1) % self.log_interval == 0:
                recent_loss = sum(losses[-self.log_interval:]) / min(len(losses), self.log_interval)
                recent_ppl = sum(perplexities[-self.log_interval:]) / min(len(perplexities), self.log_interval)
                print(f"  Step {i+1}/{steps_to_run}  "
                      f"loss={recent_loss:.4f}  "
                      f"ppl={recent_ppl:.2f}  "
                      f"temp={result.temperature:.4f}  "
                      f"ΔE={result.energy_delta:.6f}")

        elapsed = time.time() - t0
        avg_loss = sum(losses) / max(len(losses), 1)
        avg_ppl = sum(perplexities) / max(len(perplexities), 1)

        return TrainResult(
            total_steps=len(losses),
            final_loss=losses[-1] if losses else 0.0,
            final_perplexity=perplexities[-1] if perplexities else 0.0,
            avg_loss=avg_loss,
            avg_perplexity=avg_ppl,
            min_loss=min(losses) if losses else 0.0,
            total_time_seconds=elapsed,
            steps_per_second=len(losses) / max(elapsed, 1e-8),
            step_history=history,
        )

    def train_on_corpus(self, corpus_text: str, chunk_size: int = 200,
                        max_steps: Optional[int] = None,
                        verbose: bool = True) -> TrainResult:
        """
        Train on a text corpus by chunking into samples.

        Unlike traditional training:
        - Single pass through data (no epochs)
        - Each chunk is seen exactly once
        - Field learns via thermodynamic settling
        - Only output head uses gradient updates

        Args:
            corpus_text: Raw corpus text
            chunk_size: Characters per chunk
            max_steps: Limit number of chunks processed
            verbose: Print progress

        Returns:
            TrainResult
        """
        chunks = [corpus_text[i:i+chunk_size]
                  for i in range(0, len(corpus_text), chunk_size)
                  if len(corpus_text[i:i+chunk_size].strip()) > 10]

        if verbose:
            print(f"  Corpus: {len(corpus_text):,} chars → {len(chunks)} chunks "
                  f"(chunk_size={chunk_size})")

        return self.train_on_texts(chunks, max_steps=max_steps, verbose=verbose)

    def evaluate(self, eval_texts: List[str]) -> Dict[str, float]:
        """
        Evaluate model on held-out texts.

        Args:
            eval_texts: List of evaluation text samples

        Returns:
            Dict with avg_loss, avg_perplexity, etc.
        """
        self.model.eval()
        total_loss = 0.0
        count = 0

        for text in eval_texts:
            if not text.strip():
                continue
            device = self.model._device_str

            with torch.no_grad():
                wave = self.model.cse.encode(text)
                wave_vec = wave.full.mean(dim=0).to(device)

                field_features, sims, locs = self.model.field.query(wave_vec, k=4)
                combined = field_features.mean(dim=0)
                cgn_out = self.model.cgn(combined)
                merged = combined + cgn_out

                logits = self.model.output_head(merged, wave_context=wave_vec)

                targets = self._text_to_targets(text, device)
                if targets.numel() > 0:
                    target_dist = torch.zeros(256, device=device)
                    for b in targets[:50]:
                        target_dist[b.item()] += 1.0
                    target_dist = target_dist / target_dist.sum().clamp(min=1e-8)
                    loss = F.cross_entropy(logits.unsqueeze(0), target_dist.unsqueeze(0))
                    total_loss += loss.item()
                    count += 1

        self.model.train()
        avg_loss = total_loss / max(count, 1)
        return {
            'avg_loss': avg_loss,
            'avg_perplexity': math.exp(min(avg_loss, 20.0)),
            'eval_samples': count,
        }
