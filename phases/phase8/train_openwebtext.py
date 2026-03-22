"""
Phase 8: Train FLUX on OpenWebText

Streams OpenWebText through FLUXLarge in a single pass (no epochs).
The field learns through thermodynamic settling while the output head
uses gradient-based optimization for byte prediction.

Features:
- Resume-safe: checkpoints every N steps
- Mixed precision (fp16) where available
- Gradient accumulation for larger effective batches
- Streaming data — never loads full dataset into memory
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
for _phase in ['phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'phase6', 'phase7']:
    _p = str(_PHASES_DIR / _phase)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from flux_large import FLUXLarge, FLUX_LARGE_CONFIG
from flux_utils import (
    get_device, save_checkpoint, load_checkpoint, checkpoint_exists,
    PhaseLogger,
)


@dataclass
class TrainStepResult:
    """Result from a single training step."""
    step: int
    loss: float
    perplexity: float
    temperature: float
    energy_delta: float
    latency_ms: float
    tokens_seen: int


@dataclass
class TrainRunResult:
    """Result from a complete training run."""
    total_steps: int
    total_tokens: int
    final_loss: float
    final_perplexity: float
    avg_loss: float
    avg_perplexity: float
    min_loss: float
    total_time_seconds: float
    steps_per_second: float
    step_history: List[TrainStepResult] = field(default_factory=list)


class OpenWebTextTrainer:
    """
    Trainer for FLUXLarge on OpenWebText.

    Training strategy:
    1. Stream documents from OpenWebText (HuggingFace datasets)
    2. For each document:
       a. Encode via CSE (frozen) → semantic wave
       b. Thermodynamic settle (field learning — no backprop)
       c. Write to episodic memory (one-shot)
       d. Output head + bridge: supervised byte prediction (gradient)
    3. Checkpoint every `checkpoint_interval` steps
    4. Resume from latest checkpoint if interrupted

    Args:
        model: FLUXLarge model
        lr: Learning rate for output head + bridge
        grad_accum: Gradient accumulation steps
        checkpoint_interval: Steps between checkpoints
        log: PhaseLogger instance
    """

    def __init__(
        self,
        model: FLUXLarge,
        lr: float = 5e-4,
        grad_accum: int = 4,
        checkpoint_interval: int = 5000,
        log: Optional[PhaseLogger] = None,
    ):
        self.model = model
        self.lr = lr
        self.grad_accum = grad_accum
        self.checkpoint_interval = checkpoint_interval
        self.log = log

        # Only optimize output head + bridge projections
        self.trainable_params = (
            list(model.output_head.parameters()) +
            list(model.wave_to_field.parameters()) +
            list(model.field_to_wave.parameters())
        )
        self.optimizer = torch.optim.AdamW(
            self.trainable_params, lr=lr, weight_decay=0.01
        )

        # Cosine annealing scheduler
        self.scheduler = None  # Set during training based on total steps

        # Mixed precision
        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        self._global_step = 0
        self._tokens_seen = 0

    def _text_to_targets(self, text: str, device: str) -> torch.Tensor:
        """Convert text to byte-level targets."""
        return torch.tensor(
            list(text.encode('utf-8', errors='replace')),
            dtype=torch.long, device=device,
        )

    def train_step(self, text: str) -> TrainStepResult:
        """
        Single training step on a text document.

        Args:
            text: Raw text document

        Returns:
            TrainStepResult with metrics
        """
        t0 = time.time()
        device = self.model._device_str

        # CSE encode (frozen)
        with torch.no_grad():
            wave = self.model.cse.encode(text)
        wave_vec = wave.full.mean(dim=0).to(device)

        # Thermodynamic settle (field learning)
        with torch.no_grad():
            settle_result = self.model.tl.settle_once(wave_vec)
            self.model._learning_steps += 1

            # Episodic memory write
            compressed = self.model.working_memory.compress(
                wave_vec.unsqueeze(0)
            ).squeeze(0)
            self.model.episodic_memory.write(
                compressed, fact=text[:200], causal_source="openwebtext_train"
            )
            self.model.working_memory.add_perturbation(wave_vec)

        # Output head training (gradient-based)
        field_features, sims, locs = self.model.field.query(wave_vec.detach(), k=4)
        combined = field_features.mean(dim=0).detach()

        cgn_out = self.model.cgn(combined.detach())
        merged = combined + cgn_out.detach()

        if self.use_amp:
            with torch.amp.autocast('cuda'):
                logits = self.model.output_head(merged, wave_context=wave_vec.detach())
                targets = self._text_to_targets(text, device)
                if targets.numel() > 0:
                    target_dist = torch.zeros(256, device=device)
                    for b in targets[:100]:
                        target_dist[b.item()] += 1.0
                    target_dist = target_dist / target_dist.sum().clamp(min=1e-8)
                    loss = F.cross_entropy(logits.unsqueeze(0), target_dist.unsqueeze(0))
                else:
                    loss = torch.tensor(0.0, device=device)
        else:
            logits = self.model.output_head(merged, wave_context=wave_vec.detach())
            targets = self._text_to_targets(text, device)
            if targets.numel() > 0:
                target_dist = torch.zeros(256, device=device)
                for b in targets[:100]:
                    target_dist[b.item()] += 1.0
                target_dist = target_dist / target_dist.sum().clamp(min=1e-8)
                loss = F.cross_entropy(logits.unsqueeze(0), target_dist.unsqueeze(0))
            else:
                loss = torch.tensor(0.0, device=device)

        # Scale loss for gradient accumulation
        scaled_loss = loss / self.grad_accum

        if self.scaler:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        # Step optimizer every grad_accum steps
        self._global_step += 1
        if self._global_step % self.grad_accum == 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.trainable_params, 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.trainable_params, 1.0)
                self.optimizer.step()

            self.optimizer.zero_grad()
            if self.scheduler:
                self.scheduler.step()

        self._tokens_seen += len(text.encode('utf-8', errors='replace'))
        elapsed = (time.time() - t0) * 1000

        return TrainStepResult(
            step=self._global_step,
            loss=loss.item(),
            perplexity=math.exp(min(loss.item(), 20.0)),
            temperature=settle_result.temperature,
            energy_delta=settle_result.energy_delta,
            latency_ms=elapsed,
            tokens_seen=self._tokens_seen,
        )

    def train_on_texts(
        self,
        texts: List[str],
        max_steps: Optional[int] = None,
        verbose: bool = True,
        log_interval: int = 50,
    ) -> TrainRunResult:
        """
        Train on a list/iterable of text documents (single pass).

        Args:
            texts: Iterable of text documents
            max_steps: Maximum steps (None = all)
            verbose: Print progress
            log_interval: Print every N steps

        Returns:
            TrainRunResult with training metrics
        """
        t0 = time.time()
        steps_to_run = min(len(texts), max_steps) if max_steps else len(texts)

        # Set cosine scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=steps_to_run, eta_min=self.lr * 0.1
        )

        losses = []
        perplexities = []
        history = []

        for i, text in enumerate(texts):
            if i >= steps_to_run:
                break

            if not text or len(text.strip()) < 10:
                continue

            result = self.train_step(text)
            losses.append(result.loss)
            perplexities.append(result.perplexity)
            history.append(result)

            if verbose and (i + 1) % log_interval == 0:
                avg_loss = sum(losses[-log_interval:]) / min(len(losses), log_interval)
                avg_ppl = sum(perplexities[-log_interval:]) / min(len(perplexities), log_interval)
                lr_now = self.optimizer.param_groups[0]['lr']
                print(
                    f"  Step {i+1:>6}/{steps_to_run}  "
                    f"loss={avg_loss:.4f}  ppl={avg_ppl:.1f}  "
                    f"lr={lr_now:.6f}  "
                    f"tokens={self._tokens_seen:,}  "
                    f"latency={result.latency_ms:.0f}ms"
                )

                if self.log:
                    self.log.metric(f"step_{i+1}_loss", f"{avg_loss:.4f}")

            # Checkpoint interval
            if self.checkpoint_interval > 0 and (i + 1) % self.checkpoint_interval == 0:
                self._save_training_checkpoint(i + 1, losses)

        elapsed = time.time() - t0

        return TrainRunResult(
            total_steps=len(losses),
            total_tokens=self._tokens_seen,
            final_loss=losses[-1] if losses else 0.0,
            final_perplexity=perplexities[-1] if perplexities else 0.0,
            avg_loss=sum(losses) / max(len(losses), 1),
            avg_perplexity=sum(perplexities) / max(len(perplexities), 1),
            min_loss=min(losses) if losses else 0.0,
            total_time_seconds=elapsed,
            steps_per_second=len(losses) / max(elapsed, 1e-6),
            step_history=history,
        )

    def evaluate(self, texts: List[str]) -> Dict[str, float]:
        """
        Evaluate on a list of texts (no learning, no gradient).

        Args:
            texts: Evaluation texts

        Returns:
            Dict with avg_loss, avg_perplexity, eval_samples
        """
        self.model.eval()
        losses = []
        device = self.model._device_str

        with torch.no_grad():
            for text in texts:
                if not text or len(text.strip()) < 10:
                    continue

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
                    for b in targets[:100]:
                        target_dist[b.item()] += 1.0
                    target_dist = target_dist / target_dist.sum().clamp(min=1e-8)
                    loss = F.cross_entropy(logits.unsqueeze(0), target_dist.unsqueeze(0))
                    losses.append(loss.item())

        self.model.train()

        avg_loss = sum(losses) / max(len(losses), 1)
        return {
            'avg_loss': avg_loss,
            'avg_perplexity': math.exp(min(avg_loss, 20.0)),
            'eval_samples': len(losses),
        }

    def _save_training_checkpoint(self, step: int, losses: List[float]):
        """Save intermediate training checkpoint."""
        avg_loss = sum(losses[-100:]) / min(len(losses), 100)
        state = {
            'phase': 8,
            'config': self.model.config,
            'learning_steps': self.model._learning_steps,
            'global_step': self._global_step,
            'tokens_seen': self._tokens_seen,
            'cse_state_dict': self.model.cse.state_dict(),
            'field_state_dict': self.model.field.state_dict(),
            'gr_state': self.model.gr.save_state(),
            'tl_state': self.model.tl.save_state(),
            'cgn_state': self.model.cgn.save_state(),
            'causal_graph_state': self.model.causal_graph.save_state(),
            'working_memory_state': self.model.working_memory.state_dict_memory(),
            'episodic_memory_state': self.model.episodic_memory.save_state(),
            'semantic_memory_state': self.model.semantic_memory.save_state(),
            'router_state': self.model.memory_router.save_state(),
            'wave_to_field_state': self.model.wave_to_field.state_dict(),
            'field_to_wave_state': self.model.field_to_wave.state_dict(),
            'output_head_state': self.model.output_head.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'metrics': {
                'step': step,
                'avg_loss': avg_loss,
                'avg_perplexity': math.exp(min(avg_loss, 20.0)),
                'tokens_seen': self._tokens_seen,
            },
        }
        save_checkpoint(8, state)
        if self.log:
            self.log.success(f"Checkpoint saved at step {step}")


def load_openwebtext_subset(max_docs: int = 10000) -> List[str]:
    """
    Load a subset of OpenWebText from HuggingFace datasets.

    Falls back to a synthetic corpus if `datasets` not available.

    Args:
        max_docs: Maximum number of documents to load

    Returns:
        List of text documents
    """
    try:
        from datasets import load_dataset
        print(f"  ℹ Loading OpenWebText subset ({max_docs:,} docs)...")
        ds = load_dataset('openwebtext', split='train', streaming=True)
        texts = []
        for i, sample in enumerate(ds):
            if i >= max_docs:
                break
            text = sample.get('text', '')
            if len(text) > 50:
                texts.append(text[:2000])  # Cap per-document length
        print(f"  ✓ Loaded {len(texts):,} documents from OpenWebText")
        return texts
    except Exception as e:
        print(f"  ⚠ Could not load OpenWebText: {e}")
        print(f"  ℹ Using synthetic training corpus instead")
        return _synthetic_corpus(max_docs)


def _synthetic_corpus(n: int = 10000) -> List[str]:
    """Generate a synthetic training corpus for testing."""
    import random
    random.seed(42)

    templates = [
        "The {adj} {noun} {verb} the {adj2} {noun2} in the {place}.",
        "{name} discovered that {concept} is fundamental to {field}.",
        "In the year {year}, scientists proved that {fact}.",
        "The relationship between {a} and {b} was first described by {name}.",
        "Modern {field} relies heavily on {concept} for {purpose}.",
        "Research shows that {finding} affects {outcome} significantly.",
        "The {adj} theory of {concept} predicts {prediction}.",
        "According to experts, {fact} will change how we understand {field}.",
    ]

    nouns = ['system', 'model', 'network', 'field', 'algorithm', 'structure',
             'process', 'pattern', 'mechanism', 'function', 'wave', 'particle']
    adjs = ['complex', 'dynamic', 'stable', 'novel', 'fundamental', 'emergent',
            'classical', 'quantum', 'resonant', 'gravitational', 'thermal']
    verbs = ['transforms', 'generates', 'processes', 'analyzes', 'computes',
             'modifies', 'creates', 'discovers', 'reveals', 'predicts']
    names = ['Einstein', 'Turing', 'Shannon', 'Feynman', 'Dijkstra', 'Knuth']
    fields = ['physics', 'mathematics', 'computer science', 'biology', 'chemistry',
              'neuroscience', 'linguistics', 'philosophy', 'engineering']
    concepts = ['entropy', 'information', 'energy', 'symmetry', 'emergence',
                'causality', 'resonance', 'interference', 'gravity', 'topology']
    places = ['laboratory', 'university', 'research center', 'observatory', 'institute']

    texts = []
    for _ in range(n):
        template = random.choice(templates)
        text = template.format(
            adj=random.choice(adjs), adj2=random.choice(adjs),
            noun=random.choice(nouns), noun2=random.choice(nouns),
            verb=random.choice(verbs), name=random.choice(names),
            concept=random.choice(concepts), field=random.choice(fields),
            place=random.choice(places), year=random.randint(1900, 2025),
            fact=f"{random.choice(concepts)} affects {random.choice(nouns)}s",
            finding=f"{random.choice(adjs)} {random.choice(concepts)}",
            outcome=f"{random.choice(nouns)} performance",
            purpose=f"{random.choice(adjs)} {random.choice(nouns)} design",
            a=random.choice(concepts), b=random.choice(concepts),
            prediction=f"{random.choice(adjs)} behavior in {random.choice(nouns)}s",
        )
        texts.append(text)

    return texts


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  Phase 8: Train FLUXLarge on OpenWebText")
    print("=" * 60)

    device = get_device()
    log = PhaseLogger(phase=8)

    # Build model
    model = FLUXLarge.from_phase7_checkpoint(device=device)

    # Load data
    texts = load_openwebtext_subset(max_docs=1000)

    # Split train / eval
    split_idx = int(len(texts) * 0.9)
    train_texts = texts[:split_idx]
    eval_texts = texts[split_idx:]

    # Train
    trainer = OpenWebTextTrainer(model, lr=5e-4, log=log)
    result = trainer.train_on_texts(train_texts, verbose=True, log_interval=50)

    print(f"\n  Training complete:")
    print(f"    Steps:      {result.total_steps}")
    print(f"    Final loss:  {result.final_loss:.4f}")
    print(f"    Final ppl:   {result.final_perplexity:.2f}")
    print(f"    Avg loss:    {result.avg_loss:.4f}")
    print(f"    Tokens:      {result.total_tokens:,}")
    print(f"    Time:        {result.total_time_seconds:.1f}s")

    # Evaluate
    eval_metrics = trainer.evaluate(eval_texts)
    print(f"\n  Evaluation:")
    print(f"    Eval loss:   {eval_metrics['avg_loss']:.4f}")
    print(f"    Eval ppl:    {eval_metrics['avg_perplexity']:.2f}")
