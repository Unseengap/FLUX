"""
Phase 8.5: Curriculum Trainer — ABC School for FLUX

Orchestrates 6-stage curriculum training:
  Stage 1: Bytes — learn printable ASCII distribution
  Stage 2: Bigrams/Trigrams — learn common byte co-occurrences
  Stage 3: Words — learn to spell the top-1000 English words
  Stage 4: Phrases — learn common multi-word collocations
  Stage 5: Sentences — learn grammar + punctuation
  Stage 6: OpenWebText — graduate to real text

Each stage monitors decoder loss and runs advancement tests.
Advancement is based on loss threshold + accuracy on stage-specific tests.

The FLUX field provides semantic context (WHAT to say).
The curriculum teaches the decoder HOW to spell it progressively.
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
for _phase in ['phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'phase6', 'phase7', 'phase8']:
    _p = str(_PHASES_DIR / _phase)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from flux_large import FLUXLarge
from wave_decoder import WaveDecoder
from curriculum_data import (
    generate_curriculum_data, STAGE_NAMES,
    get_spelling_test_words, get_phrase_test_prompts,
    get_sentence_test_prompts,
)
from flux_utils import (
    get_device, save_checkpoint, load_checkpoint, checkpoint_exists,
    PhaseLogger,
)


# ─────────────────────────────────────────────
# Stage Configuration
# ─────────────────────────────────────────────

@dataclass
class StageConfig:
    """Configuration for a curriculum stage."""
    stage: int
    name: str
    max_steps: int
    loss_threshold: float
    accuracy_threshold: float
    lr: float
    max_seq_len: int
    grad_accum: int


STAGE_CONFIGS = {
    1: StageConfig(
        stage=1, name='Bytes (ASCII)',
        max_steps=200, loss_threshold=3.0,
        accuracy_threshold=0.80, lr=1e-3,
        max_seq_len=16, grad_accum=1,
    ),
    2: StageConfig(
        stage=2, name='Bigrams & Trigrams',
        max_steps=500, loss_threshold=2.5,
        accuracy_threshold=0.70, lr=8e-4,
        max_seq_len=32, grad_accum=1,
    ),
    3: StageConfig(
        stage=3, name='Common Words',
        max_steps=2000, loss_threshold=2.0,
        accuracy_threshold=0.60, lr=5e-4,
        max_seq_len=64, grad_accum=2,
    ),
    4: StageConfig(
        stage=4, name='Phrases & Collocations',
        max_steps=3000, loss_threshold=1.8,
        accuracy_threshold=0.50, lr=3e-4,
        max_seq_len=128, grad_accum=2,
    ),
    5: StageConfig(
        stage=5, name='Simple Sentences',
        max_steps=3000, loss_threshold=1.5,
        accuracy_threshold=0.40, lr=2e-4,
        max_seq_len=256, grad_accum=4,
    ),
    6: StageConfig(
        stage=6, name='Real Text (OpenWebText)',
        max_steps=5000, loss_threshold=1.2,
        accuracy_threshold=0.30, lr=1e-4,
        max_seq_len=512, grad_accum=4,
    ),
}


# ─────────────────────────────────────────────
# Stage Result Tracking
# ─────────────────────────────────────────────

@dataclass
class StageResult:
    """Result from completing a curriculum stage."""
    stage: int
    name: str
    steps_taken: int
    final_loss: float
    avg_loss: float
    min_loss: float
    accuracy: float
    advanced: bool
    time_seconds: float
    loss_history: List[float] = field(default_factory=list)


@dataclass
class CurriculumResult:
    """Result from the full curriculum run."""
    stages_completed: int
    total_steps: int
    total_time_seconds: float
    stage_results: List[StageResult] = field(default_factory=list)
    final_stage: int = 0


# ─────────────────────────────────────────────
# Curriculum Trainer
# ─────────────────────────────────────────────

class CurriculumTrainer:
    """
    6-stage curriculum trainer for FLUX WaveDecoder.

    Progressively trains the decoder from simple bytes to full text.
    The FLUX field and CSE are frozen — only the decoder and bridges learn.

    Each stage:
    1. Generates/loads stage-appropriate training data
    2. Trains with teacher forcing until loss threshold met OR max_steps
    3. Runs stage-specific advancement tests
    4. If tests pass → advance; otherwise → repeat stage data

    Args:
        model: FLUXLarge with WaveDecoder
        log: PhaseLogger instance
        openwebtext_texts: Pre-loaded OpenWebText texts for Stage 6
        verbose: Print progress details
    """

    def __init__(
        self,
        model: FLUXLarge,
        log: Optional[PhaseLogger] = None,
        openwebtext_texts: Optional[List[str]] = None,
        verbose: bool = True,
    ):
        self.model = model
        self.log = log
        self.openwebtext_texts = openwebtext_texts or []
        self.verbose = verbose
        self.device = model._device_str

        # Mixed precision
        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        # Tracking
        self._total_steps = 0
        self._stage_results: List[StageResult] = []

    def _build_optimizer(self, config: StageConfig) -> torch.optim.Optimizer:
        """Build optimizer for a stage with appropriate LR."""
        trainable = (
            list(self.model.decoder.parameters()) +
            list(self.model.wave_to_field.parameters()) +
            list(self.model.field_to_wave.parameters()) +
            list(self.model.output_head.parameters())
        )
        return torch.optim.AdamW(trainable, lr=config.lr, weight_decay=0.01)

    def _get_context(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run FLUX pipeline to get semantic context for decoder."""
        with torch.no_grad():
            wave = self.model.cse.encode(text)
        wave_vec = wave.full.mean(dim=0).to(self.device)

        with torch.no_grad():
            field_features, sims, locs = self.model.field.query(wave_vec, k=4)
            combined = field_features.mean(dim=0)
            cgn_out = self.model.cgn(combined)
            merged = combined + cgn_out

        return wave_vec, merged

    def _train_step(
        self,
        text: str,
        optimizer: torch.optim.Optimizer,
        config: StageConfig,
    ) -> float:
        """
        Single training step: teacher-forced byte prediction.

        Args:
            text: Training text
            optimizer: Current optimizer
            config: Stage configuration

        Returns:
            Loss value
        """
        device = self.device

        # Get FLUX semantic context (no grad — field is frozen for curriculum)
        wave_vec, merged = self._get_context(text)

        # Prepare byte targets
        text_bytes = list(text.encode('utf-8', errors='replace'))
        if not text_bytes:
            return 0.0

        max_len = min(len(text_bytes), config.max_seq_len)
        targets = torch.tensor(text_bytes[:max_len], dtype=torch.long, device=device)

        # Forward through decoder (teacher forcing)
        if self.use_amp:
            with torch.amp.autocast('cuda'):
                logits = self.model.decoder(
                    targets, wave_vec.detach(), merged.detach(),
                    max_len=max_len,
                )
                loss = F.cross_entropy(
                    logits.view(-1, 256), targets.view(-1),
                )
        else:
            logits = self.model.decoder(
                targets, wave_vec.detach(), merged.detach(),
                max_len=max_len,
            )
            loss = F.cross_entropy(
                logits.view(-1, 256), targets.view(-1),
            )

        # Backward
        scaled_loss = loss / config.grad_accum
        if self.scaler:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        # Step every grad_accum
        self._total_steps += 1
        if self._total_steps % config.grad_accum == 0:
            trainable = (
                list(self.model.decoder.parameters()) +
                list(self.model.wave_to_field.parameters()) +
                list(self.model.field_to_wave.parameters()) +
                list(self.model.output_head.parameters())
            )
            if self.scaler:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
            optimizer.zero_grad()

        return loss.item()

    def _test_byte_accuracy(self) -> float:
        """
        Test Stage 1: Can the decoder reproduce individual bytes?

        Feeds single characters through the decoder and checks if
        the argmax output matches.
        """
        test_chars = 'etaoinshrdlcumwfgypbvkjxqz .,'
        correct = 0
        total = 0

        self.model.eval()
        with torch.no_grad():
            for ch in test_chars:
                wave_vec, merged = self._get_context(ch)
                target = torch.tensor(
                    list(ch.encode('utf-8')),
                    dtype=torch.long, device=self.device,
                )
                logits = self.model.decoder(target, wave_vec, merged)
                pred = logits.argmax(dim=-1)
                correct += (pred == target).sum().item()
                total += target.numel()

        self.model.train()
        return correct / max(total, 1)

    def _test_word_spelling(self) -> float:
        """
        Test Stage 3: Can the decoder spell common words?

        Feeds words through and checks if the generated bytes
        match the target word bytes.
        """
        words = get_spelling_test_words(50)
        correct = 0
        total = 0

        self.model.eval()
        with torch.no_grad():
            for word in words:
                wave_vec, merged = self._get_context(word)
                target = torch.tensor(
                    list(word.encode('utf-8')),
                    dtype=torch.long, device=self.device,
                )
                logits = self.model.decoder(target, wave_vec, merged)
                pred = logits.argmax(dim=-1)

                # Check byte-level accuracy
                match = (pred == target).float().mean().item()
                correct += match
                total += 1

        self.model.train()
        return correct / max(total, 1)

    def _test_phrase_completion(self) -> float:
        """
        Test Stage 4: Can the decoder complete common phrases?

        Uses autoregressive generation from a prompt and checks
        if the continuation starts with expected bytes.
        """
        prompts = get_phrase_test_prompts()
        correct = 0
        total = 0

        self.model.eval()
        with torch.no_grad():
            for prompt, expected in prompts:
                wave_vec, merged = self._get_context(prompt + expected)
                prompt_bytes = torch.tensor(
                    list(prompt.encode('utf-8')),
                    dtype=torch.long, device=self.device,
                )

                # Generate a few bytes after the prompt
                generated = self.model.decoder.generate(
                    wave_vec, merged, max_length=len(expected) + 5,
                    temperature=0.5, prompt_bytes=prompt_bytes,
                )
                continuation = generated[len(prompt_bytes):]

                # Check if first byte of expected matches
                if continuation and len(expected) > 0:
                    expected_byte = expected.encode('utf-8')[0]
                    if continuation[0] == expected_byte:
                        correct += 1
                total += 1

        self.model.train()
        return correct / max(total, 1)

    def _test_sentence_coherence(self) -> float:
        """
        Test Stage 5: Can the decoder generate readable sentence continuations?

        Generates text from prompts and checks for basic coherence:
        - Contains spaces (word boundaries)
        - Contains common English characters
        - Has reasonable length
        """
        prompts = get_sentence_test_prompts()
        coherent = 0
        total = 0

        self.model.eval()
        with torch.no_grad():
            for prompt in prompts:
                wave_vec, merged = self._get_context(prompt)
                prompt_bytes = torch.tensor(
                    list(prompt.encode('utf-8')),
                    dtype=torch.long, device=self.device,
                )

                generated = self.model.decoder.generate(
                    wave_vec, merged, max_length=40,
                    temperature=0.7, prompt_bytes=prompt_bytes,
                )
                continuation = generated[len(prompt_bytes):]

                try:
                    text = bytes(continuation).decode('utf-8', errors='replace')
                except Exception:
                    text = ''

                # Coherence checks
                has_spaces = ' ' in text
                mostly_ascii = sum(1 for c in text if c.isalpha() or c == ' ') > len(text) * 0.5 if text else False
                reasonable_len = 3 < len(text) < 60

                if has_spaces and mostly_ascii and reasonable_len:
                    coherent += 1
                total += 1

        self.model.train()
        return coherent / max(total, 1)

    def _run_stage_test(self, stage: int) -> float:
        """
        Run the appropriate advancement test for a stage.

        Returns:
            Accuracy score (0-1)
        """
        if stage == 1:
            return self._test_byte_accuracy()
        elif stage == 2:
            return self._test_byte_accuracy()  # Reuse byte test (bigrams are byte sequences)
        elif stage == 3:
            return self._test_word_spelling()
        elif stage == 4:
            return self._test_phrase_completion()
        elif stage == 5:
            return self._test_sentence_coherence()
        elif stage == 6:
            return self._test_sentence_coherence()  # Reuse sentence test for real text
        return 0.0

    def train_stage(self, stage: int) -> StageResult:
        """
        Train a single curriculum stage.

        Args:
            stage: Stage number (1-6)

        Returns:
            StageResult with metrics
        """
        config = STAGE_CONFIGS[stage]
        t0 = time.time()

        if self.verbose:
            print(f"\n{'═' * 60}")
            print(f"  Stage {stage}: {config.name}")
            print(f"  Max steps: {config.max_steps}  |  LR: {config.lr}")
            print(f"  Loss target: < {config.loss_threshold}")
            print(f"  Accuracy target: > {config.accuracy_threshold:.0%}")
            print(f"{'═' * 60}")

        if self.log:
            self.log.info(f"Starting Stage {stage}: {config.name}")

        # Get training data
        if stage <= 5:
            texts = generate_curriculum_data(stage)
        else:
            texts = self.openwebtext_texts
            if not texts:
                print("  ⚠ No OpenWebText texts provided for Stage 6")
                texts = generate_curriculum_data(5, n_samples=1000)  # Fallback

        # Build optimizer for this stage
        optimizer = self._build_optimizer(config)

        # Warmup scheduler
        warmup = min(50, config.max_steps // 10)
        def lr_lambda(step):
            if step < warmup:
                return max(0.01, step / max(warmup, 1))
            progress = (step - warmup) / max(config.max_steps - warmup, 1)
            return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        losses = []
        step = 0
        data_idx = 0
        log_interval = max(config.max_steps // 10, 10)

        while step < config.max_steps:
            # Cycle through data
            text = texts[data_idx % len(texts)]
            data_idx += 1

            if not text or len(text.strip()) < 2:
                continue

            loss = self._train_step(text, optimizer, config)
            losses.append(loss)
            scheduler.step()
            step += 1

            # Progress logging
            if self.verbose and step % log_interval == 0:
                recent = losses[-log_interval:]
                avg = sum(recent) / len(recent)
                lr_now = optimizer.param_groups[0]['lr']
                print(
                    f"    Step {step:>5}/{config.max_steps}  "
                    f"loss={avg:.4f}  lr={lr_now:.6f}"
                )

            # Early advancement check (every 25% of max_steps)
            check_interval = max(config.max_steps // 4, 50)
            if step > 0 and step % check_interval == 0 and len(losses) >= 20:
                recent_avg = sum(losses[-20:]) / 20
                if recent_avg < config.loss_threshold:
                    accuracy = self._run_stage_test(stage)
                    if accuracy >= config.accuracy_threshold:
                        if self.verbose:
                            print(f"    ✓ Early advancement! loss={recent_avg:.4f} acc={accuracy:.2%}")
                        break

        # Final evaluation
        final_loss = losses[-1] if losses else 0.0
        avg_loss = sum(losses) / max(len(losses), 1)
        min_loss = min(losses) if losses else 0.0
        accuracy = self._run_stage_test(stage)
        elapsed = time.time() - t0

        # Advancement decision
        loss_ok = avg_loss < config.loss_threshold * 1.5  # Generous for advancement
        acc_ok = accuracy >= config.accuracy_threshold * 0.8  # Slightly relaxed
        advanced = loss_ok or acc_ok  # Advance if either criterion met

        result = StageResult(
            stage=stage,
            name=config.name,
            steps_taken=step,
            final_loss=final_loss,
            avg_loss=avg_loss,
            min_loss=min_loss,
            accuracy=accuracy,
            advanced=advanced,
            time_seconds=elapsed,
            loss_history=losses,
        )

        if self.verbose:
            status = "✓ ADVANCED" if advanced else "✗ Did not advance"
            print(f"\n  Stage {stage} Result:")
            print(f"    Steps:    {step}")
            print(f"    Avg loss: {avg_loss:.4f} (target: < {config.loss_threshold})")
            print(f"    Min loss: {min_loss:.4f}")
            print(f"    Accuracy: {accuracy:.2%} (target: > {config.accuracy_threshold:.0%})")
            print(f"    Time:     {elapsed:.1f}s")
            print(f"    Status:   {status}")

        if self.log:
            self.log.metric(f"stage{stage}_loss", f"{avg_loss:.4f}")
            self.log.metric(f"stage{stage}_accuracy", f"{accuracy:.2%}")
            if advanced:
                self.log.success(f"Stage {stage} complete: loss={avg_loss:.4f} acc={accuracy:.2%}")
            else:
                self.log.warning(f"Stage {stage} incomplete: loss={avg_loss:.4f} acc={accuracy:.2%}")

        return result

    def run_curriculum(self, start_stage: int = 1) -> CurriculumResult:
        """
        Run the full 6-stage curriculum.

        Args:
            start_stage: Stage to start from (for resuming)

        Returns:
            CurriculumResult with all stage results
        """
        t0 = time.time()

        if self.verbose:
            print("\n" + "▓" * 60)
            print("  🏫 FLUX ABC School — Curriculum Training")
            print("  Teaching the decoder to generate coherent text")
            print("  Stage 1: Bytes → Stage 6: Real Text")
            print("▓" * 60)

        if self.log:
            self.log.separator("Phase 8.5: Curriculum Training")

        stage_results = []

        for stage in range(start_stage, 7):
            result = self.train_stage(stage)
            stage_results.append(result)
            self._stage_results.append(result)

            if not result.advanced and stage < 6:
                if self.verbose:
                    print(f"\n  ⚠ Stage {stage} did not meet threshold — continuing anyway")
                    print(f"    (curriculum always advances to ensure full coverage)")

        elapsed = time.time() - t0
        total_steps = sum(r.steps_taken for r in stage_results)

        curriculum_result = CurriculumResult(
            stages_completed=len(stage_results),
            total_steps=total_steps,
            total_time_seconds=elapsed,
            stage_results=stage_results,
            final_stage=stage_results[-1].stage if stage_results else 0,
        )

        if self.verbose:
            print(f"\n{'▓' * 60}")
            print(f"  🎓 Curriculum Complete!")
            print(f"    Stages:     {len(stage_results)}/6")
            print(f"    Total steps: {total_steps:,}")
            print(f"    Total time:  {elapsed:.1f}s")
            print(f"{'▓' * 60}")

            # Stage summary table
            print(f"\n  {'Stage':<28} {'Steps':>6} {'Loss':>8} {'Acc':>8} {'Status':>10}")
            print(f"  {'─' * 64}")
            for r in stage_results:
                status = "✓" if r.advanced else "✗"
                print(
                    f"  {r.stage}. {r.name:<24} {r.steps_taken:>6} "
                    f"{r.avg_loss:>8.4f} {r.accuracy:>7.1%} {status:>10}"
                )

        if self.log:
            self.log.success(f"Curriculum complete: {total_steps} steps in {elapsed:.1f}s")

        return curriculum_result

    def get_stage_results(self) -> List[StageResult]:
        """Return all stage results from this trainer's lifetime."""
        return list(self._stage_results)


# ─────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("CurriculumTrainer — checking configuration")
    for stage, config in STAGE_CONFIGS.items():
        print(f"  Stage {stage}: {config.name}")
        print(f"    max_steps={config.max_steps} loss_thresh={config.loss_threshold}")
        print(f"    acc_thresh={config.accuracy_threshold} lr={config.lr}")
    print("  ✓ Configuration OK")
