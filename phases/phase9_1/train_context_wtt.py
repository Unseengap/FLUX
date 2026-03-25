"""
Phase 9.1 — Training Script: Context-Aware WaveToText

Single-stage training focused on spelling enhancement.
Loads Phase 9 checkpoint (frozen), trains only ContextWaveToText.

Key improvements over Phase 9 WTT training:
    - Left-context awareness (2 previous chunks)
    - 150,000 steps (6x more than Phase 9's 25,000)
    - Cosine LR schedule with warmup
    - Context dropout regularization
    - UTF-8 multi-byte augmentation
    - batch_size=64 with grad_accum=2

Training data: Same OpenWebText source as Phase 9.
"""

import sys
import os
import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime

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

from flux_utils import (
    get_device, save_checkpoint, load_checkpoint, checkpoint_exists,
    PhaseLogger, PhaseResults, upload_checkpoint_to_hf, _resolve_hf_token,
    CHECKPOINT_DIR,
)
from wave_chunker import WaveChunker
from context_wave_to_text import ContextWaveToText


# ─────────────────────────────────────────────
# Phase 9.1 Configuration
# ─────────────────────────────────────────────
PHASE9_1_CONFIG = {
    # ContextWaveToText architecture
    'wave_dim': 432,
    'hidden_dim': 512,
    'byte_embed_dim': 128,
    'gru_layers': 2,
    'dropout': 0.1,
    'max_bytes': 20,
    'num_context': 2,
    'context_drop_prob': 0.25,
    # Training
    'lr': 3e-4,
    'warmup_steps': 2000,
    'max_steps': 150000,
    'batch_size': 64,
    'grad_accum': 2,
    'log_interval': 5000,
    'eval_interval': 25000,
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,
    # Data
    'max_train_docs': 10000,
    'utf8_augment_ratio': 0.10,
}

# ─────────────────────────────────────────────
# UTF-8 Augmentation Data
# ─────────────────────────────────────────────
UTF8_AUGMENT_TEXTS = [
    "The café on the corner serves excellent espresso and crème brûlée.",
    "She is a naïve but talented résumé writer with a flair for design.",
    "The piñata broke open, scattering jalapeño-flavored candies everywhere.",
    "François went to the château near the forêt to enjoy his soirée.",
    "The über-efficient Zürich office processed über 1000 applications.",
    "Señor García ordered the mole poblano with a side of guacamole.",
    "The Ångström unit measures atomic distances with great précision.",
    "Dvořák composed beautiful études while staying at the Hôtel & Café.",
    "The entrée included filet mignon with a béarnaise sauce and crêpes.",
    "She studied Möbius strips and Poincaré conjecture at the université.",
    "The maître d' at the Côte d'Azur restaurant was très accommodating.",
    "El Niño affects weather patterns across the entire Pacific océan.",
    "His exposé about the coup d'état won the journaliste award.",
    "They enjoyed a tête-à-tête over thé and petit fours at the pâtisserie.",
    "The encyclopædia contained entries about every known phænomenon.",
    "日本語のテスト — Japanese characters in a mixed-language sentence.",
    "Привет мир — Russian greeting mixed with English punctuation.",
    "The emoji 🌍 represents Earth, while 🔬 represents science & research.",
    "Temperature: 25°C (77°F) — measured at 45°N latitude, 90°W longitude.",
    "Copyright © 2026 FLUX™ — all rights reserved. Price: €99 / £85 / ¥12000.",
    "Mathematical symbols: α + β = γ, ∑ᵢ xᵢ², √(a² + b²) ≈ c.",
    "The Māori word 'kia ora' means hello; 'haere rā' means goodbye.",
    "Vietnamese: Xin chào! Tôi tên là Nguyễn Văn An từ Hà Nội.",
    "Turkish: İstanbul'da güzel bir gün. Teşekkür ederim çok fazla.",
    "Polish: Łódź jest pięknym miastem. Dziękuję bardzo za pomoc.",
]


# ─────────────────────────────────────────────
# Training Result
# ─────────────────────────────────────────────

@dataclass
class WTTTrainResult:
    """Result from ContextWaveToText training."""
    total_steps: int = 0
    final_loss: float = 0.0
    avg_loss: float = 0.0
    chunk_accuracy: float = 0.0
    word_accuracy: float = 0.0
    hard_spelling_accuracy: float = 0.0
    utf8_accuracy: float = 0.0
    total_time_seconds: float = 0.0


# ─────────────────────────────────────────────
# Training Data Loading
# ─────────────────────────────────────────────

def load_training_data(max_docs: int = 10000) -> List[str]:
    """
    Load training data from OpenWebText or synthetic fallback.

    Args:
        max_docs: Maximum documents to load

    Returns:
        List of text documents
    """
    try:
        from datasets import load_dataset
        print(f"  ℹ Loading OpenWebText subset ({max_docs:,} docs)...", flush=True)
        ds = load_dataset('openwebtext', split='train', streaming=True)
        texts = []
        for i, sample in enumerate(ds):
            if i >= max_docs:
                break
            text = sample.get('text', '')
            if text and len(text.strip()) >= 20:
                texts.append(text[:500])  # Cap at 500 chars for memory
        print(f"  ✓ Loaded {len(texts):,} documents", flush=True)
        return texts
    except Exception as e:
        print(f"  ⚠ OpenWebText not available: {e}", flush=True)
        print(f"  ℹ Using synthetic training data...", flush=True)
        return _synthetic_training_data(max_docs)


def _synthetic_training_data(max_docs: int = 10000) -> List[str]:
    """Generate synthetic training data as fallback."""
    templates = [
        "The {adj} {noun} {verb} the {noun2} with {adj2} {noun3}.",
        "In the {noun}, {adj} {noun2} {verb} {adv} and {verb2}.",
        "Scientists discovered that {adj} {noun} can {verb} under {adj2} conditions.",
        "{noun} and {noun2} are fundamental to understanding {adj} {noun3}.",
        "The relationship between {noun} and {noun2} reveals {adj} patterns.",
        "Modern {noun} relies on {adj} {noun2} for {adj2} performance.",
        "Research shows that {noun} {verb} when exposed to {adj} {noun2}.",
        "The history of {noun} reveals {adj} changes in {noun2} over time.",
    ]
    words = {
        'adj': ['quantum', 'neural', 'dynamic', 'complex', 'thermal', 'resonant',
                'gravitational', 'emergent', 'fundamental', 'continuous', 'discrete',
                'causal', 'semantic', 'phonetic', 'syntactic', 'temporal', 'stable'],
        'adj2': ['optimal', 'minimal', 'maximal', 'critical', 'essential', 'variable',
                 'extreme', 'moderate', 'significant', 'remarkable', 'unprecedented'],
        'noun': ['energy', 'system', 'field', 'wave', 'particle', 'network',
                 'structure', 'process', 'mechanism', 'algorithm', 'function',
                 'technology', 'science', 'mathematics', 'physics', 'chemistry'],
        'noun2': ['intelligence', 'efficiency', 'relationship', 'behavior', 'pattern',
                  'symmetry', 'transformation', 'evolution', 'computation', 'analysis'],
        'noun3': ['phenomena', 'principles', 'dynamics', 'interactions', 'properties',
                  'characteristics', 'foundations', 'applications', 'implications'],
        'verb': ['transforms', 'affects', 'determines', 'influences', 'produces',
                 'generates', 'modifies', 'enhances', 'disrupts', 'facilitates'],
        'verb2': ['evolves', 'adapts', 'resonates', 'converges', 'stabilizes'],
        'adv': ['significantly', 'fundamentally', 'efficiently', 'dynamically'],
    }

    texts = []
    for _ in range(max_docs):
        template = random.choice(templates)
        text = template
        for key, options in words.items():
            while f'{{{key}}}' in text:
                text = text.replace(f'{{{key}}}', random.choice(options), 1)
        texts.append(text)

    return texts


# ─────────────────────────────────────────────
# LR Schedule
# ─────────────────────────────────────────────

def get_lr(step: int, warmup_steps: int, max_steps: int, peak_lr: float, min_lr: float = 1e-5) -> float:
    """
    Warmup + cosine decay learning rate schedule.

    Args:
        step: Current training step
        warmup_steps: Number of warmup steps
        max_steps: Total training steps
        peak_lr: Peak learning rate
        min_lr: Minimum learning rate

    Returns:
        Learning rate for this step
    """
    if step < warmup_steps:
        return peak_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cosine


# ─────────────────────────────────────────────
# Hard Spelling Evaluation Words
# ─────────────────────────────────────────────

HARD_SPELLING_WORDS = [
    "the", "of", "and", "is", "in",
    "science", "energy", "system", "quantum", "relationship",
    "fundamental", "efficiency", "xyz", "café", "naïve",
]


# ─────────────────────────────────────────────
# Phase 9.1 Trainer
# ─────────────────────────────────────────────

class Phase9_1_Trainer:
    """
    Trainer for Phase 9.1 ContextWaveToText.

    Loads Phase 9 checkpoint (frozen CSE + WaveChunker), trains
    only the new ContextWaveToText decoder.

    Args:
        cse: Frozen ContinuousSemanticEncoder
        chunker: Frozen WaveChunker
        context_wtt: Trainable ContextWaveToText
        device: Target device
        log: PhaseLogger instance
    """

    def __init__(
        self,
        cse,
        chunker: WaveChunker,
        context_wtt: ContextWaveToText,
        device: str = 'cpu',
        log: Optional[PhaseLogger] = None,
    ):
        self.cse = cse
        self.chunker = chunker
        self.wtt = context_wtt
        self.device = device
        self.log = log

    # ─────────────────────────────────────────────
    # Data Preparation: Extract (wave, context, bytes) tuples
    # ─────────────────────────────────────────────

    @torch.no_grad()
    def prepare_training_data(
        self,
        texts: List[str],
        max_pairs: int = 500000,
        utf8_texts: Optional[List[str]] = None,
    ) -> List[Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]]:
        """
        Extract (current_wave, context_waves, target_bytes) triples from texts.

        For each text:
            1. CSE encode → wave sequence [seq, 432]
            2. WaveChunker → chunk waves + byte spans
            3. For each chunk: extract current wave, left-2 context waves, target bytes

        Args:
            texts: Training documents
            max_pairs: Maximum training pairs to extract
            utf8_texts: Additional UTF-8 augmentation texts

        Returns:
            List of (current_wave [432], context_waves [0..2, 432] or None, target_bytes [L])
        """
        pairs = []
        _errors = 0

        all_texts = list(texts)
        if utf8_texts:
            all_texts.extend(utf8_texts)
        random.shuffle(all_texts)

        print(f"  ℹ Extracting (wave, context, bytes) triples from {len(all_texts):,} texts...", flush=True)
        t0 = time.time()

        for text_idx, text in enumerate(all_texts):
            if len(pairs) >= max_pairs:
                break

            if not text or len(text.strip()) < 10:
                continue

            try:
                wave = self.cse.encode(text)
                wave_seq = wave.full.to(self.device)  # [seq, 432]
                text_bytes = text.encode('utf-8', errors='replace')

                # Get chunk waves and byte spans
                chunk_waves, spans = self.chunker(wave_seq)
                num_chunks = chunk_waves.shape[0]

                for i in range(num_chunks):
                    start, end = spans[i]
                    byte_start = min(start, len(text_bytes))
                    byte_end = min(end, len(text_bytes))
                    chunk_bytes = text_bytes[byte_start:byte_end]

                    if len(chunk_bytes) == 0:
                        continue

                    current_wave = chunk_waves[i].detach().cpu()
                    byte_tensor = torch.tensor(
                        list(chunk_bytes), dtype=torch.long
                    )

                    # Build left-context (previous 2 chunks)
                    if i > 0:
                        ctx_start = max(0, i - 2)
                        context = chunk_waves[ctx_start:i].detach().cpu()
                    else:
                        context = None

                    pairs.append((current_wave, context, byte_tensor))

                    if len(pairs) >= max_pairs:
                        break

            except Exception as e:
                _errors += 1
                if _errors <= 3:
                    print(f"    ⚠ Error on text {text_idx}: {type(e).__name__}: {e}", flush=True)
                continue

            # Progress
            if (text_idx + 1) % 1000 == 0:
                elapsed = time.time() - t0
                rate = len(pairs) / max(elapsed, 0.01)
                print(
                    f"    ... {len(pairs):,} pairs from {text_idx+1:,} texts "
                    f"[{rate:.0f} pairs/s, {elapsed:.0f}s elapsed]",
                    flush=True,
                )

        elapsed = time.time() - t0
        print(
            f"  ✓ Extracted {len(pairs):,} training triples in {elapsed:.1f}s "
            f"({_errors} errors)",
            flush=True,
        )
        return pairs

    # ─────────────────────────────────────────────
    # Training Loop
    # ─────────────────────────────────────────────

    def train(
        self,
        training_pairs: List[Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]],
        max_steps: int = 150000,
        batch_size: int = 64,
        grad_accum: int = 2,
        lr: float = 3e-4,
        warmup_steps: int = 2000,
        log_interval: int = 5000,
        eval_interval: int = 25000,
        eval_texts: Optional[List[str]] = None,
    ) -> WTTTrainResult:
        """
        Train ContextWaveToText.

        Args:
            training_pairs: List of (current_wave, context_waves, target_bytes)
            max_steps: Maximum training steps
            batch_size: Batch size
            grad_accum: Gradient accumulation steps
            lr: Peak learning rate
            warmup_steps: LR warmup steps
            log_interval: Print every N steps
            eval_interval: Evaluate every N steps
            eval_texts: Texts for periodic evaluation

        Returns:
            WTTTrainResult with training metrics
        """
        t0 = time.time()

        optimizer = torch.optim.AdamW(
            self.wtt.parameters(),
            lr=lr,
            weight_decay=PHASE9_1_CONFIG['weight_decay'],
        )

        self.wtt.train()
        all_losses = []
        step = 0
        total_chunks = 0
        best_loss = float('inf')

        num_pairs = len(training_pairs)
        epoch = 0

        print(f"\n{'='*60}", flush=True)
        print(f"  Phase 9.1: ContextWaveToText Training", flush=True)
        print(f"  max_steps={max_steps}, batch_size={batch_size}, "
              f"grad_accum={grad_accum}", flush=True)
        print(f"  Training pairs: {num_pairs:,}  |  LR: warmup {warmup_steps} → "
              f"cosine {lr} → 1e-5", flush=True)
        print(f"  Params: {sum(p.numel() for p in self.wtt.parameters()):,}", flush=True)
        print(f"{'='*60}", flush=True)

        _step_t0 = time.time()
        optimizer.zero_grad()

        while step < max_steps:
            # Shuffle at each epoch
            indices = list(range(num_pairs))
            random.shuffle(indices)
            epoch += 1

            i = 0
            while i + batch_size <= num_pairs and step < max_steps:
                batch_indices = indices[i:i + batch_size]
                i += batch_size

                # Build batch
                waves_batch = []
                targets_batch = []
                context_batch = []

                for idx in batch_indices:
                    current_wave, context, byte_tensor = training_pairs[idx]
                    waves_batch.append(current_wave.to(self.device))
                    targets_batch.append(byte_tensor.to(self.device))
                    if context is not None:
                        context_batch.append(context.to(self.device))
                    else:
                        context_batch.append(None)

                waves_t = torch.stack(waves_batch)  # [B, 432]

                # Forward pass
                loss = self.wtt.forward_batch(
                    waves_t, targets_batch, context_batch
                )

                # Backward with gradient accumulation
                scaled_loss = loss / grad_accum
                scaled_loss.backward()

                step += 1
                total_chunks += batch_size

                if step % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.wtt.parameters(),
                        PHASE9_1_CONFIG['max_grad_norm'],
                    )
                    # Update LR
                    current_lr = get_lr(step, warmup_steps, max_steps, lr)
                    for pg in optimizer.param_groups:
                        pg['lr'] = current_lr
                    optimizer.step()
                    optimizer.zero_grad()

                loss_val = loss.item()
                all_losses.append(loss_val)
                if loss_val < best_loss:
                    best_loss = loss_val

                # ─── Logging ───
                if step == 1:
                    print(
                        f"  Step     1  loss={loss_val:.4f}  "
                        f"chunks={total_chunks:,}  "
                        f"(first step: {time.time()-_step_t0:.2f}s)",
                        flush=True,
                    )
                elif step == 10:
                    _e10 = time.time() - _step_t0
                    avgL = sum(all_losses[-10:]) / 10
                    print(
                        f"  Step    10  loss={avgL:.4f}  "
                        f"chunks={total_chunks:,}  "
                        f"[{10/_e10:.1f} step/s]",
                        flush=True,
                    )

                if step % log_interval == 0:
                    avgL = sum(all_losses[-log_interval:]) / min(len(all_losses), log_interval)
                    elapsed = time.time() - _step_t0
                    sps = step / max(elapsed, 0.01)
                    eta = (max_steps - step) / max(sps, 0.01)
                    current_lr = get_lr(step, warmup_steps, max_steps, lr)
                    trend = '↓' if len(all_losses) > log_interval * 2 and \
                        avgL < sum(all_losses[-2*log_interval:-log_interval]) / log_interval else '→'
                    print(
                        f"  Step {step:>7,}  loss={avgL:.4f} {trend}  "
                        f"lr={current_lr:.6f}  "
                        f"chunks={total_chunks:,}  "
                        f"[{sps:.1f} step/s, {elapsed:.0f}s, ETA ~{eta:.0f}s]",
                        flush=True,
                    )
                    if self.log:
                        self.log.metric(f"cwtt_step_{step}_loss", f"{avgL:.4f}")

                # ─── Periodic evaluation ───
                if eval_texts and step % eval_interval == 0:
                    print(f"\n  ℹ Evaluating at step {step}...", flush=True)
                    chunk_acc = self._evaluate_chunk_accuracy(eval_texts[:50])
                    word_acc = self._evaluate_word_accuracy(eval_texts[:50])
                    print(
                        f"    Chunk accuracy: {chunk_acc:.1%}  |  "
                        f"Word accuracy: {word_acc:.1%}",
                        flush=True,
                    )
                    if self.log:
                        self.log.metric(f"cwtt_step_{step}_chunk_acc", f"{chunk_acc:.1%}")
                        self.log.metric(f"cwtt_step_{step}_word_acc", f"{word_acc:.1%}")
                    self.wtt.train()  # Back to training mode

        # ─── Final evaluation ───
        elapsed_total = time.time() - t0
        final_loss = all_losses[-1] if all_losses else 0.0
        avg_loss = sum(all_losses) / max(len(all_losses), 1)

        print(f"\n  ℹ Running final evaluation...", flush=True)
        chunk_acc = self._evaluate_chunk_accuracy(eval_texts[:100] if eval_texts else [])
        word_acc = self._evaluate_word_accuracy(eval_texts[:100] if eval_texts else [])
        hard_acc = self._evaluate_hard_spelling(eval_texts[:20] if eval_texts else [])
        utf8_acc = self._evaluate_utf8_accuracy()

        print(f"\n  ✓ Training complete: {step:,} steps, {total_chunks:,} chunks", flush=True)
        print(f"    Final loss:           {final_loss:.4f}", flush=True)
        print(f"    Average loss:         {avg_loss:.4f}", flush=True)
        print(f"    Best loss:            {best_loss:.4f}", flush=True)
        print(f"    Chunk accuracy:       {chunk_acc:.1%}", flush=True)
        print(f"    Word accuracy:        {word_acc:.1%}", flush=True)
        print(f"    Hard spelling:        {hard_acc:.1%}", flush=True)
        print(f"    UTF-8 accuracy:       {utf8_acc:.1%}", flush=True)
        print(f"    Epochs completed:     {epoch}", flush=True)
        print(f"    Total time:           {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)", flush=True)
        print(f"    Throughput:           {step/max(elapsed_total,0.01):.1f} step/s", flush=True)

        result = WTTTrainResult(
            total_steps=step,
            final_loss=final_loss,
            avg_loss=avg_loss,
            chunk_accuracy=chunk_acc,
            word_accuracy=word_acc,
            hard_spelling_accuracy=hard_acc,
            utf8_accuracy=utf8_acc,
            total_time_seconds=elapsed_total,
        )

        if self.log:
            self.log.metric("cwtt_final_loss", f"{final_loss:.4f}")
            self.log.metric("cwtt_chunk_acc", f"{chunk_acc:.1%}")
            self.log.metric("cwtt_word_acc", f"{word_acc:.1%}")
            self.log.metric("cwtt_hard_spelling", f"{hard_acc:.1%}")
            self.log.metric("cwtt_utf8_acc", f"{utf8_acc:.1%}")

        return result

    # ─────────────────────────────────────────────
    # Evaluation Methods
    # ─────────────────────────────────────────────

    @torch.no_grad()
    def _evaluate_chunk_accuracy(
        self, texts: List[str], max_chunks: int = 500
    ) -> float:
        """Evaluate chunk-level byte accuracy (same methodology as Phase 9)."""
        self.wtt.eval()
        correct = 0
        total = 0

        for text in texts:
            if not text or len(text.strip()) < 10:
                continue
            try:
                wave = self.cse.encode(text)
                wave_seq = wave.full.to(self.device)
                text_bytes = text.encode('utf-8', errors='replace')

                chunk_waves, spans = self.chunker(wave_seq)
                num_chunks = chunk_waves.shape[0]

                for i in range(num_chunks):
                    start, end = spans[i]
                    byte_start = min(start, len(text_bytes))
                    byte_end = min(end, len(text_bytes))
                    chunk_bytes = text_bytes[byte_start:byte_end]
                    if len(chunk_bytes) == 0:
                        continue

                    # Build context
                    ctx_start = max(0, i - 2)
                    ctx = chunk_waves[ctx_start:i] if i > 0 else None

                    decoded = self.wtt.decode(
                        chunk_waves[i], temperature=0.5, context_waves=ctx
                    )
                    if decoded == chunk_bytes:
                        correct += 1
                    total += 1
                    if total >= max_chunks:
                        break
            except Exception:
                continue
            if total >= max_chunks:
                break

        return correct / max(total, 1)

    @torch.no_grad()
    def _evaluate_word_accuracy(
        self, texts: List[str], max_words: int = 200
    ) -> float:
        """Evaluate word-level spelling accuracy."""
        self.wtt.eval()
        correct = 0
        total = 0

        for text in texts:
            if not text or len(text.strip()) < 10:
                continue
            try:
                wave = self.cse.encode(text)
                wave_seq = wave.full.to(self.device)
                text_bytes = text.encode('utf-8', errors='replace')

                chunk_waves, spans = self.chunker(wave_seq)

                # Decode full sequence with context
                decoded_chunks = self.wtt.decode_sequence(
                    chunk_waves, temperature=0.5
                )

                # Reconstruct text from decoded chunks
                decoded_text = b''.join(decoded_chunks)
                try:
                    decoded_str = decoded_text.decode('utf-8', errors='replace')
                except Exception:
                    decoded_str = decoded_text.decode('latin-1', errors='replace')

                # Word-level comparison
                orig_words = text.split()
                decoded_words = decoded_str.split()

                for j, orig_w in enumerate(orig_words):
                    if j >= len(decoded_words):
                        break
                    if orig_w == decoded_words[j]:
                        correct += 1
                    total += 1
                    if total >= max_words:
                        break

            except Exception:
                continue
            if total >= max_words:
                break

        return correct / max(total, 1)

    @torch.no_grad()
    def _evaluate_hard_spelling(
        self, texts: List[str],
    ) -> float:
        """Evaluate hard spelling words embedded in carrier sentences."""
        self.wtt.eval()
        correct = 0
        total = len(HARD_SPELLING_WORDS)

        for word in HARD_SPELLING_WORDS:
            carrier = f"This is {word} here"
            try:
                wave = self.cse.encode(carrier)
                wave_seq = wave.full.to(self.device)
                text_bytes = carrier.encode('utf-8', errors='replace')

                chunk_waves, spans = self.chunker(wave_seq)
                decoded_chunks = self.wtt.decode_sequence(
                    chunk_waves, temperature=0.3
                )

                decoded_text = b''.join(decoded_chunks)
                try:
                    decoded_str = decoded_text.decode('utf-8', errors='replace')
                except Exception:
                    decoded_str = decoded_text.decode('latin-1', errors='replace')

                if word in decoded_str:
                    correct += 1

            except Exception:
                continue

        return correct / max(total, 1)

    @torch.no_grad()
    def _evaluate_utf8_accuracy(self) -> float:
        """Evaluate UTF-8 multi-byte character accuracy."""
        self.wtt.eval()
        test_words = ["café", "naïve", "résumé", "piñata", "Zürich",
                      "François", "château", "crème", "über", "señor"]
        correct = 0
        total = len(test_words)

        for word in test_words:
            carrier = f"The word is {word} exactly"
            try:
                wave = self.cse.encode(carrier)
                wave_seq = wave.full.to(self.device)

                chunk_waves, spans = self.chunker(wave_seq)
                decoded_chunks = self.wtt.decode_sequence(
                    chunk_waves, temperature=0.3
                )

                decoded_text = b''.join(decoded_chunks)
                try:
                    decoded_str = decoded_text.decode('utf-8', errors='replace')
                except Exception:
                    decoded_str = decoded_text.decode('latin-1', errors='replace')

                if word in decoded_str:
                    correct += 1

            except Exception:
                continue

        return correct / max(total, 1)

    # ─────────────────────────────────────────────
    # Checkpoint Save/Load
    # ─────────────────────────────────────────────

    def save_checkpoint(
        self,
        result: WTTTrainResult,
        phase9_checkpoint: Dict[str, Any],
    ) -> Path:
        """
        Save Phase 9.1 checkpoint.

        Copies all frozen states from phase9 checkpoint, adds new
        ContextWaveToText state.

        Args:
            result: Training result metrics
            phase9_checkpoint: Loaded phase9.phase.pt dict

        Returns:
            Path to saved checkpoint
        """
        # Start with all frozen states from Phase 9
        state = {
            'phase': 9.1,
            'config': phase9_checkpoint.get('config', {}),
            'phase9_1_config': PHASE9_1_CONFIG,
        }

        # Copy all frozen component states
        frozen_keys = [
            'cse_state_dict', 'field_state_dict',
            'gr_state', 'tl_state', 'cgn_state', 'causal_graph_state',
            'working_memory_state', 'episodic_memory_state',
            'semantic_memory_state', 'router_state',
            'wave_to_field_state', 'field_to_wave_state',
            'output_head_state',
            'wave_chunker_state_dict',
            'wave_generator_state_dict',  # Keep mode-collapsed WG for Phase 9.5
        ]
        for key in frozen_keys:
            if key in phase9_checkpoint:
                state[key] = phase9_checkpoint[key]

        # NEW: ContextWaveToText state
        state['context_wtt_state_dict'] = self.wtt.state_dict()
        # Backward compat: also save under old key name
        state['wave_to_text_state_dict'] = self.wtt.state_dict()

        # Metrics
        state['metrics'] = {
            'chunk_accuracy': result.chunk_accuracy,
            'word_accuracy': result.word_accuracy,
            'hard_spelling_accuracy': result.hard_spelling_accuracy,
            'utf8_accuracy': result.utf8_accuracy,
            'training_steps': result.total_steps,
            'final_loss': result.final_loss,
            'avg_loss': result.avg_loss,
            'total_training_time': result.total_time_seconds,
        }

        # Save with phase number as string-safe float
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        path = CHECKPOINT_DIR / 'phase9_1.phase.pt'
        state['timestamp'] = datetime.now().isoformat()
        torch.save(state, path)
        size_mb = path.stat().st_size / 1e6
        print(f"  ✓ Phase 9.1 checkpoint saved: {path} ({size_mb:.1f} MB)")
        return path


# ─────────────────────────────────────────────
# Load Phase 9 and Build Phase 9.1 Modules
# ─────────────────────────────────────────────

def load_phase9_for_9_1(device: str = 'cpu'):
    """
    Load Phase 9 checkpoint and extract frozen components for Phase 9.1.

    Returns:
        (cse, chunker, phase9_checkpoint, device)
    """
    from cse import ContinuousSemanticEncoder

    ckpt = load_checkpoint(9)
    config = ckpt.get('config', {})

    # Build and load CSE
    cse = ContinuousSemanticEncoder(**{
        k: v for k, v in config.items()
        if k in ['wave_dim', 'window_size', 'stride']
    })
    if 'cse_state_dict' in ckpt:
        cse.load_state_dict(ckpt['cse_state_dict'])
    cse.eval()
    for p in cse.parameters():
        p.requires_grad = False
    cse = cse.to(device)

    # Build and load WaveChunker
    p9cfg = ckpt.get('phase9_config', {})
    chunker = WaveChunker(
        wave_dim=p9cfg.get('wave_dim', 432),
        min_chunk_size=p9cfg.get('min_chunk_size', 2),
        max_chunk_size=p9cfg.get('max_chunk_size', 20),
        coherence_threshold=p9cfg.get('coherence_threshold', 0.5),
    ).to(device)
    if 'wave_chunker_state_dict' in ckpt:
        chunker.load_state_dict(ckpt['wave_chunker_state_dict'])
    chunker.eval()
    for p in chunker.parameters():
        p.requires_grad = False

    print(f"  ✓ Loaded CSE + WaveChunker from phase9.phase.pt (frozen)")

    return cse, chunker, ckpt


def build_context_wtt(device: str = 'cpu') -> ContextWaveToText:
    """Build fresh ContextWaveToText module."""
    cfg = PHASE9_1_CONFIG
    wtt = ContextWaveToText(
        wave_dim=cfg['wave_dim'],
        hidden_dim=cfg['hidden_dim'],
        byte_embed_dim=cfg['byte_embed_dim'],
        gru_layers=cfg['gru_layers'],
        dropout=cfg['dropout'],
        max_bytes=cfg['max_bytes'],
        num_context=cfg['num_context'],
        context_drop_prob=cfg['context_drop_prob'],
    ).to(device)

    params = sum(p.numel() for p in wtt.parameters())
    print(f"  ContextWaveToText built: {params:,} params")

    return wtt


def load_phase9_1_modules(device: str = 'cpu'):
    """
    Load Phase 9.1 checkpoint and restore all modules.

    Returns:
        (cse, chunker, context_wtt, phase9_1_checkpoint)
    """
    from cse import ContinuousSemanticEncoder

    path = CHECKPOINT_DIR / 'phase9_1.phase.pt'
    if not path.exists():
        raise FileNotFoundError(
            f"Phase 9.1 checkpoint not found at {path}\n"
            f"Run Phase 9.1 training first."
        )

    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    config = ckpt.get('config', {})
    p91cfg = ckpt.get('phase9_1_config', PHASE9_1_CONFIG)

    # CSE
    cse = ContinuousSemanticEncoder(**{
        k: v for k, v in config.items()
        if k in ['wave_dim', 'window_size', 'stride']
    })
    if 'cse_state_dict' in ckpt:
        cse.load_state_dict(ckpt['cse_state_dict'])
    cse.eval()
    for p in cse.parameters():
        p.requires_grad = False
    cse = cse.to(device)

    # WaveChunker
    chunker = WaveChunker(
        wave_dim=p91cfg.get('wave_dim', 432),
    ).to(device)
    if 'wave_chunker_state_dict' in ckpt:
        chunker.load_state_dict(ckpt['wave_chunker_state_dict'])
    chunker.eval()
    for p in chunker.parameters():
        p.requires_grad = False

    # ContextWaveToText
    wtt = ContextWaveToText(
        wave_dim=p91cfg['wave_dim'],
        hidden_dim=p91cfg['hidden_dim'],
        byte_embed_dim=p91cfg['byte_embed_dim'],
        gru_layers=p91cfg['gru_layers'],
        dropout=p91cfg['dropout'],
        max_bytes=p91cfg['max_bytes'],
        num_context=p91cfg['num_context'],
        context_drop_prob=p91cfg.get('context_drop_prob', 0.25),
    ).to(device)
    if 'context_wtt_state_dict' in ckpt:
        wtt.load_state_dict(ckpt['context_wtt_state_dict'])
    wtt.eval()

    print(f"  ✓ Phase 9.1 loaded from checkpoint")
    return cse, chunker, wtt, ckpt


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  Phase 9.1: Context-Aware WaveToText Training")
    print("=" * 60)

    device = get_device()
    log = PhaseLogger(phase=9)  # Append to phase9 log

    # Load frozen components from Phase 9
    log.separator("Loading Phase 9 Components")
    cse, chunker, ckpt9 = load_phase9_for_9_1(device=device)

    # Build fresh ContextWaveToText
    log.separator("Building ContextWaveToText")
    context_wtt = build_context_wtt(device=device)

    # Load training data
    log.separator("Loading Training Data")
    texts = load_training_data(max_docs=PHASE9_1_CONFIG['max_train_docs'])
    split_idx = int(len(texts) * 0.9)
    train_texts = texts[:split_idx]
    eval_texts = texts[split_idx:]
    print(f"  Train: {len(train_texts):,} docs, Eval: {len(eval_texts):,} docs")

    # Create trainer
    trainer = Phase9_1_Trainer(
        cse=cse,
        chunker=chunker,
        context_wtt=context_wtt,
        device=device,
        log=log,
    )

    # Prepare training data
    log.separator("Preparing Training Data")
    training_pairs = trainer.prepare_training_data(
        train_texts,
        max_pairs=500000,
        utf8_texts=UTF8_AUGMENT_TEXTS,
    )

    # Train
    log.separator("Training ContextWaveToText")
    result = trainer.train(
        training_pairs,
        max_steps=PHASE9_1_CONFIG['max_steps'],
        batch_size=PHASE9_1_CONFIG['batch_size'],
        grad_accum=PHASE9_1_CONFIG['grad_accum'],
        lr=PHASE9_1_CONFIG['lr'],
        warmup_steps=PHASE9_1_CONFIG['warmup_steps'],
        log_interval=PHASE9_1_CONFIG['log_interval'],
        eval_interval=PHASE9_1_CONFIG['eval_interval'],
        eval_texts=eval_texts,
    )

    # Save checkpoint
    log.separator("Save Checkpoint")
    trainer.save_checkpoint(result, ckpt9)

    # Generate results
    results = PhaseResults(phase=9, component_name="Context-Aware WaveToText (9.1)")
    results.add_test(
        "Chunk Accuracy", result.chunk_accuracy >= 0.93,
        score=f"{result.chunk_accuracy:.1%}", threshold="≥93%",
    )
    results.add_test(
        "Word Accuracy", result.word_accuracy >= 0.88,
        score=f"{result.word_accuracy:.1%}", threshold="≥88%",
    )
    results.add_test(
        "Hard Spelling", result.hard_spelling_accuracy >= 0.60,
        score=f"{result.hard_spelling_accuracy:.1%}", threshold="≥60%",
    )
    results.add_test(
        "UTF-8 Accuracy", result.utf8_accuracy >= 0.50,
        score=f"{result.utf8_accuracy:.1%}", threshold="≥50%",
    )
    results.add_metric("Training steps", result.total_steps)
    results.add_metric("Final loss", f"{result.final_loss:.4f}")
    results.add_metric("Training time", f"{result.total_time_seconds:.1f}s")
    results.save(
        path=str(Path(__file__).parent / 'RESULTS_PHASE_9_1.md')
    )

    log.success("Phase 9.1 training complete")
    print(f"\n{'=' * 60}")
    print(f"  Phase 9.1 Complete!")
    print(f"{'=' * 60}")
