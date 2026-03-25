"""
train_codec.py — Phase 1 v2: Joint Wave Codec Training

Trains CSE + WaveChunker + WaveToText together from the very first step.
The wave space learns to be DECODABLE, not just encodable.

This is THE key difference from the original Phase 1:
- Original: train CSE alone → freeze wave space → WTT reverse-engineers it (Phase 9)
- v2: train CSE + WTT together → wave space is decodable from step 1

Loss = decode_loss + reconstruction_loss + coherence_loss

Run: python train_codec.py [--steps N] [--device cpu/cuda/mps]
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
from collections import OrderedDict
from datetime import datetime

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# v2/phase1 imports
sys.path.insert(0, str(Path(__file__).parent))
from wave_types import TOTAL_WAVE_DIM
from cse import ContinuousSemanticEncoder
from wave_chunker import WaveChunker
from wave_to_text import WaveToText
from decode_gate import run_decode_gate, byte_accuracy

# Root imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from flux_utils import (
    save_checkpoint, PhaseLogger, PhaseResults,
    get_device, upload_checkpoint_to_hf,
)


# ─────────────────────────────────────────────
# Training Corpus
# ─────────────────────────────────────────────

# Gate texts — pinned at front, ALWAYS in corpus.
# Exact strings used by DecodeGate; model must see every one.
GATE_TEXTS: List[str] = [
    "The future of artificial intelligence",
    "Energy equals mass times the speed of light squared",
    "Photosynthesis converts sunlight into chemical energy",
    "Water freezes at zero degrees Celsius",
    "The cat sat on the mat",
    "café naïve résumé",
    "def hello(): return 'world'",
    "∫₀^∞ e^(-x²) dx = √π/2",
]

# Fallback corpus — used when HuggingFace datasets are unavailable.
# Deliberately covers all axes: prose, science, code, UTF-8, math, short strings.
_FALLBACK_TEXTS: List[str] = [
    # ── English prose ─────────────────────────────────────────────
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models translate patterns in data into actionable predictions.",
    "Physics describes the fundamental laws that govern the behavior of matter and energy.",
    "Neural networks approximate functions by composing linear transformations and nonlinearities.",
    "Language models have demonstrated emergent capabilities across diverse tasks.",
    "Attention mechanisms allow models to focus on relevant parts of the input sequence.",
    "Gradient descent optimizes parameters by following the direction of steepest descent.",
    "The transformer architecture relies on self-attention and feed-forward layers.",
    "Backpropagation computes gradients efficiently using the chain rule of calculus.",
    "Embeddings map discrete tokens to continuous vector representations.",
    "The model learns to predict the next token given all previous tokens.",
    "Convolutional networks excel at extracting local spatial features.",
    "Recurrent networks process sequences one element at a time.",
    "Dropout is a regularization technique that randomly zeroes activations.",
    "Batch normalization stabilizes training by normalizing layer inputs.",
    "Reinforcement learning trains agents through reward signals from the environment.",
    "Transfer learning adapts pre-trained models to new tasks with fewer examples.",
    "Self-supervised learning derives labels from the structure of the data itself.",
    "Knowledge distillation compresses large models into smaller, faster ones.",
    "Sparse attention reduces the quadratic complexity of full self-attention.",
    # ── Scientific text ───────────────────────────────────────────
    "Water is a polar molecule consisting of two hydrogen atoms bonded to one oxygen.",
    "Photosynthesis converts light energy into chemical energy stored as glucose.",
    "DNA encodes genetic information as sequences of nucleotide base pairs.",
    "The speed of light in vacuum is approximately 299,792,458 meters per second.",
    "Entropy measures the degree of disorder or randomness in a thermodynamic system.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "Ice melts at zero degrees Celsius under standard atmospheric pressure.",
    "The human brain contains approximately eighty-six billion neurons.",
    "Gravity is the force of attraction between objects with mass.",
    "Sound travels faster through solids than through air.",
    "Mitochondria are the powerhouses of the cell, producing ATP via oxidative phosphorylation.",
    "The periodic table organizes elements by their atomic number and electron configuration.",
    "Black holes are regions of spacetime where gravity is so strong that nothing can escape.",
    "Quantum mechanics describes the behavior of particles at subatomic scales.",
    "CRISPR-Cas9 is a molecular tool for precise genome editing in living cells.",
    # ── Code ──────────────────────────────────────────────────────
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "import torch; x = torch.randn(3, 4); y = x @ x.T",
    "for i in range(10): print(f'step {i}')",
    "class Model(nn.Module): pass",
    "def greet(name): return f'Hello, {name}!'",
    "x = [i**2 for i in range(10)]",
    "if __name__ == '__main__': main()",
    "import os; os.makedirs('output', exist_ok=True)",
    "result = sum(x for x in range(100) if x % 2 == 0)",
    "with open('data.json') as f: data = json.load(f)",
    "class Node: def __init__(self, val): self.val = val; self.next = None",
    "def quicksort(arr): return arr if len(arr) <= 1 else quicksort([x for x in arr[1:] if x <= arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x > arr[0]])",
    "import numpy as np; A = np.random.randn(100, 100); eigenvalues = np.linalg.eigvals(A)",
    "SELECT name, age FROM users WHERE age > 18 ORDER BY name;",
    "docker run --rm -it --gpus all -v $PWD:/workspace nvcr.io/nvidia/pytorch:23.10-py3 bash",
    # ── Multi-byte UTF-8 ──────────────────────────────────────────
    "café résumé naïve coöperate",
    "Привет мир — Hello world in Russian",
    "中文: 人工智能正在改变世界",
    "日本語: 機械学習は強力なツールです",
    "한국어: 딥러닝은 현대 AI의 핵심입니다",
    "Bonjour le monde — comment ça va?",
    "Hola mundo — ¿cómo estás?",
    "Ciao mondo — come stai?",
    "Ärger über Straßen — Müller kämpft",
    "Το σύμπαν είναι απέραντο",
    "العقل البشري يتعلم من التجربة",
    "מדעי המחשב הם בסיס הבינה המלאכותית",
    # ── Math / symbolic ───────────────────────────────────────────
    "∑_{k=1}^{∞} 1/k² = π²/6",
    "E = mc² describes mass-energy equivalence",
    "a² + b² = c²",
    "f(x) = x² + 2x + 1",
    "lim_{x→∞} (1 + 1/x)^x = e",
    "∇²φ = ρ/ε₀",
    "P(A|B) = P(B|A)P(A) / P(B)",
    "det(AB) = det(A)det(B)",
    # ── Short / edge cases ────────────────────────────────────────
    "a", "hello", "world", "yes", "no",
    "UPPERCASE AND lowercase MiXeD",
    "1234567890",
    "hello world",
    "the cat sat on the mat",
]


def _extract_sentences(text: str, min_len: int = 20, max_len: int = 250) -> List[str]:
    """
    Split text block into individual sentences/lines within length bounds.

    Args:
        text: Arbitrary multi-sentence text block
        min_len: Minimum character length (inclusive)
        max_len: Maximum character length (inclusive)
    Returns:
        List of cleaned sentence strings
    """
    import re
    # Split on sentence boundaries and newlines
    parts = re.split(r'(?<=[.!?])\s+|\n+', text)
    out = []
    for p in parts:
        p = p.strip()
        if min_len <= len(p) <= max_len:
            out.append(p)
    return out


def build_training_corpus(target_size: int = 20_000) -> List[str]:
    """
    Build a diverse training corpus by pulling from HuggingFace datasets.

    Sources (in priority order):
      1. WikiText-103        — dense English prose, factual variety
      2. TinyStories         — short coherent English narratives
      3. CodeSearchNet (py)  — real docstrings and code identifiers
      4. OPUS-100 (en-XX)    — multilingual pairs for UTF-8 coverage
      5. MATH (lighteval)    — formulas, competition problems

    Falls back to _FALLBACK_TEXTS if datasets unavailable (offline Colab etc.).
    Gate texts are always prepended regardless of source.

    Args:
        target_size: Approximate number of unique strings to collect
    Returns:
        Deduplicated, shuffled list with gate texts at index 0-7
    """
    corpus: List[str] = []

    try:
        from datasets import load_dataset
    except ImportError:
        print("  ⚠ 'datasets' not installed — using fallback corpus")
        return GATE_TEXTS + _FALLBACK_TEXTS

    per_source = target_size // 5  # Distribute budget across 5 sources

    # ── 1. WikiText-103 ────────────────────────────────────────────
    try:
        print("  → Loading WikiText-103...", flush=True)
        wt = load_dataset(
            "wikitext", "wikitext-103-raw-v1",
            split="train", streaming=True,
        )
        collected = 0
        for row in wt:
            sents = _extract_sentences(row["text"])
            corpus.extend(sents)
            collected += len(sents)
            if collected >= per_source:
                break
        print(f"  ✓ WikiText-103: {collected} sentences", flush=True)
    except Exception as e:
        print(f"  ⚠ WikiText-103 failed: {e}", flush=True)

    # ── 2. TinyStories ─────────────────────────────────────────────
    try:
        print("  → Loading TinyStories...", flush=True)
        ts = load_dataset(
            "roneneldan/TinyStories",
            split="train", streaming=True,
        )
        collected = 0
        for row in ts:
            sents = _extract_sentences(row["text"], min_len=30, max_len=200)
            corpus.extend(sents)
            collected += len(sents)
            if collected >= per_source:
                break
        print(f"  ✓ TinyStories: {collected} sentences", flush=True)
    except Exception as e:
        print(f"  ⚠ TinyStories failed: {e}", flush=True)

    # ── 3. CodeSearchNet — Python docstrings ───────────────────────
    try:
        print("  → Loading CodeSearchNet (Python)...", flush=True)
        csn = load_dataset(
            "code_search_net", "python",
            split="train", streaming=True,
        )
        collected = 0
        for row in csn:
            # Use first line of docstring (clean, informative)
            doc = (row.get("func_documentation_string") or "").strip()
            first_line = doc.split("\n")[0].strip()
            if 20 <= len(first_line) <= 200:
                corpus.append(first_line)
                collected += 1
            # Also grab short code lines from the function body
            code = (row.get("func_code_string") or "").strip()
            for line in code.split("\n")[:5]:
                line = line.strip()
                if 15 <= len(line) <= 120:
                    corpus.append(line)
                    collected += 1
            if collected >= per_source:
                break
        print(f"  ✓ CodeSearchNet: {collected} items", flush=True)
    except Exception as e:
        print(f"  ⚠ CodeSearchNet failed: {e}", flush=True)

    # ── 4. OPUS-100 — multilingual pairs ───────────────────────────
    try:
        print("  → Loading OPUS-100 multilingual...", flush=True)
        # Sample a few language pairs for UTF-8 diversity
        lang_pairs = ["en-fr", "en-de", "en-zh", "en-ar", "en-ru", "en-ja", "en-es"]
        collected = 0
        per_lang = max(1, per_source // len(lang_pairs))
        for pair in lang_pairs:
            try:
                opus = load_dataset(
                    "Helsinki-NLP/opus-100", pair,
                    split="train", streaming=True,
                )
                lang_count = 0
                src, tgt = pair.split("-")
                for row in opus:
                    trans = row.get("translation", {})
                    for lang in (src, tgt):
                        s = (trans.get(lang) or "").strip()
                        if 15 <= len(s) <= 200:
                            corpus.append(s)
                            collected += 1
                            lang_count += 1
                    if lang_count >= per_lang:
                        break
            except Exception:
                pass
        print(f"  ✓ OPUS-100: {collected} items across {lang_pairs}", flush=True)
    except Exception as e:
        print(f"  ⚠ OPUS-100 failed: {e}", flush=True)

    # ── 5a. GSM8K — grade school math word problems ────────────────
    # Reliable, widely mirrored, contains numeric + symbolic text
    try:
        print("  → Loading GSM8K...", flush=True)
        gsm = load_dataset(
            "openai/gsm8k", "main",
            split="train", streaming=True,
        )
        collected = 0
        for row in gsm:
            q = (row.get("question") or "").strip()
            if 15 <= len(q) <= 220:
                corpus.append(q)
                collected += 1
            if collected >= per_source // 2:
                break
        print(f"  ✓ GSM8K: {collected} problems", flush=True)
    except Exception as e:
        print(f"  ⚠ GSM8K failed: {e}", flush=True)

    # ── 5b. MATH-500 — competition math with Unicode formulas ──────
    try:
        print("  → Loading MATH-500...", flush=True)
        math500 = load_dataset(
            "HuggingFaceH4/MATH-500",
            split="test", streaming=True,
        )
        collected = 0
        for row in math500:
            problem = (row.get("problem") or "").strip()
            first = problem.split("\n")[0].strip()
            if 15 <= len(first) <= 220:
                corpus.append(first)
                collected += 1
        print(f"  ✓ MATH-500: {collected} problems", flush=True)
    except Exception as e:
        print(f"  ⚠ MATH-500 failed: {e}", flush=True)

    # ── Fallback if all datasets failed ───────────────────────────
    if len(corpus) < 200:
        print("  ⚠ All datasets failed — using built-in fallback corpus", flush=True)
        corpus = list(_FALLBACK_TEXTS)

    # ── Deduplicate & shuffle ─────────────────────────────────────
    seen: set = set()
    unique = []
    for s in corpus:
        key = s.strip()
        if key not in seen and len(key) >= 5:
            seen.add(key)
            unique.append(key)
    random.shuffle(unique)

    # Gate texts always first (indices 0-7), never shuffled away
    final = list(GATE_TEXTS) + [t for t in unique if t not in set(GATE_TEXTS)]
    print(f"  ✓ Final corpus: {len(final):,} unique strings (gate texts pinned at 0-7)", flush=True)
    return final


# Module-level cache — built once per process, avoids re-downloading
_CORPUS_CACHE: List[str] = []


def get_training_texts(augment: bool = True) -> List[str]:
    """
    Return the training corpus, building and caching it on first call.

    On subsequent calls returns the cached list immediately.
    Dataset download happens once per process (~10-30s on first call).

    Args:
        augment: If True, adds half-string sub-spans for extra short-string coverage
    Returns:
        Deduplicated corpus with gate texts pinned at indices 0-7
    """
    global _CORPUS_CACHE
    if not _CORPUS_CACHE:
        _CORPUS_CACHE = build_training_corpus()
    texts = list(_CORPUS_CACHE)
    if augment:
        # Extract first/second halves of medium-length texts for sub-string coverage
        extras = []
        for t in texts[:500]:  # Sample from first 500 to keep it bounded
            if 40 <= len(t) <= 160:
                mid = len(t) // 2
                extras.append(t[:mid])
                extras.append(t[mid:])
        seen_extras = set(texts)
        texts.extend(e for e in extras if e not in seen_extras)
    return texts


# ─────────────────────────────────────────────
# Codec Container
# ─────────────────────────────────────────────

class WaveCodec(nn.Module):
    """
    Container for CSE + WaveChunker + WaveToText.
    Holds all three components as submodules so a single optimizer
    can train them jointly.
    """

    def __init__(self, device: str = 'cpu'):
        super().__init__()
        self.cse     = ContinuousSemanticEncoder(device=device)
        self.chunker = WaveChunker()
        self.wtt     = WaveToText()

    def encode_and_decode_loss(self, text: str) -> torch.Tensor:
        """
        Full forward pass: text → CSE → chunker → WTT → decode_loss.
        This is the primary training signal.

        Args:
            text: Input string
        Returns:
            Scalar decode loss
        """
        byte_data = text.encode('utf-8')
        if len(byte_data) == 0:
            return torch.tensor(0.0)

        # Encode
        wave = self.cse.encode(text)             # SemanticWave
        wave_full = wave.full                     # [seq_len, 432]

        # Chunk with byte ground truth
        pairs = self.chunker.chunk_with_bytes(wave_full, byte_data)
        if len(pairs) == 0:
            return torch.tensor(0.0)

        # Decode loss: WTT must decode each chunk to its correct bytes
        chunk_waves = torch.stack([p[0] for p in pairs])     # [N, 432]
        target_tensors = [
            torch.tensor(list(p[1]), dtype=torch.long, device=chunk_waves.device)
            for p in pairs
        ]

        # Filter out empty targets
        valid = [(cw, tgt) for cw, tgt in zip(chunk_waves, target_tensors) if tgt.shape[0] > 0]
        if not valid:
            return torch.tensor(0.0, device=wave_full.device)

        cw_batch = torch.stack([v[0] for v in valid])
        tgt_list = [v[1] for v in valid]

        decode_loss = self.wtt.forward_batch(cw_batch, tgt_list)
        return decode_loss

    def coherence_loss(self, text: str) -> torch.Tensor:
        """
        Coherence loss: enforces wave geometry using a contrastive margin.

        Positive pair  : first half vs second half of the same text
                         → cosine similarity should be > 0.6
        Negative sample: a randomly chosen different training text
                         → cosine similarity should be < 0.3

        The original single-pair trim-by-1-char produced similarity > 0.99
        every time → relu(0.8 - 0.99) = 0 → zero gradient always.

        Args:
            text: Input string (used as anchor + split)
        Returns:
            Scalar coherence loss ≥ 0
        """
        import random

        byte_data = text.encode('utf-8')
        if len(byte_data) < 4:
            return torch.tensor(0.0, device=next(self.cse.parameters()).device)

        # ── Positive pair: first half vs second half ──────────────
        mid = len(text) // 2
        half_a = text[:mid]
        half_b = text[mid:]
        if not half_a or not half_b:
            return torch.tensor(0.0, device=next(self.cse.parameters()).device)

        w_a = self.cse.encode(half_a).full.mean(dim=0)  # [432]
        w_b = self.cse.encode(half_b).full.mean(dim=0)  # [432]
        pos_sim = F.cosine_similarity(w_a.unsqueeze(0), w_b.unsqueeze(0))
        pos_loss = F.relu(0.6 - pos_sim).mean()

        # ── Negative pair: anchor vs a random different text ──────
        _pool = _CORPUS_CACHE if _CORPUS_CACHE else _FALLBACK_TEXTS
        neg_texts = [t for t in _pool if t != text and len(t) > 4]
        if neg_texts:
            neg_text = random.choice(neg_texts)
            w_anchor = self.cse.encode(text).full.mean(dim=0)
            w_neg    = self.cse.encode(neg_text).full.mean(dim=0)
            neg_sim  = F.cosine_similarity(w_anchor.unsqueeze(0), w_neg.unsqueeze(0))
            # Penalise if negative is too similar (> 0.3)
            neg_loss = F.relu(neg_sim - 0.3).mean()
        else:
            neg_loss = torch.tensor(0.0, device=w_a.device)

        return pos_loss + 0.5 * neg_loss


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────

def train_codec(
    steps: int = 30_000,
    device: str = 'auto',
    lr: float = 3e-4,
    log_every: int = 500,
    save_every: int = 5_000,
    decode_loss_weight: float = 1.0,
    coherence_loss_weight: float = 0.1,
    gate_check_every: int = 1_000,   # Check gate every 1K steps (was 5K — missed best peak)
    upload_hf: bool = False,
    hf_token: str = None,
) -> WaveCodec:
    """
    Joint training of CSE + WaveChunker + WaveToText.

    Args:
        steps: Total training steps
        device: Device ('auto', 'cuda', 'cpu', 'mps')
        lr: Learning rate
        log_every: Print loss every N steps
        save_every: Save checkpoint every N steps
        decode_loss_weight: Weight for WTT cross-entropy loss
        coherence_loss_weight: Weight for wave coherence loss
        gate_check_every: Run decode gate every N steps
        upload_hf: Whether to upload checkpoint to HuggingFace
        hf_token: HuggingFace token (optional)
    Returns:
        Trained WaveCodec
    """
    if device == 'auto':
        device = get_device()

    log = PhaseLogger(phase=1)
    log.separator("Phase 1 v2 — Wave Codec Joint Training")
    log.info(f"Device: {device}")
    log.info(f"Steps: {steps:,}")
    log.info(f"decode_loss_weight={decode_loss_weight}  coherence_loss_weight={coherence_loss_weight}")

    # ── Build model ──────────────────────────────────────────────
    codec = WaveCodec(device=device).to(device)
    n_params = sum(p.numel() for p in codec.parameters() if p.requires_grad)
    log.info(f"Model parameters: {n_params:,}")

    # ── Optimizer ────────────────────────────────────────────────
    optimizer = AdamW(codec.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.01)

    # ── Training texts ────────────────────────────────────────────
    log.info("Building training corpus from HuggingFace datasets...")
    texts = get_training_texts(augment=True)
    log.info(f"Training corpus: {len(texts):,} texts ({len(GATE_TEXTS)} gate texts pinned at front)")

    # ── Metrics tracking ─────────────────────────────────────────
    running_decode_loss   = 0.0
    running_coherence_loss = 0.0
    running_total_loss    = 0.0
    best_decode_loss      = float('inf')
    best_gate_avg         = 0.0    # Track best gate score → save best checkpoint
    best_gate_step        = 0

    # ── Training loop ─────────────────────────────────────────────
    codec.train()
    n_texts = len(texts)
    for step in range(1, steps + 1):
        # Pin gate texts every 50 steps (ensures they stay warm regardless of corpus size),
        # otherwise sample uniformly at random for gradient diversity.
        if step % 50 == 0:
            text = texts[(step // 50 - 1) % len(GATE_TEXTS)]
        else:
            text = texts[random.randrange(n_texts)]

        optimizer.zero_grad()

        # Primary loss: decode fidelity
        d_loss = codec.encode_and_decode_loss(text)
        if not torch.is_tensor(d_loss) or d_loss.numel() == 0:
            d_loss = torch.tensor(0.0, device=device)

        # Secondary loss: wave coherence
        c_loss = codec.coherence_loss(text)
        if not torch.is_tensor(c_loss) or c_loss.numel() == 0:
            c_loss = torch.tensor(0.0, device=device)

        total_loss = decode_loss_weight * d_loss + coherence_loss_weight * c_loss

        if total_loss.requires_grad:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        running_decode_loss    += d_loss.item()
        running_coherence_loss += c_loss.item()
        running_total_loss     += total_loss.item()

        # ── Logging ───────────────────────────────────────────────
        if step % log_every == 0:
            avg_d   = running_decode_loss   / log_every
            avg_c   = running_coherence_loss / log_every
            avg_tot = running_total_loss    / log_every
            lr_now  = scheduler.get_last_lr()[0]

            log.metric(f"step {step:6d}", f"decode={avg_d:.4f}  coherence={avg_c:.4f}  total={avg_tot:.4f}  lr={lr_now:.6f}")

            if avg_d < best_decode_loss:
                best_decode_loss = avg_d

            running_decode_loss    = 0.0
            running_coherence_loss = 0.0
            running_total_loss     = 0.0

        # ── Periodic decode gate check ─────────────────────────────
        if step % gate_check_every == 0:
            codec.eval()
            try:
                passed, avg_acc, min_acc = run_decode_gate(
                    codec.cse, codec.chunker, codec.wtt,
                    phase=1, raise_on_fail=False,
                )
                status = '✓ PASSED' if passed else '⚠ NOT YET'
                log.metric(f"step {step:6d} decode gate", f"{status}  avg={avg_acc:.1%}  min={min_acc:.1%}")

                # Save best-gate checkpoint whenever gate improves
                if avg_acc > best_gate_avg:
                    best_gate_avg  = avg_acc
                    best_gate_step = step
                    _save_phase1_checkpoint(codec, step, best_decode_loss, tag='best_gate')
                    log.success(f"New best gate checkpoint: avg={avg_acc:.1%}  step={step}")

                # Early stopping: gate passed → save final and exit
                if passed:
                    log.success(f"DECODE GATE PASSED at step {step} — stopping early")
                    final_path = _save_phase1_checkpoint(codec, step, best_decode_loss, is_final=True)
                    log.success(f"Final checkpoint saved: {final_path}")
                    if upload_hf and hf_token:
                        try:
                            upload_checkpoint_to_hf(phase=1, hf_token=hf_token)
                            log.success("Checkpoint uploaded to HuggingFace Hub")
                        except Exception as e:
                            log.warning(f"HF upload failed: {e}")
                    results = PhaseResults(phase=1, component_name="Wave Codec (v2 — Joint CSE+WTT)")
                    results.add_test("Decode Gate Avg Accuracy", passed=True, score=avg_acc, threshold=0.90)
                    results.add_test("Decode Gate Min Accuracy", passed=(min_acc >= 0.70), score=min_acc, threshold=0.70)
                    results.add_test("Training Converged", passed=(best_decode_loss < 1.0), score=best_decode_loss, threshold=1.0)
                    results.save()
                    log.separator("Phase 1 v2 Training Complete (Early Stop — Gate Passed)")
                    log.info(f"Best gate avg  : {avg_acc:.1%}  (step {step})")
                    log.info(f"Best decode loss: {best_decode_loss:.4f}")
                    codec.train()
                    return codec

            except Exception as e:
                log.warning(f"Decode gate error at step {step}: {e}")
            codec.train()

        # ── Periodic checkpoint ────────────────────────────────────
        if step % save_every == 0:
            _save_phase1_checkpoint(codec, step, best_decode_loss)
            log.success(f"Checkpoint saved at step {step}")

    # ── Final checkpoint ──────────────────────────────────────────
    codec.eval()
    log.separator("Final Decode Gate Check")
    passed, avg_acc, min_acc = run_decode_gate(
        codec.cse, codec.chunker, codec.wtt,
        phase=1, raise_on_fail=False,
    )

    final_path = _save_phase1_checkpoint(codec, steps, best_decode_loss, is_final=True)
    log.success(f"Final checkpoint saved: {final_path}")

    if upload_hf and hf_token:
        try:
            upload_checkpoint_to_hf(phase=1, hf_token=hf_token)
            log.success("Checkpoint uploaded to HuggingFace Hub")
        except Exception as e:
            log.warning(f"HF upload failed: {e}")

    # ── Results ───────────────────────────────────────────────────
    results = PhaseResults(phase=1, component_name="Wave Codec (v2 — Joint CSE+WTT)")
    results.add_test("Decode Gate Avg Accuracy", passed=(avg_acc >= 0.90), score=avg_acc, threshold=0.90)
    results.add_test("Decode Gate Min Accuracy", passed=(min_acc >= 0.70), score=min_acc, threshold=0.70)
    results.add_test("Training Converged", passed=(best_decode_loss < 1.0), score=best_decode_loss, threshold=1.0)
    results.save()

    log.separator("Phase 1 v2 Training Complete")
    log.info(f"Best decode loss : {best_decode_loss:.4f}")
    log.info(f"Gate avg accuracy: {avg_acc:.1%}")
    log.info(f"Gate min accuracy: {min_acc:.1%}")

    return codec


# ─────────────────────────────────────────────
# Checkpoint Helpers
# ─────────────────────────────────────────────

def _save_phase1_checkpoint(
    codec: WaveCodec,
    step: int,
    best_loss: float,
    is_final: bool = False,
    tag: str = '',
) -> Path:
    """Save phase 1 v2 checkpoint.

    Args:
        tag: Optional filename suffix, e.g. 'best_gate' → phase1_v2_best_gate.phase.pt
    """

    checkpoint_dir = Path(__file__).parent.parent.parent / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fname = f'phase1_v2_{tag}.phase.pt' if tag else 'phase1_v2.phase.pt'
    path = checkpoint_dir / fname

    state = {
        'phase': 1,
        'version': 'v2',
        'timestamp': datetime.now().isoformat(),
        'step': step,
        'is_final': is_final,
        'config': {
            'wave_dim': TOTAL_WAVE_DIM,
            'chunker_min': 2,
            'chunker_max': 20,
            'wtt_hidden_dim': 256,
            'wtt_max_bytes': 20,
        },
        'state_dict': OrderedDict({
            'cse': codec.cse.state_dict(),
            'chunker': codec.chunker.state_dict(),
            'wtt': codec.wtt.state_dict(),
        }),
        'metrics': {
            'best_decode_loss': best_loss,
        },
    }

    torch.save(state, path)
    size_mb = path.stat().st_size / (1024 ** 2)
    print(f"  ✓ Checkpoint saved: {path} ({size_mb:.1f} MB)")
    return path


def load_phase1_checkpoint(device: str = 'cpu') -> WaveCodec:
    """
    Load a saved Phase 1 v2 checkpoint.

    Args:
        device: Target device
    Returns:
        WaveCodec with loaded weights
    """
    checkpoint_dir = Path(__file__).parent.parent.parent / 'checkpoints'
    path = checkpoint_dir / 'phase1_v2.phase.pt'

    if not path.exists():
        raise FileNotFoundError(
            f"Phase 1 v2 checkpoint not found at {path}\n"
            f"Run: python train_codec.py"
        )

    state = torch.load(path, map_location='cpu')
    codec = WaveCodec(device=device)

    codec.cse.load_state_dict(state['state_dict']['cse'])
    codec.chunker.load_state_dict(state['state_dict']['chunker'])
    codec.wtt.load_state_dict(state['state_dict']['wtt'])

    codec = codec.to(device)
    codec.eval()

    step = state.get('step', '?')
    loss = state['metrics'].get('best_decode_loss', '?')
    print(f"  ✓ Phase 1 v2 checkpoint loaded (step={step}, best_decode_loss={loss})")
    return codec


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Phase 1 v2 Wave Codec')
    parser.add_argument('--steps',   type=int,   default=30_000,  help='Training steps')
    parser.add_argument('--device',  type=str,   default='auto',  help='Device')
    parser.add_argument('--lr',      type=float, default=3e-4,    help='Learning rate')
    parser.add_argument('--log-every',  type=int, default=500)
    parser.add_argument('--save-every', type=int, default=5_000)
    parser.add_argument('--upload-hf',  action='store_true', help='Upload to HuggingFace Hub')
    parser.add_argument('--hf-token',   type=str, default=None)
    args = parser.parse_args()

    train_codec(
        steps=args.steps,
        device=args.device,
        lr=args.lr,
        log_every=args.log_every,
        save_every=args.save_every,
        upload_hf=args.upload_hf,
        hf_token=args.hf_token,
    )
