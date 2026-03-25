"""
DecodeGate: The non-negotiable invariant of the wave-first architecture.

Every v2 phase must pass this gate before completion.
A wave that can't be decoded to readable text is not a physical wave.

Usage:
    from decode_gate import run_decode_gate, byte_accuracy
    # Single coherent codec object expected:
    passed = run_decode_gate(cse, chunker, wtt)
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional

import torch

# ─────────────────────────────────────────────
# Gate Test Texts
# ─────────────────────────────────────────────

DECODE_GATE_TEXTS: List[str] = [
    "The future of artificial intelligence",
    "Energy equals mass times the speed of light squared",
    "Photosynthesis converts sunlight into chemical energy",
    "Water freezes at zero degrees Celsius",
    "The cat sat on the mat",
    "café naïve résumé",
    "def hello(): return 'world'",
    "∫₀^∞ e^(-x²) dx = √π/2",
]

# Thresholds
AVG_THRESHOLD: float = 0.90
MIN_THRESHOLD: float = 0.70


# ─────────────────────────────────────────────
# Byte Accuracy
# ─────────────────────────────────────────────

def byte_accuracy(original: str, decoded: str) -> float:
    """
    Compute byte-level accuracy between original and decoded strings.

    Aligns byte sequences and counts matching positions.
    Shorter/longer strings are handled by padding the shorter one.

    Args:
        original: The ground-truth string
        decoded: The decoded string
    Returns:
        float in [0, 1] — fraction of matching bytes
    """
    orig_bytes = original.encode('utf-8')
    dec_bytes = decoded.encode('utf-8')

    max_len = max(len(orig_bytes), len(dec_bytes), 1)
    min_len = min(len(orig_bytes), len(dec_bytes))

    matches = sum(
        1 for i in range(min_len) if orig_bytes[i] == dec_bytes[i]
    )
    return matches / max_len


def text_similarity(original: str, decoded: str) -> float:
    """
    Character-level similarity (overlap/union — Jaccard on bigrams).
    Tolerant of minor rearrangements that byte_accuracy penalizes.

    Args:
        original: ground-truth text
        decoded: decoded text
    Returns:
        float in [0, 1]
    """
    def get_bigrams(s: str):
        return set(s[i:i+2] for i in range(len(s) - 1)) if len(s) > 1 else set(s)

    bg_orig = get_bigrams(original)
    bg_dec = get_bigrams(decoded)

    if not bg_orig and not bg_dec:
        return 1.0
    if not bg_orig or not bg_dec:
        return 0.0

    intersection = len(bg_orig & bg_dec)
    union = len(bg_orig | bg_dec)
    return intersection / union


# ─────────────────────────────────────────────
# Decode Gate Runner
# ─────────────────────────────────────────────

@torch.no_grad()
def run_decode_gate(
    cse,
    chunker,
    wtt,
    texts: Optional[List[str]] = None,
    phase: int = 1,
    verbose: bool = True,
    temperature: float = 0.5,
    raise_on_fail: bool = True,
) -> Tuple[bool, float, float]:
    """
    Run the decode gate: text → CSE → chunker → WTT → text.
    Must pass at the end of every v2 phase. Non-negotiable.

    Args:
        cse: ContinuousSemanticEncoder (trained)
        chunker: WaveChunker (trained)
        wtt: WaveToText (trained)
        texts: Test texts (defaults to DECODE_GATE_TEXTS)
        phase: Phase number (for display)
        verbose: Print results
        temperature: WTT sampling temperature (lower = greedier)
        raise_on_fail: Raise AssertionError if gate fails
    Returns:
        (passed, avg_accuracy, min_accuracy)
    Raises:
        AssertionError if raise_on_fail=True and gate fails
    """
    if texts is None:
        texts = DECODE_GATE_TEXTS

    # Put everything in eval mode
    cse.eval()
    chunker.eval()
    wtt.eval()

    results = []

    if verbose:
        print(f"\n  {'─'*50}")
        print(f"  Decode Gate (Phase {phase} v2)")
        print(f"  {'─'*50}")

    for text in texts:
        try:
            wave = cse.encode(text)               # SemanticWave
            chunk_waves, _ = chunker(wave.full)    # [N, 432]
            decoded_text = wtt.decode_to_text(chunk_waves, temperature=temperature)
            acc = byte_accuracy(text, decoded_text)
        except Exception as e:
            acc = 0.0
            decoded_text = f"<ERROR: {e}>"

        results.append(acc)

        if verbose:
            status = '✓' if acc >= MIN_THRESHOLD else '✗'
            orig_preview = text[:40] + ('...' if len(text) > 40 else '')
            dec_preview = decoded_text[:40] + ('...' if len(decoded_text) > 40 else '')
            print(f"  {status} [{acc:.1%}] '{orig_preview}'")
            if acc < 0.5:
                print(f"       decoded: '{dec_preview}'")

    avg_acc = sum(results) / len(results)
    min_acc = min(results)
    passed = avg_acc >= AVG_THRESHOLD and min_acc >= MIN_THRESHOLD

    if verbose:
        print(f"  {'─'*50}")
        print(f"  Avg byte accuracy : {avg_acc:.1%}  (threshold: {AVG_THRESHOLD:.0%})")
        print(f"  Min byte accuracy : {min_acc:.1%}  (threshold: {MIN_THRESHOLD:.0%})")
        if passed:
            print(f"  ✓ DECODE GATE PASSED (Phase {phase} v2)")
        else:
            print(f"  ✗ DECODE GATE FAILED (Phase {phase} v2)")
        print(f"  {'─'*50}\n")

    if raise_on_fail:
        assert avg_acc >= AVG_THRESHOLD, (
            f"Decode gate FAILED: avg {avg_acc:.1%} < {AVG_THRESHOLD:.0%}\n"
            f"The wave space is not decodable. Retrain with higher decode_loss weight."
        )
        assert min_acc >= MIN_THRESHOLD, (
            f"Decode gate FAILED: min {min_acc:.1%} < {MIN_THRESHOLD:.0%}\n"
            f"Some inputs don't round-trip. Check UTF-8 handling and WTT max_bytes."
        )

    return passed, avg_acc, min_acc
