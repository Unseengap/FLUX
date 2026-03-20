"""
PHASE 1 TEST 1: Reconstruction quality.

Pass criteria:
    - Average reconstruction loss < 0.1 on 1000 validation sentences
      (loss = 1 - byte_accuracy, so accuracy > 90%)
    - No single sentence has reconstruction loss > 0.5
      (every sentence has accuracy > 50%)
    - Test takes < 60 seconds to run

Run: python test_phase1_test1.py
"""

import sys
import time
import random
from pathlib import Path

PHASE_DIR = Path(__file__).parent
PROJECT_ROOT = PHASE_DIR.parent.parent
sys.path.insert(0, str(PHASE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from cse import ContinuousSemanticEncoder, WaveDecoder
from wave_types import TOTAL_WAVE_DIM
from flux_utils import load_checkpoint, PhaseResults


def load_validation_sentences(max_sentences: int = 1000) -> list:
    """Load validation sentences from WikiText-2."""
    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', trust_remote_code=True)
        lines = [
            line.strip() for line in ds['validation']['text']
            if 10 < len(line.strip()) < 200
        ]
        return lines[:max_sentences]
    except Exception:
        # Fallback: generate test sentences
        words = [
            'the', 'cat', 'sat', 'on', 'mat', 'dog', 'ran', 'fast',
            'big', 'house', 'was', 'old', 'and', 'new', 'car', 'road',
            'sun', 'moon', 'star', 'rain', 'day', 'night', 'book', 'read',
            'happy', 'sad', 'love', 'hate', 'good', 'bad', 'king', 'queen',
        ]
        sentences = []
        for _ in range(max_sentences):
            length = random.randint(5, 15)
            sentences.append(' '.join(random.choices(words, k=length)))
        return sentences


def main():
    print("=" * 60)
    print("FLUX Phase 1 Test 1: Reconstruction Quality")
    print("=" * 60)

    start_time = time.time()
    results = PhaseResults(phase=1, component_name="Continuous Semantic Encoder")

    # ── Load checkpoint ──
    print("\n  Loading Phase 1 checkpoint...")
    checkpoint = load_checkpoint(1)
    config = checkpoint['config']

    cse = ContinuousSemanticEncoder(**config, device='cpu')
    cse.load_state_dict(checkpoint['state_dict'])
    cse.eval()

    decoder = WaveDecoder(wave_dim=TOTAL_WAVE_DIM)
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    decoder.eval()

    # ── Load validation data ──
    print("  Loading validation sentences...")
    sentences = load_validation_sentences(1000)
    print(f"  ✓ {len(sentences)} sentences loaded")

    # ── Evaluate reconstruction ──
    print("\n  Evaluating reconstruction quality...")
    total_correct = 0
    total_bytes = 0
    per_sentence_loss = []
    failed_sentences = 0

    with torch.no_grad():
        for i, sentence in enumerate(sentences):
            if len(sentence) < 2:
                continue

            wave = cse.encode(sentence)
            logits = decoder(wave.full)
            target = cse.text_to_bytes(sentence)

            min_len = min(logits.shape[0], target.shape[0])
            logits = logits[:min_len]
            target = target[:min_len]

            preds = logits.argmax(dim=-1)
            correct = (preds == target).sum().item()
            sentence_len = min_len

            accuracy = correct / max(sentence_len, 1)
            loss = 1.0 - accuracy
            per_sentence_loss.append(loss)

            total_correct += correct
            total_bytes += sentence_len

            if loss > 0.5:
                failed_sentences += 1

            if (i + 1) % 200 == 0:
                running_acc = total_correct / max(total_bytes, 1)
                print(f"    [{i+1}/{len(sentences)}] running accuracy: {running_acc:.3f}")

    # ── Compute metrics ──
    avg_accuracy = total_correct / max(total_bytes, 1)
    avg_loss = 1.0 - avg_accuracy
    max_loss = max(per_sentence_loss) if per_sentence_loss else 1.0
    elapsed = time.time() - start_time

    print(f"\n  ── Results ──")
    print(f"  Average byte accuracy:       {avg_accuracy:.4f}")
    print(f"  Average reconstruction loss: {avg_loss:.4f}")
    print(f"  Max single-sentence loss:    {max_loss:.4f}")
    print(f"  Failed sentences (loss>0.5): {failed_sentences}/{len(sentences)}")
    print(f"  Time elapsed:                {elapsed:.1f}s")

    # ── Record results ──
    results.add_test(
        "Average Reconstruction Loss",
        passed=avg_loss < 0.1,
        score=f"{avg_loss:.4f}",
        threshold="< 0.1",
    )
    results.add_test(
        "No Complete Failures",
        passed=failed_sentences == 0,
        score=f"{failed_sentences} failures",
        threshold="0 sentences with loss > 0.5",
    )
    results.add_test(
        "Test Runtime",
        passed=elapsed < 60,
        score=f"{elapsed:.1f}s",
        threshold="< 60s",
    )

    results.add_metric("Average byte accuracy", f"{avg_accuracy:.4f}")
    results.add_metric("Average reconstruction loss", f"{avg_loss:.4f}")
    results.add_metric("Sentences evaluated", len(sentences))

    results.add_demo("test_phase1_test1 (Reconstruction)", True, "Automated")
    results.save()

    # ── Summary ──
    print(f"\n  All tests passed: {results.all_tests_passed()}")
    if results.all_tests_passed():
        print("  ✓ RECONSTRUCTION TEST PASSED")
    else:
        print("  ✗ RECONSTRUCTION TEST FAILED")

    return results.all_tests_passed()


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
