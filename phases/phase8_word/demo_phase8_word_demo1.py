"""
Demo 1: Word-Level CSE Basic Encoding

Shows the fundamental improvement: character bytes → word units.
Demonstrates sequence compression and semantic word waves.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from flux_utils import get_device, PhaseLogger

sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1'))
from cse import ContinuousSemanticEncoder

sys.path.insert(0, str(Path(__file__).parent))
from word_cse import WordLevelCSE, compare_byte_vs_word


def main():
    log = PhaseLogger(phase=8)
    log.separator("Demo: Word-Level CSE Encoding")
    
    device = get_device()
    log.info(f"Device: {device}")
    
    # Create encoders
    log.info("Creating encoders...")
    byte_cse = ContinuousSemanticEncoder(device=device).to(device)
    word_cse = WordLevelCSE(byte_cse=byte_cse, device=device).to(device)
    
    # Test sentences
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning transforms artificial intelligence.",
        "I love programming in Python!",
        "日本語も対応 and English mixed together.",
        "Short.",
        "A significantly longer sentence that demonstrates how word-level encoding dramatically reduces sequence length compared to byte-level encoding while preserving the complete semantic meaning of each word in the input text.",
    ]
    
    log.separator("Byte vs Word Comparison")
    
    print("\n" + "="*80)
    print(f"{'Text':<40} | {'Bytes':>8} | {'Words':>8} | {'Ratio':>8}")
    print("="*80)
    
    total_bytes = 0
    total_words = 0
    
    for text in test_texts:
        comparison = compare_byte_vs_word(text, byte_cse, word_cse)
        
        display_text = text[:37] + "..." if len(text) > 40 else text
        print(f"{display_text:<40} | {comparison['byte_positions']:>8} | {comparison['word_positions']:>8} | {comparison['compression_ratio']:>7.1f}x")
        
        total_bytes += comparison['byte_positions']
        total_words += comparison['word_positions']
    
    print("="*80)
    avg_compression = total_bytes / max(total_words, 1)
    print(f"{'TOTAL':<40} | {total_bytes:>8} | {total_words:>8} | {avg_compression:>7.1f}x")
    print()
    
    log.metric("avg_compression", f"{avg_compression:.1f}x")
    
    # Detailed word breakdown
    log.separator("Word Breakdown Example")
    
    example = "The quick brown fox jumps."
    word_wave = word_cse.encode(example)
    
    print(f"\nInput: \"{example}\"")
    print(f"Words detected: {word_wave.num_words}")
    print(f"Wave shape: {word_wave.waves.shape}")
    print("\nWord boundaries:")
    
    for i, (word, (start, end)) in enumerate(zip(word_wave.words, word_wave.word_boundaries)):
        wave_norm = word_wave.waves[i].norm().item()
        print(f"  [{i}] '{word}' (bytes {start}-{end}) | wave norm: {wave_norm:.3f}")
    
    # Word similarity
    log.separator("Word Similarity Demo")
    
    word_pairs = [
        ("cat", "dog"),      # Related
        ("cat", "cats"),     # Morphological
        ("happy", "sad"),    # Antonyms
        ("king", "queen"),   # Related
        ("apple", "music"),  # Unrelated
    ]
    
    print("\nWord pair similarities (untrained model - for structure demo):")
    for w1, w2 in word_pairs:
        sim = word_cse.word_similarity(w1, w2)
        print(f"  {w1:>10} ↔ {w2:<10}: {sim:+.3f}")
    
    log.separator("Demo Complete")
    log.success("Word-level encoding working!")
    log.info("Run train_word_cse.py to train the model")
    
    return word_cse


if __name__ == '__main__':
    main()
