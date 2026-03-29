"""
Test 1: Word-Level CSE Basic Functionality

Tests:
1. Word boundary detection accuracy
2. Encoding produces valid waves
3. Sequence compression ratio
4. Word wave shapes are correct
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from flux_utils import get_device, PhaseLogger, PhaseResults

sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1'))
from cse import ContinuousSemanticEncoder
from wave_types import TOTAL_WAVE_DIM

sys.path.insert(0, str(Path(__file__).parent))
from word_cse import WordLevelCSE, WordBoundaryDetector


def test_word_boundary_detection():
    """Test that word boundaries are detected correctly."""
    detector = WordBoundaryDetector()
    
    test_cases = [
        ("Hello world", ["Hello", "world"]),
        ("The cat sat.", ["The", "cat", "sat", "."]),
        ("I love Python!", ["I", "love", "Python", "!"]),
        ("a", ["a"]),
        ("", []),
        ("word", ["word"]),
        ("multiple   spaces", ["multiple", "spaces"]),  # Multiple spaces
        ("end.", ["end", "."]),
        ("日本", ["日", "本"]),  # CJK characters
    ]
    
    passed = 0
    failed = 0
    
    for text, expected_words in test_cases:
        words, spans = detector.detect(text)
        
        if words == expected_words:
            passed += 1
        else:
            print(f"  ✗ '{text}': expected {expected_words}, got {words}")
            failed += 1
    
    return passed, failed


def test_encoding_produces_valid_waves():
    """Test that encoding produces correctly shaped waves."""
    device = get_device()
    byte_cse = ContinuousSemanticEncoder(device=device).to(device)
    word_cse = WordLevelCSE(byte_cse=byte_cse, device=device).to(device)
    
    test_cases = [
        ("Hello world", 2),      # 2 words
        ("The quick brown fox", 4),  # 4 words
        ("a", 1),                # 1 word
        ("Hello, world!", 4),    # Hello , world !
    ]
    
    passed = 0
    failed = 0
    
    for text, expected_words in test_cases:
        wave = word_cse.encode(text)
        
        # Check number of words
        if wave.num_words != expected_words:
            print(f"  ✗ '{text}': expected {expected_words} words, got {wave.num_words}")
            failed += 1
            continue
        
        # Check wave shape
        expected_shape = (expected_words, TOTAL_WAVE_DIM)
        if wave.waves.shape != expected_shape:
            print(f"  ✗ '{text}': expected shape {expected_shape}, got {wave.waves.shape}")
            failed += 1
            continue
        
        # Check waves are not all zeros
        if wave.waves.abs().sum() == 0:
            print(f"  ✗ '{text}': wave is all zeros")
            failed += 1
            continue
        
        # Check no NaN values
        if wave.waves.isnan().any():
            print(f"  ✗ '{text}': wave contains NaN")
            failed += 1
            continue
        
        passed += 1
    
    return passed, failed


def test_compression_ratio():
    """Test that word-level encoding compresses sequences significantly."""
    device = get_device()
    byte_cse = ContinuousSemanticEncoder(device=device).to(device)
    word_cse = WordLevelCSE(byte_cse=byte_cse, device=device).to(device)
    
    # Test with various length texts
    test_cases = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world of artificial intelligence.",
        "This is a longer sentence that should demonstrate significant compression.",
    ]
    
    total_bytes = 0
    total_words = 0
    
    for text in test_cases:
        byte_wave = byte_cse.encode(text)
        word_wave = word_cse.encode(text)
        
        total_bytes += byte_wave.seq_len
        total_words += word_wave.num_words
    
    compression_ratio = total_bytes / max(total_words, 1)
    
    # Should achieve at least 4x compression on average English text
    passed = compression_ratio >= 4.0
    
    return (1, 0) if passed else (0, 1), compression_ratio


def test_empty_input():
    """Test handling of empty input."""
    device = get_device()
    byte_cse = ContinuousSemanticEncoder(device=device).to(device)
    word_cse = WordLevelCSE(byte_cse=byte_cse, device=device).to(device)
    
    wave = word_cse.encode("")
    
    passed = (
        wave.num_words == 0 and
        wave.waves.shape[0] == 0 and
        len(wave.words) == 0
    )
    
    return (1, 0) if passed else (0, 1)


def test_word_similarity_basic():
    """Test that word_similarity function works."""
    device = get_device()
    byte_cse = ContinuousSemanticEncoder(device=device).to(device)
    word_cse = WordLevelCSE(byte_cse=byte_cse, device=device).to(device)
    
    # Same word should have high similarity
    sim_same = word_cse.word_similarity("cat", "cat")
    
    # Different words should have some value (may not be very different untrained)
    sim_diff = word_cse.word_similarity("cat", "elephant")
    
    # Self-similarity should be 1.0 (or very close)
    passed = abs(sim_same - 1.0) < 0.01
    
    return (1, 0) if passed else (0, 1), sim_same


def main():
    log = PhaseLogger(phase=8)
    log.separator("Test: Word-Level CSE Functionality")
    
    results = PhaseResults(phase=8, component_name="Word-Level CSE")
    device = get_device()
    log.info(f"Device: {device}")
    
    # Test 1: Word boundary detection
    log.info("Testing word boundary detection...")
    passed, failed = test_word_boundary_detection()
    total = passed + failed
    success = failed == 0
    results.add_test(
        "Word Boundary Detection",
        passed=success,
        score=passed / total if total > 0 else 0.0,
        threshold=1.0,
    )
    log.metric("boundary_accuracy", f"{passed}/{total}")
    if success:
        log.success("Word boundary detection passed")
    else:
        log.error(f"Word boundary detection failed: {failed} failures")
    
    # Test 2: Encoding produces valid waves
    log.info("Testing wave encoding...")
    passed, failed = test_encoding_produces_valid_waves()
    total = passed + failed
    success = failed == 0
    results.add_test(
        "Valid Wave Encoding",
        passed=success,
        score=passed / total if total > 0 else 0.0,
        threshold=1.0,
    )
    if success:
        log.success("Wave encoding passed")
    else:
        log.error(f"Wave encoding failed: {failed} failures")
    
    # Test 3: Compression ratio
    log.info("Testing compression ratio...")
    (passed, failed), ratio = test_compression_ratio()
    success = passed > 0
    results.add_test(
        "Compression Ratio",
        passed=success,
        score=ratio,
        threshold=4.0,
    )
    log.metric("compression_ratio", f"{ratio:.1f}x")
    if success:
        log.success(f"Compression ratio: {ratio:.1f}x (threshold: 4x)")
    else:
        log.error(f"Compression ratio too low: {ratio:.1f}x")
    
    # Test 4: Empty input handling
    log.info("Testing empty input...")
    passed, failed = test_empty_input()
    success = passed > 0
    results.add_test(
        "Empty Input Handling",
        passed=success,
    )
    if success:
        log.success("Empty input handling passed")
    else:
        log.error("Empty input handling failed")
    
    # Test 5: Word similarity function
    log.info("Testing word similarity...")
    (passed, failed), self_sim = test_word_similarity_basic()
    success = passed > 0
    results.add_test(
        "Word Similarity Function",
        passed=success,
        score=self_sim,
        threshold=0.99,
    )
    if success:
        log.success(f"Word self-similarity: {self_sim:.4f}")
    else:
        log.error(f"Word similarity test failed")
    
    # Save results
    log.separator("Test Results")
    results.save()
    
    all_passed = results.all_passed()
    if all_passed:
        log.success("All tests passed!")
    else:
        log.error("Some tests failed")
    
    return results


if __name__ == '__main__':
    results = main()
    sys.exit(0 if results.all_passed() else 1)
