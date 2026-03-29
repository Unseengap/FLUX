"""
Demo 2: Coherency Comparison — Byte-Level vs Word-Level

Shows why word-level encoding improves semantic coherency:
- Character fragments vs complete concepts
- Cleaner semantic clustering
- Better meaning preservation
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from flux_utils import get_device, PhaseLogger

sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1'))
from cse import ContinuousSemanticEncoder

sys.path.insert(0, str(Path(__file__).parent))
from word_cse import WordLevelCSE


def compute_semantic_dispersion(waves: torch.Tensor) -> float:
    """
    Measure how spread out the wave representations are.
    Higher dispersion = less coherent representation.
    """
    if waves.shape[0] < 2:
        return 0.0
    
    # Compute pairwise cosine similarities
    waves_norm = F.normalize(waves, dim=-1)
    sim_matrix = torch.mm(waves_norm, waves_norm.t())
    
    # Average off-diagonal similarity (exclude self-similarity)
    mask = 1 - torch.eye(sim_matrix.shape[0], device=sim_matrix.device)
    avg_sim = (sim_matrix * mask).sum() / mask.sum()
    
    # Dispersion = 1 - avg_similarity (higher = more spread out)
    return 1.0 - avg_sim.item()


def main():
    log = PhaseLogger(phase=8)
    log.separator("Demo: Coherency Comparison")
    
    device = get_device()
    log.info(f"Device: {device}")
    
    # Create encoders
    byte_cse = ContinuousSemanticEncoder(device=device).to(device)
    word_cse = WordLevelCSE(byte_cse=byte_cse, device=device).to(device)
    
    # Test case: "cat" as bytes vs as word
    log.separator("Example: 'cat' Representation")
    
    text = "cat"
    
    # Byte-level: 3 separate positions for c, a, t
    byte_wave = byte_cse.encode(text)
    print(f"\nByte-level encoding of 'cat':")
    print(f"  Positions: {byte_wave.seq_len} (one per character)")
    print(f"  Shape: {byte_wave.full.shape}")
    print(f"  Character waves:")
    for i, char in enumerate(text):
        wave_norm = byte_wave.full[i].norm().item()
        print(f"    [{i}] '{char}' norm={wave_norm:.3f}")
    
    # Word-level: 1 position for complete "cat"
    word_wave = word_cse.encode(text)
    print(f"\nWord-level encoding of 'cat':")
    print(f"  Positions: {word_wave.num_words} (complete concept)")
    print(f"  Shape: {word_wave.waves.shape}")
    print(f"  Word wave norm: {word_wave.waves[0].norm().item():.3f}")
    
    # Coherency analysis
    log.separator("Semantic Coherency Analysis")
    
    test_sentences = [
        "The cat sat on the mat.",
        "Machine learning enables artificial intelligence.",
        "Programming requires logical thinking and creativity.",
    ]
    
    print("\nDispersion scores (lower = more coherent):\n")
    print(f"{'Sentence':<50} | {'Byte':>8} | {'Word':>8} | {'Improve':>8}")
    print("-" * 82)
    
    for sentence in test_sentences:
        byte_wave = byte_cse.encode(sentence)
        word_wave = word_cse.encode(sentence)
        
        byte_disp = compute_semantic_dispersion(byte_wave.full)
        word_disp = compute_semantic_dispersion(word_wave.waves)
        improvement = ((byte_disp - word_disp) / byte_disp * 100) if byte_disp > 0 else 0
        
        display = sentence[:47] + "..." if len(sentence) > 50 else sentence
        print(f"{display:<50} | {byte_disp:>7.3f} | {word_disp:>7.3f} | {improvement:>+6.1f}%")
    
    # Concept clustering demo
    log.separator("Concept Clustering")
    
    concepts = {
        'animals': ['cat', 'dog', 'bird', 'fish'],
        'colors': ['red', 'blue', 'green', 'yellow'],
        'actions': ['run', 'jump', 'walk', 'swim'],
    }
    
    print("\nWithin-category similarity (word-level, untrained):")
    print("(Higher = words in same category are more similar)\n")
    
    for category, words in concepts.items():
        # Get word embeddings
        embeddings = []
        for word in words:
            emb = word_cse.get_word_embedding(word)
            embeddings.append(emb)
        
        embeddings = torch.stack(embeddings)
        
        # Compute average pairwise similarity
        embeddings_norm = F.normalize(embeddings, dim=-1)
        sim_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
        mask = 1 - torch.eye(len(words), device=sim_matrix.device)
        avg_sim = (sim_matrix * mask).sum() / mask.sum()
        
        print(f"  {category:<10}: {words} → avg sim: {avg_sim.item():.3f}")
    
    # Cross-category similarity (should be lower)
    print("\nCross-category similarity (should be lower):")
    
    all_words = []
    all_categories = []
    for cat, words in concepts.items():
        for word in words:
            all_words.append(word)
            all_categories.append(cat)
    
    embeddings = torch.stack([word_cse.get_word_embedding(w) for w in all_words])
    embeddings_norm = F.normalize(embeddings, dim=-1)
    sim_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
    
    # Average similarity for same vs different categories
    same_cat_sim = 0.0
    same_count = 0
    diff_cat_sim = 0.0
    diff_count = 0
    
    for i in range(len(all_words)):
        for j in range(i + 1, len(all_words)):
            sim = sim_matrix[i, j].item()
            if all_categories[i] == all_categories[j]:
                same_cat_sim += sim
                same_count += 1
            else:
                diff_cat_sim += sim
                diff_count += 1
    
    same_avg = same_cat_sim / max(same_count, 1)
    diff_avg = diff_cat_sim / max(diff_count, 1)
    
    print(f"  Same category avg: {same_avg:.3f}")
    print(f"  Diff category avg: {diff_avg:.3f}")
    print(f"  Separation: {same_avg - diff_avg:+.3f}")
    
    log.separator("Demo Complete")
    log.info("Note: Similarities improve significantly after training!")
    log.success("Coherency comparison complete")


if __name__ == '__main__':
    main()
