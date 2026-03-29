"""
Training script for WordLevelCSE.

Trains the word pooling attention to:
1. Reconstruct original bytes from word waves
2. Produce similar waves for similar words
3. Produce different waves for different words

Uses WikiText or similar data for training.
"""

import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from flux_utils import (
    get_device, save_checkpoint, load_checkpoint,
    PhaseLogger, PhaseResults, checkpoint_exists
)

sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1'))
from cse import ContinuousSemanticEncoder

sys.path.insert(0, str(Path(__file__).parent))
from word_cse import WordLevelCSE, WordDecoder, WordWave


# ─────────────────────────────────────────────
# Training Configuration
# ─────────────────────────────────────────────

TRAIN_CONFIG = {
    'batch_size': 32,
    'learning_rate': 1e-4,
    'epochs': 10,
    'max_word_len': 32,
    'warmup_steps': 100,
    'contrastive_weight': 0.1,
    'reconstruction_weight': 1.0,
}


# ─────────────────────────────────────────────
# Training Data
# ─────────────────────────────────────────────

def load_training_sentences(max_samples: int = 10000) -> list:
    """
    Load training sentences. Uses WikiText if available,
    otherwise falls back to simple English sentences.
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        sentences = []
        for item in dataset:
            text = item['text'].strip()
            if len(text) > 20:  # Skip very short lines
                sentences.append(text)
            if len(sentences) >= max_samples:
                break
        print(f"  ✓ Loaded {len(sentences)} sentences from WikiText")
        return sentences
    except Exception as e:
        print(f"  ⚠ WikiText unavailable ({e}), using fallback data")
        return get_fallback_sentences(max_samples)


def get_fallback_sentences(n: int = 1000) -> list:
    """Fallback training sentences for offline development."""
    base_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world of artificial intelligence.",
        "Natural language processing enables computers to understand human speech.",
        "Deep neural networks can learn complex patterns from data.",
        "The cat sat on the mat and watched the birds outside.",
        "Programming languages like Python make coding accessible to everyone.",
        "Science and technology drive innovation in modern society.",
        "Climate change is one of the most pressing issues of our time.",
        "Music and art express the deepest human emotions and experiences.",
        "Mathematics provides the foundation for understanding the universe.",
        "The ocean covers more than seventy percent of the Earth.",
        "Books have been a source of knowledge for thousands of years.",
        "Healthy eating and regular exercise improve quality of life.",
        "Space exploration continues to reveal mysteries of the cosmos.",
        "Cities around the world are becoming more sustainable and green.",
        "Education is the key to unlocking human potential.",
        "Technology connects people across continents in real time.",
        "History teaches us valuable lessons about human civilization.",
        "Philosophy asks fundamental questions about existence and meaning.",
        "Innovation requires creativity, persistence, and collaboration.",
    ]
    
    # Generate variations
    sentences = []
    for i in range(n):
        base = base_sentences[i % len(base_sentences)]
        if i >= len(base_sentences):
            # Add some variation
            words = base.split()
            if len(words) > 3:
                # Shuffle or modify slightly
                import random
                random.seed(i)
                if random.random() < 0.3:
                    words = words[:-1]  # Remove last word
                elif random.random() < 0.5:
                    words = words[1:]  # Remove first word
            base = ' '.join(words)
        sentences.append(base)
    
    return sentences


# ─────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────

def reconstruction_loss(
    word_waves: torch.Tensor,  # [num_words, wave_dim]
    target_words: list,        # List of word strings
    decoder: WordDecoder,
    max_len: int = 32,
) -> torch.Tensor:
    """
    Compute reconstruction loss: word wave → original bytes.
    """
    if len(target_words) == 0:
        return torch.tensor(0.0, device=word_waves.device)
    
    total_loss = 0.0
    count = 0
    
    for i, word in enumerate(target_words):
        if i >= word_waves.shape[0]:
            break
            
        # Get target bytes
        word_bytes = list(word.encode('utf-8'))
        if len(word_bytes) == 0:
            continue
            
        # Pad/truncate to max_len
        if len(word_bytes) > max_len:
            word_bytes = word_bytes[:max_len]
        target = torch.tensor(word_bytes, dtype=torch.long, device=word_waves.device)
        target_padded = F.pad(target, (0, max_len - len(word_bytes)), value=0)
        
        # Decode
        word_wave = word_waves[i]
        logits = decoder(word_wave, target_bytes=target_padded.unsqueeze(0), max_len=max_len)
        logits = logits.squeeze(0)  # [max_len, num_bytes]
        
        # Compute loss only on actual bytes (not padding)
        actual_len = len(word_bytes)
        loss = F.cross_entropy(logits[:actual_len], target[:actual_len])
        total_loss += loss
        count += 1
    
    if count == 0:
        return torch.tensor(0.0, device=word_waves.device)
    
    return total_loss / count


def contrastive_loss(
    word_waves: torch.Tensor,  # [num_words, wave_dim]
    words: list,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Contrastive loss: same words should be similar, different words different.
    
    Uses in-batch negatives: each word is a positive pair with itself,
    and negative with all other words in the batch.
    """
    if word_waves.shape[0] < 2:
        return torch.tensor(0.0, device=word_waves.device)
    
    # Normalize waves
    waves_norm = F.normalize(word_waves, dim=-1)
    
    # Compute similarity matrix
    sim_matrix = torch.mm(waves_norm, waves_norm.t()) / temperature
    
    # Diagonal elements are self-similarities (should be high)
    # Off-diagonal elements are different words (should be low for InfoNCE)
    
    # InfoNCE loss: maximize diagonal, minimize off-diagonal
    labels = torch.arange(sim_matrix.shape[0], device=sim_matrix.device)
    loss = F.cross_entropy(sim_matrix, labels)
    
    return loss


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────

def train_word_cse(
    device: str = None,
    epochs: int = TRAIN_CONFIG['epochs'],
    batch_size: int = TRAIN_CONFIG['batch_size'],
    learning_rate: float = TRAIN_CONFIG['learning_rate'],
    max_samples: int = 5000,
    log: PhaseLogger = None,
):
    """
    Train the WordLevelCSE model.
    """
    if device is None:
        device = get_device()
    
    if log is None:
        log = PhaseLogger(phase=8)
    
    log.separator("Training Word-Level CSE")
    log.info(f"Device: {device}")
    log.info(f"Epochs: {epochs}, Batch size: {batch_size}")
    
    # Load or create byte-level CSE
    log.info("Loading Phase 1 CSE...")
    try:
        phase1_ckpt = load_checkpoint(1)
        byte_cse = ContinuousSemanticEncoder(device=device)
        byte_cse.load_state_dict(phase1_ckpt['state_dict'])
        byte_cse = byte_cse.to(device)
        byte_cse.eval()  # Freeze byte CSE
        log.success("Loaded Phase 1 checkpoint")
    except FileNotFoundError:
        log.warning("Phase 1 checkpoint not found, using untrained CSE")
        byte_cse = ContinuousSemanticEncoder(device=device).to(device)
    
    # Freeze byte CSE
    for param in byte_cse.parameters():
        param.requires_grad = False
    
    # Create word-level CSE
    word_cse = WordLevelCSE(byte_cse=byte_cse, device=device).to(device)
    decoder = WordDecoder(wave_dim=432, max_word_len=32).to(device)
    
    # Optimizer for pooling + decoder only
    params = list(word_cse.pooling.parameters()) + list(decoder.parameters())
    optimizer = AdamW(params, lr=learning_rate, weight_decay=0.01)
    
    # Load training data
    log.info("Loading training data...")
    sentences = load_training_sentences(max_samples=max_samples)
    
    # Training loop
    best_loss = float('inf')
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_contrast = 0.0
        num_batches = 0
        
        # Shuffle sentences
        import random
        random.shuffle(sentences)
        
        pbar = tqdm(range(0, len(sentences), batch_size), desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_start in pbar:
            batch_sentences = sentences[batch_start:batch_start + batch_size]
            
            batch_loss = torch.tensor(0.0, device=device)
            batch_recon = torch.tensor(0.0, device=device)
            batch_contrast = torch.tensor(0.0, device=device)
            
            for sentence in batch_sentences:
                try:
                    # Encode to word waves
                    word_wave = word_cse.encode(sentence)
                    
                    if word_wave.num_words == 0:
                        continue
                    
                    # Reconstruction loss
                    recon = reconstruction_loss(
                        word_wave.waves, word_wave.words, decoder
                    )
                    batch_recon += recon
                    
                    # Contrastive loss
                    contrast = contrastive_loss(word_wave.waves, word_wave.words)
                    batch_contrast += contrast
                    
                except Exception as e:
                    # Skip problematic sentences
                    continue
            
            # Combine losses
            total_recon = batch_recon * TRAIN_CONFIG['reconstruction_weight']
            total_contrast = batch_contrast * TRAIN_CONFIG['contrastive_weight']
            batch_loss = total_recon + total_contrast
            
            if batch_loss.item() > 0:
                optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                
                epoch_loss += batch_loss.item()
                epoch_recon += batch_recon.item()
                epoch_contrast += batch_contrast.item()
                num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{batch_loss.item():.4f}',
                'recon': f'{batch_recon.item():.4f}',
            })
        
        # Epoch summary
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            avg_recon = epoch_recon / num_batches
            avg_contrast = epoch_contrast / num_batches
            losses.append(avg_loss)
            
            log.metric("epoch", f"{epoch+1}/{epochs}")
            log.metric("loss", f"{avg_loss:.4f}")
            log.metric("recon", f"{avg_recon:.4f}")
            log.metric("contrast", f"{avg_contrast:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                log.success(f"New best loss: {best_loss:.4f}")
    
    # Save checkpoint
    log.separator("Saving Checkpoint")
    
    checkpoint = {
        'phase': 8,
        'subphase': 'word',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'wave_dim': 432,
            'interference_radius': 3,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
        },
        'state_dict': {
            'pooling': word_cse.pooling.state_dict(),
            'decoder': decoder.state_dict(),
        },
        'metrics': {
            'final_loss': losses[-1] if losses else 0.0,
            'best_loss': best_loss,
            'losses': losses,
        },
    }
    
    # Save to checkpoint directory
    checkpoint_dir = Path(__file__).parent.parent.parent / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / 'phase8_word.phase.pt'
    torch.save(checkpoint, checkpoint_path)
    
    log.success(f"Saved checkpoint: {checkpoint_path}")
    
    return word_cse, decoder, checkpoint


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == '__main__':
    log = PhaseLogger(phase=8)
    log.separator("Phase 8-Word: Word-Level CSE Training")
    
    # Train
    word_cse, decoder, ckpt = train_word_cse(log=log)
    
    log.separator("Training Complete")
    log.success(f"Final loss: {ckpt['metrics']['final_loss']:.4f}")
    log.success(f"Best loss: {ckpt['metrics']['best_loss']:.4f}")
