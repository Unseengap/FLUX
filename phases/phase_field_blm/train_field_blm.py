"""
Training script for Field-Based BLM.

Key differences from traditional training:
- NO EPOCHS (single pass through data)
- NO BACKPROPAGATION (thermodynamic settling)
- NO OPTIMIZER (learning is just field deposits)
- NO GRADIENT ACCUMULATION (no gradients!)

The only thing trained is the byte embedding (~100K params),
which can optionally use gradient descent for warmup,
then switches to pure field-based learning.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import time
import argparse
from typing import Dict, List, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from flux_utils import get_device, PhaseLogger

from field_blm import FieldBLM, FieldBLMConfig


class TextDataset(Dataset):
    """Simple dataset for byte sequences."""
    
    def __init__(self, texts: List[str], max_len: int = 256):
        self.texts = texts
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        bytes_seq = list(text.encode('utf-8'))[:self.max_len]
        return torch.tensor(bytes_seq, dtype=torch.long)


def load_wikitext(split: str = 'train', max_samples: int = None) -> List[str]:
    """Load WikiText-103 dataset."""
    from datasets import load_dataset
    
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split)
    
    texts = []
    for item in dataset:
        text = item['text'].strip()
        if len(text) > 50:  # Skip short texts
            texts.append(text)
        if max_samples and len(texts) >= max_samples:
            break
    
    return texts


def train(
    model: FieldBLM,
    texts: List[str],
    log: PhaseLogger,
    checkpoint_every: int = 1000,
    checkpoint_dir: Path = None,
    warmup_steps: int = 0,
) -> Dict:
    """
    Train Field-Based BLM.
    
    SINGLE PASS - no epochs!
    
    Args:
        model: FieldBLM model
        texts: List of training texts
        log: Logger
        checkpoint_every: Save checkpoint every N sequences
        checkpoint_dir: Where to save checkpoints
        warmup_steps: Optional gradient warmup for embeddings
        
    Returns:
        Training statistics
    """
    device = model.device
    
    # Optional: warmup the embedding with gradient descent
    if warmup_steps > 0:
        log.info(f"Embedding warmup: {warmup_steps} steps with gradient descent")
        optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=1e-3)
        
        for step, text in enumerate(texts[:warmup_steps]):
            if len(text) < 10:
                continue
            
            bytes_seq = torch.tensor(list(text.encode('utf-8')), dtype=torch.long, device=device)
            
            # Simple contrastive loss on embeddings
            waves = model.encoder(bytes_seq.unsqueeze(0))[0]  # [seq, wave_dim]
            
            # Adjacent bytes should have similar waves
            loss = F.mse_loss(waves[:-1], waves[1:])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (step + 1) % 100 == 0:
                log.metric("warmup_step", f"{step + 1}/{warmup_steps}")
                log.metric("warmup_loss", f"{loss.item():.4f}")
        
        log.success("Embedding warmup complete")
    
    # Main training: single pass, no backprop
    log.separator("Main Training (Single Pass, No Backprop)")
    
    total_correct = 0
    total_bytes = 0
    start_time = time.time()
    
    for i, text in enumerate(texts):
        if len(text) < 10:
            continue
        
        # Train on this sequence (NO BACKPROP)
        result = model.train_on_text(text)
        
        total_correct += result.get('correct', 0)
        total_bytes += result.get('total', 0)
        
        # Logging
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            accuracy = total_correct / max(1, total_bytes)
            bytes_per_sec = total_bytes / elapsed
            
            log.metric("sequences", f"{i + 1:,}/{len(texts):,}")
            log.metric("accuracy", f"{accuracy:.2%}")
            log.metric("bytes_seen", f"{total_bytes:,}")
            log.metric("attractors", f"{model.field.unique_attractors:,}")
            log.metric("bytes/sec", f"{bytes_per_sec:.0f}")
        
        # Checkpoint
        if checkpoint_dir and (i + 1) % checkpoint_every == 0:
            ckpt_path = checkpoint_dir / f"field_blm_step_{i+1}.pt"
            model.save(str(ckpt_path))
            log.success(f"Checkpoint saved: {ckpt_path}")
    
    # Final stats
    elapsed = time.time() - start_time
    final_accuracy = total_correct / max(1, total_bytes)
    
    return {
        'total_sequences': len(texts),
        'total_bytes': total_bytes,
        'total_correct': total_correct,
        'accuracy': final_accuracy,
        'elapsed_seconds': elapsed,
        'bytes_per_second': total_bytes / elapsed,
        'unique_attractors': model.field.unique_attractors,
    }


def main():
    parser = argparse.ArgumentParser(description='Train Field-Based BLM')
    parser.add_argument('--max-samples', type=int, default=10000, help='Max training samples')
    parser.add_argument('--warmup-steps', type=int, default=100, help='Embedding warmup steps')
    parser.add_argument('--checkpoint-every', type=int, default=1000, help='Checkpoint frequency')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/field_blm', help='Checkpoint directory')
    parser.add_argument('--wave-dim', type=int, default=432, help='Wave dimension')
    parser.add_argument('--context-window', type=int, default=64, help='Context window size')
    parser.add_argument('--field-size', type=int, default=64, help='Field dimension size')
    args = parser.parse_args()
    
    # Setup
    log = PhaseLogger('field_blm')
    device = get_device()
    
    log.separator("Field-Based BLM Training")
    log.info(f"Device: {device}")
    log.info(f"Max samples: {args.max_samples:,}")
    log.info(f"Warmup steps: {args.warmup_steps}")
    
    # Create model
    config = FieldBLMConfig(
        wave_dim=args.wave_dim,
        context_window=args.context_window,
        field_dims=(args.field_size, args.field_size, args.field_size),
    )
    model = FieldBLM(config).to(device)
    
    log.info(f"Model parameters: {model.num_parameters:,}")
    log.info(f"Field size: {args.field_size}^3 = {args.field_size**3:,}")
    
    # Load data
    log.info("Loading WikiText-103...")
    texts = load_wikitext('train', max_samples=args.max_samples)
    log.info(f"Loaded {len(texts):,} texts")
    
    # Create checkpoint dir
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Train
    results = train(
        model=model,
        texts=texts,
        log=log,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=checkpoint_dir,
        warmup_steps=args.warmup_steps,
    )
    
    # Save final model
    final_path = checkpoint_dir / "field_blm_final.pt"
    model.save(str(final_path))
    log.success(f"Final model saved: {final_path}")
    
    # Print results
    log.separator("Training Complete")
    log.metric("Final accuracy", f"{results['accuracy']:.2%}")
    log.metric("Total bytes", f"{results['total_bytes']:,}")
    log.metric("Unique attractors", f"{results['unique_attractors']:,}")
    log.metric("Training time", f"{results['elapsed_seconds']:.1f}s")
    log.metric("Throughput", f"{results['bytes_per_second']:.0f} bytes/sec")
    
    # Test generation
    log.separator("Generation Test")
    prompts = ["The ", "Hello ", "In the "]
    for prompt in prompts:
        output = model.generate(prompt, max_bytes=50)
        log.info(f"'{prompt}' → '{output}'")


if __name__ == '__main__':
    main()
