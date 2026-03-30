"""
FLUX-7B: Emission Head Training

Trains ONLY the emission head (200M params).
The field already has knowledge from injection.

This is fundamentally different from LLM training:
- LLM: Train 7B params to memorize knowledge AND generate
- FLUX: Train 200M params to generate from field knowledge

Training time: 2-3 days on 8× A100 (vs 3-4 weeks for full LLM)
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass
import time
from tqdm import tqdm

# Setup paths
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from resonance_field_7b import ResonanceField7B
from emission_head import EmissionHead
from flux_7b_model import FLUX7B, FLUX7BConfig, create_flux_7b_test


@dataclass
class TrainingConfig:
    """Configuration for emission training."""
    
    # Learning rate
    lr: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.01
    
    # Schedule
    warmup_steps: int = 1000
    total_steps: int = 100000
    
    # Batching
    batch_size: int = 8
    grad_accum: int = 4
    max_seq_len: int = 512
    
    # Regularization
    dropout: float = 0.1
    label_smoothing: float = 0.1
    
    # Logging
    log_every: int = 100
    eval_every: int = 1000
    checkpoint_every: int = 5000
    
    # Field queries
    k_attractors: int = 32


class EmissionTrainer:
    """
    Trainer for emission head.
    
    Only trains the 200M emission head parameters.
    Field and physics stack are frozen.
    """
    
    def __init__(
        self,
        system: FLUX7B,
        config: Optional[TrainingConfig] = None,
        checkpoint_dir: str = './checkpoints/flux_7b',
    ):
        self.system = system
        self.config = config or TrainingConfig()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Only emission head is trainable
        self.system.physics.freeze()
        self.system.train_mode()
        
        # Optimizer (only emission params)
        self.optimizer = AdamW(
            self.system.get_trainable_parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
        )
        
        # Scheduler with warmup
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.total_steps - self.config.warmup_steps,
            eta_min=self.config.min_lr,
        )
        
        # Tracking
        self.global_step = 0
        self.total_loss = 0.0
        self.start_time = None
        
    def train(
        self,
        train_data,
        eval_data=None,
    ):
        """
        Train emission head.
        
        Args:
            train_data: Iterator of (text, target_text) pairs
            eval_data: Optional evaluation data
        """
        self.start_time = time.time()
        
        device = self.system.device
        accum_loss = 0.0
        
        # Progress bar
        pbar = tqdm(total=self.config.total_steps, desc="Training")
        
        # Training loop
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_data):
            if self.global_step >= self.config.total_steps:
                break
                
            # Process batch
            loss = self._train_step(batch)
            accum_loss += loss.item()
            
            # Gradient accumulation
            loss = loss / self.config.grad_accum
            loss.backward()
            
            if (batch_idx + 1) % self.config.grad_accum == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.system.get_trainable_parameters(),
                    max_norm=1.0,
                )
                
                # Optimizer step
                self._adjust_learning_rate()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                self.total_loss += accum_loss / self.config.grad_accum
                
                # Logging
                if self.global_step % self.config.log_every == 0:
                    self._log_progress(pbar, accum_loss / self.config.grad_accum)
                    
                # Evaluation
                if eval_data and self.global_step % self.config.eval_every == 0:
                    self._evaluate(eval_data)
                    
                # Checkpoint
                if self.global_step % self.config.checkpoint_every == 0:
                    self._save_checkpoint()
                    
                accum_loss = 0.0
                pbar.update(1)
                
        # Final checkpoint
        self._save_checkpoint("final")
        self._print_summary()
        
    def _train_step(self, batch) -> torch.Tensor:
        """Single training step."""
        device = self.system.device
        
        # Unpack batch
        if isinstance(batch, tuple):
            prompt_text, target_text = batch
        else:
            prompt_text = target_text = batch
            
        # Encode prompt
        with torch.no_grad():
            prompt_waves = self.system.physics.encode(prompt_text)
            prompt_waves = prompt_waves.to(device)
            
            # Query field
            query_wave = self.system.physics.project_to_field(
                prompt_waves.mean(dim=0)
            )
            field_features, gravity_weights, _ = self.system.field.query(
                query_wave.to(device),
                k=self.config.k_attractors,
            )
            
        # Target bytes
        target_bytes = torch.tensor(
            list(target_text.encode('utf-8')[:self.config.max_seq_len]),
            dtype=torch.long,
            device=device,
        )
        
        # Forward
        logits = self.system.forward_train(
            prompt_waves.unsqueeze(0),
            field_features.unsqueeze(0),
            gravity_weights.unsqueeze(0),
            target_bytes.unsqueeze(0),
        )
        
        # Loss with label smoothing
        loss = F.cross_entropy(
            logits.view(-1, 256),
            target_bytes.view(-1),
            label_smoothing=self.config.label_smoothing,
        )
        
        return loss
        
    def _adjust_learning_rate(self):
        """Adjust LR with warmup."""
        if self.global_step < self.config.warmup_steps:
            # Linear warmup
            lr = self.config.lr * (self.global_step + 1) / self.config.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # Cosine decay
            self.scheduler.step()
            
    def _log_progress(self, pbar, loss):
        """Update progress bar."""
        elapsed = time.time() - self.start_time
        lr = self.optimizer.param_groups[0]['lr']
        avg_loss = self.total_loss / self.global_step if self.global_step > 0 else 0
        
        pbar.set_postfix({
            'loss': f'{loss:.4f}',
            'avg': f'{avg_loss:.4f}',
            'lr': f'{lr:.2e}',
        })
        
    def _evaluate(self, eval_data):
        """Run evaluation."""
        self.system.eval_mode()
        
        total_loss = 0.0
        n_samples = 0
        
        with torch.no_grad():
            for i, batch in enumerate(eval_data):
                if i >= 50:  # Limit eval samples
                    break
                    
                loss = self._train_step(batch)
                total_loss += loss.item()
                n_samples += 1
                
        avg_loss = total_loss / max(1, n_samples)
        print(f"\n    Eval loss: {avg_loss:.4f}")
        
        # Generate sample
        sample = self.system.generate("The capital of France is", max_length=50)
        print(f"    Sample: {sample[:100]}...")
        
        self.system.train_mode()
        
    def _save_checkpoint(self, name=None):
        """Save training checkpoint."""
        name = name or f"step{self.global_step}"
        path = self.checkpoint_dir / f"emission_{name}.pt"
        
        torch.save({
            'emission_state_dict': self.system.emission.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'total_loss': self.total_loss,
        }, path)
        
        print(f"\n    ✓ Checkpoint: {path}")
        
    def _print_summary(self):
        """Print training summary."""
        elapsed = time.time() - self.start_time
        avg_loss = self.total_loss / max(1, self.global_step)
        
        print("\n" + "=" * 60)
        print("  EMISSION TRAINING COMPLETE")
        print("=" * 60)
        print(f"  Total steps: {self.global_step:,}")
        print(f"  Final avg loss: {avg_loss:.4f}")
        print(f"  Time: {elapsed / 3600:.1f} hours")
        print(f"  Steps/hour: {self.global_step / (elapsed / 3600):.0f}")
        print("=" * 60)


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

def load_training_data(max_samples: Optional[int] = None):
    """Load training data for emission training."""
    from datasets import load_dataset
    
    print("  Loading training data...")
    
    # Multiple sources for diversity
    ds = load_dataset("openwebtext", split="train", streaming=True)
    
    count = 0
    for item in ds:
        text = item['text']
        if 50 < len(text) < 2000:  # Filter by length
            yield text
            count += 1
            if max_samples and count >= max_samples:
                break
                

def load_instruction_data():
    """Load instruction-following data."""
    from datasets import load_dataset
    
    print("  Loading instruction data...")
    
    # Alpaca for instructions
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    
    for item in ds:
        instruction = item['instruction']
        input_text = item.get('input', '')
        output = item['output']
        
        if input_text:
            prompt = f"Instruction: {instruction}\nInput: {input_text}\n\nResponse:"
        else:
            prompt = f"Instruction: {instruction}\n\nResponse:"
            
        yield (prompt, output)


# ─────────────────────────────────────────────
# Main Training Functions
# ─────────────────────────────────────────────

def train_emission_full(
    system: FLUX7B,
    checkpoint_dir: str = './checkpoints/flux_7b',
    total_steps: int = 100000,
):
    """Full emission training."""
    config = TrainingConfig(
        total_steps=total_steps,
        batch_size=8,
        grad_accum=4,
    )
    
    trainer = EmissionTrainer(system, config, checkpoint_dir)
    
    train_data = load_training_data()
    eval_data = list(load_training_data(max_samples=100))
    
    trainer.train(train_data, eval_data)
    
    # Save final system
    system.save(str(Path(checkpoint_dir) / 'flux_7b_trained'))
    

def train_emission_test(system: FLUX7B, n_steps: int = 100):
    """Quick test training."""
    config = TrainingConfig(
        total_steps=n_steps,
        batch_size=1,
        grad_accum=1,
        log_every=10,
    )
    
    trainer = EmissionTrainer(system, config, './checkpoints/flux_7b_test')
    
    # Simple test data
    test_texts = [
        "The capital of France is Paris.",
        "Python is a programming language.",
        "Machine learning is a subset of AI.",
        "The Earth orbits the Sun.",
    ] * 50  # Repeat for more steps
    
    trainer.train(iter(test_texts))


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='FLUX-7B Emission Training')
    parser.add_argument('--mode', choices=['full', 'test'], default='test')
    parser.add_argument('--checkpoint-dir', default='./checkpoints/flux_7b')
    parser.add_argument('--field-path', type=str, help='Path to injected field')
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    
    # Create or load system
    if args.mode == 'test':
        system = create_flux_7b_test(args.device)
        
        # Quick injection for testing
        print("  Injecting test data...")
        system.inject_document("Paris is the capital of France. It is known for the Eiffel Tower.")
        system.inject_document("Python is a programming language created by Guido van Rossum.")
        system.inject_document("Machine learning uses algorithms to learn from data.")
        system.field.settle(steps=20)
        
        train_emission_test(system, n_steps=args.steps if args.steps < 1000 else 100)
        
    else:
        from flux_7b_model import create_flux_7b_full
        system = create_flux_7b_full(args.device)
        
        # Load injected field if provided
        if args.field_path:
            system.field = ResonanceField7B.load(args.field_path, args.device)
            print(f"  ✓ Loaded field: {system.field.stats()}")
        else:
            print("  ⚠ No field provided. Run injection first!")
            print("  Usage: python inject_corpus.py --mode full")
            sys.exit(1)
            
        train_emission_full(system, args.checkpoint_dir, args.steps)
