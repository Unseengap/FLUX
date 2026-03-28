"""
Phase 9 ARC: Training Pipeline

Train GridToWave/WaveToGrid adapters and rule inducer on ARC dataset.

Training stages:
1. Grid reconstruction — encode/decode grids accurately
2. Delta embedding — learn transformation representations
3. Rule classification — classify deltas to patterns
4. End-to-end — full pipeline optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
from typing import Dict, List, Tuple, Optional, Any
import sys
from pathlib import Path
from tqdm import tqdm
import time
import json

# Add project paths
_PHASES_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PHASES_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PHASES_DIR / 'phase8_8') not in sys.path:
    sys.path.insert(0, str(_PHASES_DIR / 'phase8_8'))

from flux_utils import get_device, PhaseLogger
from arc_loader import ARCTask, ARCExample, ARCDataset, load_arc_agi, generate_synthetic_tasks
from pattern_library import PATTERNS
from rule_inducer import RuleInducer, WaveRuleInducer

# Try to import grid adapters
try:
    from grid_adapters import GridToWave, WaveToGrid
    HAS_GRID_ADAPTERS = True
except ImportError:
    HAS_GRID_ADAPTERS = False
    print("⚠ Grid adapters not available")


# ─────────────────────────────────────────────
# Dataset for Training
# ─────────────────────────────────────────────

class ARCPairDataset(Dataset):
    """Dataset of input/output pairs for training."""
    
    def __init__(self, tasks: List[ARCTask], max_size: int = 30):
        self.pairs = []
        self.max_size = max_size
        
        for task in tasks:
            for ex in task.train:
                self.pairs.append((
                    self._to_tensor(ex.input),
                    self._to_tensor(ex.output),
                    task.task_id,
                ))
    
    def _to_tensor(self, grid: List[List[int]]) -> Tensor:
        """Convert grid to padded tensor."""
        t = torch.tensor(grid, dtype=torch.long)
        h, w = t.shape
        
        # Pad to max_size
        if h < self.max_size or w < self.max_size:
            padded = torch.zeros(self.max_size, self.max_size, dtype=torch.long)
            padded[:h, :w] = t
            return padded
        
        return t[:self.max_size, :self.max_size]
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, str]:
        return self.pairs[idx]


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────

class ARCTrainer:
    """Train ARC components."""
    
    def __init__(
        self,
        wave_dim: int = 432,
        device: str = 'cpu',
        lr: float = 1e-3,
    ):
        self.wave_dim = wave_dim
        self.device = device
        
        # Initialize models
        if HAS_GRID_ADAPTERS:
            self.grid_encoder = GridToWave(wave_dim=wave_dim, device=device)
            self.grid_decoder = WaveToGrid(wave_dim=wave_dim, device=device)
        else:
            self.grid_encoder = None
            self.grid_decoder = None
        
        self.wave_inducer = WaveRuleInducer(wave_dim=wave_dim, device=device)
        
        # Optimizer
        params = list(self.wave_inducer.parameters())
        if self.grid_encoder:
            params += list(self.grid_encoder.parameters())
        if self.grid_decoder:
            params += list(self.grid_decoder.parameters())
        
        self.optimizer = torch.optim.Adam(params, lr=lr)
        
        # Loss functions
        self.grid_loss = nn.CrossEntropyLoss()
        self.pattern_loss = nn.CrossEntropyLoss()
    
    def train_reconstruction(
        self,
        dataset: ARCPairDataset,
        epochs: int = 10,
        batch_size: int = 16,
    ) -> Dict[str, float]:
        """
        Stage 1: Train grid reconstruction (encode → decode accuracy).
        """
        if not self.grid_encoder or not self.grid_decoder:
            print("⚠ Grid adapters not available, skipping reconstruction training")
            return {"skip": True}
        
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        total_loss = 0
        total_correct = 0
        total_cells = 0
        
        self.grid_encoder.train()
        self.grid_decoder.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for input_grid, output_grid, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
                input_grid = input_grid.to(self.device)
                
                # Encode and decode each grid
                batch_loss = 0
                for i in range(input_grid.shape[0]):
                    grid = input_grid[i]
                    
                    # Find actual content (non-padded)
                    nonzero = torch.nonzero(grid)
                    if len(nonzero) == 0:
                        continue
                    h = nonzero[:, 0].max().item() + 1
                    w = nonzero[:, 1].max().item() + 1
                    grid = grid[:h, :w]
                    
                    # Encode to wave
                    wave = self.grid_encoder.encode(grid, mode='holistic')
                    
                    # Decode back
                    # (WaveToGrid needs training to work properly)
                    # For now, just measure encoding quality
                    
                self.optimizer.zero_grad()
                # batch_loss.backward()
                # self.optimizer.step()
                
                epoch_loss += batch_loss if isinstance(batch_loss, float) else 0
            
            print(f"  Epoch {epoch+1} loss: {epoch_loss:.4f}")
        
        return {
            "final_loss": epoch_loss,
            "epochs": epochs,
        }
    
    def train_delta_classifier(
        self,
        tasks: List[ARCTask],
        epochs: int = 20,
    ) -> Dict[str, float]:
        """
        Stage 2: Train delta → pattern classifier.
        
        For each task where we know the pattern, train the classifier
        to predict the correct pattern from the delta wave.
        """
        # Generate labeled training data
        # (delta_wave, pattern_id) pairs
        training_data = []
        
        rule_inducer = RuleInducer(use_waves=False, device=self.device)
        
        for task in tqdm(tasks, desc="Generating training data"):
            # Get ground truth (heuristic match)
            hypotheses = rule_inducer.induce(task)
            
            if hypotheses and hypotheses[0].confidence > 0.9:
                # This task has a clear pattern
                pattern_ids = hypotheses[0].pattern_ids
                if len(pattern_ids) == 1 and pattern_ids[0] in PATTERNS:
                    pattern_idx = list(PATTERNS.keys()).index(pattern_ids[0])
                    
                    # Compute delta waves
                    for ex in task.train:
                        input_t = torch.tensor(ex.input, dtype=torch.long, device=self.device)
                        output_t = torch.tensor(ex.output, dtype=torch.long, device=self.device)
                        
                        _, _, delta = self.wave_inducer.encode_example(input_t, output_t)
                        training_data.append((delta, pattern_idx))
        
        if not training_data:
            print("⚠ No labeled training data generated")
            return {"skip": True}
        
        print(f"  Generated {len(training_data)} labeled examples")
        
        # Train classifier
        self.wave_inducer.train()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            
            for delta, pattern_idx in training_data:
                self.optimizer.zero_grad()
                
                logits = self.wave_inducer.delta_classifier(delta)
                target = torch.tensor([pattern_idx], device=self.device)
                
                loss = self.pattern_loss(logits.unsqueeze(0), target)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                pred = logits.argmax().item()
                if pred == pattern_idx:
                    correct += 1
            
            acc = correct / len(training_data)
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}: loss={total_loss/len(training_data):.4f}, acc={acc*100:.1f}%")
        
        return {
            "final_accuracy": correct / len(training_data),
            "epochs": epochs,
            "training_samples": len(training_data),
        }


# ─────────────────────────────────────────────
# Full Training Pipeline
# ─────────────────────────────────────────────

def train_arc_system(
    n_synthetic: int = 100,
    n_real: int = 0,
    epochs: int = 20,
    device: str = None,
):
    """
    Full training pipeline for ARC system.
    """
    print("ARC System Training")
    print("=" * 50)
    
    if device is None:
        device = get_device()
    print(f"Device: {device}")
    
    # Generate training data
    print("\nLoading data...")
    tasks = []
    
    if n_synthetic > 0:
        synthetic = generate_synthetic_tasks(n_synthetic)
        tasks.extend(synthetic.tasks)
        print(f"  Synthetic tasks: {len(synthetic)}")
    
    if n_real > 0:
        try:
            real = load_arc_agi(version=2, split='training', max_tasks=n_real)
            tasks.extend(real.tasks)
            print(f"  Real ARC tasks: {len(real)}")
        except Exception as e:
            print(f"  ⚠ Failed to load real ARC: {e}")
    
    print(f"  Total tasks: {len(tasks)}")
    
    # Initialize trainer
    trainer = ARCTrainer(device=device)
    
    # Stage 1: Reconstruction (if adapters available)
    print("\n" + "=" * 50)
    print("Stage 1: Grid Reconstruction")
    print("=" * 50)
    
    pair_dataset = ARCPairDataset(tasks)
    recon_results = trainer.train_reconstruction(pair_dataset, epochs=min(epochs, 5))
    print(f"Results: {recon_results}")
    
    # Stage 2: Delta classifier
    print("\n" + "=" * 50)
    print("Stage 2: Delta Classifier")
    print("=" * 50)
    
    delta_results = trainer.train_delta_classifier(tasks, epochs=epochs)
    print(f"Results: {delta_results}")
    
    # Save trained models
    print("\n" + "=" * 50)
    print("Saving models...")
    
    checkpoint_dir = _PROJECT_ROOT / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    state = {
        'wave_inducer': trainer.wave_inducer.state_dict(),
        'training_results': {
            'reconstruction': recon_results,
            'delta_classifier': delta_results,
        },
    }
    
    if trainer.grid_encoder:
        state['grid_encoder'] = trainer.grid_encoder.state_dict()
    if trainer.grid_decoder:
        state['grid_decoder'] = trainer.grid_decoder.state_dict()
    
    save_path = checkpoint_dir / 'arc_trained.pt'
    torch.save(state, save_path)
    print(f"  ✓ Saved to {save_path}")
    
    return {
        'reconstruction': recon_results,
        'delta_classifier': delta_results,
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == '__main__':
    results = train_arc_system(
        n_synthetic=50,
        n_real=0,
        epochs=10,
    )
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
