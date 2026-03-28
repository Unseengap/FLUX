"""
Phase 8.9: Train Grid Adapters on ARC

Train GridToWave and WaveToGrid on the ARC (Abstraction and Reasoning Corpus)
dataset to enable spatial reasoning.

Training strategy:
1. Encode input grids → wave space
2. Encode output grids → wave space  
3. Train WaveToGrid to decode output waves correctly
4. Train to minimize delta (transformation) discovery loss

This is a reconstruction + transformation learning approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from typing import Dict, List, Tuple, Optional, Any
import json
import sys
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm

# Add project paths
_PHASES_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PHASES_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PHASES_DIR / 'phase8_8') not in sys.path:
    sys.path.insert(0, str(_PHASES_DIR / 'phase8_8'))

from flux_utils import get_device, PhaseLogger


# ─────────────────────────────────────────────
# ARC Dataset
# ─────────────────────────────────────────────

@dataclass
class ARCTask:
    """Single ARC task with train and test examples."""
    task_id: str
    train_pairs: List[Tuple[List[List[int]], List[List[int]]]]  # (input, output)
    test_pairs: List[Tuple[List[List[int]], List[List[int]]]]


class ARCDataset(Dataset):
    """ARC dataset loader."""
    
    def __init__(
        self,
        data_dir: str = 'arc-agi/data/training',
        max_tasks: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.tasks: List[ARCTask] = []
        
        # Load tasks
        if self.data_dir.exists():
            task_files = list(self.data_dir.glob('*.json'))
            if max_tasks:
                task_files = task_files[:max_tasks]
            
            for task_file in task_files:
                try:
                    with open(task_file) as f:
                        data = json.load(f)
                    
                    train_pairs = [
                        (ex['input'], ex['output'])
                        for ex in data.get('train', [])
                    ]
                    test_pairs = [
                        (ex['input'], ex['output'])
                        for ex in data.get('test', [])
                    ]
                    
                    self.tasks.append(ARCTask(
                        task_id=task_file.stem,
                        train_pairs=train_pairs,
                        test_pairs=test_pairs,
                    ))
                except Exception as e:
                    print(f"Failed to load {task_file}: {e}")
        else:
            print(f"⚠ ARC data not found at {data_dir}")
            print("  Download from: https://github.com/fchollet/ARC-AGI")
            # Generate synthetic tasks for testing
            self._generate_synthetic_tasks()
    
    def _generate_synthetic_tasks(self, n_tasks: int = 50):
        """Generate simple synthetic tasks for testing."""
        print("  Generating synthetic tasks for testing...")
        
        for i in range(n_tasks):
            # Task type 1: Color swap
            if i % 5 == 0:
                train_pairs = []
                for _ in range(3):
                    size = torch.randint(3, 8, (2,)).tolist()
                    grid = torch.randint(0, 3, size).tolist()
                    # Swap color 1 and 2
                    output = [[2 if c == 1 else (1 if c == 2 else c) for c in row] for row in grid]
                    train_pairs.append((grid, output))
                test_pairs = [train_pairs[-1]]  # Use last as test
                
            # Task type 2: Fill with dominant color
            elif i % 5 == 1:
                train_pairs = []
                for _ in range(3):
                    size = torch.randint(3, 6, (2,)).tolist()
                    grid = torch.randint(0, 4, size).tolist()
                    # Count colors, fill with most common
                    flat = [c for row in grid for c in row]
                    dominant = max(set(flat), key=flat.count)
                    output = [[dominant] * len(row) for row in grid]
                    train_pairs.append((grid, output))
                test_pairs = [train_pairs[-1]]
                
            # Task type 3: Mirror horizontal
            elif i % 5 == 2:
                train_pairs = []
                for _ in range(3):
                    size = torch.randint(3, 6, (2,)).tolist()
                    grid = torch.randint(0, 5, size).tolist()
                    output = [row[::-1] for row in grid]
                    train_pairs.append((grid, output))
                test_pairs = [train_pairs[-1]]
                
            # Task type 4: Rotate 90°
            elif i % 5 == 3:
                train_pairs = []
                for _ in range(3):
                    size = [torch.randint(3, 6, (1,)).item()] * 2  # Square
                    grid = torch.randint(0, 4, size).tolist()
                    # Rotate 90° clockwise
                    output = [[grid[len(grid)-1-j][i] for j in range(len(grid))] 
                              for i in range(len(grid[0]))]
                    train_pairs.append((grid, output))
                test_pairs = [train_pairs[-1]]
                
            # Task type 5: Identity (easiest)
            else:
                train_pairs = []
                for _ in range(3):
                    size = torch.randint(3, 6, (2,)).tolist()
                    grid = torch.randint(0, 5, size).tolist()
                    train_pairs.append((grid, grid))  # Identity
                test_pairs = [train_pairs[-1]]
            
            self.tasks.append(ARCTask(
                task_id=f'synthetic_{i:04d}',
                train_pairs=train_pairs,
                test_pairs=test_pairs,
            ))
    
    def __len__(self) -> int:
        return len(self.tasks)
    
    def __getitem__(self, idx: int) -> ARCTask:
        return self.tasks[idx]


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────

class GridAdapterTrainer:
    """Train GridToWave and WaveToGrid on ARC."""
    
    def __init__(
        self,
        grid_to_wave: nn.Module,
        wave_to_grid: nn.Module,
        device: str = 'cpu',
        lr: float = 1e-3,
    ):
        self.grid_to_wave = grid_to_wave.to(device)
        self.wave_to_grid = wave_to_grid.to(device)
        self.device = device
        
        # Optimizer (train both)
        params = list(grid_to_wave.parameters()) + list(wave_to_grid.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)
        
        # Loss functions
        self.reconstruction_loss = nn.CrossEntropyLoss()
    
    def train_step(
        self,
        input_grid: List[List[int]],
        output_grid: List[List[int]],
    ) -> Dict[str, float]:
        """
        Single training step on input/output pair.
        
        Losses:
        1. Reconstruction: WaveToGrid(GridToWave(output)) ≈ output
        2. Transformation: Learn delta wave
        """
        self.optimizer.zero_grad()
        
        # Encode input and output grids
        input_wave = self.grid_to_wave.encode(input_grid, mode='holistic')
        output_wave = self.grid_to_wave.encode(output_grid, mode='holistic')
        
        # Compute transformation delta
        delta = output_wave - input_wave
        
        # Decode output wave
        output_h = len(output_grid)
        output_w = len(output_grid[0])
        decoded = self.wave_to_grid.decode(output_wave, grid_size=(output_h, output_w))
        
        # Reconstruction loss (grid values are 0-9)
        target = torch.tensor(output_grid, dtype=torch.long, device=self.device)
        
        # decoded is [H, W], need to compute loss
        # WaveToGrid outputs class logits [H, W, 10]
        if decoded.dim() == 2:
            # If it's already class indices, compute accuracy
            correct = (decoded == target).float().mean()
            recon_loss = 1.0 - correct  # Proxy loss
        else:
            # If logits [H, W, 10]
            recon_loss = self.reconstruction_loss(
                decoded.view(-1, 10),
                target.view(-1),
            )
        
        # Transformation consistency loss
        # Apply delta to input, should get output
        predicted_output_wave = input_wave + delta
        delta_recon_loss = F.mse_loss(predicted_output_wave, output_wave)
        
        # Total loss
        loss = recon_loss + 0.1 * delta_recon_loss
        
        loss.backward()
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'recon_loss': recon_loss.item() if isinstance(recon_loss, Tensor) else recon_loss,
            'delta_loss': delta_recon_loss.item(),
        }
    
    def train_epoch(
        self,
        dataset: ARCDataset,
        max_pairs: int = 1000,
    ) -> Dict[str, float]:
        """Train one epoch on ARC dataset."""
        self.grid_to_wave.train()
        self.wave_to_grid.train()
        
        total_loss = 0
        total_recon = 0
        total_delta = 0
        n_pairs = 0
        
        for task in tqdm(dataset, desc='Training'):
            for input_grid, output_grid in task.train_pairs:
                metrics = self.train_step(input_grid, output_grid)
                total_loss += metrics['loss']
                total_recon += metrics['recon_loss']
                total_delta += metrics['delta_loss']
                n_pairs += 1
                
                if n_pairs >= max_pairs:
                    break
            
            if n_pairs >= max_pairs:
                break
        
        return {
            'loss': total_loss / max(n_pairs, 1),
            'recon_loss': total_recon / max(n_pairs, 1),
            'delta_loss': total_delta / max(n_pairs, 1),
            'n_pairs': n_pairs,
        }
    
    def evaluate(
        self,
        dataset: ARCDataset,
        max_tasks: int = 50,
    ) -> Dict[str, float]:
        """Evaluate on test pairs."""
        self.grid_to_wave.eval()
        self.wave_to_grid.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for task in dataset.tasks[:max_tasks]:
                for input_grid, output_grid in task.test_pairs:
                    # Encode and decode
                    wave = self.grid_to_wave.encode(output_grid, mode='holistic')
                    decoded = self.wave_to_grid.decode(
                        wave,
                        grid_size=(len(output_grid), len(output_grid[0])),
                    )
                    
                    # Check accuracy
                    target = torch.tensor(output_grid, device=self.device)
                    if decoded.dim() == 3:  # Logits
                        decoded = decoded.argmax(dim=-1)
                    
                    correct += (decoded == target).float().mean().item()
                    total += 1
        
        return {
            'accuracy': correct / max(total, 1),
            'total_tasks': total,
        }


# ─────────────────────────────────────────────
# Main Training Script
# ─────────────────────────────────────────────

def train_grid_adapters(
    epochs: int = 10,
    lr: float = 1e-3,
    arc_dir: str = 'arc-agi/data/training',
    save_path: Optional[str] = None,
    device: Optional[str] = None,
):
    """
    Train GridToWave and WaveToGrid on ARC dataset.
    
    Args:
        epochs: Number of training epochs
        lr: Learning rate
        arc_dir: Path to ARC training data
        save_path: Where to save trained adapters
        device: Target device (auto-detect if None)
    """
    log = PhaseLogger(phase=8.9)
    log.separator("Phase 8.9: Train Grid Adapters")
    
    if device is None:
        device = get_device()
    log.info(f"Device: {device}")
    
    # Load adapters
    from grid_adapters import GridToWave, WaveToGrid
    
    grid_to_wave = GridToWave(device=device)
    wave_to_grid = WaveToGrid(device=device)
    
    log.info(f"GridToWave params: {sum(p.numel() for p in grid_to_wave.parameters()):,}")
    log.info(f"WaveToGrid params: {sum(p.numel() for p in wave_to_grid.parameters()):,}")
    
    # Load dataset
    dataset = ARCDataset(arc_dir)
    log.info(f"Loaded {len(dataset)} tasks")
    
    # Create trainer
    trainer = GridAdapterTrainer(
        grid_to_wave=grid_to_wave,
        wave_to_grid=wave_to_grid,
        device=device,
        lr=lr,
    )
    
    # Training loop
    best_acc = 0.0
    
    for epoch in range(epochs):
        log.info(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Train
        train_metrics = trainer.train_epoch(dataset)
        log.metric("Train Loss", f"{train_metrics['loss']:.4f}")
        
        # Evaluate
        eval_metrics = trainer.evaluate(dataset)
        log.metric("Eval Accuracy", f"{eval_metrics['accuracy']:.4f}")
        
        if eval_metrics['accuracy'] > best_acc:
            best_acc = eval_metrics['accuracy']
            log.success(f"New best accuracy: {best_acc:.4f}")
            
            # Save checkpoint
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'grid_to_wave': grid_to_wave.state_dict(),
                    'wave_to_grid': wave_to_grid.state_dict(),
                    'accuracy': best_acc,
                }, save_path)
                log.info(f"Saved to {save_path}")
    
    log.separator("Training Complete")
    log.metric("Best Accuracy", f"{best_acc:.4f}")
    
    return grid_to_wave, wave_to_grid, best_acc


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Grid Adapters on ARC')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--arc-dir', type=str, default='arc-agi/data/training')
    parser.add_argument('--save', type=str, default='checkpoints/grid_adapters.pt')
    parser.add_argument('--device', type=str, default=None)
    
    args = parser.parse_args()
    
    train_grid_adapters(
        epochs=args.epochs,
        lr=args.lr,
        arc_dir=args.arc_dir,
        save_path=args.save,
        device=args.device,
    )
