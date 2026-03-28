"""
Phase 9 ARC: Dataset Loader

Load ARC-AGI-1, ARC-AGI-2, and ARC-AGI-3 datasets from various sources:
- Local files (JSON format)
- HuggingFace Hub (arcprize/arc-agi)
- Kaggle competition data

ARC task format:
    {
        "train": [{"input": [[...]], "output": [[...]]}],
        "test": [{"input": [[...]], "output": [[...]]}]
    }
"""

import json
import torch
from torch import Tensor
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Iterator
from dataclasses import dataclass, field
import sys

# Add project paths
_PHASES_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PHASES_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class ARCExample:
    """Single input/output pair."""
    input: List[List[int]]
    output: List[List[int]]
    
    @property
    def input_tensor(self) -> Tensor:
        """Convert input to tensor [H, W]."""
        return torch.tensor(self.input, dtype=torch.long)
    
    @property
    def output_tensor(self) -> Tensor:
        """Convert output to tensor [H, W]."""
        return torch.tensor(self.output, dtype=torch.long)
    
    @property
    def input_shape(self) -> Tuple[int, int]:
        return len(self.input), len(self.input[0]) if self.input else 0
    
    @property
    def output_shape(self) -> Tuple[int, int]:
        return len(self.output), len(self.output[0]) if self.output else 0


@dataclass
class ARCTask:
    """Complete ARC task with train and test examples."""
    task_id: str
    train: List[ARCExample]
    test: List[ARCExample]
    source: str = "unknown"  # 'arc-agi-1', 'arc-agi-2', 'arc-agi-3', 'synthetic'
    
    @property
    def num_train(self) -> int:
        return len(self.train)
    
    @property
    def num_test(self) -> int:
        return len(self.test)
    
    def get_delta_waves(self, encoder) -> List[Tensor]:
        """Compute delta waves for all training pairs."""
        deltas = []
        for ex in self.train:
            _, _, delta = encoder.encode_pair(ex.input, ex.output)
            deltas.append(delta)
        return deltas


@dataclass
class ARCDataset:
    """Collection of ARC tasks."""
    name: str
    tasks: List[ARCTask] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.tasks)
    
    def __iter__(self) -> Iterator[ARCTask]:
        return iter(self.tasks)
    
    def __getitem__(self, idx: int) -> ARCTask:
        return self.tasks[idx]
    
    def filter_by_size(self, max_h: int = 30, max_w: int = 30) -> 'ARCDataset':
        """Filter tasks to only those with grids <= max size."""
        filtered = []
        for task in self.tasks:
            valid = True
            for ex in task.train + task.test:
                h, w = ex.input_shape
                oh, ow = ex.output_shape
                if h > max_h or w > max_w or oh > max_h or ow > max_w:
                    valid = False
                    break
            if valid:
                filtered.append(task)
        return ARCDataset(name=f"{self.name}_filtered", tasks=filtered)


# ─────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────

def load_arc_task_from_json(path: Path) -> ARCTask:
    """Load single ARC task from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    train_examples = [
        ARCExample(input=ex['input'], output=ex['output'])
        for ex in data.get('train', [])
    ]
    test_examples = [
        ARCExample(input=ex['input'], output=ex.get('output', []))
        for ex in data.get('test', [])
    ]
    
    return ARCTask(
        task_id=path.stem,
        train=train_examples,
        test=test_examples,
        source='local',
    )


def load_arc_from_directory(data_dir: Path, max_tasks: Optional[int] = None) -> ARCDataset:
    """Load ARC tasks from directory of JSON files."""
    tasks = []
    
    if not data_dir.exists():
        print(f"⚠ Directory not found: {data_dir}")
        return ARCDataset(name=str(data_dir), tasks=[])
    
    json_files = sorted(data_dir.glob('*.json'))
    if max_tasks:
        json_files = json_files[:max_tasks]
    
    for path in json_files:
        try:
            task = load_arc_task_from_json(path)
            tasks.append(task)
        except Exception as e:
            print(f"  ⚠ Failed to load {path.name}: {e}")
    
    print(f"  ✓ Loaded {len(tasks)} tasks from {data_dir}")
    return ARCDataset(name=data_dir.name, tasks=tasks)


def load_arc_from_huggingface(
    repo_id: str = "arcprize/arc-agi",
    split: str = "training",
    max_tasks: Optional[int] = None,
) -> ARCDataset:
    """Load ARC dataset from HuggingFace Hub."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("⚠ datasets library not installed. Run: pip install datasets")
        return ARCDataset(name=f"hf/{repo_id}/{split}", tasks=[])
    
    try:
        dataset = load_dataset(repo_id, split=split)
        tasks = []
        
        for i, item in enumerate(dataset):
            if max_tasks and i >= max_tasks:
                break
            
            train_examples = [
                ARCExample(input=ex['input'], output=ex['output'])
                for ex in item.get('train', [])
            ]
            test_examples = [
                ARCExample(input=ex['input'], output=ex.get('output', []))
                for ex in item.get('test', [])
            ]
            
            tasks.append(ARCTask(
                task_id=item.get('id', f'task_{i:04d}'),
                train=train_examples,
                test=test_examples,
                source='huggingface',
            ))
        
        print(f"  ✓ Loaded {len(tasks)} tasks from HuggingFace")
        return ARCDataset(name=f"hf/{split}", tasks=tasks)
        
    except Exception as e:
        print(f"⚠ Failed to load from HuggingFace: {e}")
        return ARCDataset(name=f"hf/{repo_id}/{split}", tasks=[])


def download_arc_agi_dataset(
    version: int = 2,
    data_dir: Optional[Path] = None,
) -> Path:
    """
    Download ARC-AGI dataset from GitHub.
    
    Args:
        version: 1 for ARC-AGI-1, 2 for ARC-AGI-2
        data_dir: Where to save (default: project root / arc-agi-{version})
    
    Returns:
        Path to downloaded dataset directory
    """
    if data_dir is None:
        data_dir = _PROJECT_ROOT / f'arc-agi-{version}'
    
    if data_dir.exists() and (data_dir / 'data').exists():
        print(f"  ✓ ARC-AGI-{version} already downloaded at {data_dir}")
        return data_dir
    
    import subprocess
    
    if version == 1:
        repo_url = "https://github.com/fchollet/ARC-AGI.git"
    else:  # version == 2
        repo_url = "https://github.com/arcprize/arc-agi.git"
    
    print(f"  Downloading ARC-AGI-{version}...")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        subprocess.run(
            ['git', 'clone', '--depth', '1', repo_url, str(data_dir)],
            check=True,
            capture_output=True,
        )
        print(f"  ✓ Downloaded to {data_dir}")
    except subprocess.CalledProcessError as e:
        print(f"  ⚠ Git clone failed: {e}")
    
    return data_dir


def load_arc_agi(
    version: int = 2,
    split: str = 'training',
    max_tasks: Optional[int] = None,
    auto_download: bool = True,
) -> ARCDataset:
    """
    Load ARC-AGI dataset (main entry point).
    
    Args:
        version: 1 or 2
        split: 'training', 'evaluation', or 'test'
        max_tasks: Limit number of tasks (for testing)
        auto_download: Download if not found
    
    Returns:
        ARCDataset with loaded tasks
    """
    # Try local first
    data_dir = _PROJECT_ROOT / f'arc-agi-{version}' / 'data' / split
    
    if not data_dir.exists() and auto_download:
        base_dir = download_arc_agi_dataset(version)
        data_dir = base_dir / 'data' / split
    
    if data_dir.exists():
        dataset = load_arc_from_directory(data_dir, max_tasks)
        for task in dataset.tasks:
            task.source = f'arc-agi-{version}'
        return dataset
    
    # Fall back to HuggingFace
    print(f"  ⚠ Local not found, trying HuggingFace...")
    return load_arc_from_huggingface(
        repo_id="arcprize/arc-agi" if version == 2 else "fchollet/arc-agi",
        split=split,
        max_tasks=max_tasks,
    )


# ─────────────────────────────────────────────
# Synthetic Tasks for Testing
# ─────────────────────────────────────────────

def generate_synthetic_tasks(n_tasks: int = 50) -> ARCDataset:
    """Generate simple synthetic ARC tasks for testing."""
    tasks = []
    
    for i in range(n_tasks):
        task_type = i % 10
        train_pairs = []
        
        # Task type 0: Identity
        if task_type == 0:
            for _ in range(3):
                h, w = torch.randint(3, 8, (2,)).tolist()
                grid = torch.randint(0, 5, (h, w)).tolist()
                train_pairs.append(ARCExample(input=grid, output=grid))
        
        # Task type 1: Color swap (1 ↔ 2)
        elif task_type == 1:
            for _ in range(3):
                h, w = torch.randint(3, 8, (2,)).tolist()
                grid = torch.randint(0, 4, (h, w)).tolist()
                output = [[2 if c == 1 else (1 if c == 2 else c) for c in row] for row in grid]
                train_pairs.append(ARCExample(input=grid, output=output))
        
        # Task type 2: Mirror horizontal
        elif task_type == 2:
            for _ in range(3):
                h, w = torch.randint(3, 8, (2,)).tolist()
                grid = torch.randint(0, 5, (h, w)).tolist()
                output = [row[::-1] for row in grid]
                train_pairs.append(ARCExample(input=grid, output=output))
        
        # Task type 3: Mirror vertical
        elif task_type == 3:
            for _ in range(3):
                h, w = torch.randint(3, 8, (2,)).tolist()
                grid = torch.randint(0, 5, (h, w)).tolist()
                output = grid[::-1]
                train_pairs.append(ARCExample(input=grid, output=output))
        
        # Task type 4: Rotate 90° clockwise
        elif task_type == 4:
            for _ in range(3):
                size = torch.randint(3, 7, (1,)).item()
                grid = torch.randint(0, 5, (size, size)).tolist()
                h, w = len(grid), len(grid[0])
                output = [[grid[h-1-j][i] for j in range(h)] for i in range(w)]
                train_pairs.append(ARCExample(input=grid, output=output))
        
        # Task type 5: Fill with dominant color
        elif task_type == 5:
            for _ in range(3):
                h, w = torch.randint(3, 7, (2,)).tolist()
                grid = torch.randint(0, 4, (h, w)).tolist()
                flat = [c for row in grid for c in row]
                dominant = max(set(flat), key=flat.count)
                output = [[dominant] * w for _ in range(h)]
                train_pairs.append(ARCExample(input=grid, output=output))
        
        # Task type 6: Replace 0 with 5
        elif task_type == 6:
            for _ in range(3):
                h, w = torch.randint(3, 8, (2,)).tolist()
                grid = torch.randint(0, 4, (h, w)).tolist()
                output = [[5 if c == 0 else c for c in row] for row in grid]
                train_pairs.append(ARCExample(input=grid, output=output))
        
        # Task type 7: Transpose
        elif task_type == 7:
            for _ in range(3):
                h, w = torch.randint(3, 8, (2,)).tolist()
                grid = torch.randint(0, 5, (h, w)).tolist()
                output = [[grid[j][i] for j in range(h)] for i in range(w)]
                train_pairs.append(ARCExample(input=grid, output=output))
        
        # Task type 8: Invert colors (9-x)
        elif task_type == 8:
            for _ in range(3):
                h, w = torch.randint(3, 7, (2,)).tolist()
                grid = torch.randint(0, 10, (h, w)).tolist()
                output = [[9 - c for c in row] for row in grid]
                train_pairs.append(ARCExample(input=grid, output=output))
        
        # Task type 9: Clear to 0
        else:
            for _ in range(3):
                h, w = torch.randint(3, 8, (2,)).tolist()
                grid = torch.randint(0, 5, (h, w)).tolist()
                output = [[0] * w for _ in range(h)]
                train_pairs.append(ARCExample(input=grid, output=output))
        
        # Use last training pair as test
        test_pairs = [train_pairs[-1]]
        
        tasks.append(ARCTask(
            task_id=f'synthetic_{i:04d}',
            train=train_pairs[:-1],  # First 2 for training
            test=test_pairs,
            source='synthetic',
        ))
    
    print(f"  ✓ Generated {len(tasks)} synthetic tasks")
    return ARCDataset(name='synthetic', tasks=tasks)


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def visualize_task(task: ARCTask, max_examples: int = 3):
    """Print ASCII visualization of task."""
    COLORS = ['⬛', '🟦', '🟥', '🟩', '🟨', '⬜', '🟪', '🟧', '🩵', '🟫']
    
    print(f"\n{'='*50}")
    print(f"Task: {task.task_id}")
    print(f"Train: {task.num_train} | Test: {task.num_test}")
    print(f"{'='*50}")
    
    for i, ex in enumerate(task.train[:max_examples]):
        print(f"\nExample {i+1}:")
        print("Input:")
        for row in ex.input:
            print('  ' + ''.join(COLORS[min(c, 9)] for c in row))
        print("Output:")
        for row in ex.output:
            print('  ' + ''.join(COLORS[min(c, 9)] for c in row))
    
    print("\nTest:")
    for i, ex in enumerate(task.test):
        print(f"Test {i+1} Input:")
        for row in ex.input:
            print('  ' + ''.join(COLORS[min(c, 9)] for c in row))
        if ex.output:
            print("Expected Output:")
            for row in ex.output:
                print('  ' + ''.join(COLORS[min(c, 9)] for c in row))


def format_submission(
    task_id: str,
    attempt_1: List[List[int]],
    attempt_2: Optional[List[List[int]]] = None,
) -> Dict[str, Any]:
    """Format prediction for Kaggle submission."""
    result = {
        "task_id": task_id,
        "attempt_1": attempt_1,
    }
    if attempt_2:
        result["attempt_2"] = attempt_2
    return result


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("ARC Loader Test")
    print("=" * 50)
    
    # Test synthetic generation
    synthetic = generate_synthetic_tasks(10)
    print(f"\nSynthetic dataset: {len(synthetic)} tasks")
    
    if len(synthetic) > 0:
        visualize_task(synthetic[0])
        visualize_task(synthetic[2])  # Mirror task
    
    # Try loading real ARC
    print("\n" + "=" * 50)
    print("Attempting to load ARC-AGI-2...")
    arc2 = load_arc_agi(version=2, split='training', max_tasks=5, auto_download=False)
    print(f"Loaded: {len(arc2)} tasks")
    
    if len(arc2) > 0:
        visualize_task(arc2[0])
