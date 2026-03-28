"""
Phase 9 ARC: Main Solver

The complete ARC task solver that brings together:
- Object detection
- Pattern library
- Rule induction
- Verification
- Multi-attempt submission

This is what gets submitted to Kaggle.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import sys
from pathlib import Path
import json
import time

# Add project paths
_PHASES_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PHASES_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from arc_loader import ARCTask, ARCExample, ARCDataset, load_arc_agi, generate_synthetic_tasks
from object_detector import ObjectDetector, ObjectGraph
from pattern_library import PATTERNS, apply_sequence
from rule_inducer import RuleInducer, RuleHypothesis


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class Prediction:
    """Single prediction for a test case."""
    grid: List[List[int]]
    hypothesis: RuleHypothesis
    confidence: float


@dataclass
class TaskSolution:
    """Complete solution for a task."""
    task_id: str
    attempt_1: List[List[int]]
    attempt_2: Optional[List[List[int]]] = None
    hypotheses: List[RuleHypothesis] = None
    solve_time_ms: float = 0.0
    
    def to_submission_dict(self) -> Dict[str, Any]:
        """Convert to Kaggle submission format."""
        result = {
            "task_id": self.task_id,
            "attempt_1": self.attempt_1,
        }
        if self.attempt_2:
            result["attempt_2"] = self.attempt_2
        return result


# ─────────────────────────────────────────────
# ARC Solver
# ─────────────────────────────────────────────

class ARCSolver:
    """
    Main ARC task solver.
    
    Strategy:
    1. Extract features from training examples
    2. Induce transformation rule(s)
    3. Apply best rule to test input
    4. Generate 2 attempts (top 2 hypotheses)
    """
    
    def __init__(
        self,
        use_waves: bool = True,
        device: str = 'cpu',
        verbose: bool = False,
    ):
        self.rule_inducer = RuleInducer(use_waves=use_waves, device=device)
        self.detector = ObjectDetector()
        self.device = device
        self.verbose = verbose
    
    def solve(self, task: ARCTask) -> TaskSolution:
        """
        Solve a single ARC task.
        
        Returns:
            TaskSolution with up to 2 attempts
        """
        start_time = time.time()
        
        if self.verbose:
            print(f"  Solving {task.task_id}...")
        
        # Step 1: Induce rules from training examples
        hypotheses = self.rule_inducer.induce(task, max_hypotheses=5)
        
        if self.verbose:
            print(f"    Found {len(hypotheses)} hypotheses")
        
        # Step 2: Apply to test input
        if not task.test:
            return TaskSolution(
                task_id=task.task_id,
                attempt_1=[[0]],
                solve_time_ms=(time.time() - start_time) * 1000,
            )
        
        test_input = torch.tensor(task.test[0].input, dtype=torch.long)
        
        attempts = []
        for h in hypotheses[:2]:  # Top 2 hypotheses
            try:
                prediction = h.apply(test_input)
                attempts.append(prediction.tolist())
            except Exception as e:
                if self.verbose:
                    print(f"    Hypothesis {h.rule_id} failed: {e}")
                # Fallback: identity
                attempts.append(test_input.tolist())
        
        # Ensure we have at least 1 attempt
        if not attempts:
            attempts = [test_input.tolist()]
        
        solve_time = (time.time() - start_time) * 1000
        
        return TaskSolution(
            task_id=task.task_id,
            attempt_1=attempts[0],
            attempt_2=attempts[1] if len(attempts) > 1 else None,
            hypotheses=hypotheses,
            solve_time_ms=solve_time,
        )
    
    def solve_all(
        self,
        dataset: ARCDataset,
        max_tasks: Optional[int] = None,
    ) -> List[TaskSolution]:
        """
        Solve all tasks in dataset.
        """
        solutions = []
        tasks = dataset.tasks[:max_tasks] if max_tasks else dataset.tasks
        
        for i, task in enumerate(tasks):
            if self.verbose:
                print(f"[{i+1}/{len(tasks)}] ", end="")
            
            solution = self.solve(task)
            solutions.append(solution)
            
            if self.verbose:
                print(f"  Done in {solution.solve_time_ms:.1f}ms")
        
        return solutions
    
    def evaluate(
        self,
        solutions: List[TaskSolution],
        dataset: ARCDataset,
    ) -> Dict[str, Any]:
        """
        Evaluate solutions against ground truth.
        
        Returns accuracy metrics.
        """
        correct_1 = 0
        correct_2 = 0
        correct_either = 0
        total = 0
        
        task_by_id = {t.task_id: t for t in dataset.tasks}
        
        for solution in solutions:
            task = task_by_id.get(solution.task_id)
            if not task or not task.test or not task.test[0].output:
                continue
            
            expected = task.test[0].output
            
            # Check attempt 1
            match_1 = solution.attempt_1 == expected
            
            # Check attempt 2
            match_2 = (solution.attempt_2 is not None and 
                       solution.attempt_2 == expected)
            
            if match_1:
                correct_1 += 1
            if match_2:
                correct_2 += 1
            if match_1 or match_2:
                correct_either += 1
            
            total += 1
        
        return {
            "total_tasks": total,
            "correct_attempt_1": correct_1,
            "correct_attempt_2": correct_2,
            "correct_either": correct_either,
            "accuracy_1": correct_1 / max(total, 1),
            "accuracy_2": correct_2 / max(total, 1),
            "accuracy_either": correct_either / max(total, 1),
        }


# ─────────────────────────────────────────────
# Submission Generation
# ─────────────────────────────────────────────

def generate_submission(
    solutions: List[TaskSolution],
    output_path: str,
) -> Path:
    """
    Generate Kaggle submission file.
    
    Format: JSON with task_id, attempt_1, attempt_2
    """
    submission = [s.to_submission_dict() for s in solutions]
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(submission, f, indent=2)
    
    print(f"  ✓ Submission saved to {output_path}")
    return output_path


# ─────────────────────────────────────────────
# Quick Evaluation
# ─────────────────────────────────────────────

def quick_eval(
    solver: ARCSolver,
    n_tasks: int = 20,
    use_synthetic: bool = True,
) -> Dict[str, Any]:
    """
    Quick evaluation on synthetic or real tasks.
    """
    print("Quick Evaluation")
    print("=" * 50)
    
    if use_synthetic:
        print("Using synthetic tasks...")
        dataset = generate_synthetic_tasks(n_tasks)
    else:
        print("Loading ARC-AGI-2 training set...")
        dataset = load_arc_agi(version=2, split='training', max_tasks=n_tasks)
    
    print(f"Solving {len(dataset)} tasks...")
    solutions = solver.solve_all(dataset, max_tasks=n_tasks)
    
    # Evaluate
    results = solver.evaluate(solutions, dataset)
    
    print("\n" + "=" * 50)
    print("Results:")
    print(f"  Total tasks: {results['total_tasks']}")
    print(f"  Correct (attempt 1): {results['correct_attempt_1']} ({results['accuracy_1']*100:.1f}%)")
    print(f"  Correct (attempt 2): {results['correct_attempt_2']} ({results['accuracy_2']*100:.1f}%)")
    print(f"  Correct (either): {results['correct_either']} ({results['accuracy_either']*100:.1f}%)")
    
    # Timing
    total_time = sum(s.solve_time_ms for s in solutions)
    avg_time = total_time / max(len(solutions), 1)
    print(f"  Avg time/task: {avg_time:.1f}ms")
    
    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("ARC Solver Test")
    print("=" * 50)
    
    # Create solver
    solver = ARCSolver(use_waves=False, verbose=True)
    
    # Test on synthetic tasks
    print("\n--- Synthetic Tasks ---")
    synthetic = generate_synthetic_tasks(10)
    
    for task in synthetic[:5]:
        print(f"\nTask: {task.task_id} ({task.num_train} train, {task.num_test} test)")
        
        solution = solver.solve(task)
        
        print(f"  Attempt 1 shape: {len(solution.attempt_1)}x{len(solution.attempt_1[0]) if solution.attempt_1 else 0}")
        if solution.hypotheses:
            print(f"  Best hypothesis: {solution.hypotheses[0].explanation}")
        
        # Check if correct
        if task.test and task.test[0].output:
            expected = task.test[0].output
            correct = solution.attempt_1 == expected or solution.attempt_2 == expected
            print(f"  Correct: {'✓' if correct else '✗'}")
    
    # Evaluate
    print("\n" + "=" * 50)
    results = quick_eval(solver, n_tasks=20, use_synthetic=True)
