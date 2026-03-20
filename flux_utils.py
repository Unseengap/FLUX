"""
FLUX Shared Utilities
Used by all phases. Import from here to avoid duplication.
"""

import torch
import json
import platform
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional


# ─────────────────────────────────────────────
# Checkpoint Management
# ─────────────────────────────────────────────

CHECKPOINT_DIR = Path(__file__).parent / 'checkpoints'
RESULTS_DIR = Path(__file__).parent / 'results'

def save_checkpoint(phase: int, state: Dict[str, Any]) -> Path:
    """
    Save a phase checkpoint with metadata.
    Always includes phase number, timestamp, and component list.
    """
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    state['phase'] = phase
    state['timestamp'] = datetime.now().isoformat()
    path = CHECKPOINT_DIR / f'phase{phase}.phase.pt'
    torch.save(state, path)
    size_mb = path.stat().st_size / 1e6
    print(f"✓ Phase {phase} checkpoint saved: {path} ({size_mb:.1f} MB)")
    return path

def load_checkpoint(phase: int) -> Dict[str, Any]:
    """
    Load a phase checkpoint, with verification.
    Raises clear error if checkpoint missing.
    """
    path = CHECKPOINT_DIR / f'phase{phase}.phase.pt'
    if not path.exists():
        raise FileNotFoundError(
            f"Phase {phase} checkpoint not found at {path}\n"
            f"Run Phase {phase} training first."
        )
    state = torch.load(path, map_location='cpu')
    assert state.get('phase') == phase, (
        f"Checkpoint phase mismatch: expected {phase}, got {state.get('phase')}"
    )
    print(f"✓ Phase {phase} checkpoint loaded ({path.stat().st_size/1e6:.1f} MB)")
    return state

def checkpoint_exists(phase: int) -> bool:
    """Check if a phase checkpoint exists."""
    return (CHECKPOINT_DIR / f'phase{phase}.phase.pt').exists()

def verify_checkpoint_chain(up_to_phase: int) -> bool:
    """Verify all checkpoints up to the given phase exist."""
    for p in range(1, up_to_phase):
        if not checkpoint_exists(p):
            print(f"✗ Missing checkpoint for Phase {p}")
            return False
        print(f"✓ Phase {p} checkpoint present")
    return True


# ─────────────────────────────────────────────
# Hardware Detection
# ─────────────────────────────────────────────

def get_device() -> str:
    """Get best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def get_hardware_info() -> Dict[str, str]:
    """Get hardware info for RESULTS files."""
    info = {
        'device': get_device(),
        'platform': platform.platform(),
        'python': platform.python_version(),
        'torch': torch.__version__,
    }
    if torch.cuda.is_available():
        info['gpu'] = torch.cuda.get_device_name(0)
        info['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        info['cuda'] = torch.version.cuda
    return info


# ─────────────────────────────────────────────
# Results Tracking
# ─────────────────────────────────────────────

class PhaseResults:
    """
    Tracks test results, metrics, and demo outputs for a phase.
    Generates RESULTS_PHASE_N.md automatically.
    """
    
    def __init__(self, phase: int, component_name: str):
        self.phase = phase
        self.component_name = component_name
        self.start_time = datetime.now()
        self.hardware = get_hardware_info()
        self.tests = []
        self.demos = []
        self.metrics = {}
        self.issues = []
        self.notes = []
    
    def add_test(
        self, 
        name: str, 
        passed: bool, 
        score: Any = None,
        threshold: Any = None,
        notes: str = ''
    ):
        self.tests.append({
            'name': name,
            'status': 'PASS' if passed else 'FAIL',
            'score': score,
            'threshold': threshold,
            'notes': notes
        })
        status = '✓' if passed else '✗'
        print(f"  {status} {name}: {score} (threshold: {threshold})")
    
    def add_demo(self, name: str, ran: bool, quality: str = ''):
        self.demos.append({
            'name': name,
            'ran': ran,
            'quality': quality
        })
    
    def add_metric(self, key: str, value: Any):
        self.metrics[key] = value
    
    def add_issue(self, issue: str):
        self.issues.append(issue)
        print(f"  ⚠ Issue: {issue}")
    
    def all_tests_passed(self) -> bool:
        return all(t['status'] == 'PASS' for t in self.tests)
    
    def save(self, path: Optional[str] = None):
        """Generate and save RESULTS_PHASE_N.md"""
        if path is None:
            phase_dir = Path(__file__).parent / 'phases' / f'phase{self.phase}'
            path = phase_dir / f'RESULTS_PHASE_{self.phase}.md'
        
        duration = datetime.now() - self.start_time
        
        lines = [
            f"# Results: Phase {self.phase} — {self.component_name}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Hardware: {self.hardware.get('gpu', self.hardware['device'])}",
            f"Duration: {duration}",
            "",
            "## Component Status",
            f"{self.component_name}: ✓ BUILT" if self.all_tests_passed() else f"{self.component_name}: ✗ INCOMPLETE",
            f"Checkpoint: checkpoints/phase{self.phase}.phase.pt {'✓' if checkpoint_exists(self.phase) else '✗ NOT SAVED'}",
            "",
            "## Test Results",
            "| Test | Status | Score | Threshold | Pass? |",
            "|------|--------|-------|-----------|-------|",
        ]
        
        for t in self.tests:
            lines.append(
                f"| {t['name']} | {t['status']} | {t['score']} | {t['threshold']} | {'✓' if t['status']=='PASS' else '✗'} |"
            )
        
        lines += ["", "## Demo Status", "| Demo | Ran? | Quality |", "|------|------|---------|"]
        for d in self.demos:
            lines.append(f"| {d['name']} | {'✓' if d['ran'] else '✗'} | {d['quality']} |")
        
        lines += ["", "## Key Metrics"]
        for k, v in self.metrics.items():
            lines.append(f"- **{k}:** {v}")
        
        if self.issues:
            lines += ["", "## Issues Encountered"]
            for issue in self.issues:
                lines.append(f"- {issue}")
        
        all_pass = self.all_tests_passed()
        ckpt = checkpoint_exists(self.phase)
        lines += [
            "",
            f"## Phase {self.phase} → Phase {self.phase+1} Readiness",
            f"All tests passing: {'YES ✓' if all_pass else 'NO ✗'}",
            f"Checkpoint saved: {'YES ✓' if ckpt else 'NO ✗'}",
            f"Ready for Phase {self.phase+1}: {'YES ✓' if (all_pass and ckpt) else 'NO ✗'}",
        ]
        
        content = '\n'.join(lines)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content)
        
        # Also copy to results/ dir
        RESULTS_DIR.mkdir(exist_ok=True)
        results_copy = RESULTS_DIR / f'RESULTS_PHASE_{self.phase}.md'
        results_copy.write_text(content)
        
        print(f"\n{'='*50}")
        print(f"Phase {self.phase} Results saved to: {path}")
        print(f"All tests passed: {all_pass}")
        print(f"Ready for Phase {self.phase+1}: {all_pass and ckpt}")
        print('='*50)
        return content


# ─────────────────────────────────────────────
# Smoke Tests (run at start of each phase)
# ─────────────────────────────────────────────

def smoke_test_phase1(checkpoint: Dict) -> bool:
    """Quick verification that Phase 1 CSE works."""
    try:
        import sys
        sys.path.append(str(Path(__file__).parent / 'phases' / 'phase1'))
        from cse import ContinuousSemanticEncoder
        cse = ContinuousSemanticEncoder(**checkpoint.get('config', {}))
        cse.load_state_dict(checkpoint['state_dict'])
        wave = cse.encode("hello world")
        assert wave.full.shape[-1] == 432
        return True
    except Exception as e:
        print(f"Phase 1 smoke test failed: {e}")
        return False


# ─────────────────────────────────────────────
# Forgetting Score Utility
# ─────────────────────────────────────────────

def compute_forgetting_score(
    model,
    task_A_eval_fn,
    task_B_train_fn,
    task_A_eval_after_fn = None
) -> float:
    """
    Compute catastrophic forgetting score.
    
    Score = 0.0 → perfect retention (FLUX target)
    Score = 1.0 → complete forgetting
    
    Transformer baseline typically: 0.3 – 0.8
    """
    acc_before = task_A_eval_fn(model)
    task_B_train_fn(model)
    eval_after = task_A_eval_after_fn or task_A_eval_fn
    acc_after = eval_after(model)
    
    if acc_before == 0:
        return 0.0
    
    forgetting = (acc_before - acc_after) / acc_before
    return max(0.0, forgetting)
