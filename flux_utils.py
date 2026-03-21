"""
FLUX Shared Utilities
Used by all phases. Import from here to avoid duplication.

Includes:
- Checkpoint management (local + HuggingFace Hub)
- Phase logging system (phase.log per phase)
- Hardware detection
- Results tracking (RESULTS_PHASE_N.md)
- Forgetting score utility
"""

import os
import torch
import json
import platform
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional


# ─────────────────────────────────────────────
# Project Paths
# ─────────────────────────────────────────────

CHECKPOINT_DIR = Path(__file__).parent / 'checkpoints'
RESULTS_DIR = Path(__file__).parent / 'results'
LOGS_DIR = Path(__file__).parent / 'logs'

# HuggingFace Hub settings
HF_REPO_ID = "UnseenGAP/FLUX"
GITHUB_REPO_URL = "https://github.com/Unseengap/FLUX.git"


# ─────────────────────────────────────────────
# Phase Logger
# ─────────────────────────────────────────────

class PhaseLogger:
    """
    Persistent logger for a phase. Writes to logs/phaseN.log and prints.
    Every cell/step appends to the log file so progress is never lost.

    Usage:
        log = PhaseLogger(phase=1)
        log.info("Starting training")
        log.success("Checkpoint saved")
        log.error("Test failed")
        log.metric("accuracy", 0.95)
        log.separator("Training Complete")
    """

    def __init__(self, phase: int, verbose: bool = True):
        self.phase = phase
        self.verbose = verbose
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = LOGS_DIR / f'phase{phase}.log'

        # Write header if new log
        if not self.log_path.exists():
            self._write(f"{'='*60}")
            self._write(f"FLUX Phase {phase} Log")
            self._write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self._write(f"{'='*60}\n")

    def _write(self, msg: str):
        """Append a line to the log file and optionally print."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        line = f"[{timestamp}] {msg}"
        with open(self.log_path, 'a') as f:
            f.write(line + '\n')
        if self.verbose:
            print(line)

    def info(self, msg: str):
        """Log an informational message."""
        self._write(f"  ℹ {msg}")

    def success(self, msg: str):
        """Log a success message."""
        self._write(f"  ✓ {msg}")

    def warning(self, msg: str):
        """Log a warning message."""
        self._write(f"  ⚠ {msg}")

    def error(self, msg: str):
        """Log an error message."""
        self._write(f"  ✗ {msg}")

    def metric(self, key: str, value: Any):
        """Log a metric key-value pair."""
        self._write(f"  📊 {key}: {value}")

    def separator(self, title: str = ""):
        """Log a section divider."""
        if title:
            self._write(f"\n{'─'*20} {title} {'─'*20}")
        else:
            self._write(f"{'─'*60}")

    def cell_start(self, cell_name: str):
        """Mark the start of a notebook cell execution."""
        self._write(f"\n▶ CELL: {cell_name}")
        self._write(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def cell_end(self, cell_name: str, status: str = "OK"):
        """Mark the end of a notebook cell execution."""
        self._write(f"  ◼ CELL {cell_name} — {status}")

    def get_log_contents(self) -> str:
        """Read the full log file contents."""
        if self.log_path.exists():
            return self.log_path.read_text()
        return ""


# ─────────────────────────────────────────────
# Checkpoint Management (Local + HuggingFace)
# ─────────────────────────────────────────────

def save_checkpoint(phase: int, state: Dict[str, Any]) -> Path:
    """
    Save a phase checkpoint locally with metadata.
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
    Load a phase checkpoint with verification.
    Tries local first, then downloads from HuggingFace Hub if missing.
    """
    path = CHECKPOINT_DIR / f'phase{phase}.phase.pt'

    # Try local first
    if path.exists():
        state = torch.load(path, map_location='cpu', weights_only=False)
        assert state.get('phase') == phase, (
            f"Checkpoint phase mismatch: expected {phase}, got {state.get('phase')}"
        )
        print(f"✓ Phase {phase} checkpoint loaded (local, {path.stat().st_size/1e6:.1f} MB)")
        return state

    # Try HuggingFace Hub download
    print(f"  ℹ Local checkpoint not found, trying HuggingFace Hub...")
    downloaded = download_checkpoint_from_hf(phase)
    if downloaded and path.exists():
        state = torch.load(path, map_location='cpu', weights_only=False)
        assert state.get('phase') == phase
        print(f"✓ Phase {phase} checkpoint loaded (from HuggingFace Hub)")
        return state

    raise FileNotFoundError(
        f"Phase {phase} checkpoint not found at {path}\n"
        f"Not found on HuggingFace Hub ({HF_REPO_ID}) either.\n"
        f"Run Phase {phase} training first."
    )


def checkpoint_exists(phase: int) -> bool:
    """Check if a phase checkpoint exists locally."""
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
# HuggingFace Hub Integration
# ─────────────────────────────────────────────

def upload_checkpoint_to_hf(
    phase: int,
    hf_token: Optional[str] = None,
    repo_id: str = HF_REPO_ID,
) -> bool:
    """
    Upload a phase checkpoint to HuggingFace Hub.
    Token sourced from: arg > env var HF_TOKEN > Kaggle secrets.

    Args:
        phase: phase number to upload
        hf_token: HuggingFace API token (optional, falls back to env/secrets)
        repo_id: HuggingFace repo ID (default: UnseenGAP/FLUX)
    Returns:
        True if upload succeeded
    """
    path = CHECKPOINT_DIR / f'phase{phase}.phase.pt'
    if not path.exists():
        print(f"  ✗ Cannot upload — Phase {phase} checkpoint not found locally")
        return False

    token = _resolve_hf_token(hf_token)
    if not token:
        print("  ⚠ No HuggingFace token found — skipping upload")
        print("    Set HF_TOKEN env var, or add to Kaggle secrets")
        return False

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)

        # Ensure repo exists (create if not)
        try:
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        except Exception:
            pass  # Repo likely already exists

        # Upload checkpoint
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"checkpoints/phase{phase}.phase.pt",
            repo_id=repo_id,
            commit_message=f"Phase {phase} checkpoint — {datetime.now().isoformat()}",
        )
        size_mb = path.stat().st_size / 1e6
        print(f"  ✓ Phase {phase} checkpoint uploaded to HuggingFace Hub ({size_mb:.1f} MB)")
        print(f"    https://huggingface.co/{repo_id}")
        return True

    except ImportError:
        print("  ⚠ huggingface_hub not installed — run: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"  ✗ HuggingFace upload failed: {e}")
        return False


def download_checkpoint_from_hf(
    phase: int,
    hf_token: Optional[str] = None,
    repo_id: str = HF_REPO_ID,
) -> bool:
    """
    Download a phase checkpoint from HuggingFace Hub.

    Args:
        phase: phase number to download
        hf_token: HuggingFace API token (optional)
        repo_id: HuggingFace repo ID
    Returns:
        True if download succeeded
    """
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    dest = CHECKPOINT_DIR / f'phase{phase}.phase.pt'

    token = _resolve_hf_token(hf_token)

    try:
        from huggingface_hub import hf_hub_download
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"checkpoints/phase{phase}.phase.pt",
            local_dir=str(CHECKPOINT_DIR.parent),
            token=token,
        )
        print(f"  ✓ Phase {phase} checkpoint downloaded from HuggingFace Hub")
        return True
    except ImportError:
        print("  ⚠ huggingface_hub not installed")
        return False
    except Exception as e:
        print(f"  ℹ Phase {phase} not found on HuggingFace Hub: {e}")
        return False


def upload_logs_to_hf(
    phase: int,
    hf_token: Optional[str] = None,
    repo_id: str = HF_REPO_ID,
) -> bool:
    """Upload phase log file to HuggingFace Hub."""
    log_path = LOGS_DIR / f'phase{phase}.log'
    if not log_path.exists():
        return False

    token = _resolve_hf_token(hf_token)
    if not token:
        return False

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(log_path),
            path_in_repo=f"logs/phase{phase}.log",
            repo_id=repo_id,
            commit_message=f"Phase {phase} logs — {datetime.now().isoformat()}",
        )
        print(f"  ✓ Phase {phase} logs uploaded to HuggingFace Hub")
        return True
    except Exception as e:
        print(f"  ⚠ Log upload failed: {e}")
        return False


def _resolve_hf_token(token: Optional[str] = None) -> Optional[str]:
    """
    Resolve HuggingFace token from multiple sources.
    Priority: explicit arg > HF_TOKEN env var > Kaggle secrets.
    """
    if token:
        return token

    # Environment variable
    env_token = os.environ.get('HF_TOKEN')
    if env_token:
        return env_token

    # Kaggle secrets
    try:
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        kaggle_token = secrets.get_secret("HF_TOKEN")
        if kaggle_token:
            return kaggle_token
    except Exception:
        pass

    # huggingface-cli login token (cached)
    try:
        from huggingface_hub import HfFolder
        cached = HfFolder.get_token()
        if cached:
            return cached
    except Exception:
        pass

    return None


def save_and_upload_checkpoint(
    phase: int,
    state: Dict[str, Any],
    hf_token: Optional[str] = None,
) -> Path:
    """
    Save checkpoint locally AND upload to HuggingFace Hub.
    This is the recommended checkpoint save function for notebooks.
    """
    path = save_checkpoint(phase, state)
    upload_checkpoint_to_hf(phase, hf_token=hf_token)
    return path


# ─────────────────────────────────────────────
# Git Integration (auto-commit logs/results)
# ─────────────────────────────────────────────

def git_commit_and_push(
    files: List[str],
    message: str,
    repo_dir: Optional[str] = None,
) -> bool:
    """
    Stage files, commit, and push to GitHub.
    Silently skips if not in a git repo or push fails.

    Args:
        files: list of file paths (relative to repo root) to stage
        message: commit message
        repo_dir: path to git repo root (auto-detected if None)
    Returns:
        True if push succeeded
    """
    if repo_dir is None:
        repo_dir = str(Path(__file__).parent)

    try:
        for f in files:
            subprocess.run(
                ['git', 'add', f],
                cwd=repo_dir, capture_output=True, timeout=10,
            )
        subprocess.run(
            ['git', 'commit', '-m', message],
            cwd=repo_dir, capture_output=True, timeout=10,
        )
        result = subprocess.run(
            ['git', 'push'],
            cwd=repo_dir, capture_output=True, timeout=30,
        )
        if result.returncode == 0:
            print(f"  ✓ Git push: {message}")
            return True
        else:
            print(f"  ⚠ Git push failed (non-fatal): {result.stderr.decode()[:100]}")
            return False
    except Exception as e:
        print(f"  ⚠ Git commit/push skipped: {e}")
        return False


def clone_or_pull_repo(
    repo_url: str = GITHUB_REPO_URL,
    dest: str = "FLUX",
) -> str:
    """
    Clone the FLUX repo if not present, or pull latest changes.
    Designed for Kaggle/Colab notebook first-cell usage.

    Args:
        repo_url: GitHub repository URL
        dest: destination directory name
    Returns:
        Absolute path to the repo directory
    """
    dest_path = Path(dest).resolve()

    if dest_path.exists() and (dest_path / '.git').exists():
        print(f"  ℹ Repo already exists at {dest_path}, pulling latest...")
        result = subprocess.run(
            ['git', 'pull', '--ff-only'],
            cwd=str(dest_path), capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            print(f"  ✓ Pulled latest changes")
        else:
            print(f"  ⚠ Pull failed (using existing): {result.stderr[:100]}")
    else:
        print(f"  ℹ Cloning {repo_url}...")
        result = subprocess.run(
            ['git', 'clone', repo_url, str(dest_path)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            print(f"  ✓ Cloned to {dest_path}")
        else:
            raise RuntimeError(f"Git clone failed: {result.stderr}")

    return str(dest_path)


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
