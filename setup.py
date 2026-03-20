#!/usr/bin/env python3
"""
FLUX Project Setup Script
Run this once to initialize the project structure.

Usage: python setup.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent

DIRS = [
    'checkpoints',
    'results', 
    'demos',
    'phases/phase1',
    'phases/phase2',
    'phases/phase3',
    'phases/phase4',
    'phases/phase5',
    'phases/phase6',
    'phases/phase7',
    'phases/phase8',
    'shared/utils',
    'shared/data',
    'shared/eval',
]

GITKEEPS = [
    'checkpoints/.gitkeep',
    'results/.gitkeep',
]

def create_structure():
    print("Creating FLUX project structure...")
    for d in DIRS:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {d}/")
    
    for g in GITKEEPS:
        p = ROOT / g
        if not p.exists():
            p.touch()

def check_gpu():
    print("\nChecking hardware...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✓ VRAM: {mem:.1f} GB")
            if mem < 8:
                print("  ⚠ Warning: < 8GB VRAM — Phase 1-4 OK, Phase 5+ may be slow")
        else:
            print("  ⚠ No GPU detected — CPU only, training will be slow")
            print("  Phases 1-3 are fine on CPU. Phase 4+ needs GPU.")
    except ImportError:
        print("  ✗ PyTorch not installed. Run: pip install -r requirements.txt")

def check_dependencies():
    print("\nChecking dependencies...")
    deps = {
        'torch': 'PyTorch',
        'numpy': 'NumPy', 
        'matplotlib': 'Matplotlib',
        'tqdm': 'tqdm',
        'datasets': 'HuggingFace Datasets',
        'faiss': 'FAISS (try: pip install faiss-gpu or faiss-cpu)',
    }
    for module, name in deps.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} — missing")

def print_next_steps():
    print("\n" + "="*50)
    print("FLUX Project Ready!")
    print("="*50)
    print("\nNext steps:")
    print("  1. cd phases/phase1")
    print("  2. Open PHASE_1_SPEC.md in your editor")
    print("  3. Open SPECIFICATION.md alongside it")
    print("  4. Start building with Copilot: wave_types.py first")
    print("\nPhase 1 acceptance criteria:")
    print("  python test_phase1_test1.py")
    print("  python test_phase1_test2.py") 
    print("  python test_phase1_test3.py")
    print("  python demo_phase1_demo1.py")
    print("  python demo_phase1_demo2.py")
    print("\nWhen all pass → checkpoint auto-saved → Phase 2 unlocks")

if __name__ == '__main__':
    create_structure()
    check_gpu()
    check_dependencies()
    print_next_steps()
