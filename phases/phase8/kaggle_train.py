"""
Phase 8: Kaggle-Optimized Training Script

Designed for Kaggle P100/T4 GPU environment:
- Gradient accumulation (effective batch = 32)
- Mixed precision (fp16) where available
- Checkpoint every 5000 steps to Kaggle output
- Resume-safe: loads latest checkpoint if interrupted
- Memory-efficient: streams data, minimal RAM footprint
- Estimated training time: 48-72 hours on dual T4 (full)
- Quick mode: ~30 min for subset training

Usage (Kaggle notebook):
    %run kaggle_train.py --quick     # Quick mode (~1000 docs, ~30 min)
    %run kaggle_train.py --full      # Full mode (~100k docs, ~48-72 hrs)
"""

import sys
import time
import argparse
import torch
from pathlib import Path
from datetime import datetime

_PHASES_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PHASES_DIR.parent
for _phase in ['phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'phase6', 'phase7']:
    _p = str(_PHASES_DIR / _phase)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from flux_large import FLUXLarge
from train_openwebtext import OpenWebTextTrainer, load_openwebtext_subset
from flux_utils import (
    get_device, save_checkpoint, load_checkpoint, checkpoint_exists,
    upload_checkpoint_to_hf, PhaseLogger, _resolve_hf_token,
)


def kaggle_train(
    mode: str = 'quick',
    device: str = 'auto',
    log: PhaseLogger = None,
) -> dict:
    """
    Run Kaggle-optimized FLUX training.

    Args:
        mode: 'quick' (1000 docs, ~30 min) or 'full' (100k docs, ~48-72 hrs)
        device: Compute device (auto-detected if 'auto')
        log: PhaseLogger instance

    Returns:
        Dict with training and evaluation metrics
    """
    if device == 'auto':
        device = get_device()
    if log is None:
        log = PhaseLogger(phase=8)

    # ── Configuration based on mode ──
    if mode == 'quick':
        max_docs = 1000
        checkpoint_interval = 500
        log_interval = 50
    elif mode == 'full':
        max_docs = 100000
        checkpoint_interval = 5000
        log_interval = 500
    else:
        max_docs = 500
        checkpoint_interval = 100
        log_interval = 25

    print(f"\n  Mode: {mode}")
    print(f"  Max docs: {max_docs:,}")
    print(f"  Checkpoint every: {checkpoint_interval} steps")
    print(f"  Device: {device}")

    # ── Build or resume model ──
    if checkpoint_exists(8):
        print("\n  ℹ Resuming from Phase 8 checkpoint...")
        model = FLUXLarge.from_phase8_checkpoint(device=device)
        resumed = True
    else:
        print("\n  ℹ Building FLUXLarge from scratch...")
        model = FLUXLarge.from_phase7_checkpoint(device=device)
        resumed = False

    stats = model.get_stats()
    log.info(f"Model: {stats.total_params:,} params")
    print(f"  Model: {stats.total_params:,} params")

    # ── Load data ──
    texts = load_openwebtext_subset(max_docs=max_docs)
    split_idx = int(len(texts) * 0.9)
    train_texts = texts[:split_idx]
    eval_texts = texts[split_idx:]

    print(f"  Train: {len(train_texts):,} docs")
    print(f"  Eval:  {len(eval_texts):,} docs")

    # ── Train ──
    trainer = OpenWebTextTrainer(
        model,
        lr=5e-4,
        grad_accum=4,
        checkpoint_interval=checkpoint_interval,
        log=log,
    )

    log.separator("Training Start")
    t0 = time.time()

    result = trainer.train_on_texts(
        train_texts,
        verbose=True,
        log_interval=log_interval,
    )

    elapsed = time.time() - t0
    log.separator("Training Complete")

    # ── Evaluate ──
    eval_metrics = trainer.evaluate(eval_texts)
    log.metric("eval_loss", f"{eval_metrics['avg_loss']:.4f}")
    log.metric("eval_ppl", f"{eval_metrics['avg_perplexity']:.2f}")

    # ── Final checkpoint ──
    metrics = {
        'mode': mode,
        'total_steps': result.total_steps,
        'final_loss': result.final_loss,
        'final_perplexity': result.final_perplexity,
        'avg_loss': result.avg_loss,
        'avg_perplexity': result.avg_perplexity,
        'min_loss': result.min_loss,
        'eval_loss': eval_metrics['avg_loss'],
        'eval_perplexity': eval_metrics['avg_perplexity'],
        'total_tokens': result.total_tokens,
        'training_time_seconds': elapsed,
        'steps_per_second': result.steps_per_second,
        'resumed': resumed,
    }

    # Save final checkpoint
    state = {
        'phase': 8,
        'config': model.config,
        'learning_steps': model._learning_steps,
        'cse_state_dict': model.cse.state_dict(),
        'field_state_dict': model.field.state_dict(),
        'gr_state': model.gr.save_state(),
        'tl_state': model.tl.save_state(),
        'cgn_state': model.cgn.save_state(),
        'causal_graph_state': model.causal_graph.save_state(),
        'working_memory_state': model.working_memory.state_dict_memory(),
        'episodic_memory_state': model.episodic_memory.save_state(),
        'semantic_memory_state': model.semantic_memory.save_state(),
        'router_state': model.memory_router.save_state(),
        'wave_to_field_state': model.wave_to_field.state_dict(),
        'field_to_wave_state': model.field_to_wave.state_dict(),
        'output_head_state': model.output_head.state_dict(),
        'metrics': metrics,
    }
    save_checkpoint(8, state)
    log.success("Final checkpoint saved")

    # ── Upload ──
    hf_token = _resolve_hf_token()
    if hf_token:
        upload_checkpoint_to_hf(phase=8, hf_token=hf_token)
        log.success("Checkpoint uploaded to HuggingFace Hub")

    print(f"\n  ═══ Training Summary ═══")
    print(f"    Steps:       {result.total_steps:,}")
    print(f"    Final loss:  {result.final_loss:.4f}")
    print(f"    Final ppl:   {result.final_perplexity:.2f}")
    print(f"    Eval loss:   {eval_metrics['avg_loss']:.4f}")
    print(f"    Eval ppl:    {eval_metrics['avg_perplexity']:.2f}")
    print(f"    Tokens:      {result.total_tokens:,}")
    print(f"    Time:        {elapsed:.1f}s ({elapsed/60:.1f} min)")

    return metrics


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kaggle FLUX training')
    parser.add_argument('--quick', action='store_true', help='Quick mode (1000 docs)')
    parser.add_argument('--full', action='store_true', help='Full mode (100k docs)')
    parser.add_argument('--device', default='auto', help='Device (cuda/cpu/auto)')
    args = parser.parse_args()

    mode = 'full' if args.full else 'quick'
    log = PhaseLogger(phase=8)
    kaggle_train(mode=mode, device=args.device, log=log)
