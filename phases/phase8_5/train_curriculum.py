"""
Phase 8.5: Train FLUX with Curriculum Learning

Entry point for the ABC School curriculum training.
Loads FLUXLarge from Phase 8 checkpoint and runs the 6-stage curriculum.

Usage:
    python train_curriculum.py            # Full curriculum
    python train_curriculum.py --stage 3  # Resume from stage 3
"""

import sys
import time
import math
import argparse
import torch
from pathlib import Path
from typing import Optional, List

_PHASES_DIR = Path(__file__).parent.parent
_PROJECT_ROOT = _PHASES_DIR.parent
for _phase in ['phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'phase6', 'phase7', 'phase8']:
    _p = str(_PHASES_DIR / _phase)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from flux_large import FLUXLarge, FLUX_LARGE_CONFIG
from curriculum_trainer import CurriculumTrainer, CurriculumResult
from train_openwebtext import load_openwebtext_subset
from flux_utils import (
    get_device, save_checkpoint, load_checkpoint, checkpoint_exists,
    PhaseLogger, PhaseResults,
)


def load_model(device: str) -> FLUXLarge:
    """
    Load FLUXLarge from the best available checkpoint.

    Priority:
    1. Phase 8.5 checkpoint (resume curriculum)
    2. Phase 8 checkpoint (start curriculum from Phase 8 training)
    3. Phase 7 → fresh FLUXLarge (cold start)
    """
    # Try Phase 8 checkpoint first (has trained decoder)
    if checkpoint_exists(8):
        print("  ✓ Loading from Phase 8 checkpoint (trained decoder)")
        return FLUXLarge.from_phase8_checkpoint(device=device)

    # Fall back to Phase 7 → fresh
    print("  ℹ No Phase 8 checkpoint — building from Phase 7")
    return FLUXLarge.from_phase7_checkpoint(device=device)


def save_curriculum_checkpoint(
    model: FLUXLarge,
    curriculum_result: CurriculumResult,
    metrics: dict,
):
    """Save Phase 8.5 checkpoint with curriculum state."""
    from datetime import datetime

    stage_history = []
    for r in curriculum_result.stage_results:
        stage_history.append({
            'stage': r.stage,
            'name': r.name,
            'steps': r.steps_taken,
            'final_loss': r.final_loss,
            'avg_loss': r.avg_loss,
            'min_loss': r.min_loss,
            'accuracy': r.accuracy,
            'advanced': r.advanced,
            'time': r.time_seconds,
        })

    checkpoint_state = {
        'phase': 8.5,
        'timestamp': datetime.now().isoformat(),
        'config': model.config,
        'learning_steps': model._learning_steps,

        # Curriculum-specific
        'curriculum_state': {
            'stages_completed': curriculum_result.stages_completed,
            'final_stage': curriculum_result.final_stage,
            'total_steps': curriculum_result.total_steps,
            'stage_history': stage_history,
        },

        # Component states (same as Phase 8)
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
        'decoder_state_dict': model.decoder.state_dict(),

        'metrics': metrics,
    }

    # Save as phase 8.5 checkpoint
    # flux_utils expects int, so we save manually
    ckpt_path = Path('checkpoints') / 'phase8_5.phase.pt'
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_state, str(ckpt_path))
    size_mb = ckpt_path.stat().st_size / 1e6
    print(f"  ✓ Checkpoint saved: {ckpt_path} ({size_mb:.1f} MB)")
    return ckpt_path


def generate_results(
    curriculum_result: CurriculumResult,
    metrics: dict,
):
    """Generate RESULTS_PHASE_8_5.md via PhaseResults."""
    results = PhaseResults(phase=8, component_name="Curriculum Learning (Phase 8.5)")

    # Stage results as tests
    for r in curriculum_result.stage_results:
        results.add_test(
            f"Stage {r.stage}: {r.name}",
            passed=r.advanced,
            score=f"loss={r.avg_loss:.4f} acc={r.accuracy:.2%}",
            threshold=f"steps={r.steps_taken}",
        )

    # Metrics
    results.add_metric("total_steps", curriculum_result.total_steps)
    results.add_metric("stages_completed", f"{curriculum_result.stages_completed}/6")
    results.add_metric("total_time", f"{curriculum_result.total_time_seconds:.1f}s")

    for r in curriculum_result.stage_results:
        results.add_metric(f"stage{r.stage}_loss", f"{r.avg_loss:.4f}")
        results.add_metric(f"stage{r.stage}_accuracy", f"{r.accuracy:.2%}")

    # Save to phase8_5 directory
    results_path = Path(__file__).parent / 'RESULTS_PHASE_8_5.md'
    results.save(path=str(results_path))
    print(f"  ✓ Results saved: {results_path}")


def main(start_stage: int = 1, max_owt_docs: int = 1000):
    """
    Run the full curriculum training pipeline.

    Args:
        start_stage: Stage to start from (1-6)
        max_owt_docs: Max OpenWebText documents for Stage 6
    """
    print("=" * 60)
    print("  Phase 8.5: Curriculum Learning — ABC School for FLUX")
    print("=" * 60)

    device = get_device()
    log = PhaseLogger(phase=8)  # Log to phase8 log (8.5 is an extension)

    # ── Build model ──
    print("\n  Loading FLUXLarge...")
    model = load_model(device)
    stats = model.get_stats()
    print(f"  ✓ FLUXLarge: {stats.total_params:,} params")

    # ── Load OpenWebText for Stage 6 ──
    print("\n  Loading OpenWebText for Stage 6...")
    owt_texts = load_openwebtext_subset(max_docs=max_owt_docs)
    print(f"  ✓ {len(owt_texts)} documents loaded")

    # ── Run curriculum ──
    trainer = CurriculumTrainer(
        model=model,
        log=log,
        openwebtext_texts=owt_texts,
        verbose=True,
    )

    t0 = time.time()
    result = trainer.run_curriculum(start_stage=start_stage)
    elapsed = time.time() - t0

    # ── Build metrics ──
    metrics = {
        'stages_completed': result.stages_completed,
        'total_steps': result.total_steps,
        'total_time_seconds': elapsed,
        'final_stage': result.final_stage,
    }
    for r in result.stage_results:
        metrics[f'stage{r.stage}_loss'] = r.avg_loss
        metrics[f'stage{r.stage}_accuracy'] = r.accuracy
        metrics[f'stage{r.stage}_steps'] = r.steps_taken

    # ── Save checkpoint ──
    print("\n  Saving checkpoint...")
    save_curriculum_checkpoint(model, result, metrics)

    # ── Generation showcase ──
    print("\n" + "=" * 60)
    print("  Post-Curriculum Generation Test")
    print("=" * 60)

    test_prompts = [
        "The ",
        "I think ",
        "The world is ",
        "She said that ",
        "In the morning ",
    ]

    for prompt in test_prompts:
        generated = model.generate(prompt, max_length=30, temperature=0.7)
        continuation = generated[len(prompt):]
        print(f"  '{prompt}' → '{continuation}'")

    # ── Generate results ──
    generate_results(result, metrics)

    print(f"\n  ✓ Phase 8.5 complete in {elapsed:.1f}s")
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 8.5: Curriculum Training')
    parser.add_argument('--stage', type=int, default=1, help='Start stage (1-6)')
    parser.add_argument('--owt-docs', type=int, default=1000, help='OpenWebText docs for Stage 6')
    args = parser.parse_args()

    main(start_stage=args.stage, max_owt_docs=args.owt_docs)
