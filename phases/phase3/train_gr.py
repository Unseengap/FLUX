import sys
import time
from pathlib import Path
from datetime import datetime
import torch
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'phase1'))
sys.path.append(str(Path(__file__).parent.parent / 'phase2'))
from flux_utils import load_checkpoint, save_checkpoint, PhaseLogger, get_device, checkpoint_exists
from cse import ContinuousSemanticEncoder
from field import ResonanceField
from gravity import GravitationalRelevance
from sanity_decoder import SanityDecoder, pipeline_check
from benchmark_attention import run_benchmark

# ─────────────────────────────────────────────
# Pipeline test sentences
# ─────────────────────────────────────────────
PIPELINE_TEST_SENTENCES = [
    "the quick brown fox",
    "hello world",
    "artificial intelligence",
    "the cat sat on the mat",
    "once upon a time",
    "machine learning algorithms",
    "the ocean covers earth",
    "quantum mechanics",
    "stars are giant plasma balls",
    "democracy requires participation",
]


def main():
    log = PhaseLogger(phase=3)
    log.separator("Phase 3: Gravitational Relevance — train_gr.py")

    if checkpoint_exists(3):
        log.info("Phase 3 checkpoint exists. Skipping.")
        return

    device = get_device()
    log.info(f"Device: {device}")

    # ─────────────────────────────────────────────
    # Load Phase 1 (CSE) and Phase 2 (Field)
    # ─────────────────────────────────────────────
    log.cell_start("Load Phase 1 & 2 checkpoints")
    ckpt1 = load_checkpoint(1)
    cse = ContinuousSemanticEncoder(**ckpt1.get('config', {})).to(device)
    cse.load_state_dict(ckpt1['state_dict'])
    cse.eval()
    log.success(f"Phase 1 CSE loaded: {sum(p.numel() for p in cse.parameters()):,} params")

    ckpt2 = load_checkpoint(2)
    field_cfg = ckpt2.get('config', {}).get('field', {})
    field = ResonanceField(**field_cfg).to(device)
    field.load_state_dict(ckpt2['state_dict'])
    field.eval()
    log.success(f"Phase 2 Field loaded: {sum(p.numel() for p in field.parameters()):,} params")
    log.cell_end("Load Phase 1 & 2 checkpoints", "PASS")

    # ─────────────────────────────────────────────
    # Build Phase 3 components
    # ─────────────────────────────────────────────
    gr = GravitationalRelevance(feature_dim=512, device=device).to(device)
    decoder = SanityDecoder(feature_dim=512, device=device).to(device)
    log.info(f"GR parameters: {sum(p.numel() for p in gr.parameters()):,}")
    log.info(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")

    # ─────────────────────────────────────────────
    # Stage A: Benchmark GR vs attention
    # ─────────────────────────────────────────────
    log.cell_start("Benchmark GR vs Attention")
    benchmark_results = run_benchmark(gr, device=device)
    # Capture specific speedups for metrics
    speedup_1024 = next(
        (r.get('speedup', 0.0) for r in benchmark_results if r['seq_len'] == 1024),
        0.0,
    )
    speedup_4096 = next(
        (r.get('speedup', 0.0) for r in benchmark_results if r['seq_len'] == 4096),
        0.0,
    )
    log.metric("speedup_1024", str(speedup_1024))
    log.metric("speedup_4096", str(speedup_4096))
    log.cell_end("Benchmark GR vs Attention", "PASS")

    # ─────────────────────────────────────────────
    # Stage B: GR warmup via field output
    # ─────────────────────────────────────────────
    log.cell_start("GR Warmup")
    warmup_texts = [
        "The cat sat on the mat.", "Quantum mechanics at atomic scales.",
        "Machine learning algorithms discover patterns.", "Ancient pyramids still stand.",
        "Photosynthesis converts sunlight into glucose.", "Democracy requires participation.",
        "Neural networks learn from data.", "The orchestra performed Beethoven.",
        "Climate change challenges agriculture.", "Stars are plasma fusing hydrogen.",
    ] * 5
    start_time = time.time()
    gr.train()
    with torch.no_grad():
        for text in warmup_texts:
            wave = cse.encode(text)
            vec  = wave.full.mean(dim=0).to(device)
            field_out, _, _ = field.query(vec, k=8)
            gr(field_out.unsqueeze(0))
    log.success(f"Warmup complete: {gr.mass_tracker.count} concepts, {time.time()-start_time:.1f}s")
    log.cell_end("GR Warmup", "PASS")

    # ─────────────────────────────────────────────
    # Stage C: End-to-end pipeline check
    # ─────────────────────────────────────────────
    log.cell_start("Pipeline Check")
    pipeline_results = []
    for text in PIPELINE_TEST_SENTENCES:
        result = pipeline_check(cse, field, gr, decoder, text, verbose=True)
        pipeline_results.append(result)
    pass_rate = sum(r['recognizable'] for r in pipeline_results) / len(pipeline_results)
    log.metric("pipeline_check_pass_rate", f"{pass_rate:.2f}")
    log.success(f"Pipeline: {sum(r['recognizable'] for r in pipeline_results)}/{len(pipeline_results)} recognizable")
    log.cell_end("Pipeline Check", "PASS")

    # ─────────────────────────────────────────────
    # Save checkpoint — full Phase 3 schema
    # ─────────────────────────────────────────────
    log.cell_start("Save Checkpoint")
    mass_stats = gr.mass_tracker.stats()
    checkpoint_state = {
        'phase': 3,
        'timestamp': datetime.now().isoformat(),
        # Previous phase configs (chain intact)
        'phase1_config': ckpt1.get('config', {}),
        'phase2_config': ckpt2.get('config', {}),
        # Phase 3 components
        'phase3_gr_state': gr.save_state(),
        'phase3_decoder_state': decoder.state_dict(),
        # Benchmark recorded at checkpoint time
        'benchmark_results': benchmark_results,
        # Metrics
        'metrics': {
            'pipeline_check_pass_rate': pass_rate,
            'speedup_vs_attention_1024': speedup_1024,
            'speedup_vs_attention_4096': speedup_4096,
            'mass_tracker_count': gr.mass_tracker.count,
            'mass_tracker_mean': mass_stats.get('mean_mass', 0.0),
            'negative_mass_count': mass_stats.get('negative_count', 0),
        },
    }
    save_checkpoint(3, checkpoint_state)
    log.success("Phase 3 checkpoint saved to checkpoints/phase3.phase.pt")
    log.cell_end("Save Checkpoint", "PASS")


if __name__ == "__main__":
    main()
