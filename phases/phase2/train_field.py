"""
Phase 2 Training: Resonance Field Core

Training has three phases:
  A) Projection warmup (brief gradient-based, ~500 steps)
     - Train wave_to_feature to preserve semantic similarities
     - Train wave_to_location for spatial diversity
  B) Field formation (physics-based, ~3000 perturbations)
     - Feed repeated patterns into field via perturb()
     - Settle periodically for energy minimization
     - Track attractor formation and stability
  C) Validation and checkpoint save
     - Query field for known patterns → measure retrieval accuracy
     - Verify no-forgetting property
     - Save phase2.phase.pt

FIX v2.1: Spatial diversity loss updated to work with SpatialProjection.
  - warmup_projections now calls field.wave_to_location(batch) directly
    which returns tanh-normalized coords in (-1, 1)
  - Spatial diversity loss maximizes pairwise distances in (-1,1) space
  - Added center-avoidance term to penalize collapse toward center
  - loss_div weight increased from 0.1 → 0.3 for stronger spread signal

Usage:
    python train_field.py                       # Auto-detect device
    python train_field.py --device cuda          # Force GPU
    python train_field.py --num-patterns 20      # Fewer patterns (faster)
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from tqdm import tqdm

# ─────────────────────────────────────────────
# Path Setup
# ─────────────────────────────────────────────

PHASE_DIR = Path(__file__).parent
PROJECT_ROOT = PHASE_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PHASE_DIR.parent / 'phase1'))

from flux_utils import (
    get_device, save_checkpoint, load_checkpoint,
    verify_checkpoint_chain, PhaseResults, PhaseLogger,
)
from cse import ContinuousSemanticEncoder
from wave_types import TOTAL_WAVE_DIM
from field import ResonanceField, FIELD_H, FIELD_W, FIELD_D, FIELD_FEATURES
from attractor import AttractorCatalog
from field_ops import compute_field_statistics, find_top_attractors, normalize_field_energy


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Field dimensions
    'field_h': FIELD_H,
    'field_w': FIELD_W,
    'field_d': FIELD_D,
    'field_features': FIELD_FEATURES,
    'wave_dim': TOTAL_WAVE_DIM,

    # Projection warmup
    'warmup_steps': 500,
    'warmup_lr': 1e-3,
    'warmup_batch_size': 32,

    # Field formation
    'num_patterns': 50,
    'repetitions': 20,
    'settle_every': 50,
    'settle_steps': 10,
    'perturbation_strength': 1.0,

    # Validation
    'retrieval_threshold': 0.3,
}


# ─────────────────────────────────────────────
# Phase 1 Loading
# ─────────────────────────────────────────────

def load_phase1_cse(device: str) -> ContinuousSemanticEncoder:
    """Load and freeze the Phase 1 CSE."""
    print("\n── Loading Phase 1 CSE ──")
    checkpoint = load_checkpoint(1)
    config = checkpoint.get('config', {})
    cse = ContinuousSemanticEncoder(**config)
    cse.load_state_dict(checkpoint['state_dict'])
    cse = cse.to(device)
    cse.eval()
    for param in cse.parameters():
        param.requires_grad = False
    print(f"  ✓ CSE loaded and frozen ({sum(p.numel() for p in cse.parameters()):,} params)")
    return cse


# ─────────────────────────────────────────────
# Data Preparation
# ─────────────────────────────────────────────

def prepare_training_texts(num_texts: int = 200) -> list:
    """
    Extract clean text passages from WikiText-2.

    Args:
        num_texts: how many unique texts to extract
    Returns:
        List of text strings
    """
    print("\n── Preparing training data ──")
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = []
        for item in dataset:
            text = item['text'].strip()
            if len(text) > 50 and not text.startswith('='):
                texts.append(text[:200])  # Cap at 200 chars
                if len(texts) >= num_texts:
                    break
        print(f"  ✓ Extracted {len(texts)} text passages from WikiText-2")
        return texts
    except Exception as e:
        print(f"  ⚠ WikiText-2 not available ({e}), using built-in sentences")
        return _fallback_texts(num_texts)


def _fallback_texts(num_texts: int) -> list:
    """Built-in text patterns for when WikiText-2 is unavailable."""
    base = [
        "The cat sat on the mat and watched the birds fly past the window.",
        "Quantum mechanics describes the behavior of particles at atomic scales.",
        "The economy grew significantly during the second quarter of the fiscal year.",
        "Machine learning algorithms process data to discover hidden patterns.",
        "The ancient civilization built pyramids that still stand thousands of years later.",
        "Photosynthesis converts carbon dioxide and water into glucose using sunlight.",
        "Democracy requires the active participation of informed citizens to function.",
        "Neural networks consist of layers of interconnected artificial neurons.",
        "The orchestra performed Beethoven's Fifth Symphony to a standing ovation.",
        "Climate change poses significant challenges to agricultural production worldwide.",
        "Shakespeare wrote thirty-seven plays that explored the human condition.",
        "The human genome contains approximately three billion base pairs of DNA.",
        "Artificial intelligence is transforming how we approach scientific discovery.",
        "The Renaissance period saw remarkable advances in art, science, and philosophy.",
        "Gravitational waves were first detected in September of twenty fifteen.",
        "The industrial revolution fundamentally changed manufacturing processes globally.",
        "Coral reefs support approximately twenty-five percent of all marine species.",
        "The speed of light in a vacuum is approximately three hundred million meters per second.",
        "Modern cryptography relies on the difficulty of factoring large prime numbers.",
        "Plate tectonics explains the movement and formation of Earth's continental plates.",
        "The French Revolution began in seventeen eighty-nine with the storming of the Bastille.",
        "Antibiotics revolutionized medicine by providing effective treatments for bacterial infections.",
        "The Hubble Space Telescope has captured images of galaxies billions of light-years away.",
        "Deep ocean hydrothermal vents support unique ecosystems independent of sunlight.",
        "The theory of evolution by natural selection was proposed by Charles Darwin.",
        "Superconductors carry electrical current with zero resistance below a critical temperature.",
        "The Amazon rainforest produces approximately twenty percent of the world's oxygen.",
        "Fibonacci numbers appear frequently in biological structures like flower petals.",
        "Black holes are regions of spacetime where gravity is so strong nothing can escape.",
        "The printing press invented by Gutenberg transformed the spread of knowledge in Europe.",
    ]
    result = []
    for i in range(num_texts):
        result.append(base[i % len(base)])
    return result


def encode_texts(
    cse: ContinuousSemanticEncoder,
    texts: list,
    device: str,
) -> torch.Tensor:
    """
    Encode texts with CSE and mean-pool to single vectors.

    Args:
        cse: frozen CSE from Phase 1
        texts: list of text strings
        device: computation device
    Returns:
        [N, wave_dim] tensor of wave vectors
    """
    print("\n── Encoding texts with CSE ──")
    vectors = []
    with torch.no_grad():
        for text in tqdm(texts, desc="  Encoding"):
            wave = cse.encode(text)
            vec = wave.full.mean(dim=0)  # [432]
            vectors.append(vec)

    batch = torch.stack(vectors).to(device)
    print(f"  ✓ Encoded {len(texts)} texts → [{batch.shape[0]}, {batch.shape[1]}]")
    return batch


# ─────────────────────────────────────────────
# Phase A: Projection Warmup (FIXED v2.1)
# ─────────────────────────────────────────────

def spatial_diversity_loss(coords: torch.Tensor) -> torch.Tensor:
    """
    Penalize spatial collapse — force coordinates to spread across field.

    coords: [batch, 3] in range (-1, 1) from SpatialProjection tanh output

    Two terms:
      1. Repulsion: maximize pairwise distances between all coordinate pairs
         so different concepts map to different field regions
      2. Center-avoidance: penalize clustering at origin (0,0,0)
         which is the natural attractor of tanh-initialized networks

    Returns scalar loss (minimize to maximize spread).
    """
    # Term 1: pairwise repulsion — push coordinates apart
    pairwise_dists = torch.cdist(coords, coords)  # [bs, bs]
    # Mean of all pairwise distances (excluding diagonal)
    bs = coords.shape[0]
    mask = ~torch.eye(bs, dtype=torch.bool, device=coords.device)
    repulsion = -pairwise_dists[mask].mean()  # maximize distance = minimize negative

    # Term 2: center-avoidance — penalize collapse to origin
    # coords are in (-1,1); we want them spread toward ±1, not clustering at 0
    center_penalty = torch.exp(-coords.pow(2).sum(dim=-1)).mean()

    return repulsion + 0.5 * center_penalty


def warmup_projections(
    field: ResonanceField,
    wave_batch: torch.Tensor,
    config: dict,
    log: PhaseLogger,
):
    """
    Train wave_to_feature and wave_to_location with gradient-based
    optimization for a brief warmup period.

    wave_to_feature: preserve pairwise cosine similarities from waves.
    wave_to_location: encourage spatial diversity (spread patterns out).

    FIX v2.1:
      - wave_to_location is now SpatialProjection (MLP + tanh + normalize)
      - Diversity loss operates on tanh coords in (-1, 1) space
      - loss_div weight increased to 0.3 for stronger spread signal
      - Added center-avoidance term (see spatial_diversity_loss)
      - Diagnostic: prints coordinate spread stats during warmup

    Args:
        field: the ResonanceField (projections will be updated)
        wave_batch: [N, wave_dim] pre-encoded wave vectors
        config: training configuration
        log: phase logger
    """
    print("\n── Phase A: Projection Warmup (v2.1 — spatial diversity fix) ──")
    log.info("Phase A: Projection warmup starting (v2.1)")

    optimizer = torch.optim.Adam(
        list(field.wave_to_location.parameters()) +
        list(field.wave_to_feature.parameters()),
        lr=config['warmup_lr'],
    )

    N = wave_batch.shape[0]
    bs = config['warmup_batch_size']
    steps = config['warmup_steps']

    for step in range(steps):
        # Sample a batch
        idx = torch.randint(0, N, (bs,), device=wave_batch.device)
        batch = wave_batch[idx]

        # ── Feature similarity preservation ──
        features = field.wave_to_feature(batch)  # [bs, features]
        wave_sim = F.cosine_similarity(
            batch.unsqueeze(1), batch.unsqueeze(0), dim=-1
        )  # [bs, bs]
        feat_sim = F.cosine_similarity(
            features.unsqueeze(1), features.unsqueeze(0), dim=-1
        )  # [bs, bs]
        loss_sim = F.mse_loss(feat_sim, wave_sim)

        # ── Location spatial diversity (FIXED) ──
        # SpatialProjection normalizes input and returns tanh coords in (-1, 1)
        # We call the network directly (not wave_to_field_coords which uses no_grad)
        v_norm = F.normalize(batch, dim=-1)               # [bs, wave_dim]
        coords = torch.tanh(field.wave_to_location.net(v_norm))  # [bs, 3]
        loss_div = spatial_diversity_loss(coords)

        # Combined loss — stronger diversity weight than v2.0 (0.1 → 0.3)
        loss = loss_sim + 0.3 * loss_div

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            # Diagnostic: show actual coordinate spread
            with torch.no_grad():
                coord_std = coords.std(dim=0)
                coord_range = coords.max(dim=0).values - coords.min(dim=0).values
            msg = (
                f"Warmup {step}/{steps}: "
                f"loss_sim={loss_sim.item():.4f} "
                f"loss_div={loss_div.item():.4f} "
                f"coord_std=[{coord_std[0]:.3f},{coord_std[1]:.3f},{coord_std[2]:.3f}] "
                f"coord_range=[{coord_range[0]:.3f},{coord_range[1]:.3f},{coord_range[2]:.3f}]"
            )
            print(f"  {msg}")
            log.info(msg)

    # Final diagnostic: verify spread after warmup
    with torch.no_grad():
        sample_idx = torch.randint(0, N, (min(100, N),), device=wave_batch.device)
        sample = wave_batch[sample_idx]
        v_norm = F.normalize(sample, dim=-1)
        final_coords = torch.tanh(field.wave_to_location.net(v_norm))
        final_std = final_coords.std(dim=0)
        final_range = final_coords.max(dim=0).values - final_coords.min(dim=0).values

    print(f"  ✓ Warmup complete ({steps} steps)")
    print(f"  Final coord std:  [{final_std[0]:.3f}, {final_std[1]:.3f}, {final_std[2]:.3f}]")
    print(f"  Final coord range: [{final_range[0]:.3f}, {final_range[1]:.3f}, {final_range[2]:.3f}]")

    # Warn if still collapsed
    if final_std.mean().item() < 0.2:
        print("  ⚠ WARNING: Coordinates still clustered (std < 0.2). "
              "Consider increasing warmup_steps or loss_div weight.")
        log.info("WARNING: Spatial collapse may persist after warmup")
    else:
        print("  ✓ Coordinates well-spread across field")
        log.success(f"Warmup complete — coord spread OK (mean std={final_std.mean():.3f})")


# ─────────────────────────────────────────────
# Phase B: Field Formation
# ─────────────────────────────────────────────

def field_formation(
    field: ResonanceField,
    wave_batch: torch.Tensor,
    config: dict,
    log: PhaseLogger,
) -> list:
    """
    Physics-based field formation via repeated perturbation + settling.
    No gradients — pure field dynamics.

    Each pattern is perturbed 'repetitions' times, creating attractors
    at stable locations. Periodic settling smooths the energy landscape.

    Args:
        field: the ResonanceField
        wave_batch: [N, wave_dim] wave vectors for formation patterns
        config: training configuration
        log: phase logger
    Returns:
        List of (pattern_index, FieldLocation) tuples for tracking
    """
    num_patterns = min(config['num_patterns'], wave_batch.shape[0])
    repetitions = config['repetitions']
    settle_every = config['settle_every']
    settle_steps = config['settle_steps']
    strength = config['perturbation_strength']

    patterns = wave_batch[:num_patterns]

    print(f"\n── Phase B: Field Formation ──")
    print(f"  Patterns: {num_patterns}")
    print(f"  Repetitions: {repetitions}")
    print(f"  Total perturbations: {num_patterns * repetitions}")
    log.info(f"Phase B: {num_patterns} patterns × {repetitions} reps")

    # Show where patterns actually map to (diagnostic for spatial spread)
    print(f"\n  Pattern location sample (first 10):")
    with torch.no_grad():
        unique_locs = set()
        for i in range(min(10, num_patterns)):
            loc = field.wave_to_field_coords(patterns[i])
            unique_locs.add((loc.h, loc.w, loc.d))
            print(f"    Pattern {i:2d} → ({loc.h:3d}, {loc.w:3d}, {loc.d:3d})")
        collision_rate = 1.0 - len(unique_locs) / min(10, num_patterns)
        print(f"  Collision rate (first 10): {collision_rate:.0%} "
              f"({'⚠ HIGH — spatial collapse still present' if collision_rate > 0.3 else '✓ OK'})")
    log.info(f"Pattern collision rate (first 10): {collision_rate:.0%}")

    pattern_locations = []
    total_steps = 0
    start_time = time.time()

    with torch.no_grad():
        for rep in range(repetitions):
            for i in range(num_patterns):
                loc = field.perturb(patterns[i], strength=strength)
                total_steps += 1

                if i == 0 and rep == 0:
                    pattern_locations.append((i, loc))

                # Periodic settling
                if total_steps % settle_every == 0:
                    field.settle(steps=settle_steps)

            # Status update each repetition
            if rep % 5 == 0 or rep == repetitions - 1:
                stats = field.get_field_stats()
                elapsed = time.time() - start_time
                msg = (
                    f"Rep {rep+1}/{repetitions}: "
                    f"attractors={stats['num_attractors']}, "
                    f"energy={stats['total_energy']:.1f}, "
                    f"max_mass={stats['max_mass']:.4f}, "
                    f"elapsed={elapsed:.1f}s"
                )
                print(f"  {msg}")
                log.info(msg)

        # Final settling
        print("  Performing final settling (50 steps)...")
        field.settle(steps=50)
        normalize_field_energy(field)

    elapsed = time.time() - start_time
    final_stats = field.get_field_stats()
    print(f"  ✓ Field formation complete in {elapsed:.1f}s")
    print(f"    Final attractors: {final_stats['num_attractors']}")
    print(f"    Final energy: {final_stats['total_energy']:.1f}")
    print(f"    Max mass: {final_stats['max_mass']:.4f}")
    log.success(f"Field formation complete: {final_stats['num_attractors']} attractors in {elapsed:.1f}s")

    # Record all pattern locations after formation
    pattern_locations = []
    for i in range(num_patterns):
        loc = field.wave_to_field_coords(patterns[i])
        pattern_locations.append((i, loc))

    return pattern_locations


# ─────────────────────────────────────────────
# Phase C: Validation
# ─────────────────────────────────────────────

def validate_field(
    field: ResonanceField,
    wave_batch: torch.Tensor,
    num_patterns: int,
    texts: list,
    config: dict,
    log: PhaseLogger,
) -> dict:
    """
    Validate the trained field by querying for known patterns.

    Args:
        field: the trained ResonanceField
        wave_batch: [N, wave_dim] wave vectors
        num_patterns: number of patterns used in formation
        texts: original text strings
        config: training configuration
        log: phase logger
    Returns:
        Dict with retrieval accuracy and similarity scores
    """
    print("\n── Phase C: Validation ──")
    log.info("Phase C: Validation starting")

    threshold = config['retrieval_threshold']
    patterns = wave_batch[:num_patterns]

    hits = 0
    total_sim = 0.0
    sims_list = []

    with torch.no_grad():
        for i in range(num_patterns):
            features, sims, locs = field.query(patterns[i])
            top_sim = sims[0].item() if sims.numel() > 0 else 0.0
            total_sim += top_sim
            sims_list.append(top_sim)
            if top_sim > threshold:
                hits += 1

    accuracy = hits / num_patterns
    avg_sim = total_sim / num_patterns

    print(f"  Retrieval accuracy (sim > {threshold}): {hits}/{num_patterns} = {accuracy:.1%}")
    print(f"  Average top-1 similarity: {avg_sim:.4f}")
    print(f"  Min similarity: {min(sims_list):.4f}")
    print(f"  Max similarity: {max(sims_list):.4f}")
    log.metric("retrieval_accuracy", f"{accuracy:.4f}")
    log.metric("avg_similarity", f"{avg_sim:.4f}")

    # Check attractor formation
    catalog = AttractorCatalog(field)
    new_attractors = catalog.scan_and_update(mass_threshold=0.1)
    print(f"  Discovered {new_attractors} attractors via scan")
    log.metric("discovered_attractors", new_attractors)

    # Top attractors
    top = find_top_attractors(field, k=5)
    if top:
        print("  Top 5 attractors by mass:")
        for a in top:
            print(f"    {a['location']}: mass={a['mass']:.4f}, energy={a['energy']:.4f}")

    # Spatial spread diagnostic
    print(f"\n  Spatial spread of top {min(20, num_patterns)} patterns:")
    locs_seen = []
    with torch.no_grad():
        for i in range(min(20, num_patterns)):
            loc = field.wave_to_field_coords(patterns[i])
            locs_seen.append((loc.h, loc.w, loc.d))
    unique_locs = len(set(locs_seen))
    print(f"  Unique locations: {unique_locs}/{min(20, num_patterns)} "
          f"({'✓ well spread' if unique_locs > min(20, num_patterns) * 0.7 else '⚠ still some collision'})")
    log.metric("unique_locations_20", unique_locs)

    return {
        'retrieval_accuracy': accuracy,
        'avg_similarity': avg_sim,
        'min_similarity': min(sims_list),
        'max_similarity': max(sims_list),
        'all_similarities': sims_list,
        'num_attractors': new_attractors,
        'catalog': catalog,
    }


# ─────────────────────────────────────────────
# Main Training Script
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Resonance Field Core Training")
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/mps/cpu)')
    parser.add_argument('--num-patterns', type=int, default=None, help='Number of patterns')
    parser.add_argument('--repetitions', type=int, default=None, help='Repetitions per pattern')
    parser.add_argument('--warmup-steps', type=int, default=None, help='Projection warmup steps')
    args = parser.parse_args()

    # ── Setup ──
    device = args.device or get_device()
    config = dict(DEFAULT_CONFIG)
    if args.num_patterns:
        config['num_patterns'] = args.num_patterns
    if args.repetitions:
        config['repetitions'] = args.repetitions
    if args.warmup_steps:
        config['warmup_steps'] = args.warmup_steps

    log = PhaseLogger(phase=2)
    log.separator("Phase 2: Resonance Field Core")
    log.info(f"Device: {device}")
    log.info(f"Config: {config}")

    print("=" * 60)
    print("  Phase 2: Resonance Field Core — Training (v2.1)")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Field: {config['field_h']}×{config['field_w']}×{config['field_d']}×{config['field_features']}")

    start_time = time.time()

    # ── Verify Phase 1 ──
    assert verify_checkpoint_chain(up_to_phase=2), (
        "Phase 1 checkpoint missing. Run Phase 1 training first."
    )

    # ── Load Phase 1 CSE ──
    cse = load_phase1_cse(device)

    # ── Create ResonanceField ──
    print("\n── Creating ResonanceField ──")
    field = ResonanceField(
        h=config['field_h'],
        w=config['field_w'],
        d=config['field_d'],
        features=config['field_features'],
        wave_dim=config['wave_dim'],
    ).to(device)

    total_params = sum(p.numel() for p in field.parameters())
    buffer_size = sum(b.numel() for b in field.buffers())
    print(f"  ✓ Field created: {total_params:,} trainable params, {buffer_size:,} buffer elements")
    print(f"  Buffer memory: ~{buffer_size * 4 / 1e6:.0f} MB (float32)")
    log.info(f"Field: {total_params:,} params, {buffer_size:,} buffers (~{buffer_size*4/1e6:.0f} MB)")

    # ── Prepare Data ──
    texts = prepare_training_texts(num_texts=max(200, config['num_patterns'] * 2))
    wave_batch = encode_texts(cse, texts, device)

    # ── Phase A: Warmup ──
    warmup_projections(field, wave_batch, config, log)

    # ── Phase B: Field Formation ──
    pattern_locations = field_formation(field, wave_batch, config, log)

    # ── Phase C: Validation ──
    val_results = validate_field(
        field, wave_batch, config['num_patterns'], texts, config, log
    )

    # ── Save Checkpoint ──
    print("\n── Saving Phase 2 Checkpoint ──")
    total_time = time.time() - start_time
    field_stats = compute_field_statistics(field)

    checkpoint_state = {
        'config': {
            'field': {
                'h': config['field_h'],
                'w': config['field_w'],
                'd': config['field_d'],
                'features': config['field_features'],
                'wave_dim': config['wave_dim'],
            },
            'cse': cse.wave_dims if hasattr(cse, 'wave_dims') else {},
        },
        'state_dict': field.state_dict(),
        'cse_state_dict': cse.state_dict(),
        'metrics': {
            'retrieval_accuracy': val_results['retrieval_accuracy'],
            'avg_similarity': val_results['avg_similarity'],
            'num_attractors': field.num_attractors(),
            'total_energy': field.total_energy(),
            'training_time_seconds': total_time,
            'field_stats': field_stats,
        },
        'attractor_catalog': val_results['catalog'].to_dict(),
        'training_config': config,
        'version': '2.1',
    }
    save_checkpoint(2, checkpoint_state)

    # ── Save .flx format too ──
    from flux_format import save_flux
    flx_path = PROJECT_ROOT / 'checkpoints' / 'phase2.flx'
    save_flux(field, cse, str(flx_path), val_results['catalog'],
              metadata={'steps': field.step_count, 'version': '2.1'})

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  Phase 2 Training Complete! (v2.1 — spatial diversity fix)")
    print("=" * 60)
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Retrieval accuracy: {val_results['retrieval_accuracy']:.1%}")
    print(f"  Average similarity: {val_results['avg_similarity']:.4f}")
    print(f"  Field attractors: {field.num_attractors()}")
    print(f"  Total energy: {field.total_energy():.1f}")
    print(f"  Checkpoint: checkpoints/phase2.phase.pt")
    print(f"  FLUX file: checkpoints/phase2.flx")

    log.separator("Training Summary")
    log.metric("total_time", f"{total_time:.1f}s")
    log.metric("retrieval_accuracy", f"{val_results['retrieval_accuracy']:.4f}")
    log.metric("num_attractors", field.num_attractors())
    log.success("Phase 2 v2.1 training complete")


if __name__ == '__main__':
    main()