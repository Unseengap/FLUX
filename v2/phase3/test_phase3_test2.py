"""
test_phase3_test2.py — Context Sensitivity: Different prompts → different waves

The context collapse problem (legacy Phase 9 failure) was: all prompts produced
nearly identical context vectors, so all generated output was the same.

This test verifies that:
    - Semantically different prompts produce measurably different output waves
    - Average pairwise cosine of generated waves < 0.85  (diversity threshold)
    - Prompts that are similar to each other produce more similar output than
      prompts that are dissimilar (ordering check)

Pass criteria:
    avg inter-prompt cosine  < 0.85   (diverse generation)
    context sensitivity ordering OK   (similar prompts → more similar output)
    self-similarity == 1.0            (deterministic: same prompt → same output)
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from itertools import combinations
from typing import List, Tuple

# ─────────────────────────────────────────────
# Path setup
# ─────────────────────────────────────────────
_V2_DIR       = Path(__file__).parent.parent
_PHASE1_DIR   = _V2_DIR / 'phase1'
_PHASE2_DIR   = _V2_DIR / 'phase2'
_PROJECT_ROOT = _V2_DIR.parent

for _p in [str(_PHASE1_DIR), str(_PHASE2_DIR), str(Path(__file__).parent), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cse            import ContinuousSemanticEncoder
from wave_to_field  import WaveToField
from wave_generator import WaveGenerator
from flux_utils     import PhaseResults, get_device

# ─────────────────────────────────────────────
# Test prompts — deliberately diverse
# ─────────────────────────────────────────────
DIVERSE_PROMPTS = [
    "The cat sat on the mat",                              # simple English
    "Water freezes at zero degrees Celsius",               # science fact
    "def hello(): return 'world'",                         # code
    "café naïve résumé",                                   # UTF-8
    "The future of artificial intelligence",               # abstract concept
    "∫₀^∞ e^(-x²) dx = √π/2",                            # math
    "Photosynthesis converts sunlight into energy",        # biology
    "The quick brown fox jumps over the lazy dog",         # pangram
]

# Similar pairs — these two SHOULD produce more similar outputs than cross-domain pairs
SIMILAR_PAIRS = [
    ("The cat sat on the mat", "The dog sat on the rug"),
    ("Water is H₂O", "Water freezes at zero degrees Celsius"),
    ("def foo(): pass", "def hello(): return 'world'"),
]

DISSIMILAR_PAIRS = [
    ("The cat sat on the mat", "∫₀^∞ e^(-x²) dx = √π/2"),
    ("Water freezes at zero degrees Celsius", "def hello(): return 'world'"),
    ("café naïve résumé", "The future of artificial intelligence"),
]

AVG_COSINE_THRESHOLD = 0.85   # generated waves should differ across prompts
ORDERING_MUST_PASS   = 2       # out of 3 similar/dissimilar ordering checks


def load_components(ckpt_dir: Path, device: str):
    """Load CSE, WaveToField, WaveGenerator."""
    p1  = torch.load(ckpt_dir / 'phase1_v2.phase.pt', map_location='cpu')
    cfg1 = p1['config']
    cse = ContinuousSemanticEncoder(
        wave_dim=cfg1.get('wave_dim', 432),
        window_size=cfg1.get('window_size', 8),
        stride=cfg1.get('stride', 1),
    )
    cse.load_state_dict(p1['state_dict']['cse'])
    cse.to(device).eval()

    p2  = torch.load(ckpt_dir / 'phase2_v2.phase.pt', map_location='cpu')
    cfg2 = p2['config']
    w2f = WaveToField(
        wave_dim=cfg2.get('wave_dim', 432),
        field_dim=cfg2.get('field_features', 512),
    )
    w2f.load_state_dict(p2['state_dict']['wave_to_field'])
    w2f.to(device).eval()

    p3  = torch.load(ckpt_dir / 'phase3_v2.phase.pt', map_location='cpu')
    cfg3 = p3['config']
    generator = WaveGenerator(
        wave_dim=cfg3.get('wave_dim', 432),
        field_features=cfg3.get('field_features', 512),
        gru_hidden=cfg3.get('gru_hidden', 512),
        gru_layers=cfg3.get('gru_layers', 1),
        dropout=0.0,
    )
    generator.load_state_dict(p3['state_dict']['generator'])
    generator.to(device).eval()

    return cse, w2f, generator


@torch.no_grad()
def get_mean_generated_wave(
    text:      str,
    cse:       ContinuousSemanticEncoder,
    w2f:       WaveToField,
    generator: WaveGenerator,
    device:    str,
    max_waves: int = 10,
) -> torch.Tensor:
    """Encode text → field context → generate waves → return mean wave [432]."""
    wave      = cse.encode(text)
    mean_wave = wave.full.mean(dim=0).to(device)
    ctx       = w2f(mean_wave)
    waves, _  = generator.generate(field_context=ctx, max_waves=max_waves)
    return waves.mean(dim=0)   # [432]


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two 1D tensors."""
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    device   = get_device()
    ckpt_dir = _PROJECT_ROOT / 'checkpoints'

    assert (ckpt_dir / 'phase3_v2.phase.pt').exists(), \
        "Phase 3 checkpoint not found. Run train_generator.py first."

    cse, w2f, generator = load_components(ckpt_dir, device)
    results = PhaseResults(phase=3, component_name="Wave Generator")

    # ── Part A: Diversity across diverse prompts ──
    print("\n── Test 2a: Context Diversity (anti-collapse) ──\n")
    gen_waves = {}
    for text in DIVERSE_PROMPTS:
        gen_waves[text] = get_mean_generated_wave(text, cse, w2f, generator, device)

    pairwise = []
    for (t1, w1), (t2, w2) in combinations(gen_waves.items(), 2):
        sim = cosine(w1, w2)
        pairwise.append(sim)

    avg_cosine = sum(pairwise) / len(pairwise)
    print(f"  Pairwise cosine across {len(DIVERSE_PROMPTS)} prompts:")
    print(f"    avg={avg_cosine:.4f}  max={max(pairwise):.4f}  min={min(pairwise):.4f}")
    diversity_ok = avg_cosine < AVG_COSINE_THRESHOLD

    results.add_test(
        f"Avg inter-prompt cosine < {AVG_COSINE_THRESHOLD}",
        passed=diversity_ok,
        score=avg_cosine,
        threshold=AVG_COSINE_THRESHOLD,
    )
    print(f"  {'✓' if diversity_ok else '✗'} avg_cosine={avg_cosine:.4f} < {AVG_COSINE_THRESHOLD}")

    # ── Part B: Similar prompts → more similar output than dissimilar prompts ──
    print("\n── Test 2b: Context Sensitivity Ordering ──\n")
    ordering_passes = 0
    for (s1, s2), (d1, d2) in zip(SIMILAR_PAIRS, DISSIMILAR_PAIRS):
        w_s1 = get_mean_generated_wave(s1, cse, w2f, generator, device)
        w_s2 = get_mean_generated_wave(s2, cse, w2f, generator, device)
        w_d1 = get_mean_generated_wave(d1, cse, w2f, generator, device)
        w_d2 = get_mean_generated_wave(d2, cse, w2f, generator, device)

        sim_similar    = cosine(w_s1, w_s2)
        sim_dissimilar = cosine(w_d1, w_d2)
        ok             = sim_similar > sim_dissimilar

        ordering_passes += int(ok)
        print(f"  {'✓' if ok else '✗'}  similar={sim_similar:.3f}  dissimilar={sim_dissimilar:.3f}")
        print(f"      similar:    {s1[:35]!r}")
        print(f"      dissimilar: {d1[:35]!r}")

    ordering_ok = ordering_passes >= ORDERING_MUST_PASS
    results.add_test(
        f"Context ordering: {ORDERING_MUST_PASS}/3 pairs correct",
        passed=ordering_ok,
        score=ordering_passes / 3,
        threshold=ORDERING_MUST_PASS / 3,
    )
    print(f"\n  {'✓' if ordering_ok else '✗'} ordering_passes={ordering_passes}/3")

    # ── Part C: Determinism (same prompt → same output every time) ──
    print("\n── Test 2c: Determinism ──\n")
    test_text = "The cat sat on the mat"
    w_a = get_mean_generated_wave(test_text, cse, w2f, generator, device)
    w_b = get_mean_generated_wave(test_text, cse, w2f, generator, device)
    self_sim   = cosine(w_a, w_b)
    deterministic = self_sim > 0.999

    results.add_test(
        "Self-similarity == 1.0 (deterministic)",
        passed=deterministic,
        score=self_sim,
        threshold=1.0,
    )
    print(f"  {'✓' if deterministic else '✗'} self_sim={self_sim:.6f}")

    results.save()

    assert diversity_ok,  f"FAIL avg_cosine={avg_cosine:.4f} ≥ {AVG_COSINE_THRESHOLD}"
    assert ordering_ok,   f"FAIL ordering_passes={ordering_passes}/3 < {ORDERING_MUST_PASS}"
    assert deterministic, f"FAIL self_sim={self_sim:.6f} not deterministic"

    print("\n  ✓ Test 2 PASSED")


if __name__ == '__main__':
    main()
