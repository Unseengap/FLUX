"""
test_phase3_test1.py — Decode Gate: Generated waves decode to readable text

Loads phase3_v2.phase.pt + all frozen components and measures how well
the generator's output decodes back to the original prompt text.

Pass criteria:
    avg byte accuracy  > 50%  (generation quality threshold — lower than Phase 1
                                because we're comparing generated output vs original,
                                not a round-trip encode/decode)
    min byte accuracy  > 20%  (no complete failures)
    at least 5/8 texts score > 30%
"""

import sys
import torch
from pathlib import Path
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

from cse           import ContinuousSemanticEncoder
from wave_chunker  import WaveChunker
from wave_to_text  import WaveToText
from wave_to_field import WaveToField
from field_to_wave import FieldToWave
from wave_generator import WaveGenerator
from flux_utils    import PhaseResults, get_device

# ─────────────────────────────────────────────
# Test texts (same 8 as decode gate)
# ─────────────────────────────────────────────
TEST_TEXTS = [
    "The future of artificial intelligence",
    "Energy equals mass times the speed of light squared",
    "Photosynthesis converts sunlight into chemical energy",
    "Water freezes at zero degrees Celsius",
    "The cat sat on the mat",
    "café naïve résumé",
    "def hello(): return 'world'",
    "∫₀^∞ e^(-x²) dx = √π/2",
]

AVG_THRESHOLD   = 0.50
MIN_THRESHOLD   = 0.20
PER_TEXT_THRESH = 0.30
N_MUST_PASS     = 5


def byte_accuracy(a: bytes, b: bytes) -> float:
    """Fraction of bytes that match (padded to longer length)."""
    n     = max(len(a), len(b), 1)
    match = sum(x == y for x, y in zip(a, b))
    return match / n


def load_all_components(checkpoint_dir: Path, device: str):
    """Load all frozen + generator components from checkpoints."""
    # Phase 1
    p1 = torch.load(checkpoint_dir / 'phase1_v2.phase.pt', map_location='cpu')
    cfg1 = p1['config']

    cse = ContinuousSemanticEncoder(
        wave_dim=cfg1.get('wave_dim', 432),
        window_size=cfg1.get('window_size', 8),
        stride=cfg1.get('stride', 1),
    )
    cse.load_state_dict(p1['state_dict']['cse'])
    cse.to(device).eval()

    chunker = WaveChunker(
        wave_dim=cfg1.get('wave_dim', 432),
        min_chunk=cfg1.get('min_chunk', 2),
        max_chunk=cfg1.get('max_chunk', 20),
    )
    chunker.load_state_dict(p1['state_dict']['chunker'])
    chunker.to(device).eval()

    wtt = WaveToText(
        wave_dim=cfg1.get('wave_dim', 432),
        hidden_dim=cfg1.get('wtt_hidden_dim', 256),
        max_bytes=cfg1.get('max_bytes', 20),
    )
    wtt.load_state_dict(p1['state_dict']['wtt'])
    wtt.to(device).eval()

    # Phase 2
    p2  = torch.load(checkpoint_dir / 'phase2_v2.phase.pt', map_location='cpu')
    cfg2 = p2['config']

    w2f = WaveToField(
        wave_dim=cfg2.get('wave_dim', 432),
        field_dim=cfg2.get('field_features', 512),
    )
    w2f.load_state_dict(p2['state_dict']['wave_to_field'])
    w2f.to(device).eval()

    f2w = FieldToWave(
        field_dim=cfg2.get('field_features', 512),
        wave_dim=cfg2.get('wave_dim', 432),
    )
    f2w.load_state_dict(p2['state_dict']['field_to_wave'])
    f2w.to(device).eval()

    # Phase 3
    p3  = torch.load(checkpoint_dir / 'phase3_v2.phase.pt', map_location='cpu')
    cfg3 = p3['config']

    generator = WaveGenerator(
        wave_dim=cfg3.get('wave_dim', 432),
        field_features=cfg3.get('field_features', 512),
        gru_hidden=cfg3.get('gru_hidden', 512),
        gru_layers=cfg3.get('gru_layers', 1),
        dropout=0.0,  # no dropout at eval
    )
    generator.load_state_dict(p3['state_dict']['generator'])
    generator.to(device).eval()

    print(f"  ✓ All components loaded from {checkpoint_dir}")
    return cse, chunker, wtt, w2f, f2w, generator


@torch.no_grad()
def eval_text(
    text:      str,
    cse:       ContinuousSemanticEncoder,
    chunker:   WaveChunker,
    wtt:       WaveToText,
    w2f:       WaveToField,
    generator: WaveGenerator,
    device:    str,
) -> Tuple[float, str]:
    """Encode text → get field context → generate waves → decode → measure accuracy."""
    text_bytes = text.encode('utf-8')
    wave       = cse.encode(text)
    mean_wave  = wave.full.mean(dim=0).to(device)
    ctx        = w2f(mean_wave)

    generated_waves, _ = generator.generate(
        field_context=ctx,
        max_waves=len(text_bytes) // 3 + 10,
    )

    decoded_bytes = b''
    for i in range(generated_waves.shape[0]):
        chunk_bytes = wtt.decode(generated_waves[i])
        if chunk_bytes:
            decoded_bytes += bytes(chunk_bytes)

    acc           = byte_accuracy(text_bytes, decoded_bytes)
    decoded_str   = decoded_bytes.decode('utf-8', errors='replace')
    return acc, decoded_str


def main():
    device    = get_device()
    ckpt_dir  = _PROJECT_ROOT / 'checkpoints'

    assert (ckpt_dir / 'phase3_v2.phase.pt').exists(), \
        "Phase 3 checkpoint not found. Run train_generator.py first."

    cse, chunker, wtt, w2f, f2w, generator = load_all_components(ckpt_dir, device)

    results    = PhaseResults(phase=3, component_name="Wave Generator")
    per_text   = {}
    accs       = []
    n_pass     = 0

    print("\n── Test 1: Decode Gate (Generated Output Quality) ──\n")
    for text in TEST_TEXTS:
        acc, decoded = eval_text(text, cse, chunker, wtt, w2f, generator, device)
        accs.append(acc)
        per_text[text[:40]] = acc

        label  = "✓" if acc >= PER_TEXT_THRESH else "✗"
        n_pass += int(acc >= PER_TEXT_THRESH)

        print(f"  {label} [{acc:.1%}]  {text[:40]}")
        print(f"         decoded: {decoded[:60]!r}")

    avg_acc = sum(accs) / len(accs)
    min_acc = min(accs)

    print(f"\n  avg={avg_acc:.1%}  min={min_acc:.1%}  texts_pass={n_pass}/{len(TEST_TEXTS)}")

    results.add_test(
        "Generated decode avg > 50%",
        passed=(avg_acc >= AVG_THRESHOLD),
        score=avg_acc,
        threshold=AVG_THRESHOLD,
    )
    results.add_test(
        "Generated decode min > 20%",
        passed=(min_acc >= MIN_THRESHOLD),
        score=min_acc,
        threshold=MIN_THRESHOLD,
    )
    results.add_test(
        f"At least {N_MUST_PASS}/8 texts score > 30%",
        passed=(n_pass >= N_MUST_PASS),
        score=n_pass / len(TEST_TEXTS),
        threshold=N_MUST_PASS / len(TEST_TEXTS),
    )

    results.save()

    assert avg_acc >= AVG_THRESHOLD, f"FAIL avg={avg_acc:.1%} < {AVG_THRESHOLD:.0%}"
    assert min_acc >= MIN_THRESHOLD, f"FAIL min={min_acc:.1%} < {MIN_THRESHOLD:.0%}"
    assert n_pass  >= N_MUST_PASS,   f"FAIL only {n_pass}/{len(TEST_TEXTS)} texts pass"

    print("\n  ✓ Test 1 PASSED")


if __name__ == '__main__':
    main()
