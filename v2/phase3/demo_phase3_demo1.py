"""
demo_phase3_demo1.py — Prompt → Wave Generation → Text

Shows the full Phase 3 pipeline end-to-end:
    1. User gives a text prompt
    2. CSE encodes it to waves
    3. WaveToField projects the mean wave into field space (context)
    4. WaveGenerator predicts the next N waves from context
    5. WaveToText decodes each wave back to bytes
    6. Output text is printed alongside confidence scores

Run: python demo_phase3_demo1.py
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path

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
from wave_chunker   import WaveChunker
from wave_to_text   import WaveToText
from wave_to_field  import WaveToField
from field_to_wave  import FieldToWave
from wave_generator import WaveGenerator
from flux_utils     import get_device

DEMO_PROMPTS = [
    "The capital of France is Paris",
    "Water freezes at zero degrees Celsius",
    "def hello(): return 'world'",
    "The future of artificial intelligence",
    "café naïve résumé",
    "The cat sat on the mat",
    "Machine learning requires large datasets",
    "∫₀^∞ e^(-x²) dx = √π/2",
]


def load_components(ckpt_dir: Path, device: str):
    p1   = torch.load(ckpt_dir / 'phase1_v2.phase.pt', map_location='cpu')
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

    p2   = torch.load(ckpt_dir / 'phase2_v2.phase.pt', map_location='cpu')
    cfg2 = p2['config']

    w2f = WaveToField(
        wave_dim=cfg2.get('wave_dim', 432),
        field_dim=cfg2.get('field_features', 512),
    )
    w2f.load_state_dict(p2['state_dict']['bridge_wtf'])
    w2f.to(device).eval()

    f2w = FieldToWave(
        field_dim=cfg2.get('field_features', 512),
        wave_dim=cfg2.get('wave_dim', 432),
    )
    f2w.load_state_dict(p2['state_dict']['bridge_ftw'])
    f2w.to(device).eval()

    p3   = torch.load(ckpt_dir / 'phase3_v2.phase.pt', map_location='cpu')
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

    return cse, chunker, wtt, w2f, f2w, generator


@torch.no_grad()
def run_pipeline(
    prompt:    str,
    cse:       ContinuousSemanticEncoder,
    chunker:   WaveChunker,
    wtt:       WaveToText,
    w2f:       WaveToField,
    f2w:       FieldToWave,
    generator: WaveGenerator,
    device:    str,
    max_waves: int = 20,
):
    """Full pipeline: text → waves → field context → generate → bridge → decode."""
    # Encode prompt
    wave       = cse.encode(prompt)
    mean_wave  = wave.full.mean(dim=0).to(device)
    ctx        = w2f(mean_wave)

    # Generate
    gen_waves, confs = generator.generate(
        field_context=ctx,
        max_waves=max_waves,
    )

    # Snap onto Phase 2 CSE manifold before WTT decoding
    bridged_waves = f2w(w2f(gen_waves))  # [N, 432]

    # Decode each bridged wave
    decoded_chunks = []
    for i in range(bridged_waves.shape[0]):
        chunk_bytes = wtt.decode(bridged_waves[i])
        text_chunk  = bytes(chunk_bytes).decode('utf-8', errors='replace') if chunk_bytes else ''
        decoded_chunks.append((text_chunk, confs[i]))

    # Full decoded output
    full_output = ''.join(t for t, _ in decoded_chunks)

    # Wave diversity metric
    if gen_waves.shape[0] > 1:
        norms     = F.normalize(gen_waves, dim=-1)
        sim_mat   = norms @ norms.T
        n         = sim_mat.shape[0]
        off_diag  = sim_mat[~torch.eye(n, dtype=torch.bool, device=device)]
        avg_sim   = off_diag.mean().item()
    else:
        avg_sim = 1.0

    return full_output, decoded_chunks, avg_sim, gen_waves.shape[0]


def print_bar(value: float, width: int = 20, char: str = '█') -> str:
    """Simple ASCII progress bar."""
    filled = int(value * width)
    return char * filled + '░' * (width - filled)


def main():
    device   = get_device()
    ckpt_dir = _PROJECT_ROOT / 'checkpoints'

    assert (ckpt_dir / 'phase3_v2.phase.pt').exists(), \
        "Phase 3 checkpoint not found. Run train_generator.py first."

    print("\n")
    print("=" * 65)
    print("  FLUX v2 Phase 3 — Demo 1: Prompt → Wave Generation → Text")
    print("=" * 65)
    print(f"  Device: {device}")
    print()

    cse, chunker, wtt, w2f, f2w, generator = load_components(ckpt_dir, device)

    for prompt in DEMO_PROMPTS:
        output, chunks, avg_sim, n_waves = run_pipeline(
            prompt, cse, chunker, wtt, w2f, f2w, generator, device
        )

        print(f"  Prompt  : {prompt}")
        print(f"  Output  : {output[:80]!r}")
        print(f"  Waves   : {n_waves}  avg_inter_wave_cosine={avg_sim:.3f}")

        # Per-wave confidence bar chart (first 8 waves)
        print("  Confidence per wave:")
        for i, (chunk_text, conf) in enumerate(chunks[:8]):
            bar = print_bar(conf, width=12)
            print(f"    [{i:2d}] {bar} {conf:.2f}  {chunk_text!r}")

        print()

    # ── Interactive mode ──
    print("─" * 65)
    print("  Interactive mode — type a prompt and see it generated.")
    print("  (Press Ctrl+C to exit)\n")
    try:
        while True:
            try:
                user_prompt = input("  Prompt > ").strip()
            except EOFError:
                break
            if not user_prompt:
                continue

            output, chunks, avg_sim, n_waves = run_pipeline(
                user_prompt, cse, chunker, wtt, w2f, f2w, generator, device
            )
            print(f"  Output  : {output[:120]!r}")
            print(f"  Waves: {n_waves}  avg_cosine={avg_sim:.3f}\n")
    except KeyboardInterrupt:
        print("\n  Done.")


if __name__ == '__main__':
    main()
