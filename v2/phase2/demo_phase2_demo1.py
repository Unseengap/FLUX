"""
demo_phase2_demo1.py — Wave → Field → Wave → TEXT Round-Trip

Shows the complete Phase 2 v2 pipeline with the field bridge:
  text → CSE → WaveChunker → WaveToField → FieldToWave → WaveToText → text

Demonstrates that the field projection preserves decodability,
proving that the bridge is not a random bottleneck (the original bug).

Run: python demo_phase2_demo1.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F
from decode_gate import byte_accuracy

DEMO_TEXTS = [
    "The future of artificial intelligence is wave-first.",
    "café naïve résumé",
    "def encode(text): return waves",
    "∫₀^∞ e^(-x²) dx = √π/2",
    "Physics governs the behavior of waves in fields.",
    "Water is a polar molecule — H₂O",
]


def print_bar(width: int = 65, char: str = '─'):
    print(char * width)


@torch.no_grad()
def run_demo():
    print()
    print_bar(65, '═')
    print("  FLUX v2 Phase 2 — Wave ↔ Field ↔ Wave ↔ Text Round-Trip Demo")
    print_bar(65, '═')

    # ── Load checkpoints ───────────────────────────────────────────
    try:
        from train_codec import load_phase1_checkpoint
        from train_field import load_phase2_checkpoint
        codec         = load_phase1_checkpoint(device='cpu')
        _, bridge, catalog = load_phase2_checkpoint(device='cpu')
        cse     = codec.cse
        chunker = codec.chunker
        wtt     = codec.wtt
        wtf     = bridge.wtf
        ftw     = bridge.ftw
        print(f"  ✓ Phase 1 + Phase 2 v2 loaded  (attractors: {catalog.count()})")
    except FileNotFoundError:
        print("  ⚠ Checkpoints not found — using UNTRAINED model")
        print("  Train: python train_codec.py && python train_field.py\n")
        from cse import ContinuousSemanticEncoder
        from wave_chunker import WaveChunker
        from wave_to_text import WaveToText
        from wave_to_field import WaveToField
        from field_to_wave import FieldToWave
        cse, chunker, wtt = ContinuousSemanticEncoder(), WaveChunker(), WaveToText()
        wtf, ftw = WaveToField(), FieldToWave()
        catalog = None

    for i, text in enumerate(DEMO_TEXTS, 1):
        print()
        print_bar()
        print(f"  Example {i}: '{text}'")
        print_bar()

        wave = cse.encode(text)
        chunks, spans = chunker(wave.full)

        # Through field bridge
        field_vecs    = wtf(chunks)
        reconstructed = ftw(field_vecs)

        # Direct decode (Phase 1 path — baseline)
        decoded_direct = wtt.decode_to_text(chunks, temperature=0.3)
        acc_direct     = byte_accuracy(text, decoded_direct)

        # Field-bridge decode (Phase 2 path)
        decoded_field = wtt.decode_to_text(reconstructed, temperature=0.3)
        acc_field     = byte_accuracy(text, decoded_field)

        # Round-trip cosine
        cos = F.cosine_similarity(chunks, reconstructed, dim=-1).mean()

        print(f"  Bytes          : {len(text.encode('utf-8'))}")
        print(f"  Chunks         : {chunks.shape[0]}")
        print(f"  Round-trip cos : {cos:.3f}  (threshold: ≥ 0.85)")
        print(f"  Direct decode  : '{decoded_direct[:50]}'  [{acc_direct:.1%}]")
        print(f"  Field decode   : '{decoded_field[:50]}'  [{acc_field:.1%}]")

        delta = acc_field - acc_direct
        if abs(delta) < 0.05:
            quality = "✓ Field preserves accuracy"
        elif delta > 0:
            quality = "✓ Field actually improved accuracy"
        else:
            quality = f"⚠ Field degraded accuracy by {abs(delta):.1%}"

        print(f"  Status         : {quality}")

    # ── Summary ────────────────────────────────────────────────────
    print()
    print_bar(65, '═')
    print("  Pipeline Summary")
    print_bar()

    direct_accs = []
    field_accs  = []
    cos_sims    = []

    for text in DEMO_TEXTS:
        wave = cse.encode(text)
        chunks, _ = chunker(wave.full)
        fv    = wtf(chunks)
        recon = ftw(fv)

        direct_accs.append(byte_accuracy(text, wtt.decode_to_text(chunks, temperature=0.3)))
        field_accs.append(byte_accuracy(text, wtt.decode_to_text(recon, temperature=0.3)))
        cos_sims.append(F.cosine_similarity(chunks, recon, dim=-1).mean().item())

    print(f"  Phase 1 direct decode avg  : {sum(direct_accs)/len(direct_accs):.1%}")
    print(f"  Phase 2 field decode avg   : {sum(field_accs)/len(field_accs):.1%}")
    print(f"  Round-trip cosine avg      : {sum(cos_sims)/len(cos_sims):.3f}")
    print()

    gate_passed = sum(field_accs) / len(field_accs) >= 0.90
    if gate_passed:
        print("  ✓ Phase 2 decode gate: PASSED")
    else:
        print("  ⚠ Phase 2 decode gate: NOT YET (train more steps)")

    print_bar(65, '═')


if __name__ == '__main__':
    run_demo()
