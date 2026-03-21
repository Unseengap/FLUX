import sys, torch
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'phase1'))
sys.path.append(str(Path(__file__).parent.parent / 'phase2'))
from flux_utils import load_checkpoint, get_device
from cse import ContinuousSemanticEncoder
from field import ResonanceField
from gravity import GravitationalRelevance
from sanity_decoder import SanityDecoder, pipeline_check

DEMO_TEXTS = [
    "the cat sat on the mat",
    "hello world",
    "artificial intelligence is transforming the world",
    "the quick brown fox jumps over the lazy dog",
    "once upon a time in a land far away",
    "stars are giant balls of plasma fusing hydrogen",
    "the ocean covers seventy percent of Earth",
    "knowledge is power",
]

def main():
    device = get_device()

    # ── Load Phase 1 (CSE) ──
    c1 = load_checkpoint(1)
    cse = ContinuousSemanticEncoder(**c1.get('config', {})).to(device)
    cse.load_state_dict(c1['state_dict'])
    cse.eval()

    # ── Load Phase 2 (Field) ──
    c2 = load_checkpoint(2)
    field = ResonanceField(**c2.get('config', {}).get('field', {})).to(device)
    field.load_state_dict(c2['state_dict'])
    field.eval()

    # ── Load Phase 3 (GR + trained Decoder) from checkpoint ──
    c3 = load_checkpoint(3)
    gr = GravitationalRelevance.load_state(c3['phase3_gr_state'], device=device)
    gr = gr.to(device).eval()

    dec = SanityDecoder(feature_dim=512, device=device).to(device)
    dec.load_state_dict(c3['phase3_decoder_state'])
    dec.eval()

    # ── ★ FIRST TEXT OUTPUT FROM FLUX ★ ──
    print("\n" + "=" * 65)
    print("  ★  FLUX FIRST TEXT OUTPUT — Full Pipeline Demo  ★")
    print("  CSE (Phase 1) → Field (Phase 2) → GR (Phase 3) → Decoder → TEXT")
    print("  [wave identity + field context → text reconstruction]")
    print("=" * 65)

    for text in DEMO_TEXTS:
        result = pipeline_check(cse, field, gr, dec, text, verbose=True)
        print(f"  LCS ratio={result['lcs_ratio']:.2%}  "
              f"trigram={result['trigram_overlap']:.2%}  "
              f"char_acc={result['char_accuracy']:.2%}")

    # ── Uniqueness check: different inputs should produce different outputs ──
    print("\n" + "-" * 65)
    print("  Input-specificity check (v2 wave-aware decoder):")
    print("-" * 65)
    outputs = set()
    for text in DEMO_TEXTS:
        result = pipeline_check(cse, field, gr, dec, text, verbose=False)
        outputs.add(result['reconstructed'][:30])
    unique_pct = len(outputs) / len(DEMO_TEXTS) * 100
    print(f"  Unique outputs: {len(outputs)}/{len(DEMO_TEXTS)} ({unique_pct:.0f}%)")
    if unique_pct >= 75:
        print(f"  ✓ Decoder is input-specific — different inputs → different outputs")
    else:
        print(f"  ✗ Decoder is still collapsing inputs — wave signal may not be working")

    print("\n" + "=" * 65)
    print("  ★  MILESTONE: First text output from FLUX demonstrated  ★")
    print("=" * 65)

if __name__ == "__main__": main()
