import sys, torch
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'phase1'))
sys.path.append(str(Path(__file__).parent.parent / 'phase2'))
from flux_utils import load_checkpoint, get_device, PhaseResults
from cse import ContinuousSemanticEncoder
from field import ResonanceField
from gravity import GravitationalRelevance
from sanity_decoder import SanityDecoder, pipeline_check

TEST_SENTENCES = [
    "the cat sat on the mat",
    "hello world",
    "artificial intelligence",
    "the quick brown fox",
    "once upon a time",
    "machine learning algorithms",
    "the ocean covers earth",
    "quantum mechanics",
    "stars are giant plasma balls",
    "democracy requires participation",
    "the sun rises in the east",
    "neural networks learn from data",
    "gravity pulls objects together",
    "the sky is blue today",
    "water freezes at zero degrees",
    "music soothes the soul",
    "birds fly south in winter",
    "computers process binary data",
    "the earth orbits the sun",
    "knowledge is power",
]

def main():
    device = get_device()
    results = PhaseResults(phase=3, component_name="End-to-End Text Reconstruction")

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

    # ── Run pipeline on 20 sentences ──
    recognizable_count = 0
    print(f"\n{'Input':<35} {'Reconstructed':<35} {'Recognize':>9}")
    print("-" * 82)
    for text in TEST_SENTENCES:
        result = pipeline_check(cse, field, gr, dec, text, verbose=False)
        icon = "✓" if result['recognizable'] else "✗"
        recon = result['reconstructed'][:32]
        print(f"  {icon} {text:<33} {recon:<33} {result['char_overlap']:.2f}")
        if result['recognizable']:
            recognizable_count += 1

    passed = recognizable_count >= 15
    print(f"\nResult: {recognizable_count}/{len(TEST_SENTENCES)} recognizable (need ≥15)")
    print(f"{'✓ PASS' if passed else '✗ FAIL'}")

    results.add_test(
        "End-to-End Text Reconstruction",
        passed=passed,
        score=recognizable_count,
        threshold=15,
    )
    results.save()

if __name__ == "__main__": main()
