"""
FLUX Phase 1.5 Test 1: Order Sensitivity (Tension-Based)

The causal_coherence() metric hits a ceiling — both ordered and shuffled
sentences score 0.9998-0.9999. This test uses TENSION instead.

Shuffled sequences are incoherent. Incoherence IS internal contradiction.
The TensionDetector should give shuffled sequences HIGHER tension.
This is consistent with 20/20 contradiction detection in training.

Pass criteria:
    - 35/50 shuffled sequences have higher tension than ordered
    - Mean tension gap (shuffled - ordered) > 0.005
    - Runs in < 60 seconds
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'phases' / 'phase1'))
sys.path.insert(0, str(ROOT / 'phases' / 'phase1_5'))

import torch
from flux_utils import load_checkpoint
from cse import ContinuousSemanticEncoder
from causal_encoder import CausalWaveChainer

TEST_SENTENCES = [
    "the dog chased the cat across the yard into the garden",
    "scientists discovered a new species in the deep ocean",
    "the economy grew significantly during the second quarter of the fiscal year",
    "machine learning algorithms process data to discover hidden patterns",
    "the ancient civilization built monuments that still stand today",
    "photosynthesis converts carbon dioxide and water into glucose using sunlight",
    "democracy requires the active participation of informed citizens to function",
    "neural networks consist of layers of interconnected artificial neurons",
    "the orchestra performed the symphony to an enthusiastic standing ovation",
    "climate change poses significant challenges to agricultural production worldwide",
    "the children played happily in the park until the sun went down",
    "researchers published their groundbreaking findings in a major scientific journal",
    "the train departed from the station exactly on schedule this morning",
    "she carefully read through all the instructions before starting the assembly",
    "the storm moved quickly across the mountains and into the valley below",
    "the chef prepared the meal with fresh ingredients from the local market",
    "students gathered in the library to study for their upcoming examinations",
    "the spacecraft traveled millions of miles through the void of outer space",
    "the doctor carefully examined the patient and recommended further diagnostic tests",
    "the river flowed steadily through the canyon carving deep channels in the rock",
    "the musician practiced the difficult passage repeatedly until it became effortless",
    "the election results were announced after all the ballots had been carefully counted",
    "the engineers designed the bridge to withstand extremely strong winds and heavy loads",
    "the teacher explained the complex mathematical concept using simple clear examples",
    "the explorers navigated through the dense forest using only a compass and a map",
    "the company released its quarterly earnings report showing strong revenue growth",
    "the athlete trained every morning for months to prepare for the competition",
    "the library contains thousands of books on every subject imaginable",
    "the volcano erupted sending clouds of ash high into the atmosphere",
    "the detective examined the evidence carefully before drawing any conclusions",
    "the farmer planted seeds in the spring and harvested crops in the autumn",
    "the satellite transmitted data continuously from its orbit around the earth",
    "the journalist interviewed multiple witnesses before writing the final article",
    "the surgeon performed the delicate operation with exceptional skill and precision",
    "the pilot guided the aircraft safely through the turbulent storm clouds",
    "the programmer wrote efficient code to solve the complex computational problem",
    "the diplomat negotiated a peaceful agreement between the two competing nations",
    "the historian researched ancient documents to uncover long forgotten truths",
    "the architect designed the building to maximize natural light and ventilation",
    "the nurse monitored the patient vital signs throughout the long night",
    "the fire spread rapidly through the dry forest driven by strong gusting winds",
    "the submarine descended slowly into the dark depths of the ocean",
    "the translator worked carefully to preserve the meaning of the original text",
    "the composer wrote the symphony over many years of dedicated creative work",
    "the geologist studied rock formations to understand the history of the earth",
    "the biologist observed the behavior of the colony over several months",
    "the economist analyzed market trends to forecast future economic conditions",
    "the photographer captured the fleeting moment with perfect timing and skill",
    "the sailor navigated the ship through stormy waters using stars for guidance",
    "the software update fixed several critical bugs and improved overall performance",
]


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start  = time.time()

    print("=" * 60)
    print("FLUX Phase 1.5 Test 1: Order Sensitivity (Tension-Based)")
    print("=" * 60)
    print("\n  Signal: shuffled text should have HIGHER tension than ordered")
    print("  (incoherence registers as internal contradiction)\n")

    ckpt1 = load_checkpoint(1)
    cse   = ContinuousSemanticEncoder(**ckpt1['config'])
    cse.load_state_dict(ckpt1['state_dict'])
    cse   = cse.to(device).eval()
    for p in cse.parameters():
        p.requires_grad = False

    ckpt15 = load_checkpoint(1.5)
    cwc    = CausalWaveChainer(**ckpt15['config'], device=device).to(device)
    cwc.load_state_dict(ckpt15['state_dict'])
    cwc.eval()
    print(f"  ✓ Models loaded\n")

    wins         = 0
    total        = 0
    tension_gaps = []

    print(f"  {'#':>3}  {'Ordered':>10}  {'Shuffled':>10}  {'Gap':>10}  {'Pass'}")
    print(f"  {'-'*55}")

    with torch.no_grad():
        for i, sentence in enumerate(TEST_SENTENCES):
            words = sentence.split()
            if len(words) < 4:
                continue

            cw_orig = cwc.encode(cse, sentence)
            t_orig  = cw_orig.tension_score()

            shuf_tensions = []
            for _ in range(3):
                idx      = torch.randperm(len(words)).tolist()
                shuffled = ' '.join([words[j] for j in idx])
                cw_shuf  = cwc.encode(cse, shuffled)
                shuf_tensions.append(cw_shuf.tension_score())

            mean_shuf = sum(shuf_tensions) / len(shuf_tensions)
            gap       = mean_shuf - t_orig
            passed    = gap > 0

            tension_gaps.append(gap)
            if passed:
                wins += 1
            total += 1

            status = "✓" if passed else "✗"
            print(f"  {total:>3}  {t_orig:>10.4f}  {mean_shuf:>10.4f}  {gap:>+10.4f}  {status}")

    elapsed  = time.time() - start
    mean_gap = sum(tension_gaps) / max(len(tension_gaps), 1)

    wins_pass = wins >= 35
    gap_pass  = mean_gap > 0.005
    time_pass = elapsed < 60
    all_pass  = wins_pass and gap_pass and time_pass

    print(f"\n  ── Results ──")
    print(f"  Tension wins:     {wins}/{total}")
    print(f"  Mean tension gap: {mean_gap:.6f}  (threshold > 0.005)")
    print(f"  Time elapsed:     {elapsed:.1f}s")
    print()
    print(f"  {'✓' if wins_pass else '✗'} Tension wins: {wins}/{total} (threshold: >= 35/50)")
    print(f"  {'✓' if gap_pass else '✗'} Mean tension gap: {mean_gap:.6f} (threshold: > 0.005)")
    print(f"  {'✓' if time_pass else '✗'} Runtime: {elapsed:.1f}s (threshold: < 60s)")

    # Write results manually
    results_dir  = ROOT / 'phases' / 'phase1_5'
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / 'RESULTS_PHASE_1_5.md'
    result_text  = (
        f"\n## Test 1: Order Sensitivity\n"
        f"| Metric | Score | Threshold | Pass? |\n"
        f"|--------|-------|-----------|-------|\n"
        f"| Tension wins | {wins}/{total} | >= 35/50 | {'PASS' if wins_pass else 'FAIL'} |\n"
        f"| Mean tension gap | {mean_gap:.6f} | > 0.005 | {'PASS' if gap_pass else 'FAIL'} |\n"
        f"| Runtime | {elapsed:.1f}s | < 60s | {'PASS' if time_pass else 'FAIL'} |\n"
    )
    try:
        existing = results_path.read_text() if results_path.exists() else ""
        results_path.write_text(existing + result_text)
        print(f"\n  Results saved to: {results_path}")
    except Exception as e:
        print(f"  ℹ Results save skipped: {e}")

    print(f"\n{'='*60}")
    print(f"All tests passed: {all_pass}")
    print(f"Ready for Phase 2: {all_pass}")
    print(f"{'='*60}")
    if all_pass:
        print("  ✓ ORDER SENSITIVITY TEST PASSED")
    else:
        print("  ✗ ORDER SENSITIVITY TEST FAILED")

    return all_pass


if __name__ == '__main__':
    main()