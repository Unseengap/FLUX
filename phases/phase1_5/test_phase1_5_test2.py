"""
PHASE 1.5 TEST 2: Contradicting pairs produce higher tension than neutral pairs.

Pass criteria:
    - 45/50 contradiction pairs correctly detected (higher tension)
    - Mean tension gap (contra - neutral) > 0.2
    - No false positives: neutral pairs have tension < 0.3 on average
    - Runs in < 30 seconds
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'phases' / 'phase1'))
sys.path.insert(0, str(ROOT / 'phases' / 'phase1_5'))

import torch
from flux_utils import load_checkpoint, PhaseResults
from cse import ContinuousSemanticEncoder
from causal_encoder import CausalWaveChainer

CONTRADICTION_PAIRS = [
    ("the sky is blue",           "the sky is green",              "birds can fly"),
    ("water is liquid",           "water is solid",                "fish swim in water"),
    ("dogs are animals",          "dogs are not animals",          "cats purr loudly"),
    ("the earth is round",        "the earth is flat",             "stars shine at night"),
    ("fire is hot",               "fire is cold",                  "ice melts slowly"),
    ("Paris is in France",        "Paris is in Germany",           "London is a large city"),
    ("humans need oxygen",        "humans do not need oxygen",     "plants grow in sunlight"),
    ("the sun rises in the east", "the sun rises in the west",     "the moon orbits earth"),
    ("cats are mammals",          "cats are reptiles",             "dogs bark loudly"),
    ("winter is cold",            "winter is hot",                 "summer brings sunshine"),
    ("the car moved forward",     "the car moved backward",        "the driver wore a seatbelt"),
    ("the door was open",         "the door was closed",           "the window had curtains"),
    ("the light was on",          "the light was off",             "the room had furniture"),
    ("she was happy",             "she was miserable",             "she wore a blue coat"),
    ("the ball is red",           "the ball is blue",              "the field is green"),
    ("he ran fast",               "he ran slowly",                 "she carried a bag"),
    ("the water was warm",        "the water was freezing",        "the towel was dry"),
    ("the music was loud",        "the music was silent",          "the crowd was large"),
    ("the answer is yes",         "the answer is no",              "the question was clear"),
    ("the road was empty",        "the road was crowded",          "the weather was fine"),
    ("the bird flew high",        "the bird stayed on the ground", "the tree was tall"),
    ("the book was fiction",      "the book was nonfiction",       "the cover was blue"),
    ("the child was awake",       "the child was asleep",          "the toy was broken"),
    ("the engine started",        "the engine would not start",    "the fuel was full"),
    ("the team won",              "the team lost",                 "the crowd cheered"),
    ("the glass was full",        "the glass was empty",           "the table was wooden"),
    ("he told the truth",         "he told a lie",                 "she listened carefully"),
    ("the plant was alive",       "the plant was dead",            "the soil was moist"),
    ("the signal was strong",     "the signal was weak",           "the antenna was tall"),
    ("the price went up",         "the price went down",           "the market was busy"),
    ("the flight was on time",    "the flight was delayed",        "the airport was crowded"),
    ("the test was easy",         "the test was very difficult",   "the room was quiet"),
    ("she agreed with him",       "she strongly disagreed",        "they sat together"),
    ("the window was clean",      "the window was dirty",          "the view was pleasant"),
    ("the soup was hot",          "the soup was cold",             "the bread was fresh"),
    ("the news was good",         "the news was terrible",         "the reporter spoke clearly"),
    ("he was awake all night",    "he slept soundly all night",    "the bed was comfortable"),
    ("the lock was secure",       "the lock was broken",           "the key was silver"),
    ("the path was clear",        "the path was completely blocked","the trees were tall"),
    ("the battery was charged",   "the battery was dead",          "the screen was bright"),
    ("the connection was fast",   "the connection was very slow",  "the server was running"),
    ("the river was rising",      "the river was falling",         "the bridge was old"),
    ("it was raining heavily",    "the sky was perfectly clear",   "people carried bags"),
    ("the meeting started early", "the meeting started very late", "the chairs were arranged"),
    ("the food was fresh",        "the food was stale",            "the menu had options"),
    ("the road was wet",          "the road was completely dry",   "the car drove on"),
    ("the stars were visible",    "the stars were hidden by clouds","the moon was full"),
    ("the engine was running",    "the engine had stopped",        "the driver was seated"),
    ("the crowd was silent",      "the crowd was very noisy",      "the stage was lit"),
    ("the building was new",      "the building was ancient",      "the lobby was large"),
]


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start  = time.time()

    print("=" * 60)
    print("FLUX Phase 1.5 Test 2: Contradiction Detection")
    print("=" * 60)

    # Load models
    print("\n  Loading Phase 1.5 checkpoint...")
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
    print(f"  ✓ Models loaded")

    detected     = 0
    tension_gaps = []
    neutral_tensions = []

    print(f"\n  {'#':>3}  {'Contra':>8}  {'Neutral':>8}  {'Gap':>8}  {'Pass'}")
    print(f"  {'-'*50}")

    with torch.no_grad():
        for i, (stmt, contra, neutral) in enumerate(CONTRADICTION_PAIRS):
            cw_contra  = cwc.encode(cse, stmt + ' ' + contra)
            cw_neutral = cwc.encode(cse, stmt + ' ' + neutral)

            t_contra  = cw_contra.tension_score()
            t_neutral = cw_neutral.tension_score()
            gap       = t_contra - t_neutral
            passed    = gap > 0

            tension_gaps.append(gap)
            neutral_tensions.append(t_neutral)
            if passed:
                detected += 1

            status = "✓" if passed else "✗"
            print(f"  {i+1:>3}  {t_contra:>8.4f}  {t_neutral:>8.4f}  {gap:>+8.4f}  {status}")

    elapsed          = time.time() - start
    mean_gap         = sum(tension_gaps) / max(len(tension_gaps), 1)
    mean_neutral     = sum(neutral_tensions) / max(len(neutral_tensions), 1)
    detected_pass    = detected >= 45
    gap_pass         = mean_gap > 0.2
    fp_pass          = mean_neutral < 0.3
    time_pass        = elapsed < 30

    print(f"\n  ── Results ──")
    print(f"  Detected:       {detected}/{len(CONTRADICTION_PAIRS)}")
    print(f"  Mean gap:       {mean_gap:.4f}  (threshold > 0.2)")
    print(f"  Mean neutral:   {mean_neutral:.4f}  (threshold < 0.3)")
    print(f"  Time elapsed:   {elapsed:.1f}s  (threshold < 30s)")
    print()

    all_pass = detected_pass and gap_pass and fp_pass and time_pass
    print(f"  {'✓' if detected_pass else '✗'} Detected: {detected}/50 (threshold: ≥ 45/50)")
    print(f"  {'✓' if gap_pass else '✗'} Mean tension gap: {mean_gap:.4f} (threshold: > 0.2)")
    print(f"  {'✓' if fp_pass else '✗'} False positive rate: neutral avg={mean_neutral:.4f} (threshold: < 0.3)")
    print(f"  {'✓' if time_pass else '✗'} Runtime: {elapsed:.1f}s (threshold: < 30s)")

    try:
        results = PhaseResults(phase=1.5, component_name="CausalWaveChainer")
    except TypeError:
        results = PhaseResults(phase=1.5)
    results.add("Contradiction detected", detected, ">= 45/50", detected_pass)
    results.add("Mean tension gap", mean_gap, "> 0.2", gap_pass)
    results.add("Neutral tension (FP)", mean_neutral, "< 0.3", fp_pass)
    results.add("Runtime", elapsed, "< 30s", time_pass)
    try:
        results.save()
    except Exception as e:
        print(f"  ℹ Results save: {e}")

    print(f"\n{'='*60}")
    print(f"All tests passed: {all_pass}")
    print(f"{'='*60}")
    if all_pass:
        print("  ✓ CONTRADICTION DETECTION TEST PASSED")
    else:
        print("  ✗ CONTRADICTION DETECTION TEST FAILED")

    return all_pass


if __name__ == '__main__':
    main()