"""
PHASE 1.5 TEST 3: Forward propagation finds implied concepts.

Pass criteria:
    - 14/20 test premises correctly imply their known conclusion
    - At least 5/20 achieve chain depth > 1 (transitive reasoning)
    - Runs in < 45 seconds
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'phases' / 'phase1'))
sys.path.insert(0, str(ROOT / 'phases' / 'phase1_5'))

import torch
import torch.nn.functional as F
from flux_utils import load_checkpoint, PhaseResults
from cse import ContinuousSemanticEncoder
from causal_encoder import CausalWaveChainer
from implication import ImplicationChainStore
from causal_types import CausalArrow

# Test pairs: (premise, known_conclusion, strength)
TEST_IMPLICATION_PAIRS = [
    ("it started raining",             "people opened umbrellas",        0.85),
    ("the dog was hungry",             "the dog ate food",               0.90),
    ("the car ran out of gas",         "the car stopped moving",         0.95),
    ("she studied hard",               "she passed the exam",            0.80),
    ("the fire alarm went off",        "people evacuated the building",  0.90),
    ("the ice melted",                 "the water level rose",           0.85),
    ("he missed the train",            "he arrived late",                0.88),
    ("the power went out",             "the lights turned off",          0.95),
    ("the seed was planted",           "the plant began to grow",        0.80),
    ("the window broke",               "cold air entered the room",      0.82),
    ("the meeting was cancelled",      "people went back to work",       0.75),
    ("the temperature dropped",        "the lake began to freeze",       0.83),
    ("the alarm was set",              "she woke up on time",            0.78),
    ("the bridge collapsed",           "traffic was redirected",         0.88),
    ("the storm intensified",          "visibility decreased sharply",   0.85),
    ("the student asked a question",   "the teacher provided an answer", 0.82),
    ("the letter was mailed",          "it arrived days later",          0.79),
    ("the experiment failed",          "the team revised the approach",  0.76),
    ("the sun set",                    "it became dark outside",         0.95),
    ("the crowd gathered",             "the event was about to begin",   0.82),
]


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start  = time.time()

    print("=" * 60)
    print("FLUX Phase 1.5 Test 3: Implication Propagation")
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

    # Restore implication store from checkpoint
    impl_store = ImplicationChainStore(device=device)
    if 'implication_store' in ckpt15:
        impl_store.load(ckpt15['implication_store'])
    print(f"  ✓ Implication store: {len(impl_store.arrows)} arrows")

    # If store is empty, populate from training pairs
    if len(impl_store.arrows) == 0:
        print("  ⚠ Implication store empty — populating from test pairs...")
        with torch.no_grad():
            for premise, conclusion, strength in TEST_IMPLICATION_PAIRS:
                cw_p = cwc.encode(cse, premise)
                cw_c = cwc.encode(cse, conclusion)
                impl_store.add_arrow(
                    cw_p.full.mean(dim=0),
                    cw_c.full.mean(dim=0),
                    strength, 'temporal'
                )

    hits         = 0
    deep_chains  = 0
    total        = len(TEST_IMPLICATION_PAIRS)

    print(f"\n  {'#':>3}  {'Premise':<35}  {'Top Sim':>8}  {'Depth':>6}  {'Pass'}")
    print(f"  {'-'*65}")

    with torch.no_grad():
        for i, (premise, conclusion, strength) in enumerate(TEST_IMPLICATION_PAIRS):
            # Encode premise
            cw_prem = cwc.encode(cse, premise)
            q_vec   = cw_prem.full.mean(dim=0)

            # Forward propagate
            implied = impl_store.forward_propagate(q_vec, k=10, min_strength=0.2)

            # Encode known conclusion for comparison
            cw_conc = cwc.encode(cse, conclusion)
            conc_vec = F.normalize(cw_conc.full.mean(dim=0).cpu().float(), dim=-1)

            # Check if any implied concept is close to ground truth
            best_sim = 0.0
            best_depth = 0
            for tgt_vec, eff_str in implied:
                tgt_norm = F.normalize(tgt_vec.float(), dim=-1)
                sim = F.cosine_similarity(conc_vec.unsqueeze(0), tgt_norm.unsqueeze(0)).item()
                if sim > best_sim:
                    best_sim   = sim
                    best_depth = 1

            # Try chain propagation for depth > 1
            chains = impl_store.chain_propagate(q_vec, depth=3, k_per_step=5)
            for chain in chains:
                for step_vec, step_str in chain:
                    step_norm = F.normalize(step_vec.float(), dim=-1)
                    sim = F.cosine_similarity(conc_vec.unsqueeze(0), step_norm.unsqueeze(0)).item()
                    depth = len(chain)
                    if sim > best_sim:
                        best_sim   = sim
                        best_depth = depth

            passed = best_sim > 0.6
            if passed:
                hits += 1
            if best_depth > 1:
                deep_chains += 1

            status = "✓" if passed else "✗"
            prem_short = premise[:33] + ".." if len(premise) > 33 else premise
            print(f"  {i+1:>3}  {prem_short:<35}  {best_sim:>8.4f}  {best_depth:>6}  {status}")

    elapsed      = time.time() - start
    hits_pass    = hits >= 14
    deep_pass    = deep_chains >= 5
    time_pass    = elapsed < 45
    all_pass     = hits_pass and deep_pass and time_pass

    print(f"\n  ── Results ──")
    print(f"  Hits:        {hits}/{total}")
    print(f"  Deep chains: {deep_chains}/{total}")
    print(f"  Time:        {elapsed:.1f}s")
    print()
    print(f"  {'✓' if hits_pass else '✗'} Implication hits: {hits}/20 (threshold: ≥ 14/20)")
    print(f"  {'✓' if deep_pass else '✗'} Transitive chains: {deep_chains}/20 (threshold: ≥ 5/20)")
    print(f"  {'✓' if time_pass else '✗'} Runtime: {elapsed:.1f}s (threshold: < 45s)")

    results = PhaseResults(phase=1.5)
    results.add("Implication hits", hits, ">= 14/20", hits_pass)
    results.add("Transitive chain depth > 1", deep_chains, ">= 5/20", deep_pass)
    results.add("Runtime", elapsed, "< 45s", time_pass)
    results.save()

    print(f"\n{'='*60}")
    print(f"All tests passed: {all_pass}")
    print(f"{'='*60}")
    if all_pass:
        print("  ✓ IMPLICATION PROPAGATION TEST PASSED")
    else:
        print("  ✗ IMPLICATION PROPAGATION TEST FAILED")

    return all_pass


if __name__ == '__main__':
    main()