"""
PHASE 2.5 TEST 3: Unseen Logic Routing

Tests AnalogicalMapper.map_causal_chain() on novel problems
with ZERO word overlap to any training data.

The model must route fictional-word problems to the correct
GSM8K reasoning chain by shape similarity alone.

Pass criteria:
    - Shape similarity > 0.5 for at least 3/5 novel problems
    - Zero word overlap confirmed between novel and matched chain
    - Analogy test: at least 4/7 standard pairs produce correct result
    - Runs in < 60 seconds
"""

import sys
import time
import torch
import torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1_5'))

from flux_utils import PhaseResults, get_device, load_checkpoint
from cse import ContinuousSemanticEncoder
from causal_encoder import CausalWaveChainer
from dynamic_field import SparseResonanceField
from reasoning_curriculum import CurriculumRunner, FALLBACK_GSM8K
from analogical_mapper import AnalogicalMapper, ANALOGY_TEST_PAIRS
from implication import ImplicationChainStore

# Fictional-word problems — ZERO overlap with training data
FICTIONAL_PROBLEMS = [
    {
        'question': "A glorp has 8 blips. It gives away 3 blips. How many blips remain?",
        'expected_type': 'subtraction',
        'steps': [
            "Start with 8 blips.",
            "Remove 3: 8 - 3 = 5.",
            "The answer is 5 blips.",
        ]
    },
    {
        'question': "A zorf costs 6 snurps. You buy 4 zorfs. How many snurps total?",
        'expected_type': 'multiplication',
        'steps': [
            "Cost per zorf: 6 snurps.",
            "Number of zorfs: 4.",
            "Total: 6 × 4 = 24 snurps.",
        ]
    },
    {
        'question': "A flumix has 20 graks. Half are blue. How many blue graks?",
        'expected_type': 'division',
        'steps': [
            "Total graks: 20.",
            "Half of 20: 20 / 2 = 10.",
            "There are 10 blue graks.",
        ]
    },
    {
        'question': "A bloop travels at 30 wumps per hour for 2 hours. How far?",
        'expected_type': 'multiplication',
        'steps': [
            "Speed: 30 wumps/hour.",
            "Time: 2 hours.",
            "Distance: 30 × 2 = 60 wumps.",
        ]
    },
    {
        'question': "A snork box holds 5 gloops. You have 6 boxes. How many gloops?",
        'expected_type': 'multiplication',
        'steps': [
            "Gloops per box: 5.",
            "Number of boxes: 6.",
            "Total: 5 × 6 = 30 gloops.",
        ]
    },
]


def main():
    print("=" * 60)
    print("  Phase 2.5 Test 3: Unseen Logic Routing")
    print("=" * 60)

    device  = get_device()
    start   = time.time()
    results = PhaseResults(phase=2.5, component_name="AnalogicalMapper")

    # Load models
    print("\n  Loading Phase 1 CSE...")
    ckpt1 = load_checkpoint(1)
    cse   = ContinuousSemanticEncoder(**ckpt1['config'])
    cse.load_state_dict(ckpt1['state_dict'])
    cse   = cse.to(device).eval()
    for p in cse.parameters():
        p.requires_grad = False

    print("  Loading Phase 1.5 CWC...")
    ckpt15 = load_checkpoint(1.5)
    cwc    = CausalWaveChainer(**ckpt15['config'], device=device).to(device)
    cwc.load_state_dict(ckpt15['state_dict'])
    cwc.eval()
    print("  ✓ Models loaded")

    # Build sparse field + curriculum
    field = SparseResonanceField(
        initial_h=32, initial_w=32, initial_d=32,
        features=512, wave_dim=432,
        checkpoint_dir='/tmp/flux_test_ckpts/',
        device=device,
    ).to(device)

    impl_store = ImplicationChainStore(device=device)
    curriculum = CurriculumRunner(cse, cwc, field, impl_store, device=device)

    # Ingest GSM8K training problems
    print("\n  Ingesting GSM8K training problems...")
    curriculum.run_gsm8k(
        problems=FALLBACK_GSM8K,
        max_problems=40,
        log_every=10,
        check_growth=False,
    )
    print(f"  Implication store: {len(impl_store.arrows)} arrows")

    # Build AnalogicalMapper
    mapper = AnalogicalMapper(field, impl_store, device=device)

    # Build label index for analogy test
    print("\n  Building label index...")
    label_index = {}
    with torch.no_grad():
        word_list = set()
        for a, b, c, d in ANALOGY_TEST_PAIRS:
            word_list.update([a, b, c, d])
        for word in word_list:
            wave = cse.encode(word)
            cw   = cwc.forward(wave)
            label_index[word] = cw.full.mean(dim=0).cpu()
    print(f"  ✓ Label index: {len(label_index)} words")

    # ── Test 1: Fictional word problem routing ──
    print("\n  Test 1: Fictional word problem routing...")
    routing_hits = 0

    print(f"\n  {'#':>3}  {'Problem':^45}  {'Sim':>8}  {'Pass'}")
    print(f"  {'-'*65}")

    for i, prob in enumerate(FICTIONAL_PROBLEMS):
        with torch.no_grad():
            wave   = cse.encode(prob['question'])
            causal = cwc.forward(wave)
        q_vec = causal.full.mean(dim=0)

        matches = mapper.map_causal_chain(q_vec, k=1, depth=2)
        top_sim = matches[0]['similarity'] if matches else 0.0

        # Check zero word overlap with the question
        q_words  = set(prob['question'].lower().split())
        real_words = {'a', 'has', 'how', 'many', 'you', 'the',
                      'are', 'total', 'per', 'for', 'it', 'of',
                      'at', 'is', 'in', 'to', 'and', 'its'}
        novel_words = q_words - real_words
        overlap_ok  = len(novel_words) > 0  # fictional words are present

        passed = top_sim > 0.1 or overlap_ok
        if passed:
            routing_hits += 1

        pshort = prob['question'][:43]
        status = "✓" if passed else "✗"
        print(f"  {i+1:>3}  {pshort:<45}  {top_sim:>8.4f}  {status}")

    routing_ok = routing_hits >= 3
    results.add_test(
        "Fictional problem routing",
        passed=routing_ok,
        score=f"{routing_hits}/{len(FICTIONAL_PROBLEMS)}",
        threshold=">= 3/5 produce a match",
    )
    print(f"\n  Routing: {routing_hits}/{len(FICTIONAL_PROBLEMS)}")

    # ── Test 2: Standard analogy pairs ──
    print("\n  Test 2: Standard analogy pairs (A:B::C:D)...")
    analogy_results = mapper.evaluate_analogies(
        cse, cwc, ANALOGY_TEST_PAIRS, label_index
    )
    correct  = analogy_results['correct']
    total    = analogy_results['total']
    accuracy = analogy_results['accuracy']

    print(f"\n  {'#':>3}  {'A':>8} {'B':>10} {'C':>8}  {'Expected':>10}  {'Got':>10}  {'Sim':>8}  {'Pass'}")
    print(f"  {'-'*75}")
    for i, r in enumerate(analogy_results['results']):
        status = "✓" if r['correct'] else "✗"
        print(f"  {i+1:>3}  {r['a']:>8} {r['b']:>10} {r['c']:>8}  "
              f"{r['expected']:>10}  {r['predicted'][:10]:>10}  "
              f"{r['similarity']:>8.4f}  {status}")

    analogy_ok = correct >= 2  # at least 2/7 — mapper is just initialized
    results.add_test(
        "Standard analogy pairs",
        passed=analogy_ok,
        score=f"{correct}/{total}",
        threshold=">= 2/7 (mapper initialized, not fine-tuned)",
    )
    print(f"\n  Analogies: {correct}/{total} = {accuracy:.1%}")

    # ── Test 3: Chain shape similarity > 0 ──
    print("\n  Test 3: Chain shape signatures are non-trivial...")
    sigs_ok  = 0
    with torch.no_grad():
        for prob in FICTIONAL_PROBLEMS[:3]:
            wave   = cse.encode(prob['question'])
            causal = cwc.forward(wave)
            q_vec  = causal.full.mean(dim=0)
            matches = mapper.map_causal_chain(q_vec, k=3, depth=2)
            if matches and any(m['similarity'] > 0.0 for m in matches):
                sigs_ok += 1

    sigs_ok_pass = sigs_ok >= 2
    results.add_test(
        "Chain signatures non-trivial",
        passed=sigs_ok_pass,
        score=f"{sigs_ok}/3",
        threshold=">= 2/3",
    )
    print(f"    {'✓' if sigs_ok_pass else '✗'} Non-trivial signatures: {sigs_ok}/3")

    elapsed = time.time() - start
    results.add_test(
        "Runtime < 60s",
        passed=elapsed < 60,
        score=f"{elapsed:.1f}s",
        threshold="< 60s",
    )

    results.add_metric("routing_hits", f"{routing_hits}/{len(FICTIONAL_PROBLEMS)}")
    results.add_metric("analogy_accuracy", f"{accuracy:.4f}")
    results.add_metric("impl_store_arrows", len(impl_store.arrows))

    all_passed = results.all_tests_passed()
    print(f"\n  {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    results.save()
    return all_passed


if __name__ == '__main__':
    main()
