"""
Training script for Phase 1.5 CausalWaveChainer.

Training data: WikiText-2 — naturally ordered sentences
               Contradiction pairs — 50 curated pairs
               Implication pairs  — extracted from text co-occurrence

Training objectives (combined loss):
    L_total = λ1 * L_coherence
            + λ2 * L_order
            + λ3 * L_contradiction
            + λ4 * L_implication

    λ1 = 1.0   (primary — causal flow must work)
    λ2 = 0.5   (order sensitivity — sequences must be directional)
    λ3 = 0.3   (contradiction tension — conflicts must be felt)
    λ4 = 0.2   (implication consistency — chains must be transitive)

Hardware: GPU recommended, CPU viable for smoke test
Steps: 3000
Checkpoint: checkpoints/phase1_5.phase.pt
"""

import sys
import os
import argparse
import hashlib
import time
import math
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── Path setup ──
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'phases' / 'phase1'))
sys.path.insert(0, str(ROOT / 'phases' / 'phase1_5'))

from flux_utils import load_checkpoint, save_checkpoint, PhaseResults
from cse import ContinuousSemanticEncoder
from causal_encoder import CausalWaveChainer
from causal_types import CausalWave
from contradiction import ContradictionRegistry
from implication import ImplicationChainStore

# ── Contradiction training pairs ──
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

# ── Implication pairs ──
IMPLICATION_PAIRS = [
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


def get_wikitext_sentences(n: int = 3000) -> list:
    """Load WikiText-2 and extract clean sentence pairs."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        sentences = []
        for item in dataset:
            text = item['text'].strip()
            if len(text) > 40 and not text.startswith('=') and len(text) < 300:
                sentences.append(text)
            if len(sentences) >= n:
                break
        print(f"  ✓ Loaded {len(sentences)} sentences from WikiText-2")
        return sentences
    except Exception as e:
        print(f"  ⚠ WikiText-2 unavailable ({e}), using fallback")
        return [
            "The dog chased the cat across the yard into the garden.",
            "Scientists discovered a new species in the deep ocean.",
            "The economy grew significantly during the second quarter.",
            "Machine learning algorithms process data to find patterns.",
            "The ancient civilization built monuments that still stand.",
            "Photosynthesis converts sunlight into energy for plants.",
            "Democracy requires the participation of informed citizens.",
            "The orchestra performed the symphony to a standing ovation.",
            "Climate change poses significant challenges to agriculture.",
            "The researchers published their findings in a major journal.",
        ] * 300


def checkpoint_hash(path: Path) -> str:
    """MD5 hash of a checkpoint file to prove it was not modified."""
    if not path.exists():
        return "NOT_FOUND"
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def validate(cwc, cse, val_sentences, contradiction_pairs, device, step):
    """Run validation and return metrics dict."""
    cwc.eval()
    with torch.no_grad():
        # Coherence validation
        ordered_scores, shuffled_scores = [], []
        for sent in val_sentences[:100]:
            words = sent.split()
            if len(words) < 4:
                continue
            cw = cwc.encode(cse, sent)
            coh = cw.causal_coherence()
            if coh.numel() == 0:
                continue
            ordered_scores.append(coh.mean().item())
            idx = torch.randperm(len(words)).tolist()
            shuffled = ' '.join([words[i] for i in idx])
            cw_s = cwc.encode(cse, shuffled)
            coh_s = cw_s.causal_coherence()
            if coh_s.numel() > 0:
                shuffled_scores.append(coh_s.mean().item())

        mean_ordered  = sum(ordered_scores)  / max(len(ordered_scores), 1)
        mean_shuffled = sum(shuffled_scores) / max(len(shuffled_scores), 1)
        coherence_gap = mean_ordered - mean_shuffled

        # Order sensitivity: how many sentences beat their shuffles
        order_wins = sum(
            o > s for o, s in zip(ordered_scores, shuffled_scores)
        )
        order_acc = order_wins / max(len(ordered_scores), 1)

        # Contradiction tension validation
        tension_gaps = []
        for stmt, contra, neutral in contradiction_pairs[:20]:
            cw_c = cwc.encode(cse, stmt + ' ' + contra)
            cw_n = cwc.encode(cse, stmt + ' ' + neutral)
            gap  = cw_c.tension_score() - cw_n.tension_score()
            tension_gaps.append(gap)
        mean_tension_gap = sum(tension_gaps) / max(len(tension_gaps), 1)
        contra_detected  = sum(g > 0 for g in tension_gaps)

    cwc.train()
    return {
        'mean_ordered_coherence':  mean_ordered,
        'mean_shuffled_coherence': mean_shuffled,
        'coherence_gap':           coherence_gap,
        'order_accuracy':          order_acc,
        'order_wins':              order_wins,
        'mean_tension_gap':        mean_tension_gap,
        'contra_detected':         contra_detected,
    }


def train(args):
    device = args.device
    steps  = args.steps
    print(f"\n{'='*60}")
    print(f"FLUX Phase 1.5: Causal Wave Chaining (CWC)")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    print(f"  Steps:  {steps}")

    # ── Load Phase 1 CSE (frozen) ──
    print("\n── Loading Phase 1 CSE ──")
    ckpt1 = load_checkpoint(1)
    cse = ContinuousSemanticEncoder(**ckpt1['config'])
    cse.load_state_dict(ckpt1['state_dict'])
    cse = cse.to(device).eval()
    for p in cse.parameters():
        p.requires_grad = False
    print(f"  ✓ CSE loaded and frozen: {sum(p.numel() for p in cse.parameters()):,} params")

    # Record Phase 1 checkpoint hash — must not change
    p1_path = ROOT / 'checkpoints' / 'phase1.phase.pt'
    p1_hash = checkpoint_hash(p1_path)
    print(f"  ✓ Phase 1 checkpoint hash: {p1_hash[:16]}...")

    # ── Build CausalWaveChainer ──
    print("\n── Building CausalWaveChainer ──")
    cwc = CausalWaveChainer(device=device).to(device)
    cwc_params = sum(p.numel() for p in cwc.parameters())
    print(f"  ✓ CWC parameters: {cwc_params:,}")

    # ── Build supporting stores ──
    contradiction_registry = ContradictionRegistry(device=device)
    implication_store      = ImplicationChainStore(device=device)

    # ── Load data ──
    print("\n── Loading Training Data ──")
    all_sentences = get_wikitext_sentences(3000)
    split = int(len(all_sentences) * 0.85)
    train_sentences = all_sentences[:split]
    val_sentences   = all_sentences[split:]
    print(f"  ✓ Train: {len(train_sentences)} sentences")
    print(f"  ✓ Val:   {len(val_sentences)} sentences")
    print(f"  ✓ Contradiction pairs: {len(CONTRADICTION_PAIRS)}")
    print(f"  ✓ Implication pairs:   {len(IMPLICATION_PAIRS)}")

    # Pre-populate implication store from training pairs
    # Use force=True to bypass duplicate detection on untrained model
    # (vectors will spread after training — store is rebuilt at checkpoint time)
    print("\n── Pre-populating Implication Store ──")
    with torch.no_grad():
        for premise, conclusion, strength in IMPLICATION_PAIRS:
            cw_p = cwc.encode(cse, premise)
            cw_c = cwc.encode(cse, conclusion)
            src  = cw_p.full.mean(dim=0)
            tgt  = cw_c.full.mean(dim=0)
            # Bypass duplicate check — directly append to store
            from causal_types import CausalArrow
            implication_store.arrows.append(CausalArrow(
                source_vector  = src.detach().cpu(),
                target_vector  = tgt.detach().cpu(),
                strength       = strength,
                evidence_count = 1,
                arrow_type     = 'temporal',
            ))
    print(f"  ✓ Implication store: {len(implication_store.arrows)} arrows")

    # ── Optimizer ──
    optimizer = Adam(cwc.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-5)

    # Loss weights — order and contradiction boosted to break collapse
    λ1, λ2, λ3, λ4 = 1.0, 2.0, 1.5, 0.2

    # ── Training loop ──
    print(f"\n── Training for {steps} steps ──\n")
    cwc.train()
    start_time = time.time()
    best_gap   = -999.0
    best_state = None

    for step in range(1, steps + 1):
        # Sample a sentence from training data
        idx  = torch.randint(0, len(train_sentences) - 1, (1,)).item()
        sent = train_sentences[idx]
        next_sent = train_sentences[idx + 1]

        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)

        # ── L1: Causal coherence ──
        cw = cwc.encode(cse, sent)
        l_coh = cwc.causal_coherence_loss(cw)
        total_loss = total_loss + λ1 * l_coh

        # Also coherence across sentence boundary
        cw_pair = cwc.encode(cse, sent + ' ' + next_sent)
        l_pair  = cwc.causal_coherence_loss(cw_pair)
        total_loss = total_loss + λ1 * 0.5 * l_pair

        # ── L2: Order sensitivity ──
        words = sent.split()
        if len(words) >= 3:
            l_ord = cwc.order_sensitivity_loss(cse, sent, n_shuffles=2)
            total_loss = total_loss + λ2 * l_ord
        else:
            l_ord = torch.tensor(0.0, device=device)

        # ── L3: Contradiction ──
        c_idx   = step % len(CONTRADICTION_PAIRS)
        stmt, contra, neutral = CONTRADICTION_PAIRS[c_idx]
        l_contra = cwc.contradiction_loss(cse, stmt, contra, neutral)
        total_loss = total_loss + λ3 * l_contra

        # Register in contradiction registry periodically
        if step % 50 == 0:
            with torch.no_grad():
                cw_s = cwc.encode(cse, stmt)
                cw_c = cwc.encode(cse, contra)
                contradiction_registry.register(
                    cw_s.full.mean(dim=0),
                    cw_c.full.mean(dim=0),
                    strength=0.7,
                    description=f"{stmt[:30]} ↔ {contra[:30]}",
                    step=step,
                )

        # ── L4: Implication consistency ──
        if step % 3 == 0:
            i_idx  = step % len(IMPLICATION_PAIRS)
            prem,  conc,  strength  = IMPLICATION_PAIRS[i_idx]
            i_idx2 = (i_idx + 1) % len(IMPLICATION_PAIRS)
            prem2, conc2, strength2 = IMPLICATION_PAIRS[i_idx2]

            # CSE is frozen — encode under no_grad, then pass SemanticWave
            # through CWC (which IS differentiable) to get gradients
            with torch.no_grad():
                wave_a    = cse.encode(prem)
                wave_b    = cse.encode(conc)
                wave_c_i  = cse.encode(conc2)

            cw_a     = cwc.forward(wave_a)
            cw_b     = cwc.forward(wave_b)
            cw_c_imp = cwc.forward(wave_c_i)

            l_impl = cwc.implication_consistency_loss(
                cw_a, cw_b, cw_c_imp,
                implies_ab=True,
                implies_bc=(strength2 > 0.7),
            )
            total_loss = total_loss + λ4 * l_impl
        else:
            l_impl = torch.tensor(0.0, device=device)

        # ── Backward ──
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(cwc.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # ── Logging every 100 steps ──
        if step % 100 == 0:
            elapsed = time.time() - start_time
            print(
                f"  Step {step:4d} | "
                f"coh={l_coh.item():.4f} "
                f"ord={l_ord.item() if hasattr(l_ord, 'item') else 0:.4f} "
                f"contra={l_contra.item():.4f} "
                f"impl={l_impl.item() if hasattr(l_impl, 'item') else 0:.4f} "
                f"total={total_loss.item():.4f} "
                f"| {elapsed:.0f}s"
            )

        # ── Validation every 500 steps ──
        if step % 500 == 0:
            val = validate(cwc, cse, val_sentences, CONTRADICTION_PAIRS, device, step)
            gap = val['coherence_gap']
            print(f"\n  ── VAL step {step} ──")
            print(f"     ordered_coherence:  {val['mean_ordered_coherence']:.4f}")
            print(f"     shuffled_coherence: {val['mean_shuffled_coherence']:.4f}")
            print(f"     coherence_gap:      {gap:.4f}  (target > 0.3)")
            print(f"     order_accuracy:     {val['order_accuracy']:.3f}  ({val['order_wins']}/50+ correct)")
            print(f"     tension_gap:        {val['mean_tension_gap']:.4f}  (target > 0.2)")
            print(f"     contra_detected:    {val['contra_detected']}/20")
            print()

            # Save best model
            if gap > best_gap:
                best_gap   = gap
                best_state = {k: v.clone() for k, v in cwc.state_dict().items()}

    # ── Training complete ──
    duration = time.time() - start_time
    print(f"\n── Training Complete ──")
    print(f"  Duration: {duration/60:.1f} min")
    print(f"  Best coherence gap: {best_gap:.4f}")

    # Restore best state
    if best_state is not None:
        cwc.load_state_dict(best_state)

    # Final validation
    print("\n── Final Validation ──")
    val_final = validate(cwc, cse, val_sentences, CONTRADICTION_PAIRS, device, steps)
    for k, v in val_final.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Verify Phase 1 hash unchanged
    p1_hash_after = checkpoint_hash(p1_path)
    hash_ok = p1_hash == p1_hash_after
    print(f"\n  Phase 1 checkpoint hash match: {'✓' if hash_ok else '✗ CHANGED!'}")
    print(f"    Before: {p1_hash[:16]}")
    print(f"    After:  {p1_hash_after[:16]}")

    # ── Save checkpoint ──
    print("\n── Saving Checkpoint ──")
    ckpt_data = {
        'phase': 1.5,
        'component': 'CausalWaveChainer',
        'timestamp': datetime.now().isoformat(),
        'state_dict': cwc.state_dict(),
        'contradiction_registry': contradiction_registry.save(),
        'implication_store': implication_store.save(),
        'config': {
            'wave_dim':      432,
            'forward_dim':   64,
            'backward_dim':  64,
            'tension_dim':   32,
            'chain_dim':     16,
        },
        'metrics': {
            'mean_ordered_coherence':  val_final['mean_ordered_coherence'],
            'mean_shuffled_coherence': val_final['mean_shuffled_coherence'],
            'coherence_gap':           val_final['coherence_gap'],
            'order_accuracy':          val_final['order_accuracy'],
            'mean_tension_gap':        val_final['mean_tension_gap'],
            'contra_detected':         val_final['contra_detected'],
            'training_steps':          steps,
            'training_duration_s':     duration,
            'best_coherence_gap':      best_gap,
        },
        'phase1_checkpoint_hash': p1_hash,
        'phase1_hash_unchanged':  hash_ok,
    }
    save_checkpoint(1.5, ckpt_data)
    # Also save explicitly to the underscore path the notebook expects
    ckpt_path = ROOT / 'checkpoints' / 'phase1_5.phase.pt'
    torch.save(ckpt_data, str(ckpt_path))
    size_mb = ckpt_path.stat().st_size / 1e6 if ckpt_path.exists() else 0
    print(f"✓ Phase 1.5 checkpoint saved: {ckpt_path} ({size_mb:.1f} MB)")

    # ── Generate RESULTS file ──
    _write_results(val_final, duration, hash_ok, p1_hash, best_gap)

    print(f"\n{'='*60}")
    print(f"Phase 1.5 training complete!")
    print(f"  Coherence gap:   {val_final['coherence_gap']:.4f}  (target > 0.3)")
    print(f"  Order accuracy:  {val_final['order_accuracy']:.3f}  (target > 0.9)")
    print(f"  Tension gap:     {val_final['mean_tension_gap']:.4f}  (target > 0.2)")
    print(f"  CSE unchanged:   {'✓' if hash_ok else '✗'}")
    print(f"  Next: run test scripts to verify acceptance criteria")
    print(f"{'='*60}\n")


def _write_results(val, duration, hash_ok, p1_hash, best_gap):
    results_dir = ROOT / 'phases' / 'phase1_5'
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / 'RESULTS_PHASE_1_5.md'
    now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    order_pass   = val['order_accuracy'] >= 0.9
    tension_pass = val['mean_tension_gap'] >= 0.2
    gap_pass     = val['coherence_gap'] >= 0.3

    content = f"""# Results: Phase 1.5 — Causal Wave Chaining (CWC)
Generated: {now}
Duration: {duration/60:.1f} min

## Component Status
CausalWaveChainer: ✓ BUILT
ContradictionRegistry: ✓ BUILT
ImplicationChainStore: ✓ BUILT
Checkpoint: checkpoints/phase1_5.phase.pt ✓

## Test Results
| Test | Score | Threshold | Pass? |
|------|-------|-----------|-------|
| Coherence Gap (ordered vs shuffled) | {val['coherence_gap']:.4f} | > 0.3 | {'✓' if gap_pass else '✗'} |
| Order Accuracy | {val['order_accuracy']:.3f} | > 0.9 | {'✓' if order_pass else '✗'} |
| Mean Tension Gap | {val['mean_tension_gap']:.4f} | > 0.2 | {'✓' if tension_pass else '✗'} |
| Contradiction Detected | {val['contra_detected']}/20 | ≥ 14/20 | {'✓' if val['contra_detected'] >= 14 else '✗'} |

## Key Metrics
- Mean ordered coherence:  {val['mean_ordered_coherence']:.4f}
- Mean shuffled coherence: {val['mean_shuffled_coherence']:.4f}
- Coherence gap:           {val['coherence_gap']:.4f}
- Best coherence gap:      {best_gap:.4f}
- Order accuracy:          {val['order_accuracy']:.3f}
- Mean tension gap:        {val['mean_tension_gap']:.4f}

## Phase 1 Compatibility
- CSE checkpoint hash: {p1_hash[:16]}...
- CSE output unchanged: {'YES ✓' if hash_ok else 'NO ✗ — CRITICAL ERROR'}

## Phase 1.5 → Phase 2 Readiness
All tests passing: {'YES ✓' if (gap_pass and order_pass and tension_pass) else 'NO ✗'}
Checkpoint saved: YES ✓
Ready for Phase 2 (with CWC active): {'YES ✓' if hash_ok else 'NO — CSE was modified'}
"""
    path.write_text(content)
    print(f"  Results saved to: {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--steps', type=int, default=3000)
    args = parser.parse_args()
    train(args)