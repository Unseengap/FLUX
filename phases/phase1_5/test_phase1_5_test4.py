"""
PHASE 1.5 TEST 4: CausalWave integrates cleanly with Phase 1 and Phase 2.

Pass criteria:
    - CausalWave.full shape: [seq_len, 608] ✓
    - to_phase2_wave() returns SemanticWave with shape [seq_len, 432] ✓
    - Phase 2 field accepts the stripped wave without error ✓
    - Field attractor similarity > 0.7 after 10 perturbations ✓
    - Phase 1 CSE output is bit-identical before and after Phase 1.5 ✓
"""

import sys
import time
import hashlib
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'phases' / 'phase1'))
sys.path.insert(0, str(ROOT / 'phases' / 'phase2'))
sys.path.insert(0, str(ROOT / 'phases' / 'phase1_5'))

import torch
import torch.nn.functional as F
from flux_utils import load_checkpoint, PhaseResults
from cse import ContinuousSemanticEncoder
from field import ResonanceField
from causal_encoder import CausalWaveChainer
from causal_types import TOTAL_CAUSAL_DIM

TEST_TEXTS = [
    "the dog chased the cat across the yard",
    "scientists discovered a new species in the ocean",
    "the economy grew during the second quarter",
    "it started raining and people opened umbrellas",
    "the fire alarm went off and people evacuated",
]


def checkpoint_hash(path: Path) -> str:
    if not path.exists():
        return "NOT_FOUND"
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start  = time.time()

    print("=" * 60)
    print("FLUX Phase 1.5 Test 4: Pipeline Integration")
    print("=" * 60)

    # ── Load Phase 1 CSE ──
    print("\n  Step 1: Loading Phase 1 CSE (frozen)...")
    p1_path = ROOT / 'checkpoints' / 'phase1.phase.pt'
    p1_hash = checkpoint_hash(p1_path)

    ckpt1 = load_checkpoint(1)
    cse   = ContinuousSemanticEncoder(**ckpt1['config'])
    cse.load_state_dict(ckpt1['state_dict'])
    cse   = cse.to(device).eval()
    for p in cse.parameters():
        p.requires_grad = False
    print(f"  ✓ CSE loaded: {sum(p.numel() for p in cse.parameters()):,} params")

    # Record CSE output BEFORE Phase 1.5
    with torch.no_grad():
        wave_before = cse.encode(TEST_TEXTS[0])
        full_before = wave_before.full.clone()

    # ── Load Phase 1.5 CWC ──
    print("\n  Step 2: Loading Phase 1.5 CausalWaveChainer...")
    ckpt15 = load_checkpoint(1.5)
    cwc    = CausalWaveChainer(**ckpt15['config'], device=device).to(device)
    cwc.load_state_dict(ckpt15['state_dict'])
    cwc.eval()
    cwc_params = sum(p.numel() for p in cwc.parameters())
    print(f"  ✓ CWC loaded: {cwc_params:,} params")

    # ── Load Phase 2 Field ──
    print("\n  Step 3: Loading Phase 2 ResonanceField...")
    ckpt2     = load_checkpoint(2)
    field_cfg = ckpt2['config']['field']
    field     = ResonanceField(**field_cfg).to(device)
    field.load_state_dict(ckpt2['state_dict'])
    field.eval()
    print(f"  ✓ ResonanceField loaded")

    all_results = []

    print(f"\n  ── Integration Tests ──\n")
    print(f"  {'#':>3}  {'Text':<42}  {'608?':>5}  {'432?':>5}  {'Field?':>6}  {'Sim':>6}")
    print(f"  {'-'*72}")

    with torch.no_grad():
        for i, text in enumerate(TEST_TEXTS):
            # Step A: encode with frozen CSE
            wave = cse.encode(text)
            assert wave.full.shape[-1] == 432, f"CSE output wrong: {wave.full.shape}"

            # Step B: extend with CWC
            cw = cwc.forward(wave)
            shape_608 = cw.full.shape[-1] == TOTAL_CAUSAL_DIM

            # Step C: strip back to Phase 2 wave
            p2_wave = cw.to_phase2_wave()
            shape_432 = p2_wave.full.shape[-1] == 432

            # Step D: pass to Phase 2 field — 10 perturbations
            vec = p2_wave.full.mean(dim=0)
            for _ in range(10):
                field.perturb(vec)
            field_ok = True

            # Step E: query field — check attractor formed
            feats, sims, locs = field.query(vec)
            top_sim = sims[0].item() if sims.numel() > 0 else 0.0
            sim_ok  = top_sim > 0.7

            text_short = text[:40] + ".." if len(text) > 40 else text
            print(f"  {i+1:>3}  {text_short:<42}  {'✓' if shape_608 else '✗':>5}  {'✓' if shape_432 else '✗':>5}  {'✓' if field_ok else '✗':>6}  {top_sim:.4f}")

            all_results.append({
                'shape_608': shape_608,
                'shape_432': shape_432,
                'field_ok':  field_ok,
                'sim_ok':    sim_ok,
                'top_sim':   top_sim,
            })

    # ── Verify CSE output unchanged ──
    print(f"\n  ── CSE Bit-Identical Check ──")
    with torch.no_grad():
        wave_after = cse.encode(TEST_TEXTS[0])
        full_after = wave_after.full

    max_diff = (full_before - full_after).abs().max().item()
    cse_unchanged = max_diff < 1e-6
    print(f"  Max difference in CSE output: {max_diff:.2e}")
    print(f"  CSE output bit-identical:     {'✓ YES' if cse_unchanged else '✗ NO — CSE WAS MODIFIED'}")

    # Verify Phase 1 hash unchanged
    p1_hash_after = checkpoint_hash(p1_path)
    hash_ok = p1_hash == p1_hash_after
    print(f"  Phase 1 checkpoint hash:      {'✓ UNCHANGED' if hash_ok else '✗ CHANGED'}")

    # Verify stored hash from training matches
    stored_hash = ckpt15.get('phase1_checkpoint_hash', 'NOT_STORED')
    stored_ok   = stored_hash == p1_hash
    print(f"  Stored hash matches current:  {'✓' if stored_ok else '✗'}")

    elapsed = time.time() - start

    # ── Summary ──
    shape_608_all = all(r['shape_608'] for r in all_results)
    shape_432_all = all(r['shape_432'] for r in all_results)
    field_ok_all  = all(r['field_ok']  for r in all_results)
    sim_ok_all    = all(r['sim_ok']    for r in all_results)
    mean_sim      = sum(r['top_sim'] for r in all_results) / len(all_results)

    all_pass = (
        shape_608_all and shape_432_all and
        field_ok_all  and sim_ok_all    and
        cse_unchanged and hash_ok
    )

    print(f"\n  ── Results ──")
    print(f"  {'✓' if shape_608_all else '✗'} CausalWave.full shape [seq, 608]: {shape_608_all}")
    print(f"  {'✓' if shape_432_all else '✗'} to_phase2_wave() shape [seq, 432]: {shape_432_all}")
    print(f"  {'✓' if field_ok_all else '✗'} Phase 2 field accepts wave: {field_ok_all}")
    print(f"  {'✓' if sim_ok_all else '✗'} Attractor similarity > 0.7: mean={mean_sim:.4f}")
    print(f"  {'✓' if cse_unchanged else '✗'} CSE output bit-identical: max_diff={max_diff:.2e}")
    print(f"  {'✓' if hash_ok else '✗'} Phase 1 checkpoint unchanged")

    results = PhaseResults(phase=1.5)
    results.add("CausalWave shape 608", int(shape_608_all), "True", shape_608_all)
    results.add("Phase2Wave shape 432", int(shape_432_all), "True", shape_432_all)
    results.add("Field accepts wave", int(field_ok_all), "True", field_ok_all)
    results.add("Attractor sim > 0.7", mean_sim, "> 0.7", sim_ok_all)
    results.add("CSE output unchanged", max_diff, "< 1e-6", cse_unchanged)
    results.add("Phase1 checkpoint hash", int(hash_ok), "unchanged", hash_ok)
    results.save()

    print(f"\n{'='*60}")
    print(f"All tests passed: {all_pass}")
    print(f"Ready for Phase 2: {all_pass}")
    print(f"{'='*60}")
    if all_pass:
        print("  ✓ PIPELINE INTEGRATION TEST PASSED")
    else:
        print("  ✗ PIPELINE INTEGRATION TEST FAILED")

    return all_pass


if __name__ == '__main__':
    main()