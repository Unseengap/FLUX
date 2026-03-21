[01:14:35] 
▶ CELL: Cell 15 — Save Kaggle Artifacts
[01:14:35]   Started: 2026-03-21 01:14:35
  ✓ Copied phase2.phase.pt (545.3 MB)
  ✓ Copied phase2.flx (546.3 MB)
  ✓ Copied RESULTS_PHASE_2.md
  ✓ Copied phase2.log
  ✓ Copied demo2_no_forgetting.png
  ✓ Copied demo1_attractor_formation.png

  Kaggle output: [PosixPath('/kaggle/working/output/phase2.flx'), PosixPath('/kaggle/working/output/demo2_no_forgetting.png'), PosixPath('/kaggle/working/output/phase2.phase.pt'), PosixPath('/kaggle/working/output/RESULTS_PHASE_2.md'), PosixPath('/kaggle/working/output/phase2.log'), PosixPath('/kaggle/working/output/demo1_attractor_formation.png')]
[01:14:35]   ◼ CELL Cell 15 — Save Kaggle Artifacts — PASS

============================================================
  PHASE 2 COMPLETE ✓
============================================================


  ✓ Phase 2 checkpoint uploaded to HuggingFace Hub (545.3 MB)
    https://huggingface.co/UnseenGAP/FLUX
[01:12:45]   ◼ CELL Cell 6 — Save & Upload Checkpoint — PASS



[01:13:02] 
▶ CELL: Cell 7 — Test 1: Attractor Formation & No-Forgetting
[01:13:02]   Started: 2026-03-21 01:13:02
============================================================
  Phase 2 Test 1: Attractor Formation & No-Forgetting
============================================================
  Device: cuda
✓ Phase 1 checkpoint loaded (local, 7.0 MB)
  ✓ Using CSE-encoded wave vectors

  Step 1: Training Pattern A (20 repetitions)...
    Attractor A formed at (7, 6, 7), similarity: 0.9940
  Step 2: Training Pattern B (20 repetitions)...
    Attractor B formed at (7, 7, 7)
  Step 3: Training Pattern C (20 repetitions)...
    Attractor C formed at (7, 7, 7)

  Step 4: Verifying attractor persistence...
    Total: 3
    Alive: 3
    Dead:  0
    Survival rate: 100.00%
    ✓ Pattern A: similarity = 0.9704
    ✓ Pattern B: similarity = 0.9975
    ✓ Pattern C: similarity = 1.0000
  ✓ Attractor A survives B+C training: 0.9703508019447327 (threshold: 0.7)
  ✓ Attractor B survives C training: 0.9974833726882935 (threshold: 0.7)
  ✓ All 3 attractors coexist: 1.0 (threshold: 0.9)
  ✓ Forgetting score < 0.2: 0.0 (threshold: 0.2)
  ✓ Field has attractors: 693 (threshold: 3)

  ✓ ALL TESTS PASSED
  Forgetting score: 0.0000 (target: < 0.2)

==================================================
Phase 2 Results saved to: /kaggle/working/FLUX/phases/phase2/RESULTS_PHASE_2.md
All tests passed: True
Ready for Phase 3: True
==================================================
[01:13:02]   ◼ CELL Cell 7 — Test 1 — PASS


[01:13:10] 
▶ CELL: Cell 8 — Test 2: Local Update Verification
[01:13:10]   Started: 2026-03-21 01:13:10
============================================================
  Phase 2 Test 2: Local Update Verification
============================================================
  Device: cuda

  Perturbation center: (8, 8, 8)
  Selected 100 far locations (distance > 6)
  Recording field state before perturbation...
  Applying perturbation at center...
  Updated 729 locations
  Recording field state after perturbation...

  Results:
    Avg change at far locations: 0.00000000
    Max change at far locations: 0.00000000
    Non-zero changes: 0/100
    Center state norm: 0.131812
  ✓ Average far change < 0.001: 0.0 (threshold: 0.001)
  ✓ Max far change < 0.01: 0.0 (threshold: 0.01)
  ✓ Zero far changes (exact locality): 0 (threshold: 0)

  ✓ ALL TESTS PASSED
  Updates are TRULY LOCAL ✓

==================================================
Phase 2 Results saved to: /kaggle/working/FLUX/phases/phase2/RESULTS_PHASE_2.md
All tests passed: True
Ready for Phase 3: True
==================================================
[01:13:10]   ◼ CELL Cell 8 — Test 2 — PASS


[01:13:18] 
▶ CELL: Cell 9 — Test 3: Field Stability
[01:13:18]   Started: 2026-03-21 01:13:18
============================================================
  Phase 2 Test 3: Field Stability Over 1000 Steps
============================================================
  Device: cuda

  Initial energy: 4096.0
  Ceiling (×10): 40960.0
  Floor (×0.01): 40.9600

  Running 1000 perturbations...
    Step 100: energy=3967.0, attractors=912, max_mass=0.4294, elapsed=0.1s
    Step 200: energy=3609.6, attractors=1594, max_mass=0.8589, elapsed=0.2s
    Step 300: energy=3288.1, attractors=1901, max_mass=1.2883, elapsed=0.3s
    Step 400: energy=2993.1, attractors=2129, max_mass=1.7177, elapsed=0.4s
    Step 500: energy=2718.2, attractors=2241, max_mass=2.1472, elapsed=0.5s
    Step 600: energy=2463.8, attractors=2367, max_mass=2.5766, elapsed=0.6s
    Step 700: energy=2230.8, attractors=2523, max_mass=3.0060, elapsed=0.7s
    Step 800: energy=2017.4, attractors=2682, max_mass=3.4355, elapsed=0.8s
    Step 900: energy=1820.9, attractors=2849, max_mass=3.8649, elapsed=0.9s
    Step 1000: energy=1639.2, attractors=2912, max_mass=4.2943, elapsed=1.0s

  Results:
    Total time: 1.0s
    Energy range: [1639.21, 4094.69]
    Energy exploded: False
    Energy collapsed: False
    Final attractors: 2912
    Attractors decreased at any point: False
    Attractor history: [912, 1594, 1901, 2129, 2241, 2367, 2523, 2682, 2849, 2912]
  ✓ Energy never explodes (< init×10): 4094.693603515625 (threshold: 40960.0)
  ✓ Energy never collapses (> init×0.01): 1639.21044921875 (threshold: 40.96)
  ✓ Attractors grow monotonically: monotonic (threshold: monotonic)
  ✓ Completes in < 120 seconds: 1.0s (threshold: 120s)
  ✓ At least 5 attractors after 1000 steps: 2912 (threshold: 5)

  ✓ ALL TESTS PASSED
  Field is STABLE ✓

==================================================
Phase 2 Results saved to: /kaggle/working/FLUX/phases/phase2/RESULTS_PHASE_2.md
All tests passed: True
Ready for Phase 3: True
==================================================
[01:13:19]   ◼ CELL Cell 9 — Test 3 — PASS



▶ CELL: Cell 10 — Demo 1: Attractor Formation Visualization
[01:13:24]   Started: 2026-03-21 01:13:24
============================================================
  Phase 2 Demo 1: Attractor Formation Visualization
============================================================
  Device: cuda
✓ Phase 1 checkpoint loaded (local, 7.0 MB)
  ✓ Using CSE-encoded patterns

  Feeding 5 patterns × 15 repetitions = 75 perturbations
  Rep 1/15: attractors=0, energy=4089.5, max_mass=0.0478
  Rep 5/15: attractors=211, energy=3835.1, max_mass=0.2389
  Rep 10/15: attractors=685, energy=3444.7, max_mass=0.4779
  Rep 15/15: attractors=729, energy=3136.6, max_mass=0.7168

  Generating visualization...
  ✓ Saved: /kaggle/working/FLUX/phases/phase2/demo1_attractor_formation.png

  Final field state:
    Attractors: 729
    Total energy: 2723.0
    Max mass: 0.7168
    State std: 0.024026

  Pattern locations:
    Science    → ( 8,  5,  7) mass=0.7168 energy=0.7653
    Music      → ( 8,  6,  7) mass=0.6173 energy=0.7748
    Sports     → ( 8,  5,  7) mass=0.7168 energy=0.7653
    Cooking    → ( 8,  5,  7) mass=0.7168 energy=0.7653
    History    → ( 8,  5,  7) mass=0.7168 energy=0.7653

  ✓ Demo complete


  [01:13:39] 
▶ CELL: Cell 11 — Demo 2: No-Forgetting Demo
[01:13:39]   Started: 2026-03-21 01:13:39
============================================================
  Phase 2 Demo 2: The No-Forgetting Demonstration
============================================================
  Device: cuda
✓ Phase 1 checkpoint loaded (local, 7.0 MB)
  ✓ Using CSE-encoded patterns

  Step 1: Training on Pattern A (20 repetitions)...
    ✓ Attractor A at (8, 6, 8), similarity: 0.9951

  Step 2: Training on Pattern B (20 repetitions)...
    ✓ Attractor B at (9, 6, 8)
    Pattern A still alive: True (sim=0.9854)

  Step 3: Training on Pattern C (20 repetitions)...

  Step 4: Querying for all patterns...
    Query 'A': similarity = 0.9849, location = (8, 6, 8)
    Query 'B': similarity = 0.9947, location = (9, 6, 8)
    Query 'C': similarity = 0.9910, location = (8, 6, 8)

  Step 5: Computing forgetting score...

  ==================================================
  CATASTROPHIC FORGETTING SCORE: 0.00
  Transformer baseline:          ~0.45
  FLUX advantage:                COMPLETE RETENTION ✓
  ==================================================

  Attractor Survival Report:
    Pattern A   : similarity = 0.9745  ✓ ALIVE
    Pattern B   : similarity = 0.9972  ✓ ALIVE
    Pattern C   : similarity = 1.0000  ✓ ALIVE
    Survival rate: 100%

  Generating visualization...
  ✓ Saved: /kaggle/working/FLUX/phases/phase2/demo2_no_forgetting.png

  ✓ Demo complete
[01:13:53] 
▶ CELL: Cell 12 — Interactive Exploration
[01:13:53]   Started: 2026-03-21 01:13:53
  Field Statistics:
    total_energy: 262143.984375
    mean_energy: 0.9999999403953552
    min_energy: 0.013209173455834389
    max_energy: 1.320917010307312
    energy_std: 0.3772086501121521
    total_mass: 2180.15966796875
    mean_mass: 0.008316649124026299
    max_mass: 0.34574148058891296
    active_locations: 25145
    active_fraction: 0.09592056274414062
    state_mean_norm: 0.08341361582279205
    state_max_norm: 1.788787841796875
    state_std: 0.00889055710285902
    num_attractors_01: 6500
    num_attractors_03: 31
    num_attractors_05: 0
    step_count: 1000

  Top 10 Attractors:
    #1: loc=(37, 9, 62) mass=0.3457 energy=0.4108
    #2: loc=(60, 0, 23) mass=0.3404 energy=0.1372
    #3: loc=(60, 1, 48) mass=0.3404 energy=0.2735
    #4: loc=(61, 0, 22) mass=0.3404 energy=0.1040
    #5: loc=(61, 2, 48) mass=0.3404 energy=0.3144
    #6: loc=(47, 1, 62) mass=0.3305 energy=0.1336
    #7: loc=(37, 8, 61) mass=0.3203 energy=0.5875
    #8: loc=(37, 9, 61) mass=0.3195 energy=0.6009
    #9: loc=(37, 8, 62) mass=0.3154 energy=0.4017
    #10: loc=(47, 2, 60) mass=0.3133 energy=0.3843

  Custom queries:
    'The cat sat on the mat...' → (6,62,39) top_sim=0.9580
    'Quantum physics experiment...' → (57,0,39) top_sim=0.9688
    'Stock market analysis...' → (33,0,62) top_sim=0.9889
[01:13:53]   ◼ CELL Cell 12 — Interactive Exploration — PASS




[01:14:01] 
▶ CELL: Cell 13 — Results & Log
[01:14:01]   Started: 2026-03-21 01:14:01
# Results: Phase 2 — Resonance Field Core
Generated: 2026-03-21 01:13:19
Hardware: Tesla T4
Duration: 0:00:01.027706

## Component Status
Resonance Field Core: ✓ BUILT
Checkpoint: checkpoints/phase2.phase.pt ✓

## Test Results
| Test | Status | Score | Threshold | Pass? |
|------|--------|-------|-----------|-------|
| Energy never explodes (< init×10) | PASS | 4094.693603515625 | 40960.0 | ✓ |
| Energy never collapses (> init×0.01) | PASS | 1639.21044921875 | 40.96 | ✓ |
| Attractors grow monotonically | PASS | monotonic | monotonic | ✓ |
| Completes in < 120 seconds | PASS | 1.0s | 120s | ✓ |
| At least 5 attractors after 1000 steps | PASS | 2912 | 5 | ✓ |

## Demo Status
| Demo | Ran? | Quality |
|------|------|---------|

## Key Metrics
- **total_time:** 1.0s
- **initial_energy:** 4096.0
- **final_energy:** 1556.2
- **min_energy:** 1639.21
- **max_energy:** 4094.69
- **final_attractors:** 2912
- **attractor_history:** [912, 1594, 1901, 2129, 2241, 2367, 2523, 2682, 2849, 2912]
- **steps_per_second:** 979.6

## Phase 2 → Phase 3 Readiness
All tests passing: YES ✓
Checkpoint saved: YES ✓
Ready for Phase 3: YES ✓

============================================================
FULL PHASE 2 LOG:
============================================================
[00:58:54] ============================================================
[00:58:54] FLUX Phase 2 Log
[00:58:54] Started: 2026-03-21 00:58:54
[00:58:54] ============================================================

[00:58:54] 
──────────────────── Phase 2: Resonance Field Core ────────────────────
[00:58:54] 
▶ CELL: Cell 3 — Hardware & Secrets
[00:58:54]   Started: 2026-03-21 00:58:54
[00:58:54]   ℹ device: cuda
[00:58:54]   ℹ platform: Linux-6.6.113+-x86_64-with-glibc2.35
[00:58:54]   ℹ python: 3.12.12
[00:58:54]   ℹ torch: 2.10.0+cu128
[00:58:54]   ℹ gpu: Tesla T4
[00:58:54]   ℹ gpu_memory: 15.6 GB
[00:58:54]   ℹ cuda: 12.8
[00:58:54]   ✓ HuggingFace token loaded
[00:58:54]   ◼ CELL Cell 3 — Hardware & Secrets — PASS
[00:58:54] 
▶ CELL: Cell 4 — Load Phase 1 & Smoke Test
[00:58:54]   Started: 2026-03-21 00:58:54
[01:02:28] 
▶ CELL: Cell 4 — Load Phase 1 & Smoke Test
[01:02:28]   Started: 2026-03-21 01:02:28
[01:02:30] 
▶ CELL: Cell 4 — Load Phase 1 & Smoke Test
[01:02:30]   Started: 2026-03-21 01:02:30
[01:04:21] 
▶ CELL: Cell 4 — Load Phase 1 & Smoke Test
[01:04:21]   Started: 2026-03-21 01:04:21
[01:04:23]   ✓ CSE loaded: 1,337,264 params
[01:04:24]   ✓ CSE smoke test passed: wave shape torch.Size([27, 432])
[01:04:24]   ✓ ResonanceField smoke: perturb→(3,2,2), query→torch.Size([8, 64])
[01:04:24]   ◼ CELL Cell 4 — Load Phase 1 & Smoke Test — PASS
[01:04:41] 
▶ CELL: Cell 5 — Training
[01:04:41]   Started: 2026-03-21 01:04:41
[01:04:42]   ℹ Field: 222,995 params, 134,742,016 buffers (~539 MB)
[01:04:46]   ✓ Loaded 200 texts from WikiText-2
[01:04:47]   ✓ Encoded 100 texts → torch.Size([100, 432])
[01:04:50]   ℹ Warmup 0/500: loss_sim=0.0000 loss_div=-0.0190
[01:04:50]   ℹ Warmup 100/500: loss_sim=0.0000 loss_div=-0.3984
[01:04:51]   ℹ Warmup 200/500: loss_sim=0.0000 loss_div=-0.5866
[01:04:51]   ℹ Warmup 300/500: loss_sim=0.0000 loss_div=-0.6601
[01:04:51]   ℹ Warmup 400/500: loss_sim=0.0000 loss_div=-0.7471
[01:04:51]   ✓ Projection warmup complete
[01:04:53]   ℹ Rep 1/20: attractors=0, energy=255047.9, max_mass=0.0173
[01:05:02]   ℹ Rep 6/20: attractors=5, energy=234642.7, max_mass=0.1037
[01:05:11]   ℹ Rep 11/20: attractors=1007, energy=222016.4, max_mass=0.1902
[01:05:20]   ℹ Rep 16/20: attractors=3887, energy=212217.3, max_mass=0.2766
[01:05:28]   ℹ Rep 20/20: attractors=6500, energy=205546.1, max_mass=0.3457
[01:05:36]   ✓ Field formation complete: 1000 perturbations
[01:05:37]   📊 retrieval_accuracy: 1.0000
[01:05:37]   📊 avg_similarity: 0.9997
[01:05:37]   📊 num_attractors: 6500
[01:05:37]   📊 training_time: 55.6s
[01:05:37]   ◼ CELL Cell 5 — Training — PASS
[01:12:32] 
▶ CELL: Cell 6 — Save & Upload Checkpoint
[01:12:32]   Started: 2026-03-21 01:12:32
[01:12:45]   ◼ CELL Cell 6 — Save & Upload Checkpoint — PASS
[01:13:02] 
▶ CELL: Cell 7 — Test 1: Attractor Formation & No-Forgetting
[01:13:02]   Started: 2026-03-21 01:13:02
[01:13:02]   ◼ CELL Cell 7 — Test 1 — PASS
[01:13:10] 
▶ CELL: Cell 8 — Test 2: Local Update Verification
[01:13:10]   Started: 2026-03-21 01:13:10
[01:13:10]   ◼ CELL Cell 8 — Test 2 — PASS
[01:13:18] 
▶ CELL: Cell 9 — Test 3: Field Stability
[01:13:18]   Started: 2026-03-21 01:13:18
[01:13:19]   ◼ CELL Cell 9 — Test 3 — PASS
[01:13:24] 
▶ CELL: Cell 10 — Demo 1: Attractor Formation Visualization
[01:13:24]   Started: 2026-03-21 01:13:24
[01:13:27]   ◼ CELL Cell 10 — Demo 1 — PASS
[01:13:39] 
▶ CELL: Cell 11 — Demo 2: No-Forgetting Demo
[01:13:39]   Started: 2026-03-21 01:13:39
[01:13:41]   ◼ CELL Cell 11 — Demo 2 — PASS
[01:13:50] 
▶ CELL: Cell 12 — Interactive Exploration
[01:13:50]   Started: 2026-03-21 01:13:50
[01:13:50]   ◼ CELL Cell 12 — Interactive Exploration — PASS
[01:13:53] 
▶ CELL: Cell 12 — Interactive Exploration
[01:13:53]   Started: 2026-03-21 01:13:53
[01:13:53]   ◼ CELL Cell 12 — Interactive Exploration — PASS
[01:14:01] 
▶ CELL: Cell 13 — Results & Log
[01:14:01]   Started: 2026-03-21 01:14:01

[01:14:01]   ◼ CELL Cell 13 — Results & Log — PASS