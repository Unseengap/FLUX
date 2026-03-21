[02:09:41]   ✓ Encoded 100 texts → torch.Size([100, 432])

  Phase A: Projection Warmup (500 steps)...
[02:09:41]   ℹ Warmup 0/500: loss_sim=0.0000  loss_div=1.8468  coord_std=[0.009,0.007,0.011]  coord_range=[0.041,0.029,0.048]
    Warmup 0/500: loss_sim=0.0000  loss_div=1.8468  coord_std=[0.009,0.007,0.011]  coord_range=[0.041,0.029,0.048]
[02:09:41]   ℹ Warmup 100/500: loss_sim=0.0000  loss_div=-1.2224  coord_std=[0.695,0.631,0.638]  coord_range=[1.963,1.816,1.977]
    Warmup 100/500: loss_sim=0.0000  loss_div=-1.2224  coord_std=[0.695,0.631,0.638]  coord_range=[1.963,1.816,1.977]
[02:09:42]   ℹ Warmup 200/500: loss_sim=0.0000  loss_div=-1.5827  coord_std=[0.792,0.740,0.786]  coord_range=[1.997,1.998,1.969]
    Warmup 200/500: loss_sim=0.0000  loss_div=-1.5827  coord_std=[0.792,0.740,0.786]  coord_range=[1.997,1.998,1.969]
[02:09:42]   ℹ Warmup 300/500: loss_sim=0.0000  loss_div=-1.7628  coord_std=[0.891,0.813,0.792]  coord_range=[1.999,2.000,1.996]
    Warmup 300/500: loss_sim=0.0000  loss_div=-1.7628  coord_std=[0.891,0.813,0.792]  coord_range=[1.999,2.000,1.996]
[02:09:43]   ℹ Warmup 400/500: loss_sim=0.0000  loss_div=-1.8351  coord_std=[0.921,0.888,0.806]  coord_range=[2.000,2.000,2.000]
    Warmup 400/500: loss_sim=0.0000  loss_div=-1.8351  coord_std=[0.921,0.888,0.806]  coord_range=[2.000,2.000,2.000]

  ── Post-warmup spatial spread check ──
  Coord std   (want > 0.3 each): [0.939, 0.874, 0.880]
  Coord range (want > 1.0 each): [2.000, 2.000, 2.000]
  ✓ All 3 axes well-spread — topology collapse resolved
[02:09:43]   ✓ Spatial spread OK: std=[0.9390402436256409, 0.8740364909172058, 0.8796557188034058]
[02:09:43]   ✓ Projection warmup complete
  ✓ wave_to_field_coords patched to use SpatialProjection

  Phase B: Field Formation (50 patterns × 20 reps)...
  Pattern location sample (first 10):
    Pattern  0 → (  4,   0,   0)
    Pattern  1 → ( 23,  10,  62)
    Pattern  2 → (  0,   0,  62)
    Pattern  3 → (  0,   0,  62)
    Pattern  4 → (  0,   0,  32)
    Pattern  5 → (  4,   5,  62)
    Pattern  6 → (  0,   0,  62)
    Pattern  7 → (  0,  61,  59)
    Pattern  8 → (  0,   0,  58)
    Pattern  9 → ( 23,  62,  62)
  Collision rate (first 10): 20%  ✓ OK
[02:09:43]   ℹ Pre-formation collision rate: 20%
[02:09:45]   ℹ Rep 1/20: attractors=7, energy=255075.2, max_mass=0.1309
    Rep 1/20: attractors=7, energy=255075.2, max_mass=0.1309
[02:09:56]   ℹ Rep 6/20: attractors=261, energy=234823.3, max_mass=0.7855
    Rep 6/20: attractors=261, energy=234823.3, max_mass=0.7855
[02:10:07]   ℹ Rep 11/20: attractors=622, energy=222370.7, max_mass=1.4400
    Rep 11/20: attractors=622, energy=222370.7, max_mass=1.4400
[02:10:18]   ℹ Rep 16/20: attractors=2009, energy=212762.4, max_mass=2.0946
    Rep 16/20: attractors=2009, energy=212762.4, max_mass=2.0946
[02:10:27]   ℹ Rep 20/20: attractors=3224, energy=206254.5, max_mass=2.6182
    Rep 20/20: attractors=3224, energy=206254.5, max_mass=2.6182
[02:10:37]   ✓ Field formation complete: 1000 perturbations

  Phase C: Validation...

  Final spatial spread (50 patterns):
  Unique locations: 37/50 (74%)
  ✓ Good spatial distribution — significant improvement over v2.0
[02:10:38]   ✓ Spatial distribution: 37/50 unique locations
[02:10:38]   📊 retrieval_accuracy: 1.0000
[02:10:38]   📊 avg_similarity: 0.9985
[02:10:38]   📊 num_attractors: 3224
[02:10:38]   📊 training_time: 59.2s
[02:10:38]   📊 unique_locations: 37/50

  ✓ Training complete in 59.2s (1.0 min)
  Retrieval accuracy: 50/50 = 100.0%
  Average similarity: 0.9985
  Attractors:         3224
  Discovered via scan: 15
[02:10:38]   ◼ CELL Cell 5 — Training — PASS



  ✓ Phase 2 checkpoint uploaded to HuggingFace Hub (545.6 MB)
    https://huggingface.co/UnseenGAP/FLUX
[02:10:50]   ◼ CELL Cell 6 — Save & Upload Checkpoint — PASS


[02:10:50] 
▶ CELL: Cell 7 — Test 1: Attractor Formation & No-Forgetting
[02:10:50]   Started: 2026-03-21 02:10:50
============================================================
  Phase 2 Test 1: Attractor Formation & No-Forgetting
============================================================
  Device: cuda
✓ Phase 1 checkpoint loaded (local, 7.0 MB)
  ✓ Using CSE-encoded wave vectors

  Step 1: Training Pattern A (20 repetitions)...
    Attractor A formed at (5, 7, 10), similarity: 0.9931
  Step 2: Training Pattern B (20 repetitions)...
    Attractor B formed at (5, 7, 10)
  Step 3: Training Pattern C (20 repetitions)...
    Attractor C formed at (5, 7, 10)

  Step 4: Verifying attractor persistence...
    Total: 3
    Alive: 3
    Dead:  0
    Survival rate: 100.00%
    ✓ Pattern A: similarity = 0.9731
    ✓ Pattern B: similarity = 0.9980
    ✓ Pattern C: similarity = 1.0000
  ✓ Attractor A survives B+C training: 0.9730842113494873 (threshold: 0.7)
  ✓ Attractor B survives C training: 0.9979535341262817 (threshold: 0.7)
  ✓ All 3 attractors coexist: 1.0 (threshold: 0.9)
  ✓ Forgetting score < 0.2: 0.0 (threshold: 0.2)
  ✓ Field has attractors: 729 (threshold: 3)

  ✓ ALL TESTS PASSED
  Forgetting score: 0.0000 (target: < 0.2)

==================================================
Phase 2 Results saved to: /kaggle/working/FLUX/phases/phase2/RESULTS_PHASE_2.md
All tests passed: True
Ready for Phase 3: True
==================================================
[02:10:50]   ◼ CELL Cell 7 — Test 1 — PASS







[02:10:50] 
▶ CELL: Cell 8 — Test 2: Local Update Verification
[02:10:50]   Started: 2026-03-21 02:10:50
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
    Center state norm: 0.123350
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
[02:10:50]   ◼ CELL Cell 8 — Test 2 — PASS






[02:10:50] 
▶ CELL: Cell 9 — Test 3: Field Stability
[02:10:50]   Started: 2026-03-21 02:10:50
============================================================
  Phase 2 Test 3: Field Stability Over 1000 Steps
============================================================
  Device: cuda

  Initial energy: 4096.0
  Ceiling (×10): 40960.0
  Floor (×0.01): 40.9600

  Running 1000 perturbations...
    Step 100: energy=3965.9, attractors=783, max_mass=0.7822, elapsed=0.1s
    Step 200: energy=3607.7, attractors=952, max_mass=1.5645, elapsed=0.3s
    Step 300: energy=3286.0, attractors=968, max_mass=2.3467, elapsed=0.4s
    Step 400: energy=3001.2, attractors=975, max_mass=3.1289, elapsed=0.5s
    Step 500: energy=2757.4, attractors=980, max_mass=3.9112, elapsed=0.7s
    Step 600: energy=2550.4, attractors=1016, max_mass=4.6934, elapsed=0.8s
    Step 700: energy=2374.1, attractors=1096, max_mass=5.4757, elapsed=0.9s
    Step 800: energy=2220.4, attractors=1160, max_mass=6.2579, elapsed=1.1s
    Step 900: energy=2083.0, attractors=1261, max_mass=7.0402, elapsed=1.2s
    Step 1000: energy=1957.5, attractors=1295, max_mass=7.8224, elapsed=1.3s

  Results:
    Total time: 1.3s
    Energy range: [1957.52, 4094.69]
    Energy exploded: False
    Energy collapsed: False
    Final attractors: 1295
    Attractors decreased at any point: False
    Attractor history: [783, 952, 968, 975, 980, 1016, 1096, 1160, 1261, 1295]
  ✓ Energy never explodes (< init×10): 4094.693603515625 (threshold: 40960.0)
  ✓ Energy never collapses (> init×0.01): 1957.51904296875 (threshold: 40.96)
  ✓ Attractors grow monotonically: monotonic (threshold: monotonic)
  ✓ Completes in < 120 seconds: 1.3s (threshold: 120s)
  ✓ At least 5 attractors after 1000 steps: 1295 (threshold: 5)

  ✓ ALL TESTS PASSED
  Field is STABLE ✓

==================================================
Phase 2 Results saved to: /kaggle/working/FLUX/phases/phase2/RESULTS_PHASE_2.md
All tests passed: True
Ready for Phase 3: True
==================================================
[02:10:51]   ◼ CELL Cell 9 — Test 3 — PASS




CELL: Cell 10 — Demo 1: Attractor Formation Visualization
[02:10:51]   Started: 2026-03-21 02:10:51
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
  Rep 15/15: attractors=729, energy=3136.7, max_mass=0.7168

  Generating visualization...
  ✓ Saved: /kaggle/working/FLUX/phases/phase2/demo1_attractor_formation.png

  Final field state:
    Attractors: 729
    Total energy: 2723.5
    Max mass: 0.7168
    State std: 0.025796

  Pattern locations:
    Science    → ( 6,  8,  9) mass=0.6173 energy=0.7746
    Music      → ( 5,  8,  9) mass=0.7168 energy=0.7651
    Sports     → ( 5,  8,  9) mass=0.7168 energy=0.7651
    Cooking    → ( 5,  8,  9) mass=0.7168 energy=0.7651
    History    → ( 5,  8,  9) mass=0.7168 energy=0.7651

  ✓ Demo complete


  CELL: Cell 11 — Demo 2: No-Forgetting Demo
[02:10:53]   Started: 2026-03-21 02:10:53
============================================================
  Phase 2 Demo 2: The No-Forgetting Demonstration
============================================================
  Device: cuda
✓ Phase 1 checkpoint loaded (local, 7.0 MB)
  ✓ Using CSE-encoded patterns

  Step 1: Training on Pattern A (20 repetitions)...
    ✓ Attractor A at (4, 7, 10), similarity: 0.9955

  Step 2: Training on Pattern B (20 repetitions)...
    ✓ Attractor B at (4, 7, 10)
    Pattern A still alive: True (sim=0.9829)

  Step 3: Training on Pattern C (20 repetitions)...

  Step 4: Querying for all patterns...
    Query 'A': similarity = 0.9838, location = (4, 7, 10)
    Query 'B': similarity = 0.9854, location = (4, 7, 10)
    Query 'C': similarity = 0.9880, location = (4, 7, 10)

  Step 5: Computing forgetting score...

  ==================================================
  CATASTROPHIC FORGETTING SCORE: 0.00
  Transformer baseline:          ~0.45
  FLUX advantage:                COMPLETE RETENTION ✓
  ==================================================

  Attractor Survival Report:
    Pattern A   : similarity = 0.9743  ✓ ALIVE
    Pattern B   : similarity = 0.9954  ✓ ALIVE
    Pattern C   : similarity = 1.0000  ✓ ALIVE
    Survival rate: 100%

  Generating visualization...
  ✓ Saved: /kaggle/working/FLUX/phases/phase2/demo2_no_forgetting.png

  ✓ Demo complete



[02:10:55] 
▶ CELL: Cell 12 — Interactive Exploration
[02:10:55]   Started: 2026-03-21 02:10:55
  Field Statistics:
    total_energy: 262144.0
    mean_energy: 1.0
    min_energy: 0.013158535584807396
    max_energy: 1.315853238105774
    energy_std: 0.3774619698524475
    total_mass: 1157.04541015625
    mean_mass: 0.0044137779623270035
    max_mass: 2.618227005004883
    active_locations: 10528
    active_fraction: 0.0401611328125
    state_mean_norm: 0.035156283527612686
    state_max_norm: 5.566548824310303
    state_std: 0.0065038930624723434
    num_attractors_01: 3224
    num_attractors_03: 274
    num_attractors_05: 193
    step_count: 1000

  Top 10 Attractors:
    #1: loc=(0, 0, 62) mass=2.6182 energy=0.0132
    #2: loc=(0, 1, 62) mass=2.2148 energy=0.0132
    #3: loc=(0, 0, 61) mass=2.1603 energy=0.0132
    #4: loc=(1, 0, 62) mass=2.1372 energy=0.0132
    #5: loc=(0, 1, 61) mass=2.0405 energy=0.0132
    #6: loc=(1, 1, 62) mass=2.0288 energy=0.0132
    #7: loc=(0, 0, 63) mass=2.0181 energy=0.0132
    #8: loc=(1, 0, 61) mass=1.9940 energy=0.0132
    #9: loc=(1, 1, 61) mass=1.9273 energy=0.0132
    #10: loc=(0, 1, 63) mass=1.9068 energy=0.0132

  Custom queries:
    'The cat sat on the mat...' → (4,0,1) top_sim=0.9512
    'Quantum physics experiment...' → (4,1,62) top_sim=0.9600
    'Stock market analysis...' → (0,0,62) top_sim=0.9872
[02:10:55]   ◼ CELL Cell 12 — Interactive Exploration — PASS



[02:10:55] 
▶ CELL: Cell 13 — Results & Log
[02:10:55]   Started: 2026-03-21 02:10:55
# Results: Phase 2 — Resonance Field Core
Generated: 2026-03-21 02:10:51
Hardware: Tesla T4
Duration: 0:00:01.348885

## Component Status
Resonance Field Core: ✓ BUILT
Checkpoint: checkpoints/phase2.phase.pt ✓

## Test Results
| Test | Status | Score | Threshold | Pass? |
|------|--------|-------|-----------|-------|
| Energy never explodes (< init×10) | PASS | 4094.693603515625 | 40960.0 | ✓ |
| Energy never collapses (> init×0.01) | PASS | 1957.51904296875 | 40.96 | ✓ |
| Attractors grow monotonically | PASS | monotonic | monotonic | ✓ |
| Completes in < 120 seconds | PASS | 1.3s | 120s | ✓ |
| At least 5 attractors after 1000 steps | PASS | 1295 | 5 | ✓ |

## Demo Status
| Demo | Ran? | Quality |
|------|------|---------|

## Key Metrics
- **total_time:** 1.3s
- **initial_energy:** 4096.0
- **final_energy:** 1876.5
- **min_energy:** 1957.52
- **max_energy:** 4094.69
- **final_attractors:** 1295
- **attractor_history:** [783, 952, 968, 975, 980, 1016, 1096, 1160, 1261, 1295]
- **steps_per_second:** 745.2

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
[01:14:18] 
▶ CELL: Cell 14 — Final Upload
[01:14:18]   Started: 2026-03-21 01:14:18
[01:14:20]   ◼ CELL Cell 14 — Final Upload — PASS
[01:14:35] 
▶ CELL: Cell 15 — Save Kaggle Artifacts
[01:14:35]   Started: 2026-03-21 01:14:35
[01:14:35]   ◼ CELL Cell 15 — Save Kaggle Artifacts — PASS
[01:43:01] 
──────────────────── Phase 2: Resonance Field Core ────────────────────
[01:43:01] 
▶ CELL: Cell 3 — Hardware & Secrets
[01:43:01]   Started: 2026-03-21 01:43:01
[01:43:01]   ℹ device: cuda
[01:43:01]   ℹ platform: Linux-6.6.113+-x86_64-with-glibc2.35
[01:43:01]   ℹ python: 3.12.12
[01:43:01]   ℹ torch: 2.10.0+cu128
[01:43:01]   ℹ gpu: Tesla T4
[01:43:01]   ℹ gpu_memory: 15.6 GB
[01:43:01]   ℹ cuda: 12.8
[01:43:01]   ✓ HuggingFace token loaded
[01:43:01]   ◼ CELL Cell 3 — Hardware & Secrets — PASS
[01:43:01] 
▶ CELL: Cell 4 — Load Phase 1 & Smoke Test
[01:43:01]   Started: 2026-03-21 01:43:01
[01:43:01]   ✓ CSE loaded: 1,337,264 params
[01:43:01]   ✓ CSE smoke test passed: wave shape torch.Size([27, 432])
[01:43:01]   ✓ ResonanceField smoke: perturb→(3,3,3), query→torch.Size([8, 64])
[01:43:01]   ◼ CELL Cell 4 — Load Phase 1 & Smoke Test — PASS
[01:43:01] 
▶ CELL: Cell 5 — Training
[01:43:01]   Started: 2026-03-21 01:43:01
[01:43:03]   ℹ Field: 222,995 params, 134,742,016 buffers (~539 MB)
[01:43:04]   ✓ Loaded 200 texts from WikiText-2
[01:43:04]   ✓ Encoded 100 texts → torch.Size([100, 432])
[01:43:04]   ℹ Warmup 0/500: loss_sim=0.0000 loss_div=-0.0246
[01:43:04]   ℹ Warmup 100/500: loss_sim=0.0000 loss_div=-0.4812
[01:43:04]   ℹ Warmup 200/500: loss_sim=0.0000 loss_div=-0.5727
[01:43:05]   ℹ Warmup 300/500: loss_sim=0.0000 loss_div=-0.7019
[01:43:05]   ℹ Warmup 400/500: loss_sim=0.0000 loss_div=-0.7442
[01:43:05]   ✓ Projection warmup complete
[01:43:07]   ℹ Rep 1/20: attractors=0, energy=255049.8, max_mass=0.0149
[01:43:19]   ℹ Rep 6/20: attractors=0, energy=234654.8, max_mass=0.0896
[01:43:30]   ℹ Rep 11/20: attractors=304, energy=222039.5, max_mass=0.1642
[01:43:41]   ℹ Rep 16/20: attractors=2271, energy=212251.5, max_mass=0.2389
[01:43:49]   ℹ Rep 20/20: attractors=4823, energy=205589.3, max_mass=0.2986
[01:43:59]   ✓ Field formation complete: 1000 perturbations
[01:43:59]   📊 retrieval_accuracy: 1.0000
[01:43:59]   📊 avg_similarity: 1.0000
[01:43:59]   📊 num_attractors: 4823
[01:43:59]   📊 training_time: 57.6s
[01:43:59]   ◼ CELL Cell 5 — Training — PASS
[01:43:59] 
▶ CELL: Cell 6 — Save & Upload Checkpoint
[01:43:59]   Started: 2026-03-21 01:43:59
[01:44:11]   ◼ CELL Cell 6 — Save & Upload Checkpoint — PASS
[01:44:11] 
▶ CELL: Cell 7 — Test 1: Attractor Formation & No-Forgetting
[01:44:11]   Started: 2026-03-21 01:44:11
[01:44:11]   ◼ CELL Cell 7 — Test 1 — PASS
[01:44:11] 
▶ CELL: Cell 8 — Test 2: Local Update Verification
[01:44:11]   Started: 2026-03-21 01:44:11
[01:44:11]   ◼ CELL Cell 8 — Test 2 — PASS
[01:44:11] 
▶ CELL: Cell 9 — Test 3: Field Stability
[01:44:11]   Started: 2026-03-21 01:44:11
[01:44:12]   ◼ CELL Cell 9 — Test 3 — PASS
[01:44:12] 
▶ CELL: Cell 10 — Demo 1: Attractor Formation Visualization
[01:44:12]   Started: 2026-03-21 01:44:12
[01:44:14]   ◼ CELL Cell 10 — Demo 1 — PASS
[01:44:14] 
▶ CELL: Cell 11 — Demo 2: No-Forgetting Demo
[01:44:14]   Started: 2026-03-21 01:44:14
[01:44:16]   ◼ CELL Cell 11 — Demo 2 — PASS
[01:44:16] 
▶ CELL: Cell 12 — Interactive Exploration
[01:44:16]   Started: 2026-03-21 01:44:16
[01:44:16]   ◼ CELL Cell 12 — Interactive Exploration — PASS
[01:44:16] 
▶ CELL: Cell 13 — Results & Log
[01:44:16]   Started: 2026-03-21 01:44:16
[01:44:16]   ◼ CELL Cell 13 — Results & Log — PASS
[01:44:16] 
▶ CELL: Cell 14 — Final Upload
[01:44:16]   Started: 2026-03-21 01:44:16
[01:44:18]   ◼ CELL Cell 14 — Final Upload — PASS
[01:44:18] 
▶ CELL: Cell 15 — Save Kaggle Artifacts
[01:44:18]   Started: 2026-03-21 01:44:18
[01:44:20]   ◼ CELL Cell 15 — Save Kaggle Artifacts — PASS
[01:48:04] 
▶ CELL: Cell 5 — Training
[01:48:04]   Started: 2026-03-21 01:48:04
[01:48:05]   ℹ Field: 222,995 params, 134,742,016 buffers (~539 MB)
[01:48:06]   ✓ Loaded 200 texts from WikiText-2
[01:48:07]   ✓ Encoded 100 texts → torch.Size([100, 432])
[01:49:29] 
▶ CELL: Cell 5 — Training
[01:49:29]   Started: 2026-03-21 01:49:29
[01:49:30]   ℹ Field: 249,603 params, 134,742,016 buffers (~539 MB)
[01:49:31]   ✓ Loaded 200 texts from WikiText-2
[01:49:31]   ✓ Encoded 100 texts → torch.Size([100, 432])
[01:49:31]   ℹ Warmup 0/500: loss_sim=0.0000  loss_div=0.4662  coord_std=[0.011,0.008,0.005]  coord_range=[0.044,0.036,0.018]
[01:49:32]   ℹ Warmup 100/500: loss_sim=0.0000  loss_div=-0.8015  coord_std=[0.001,0.745,0.007]  coord_range=[0.005,1.950,0.032]
[01:49:32]   ℹ Warmup 200/500: loss_sim=0.0000  loss_div=-0.8751  coord_std=[0.003,0.810,0.026]  coord_range=[0.010,1.997,0.094]
[01:49:32]   ℹ Warmup 300/500: loss_sim=0.0000  loss_div=-1.2352  coord_std=[0.003,0.807,0.817]  coord_range=[0.013,2.000,1.997]
[01:49:33]   ℹ Warmup 400/500: loss_sim=0.0000  loss_div=-1.3201  coord_std=[0.002,0.881,0.786]  coord_range=[0.014,2.000,2.000]
[01:49:33]   ℹ WARNING: Spatial collapse may persist. std=[0.0034824132453650236, 0.8330397605895996, 0.8981401324272156]
[01:49:33]   ✓ Projection warmup complete (v2.1)
[01:49:33]   ℹ Pre-formation collision rate: 10%
[01:49:35]   ℹ Rep 1/20: attractors=8, energy=255083.2, max_mass=0.1214
[01:49:47]   ℹ Rep 6/20: attractors=720, energy=234872.3, max_mass=0.7284
[01:49:59]   ℹ Rep 11/20: attractors=1312, energy=222462.2, max_mass=1.3354
[01:50:09]   ℹ Rep 16/20: attractors=1752, energy=212898.6, max_mass=1.9423
[01:50:17]   ℹ Rep 20/20: attractors=2015, energy=206427.8, max_mass=2.4279
[01:50:27]   ✓ Field formation complete: 1000 perturbations
[01:50:27]   ℹ WARNING: Only 29/50 unique locations
[01:50:27]   📊 retrieval_accuracy: 1.0000
[01:50:27]   📊 avg_similarity: 0.9960
[01:50:27]   📊 num_attractors: 2015
[01:50:27]   📊 training_time: 58.6s
[01:50:27]   📊 unique_locations: 29/50
[01:50:27]   ◼ CELL Cell 5 — Training — PASS
[02:01:55] 
──────────────────── Phase 2: Resonance Field Core ────────────────────
[02:01:55] 
▶ CELL: Cell 3 — Hardware & Secrets
[02:01:55]   Started: 2026-03-21 02:01:55
[02:01:55]   ℹ device: cuda
[02:01:55]   ℹ platform: Linux-6.6.113+-x86_64-with-glibc2.35
[02:01:55]   ℹ python: 3.12.12
[02:01:55]   ℹ torch: 2.10.0+cu128
[02:01:55]   ℹ gpu: Tesla T4
[02:01:55]   ℹ gpu_memory: 15.6 GB
[02:01:55]   ℹ cuda: 12.8
[02:01:55]   ✓ HuggingFace token loaded
[02:01:55]   ◼ CELL Cell 3 — Hardware & Secrets — PASS
[02:01:55] 
▶ CELL: Cell 4 — Load Phase 1 & Smoke Test
[02:01:55]   Started: 2026-03-21 02:01:55
[02:01:56]   ✓ CSE loaded: 1,337,264 params
[02:01:56]   ✓ CSE smoke test passed: wave shape torch.Size([27, 432])
[02:01:56]   ✓ ResonanceField smoke: perturb→(2,3,4), query→torch.Size([8, 64])
[02:01:56]   ◼ CELL Cell 4 — Load Phase 1 & Smoke Test — PASS
[02:01:56] 
▶ CELL: Cell 5 — Training
[02:01:56]   Started: 2026-03-21 02:01:56
[02:01:57]   ℹ Field: 305,027 params, 134,742,016 buffers (~539 MB)
[02:02:00]   ✓ Loaded 200 texts from WikiText-2
[02:02:00]   ✓ Encoded 100 texts → torch.Size([100, 432])
[02:02:02]   ℹ Warmup 0/500: loss_sim=0.0000 loss_div=-0.0049
[02:02:02]   ℹ Warmup 100/500: loss_sim=0.0000 loss_div=-0.3806
[02:02:02]   ℹ Warmup 200/500: loss_sim=0.0000 loss_div=-0.4287
[02:02:03]   ℹ Warmup 300/500: loss_sim=0.0000 loss_div=-0.4391
[02:02:03]   ℹ Warmup 400/500: loss_sim=0.0000 loss_div=-0.4661
[02:02:03]   ✓ Projection warmup complete
[02:02:05]   ℹ Rep 1/20: attractors=0, energy=255074.2, max_mass=0.0573
[02:02:14]   ℹ Rep 6/20: attractors=530, energy=234819.4, max_mass=0.3439
[02:02:24]   ℹ Rep 11/20: attractors=1233, energy=222364.7, max_mass=0.6305
[02:02:33]   ℹ Rep 16/20: attractors=2330, energy=212753.6, max_mass=0.9171
[02:02:40]   ℹ Rep 20/20: attractors=3391, energy=206244.1, max_mass=1.1464
[02:02:49]   ✓ Field formation complete: 1000 perturbations
[02:02:49]   📊 retrieval_accuracy: 1.0000
[02:02:49]   📊 avg_similarity: 0.9990
[02:02:49]   📊 num_attractors: 3391
[02:02:49]   📊 training_time: 53.4s
[02:02:49]   ◼ CELL Cell 5 — Training — PASS
[02:02:50] 
▶ CELL: Cell 6 — Save & Upload Checkpoint
[02:02:50]   Started: 2026-03-21 02:02:50
[02:03:03]   ◼ CELL Cell 6 — Save & Upload Checkpoint — PASS
[02:03:03] 
▶ CELL: Cell 7 — Test 1: Attractor Formation & No-Forgetting
[02:03:03]   Started: 2026-03-21 02:03:03
[02:03:03]   ◼ CELL Cell 7 — Test 1 — PASS
[02:03:03] 
▶ CELL: Cell 8 — Test 2: Local Update Verification
[02:03:03]   Started: 2026-03-21 02:03:03
[02:03:03]   ◼ CELL Cell 8 — Test 2 — PASS
[02:03:03] 
▶ CELL: Cell 9 — Test 3: Field Stability
[02:03:03]   Started: 2026-03-21 02:03:03
[02:03:04]   ◼ CELL Cell 9 — Test 3 — PASS
[02:03:04] 
▶ CELL: Cell 10 — Demo 1: Attractor Formation Visualization
[02:03:04]   Started: 2026-03-21 02:03:04
[02:03:06]   ◼ CELL Cell 10 — Demo 1 — PASS
[02:03:06] 
▶ CELL: Cell 11 — Demo 2: No-Forgetting Demo
[02:03:06]   Started: 2026-03-21 02:03:06
[02:03:08]   ◼ CELL Cell 11 — Demo 2 — PASS
[02:03:08] 
▶ CELL: Cell 12 — Interactive Exploration
[02:03:08]   Started: 2026-03-21 02:03:08
[02:03:08]   ◼ CELL Cell 12 — Interactive Exploration — PASS
[02:03:08] 
▶ CELL: Cell 13 — Results & Log
[02:03:08]   Started: 2026-03-21 02:03:08
[02:03:08]   ◼ CELL Cell 13 — Results & Log — PASS
[02:03:08] 
▶ CELL: Cell 14 — Final Upload
[02:03:08]   Started: 2026-03-21 02:03:08
[02:03:12]   ◼ CELL Cell 14 — Final Upload — PASS
[02:03:12] 
▶ CELL: Cell 15 — Save Kaggle Artifacts
[02:03:12]   Started: 2026-03-21 02:03:12
[02:03:13]   ◼ CELL Cell 15 — Save Kaggle Artifacts — PASS
[02:04:41] 
▶ CELL: Cell 5 — Training
[02:04:41]   Started: 2026-03-21 02:04:41
[02:04:43]   ℹ Field: 305,027 params, 134,742,016 buffers (~539 MB)
[02:04:44]   ✓ Loaded 200 texts from WikiText-2
[02:04:44]   ✓ Encoded 100 texts → torch.Size([100, 432])
[02:04:44]   ℹ Warmup 0/500: loss_sim=0.0000 loss_div=-0.0028
[02:04:45]   ℹ Warmup 100/500: loss_sim=0.0000 loss_div=-0.3795
[02:04:45]   ℹ Warmup 200/500: loss_sim=0.0000 loss_div=-0.4388
[02:04:45]   ℹ Warmup 300/500: loss_sim=0.0000 loss_div=-0.4442
[02:04:46]   ℹ Warmup 400/500: loss_sim=0.0000 loss_div=-0.4362
[02:04:46]   ✓ Projection warmup complete
[02:04:48]   ℹ Rep 1/20: attractors=0, energy=255076.1, max_mass=0.0548
[02:04:58]   ℹ Rep 6/20: attractors=559, energy=234830.2, max_mass=0.3291
[02:05:07]   ℹ Rep 11/20: attractors=1471, energy=222384.0, max_mass=0.6033
[02:05:18]   ℹ Rep 16/20: attractors=2599, energy=212781.3, max_mass=0.8776
[02:05:26]   ℹ Rep 20/20: attractors=3418, energy=206278.5, max_mass=1.0969
[02:05:37]   ✓ Field formation complete: 1000 perturbations
[02:05:37]   📊 retrieval_accuracy: 1.0000
[02:05:37]   📊 avg_similarity: 0.9988
[02:05:37]   📊 num_attractors: 3418
[02:05:37]   📊 training_time: 55.3s
[02:05:37]   ◼ CELL Cell 5 — Training — PASS
[02:07:06] 
▶ CELL: Cell 5 — Training
[02:07:06]   Started: 2026-03-21 02:07:06
[02:07:07]   ℹ Field: 305,027 params, 134,742,016 buffers (~539 MB)
[02:07:08]   ✓ Loaded 200 texts from WikiText-2
[02:07:09]   ✓ Encoded 100 texts → torch.Size([100, 432])
[02:07:09]   ℹ Warmup 0/500: loss_sim=0.0000  loss_div=1.8037  coord_std=[0.008,0.007,0.006]  coord_range=[0.034,0.023,0.023]
[02:07:09]   ℹ Warmup 100/500: loss_sim=0.0000  loss_div=-1.5319  coord_std=[0.800,0.707,0.707]  coord_range=[1.986,1.942,1.972]
[02:07:10]   ℹ Warmup 200/500: loss_sim=0.0000  loss_div=-1.7950  coord_std=[0.818,0.838,0.810]  coord_range=[1.999,1.999,1.998]
[02:07:10]   ℹ Warmup 300/500: loss_sim=0.0000  loss_div=-1.7416  coord_std=[0.777,0.854,0.787]  coord_range=[2.000,2.000,2.000]
[02:07:11]   ℹ Warmup 400/500: loss_sim=0.0000  loss_div=-1.9407  coord_std=[0.902,0.893,0.860]  coord_range=[2.000,1.999,2.000]
[02:07:11]   ✓ Spatial spread OK: std=[0.915019690990448, 0.9254288077354431, 0.9101656675338745]
[02:07:11]   ✓ Projection warmup complete
[02:07:11]   ℹ Pre-formation collision rate: 0%
[02:07:13]   ℹ Rep 1/20: attractors=0, energy=255076.8, max_mass=0.0400
[02:07:24]   ℹ Rep 6/20: attractors=487, energy=234832.1, max_mass=0.2400
[02:07:36]   ℹ Rep 11/20: attractors=1339, energy=222384.4, max_mass=0.4400
[02:07:47]   ℹ Rep 16/20: attractors=2338, energy=212777.8, max_mass=0.6400
[02:07:55]   ℹ Rep 20/20: attractors=3288, energy=206270.2, max_mass=0.8000
[02:08:05]   ✓ Field formation complete: 1000 perturbations
[02:08:05]   ✓ Spatial distribution: 40/50 unique locations
[02:08:05]   📊 retrieval_accuracy: 1.0000
[02:08:05]   📊 avg_similarity: 0.9989
[02:08:05]   📊 num_attractors: 3288
[02:08:05]   📊 training_time: 59.6s
[02:08:05]   📊 unique_locations: 40/50
[02:08:05]   ◼ CELL Cell 5 — Training — PASS
[02:09:38] 
──────────────────── Phase 2: Resonance Field Core ────────────────────
[02:09:38] 
▶ CELL: Cell 3 — Hardware & Secrets
[02:09:38]   Started: 2026-03-21 02:09:38
[02:09:38]   ℹ device: cuda
[02:09:38]   ℹ platform: Linux-6.6.113+-x86_64-with-glibc2.35
[02:09:38]   ℹ python: 3.12.12
[02:09:38]   ℹ torch: 2.10.0+cu128
[02:09:38]   ℹ gpu: Tesla T4
[02:09:38]   ℹ gpu_memory: 15.6 GB
[02:09:38]   ℹ cuda: 12.8
[02:09:38]   ✓ HuggingFace token loaded
[02:09:38]   ◼ CELL Cell 3 — Hardware & Secrets — PASS
[02:09:38] 
▶ CELL: Cell 4 — Load Phase 1 & Smoke Test
[02:09:38]   Started: 2026-03-21 02:09:38
[02:09:38]   ✓ CSE loaded: 1,337,264 params
[02:09:38]   ✓ CSE smoke test passed: wave shape torch.Size([27, 432])
[02:09:38]   ✓ ResonanceField smoke: perturb→(3,3,3), query→torch.Size([8, 64])
[02:09:38]   ◼ CELL Cell 4 — Load Phase 1 & Smoke Test — PASS
[02:09:38] 
▶ CELL: Cell 5 — Training
[02:09:38]   Started: 2026-03-21 02:09:38
[02:09:40]   ℹ Field: 305,027 params, 134,742,016 buffers (~539 MB)
[02:09:41]   ✓ Loaded 200 texts from WikiText-2
[02:09:41]   ✓ Encoded 100 texts → torch.Size([100, 432])
[02:09:41]   ℹ Warmup 0/500: loss_sim=0.0000  loss_div=1.8468  coord_std=[0.009,0.007,0.011]  coord_range=[0.041,0.029,0.048]
[02:09:41]   ℹ Warmup 100/500: loss_sim=0.0000  loss_div=-1.2224  coord_std=[0.695,0.631,0.638]  coord_range=[1.963,1.816,1.977]
[02:09:42]   ℹ Warmup 200/500: loss_sim=0.0000  loss_div=-1.5827  coord_std=[0.792,0.740,0.786]  coord_range=[1.997,1.998,1.969]
[02:09:42]   ℹ Warmup 300/500: loss_sim=0.0000  loss_div=-1.7628  coord_std=[0.891,0.813,0.792]  coord_range=[1.999,2.000,1.996]
[02:09:43]   ℹ Warmup 400/500: loss_sim=0.0000  loss_div=-1.8351  coord_std=[0.921,0.888,0.806]  coord_range=[2.000,2.000,2.000]
[02:09:43]   ✓ Spatial spread OK: std=[0.9390402436256409, 0.8740364909172058, 0.8796557188034058]
[02:09:43]   ✓ Projection warmup complete
[02:09:43]   ℹ Pre-formation collision rate: 20%
[02:09:45]   ℹ Rep 1/20: attractors=7, energy=255075.2, max_mass=0.1309
[02:09:56]   ℹ Rep 6/20: attractors=261, energy=234823.3, max_mass=0.7855
[02:10:07]   ℹ Rep 11/20: attractors=622, energy=222370.7, max_mass=1.4400
[02:10:18]   ℹ Rep 16/20: attractors=2009, energy=212762.4, max_mass=2.0946
[02:10:27]   ℹ Rep 20/20: attractors=3224, energy=206254.5, max_mass=2.6182
[02:10:37]   ✓ Field formation complete: 1000 perturbations
[02:10:38]   ✓ Spatial distribution: 37/50 unique locations
[02:10:38]   📊 retrieval_accuracy: 1.0000
[02:10:38]   📊 avg_similarity: 0.9985
[02:10:38]   📊 num_attractors: 3224
[02:10:38]   📊 training_time: 59.2s
[02:10:38]   📊 unique_locations: 37/50
[02:10:38]   ◼ CELL Cell 5 — Training — PASS
[02:10:38] 
▶ CELL: Cell 6 — Save & Upload Checkpoint
[02:10:38]   Started: 2026-03-21 02:10:38
[02:10:50]   ◼ CELL Cell 6 — Save & Upload Checkpoint — PASS
[02:10:50] 
▶ CELL: Cell 7 — Test 1: Attractor Formation & No-Forgetting
[02:10:50]   Started: 2026-03-21 02:10:50
[02:10:50]   ◼ CELL Cell 7 — Test 1 — PASS
[02:10:50] 
▶ CELL: Cell 8 — Test 2: Local Update Verification
[02:10:50]   Started: 2026-03-21 02:10:50
[02:10:50]   ◼ CELL Cell 8 — Test 2 — PASS
[02:10:50] 
▶ CELL: Cell 9 — Test 3: Field Stability
[02:10:50]   Started: 2026-03-21 02:10:50
[02:10:51]   ◼ CELL Cell 9 — Test 3 — PASS
[02:10:51] 
▶ CELL: Cell 10 — Demo 1: Attractor Formation Visualization
[02:10:51]   Started: 2026-03-21 02:10:51
[02:10:53]   ◼ CELL Cell 10 — Demo 1 — PASS
[02:10:53] 
▶ CELL: Cell 11 — Demo 2: No-Forgetting Demo
[02:10:53]   Started: 2026-03-21 02:10:53
[02:10:55]   ◼ CELL Cell 11 — Demo 2 — PASS
[02:10:55] 
▶ CELL: Cell 12 — Interactive Exploration
[02:10:55]   Started: 2026-03-21 02:10:55
[02:10:55]   ◼ CELL Cell 12 — Interactive Exploration — PASS
[02:10:55] 
▶ CELL: Cell 13 — Results & Log
[02:10:55]   Started: 2026-03-21 02:10:55

[02:10:55]   ◼ CELL Cell 13 — Results & Log — PASS