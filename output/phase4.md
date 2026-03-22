[04:59:15] 
▶ CELL: Cell 5 — Training / Load from Checkpoint
[04:59:15]   Started: 2026-03-22 04:59:15
[04:59:16]   ✓ ThermodynamicLearner + OnlineLearner built

=================================================================
  Stage A: One-Shot Fact Learning
=================================================================
  ✓ [ 1] energy: 0.2904 → 0.2873 (Δ=-0.0031)  temp=0.9950  surprise=0.2800
  ✗ [ 2] energy: 2.3158 → 2.3363 (Δ=+0.0204)  temp=0.9900  surprise=0.2331
  ✗ [ 3] energy: 1.7457 → 1.7559 (Δ=+0.0102)  temp=0.9851  surprise=0.3157
  ✓ [ 4] energy: 1.1129 → 1.1098 (Δ=-0.0031)  temp=0.9801  surprise=0.2827
  ✓ [ 5] energy: 0.3955 → 0.3908 (Δ=-0.0047)  temp=0.9752  surprise=0.3915
  ✓ [ 6] energy: 0.3749 → 0.3707 (Δ=-0.0041)  temp=0.9704  surprise=0.3663
  ✓ [ 7] energy: 0.4320 → 0.4294 (Δ=-0.0026)  temp=0.9655  surprise=0.2924
  ✓ [ 8] energy: 0.3214 → 0.3178 (Δ=-0.0036)  temp=0.9607  surprise=0.3172
  ✗ [ 9] energy: 2.7398 → 2.7586 (Δ=+0.0189)  temp=0.9559  surprise=0.2923
  ✗ [10] energy: 3.2459 → 3.2666 (Δ=+0.0207)  temp=0.9511  surprise=0.2818
[04:59:34]   📊 facts_stored: 6/10

  Facts stored (energy decreased): 6/10

=================================================================
  Stage B: Retention After 100 Distractor Updates
=================================================================
  Fact: 'The capital of Mars colony Alpha is New Houston'
  Similarity immediately after learning: 1.0000
  Similarity after 100 distractors: 0.9904
  Similarity drop: 0.0096
  Retained: True
[05:02:50]   📊 retention_sim_immediately: 1.0000
[05:02:50]   📊 retention_sim_after_100: 0.9904
[05:02:50]   📊 retention_passed: True

=================================================================
  Stage C: Temperature Annealing Over 200 Steps
=================================================================
    [  10] temp=0.5452  energy=1.4228  surprise=0.3315  stored=9/10
    [  20] temp=0.5186  energy=1.4325  surprise=0.3323  stored=8/10
    [  30] temp=0.4932  energy=1.4384  surprise=0.3328  stored=10/10
    [  40] temp=0.4691  energy=1.4438  surprise=0.3330  stored=10/10
    [  50] temp=0.4462  energy=1.4488  surprise=0.3332  stored=10/10
    [  60] temp=0.4244  energy=1.4534  surprise=0.3333  stored=10/10
    [  70] temp=0.4036  energy=1.4579  surprise=0.3334  stored=10/10
    [  80] temp=0.3839  energy=1.4623  surprise=0.3334  stored=10/10
    [  90] temp=0.3651  energy=1.4663  surprise=0.3335  stored=9/10
    [ 100] temp=0.3473  energy=1.4692  surprise=0.3336  stored=8/10
    [ 110] temp=0.3303  energy=1.4706  surprise=0.3338  stored=10/10
    [ 120] temp=0.3141  energy=1.4718  surprise=0.3339  stored=10/10

  Final temperature:    0.314146
  Steps completed:      231
  Is cold:              False
  Error trend:          -0.006747
[05:07:04]   📊 final_temperature: 0.314146

=================================================================
  Stage D: Thermodynamic Learning vs SGD Comparison
=================================================================
  Facts compared:       5
  TL time:              10.524s
  SGD time:             3.443s
  TL mean energy:       1.501908
  SGD mean energy:      1.271012
  TL faster:            False
  Speedup factor:       0.33x
[05:07:18]   📊 tl_vs_sgd_speedup: 0.33x

=================================================================
  Stage E: Verify No Global Gradients
=================================================================
  ✓ No global gradients found in field parameters
  ✓ All updates were local (through physics, not backprop)
[05:07:18]   📊 no_global_gradients: True
[05:07:18]   📊 training_time: 483.2s

  ✓ Training complete in 483.2s (8.1 min)
[05:07:18]   ◼ CELL Cell 5 — Training / Load from Checkpoint — PASS




[05:12:31] 
▶ CELL: Cell 7 — Test 1: Single-Shot Learning Retention
[05:12:31]   Started: 2026-03-22 05:12:31
============================================================
FLUX Phase 4 Test 1: Single-Shot Learning Retention
============================================================
✓ Phase 1 checkpoint loaded (local, 7.0 MB)
✓ Phase 2 checkpoint loaded (local, 545.6 MB)
  ✓ Models loaded

  Sub-test 1: One-shot learning (energy should decrease)...
    ✓ 'The capital of Mars colony Alpha is New ...'  energy: 0.2904 → 0.2873
    ✗ 'Water boils at 100 degrees Celsius at se...'  energy: 2.3158 → 2.3362
    ✗ 'The speed of light is approximately 3000...'  energy: 1.7456 → 1.7558
    ✓ 'Photosynthesis converts carbon dioxide i...'  energy: 1.1129 → 1.1097
    ✓ 'The deepest ocean trench is the Mariana ...'  energy: 0.3955 → 0.3907
  ✓ One-Shot Learning: 0.6 (threshold: 0.5)
  One-shot store rate: 60% (3/5)

  Sub-test 2: Retention after 100 distractors...
    ✓ 'The capital of Mars colony Alpha is New ...'  sim: 1.0000 → 0.9944 (drop: 0.0056)
    ✓ 'Water boils at 100 degrees Celsius at se...'  sim: 0.9972 → 0.9815 (drop: 0.0157)
    ✓ 'The speed of light is approximately 3000...'  sim: 0.9997 → 0.9946 (drop: 0.0051)
  ✓ Retention After 100 Updates: 0.9901522000630697 (threshold: 0.5)
  Retention rate: 100% (3/3)
  Avg similarity after distractors: 0.9902

==================================================
Phase 4 Results saved to: /kaggle/working/FLUX/phases/phase4/RESULTS_PHASE_4.md
All tests passed: True
Ready for Phase 5: True
==================================================

==================================================
All tests passed: True
==================================================
  ✓ SINGLE-SHOT LEARNING RETENTION TEST PASSED
[05:23:18]   ◼ CELL Cell 7 — Test 1: Single-Shot Learning Retention — PASS


[05:23:18] 
▶ CELL: Cell 8 — Test 2: No Global Gradient Required
[05:23:18]   Started: 2026-03-22 05:23:18
============================================================
FLUX Phase 4 Test 2: No Global Gradient Required
============================================================
✓ Phase 1 checkpoint loaded (local, 7.0 MB)
✓ Phase 2 checkpoint loaded (local, 545.6 MB)
  ✓ Models loaded

  Sub-test 1: Learn 20 facts, verify no gradients accumulated...
    ✓ Zero gradients in all field parameters after 20 facts
  ✓ No Global Gradients After Learning: 0.0 (threshold: 0.0)

  Sub-test 2: Verify torch.no_grad() context during settling...
    ✓ No gradient leaks — settling uses torch.no_grad() correctly
  ✓ No Gradient Leaks During Settle: 0.0 (threshold: 0.0)

  Sub-test 3: Verify field state changes without gradients...
    Field state delta: 2763.106934
    ✓ Field state changed (learning happened via physics, not gradients)
  ✓ Field State Changes Without Gradients: 2763.10693359375 (threshold: 0.0)

==================================================
Phase 4 Results saved to: /kaggle/working/FLUX/phases/phase4/RESULTS_PHASE_4.md
All tests passed: True
Ready for Phase 5: True
==================================================

==================================================
All tests passed: True
==================================================
  ✓ NO GLOBAL GRADIENT TEST PASSED
[05:24:06]   ◼ CELL Cell 8 — Test 2: No Global Gradient Required — PASS



[05:24:06] 
▶ CELL: Cell 9 — Test 3: Temperature Annealing Behavior
[05:24:06]   Started: 2026-03-22 05:24:06
============================================================
FLUX Phase 4 Test 3: Temperature Annealing Behavior
============================================================

  Sub-test 1: Base temperature decay...
    Initial: 0.9900
    After 200 steps: 0.1340
    Monotonically decreasing: True
  ✓ Base Temperature Decay: 0.13397967485796175 (threshold: 0.5)
    ✓ Temperature decays correctly

  Sub-test 2: High error causes heating...
    Temp before error spike: 0.1202
    Temp after error spike:  1.1202
    Heated up: True
  ✓ High Error Causes Heating: 1.0 (threshold: 0.0)
    ✓ Temperature rises on high error

  Sub-test 3: Consistently low error accelerates cooling...
    No-error temp after 100 steps: 0.183016
    Low-error temp after 100 steps: 0.073332
    Low error cooled faster: True
  ✓ Low Error Accelerates Cooling: 0.10968439982056291 (threshold: 0.0)
    ✓ Low error accelerates cooling

  Sub-test 4: Temperature stays in bounds...
    Final temp: 2.000000
    Bounds respected over 500 steps: True
  ✓ Temperature Bounds Respected: 1.0 (threshold: 1.0)
    ✓ Temperature always in [0.01, 2.0]

  Sub-test 5: Temperature anneals during actual learning...
✓ Phase 1 checkpoint loaded (local, 7.0 MB)
✓ Phase 2 checkpoint loaded (local, 545.6 MB)
    Initial temperature: 1.0000
    After 50 facts:     0.778313
    Temperature decreased: True
  ✓ Temperature Decrease During Learning: 0.221687442931358 (threshold: 0.0)
    ✓ Temperature annealed during real learning

==================================================
Phase 4 Results saved to: /kaggle/working/FLUX/phases/phase4/RESULTS_PHASE_4.md
All tests passed: True
Ready for Phase 5: True
==================================================

==================================================
All tests passed: False
==================================================
  ✗ Test failed — check sub-tests above
[05:25:54]   ◼ CELL Cell 9 — Test 3: Temperature Annealing Behavior — PASS






[05:26:29] 
▶ CELL: Cell 10 — Demo 1: One-Shot Fact Learning
[05:26:29]   Started: 2026-03-22 05:26:29
============================================================
FLUX Phase 4 Demo 1: One-Shot Fact Learning
============================================================
✓ Phase 1 checkpoint loaded (local, 7.0 MB)
✓ Phase 2 checkpoint loaded (local, 545.6 MB)
  ✓ 'The capital of Mars colony Alpha is New Houston...'
    energy: 0.2904 → 0.2873  temp=0.9950  surprise=0.2800
  ✗ 'Water boils at 100 degrees Celsius at sea level...'
    energy: 2.3158 → 2.3363  temp=0.9900  surprise=0.2331
  ✗ 'The speed of light is approximately 300000 km per ...'
    energy: 1.7457 → 1.7559  temp=0.9851  surprise=0.3157
  ✓ 'Photosynthesis converts carbon dioxide into oxygen...'
    energy: 1.1129 → 1.1098  temp=0.9801  surprise=0.2827
  ✓ 'The deepest ocean trench is the Mariana Trench...'
    energy: 0.3955 → 0.3908  temp=0.9752  surprise=0.3915
  ✓ 'DNA carries genetic information in all living orga...'
    energy: 0.3749 → 0.3707  temp=0.9704  surprise=0.3663
  ✓ 'The Earth orbits the Sun once every 365 days...'
    energy: 0.4320 → 0.4294  temp=0.9655  surprise=0.2924
  ✓ 'Gravity on the Moon is one sixth of Earth gravity...'
    energy: 0.3214 → 0.3178  temp=0.9607  surprise=0.3172
  ✗ 'The human brain contains approximately 86 billion ...'
    energy: 2.7398 → 2.7586  temp=0.9559  surprise=0.2923
  ✗ 'Sound travels faster in water than in air...'
    energy: 3.2459 → 3.2666  temp=0.9511  surprise=0.2818
  ✓ 'The Milky Way contains hundreds of billions of sta...'
    energy: 0.3435 → 0.3409  temp=0.9464  surprise=0.3162
  ✓ 'Antibiotics treat bacterial infections but not vir...'
    energy: 0.5681 → 0.5651  temp=0.9416  surprise=0.3012
  ✓ 'Lightning is a discharge of atmospheric electricit...'
    energy: 1.4118 → 1.4053  temp=0.9369  surprise=0.3895
  ✓ 'Tectonic plates move a few centimeters per year...'
    energy: 0.7609 → 0.7553  temp=0.9322  surprise=0.3457
  ✓ 'Oxygen makes up 21 percent of the atmosphere...'
    energy: 0.7423 → 0.7385  temp=0.9276  surprise=0.3366

── Querying facts back ──
  Query: 'The capital of Mars colony Alpha is New ...'  → similarity: 0.9996
  Query: 'Water boils at 100 degrees Celsius at se...'  → similarity: 0.9710
  Query: 'The speed of light is approximately 3000...'  → similarity: 0.9955
  Query: 'Photosynthesis converts carbon dioxide i...'  → similarity: 0.9771
  Query: 'The deepest ocean trench is the Mariana ...'  → similarity: 0.9999

  ✓ Saved: /kaggle/working/FLUX/phases/phase4/demo4_oneshot_learning.png
  ✓ Demo complete






  [05:27:46] 
▶ CELL: Cell 11 — Demo 2: TL vs SGD Comparison
[05:27:46]   Started: 2026-03-22 05:27:46
============================================================
FLUX Phase 4 Demo 2: Thermodynamic Learning vs SGD
============================================================
✓ Phase 1 checkpoint loaded (local, 7.0 MB)
✓ Phase 2 checkpoint loaded (local, 545.6 MB)

── Thermodynamic Learning (1 settle pass per fact) ──
  TL: The capital of Mars colony Alpha is New  energy=0.2873  time=2.0263s
  TL: Water boils at 100 degrees Celsius at se energy=2.3363  time=2.0205s
  TL: The speed of light is approximately 3000 energy=1.7559  time=2.0702s
  TL: Photosynthesis converts carbon dioxide i energy=1.1098  time=2.0907s
  TL: The deepest ocean trench is the Mariana  energy=0.3908  time=2.1262s
  TL: DNA carries genetic information in all l energy=0.3707  time=2.1599s
  TL: The Earth orbits the Sun once every 365  energy=0.4294  time=2.1886s
  TL: Gravity on the Moon is one sixth of Eart energy=0.3178  time=2.2325s
  TL: The human brain contains approximately 8 energy=2.7586  time=2.2594s
  TL: Sound travels faster in water than in ai energy=3.2666  time=2.2993s

── SGD Baseline (100 steps per fact) ──
  SGD (  1 steps): avg_energy=1.3671  avg_time=0.0004s
  SGD (  5 steps): avg_energy=1.2832  avg_time=0.0018s
  SGD ( 10 steps): avg_energy=1.3791  avg_time=0.0036s
  SGD ( 25 steps): avg_energy=1.3674  avg_time=0.0089s
  SGD ( 50 steps): avg_energy=1.3199  avg_time=0.0180s
  SGD (100 steps): avg_energy=1.3219  avg_time=0.0357s
  SGD (200 steps): avg_energy=1.2869  avg_time=0.0710s

  ✓ Saved: /kaggle/working/FLUX/phases/phase4/demo4_tl_vs_sgd.png
  ✓ Demo complete







