[01:03:42] 
▶ CELL: Cell 6a — WTT Fresh
[01:03:42]   Started: 2026-03-25 01:03:42
✓ Phase 7 checkpoint loaded (local, 618.2 MB)
  ✓ Bridges patched from field.wave_to_feature (Phase 2 trained)
    Round-trip cosine: 1.000 (random was ~0.001)
  ✓ Loaded from phase7.phase.pt: CSE, Field, GR, TL, CGN, Memory, OutputHead, wave_to_field(patched), field_to_wave(pinv)
  ✓ Loaded FLUXModel from phase7.phase.pt (bridges patched from field)
  Phase 9 modules built:
    WaveChunker:      374,112 params
    WaveGenerator:  3,187,234 params
    WaveToText:       440,705 params
    Total new:      4,002,051 params
  ✓ Field projection: 432→512 (trained Phase 2)
  ✓ Bridge wave_to_field: 432→512 (patched from field)
  ✓ Bridge field_to_wave: 512→432 (pseudoinverse)
[01:03:44]   ◼ CELL Cell 6a — WTT Fresh — PASS
  ✓ Phase 9 trainer ready — all bridges verified


[01:03:50] 
▶ CELL: Cell 6b — WTT Train
[01:03:50]   Started: 2026-03-25 01:03:50

============================================================
  Stage 1: WaveToText Pre-Training — max_steps=25000, batch_size=32
  Training texts: 9,000  |  log_interval: 5000
============================================================
  ℹ Collecting (wave, word) pairs...
  WTT Step     1  loss=5.5608  chunks=32  (first step: 0.45s)
  WTT Step    10  loss=5.5161  chunks=320  [20.5 step/s]
  WTT Step   5000  loss=0.9669 →  chunks=160,000  [74.1 step/s, elapsed 67s, ETA ~270s]
[01:04:57]   📊 wtt_step_5000_loss: 0.9669
  WTT Step  10000  loss=0.3316 ↓  chunks=320,000  [74.2 step/s, elapsed 135s, ETA ~202s]
[01:06:05]   📊 wtt_step_10000_loss: 0.3316
  WTT Step  15000  loss=0.2287 ↓  chunks=480,000  [74.4 step/s, elapsed 202s, ETA ~134s]
[01:07:11]   📊 wtt_step_15000_loss: 0.2287
  WTT Step  20000  loss=0.1881 ↓  chunks=640,000  [74.5 step/s, elapsed 269s, ETA ~67s]
[01:08:18]   📊 wtt_step_20000_loss: 0.1881
  WTT Step  25000  loss=0.1647 ↓  chunks=800,000  [74.6 step/s, elapsed 335s, ETA ~0s]
[01:09:25]   📊 wtt_step_25000_loss: 0.1647

  ℹ Evaluating WTT word accuracy on 50 texts...

  ✓ Stage 1 complete: 25000 steps, 800,000 chunks
    Final loss:    0.1319
    Average loss:  0.3760
    Word accuracy: 79.8%
    Total time:    336.6s (5.6 min)
    Throughput:    74.3 step/s, 2377 chunks/s
[01:09:27]   📊 wtt_final_loss: 0.1319
[01:09:27]   📊 wtt_word_acc: 79.8%
[01:09:27]   ◼ CELL Cell 6b — WTT Train — PASS



[01:15:45] 
▶ CELL: Cell 6c — WTT Diag
[01:15:45]   Started: 2026-03-25 01:15:45

  WTT Diagnostic — Word Spelling Accuracy (in-context)
  ============================================================
  (Words encoded in carrier sentence, chunks extracted by byte offset)

  ✗ the                  → 'the ee'                  GT='the e'         (3 chunks)
  ✓ of                   → 'of'                      GT='of'            (1 chunks)
  ✗ and                  → 'an d'                    GT='and '          (2 chunks)
  ✓ is                   → 'This isis'               GT='This isis'     (4 chunks)
  ✓ in                   → 'inn '                    GT='inn '          (2 chunks)
  ✓ science              → 'science  sennce'         GT='science  sennce' (7 chunks)
  ✗ energy               → 'te engry ene'            GT='t energy ene'  (6 chunks)
  ✗ system               → 'ysstme'                  GT='system'        (3 chunks)
  ✓ quantum              → 'quantumnt'               GT='quantumnt'     (4 chunks)
  ✗ relationship         → 'relations pe'            GT='relationsp e'  (6 chunks)
  ✗ fundamental          → 'fundamenat lenee'        GT='fundamental ene' (8 chunks)
  ✗ efficiency           → 'fficiencyen'             GT='efficiencyen'  (5 chunks)
  ✗ xyz                  → 'xy z'                    GT='xyz '          (2 chunks)
  ✗ café                 → 'caQ7 J'                  GT='caf�� '        (3 chunks)
  ✗ naïve                → 'naave hoee'              GT='naïve e'       (3 chunks)

  Accuracy: 5/15 (33%)

  📊 Chunk-Level Accuracy (same as training eval, 50 texts)
  ------------------------------------------------------------
    ✓ GT='Po'            Pred='Po'           
    ✗ GT='rt'            Pred='tr'           
    ✓ GT='-a'            Pred='-a'           
    ✗ GT='u-'            Pred='-u'           
    ✓ GT='Pr'            Pred='Pr'           
    ✓ GT='in'            Pred='in'           
    ✓ GT='ce'            Pred='ce'           
    ✓ GT=', '            Pred=', '           
    ✓ GT='Ha'            Pred='Ha'           
    ✓ GT='it'            Pred='it'           

  Chunk-level accuracy: 414/500 (82.8%)
  (This matches the training eval methodology)
[01:15:47]   ◼ CELL Cell 6c — WTT Diag — PASS





[01:17:14] 
▶ CELL: Cell 7b — Field Pop
[01:17:14]   Started: 2026-03-25 01:17:14
  ℹ Populating field with chunk-level attractors (max 200,000)...
    Field attractors before: 3,228
    ... 1st perturb done in 0.36s
    ... 100/200,000 perturbs (0.1%)  [200/s, ETA 16.6 min]
    ... 200/200,000 perturbs (0.1%)  [313/s, ETA 10.6 min]
    ... 300/200,000 perturbs (0.1%)  [385/s, ETA 8.6 min]
    ... 400/200,000 perturbs (0.2%)  [435/s, ETA 7.7 min]
    ... 500/200,000 perturbs (0.2%)  [472/s, ETA 7.0 min]
    ... 600/200,000 perturbs (0.3%)  [501/s, ETA 6.6 min]
    ... 700/200,000 perturbs (0.4%)  [524/s, ETA 6.3 min]
    ... 800/200,000 perturbs (0.4%)  [542/s, ETA 6.1 min]
    ... 900/200,000 perturbs (0.4%)  [558/s, ETA 6.0 min]
    continuation .....
    ... 199,500/200,000 perturbs (99.8%)  [623/s, ETA 0.0 min]
    ... 199,600/200,000 perturbs (99.8%)  [623/s, ETA 0.0 min]
    ... 199,700/200,000 perturbs (99.9%)  [623/s, ETA 0.0 min]
    ... 199,800/200,000 perturbs (99.9%)  [623/s, ETA 0.0 min]
    ... 199,900/200,000 perturbs (100.0%)  [623/s, ETA 0.0 min]
    ... 200,000/200,000 perturbs (100.0%)  [623/s, ETA 0.0 min]
    ℹ Bulk settling field (50 steps)...
    ✓ Field settled in 0.0s
  ✓ Field populated in 324.0s
    Perturbs: 200,000
    Texts processed: 153  |  Errors: 0
    Attractors: 3,228 → 75,561 (+72,333)

  Population complete:
    Perturbs:    200,000
    Attractors:  3,228 → 75,561
    New:         +72,333
[01:22:38]   ◼ CELL Cell 7b — Field Pop — PASS





[01:22:58] 
▶ CELL: Cell 7c — Field Diag
[01:22:58]   Started: 2026-03-25 01:22:58
  Total attractors: 75,561

  Field Query Test (8 phrases)
  ======================================================================
  ✓ The future of artificial intelligence              sim=0.948  rt_cos=0.080
  ✓ Energy equals mass times the speed of light square sim=0.978  rt_cos=0.336
  ✓ Quantum mechanics describes the behavior of partic sim=0.872  rt_cos=0.027
  ✓ The cat sat on the mat                             sim=0.581  rt_cos=0.071
  ✓ Photosynthesis converts sunlight into chemical ene sim=0.976  rt_cos=0.121
  ✓ Neural networks learn from data through backpropag sim=0.873  rt_cos=0.010
  ✓ Water freezes at zero degrees Celsius              sim=0.906  rt_cos=0.006
  ✓ Democracy is a form of government                  sim=0.964  rt_cos=0.082

  Query hit rate: 8/8
  Avg round-trip cosine: 0.092
  Min round-trip cosine: 0.006
  ℹ Field round-trip cosine is low (0.092) — field may be sparse
    This is expected: attractors store their OWN features, not the query's.
[01:22:58]   ◼ CELL Cell 7c — Field Diag — PASS





[01:24:35] 
▶ CELL: Cell 8a — WG Fresh (Precompute)
[01:24:35]   Started: 2026-03-25 01:24:35
  ℹ Pre-computing frozen pipeline outputs for up to 8,500 samples...
    Pipeline: CSE → wave_to_field → GR → CGN → field.query → WaveChunker
    ... 1st sample done: 40 chunks in 0.38s
    ℹ Estimated total: ~53 min for 8,500 samples
    ... 50/8,500 samples (1%)  [3.4 sample/s, elapsed 15s, ETA 2458s]
    ... 100/8,500 samples (1%)  [3.4 sample/s, elapsed 29s, ETA 2462s]
    ... 150/8,500 samples (2%)  [3.6 sample/s, elapsed 42s, ETA 2318s]
    ... 200/8,500 samples (2%)  [3.6 sample/s, elapsed 56s, ETA 2310s]
    ... 250/8,500 samples (3%)  [3.6 sample/s, elapsed 70s, ETA 2311s]
    ... 300/8,500 samples (4%)  [3.6 sample/s, elapsed 83s, ETA 2272s]
    ... 350/8,500 samples (4%)  [3.6 sample/s, elapsed 98s, ETA 2289s]
    ... 400/8,500 samples (5%)  [3.6 sample/s, elapsed 112s, ETA 2268s]
    ... 450/8,500 samples (5%)  [3.6 sample/s, elapsed 126s, ETA 2247s]
    ... 500/8,500 samples (6%)  [3.6 sample/s, elapsed 138s, ETA 2214s]
    ... 1,000/8,500 samples (12%)  [3.6 sample/s, elapsed 276s, ETA 2071s]
    ... 1,500/8,500 samples (18%)  [3.6 sample/s, elapsed 415s, ETA 1935s]
    ... 2,000/8,500 samples (24%)  [3.6 sample/s, elapsed 555s, ETA 1802s]
    ... 2,500/8,500 samples (29%)  [3.6 sample/s, elapsed 696s, ETA 1671s]
    ... 3,000/8,500 samples (35%)  [3.6 sample/s, elapsed 839s, ETA 1539s]
    ... 3,500/8,500 samples (41%)  [3.6 sample/s, elapsed 980s, ETA 1401s]
    ... 4,000/8,500 samples (47%)  [3.6 sample/s, elapsed 1123s, ETA 1264s]
    ... 4,500/8,500 samples (53%)  [3.6 sample/s, elapsed 1265s, ETA 1125s]
    ... 5,000/8,500 samples (59%)  [3.6 sample/s, elapsed 1408s, ETA 986s]
    ... 5,500/8,500 samples (65%)  [3.5 sample/s, elapsed 1550s, ETA 845s]
    ... 6,000/8,500 samples (71%)  [3.5 sample/s, elapsed 1693s, ETA 706s]
    ... 6,500/8,500 samples (76%)  [3.5 sample/s, elapsed 1837s, ETA 565s]
    ... 7,000/8,500 samples (82%)  [3.5 sample/s, elapsed 1976s, ETA 424s]
    ... 7,500/8,500 samples (88%)  [3.5 sample/s, elapsed 2118s, ETA 282s]
    ... 8,000/8,500 samples (94%)  [3.5 sample/s, elapsed 2259s, ETA 141s]
    ... 8,500/8,500 samples (100%)  [3.5 sample/s, elapsed 2402s, ETA 0s]
  ✓ Pre-computed 8,500 samples in 2401.6s (40.0 min)
    Rate: 3.5 samples/s  |  Skipped: 0

  Precomputed: 8,500 samples
  Sample shape: merged=[512], target_waves=[40, 432], wave_vec=[432]
[02:04:36]   ◼ CELL Cell 8a — WG Fresh (Precompute) — PASS


[02:04:59] 
▶ CELL: Cell 8a Diag — Precompute Sanity
[02:04:59]   Started: 2026-03-25 02:04:59
  Total precomputed samples: 8,500

  📐 Shape Check (sample 0):
    merged:       [512]  dtype=torch.float32
    target_waves: [40, 432]  dtype=torch.float32
    wave_vec:     [432]  dtype=torch.float32

  📊 Target Waves Per Sample:
    Min:    40
    Max:    40
    Mean:   40.0
    Median: 40
    <3 waves: 0/8500 (0%)

  📊 Merged Context Diversity (first 20 samples):
    Pairwise cosine: avg=0.943  min=0.751  max=1.000
    ✓ Good diversity — contexts are distinguishable

  📊 Merged Context Norms:
    Mean: 1.171
    Std:  0.334
    Min:  0.407
    Max:  1.648
    ✓ All contexts have meaningful magnitude

  📊 Target Wave Quality (first 10 samples):
    Wave norms:  mean=2.911  std=0.440
    Consecutive cosine: mean=0.432  min=-0.106  max=0.986

  🔍 NaN/Inf Check (first 100 samples):
    NaN tensors: 0  |  Inf tensors: 0
    ✓ All clean
[02:04:59]   ◼ CELL Cell 8a Diag — Precompute Sanity — PASS

  ✓ Precomputed data looks healthy — ready for WG training










[02:07:36] 
▶ CELL: Cell 8b — WG Train
[02:07:36]   Started: 2026-03-25 02:07:36

============================================================
  WaveGenerator Training — max_steps=8000, precomputed=YES
============================================================
  ✓ Using 8,500 pre-computed samples (skipping pre-computation)
  ✓ Gradient check: 19/19 generator params trainable

  ℹ Starting WG training loop: 8,000 steps over 8,500 samples
    Scheduled sampling: 0%→50% (warmup=1600)
    Context loss weight: 2.0x on Wave 0
  WG Step     1/8000  loss=3.0638  cos_acc=-0.009  ctx_loss=2.0313  ss_p=0.00  (first step: 0.06s)
  WG Step    10/8000  loss=3.0602  [17.4 step/s]
  WG Step    200/8000  loss=2.3076  cos_acc=0.394  lr=0.000150  ss_p=0.00  [17.3 step/s, ETA 451s]
[02:07:48]   📊 wg_step_200_loss: 2.3076
  WG Step    400/8000  loss=1.2949  cos_acc=0.673  lr=0.000300  ss_p=0.00  [17.3 step/s, ETA 438s]
[02:07:59]   📊 wg_step_400_loss: 1.2949
  WG Step    600/8000  loss=1.3532  cos_acc=0.680  lr=0.000300  ss_p=0.00  [17.4 step/s, ETA 426s]
[02:08:11]   📊 wg_step_600_loss: 1.3532
  WG Step    800/8000  loss=1.3374  cos_acc=0.683  lr=0.000300  ss_p=0.00  [17.4 step/s, ETA 414s]
[02:08:22]   📊 wg_step_800_loss: 1.3374
  WG Step   1000/8000  loss=1.3399  cos_acc=0.689  lr=0.000300  ss_p=0.00  [17.4 step/s, ETA 403s]
[02:08:34]   📊 wg_step_1000_loss: 1.3399
  WG Step   1200/8000  loss=1.3419  cos_acc=0.688  lr=0.000300  ss_p=0.00  [17.4 step/s, ETA 392s]
[02:08:46]   📊 wg_step_1200_loss: 1.3419
  WG Step   1400/8000  loss=1.3147  cos_acc=0.692  lr=0.000299  ss_p=0.00  [17.4 step/s, ETA 380s]
[02:08:57]   📊 wg_step_1400_loss: 1.3147
  WG Step   1600/8000  loss=1.3139  cos_acc=0.692  lr=0.000299  ss_p=0.00  [17.4 step/s, ETA 369s]
[02:09:09]   📊 wg_step_1600_loss: 1.3139
  WG Step   1800/8000  loss=1.2915  cos_acc=0.695  lr=0.000299  ss_p=0.02  [17.4 step/s, ETA 357s]
[02:09:20]   📊 wg_step_1800_loss: 1.2915
  WG Step   2000/8000  loss=1.3357  cos_acc=0.698  lr=0.000298  ss_p=0.03  [17.3 step/s, ETA 346s]
[02:09:32]   📊 wg_step_2000_loss: 1.3357
  WG Step   2200/8000  loss=1.3279  cos_acc=0.698  lr=0.000298  ss_p=0.05  [17.3 step/s, ETA 334s]
[02:09:43]   📊 wg_step_2200_loss: 1.3279
  WG Step   2400/8000  loss=1.2945  cos_acc=0.698  lr=0.000297  ss_p=0.06  [17.3 step/s, ETA 323s]
[02:09:55]   📊 wg_step_2400_loss: 1.2945
  WG Step   2600/8000  loss=1.3051  cos_acc=0.699  lr=0.000296  ss_p=0.08  [17.3 step/s, ETA 312s]
[02:10:06]   📊 wg_step_2600_loss: 1.3051
  WG Step   2800/8000  loss=1.3155  cos_acc=0.702  lr=0.000296  ss_p=0.09  [17.3 step/s, ETA 300s]
[02:10:18]   📊 wg_step_2800_loss: 1.3155
  WG Step   3000/8000  loss=1.2929  cos_acc=0.699  lr=0.000295  ss_p=0.11  [17.3 step/s, ETA 289s]
[02:10:29]   📊 wg_step_3000_loss: 1.2929
  WG Step   3200/8000  loss=1.2794  cos_acc=0.701  lr=0.000294  ss_p=0.12  [17.3 step/s, ETA 277s]
[02:10:41]   📊 wg_step_3200_loss: 1.2794
  WG Step   3400/8000  loss=1.3051  cos_acc=0.701  lr=0.000293  ss_p=0.14  [17.3 step/s, ETA 265s]
[02:10:53]   📊 wg_step_3400_loss: 1.3051
  WG Step   3600/8000  loss=1.3104  cos_acc=0.701  lr=0.000292  ss_p=0.16  [17.3 step/s, ETA 254s]
[02:11:04]   📊 wg_step_3600_loss: 1.3104
  WG Step   3800/8000  loss=1.3145  cos_acc=0.703  lr=0.000292  ss_p=0.17  [17.3 step/s, ETA 242s]
[02:11:16]   📊 wg_step_3800_loss: 1.3145
  WG Step   4000/8000  loss=1.3164  cos_acc=0.702  lr=0.000290  ss_p=0.19  [17.3 step/s, ETA 231s]
[02:11:27]   📊 wg_step_4000_loss: 1.3164
  WG Step   4200/8000  loss=1.3192  cos_acc=0.702  lr=0.000289  ss_p=0.20  [17.3 step/s, ETA 219s]
[02:11:39]   📊 wg_step_4200_loss: 1.3192
  WG Step   4400/8000  loss=1.2508  cos_acc=0.702  lr=0.000288  ss_p=0.22  [17.3 step/s, ETA 208s]
[02:11:50]   📊 wg_step_4400_loss: 1.2508
  WG Step   4600/8000  loss=1.3088  cos_acc=0.702  lr=0.000287  ss_p=0.23  [17.3 step/s, ETA 196s]
[02:12:02]   📊 wg_step_4600_loss: 1.3088
  WG Step   4800/8000  loss=1.3348  cos_acc=0.697  lr=0.000286  ss_p=0.25  [17.3 step/s, ETA 184s]
[02:12:13]   📊 wg_step_4800_loss: 1.3348
  WG Step   5000/8000  loss=1.3270  cos_acc=0.703  lr=0.000285  ss_p=0.27  [17.3 step/s, ETA 173s]
[02:12:25]   📊 wg_step_5000_loss: 1.3270
  WG Step   5200/8000  loss=1.3256  cos_acc=0.702  lr=0.000283  ss_p=0.28  [17.4 step/s, ETA 161s]
[02:12:36]   📊 wg_step_5200_loss: 1.3256
  WG Step   5400/8000  loss=1.3129  cos_acc=0.699  lr=0.000282  ss_p=0.30  [17.4 step/s, ETA 150s]
[02:12:48]   📊 wg_step_5400_loss: 1.3129
  WG Step   5600/8000  loss=1.3228  cos_acc=0.699  lr=0.000280  ss_p=0.31  [17.4 step/s, ETA 138s]
[02:12:59]   📊 wg_step_5600_loss: 1.3228
  WG Step   5800/8000  loss=1.3200  cos_acc=0.700  lr=0.000279  ss_p=0.33  [17.4 step/s, ETA 127s]
[02:13:11]   📊 wg_step_5800_loss: 1.3200
  WG Step   6000/8000  loss=1.3333  cos_acc=0.698  lr=0.000277  ss_p=0.34  [17.4 step/s, ETA 115s]
[02:13:22]   📊 wg_step_6000_loss: 1.3333
  WG Step   6200/8000  loss=1.3278  cos_acc=0.698  lr=0.000276  ss_p=0.36  [17.4 step/s, ETA 104s]
[02:13:34]   📊 wg_step_6200_loss: 1.3278
  WG Step   6400/8000  loss=1.3162  cos_acc=0.699  lr=0.000274  ss_p=0.38  [17.4 step/s, ETA 92s]
[02:13:45]   📊 wg_step_6400_loss: 1.3162
  WG Step   6600/8000  loss=1.2896  cos_acc=0.699  lr=0.000272  ss_p=0.39  [17.4 step/s, ETA 81s]
[02:13:57]   📊 wg_step_6600_loss: 1.2896
  WG Step   6800/8000  loss=1.3205  cos_acc=0.699  lr=0.000271  ss_p=0.41  [17.4 step/s, ETA 69s]
[02:14:08]   📊 wg_step_6800_loss: 1.3205
  WG Step   7000/8000  loss=1.3038  cos_acc=0.699  lr=0.000269  ss_p=0.42  [17.4 step/s, ETA 58s]
[02:14:20]   📊 wg_step_7000_loss: 1.3038
  WG Step   7200/8000  loss=1.2977  cos_acc=0.699  lr=0.000267  ss_p=0.44  [17.4 step/s, ETA 46s]
[02:14:31]   📊 wg_step_7200_loss: 1.2977
  WG Step   7400/8000  loss=1.3270  cos_acc=0.698  lr=0.000265  ss_p=0.45  [17.4 step/s, ETA 35s]
[02:14:43]   📊 wg_step_7400_loss: 1.3270
  WG Step   7600/8000  loss=1.3088  cos_acc=0.694  lr=0.000263  ss_p=0.47  [17.4 step/s, ETA 23s]
[02:14:54]   📊 wg_step_7600_loss: 1.3088
  WG Step   7800/8000  loss=1.3394  cos_acc=0.694  lr=0.000261  ss_p=0.48  [17.4 step/s, ETA 12s]
[02:15:06]   📊 wg_step_7800_loss: 1.3394
  WG Step   8000/8000  loss=1.3201  cos_acc=0.697  lr=0.000259  ss_p=0.50  [17.4 step/s, ETA 0s]
[02:15:17]   📊 wg_step_8000_loss: 1.3201

  ✓ Stage 2 complete: 8000 steps
    Final loss: 1.6563
    Avg cosine accuracy: 0.689
    Pre-compute time: 0.0s
    Training time: 460.8s
    Total time: 460.8s
[02:15:17]   📊 wg_final_loss: 1.6563
[02:15:17]   📊 wg_cos_acc: 0.689
[02:15:17]   ◼ CELL Cell 8b — WG Train — PASS
[02:20:36] 
▶ CELL: Cell 8c — WG Diag
[02:20:36]   Started: 2026-03-25 02:20:36
  📊 Teacher-Forced Prediction (5 samples)
  ============================================================
  Sample 0: 40 waves, avg cosine=0.678, range=[0.292, 0.924]
  Sample 1: 40 waves, avg cosine=0.722, range=[0.473, 0.868]
  Sample 2: 40 waves, avg cosine=0.757, range=[0.553, 0.952]
  Sample 3: 40 waves, avg cosine=0.718, range=[0.371, 0.887]
  Sample 4: 40 waves, avg cosine=0.673, range=[0.380, 0.936]

  Overall teacher-forced cosine: 0.710

  📊 Merged Context Diversity Check
  ============================================================
  merged[0] vs merged[1]: cosine=0.999
  merged[0] vs merged[2]: cosine=0.995
  merged[0] vs merged[3]: cosine=0.967
  merged[0] vs merged[4]: cosine=0.994
  merged[1] vs merged[2]: cosine=0.995
  merged[1] vs merged[3]: cosine=0.968
  merged[1] vs merged[4]: cosine=0.994
  merged[2] vs merged[3]: cosine=0.949
  merged[2] vs merged[4]: cosine=0.998
  merged[3] vs merged[4]: cosine=0.941
  Average merged cosine: 0.980 (lower = more diverse inputs)

  📊 Context Dependency Test
  ============================================================
  Wave 0 cross-context cosines: 1.000, 1.000, 1.000
  Average: 1.000 (lower = more context-dependent)
  ⚠ Generator may be ignoring context

  Hidden init cross-context: 0.997, 0.992, 0.998
  ⚠ Hidden init may have collapsed (avg 0.996)

  📊 Free Generation (3 prompts)
  ============================================================
  Prompt: The future of artificial intelligence
  Output: "a s y  y   u u m t Fk  u  u n h  u  u at t  t

  Prompt: Scientists have discovered
  Output: (r at e p be   u on  u t  u  a p  u t  t  t  e

  Prompt: In the beginning
  Output: zo s  Go  o ma r F t   u t   t ar m   u t  at

[02:20:37]   ◼ CELL Cell 8c — WG Diag — PASS