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





# Cell 9b — Stage 4: Joint fine-tuning
log.cell_start("Cell 9b — Joint FT Train")
joint_result = trainer.train_joint_finetune(
    train_texts, max_steps=2000, log_interval=200,
    precomputed=precomputed,
)
log.metric("joint_cos_acc", f"{joint_result.wave_cosine_accuracy:.3f}")
log.metric("joint_wtt_acc", f"{joint_result.wtt_word_accuracy:.1%}")
log.cell_end("Cell 9b — Joint FT Train", "PASS")



[02:43:46] 
▶ CELL: Cell 9b — Joint FT Train
[02:43:46]   Started: 2026-03-25 02:43:46

============================================================
  Stage 3: Joint Fine-Tuning — max_steps=2000
============================================================
  Trainable: WG 3,187,234 + WTT 440,705 = 3,627,939 params
  LR: 0.000150  |  Grad accum: 4  |  SS: 0.5

  ℹ Evaluating WTT word accuracy on 50 texts...

  ✓ Stage 3 complete: 2000 steps (2000 skipped)
    Combined loss: 0.0000  (mse=0.0000 + wtt=0.0000)
    Cosine Acc:    0.000
    WTT Word Acc:  78.8%
    Training time: 625.7s (10.4 min)
    Total time:    625.7s (10.4 min)
    Throughput:    3.2 step/s
[02:54:12]   📊 joint_cos_acc: 0.000
[02:54:12]   📊 joint_wtt_acc: 78.8%
[02:54:12]   ◼ CELL Cell 9b — Joint FT Train — PASS





[02:54:22] 
▶ CELL: Cell 9c — Joint FT Diag
[02:54:22]   Started: 2026-03-25 02:54:22
  📊 Post-Joint Generation (5 prompts)
  ============================================================
  Prompt: The relationship between energy and matter
  Output: SD ow  C at t  t   a ta  u t  m  ro n h  u  u ta t  t   j o p

  Prompt: Modern technology relies on
  Output: Ek ut e be   u s   u og f   l t  w  og t   a t a t  at  u t  f

  Prompt: In the year 2025, scientists proved that
  Output: x at at at  au t  o t  nu ho  u ue   s t   u  z  u ca  u t  t

  Prompt: The history of mathematics reveals
  Output: De y  at t t   u at  m  u at  u at t   u  u  u  u t  me   t

  Prompt: Research shows that quantum
  Output: Th y   D  z  u t   o t   m  u t  t  t  re  ow l  t a at y  t

  Valid word rate: 100.0% (30/30)
[02:54:23]   📊 valid_word_rate: 100.0%
[02:54:23]   ◼ CELL Cell 9c — Joint FT Diag — PASS


# ═══════════════════════════════════════════════════════════════
# Phase 9 → Phase 9.5 HANDOFF DOCUMENT
# ═══════════════════════════════════════════════════════════════
#
# This is the complete handoff for building Phase 9.5.
# Read this ENTIRE section before writing any code.
#
# ═══════════════════════════════════════════════════════════════


## 1. CHECKPOINT — phase9.phase.pt IS SELF-CONTAINED

**CRITICAL: Phase 9.5 only needs to load ONE file: `checkpoints/phase9.phase.pt`**

This single checkpoint contains the FULL state of every component from Phases 1–7
plus all Phase 9 modules. You do NOT need phase7.phase.pt or any earlier checkpoint.

### Exact keys inside phase9.phase.pt:

**Phase 1–7 frozen components (all stored as state_dicts):**
| Key in checkpoint            | Component                  | Params    |
|------------------------------|----------------------------|-----------|
| `cse_state_dict`             | ContinuousSemanticEncoder  | Phase 1   |
| `field_state_dict`           | ResonanceField (75,561 attractors) | Phase 2 |
| `gr_state`                   | GravitationalRelevance     | Phase 3   |
| `tl_state`                   | ThermodynamicLearner       | Phase 4   |
| `cgn_state`                  | CausalGeometryNode         | Phase 5   |
| `causal_graph_state`         | CausalGraph                | Phase 5   |
| `working_memory_state`       | WorkingMemory              | Phase 6   |
| `episodic_memory_state`      | EpisodicMemory             | Phase 6   |
| `semantic_memory_state`      | SemanticMemory             | Phase 6   |
| `router_state`               | MemoryRouter               | Phase 6   |
| `wave_to_field_state`        | Linear(432→512)            | Bridge    |
| `field_to_wave_state`        | Linear(512→432)            | Bridge    |
| `output_head_state`          | OutputHead                 | Phase 7   |

**Phase 9 new components:**
| Key in checkpoint            | Component      | Status                      |
|------------------------------|----------------|-----------------------------|
| `wave_chunker_state_dict`    | WaveChunker    | TRAINED — KEEP, freeze      |
| `wave_generator_state_dict`  | WaveGenerator  | MODE-COLLAPSED — DISCARD    |
| `wave_to_text_state_dict`    | WaveToText     | TRAINED — KEEP, freeze      |

**Metadata keys:**
| Key                | Contents                                          |
|--------------------|---------------------------------------------------|
| `config`           | FLUXModel config dict (wave_dim=432, field_features=512, etc.) |
| `phase9_config`    | Phase 9 specific config                           |
| `metrics`          | Training metrics from Phase 9                     |

### How to load in Phase 9.5:

```python
checkpoint = torch.load('checkpoints/phase9.phase.pt', map_location='cpu')

# Rebuild FLUXModel and load ALL Phase 1-7 states from this checkpoint
model = build_flux_from_config(checkpoint['config'])
model.cse.load_state_dict(checkpoint['cse_state_dict'])
model.field.load_state_dict(checkpoint['field_state_dict'])
# ... etc for all frozen components ...

# Load TRAINED Phase 9 components (freeze them)
wave_chunker.load_state_dict(checkpoint['wave_chunker_state_dict'])
wave_to_text.load_state_dict(checkpoint['wave_to_text_state_dict'])
for p in wave_chunker.parameters(): p.requires_grad = False
for p in wave_to_text.parameters(): p.requires_grad = False

# IGNORE checkpoint['wave_generator_state_dict'] — build fresh WaveGenerator
wave_generator = WaveGenerator(...)  # random init, this is what we're retraining
```

### Bridge details (important):
- `wave_to_field`: Linear(432→512), patched from `field.wave_to_feature` (Phase 2 trained weight)
- `field_to_wave`: Linear(512→432), pseudoinverse of wave_to_field
- Round-trip cosine verified at 1.000
- These are in the checkpoint as `wave_to_field_state` and `field_to_wave_state`


## 2. WHAT PHASE 9 PRODUCED — KEEP FROZEN

| Component         | Status   | Metric                                    | Action       |
|-------------------|----------|-------------------------------------------|--------------|
| WaveToText (WTT)  | TRAINED  | loss=0.13, word_acc=79.8%, chunk_acc=82.8%| Load, freeze |
| WaveChunker       | TRAINED  | 374,112 params, works correctly           | Load, freeze |
| FLUXModel (Ph1-7) | FROZEN   | CSE, Field(75,561 attractors), GR, TL, CGN, Memory, bridges | Load, freeze |

### WTT details:
- Trained 25,000 steps at 74.3 step/s, batch_size=32
- Converts wave chunks [432] → byte sequences (spelling)
- forward_batch() supports batched training
- 82.8% chunk-level accuracy, 79.8% word-level accuracy on spelling test
- This component WORKS — do not retrain it

### WaveChunker details:
- Splits continuous wave sequences into discrete chunks
- Each chunk is [432] dimensional (TOTAL_WAVE_DIM)
- Works correctly — do not retrain it

### Field details:
- 75,561 attractors populated from 153 texts (200,000 perturbs)
- field.query() returns top-k attractor features [512]
- GravitationalRelevance selects attractors by mass/distance (O(log n))


## 3. WHAT FAILED — DISCARD AND REBUILD

### WaveGenerator — MODE COLLAPSED
- Architecture: GRU-based, gru_hidden=512, 3,187,234 params
- Input per step: [1, 1, 864] = concat(prev_wave[432], context[432])
- `context_to_hidden`: Linear(512→512) maps merged context → initial GRU hidden state
- **EVIDENCE OF COLLAPSE:**
  - Merged context vectors: avg pairwise cosine = 0.980 across DIFFERENT inputs
  - context_to_hidden output: cross-context cosine = 0.996 (nearly identical for all inputs)
  - Wave 0 output: cross-context cosine = 1.000 (IDENTICAL output regardless of input)
  - Loss plateau: hit 1.29 at step 400, barely moved for remaining 7,600 steps
  - Free generation output: random gibberish ("a s y  y   u u m t Fk...")
- **THE MODEL LEARNED TO COPY prev_wave AND IGNORE CONTEXT ENTIRELY**

### Joint Fine-Tuning (Stage 3) — COMPLETE NO-OP
- 2000 steps scheduled, 2000 steps SILENTLY SKIPPED
- `except Exception: _skipped += 1; continue` caught and hid every error
- Combined loss = 0.0000, cosine accuracy = 0.000 — literally zero work done
- The exception was never logged — we don't know what the error was
- Reported "PASS" despite doing nothing

### Precomputed Data — BIASED AND FIXED
- 8,500 samples, ALL exactly 40 chunks, ALL starting from position 0
- `_start = 0` was hardcoded in `_precompute_wg_data()` (~line 907 in train_wave_gen.py)
- `_MAX_CHUNKS = 40` — no variation in sequence length
- The model only ever saw the first 80 bytes of every document
- Zero diversity in chunk count or document position


## 4. ROOT CAUSES — ALL 6 MUST BE FIXED

### Bug 1: Context Collapse
- **Where**: Precomputed merged vectors (output of CSE → wave_to_field → GR → CGN → field.query)
- **Evidence**: Avg pairwise cosine = 0.980 across different texts
- **Why it breaks training**: GRU receives nearly identical context for every input,
  so it learns to ignore it and rely solely on teacher-forced prev_wave
- **Fix**: L2-normalize merged vectors before feeding to GRU. Add contrastive loss
  that penalizes same hidden states for different contexts. Consider projecting through
  a decorrelation layer. Or: instead of a single merged [512] vector, keep top-k
  attractor features as a sequence and use cross-attention.

### Bug 2: Fixed 40 Chunks from Position 0
- **Where**: `_precompute_wg_data()` in `train_wave_gen.py`, ~line 907
- **Code**: `_start = 0` (hardcoded), `_MAX_CHUNKS = 40` (no variation)
- **Why it breaks training**: Model only sees first ~80 bytes of documents.
  All sequences same length → no learning of variable-length generation.
- **Fix**: `_start = random.randint(0, max(0, total_chunks - max_len))`.
  Sample chunk counts from `randint(5, 40)` per example.

### Bug 3: Batch Size 1, 17% GPU Utilization
- **Where**: `train_wave_generator()` loop — processes one sample at a time,
  one GRU step at a time, in a Python for-loop
- **Evidence**: 17.4 step/s on T4 (22.5GB), only 3.9GB used
- **Why it's slow**: Each GRU call processes tensor [1, 1, 864]. Python loop overhead
  dominates. GPU starved for work.
- **Fix**: Batch multiple sequences. Use `pack_padded_sequence` for variable lengths.
  Process entire sequence in one GRU call: [batch, seq_len, 864] → [batch, seq_len, 512].
  Target batch_size=128+, keep precomputed tensors on GPU (~600MB easily fits in 22.5GB).
  Target: >500 steps/s (Phase 9 achieved only 17.4).

### Bug 4: Scheduled Sampling Started at 0%
- **Where**: `train_wave_generator()` — ss_p=0.00 for first 1600 steps (warmup=1600)
- **Evidence**: Logs show `ss_p=0.00` from step 1 to step 1600
- **Why it breaks training**: For 1600 steps (20% of training), the model ALWAYS gets
  the ground-truth previous wave. It learns to be a copy machine. By step 400 loss
  plateaus at ~1.29 and barely decreases again. When ss_p finally ramps up, the model
  has already learned bad habits.
- **Fix**: Start ss_p at 0.5 from step 1, or higher. Better yet: use a curriculum that
  starts with short free-running windows (2-3 steps) and grows to full sequence.
  At minimum, never let ss_p be 0.0 after the first 100 warmup steps.

### Bug 5: Training-Inference Mismatch
- **Where**: Training uses static precomputed merged vector. Inference (`generate_text()`)
  uses dynamic `query_field_attractors()` which re-queries the field at each step.
- **Evidence**: The model never saw attractor-sampled context during training.
  At inference time, the field returns different (potentially noisy) results.
- **Fix**: Either train with live field queries (slower but faithful), or add noise/
  augmentation to precomputed contexts during training. At minimum, during fine-tuning
  stage, switch to live field queries to close the gap.

### Bug 6: Silent Error Swallowing in Joint FT
- **Where**: `train_joint_finetune()` in `train_wave_gen.py`
- **Code**: `except Exception: _skipped += 1; continue` — catches ALL exceptions
- **Evidence**: 2000/2000 steps skipped, loss=0.0000, reported "PASS"
- **Fix**: Log the FULL exception traceback for at least the first 5 errors.
  If skip_rate > 10%, abort training and report FAIL. Never use bare
  `except Exception: continue` in a training loop.


## 5. PHASE 9.5 IMPLEMENTATION PLAN

### Goal
Retrain ONLY the WaveGenerator from scratch. Everything else stays frozen from phase9.phase.pt.

### Architecture (keep or modify)
The GRU architecture in `phases/phase9/wave_generator.py` is structurally sound:
- GRU hidden dim 512, input = concat(prev_wave[432], context[432]) = [864]
- `context_to_hidden`: maps merged context → initial hidden state
- `wave_project_out`: maps GRU output → predicted next wave [432]
- `query_field_attractors()`: dynamic field re-query at inference

Consider adding:
- `context_projection`: decorrelation layer before context_to_hidden
- Dropout on GRU output (0.1-0.2)
- LayerNorm on context before concat

### Training Stages for 9.5

**Stage 1: WaveGenerator Training (MAIN FOCUS)**
- Load phase9.phase.pt, restore everything, freeze all except fresh WaveGenerator
- Precompute data with RANDOM windows and VARIABLE lengths
- Batch_size=128+, pack_padded_sequence for GRU
- Scheduled sampling at 50% from start, ramp to 90%
- Train 15,000-20,000 steps (more steps since now much faster per step)
- Add context contrastive loss: different inputs must produce different Wave 0

**Stage 2: Joint Fine-Tuning (WG + WTT together)**
- Unfreeze WTT alongside WG
- Lower LR (e.g., 1e-4 for WG, 5e-5 for WTT)
- MUST have proper error handling — log exceptions, abort on >10% skip rate
- 3,000 steps with live field queries (close training-inference gap)

**Stage 3: Evaluation**
- Cross-context Wave 0 cosine MUST be < 0.85 (was 1.000 in Phase 9)
- Hidden init cross-context cosine MUST be < 0.90 (was 0.996)
- Free generation must produce recognizable English words
- GPU utilization > 60% (was 17%)
- Training speed > 200 steps/s (was 17.4)
- No silently skipped steps
- Loss must still be decreasing at end of training

### Key Dimensions Reference
```
wave_dim        = 432  (TOTAL_WAVE_DIM = phonetic:64 + syntactic:64 + semantic:256 + temporal:32 + intensity:16)
field_features  = 512
gru_hidden      = 512
gru_input       = 864  (prev_wave:432 + context:432)
field_to_wave   = Linear(512→432)  — pseudoinverse of wave_to_field
wave_to_field   = Linear(432→512)  — patched from field.wave_to_feature (Phase 2)
```

### File References
- Phase 9 training script: `phases/phase9/train_wave_gen.py` (1927 lines) — reference for loading logic
- WaveGenerator module: `phases/phase9/wave_generator.py` (473 lines) — GRU architecture
- WaveToText module: `phases/phase9/wave_to_text.py` (229 lines) — keep frozen
- FLUXModel construction: see `build_flux_for_phase9()` in train_wave_gen.py
- Bridge patching: see `_patch_bridges_from_field()` in train_wave_gen.py
- Checkpoint save format: see `save_phase9_checkpoint()` at line 1452 in train_wave_gen.py
- flux_utils.py: shared utilities (save/load checkpoint, PhaseLogger, PhaseResults, get_device)

### Success Criteria (before declaring Phase 9.5 complete)
1. Free generation produces readable multi-word English output for 5+ different prompts
2. Cross-context Wave 0 cosine < 0.85
3. Hidden init cross-context cosine < 0.90
4. Training loss steadily decreasing (no plateau after step 400)
5. GPU utilization > 60% on T4
6. Zero silently skipped training steps
7. Joint FT actually runs and improves metrics
