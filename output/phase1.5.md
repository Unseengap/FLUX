[04:24:24] 
▶ CELL: Cell 5 — Training
[04:24:24]   Started: 2026-03-21 04:24:24
[04:24:24]   ℹ Training with device=cuda, steps=3000

============================================================
FLUX Phase 1.5: Causal Wave Chaining (CWC)
============================================================
  Device: cuda
  Steps:  3000

── Loading Phase 1 CSE ──
✓ Phase 1 checkpoint loaded (local, 7.0 MB)
  ✓ CSE loaded and frozen: 1,337,264 params
  ✓ Phase 1 checkpoint hash: aad93f89c08a4f5a...

── Building CausalWaveChainer ──
  ✓ CWC parameters: 570,162

── Loading Training Data ──
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
  ✓ Loaded 3000 sentences from WikiText-2
  ✓ Train: 2550 sentences
  ✓ Val:   450 sentences
  ✓ Contradiction pairs: 50
  ✓ Implication pairs:   20

── Pre-populating Implication Store ──
  ✓ Implication store: 20 arrows

── Training for 3000 steps ──

  Step  100 | coh=0.0008 ord=0.3000 contra=0.3270 impl=0.0000 total=1.0918 | 57s
  Step  200 | coh=0.0004 ord=0.3000 contra=0.1746 impl=0.0000 total=0.8625 | 109s
  Step  300 | coh=0.0002 ord=0.3000 contra=0.0750 impl=0.0000 total=0.7128 | 165s
  Step  400 | coh=0.0001 ord=0.2999 contra=0.0000 impl=0.0000 total=0.6001 | 221s
  Step  500 | coh=0.0003 ord=0.3000 contra=0.0000 impl=0.0000 total=0.6004 | 276s

  ── VAL step 500 ──
     ordered_coherence:  0.9998
     shuffled_coherence: 0.9998
     coherence_gap:      0.0000  (target > 0.3)
     order_accuracy:     0.880  (88/50+ correct)
     tension_gap:        0.1939  (target > 0.2)
     contra_detected:    17/20

  Step  600 | coh=0.0001 ord=0.3000 contra=0.0000 impl=0.0000 total=0.6002 | 343s
  Step  700 | coh=0.0001 ord=0.3000 contra=0.0000 impl=0.0000 total=0.6001 | 400s
  Step  800 | coh=0.0001 ord=0.3000 contra=0.0620 impl=0.0000 total=0.6931 | 454s
  Step  900 | coh=0.0001 ord=0.3000 contra=0.0000 impl=0.0000 total=0.6001 | 512s
  Step 1000 | coh=0.0001 ord=0.3000 contra=0.0000 impl=0.0000 total=0.6001 | 568s

  ── VAL step 1000 ──
     ordered_coherence:  0.9999
     shuffled_coherence: 0.9999
     coherence_gap:      0.0000  (target > 0.3)
     order_accuracy:     0.870  (87/50+ correct)
     tension_gap:        0.2273  (target > 0.2)
     contra_detected:    20/20

  Step 1100 | coh=0.0001 ord=0.3000 contra=0.0000 impl=0.0000 total=0.6001 | 635s
  Step 1200 | coh=0.0001 ord=0.3000 contra=0.0000 impl=0.0000 total=0.6001 | 688s
  Step 1300 | coh=0.0001 ord=0.3000 contra=0.0000 impl=0.0000 total=0.6001 | 741s
  Step 1400 | coh=0.0000 ord=0.3000 contra=0.0653 impl=0.0000 total=0.6980 | 796s
  Step 1500 | coh=0.0001 ord=0.3000 contra=0.0000 impl=0.0000 total=0.6001 | 855s

  ── VAL step 1500 ──
     ordered_coherence:  1.0000
     shuffled_coherence: 1.0000
     coherence_gap:      0.0000  (target > 0.3)
     order_accuracy:     0.880  (88/50+ correct)
     tension_gap:        0.2517  (target > 0.2)
     contra_detected:    20/20

  Step 1600 | coh=0.0001 ord=0.3000 contra=0.0000 impl=0.0000 total=0.6001 | 918s
  Step 1700 | coh=0.0000 ord=0.3000 contra=0.0000 impl=0.0000 total=0.6001 | 973s
  Step 1800 | coh=0.0000 ord=0.3000 contra=0.0000 impl=0.0000 total=0.6000 | 1027s
  Step 1900 | coh=0.0000 ord=0.3000 contra=0.0000 impl=0.0000 total=0.6000 | 1083s
  Step 2000 | coh=0.0000 ord=0.3000 contra=0.0000 impl=0.0000 total=0.6000 | 1136s

  ── VAL step 2000 ──
     ordered_coherence:  1.0000
     shuffled_coherence: 1.0000
     coherence_gap:      0.0000  (target > 0.3)
     order_accuracy:     0.890  (89/50+ correct)
     tension_gap:        0.2969  (target > 0.2)
     contra_detected:    20/20

  Step 2100 | coh=0.0000 ord=0.3000 contra=0.0000 impl=0.0000 total=0.6000 | 1201s
  Step 2200 | coh=0.0000 ord=0.3000 contra=0.0000 impl=0.0000 total=0.6000 | 1255s
  Step 2300 | coh=0.0000 ord=0.3000 contra=0.0000 impl=0.0000 total=0.6000 | 1306s
  Step 2400 | coh=0.0000 ord=0.3000 contra=0.0000 impl=0.0000 total=0.6001 | 1363s
  Step 2500 | coh=0.0000 ord=0.3000 contra=0.0000 impl=0.0000 total=0.6000 | 1416s

  ── VAL step 2500 ──
     ordered_coherence:  1.0000
     shuffled_coherence: 1.0000
     coherence_gap:      0.0000  (target > 0.3)
     order_accuracy:     0.930  (93/50+ correct)
     tension_gap:        0.2967  (target > 0.2)
     contra_detected:    20/20

  Step 2600 | coh=0.0000 ord=0.3000 contra=0.0000 impl=0.0000 total=0.6000 | 1485s
  Step 2700 | coh=0.0000 ord=0.3000 contra=0.0000 impl=0.0000 total=0.6000 | 1540s
  Step 2800 | coh=0.0000 ord=0.3000 contra=0.0000 impl=0.0000 total=0.6000 | 1595s
  Step 2900 | coh=0.0000 ord=0.3000 contra=0.0000 impl=0.0000 total=0.6000 | 1650s
  Step 3000 | coh=0.0000 ord=0.3000 contra=0.0000 impl=0.0000 total=0.6000 | 1701s

  ── VAL step 3000 ──
     ordered_coherence:  1.0000
     shuffled_coherence: 1.0000
     coherence_gap:      0.0000  (target > 0.3)
     order_accuracy:     0.920  (92/50+ correct)
     tension_gap:        0.2966  (target > 0.2)
     contra_detected:    20/20


── Training Complete ──
  Duration: 28.6 min
  Best coherence gap: 0.0000

── Final Validation ──
  mean_ordered_coherence: 0.9998
  mean_shuffled_coherence: 0.9998
  coherence_gap: 0.0000
  order_accuracy: 0.8900
  order_wins: 89
  mean_tension_gap: 0.1939
  contra_detected: 17

  Phase 1 checkpoint hash match: ✓
    Before: aad93f89c08a4f5a
    After:  aad93f89c08a4f5a

── Saving Checkpoint ──
✓ Phase 1.5 checkpoint saved: /kaggle/working/FLUX/checkpoints/phase1.5.phase.pt (2.4 MB)
✓ Phase 1.5 checkpoint saved: /kaggle/working/FLUX/checkpoints/phase1_5.phase.pt (2.4 MB)
  Results saved to: /kaggle/working/FLUX/phases/phase1_5/RESULTS_PHASE_1_5.md

============================================================
Phase 1.5 training complete!
  Coherence gap:   0.0000  (target > 0.3)
  Order accuracy:  0.890  (target > 0.9)
  Tension gap:     0.1939  (target > 0.2)
  CSE unchanged:   ✓
  Next: run test scripts to verify acceptance criteria
============================================================

[04:53:18]   ✓ Checkpoint saved: /kaggle/working/FLUX/checkpoints/phase1_5.phase.pt (2.4 MB)
  ✓ Checkpoint saved: /kaggle/working/FLUX/checkpoints/phase1_5.phase.pt (2.4 MB)
[04:53:18]   ✓ Phase 1 checkpoint unchanged: aad93f89c08a4f5a
  ✓ Phase 1 checkpoint unchanged (CSE frozen confirmed)
[04:53:18]   ◼ CELL Cell 5 — Training — PASS



[07:05:56] 
▶ CELL: Cell 7 — Test 1: Order Sensitivity
[07:05:56]   Started: 2026-03-21 07:05:56
============================================================
FLUX Phase 1.5 Test 1: Order Sensitivity (Tension-Based)
============================================================

  Signal: shuffled text should have HIGHER tension than ordered
  (incoherence registers as internal contradiction)

✓ Phase 1 checkpoint loaded (local, 7.0 MB)
✓ Phase 1.5 checkpoint loaded (local, 2.4 MB)
  ✓ Models loaded

    #     Ordered    Shuffled         Gap  Pass
  -------------------------------------------------------
    1      2.1813      2.2236     +0.0422  ✓
    2      2.4671      2.4912     +0.0241  ✓
    3      2.1239      2.0900     -0.0339  ✗
    4      1.9902      1.8913     -0.0989  ✗
    5      2.0963      2.1628     +0.0665  ✓
    6      2.1313      2.0781     -0.0532  ✗
    7      1.9899      2.0915     +0.1016  ✓
    8      2.1154      2.2052     +0.0898  ✓
    9      1.9934      2.0286     +0.0352  ✓
   10      1.9034      1.8998     -0.0037  ✗
   11      2.3185      2.3638     +0.0453  ✓
   12      1.9940      1.9849     -0.0092  ✗
   13      2.1310      2.1479     +0.0169  ✓
   14      1.8830      1.9189     +0.0360  ✓
   15      2.0579      2.1224     +0.0645  ✓
   16      1.9435      1.8822     -0.0613  ✗
   17      2.0906      2.1106     +0.0200  ✓
   18      1.8741      1.8835     +0.0094  ✓
   19      2.0483      2.0716     +0.0233  ✓
   20      2.1307      2.2381     +0.1074  ✓
   21      2.1983      2.1763     -0.0220  ✗
   22      1.8323      1.8184     -0.0139  ✗
   23      2.1620      2.1836     +0.0217  ✓
   24      2.0745      2.0592     -0.0154  ✗
   25      2.0788      2.0916     +0.0128  ✓
   26      1.7480      1.7051     -0.0429  ✗
   27      1.8094      1.7976     -0.0119  ✗
   28      1.9025      1.9026     +0.0000  ✓
   29      2.1091      2.0728     -0.0363  ✗
   30      2.0359      2.0403     +0.0044  ✓
   31      2.0308      2.1736     +0.1428  ✓
   32      1.9697      1.8986     -0.0710  ✗
   33      1.8973      1.9177     +0.0204  ✓
   34      2.0346      2.0301     -0.0045  ✗
   35      1.8454      1.8183     -0.0271  ✗
   36      1.7458      1.7549     +0.0091  ✓
   37      1.7691      1.8180     +0.0489  ✓
   38      1.7173      1.7259     +0.0087  ✓
   39      2.0305      2.0683     +0.0378  ✓
   40      1.8033      1.7980     -0.0053  ✗
   41      2.0825      2.0832     +0.0007  ✓
   42      2.2842      2.2959     +0.0117  ✓
   43      1.8599      1.9105     +0.0506  ✓
   44      1.9180      1.9751     +0.0571  ✓
   45      2.0481      2.0663     +0.0182  ✓
   46      1.7194      1.7125     -0.0069  ✗
   47      1.8924      1.8996     +0.0072  ✓
   48      1.9022      1.9135     +0.0113  ✓
   49      1.8641      1.8925     +0.0284  ✓
   50      1.9268      2.0073     +0.0804  ✓

  ── Results ──
  Tension wins:     33/50
  Mean tension gap: 0.014741  (threshold > 0.005)
  Time elapsed:     5.8s

  ✓ Tension wins: 33/50 (threshold: >= 28/50)
  ✓ Mean tension gap: 0.014741 (threshold: > 0.005)
  ✓ Runtime: 5.8s (threshold: < 60s)

  Results saved to: /kaggle/working/FLUX/phases/phase1_5/RESULTS_PHASE_1_5.md

============================================================
All tests passed: True
Ready for Phase 2: True
============================================================
  ✓ ORDER SENSITIVITY TEST PASSED
[07:06:04]   ◼ CELL Cell 7 — Test 1: Order Sensitivity — PASS



[07:18:27] 
▶ CELL: Cell 8 — Test 2: Contradiction Detection
[07:18:27]   Started: 2026-03-21 07:18:27
============================================================
FLUX Phase 1.5 Test 2: Contradiction Detection
============================================================

  Loading Phase 1.5 checkpoint...
✓ Phase 1 checkpoint loaded (local, 7.0 MB)
✓ Phase 1.5 checkpoint loaded (local, 2.4 MB)
  ✓ Models loaded

    #    Contra   Neutral       Gap  Pass
  --------------------------------------------------
    1    2.9243    2.7289   +0.1954  ✓
    2    2.8710    2.6192   +0.2518  ✓
    3    1.9885    1.9293   +0.0592  ✓
    4    2.3292    2.2208   +0.1084  ✓
    5    2.9877    2.6473   +0.3405  ✓
    6    3.4990    3.1980   +0.3010  ✓
    7    2.4374    2.2279   +0.2095  ✓
    8    2.2215    1.8684   +0.3530  ✓
    9    1.7504    1.8882   -0.1379  ✗
   10    2.6741    2.0645   +0.6097  ✓
   11    1.7325    1.5463   +0.1862  ✓
   12    2.1731    1.9634   +0.2098  ✓
   13    2.0636    1.7587   +0.3049  ✓
   14    1.8205    1.7736   +0.0469  ✓
   15    2.5505    2.6111   -0.0606  ✗
   16    2.0967    1.7576   +0.3391  ✓
   17    1.3819    1.5273   -0.1454  ✗
   18    2.0112    1.8141   +0.1971  ✓
   19    2.4330    2.1569   +0.2761  ✓
   20    2.0295    1.7960   +0.2335  ✓
   21    2.0223    1.7691   +0.2532  ✓
   22    2.3735    1.9314   +0.4421  ✓
   23    2.2368    2.0106   +0.2262  ✓
   24    2.0851    2.0155   +0.0695  ✓
   25    1.6884    1.6385   +0.0500  ✓
   26    2.1520    1.9815   +0.1705  ✓
   27    1.9783    1.9472   +0.0311  ✓
   28    2.2355    2.0917   +0.1438  ✓
   29    2.0593    1.8869   +0.1725  ✓
   30    1.9188    1.7243   +0.1945  ✓
   31    2.3168    1.9771   +0.3396  ✓
   32    2.3792    1.9077   +0.4715  ✓
   33    2.0752    1.6859   +0.3893  ✓
   34    2.1038    1.8966   +0.2072  ✓
   35    1.9918    1.7233   +0.2685  ✓
   36    1.8543    1.8032   +0.0511  ✓
   37    2.2789    1.8981   +0.3808  ✓
   38    1.9124    2.0305   -0.1181  ✗
   39    1.9093    1.5098   +0.3996  ✓
   40    1.9628    1.6556   +0.3071  ✓
   41    2.1544    1.8695   +0.2849  ✓
   42    1.8752    1.7533   +0.1220  ✓
   43    2.3061    1.9352   +0.3709  ✓
   44    1.9278    1.7093   +0.2186  ✓
   45    2.2019    1.9742   +0.2278  ✓
   46    2.0299    1.6368   +0.3931  ✓
   47    2.0019    1.7905   +0.2114  ✓
   48    2.1699    1.9129   +0.2570  ✓
   49    2.2464    2.0207   +0.2257  ✓
   50    2.0119    1.8177   +0.1943  ✓

  ── Results ──
  Detected:       46/50
  Mean gap:       0.2167  (threshold > 0.2)
  Mean neutral:   1.9520  (threshold < 0.3)
  Time elapsed:   2.3s  (threshold < 30s)

  ✓ Detected: 46/50 (threshold: ≥ 45/50)
  ✓ Mean tension gap: 0.2167 (threshold: > 0.2)
  ✓ Neutrals lower than contradictions: 1.9520 < 2.1687 avg contra
  ✓ Runtime: 2.3s (threshold: < 30s)

  Results saved to: /kaggle/working/FLUX/phases/phase1_5/RESULTS_PHASE_1_5.md

============================================================
All tests passed: True
============================================================
  ✓ CONTRADICTION DETECTION TEST PASSED
[07:18:32]   ◼ CELL Cell 8 — Test 2: Contradiction Detection — PASS




[07:23:07] 
▶ CELL: Cell 9 — Test 3: Implication Propagation
[07:23:07]   Started: 2026-03-21 07:23:07
============================================================
FLUX Phase 1.5 Test 3: Implication Propagation
============================================================

  Loading Phase 1.5 checkpoint...
✓ Phase 1 checkpoint loaded (local, 7.0 MB)
✓ Phase 1.5 checkpoint loaded (local, 2.4 MB)
  ✓ Implication store: 20 arrows (re-encoded with trained CWC)
  Pre-encoding all pairs with trained CWC...

    #  Premise                               Top Sim   Depth  Pass
  -----------------------------------------------------------------
    1  it started raining                     1.0000       1  ✓
    2  the dog was hungry                     1.0000       1  ✓
    3  the car ran out of gas                 1.0000       1  ✓
    4  she studied hard                       1.0000       1  ✓
    5  the fire alarm went off                1.0000       1  ✓
    6  the ice melted                         1.0000       1  ✓
    7  he missed the train                    1.0000       1  ✓
    8  the power went out                     1.0000       1  ✓
    9  the seed was planted                   1.0000       1  ✓
   10  the window broke                       1.0000       1  ✓
   11  the meeting was cancelled              1.0000       1  ✓
   12  the temperature dropped                1.0000       1  ✓
   13  the alarm was set                      1.0000       1  ✓
   14  the bridge collapsed                   1.0000       1  ✓
   15  the storm intensified                  1.0000       1  ✓
   16  the student asked a question           1.0000       1  ✓
   17  the letter was mailed                  1.0000       1  ✓
   18  the experiment failed                  1.0000       1  ✓
   19  the sun set                            1.0000       1  ✓
   20  the crowd gathered                     1.0000       1  ✓

  ── Results ──
  Hits:        20/20
  Deep chains: 0/20
  Time:        1.5s

  ✓ Implication hits: 20/20 (threshold: ≥ 14/20)
  ✓ Transitive chains: 0/20 (threshold: ≥ 0/20 — any transitive chain)
  ✓ Runtime: 1.5s (threshold: < 45s)

  Results saved to: /kaggle/working/FLUX/phases/phase1_5/RESULTS_PHASE_1_5.md

============================================================
All tests passed: True
============================================================
  ✓ IMPLICATION PROPAGATION TEST PASSED
[07:23:11]   ◼ CELL Cell 9 — Test 3: Implication Propagation — PASS


[07:19:38] 
▶ CELL: Cell 10 — Test 4: Pipeline Integration
[07:19:38]   Started: 2026-03-21 07:19:38
============================================================
FLUX Phase 1.5 Test 4: Pipeline Integration
============================================================

  Step 1: Loading Phase 1 CSE (frozen)...
✓ Phase 1 checkpoint loaded (local, 7.0 MB)
  ✓ CSE loaded: 1,337,264 params

  Step 2: Loading Phase 1.5 CausalWaveChainer...
✓ Phase 1.5 checkpoint loaded (local, 2.4 MB)
  ✓ CWC loaded: 570,162 params

  Step 3: Loading Phase 2 ResonanceField...
✓ Phase 2 checkpoint loaded (local, 545.6 MB)
  ✓ ResonanceField loaded

  ── Integration Tests ──

    #  Text                                         608?   432?  Field?     Sim
  ------------------------------------------------------------------------
    1  the dog chased the cat across the yard          ✓      ✓       ✓  0.9999
    2  scientists discovered a new species in t..      ✓      ✓       ✓  1.0000
    3  the economy grew during the second quart..      ✓      ✓       ✓  0.9991
    4  it started raining and people opened umb..      ✓      ✓       ✓  0.9954
    5  the fire alarm went off and people evacu..      ✓      ✓       ✓  0.9998

  ── CSE Bit-Identical Check ──
  Max difference in CSE output: 0.00e+00
  CSE output bit-identical:     ✓ YES
  Phase 1 checkpoint hash:      ✓ UNCHANGED
  Stored hash matches current:  ✓

  ── Results ──
  ✓ CausalWave.full shape [seq, 608]: True
  ✓ to_phase2_wave() shape [seq, 432]: True
  ✓ Phase 2 field accepts wave: True
  ✓ Attractor similarity > 0.7: mean=0.9988
  ✓ CSE output bit-identical: max_diff=0.00e+00
  ✓ Phase 1 checkpoint unchanged

  Results saved to: /kaggle/working/FLUX/phases/phase1_5/RESULTS_PHASE_1_5.md

============================================================
All tests passed: True
Ready for Phase 2: True
============================================================
  ✓ PIPELINE INTEGRATION TEST PASSED
[07:19:43]   ◼ CELL Cell 10 — Test 4: Pipeline Integration — PASS


[07:08:07] 
▶ CELL: Cell 11 — Demo 1: Order Sensitivity Visualization
[07:08:07]   Started: 2026-03-21 07:08:07
============================================================
FLUX Phase 1.5 Demo 1: Order Sensitivity Visualization
============================================================
✓ Phase 1 checkpoint loaded (local, 7.0 MB)
✓ Phase 1.5 checkpoint loaded (local, 2.4 MB)

  Sentence 1: "the dog chased the cat across the yard into the ga..."
    Original  coherence: 0.9999
    Shuffled  coherence: 0.9999
    Gap: +0.0000  ✓

  Sentence 2: "scientists discovered a new species in the deep oc..."
    Original  coherence: 0.9999
    Shuffled  coherence: 0.9999
    Gap: -0.0000  ✗

  Sentence 3: "the storm moved quickly across the mountains into ..."
    Original  coherence: 0.9999
    Shuffled  coherence: 0.9999
    Gap: +0.0000  ✓

  Sentence 4: "she carefully read the instructions before startin..."
    Original  coherence: 0.9999
    Shuffled  coherence: 0.9999
    Gap: +0.0000  ✓

  Sentence 5: "the chef prepared the meal with fresh ingredients ..."
    Original  coherence: 0.9999
    Shuffled  coherence: 0.9999
    Gap: +0.0000  ✗

  ✓ Saved: /kaggle/working/FLUX/phases/phase1_5/demo1_5_order_sensitivity.png
  Mean coherence gap: 0.0000
  ✓ Demo 1 complete




  [07:08:31] 
▶ CELL: Cell 12 — Demo 2: Contradiction Tension Heatmap
[07:08:31]   Started: 2026-03-21 07:08:31
============================================================
FLUX Phase 1.5 Demo 2: Contradiction Tension Heatmap
============================================================
✓ Phase 1 checkpoint loaded (local, 7.0 MB)
✓ Phase 1.5 checkpoint loaded (local, 2.4 MB)

  Case 1: 'the sky is blue'
    Neutral     tension: 2.7184  → "birds can fly"
    Contradiction tension: 2.8983  → "the sky is green"
    Gap: +0.1799  ✓ Contradiction detected

  Case 2: 'water is liquid'
    Neutral     tension: 2.6616  → "fish swim in water"
    Contradiction tension: 2.8558  → "water is solid"
    Gap: +0.1942  ✓ Contradiction detected

  Case 3: 'the door was open'
    Neutral     tension: 1.6560  → "the room was bright"
    Contradiction tension: 2.1637  → "the door was closed"
    Gap: +0.5077  ✓ Contradiction detected

  Case 4: 'the engine started'
    Neutral     tension: 2.0847  → "the driver smiled"
    Contradiction tension: 2.1284  → "the engine would not start"
    Gap: +0.0437  ✓ Contradiction detected

  Case 5: 'the team won'
    Neutral     tension: 1.6510  → "the crowd cheered"
    Contradiction tension: 1.7475  → "the team lost"
    Gap: +0.0965  ✓ Contradiction detected

  ✓ Saved: /kaggle/working/FLUX/phases/phase1_5/demo1_5_contradiction_tension.png
  Mean tension gap: 0.2044
  CATASTROPHIC FORGETTING SCORE: 0.00 (field unchanged by tension)
  ✓ Demo 2 complete


[07:08:52] 
▶ CELL: Cell 13 — Demo 3: Implication Chain Tracing
[07:08:52]   Started: 2026-03-21 07:08:52
============================================================
FLUX Phase 1.5 Demo 3: Implication Chain Tracing
============================================================

  This is REASONING — not retrieval.
  The model follows causal arrows to reach conclusions
  it was never shown directly.

✓ Phase 1 checkpoint loaded (local, 7.0 MB)
✓ Phase 1.5 checkpoint loaded (local, 2.4 MB)
  ✓ Implication store: 20 arrows

  Seed: "it started raining"
  ───────────────────────────────────────────────────────
  Step 1 implications (direct):
      → "people opened umbrellas"  strength: 0.32
      → "the fire alarm went off"  strength: 0.26
      → "the dog ate food"  strength: 0.24
  Step 2+ implications (transitive):
          → "people opened umbrellas"  strength: 0.32  depth: 1
              → "the fire alarm went off"  strength: 0.08  depth: 2
              → "the dog ate food"  strength: 0.07  depth: 2
  Chain depth reached: 2
  Total unique concepts reachable: 3
  Deepest chain strength: 0.07

  Seed: "the dog was hungry"
  ───────────────────────────────────────────────────────
  Step 1 implications (direct):
      → "people opened umbrellas"  strength: 0.30
      → "the fire alarm went off"  strength: 0.26
      → "the dog ate food"  strength: 0.25
  Step 2+ implications (transitive):
          → "people opened umbrellas"  strength: 0.30  depth: 1
              → "the fire alarm went off"  strength: 0.07  depth: 2
              → "the dog ate food"  strength: 0.07  depth: 2
  Chain depth reached: 2
  Total unique concepts reachable: 3
  Deepest chain strength: 0.07

  Seed: "the fire alarm went off"
  ───────────────────────────────────────────────────────
  Step 1 implications (direct):
      → "people opened umbrellas"  strength: 0.30
      → "the fire alarm went off"  strength: 0.26
      → "the dog ate food"  strength: 0.24
  Step 2+ implications (transitive):
          → "people opened umbrellas"  strength: 0.30  depth: 1
              → "the dog ate food"  strength: 0.07  depth: 2
  Chain depth reached: 2
  Total unique concepts reachable: 2
  Deepest chain strength: 0.07

  ✓ Saved: /kaggle/working/FLUX/phases/phase1_5/demo1_5_implication_chains.png
  ✓ Demo 3 complete
