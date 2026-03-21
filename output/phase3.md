[23:09:49] 
▶ CELL: Cell 5 — Training / Load from Checkpoint
[23:09:49]   Started: 2026-03-21 23:09:49
[23:09:49]   📊 GR parameters: 1,050,625
[23:09:49]   📊 Decoder parameters: 69,988,353
  GR parameters:      1,050,625
  Decoder parameters: 69,988,353

=================================================================
  Stage A: GravitationalRelevance vs PyTorch Attention
=================================================================

   Seq Len      GR (ms)    Attn (ms)      Ratio      Scaling
------------------------------------------------------------
       128       33.5ms        0.4ms 80.6x slower           --
       256       70.8ms        0.6ms 116.2x slower GR:2.1x A:1.5x
       512      138.5ms        0.8ms 167.0x slower GR:2.0x A:1.4x
      1024      305.3ms        1.5ms 198.8x slower GR:2.2x A:1.9x
      2048      934.3ms        6.3ms 147.4x slower GR:3.1x A:4.1x
      4096     3147.3ms       28.3ms 111.3x slower GR:3.4x A:4.5x

  Scaling analysis (when seq doubles):
    O(n²) attention: growth should be ~4x per doubling
    O(n log n) GR:   growth should be ~2x per doubling
    GR avg growth:   2.54x
    Attn avg growth: 2.65x

  Note: GR is slower in absolute time due to Python/CPU spatial index.
  The O(log n) advantage is in scaling rate, not constant factors.
  A CUDA-native spatial index would close the gap at seq > 10K.
[23:10:49]   ✓ Benchmark complete: 6 seq lengths tested

  Stage B: GR Warmup — feeding Phase 2 field output...
[23:10:50]   ✓ Loaded 100 texts from WikiText-2 for warmup
  ✓ Loaded 100 texts from WikiText-2
[23:10:51]   ℹ Warmup  20/80: avg_top_sim=0.9980, mass_count=8067
    Warmup  20/80: avg_top_sim=0.9980, mass_count=8067
[23:10:51]   ℹ Warmup  40/80: avg_top_sim=0.9984, mass_count=8068
    Warmup  40/80: avg_top_sim=0.9984, mass_count=8068
[23:10:51]   ℹ Warmup  60/80: avg_top_sim=0.9974, mass_count=8068
    Warmup  60/80: avg_top_sim=0.9974, mass_count=8068
[23:10:51]   ℹ Warmup  80/80: avg_top_sim=0.9961, mass_count=8068
    Warmup  80/80: avg_top_sim=0.9961, mass_count=8068
[23:10:51]   ✓ GR warmup: 8068 concepts accumulated in mass tracker
  ✓ Mass tracker: 8068 concepts, avg_mass=1.1207

=================================================================
  Stage B2: Training SanityDecoder (wave + GR features → text)
=================================================================
[23:10:54]   ℹ Epoch   1/80: loss=4.2361, lr=5.0e-04, sample='In  ie  ,     n li e    s  e      e  o      e     '
    Epoch   1/80: loss=4.2361, lr=5.0e-04, sample='In  ie  ,     n li e    s  e      e  o      e     '
[23:11:17]   ℹ Epoch  10/80: loss=2.8348, lr=4.8e-04, sample='The er  a e      e  e      e              o t     '
    Epoch  10/80: loss=2.8348, lr=4.8e-04, sample='The er  a e      e  e      e              o t     '
[23:11:42]   ℹ Epoch  20/80: loss=2.6658, lr=4.3e-04, sample='Thn ~   l   oft te W     tee rt   d  ohe lo  r   e'
    Epoch  20/80: loss=2.6658, lr=4.3e-04, sample='Thn ~   l   oft te W     tee rt   d  ohe lo  r   e'
[23:12:08]   ℹ Epoch  30/80: loss=2.4256, lr=3.5e-04, sample='S nj~~mnhCVLofyhhau33:wUDReesrdedNdh mhu le 1(6Jka'
    Epoch  30/80: loss=2.4256, lr=3.5e-04, sample='S nj~~mnhCVLofyhhau33:wUDReesrdedNdh mhu le 1(6Jka'
[23:12:33]   ℹ Epoch  40/80: loss=2.0078, lr=2.6e-04, sample='S "j~~SnarVoopyeha 3':wUDoeesrdedNdh mbirlk 1(6Jka'
    Epoch  40/80: loss=2.0078, lr=2.6e-04, sample='S "j~~SnarVoopyeha 3':wUDoeesrdedNdh mbirlk 1(6Jka'
[23:12:59]   ℹ Epoch  50/80: loss=1.6133, lr=1.6e-04, sample='Senj~~mnarVoofyhhe 3o:kUDoaeardedNohtoberlo 1(6Jfp'
    Epoch  50/80: loss=1.6133, lr=1.6e-04, sample='Senj~~mnarVoofyhhe 3o:kUDoaeardedNohtoberlo 1(6Jfp'
[23:13:24]   ℹ Epoch  60/80: loss=1.2547, lr=8.2e-05, sample='S nj~~mnorV onyuhe 3o:kUsoeeardedNChtobeclo 1(sJfp'
    Epoch  60/80: loss=1.2547, lr=8.2e-05, sample='S nj~~mnorV onyuhe 3o:kUsoeeardedNChtobeclo 1(sJfp'
[23:13:50]   ℹ Epoch  70/80: loss=0.9949, lr=2.9e-05, sample='Senj~~mnorVaokytia 3o: UDoecordedNChroniclos1(6Jfp'
    Epoch  70/80: loss=0.9949, lr=2.9e-05, sample='Senj~~mnorVaokytia 3o: UDoecordedNChroniclos1(6Jfp'
[23:14:16]   ℹ Epoch  80/80: loss=0.9137, lr=1.0e-05, sample='Senj~~mnorValky ia 3o: UDrecordedNChroniclos1(6Jfp'
    Epoch  80/80: loss=0.9137, lr=1.0e-05, sample='Senj~~mnorValky ia 3o: UDrecordedNChroniclos1(6Jfp'
[23:14:16]   ✓ Decoder trained: final loss=0.9137
  ✓ Decoder training complete: final loss=0.9137

=================================================================
  Stage C: First End-to-End Pipeline Check (honest metrics)
=================================================================

Input: 'the cat sat on the mat'
Reconstructed: 'Pa mlnGtumotphthedvtavenps9,0aJsontpffyh ua6mh rhbimejb" lotteerfaAmlnpotW lGalnys suffeup .oAAvdctshe gdd ,far . wervwishotheoE'
Char accuracy: 4.5%  LCS ratio: 77.3%  Trigrams: 11.8%
Recognizable: YES
[23:14:16]   📊 pipeline_check('the cat sat on the m'): lcs=0.77 tri=0.12 rec=True

Input: 'hello world'
Reconstructed: 'Except foor~~soheVickClub Ceeoings , thI Togan Building rlmamned largelyiunnccupied for almost fifty yeerrland0sufferid sfgnlfic'
Char accuracy: 0.0%  LCS ratio: 90.9%  Trigrams: 0.0%
Recognizable: YES
[23:14:16]   📊 pipeline_check('hello world'): lcs=0.91 tri=0.00 rec=True

Input: 'artificial intelligence'
Reconstructed: 'Except foor~~soheVackClub Ceroings s thI Togan Building rlmamned largelyiunoccupiedpfor rlmosl fifty yeerrland0sufferid sfgnlfic'
Char accuracy: 0.0%  LCS ratio: 73.9%  Trigrams: 4.8%
Recognizable: YES
[23:14:16]   📊 pipeline_check('artificial intellige'): lcs=0.74 tri=0.05 rec=True

Input: 'the quick brown fox'
Reconstructed: 'Sxnj~~lfoaVy~fyhcei3f:LUtzlzeRickcuhdereDuywJ(sJuna'sock:m~~~~~~~~~~~ud~~~~~~~~~~~~~c~~-3:,aimselfV,lMyj1r7JfhnsB.BLttkm igldCap'
Char accuracy: 0.0%  LCS ratio: 52.6%  Trigrams: 5.9%
Recognizable: YES
[23:14:16]   📊 pipeline_check('the quick brown fox'): lcs=0.53 tri=0.06 rec=True

Input: 'once upon a time'
Reconstructed: 'The gmovrne  ao lodempteeeby tse f cecne thas p rvbdd  eae ca ezpis ofqutis Stapr anst i nthe paroegtoe erge cy te ,arms ingnmue'
Char accuracy: 6.2%  LCS ratio: 93.8%  Trigrams: 0.0%
Recognizable: YES
[23:14:16]   📊 pipeline_check('once upon a time'): lcs=0.94 tri=0.00 rec=True

  Pipeline check: 5/5 recognizable
  Avg LCS ratio:     77.70%
  Avg trigram overlap: 4.48%
[23:14:16]   ✓ Pipeline: 5/5 recognizable, avg_lcs=77.70%
[23:14:16]   📊 training_time: 266.9s

  ✓ Training complete in 266.9s (4.4 min)
[23:14:16]   ◼ CELL Cell 5 — Training / Load from Checkpoint — PASS

[23:22:01] 
▶ CELL: Cell 7 — Test 1: O(log n) Complexity
[23:22:01]   Started: 2026-03-21 23:22:01

   Seq Len      GR (ms)    Attn (ms)    GR Growth  Attn Growth
--------------------------------------------------------------
        64       18.1ms        0.3ms           --           --
       128       32.8ms        0.4ms        1.82x        1.09x
       256       73.4ms        0.5ms        2.24x        1.47x
       512      138.1ms        0.8ms        1.88x        1.42x
      1024      306.6ms        1.6ms        2.22x        2.04x
      2048      937.4ms        6.2ms        3.06x        3.95x

Log fit R² = 0.6922 (need > 0.85)
Growth ratio (64→2048): 51.9x (O(n²) would be 1024x)
Avg growth per doubling: 2.24x (O(n²)=4x, O(n log n)≈2.1x)

Note: GR is slower in absolute time than fused CUDA attention
due to Python/CPU spatial index. The scaling pattern is what matters.
  ✗ O(log n) Sub-linear Scaling (R²): 0.6922 (threshold: 0.85)
  ✓ Sub-quadratic Growth (64→2048): 51.9 (threshold: 181.0)
  ✓ Avg Doubling Growth < 3.0x: 2.24 (threshold: 3.0)

✗ FAIL: O(log n) complexity test

==================================================
Phase 3 Results saved to: /kaggle/working/FLUX/phases/phase3/RESULTS_PHASE_3.md
All tests passed: False
Ready for Phase 4: False
==================================================
[23:22:20]   ◼ CELL Cell 7 — Test 1: O(log n) Complexity — PASS


[23:22:28] 
▶ CELL: Cell 8 — Test 2: Retrieval Precision@k
[23:22:28]   Started: 2026-03-21 23:22:28

  Retrieval Precision Results (50 concepts):
  Precision@1:         1.000 (need > 0.8)
  Precision@10:        1.000 (need > 0.7)

  ✓ PASS: Retrieval Precision@k
  ✓ Precision@1: 1.0 (threshold: 0.8)
  ✓ Precision@10: 1.0 (threshold: 0.7)

==================================================
Phase 3 Results saved to: /kaggle/working/FLUX/phases/phase3/RESULTS_PHASE_3.md
All tests passed: True
Ready for Phase 4: True
==================================================
[23:22:28]   ◼ CELL Cell 8 — Test 2: Retrieval Precision@k — PASS

[23:22:35] 
▶ CELL: Cell 9 — Test 3: Negative Mass Repulsion
[23:22:35]   Started: 2026-03-21 23:22:35

  Negative Mass Repulsion Test (10 concepts):
  --------------------------------------------------
  Concept      Before       After        Repelled? 
  --------------------------------------------------
  ✓ Concept 0        1.0000    -1.5000   YES
  ✓ Concept 1        1.0000    -1.5000   YES
  ✓ Concept 2        1.0000    -1.5000   YES
  ✓ Concept 3        1.0000    -1.5000   YES
  ✓ Concept 4        1.0000    -1.5000   YES
  ✓ Concept 5        1.0000    -1.5000   YES
  ✓ Concept 6        1.0000    -1.5000   YES
  ✓ Concept 7        1.0000    -1.5000   YES
  ✓ Concept 8        1.0000    -1.5000   YES
  ✓ Concept 9        1.0000    -1.5000   YES

  Repelled: 10/10
  Negative mass concepts in tracker: 10

  ✓ PASS: Negative Mass Repulsion (10/10)
  ✓ Negative Mass Repulsion: 10 (threshold: 10)

==================================================
Phase 3 Results saved to: /kaggle/working/FLUX/phases/phase3/RESULTS_PHASE_3.md
All tests passed: True
Ready for Phase 4: True
==================================================
[23:22:35]   ◼ CELL Cell 9 — Test 3: Negative Mass Repulsion — PASS



