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


