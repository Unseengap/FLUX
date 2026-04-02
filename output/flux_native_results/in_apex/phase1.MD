[22:47:55] 
▶ CELL: Cell 5 — Training
[22:47:55]   Started: 2026-03-20 22:47:55
[22:47:55]   ℹ Training with device=cuda, steps=5000
============================================================
FLUX Phase 1: Continuous Semantic Encoder (CSE)
============================================================

  Device: cuda
  Steps:  5000
  LR:     0.0003

── Loading Data ──
  Loading WikiText-2 dataset...
`trust_remote_code` is not supported anymore.
Please check that the Hugging Face dataset 'wikitext' isn't based on a loading script and remove `trust_remote_code`.
If the dataset is based on a loading script, please ask the dataset author to remove it and convert it to a standard format like Parquet.
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
README.md: 10.5kB [00:00, 30.4MB/s]
wikitext-2-raw-v1/test-00000-of-00001.pa(…): 100%|█| 733k/733k [00:00<00:00, 1.5
wikitext-2-raw-v1/train-00000-of-00001.p(…): 100%|█| 6.36M/6.36M [00:00<00:00, 1
wikitext-2-raw-v1/validation-00000-of-00(…): 100%|█| 657k/657k [00:00<00:00, 2.9
Generating test split: 100%|██████| 4358/4358 [00:00<00:00, 84746.77 examples/s]
Generating train split: 100%|██| 36718/36718 [00:00<00:00, 663414.87 examples/s]
Generating validation split: 100%|█| 3760/3760 [00:00<00:00, 569396.80 examples/
  ✓ Loaded 5000 train, 1000 val texts

── Building CSE + Decoder ──
  ✓ CSE parameters:     1,337,264
  ✓ Decoder parameters: 418,816
  ✓ Total parameters:   1,756,080

── Training for 5000 steps ──

  Step   100 | recon=0.6775 sem=0.2297 ant=0.0415 total=0.7025                                      
  Step   200 | recon=0.0355 sem=0.0198 ant=0.0204 total=0.0415                                      
  Step   300 | recon=0.0071 sem=0.0041 ant=0.0016 total=0.0086                                      
  Step   400 | recon=0.0046 sem=0.0004 ant=0.0003 total=0.0048                                      
  Step   500 | recon=0.0019 sem=0.0018 ant=0.0020 total=0.0033                                      
  ── VAL step 500 ── accuracy=0.998 recon_loss=0.002 CE=0.010 semantic=10/10                        
  Step   600 | recon=0.0014 sem=0.0004 ant=0.0003 total=0.0017                                      
  Step   700 | recon=0.0058 sem=0.0024 ant=0.0029 total=0.0085                                      
  Step   800 | recon=0.0023 sem=0.0005 ant=0.0009 total=0.0031                                      
  Step   900 | recon=0.0018 sem=0.0016 ant=0.0025 total=0.0044                                      
  Step  1000 | recon=0.0003 sem=0.0008 ant=0.0012 total=0.0018                                      
  ── VAL step 1000 ── accuracy=0.999 recon_loss=0.001 CE=0.005 semantic=10/10                       
  Step  1100 | recon=0.0012 sem=0.0003 ant=0.0015 total=0.0022                                      
  Step  1200 | recon=0.1495 sem=0.0012 ant=0.0005 total=0.1509                                      
  Step  1300 | recon=0.0012 sem=0.0001 ant=0.0010 total=0.0018                                      
  Step  1400 | recon=0.0294 sem=0.0002 ant=0.0002 total=0.0298                                      
  Step  1500 | recon=0.0065 sem=0.0002 ant=0.0001 total=0.0067                                      
  ── VAL step 1500 ── accuracy=1.000 recon_loss=0.000 CE=0.001 semantic=10/10                       
  Step  1600 | recon=0.0001 sem=0.0002 ant=0.0002 total=0.0004                                      
  Step  1700 | recon=0.0002 sem=0.0002 ant=0.0004 total=0.0006                                      
  Step  1800 | recon=0.0012 sem=0.0003 ant=0.0002 total=0.0016                                      
  Step  1900 | recon=0.0002 sem=0.0011 ant=0.0010 total=0.0019                                      
  Step  2000 | recon=0.0003 sem=0.0003 ant=0.0019 total=0.0015                                      
  ── VAL step 2000 ── accuracy=1.000 recon_loss=0.000 CE=0.002 semantic=10/10                       
  Step  2100 | recon=0.0002 sem=0.0003 ant=0.0011 total=0.0010                                      
  Step  2200 | recon=0.0003 sem=0.0003 ant=0.0001 total=0.0006                                      
  Step  2300 | recon=0.0001 sem=0.0001 ant=0.0002 total=0.0003                                      
  Step  2400 | recon=0.0002 sem=0.0002 ant=0.0011 total=0.0009                                      
  Step  2500 | recon=0.0013 sem=0.0005 ant=0.0002 total=0.0019                                      
  ── VAL step 2500 ── accuracy=1.000 recon_loss=0.000 CE=0.000 semantic=10/10                       
  Step  2600 | recon=0.0004 sem=0.0003 ant=0.0001 total=0.0009                                      
  Step  2700 | recon=0.0004 sem=0.0002 ant=0.0003 total=0.0008                                      
  Step  2800 | recon=0.0004 sem=0.0002 ant=0.0003 total=0.0007                                      
  Step  2900 | recon=0.0004 sem=0.0004 ant=0.0008 total=0.0012                                      
  Step  3000 | recon=0.0006 sem=0.0006 ant=0.0005 total=0.0014                                      
  ── VAL step 3000 ── accuracy=0.999 recon_loss=0.001 CE=0.002 semantic=10/10                       
  Step  3100 | recon=0.0001 sem=0.0004 ant=0.0006 total=0.0008                                      
  Step  3200 | recon=0.0009 sem=0.0012 ant=0.0001 total=0.0021                                      
  Step  3300 | recon=0.0001 sem=0.0001 ant=0.0003 total=0.0004                                      
  Step  3400 | recon=0.0001 sem=0.0001 ant=0.0001 total=0.0002                                      
  Step  3500 | recon=0.0000 sem=0.0001 ant=0.0000 total=0.0002                                      
  ── VAL step 3500 ── accuracy=0.999 recon_loss=0.001 CE=0.002 semantic=10/10                       
  Step  3600 | recon=0.0001 sem=0.0001 ant=0.0000 total=0.0002                                      
  Step  3700 | recon=0.0002 sem=0.0001 ant=0.0000 total=0.0003                                      
  Step  3800 | recon=0.0001 sem=0.0000 ant=0.0000 total=0.0001                                      
  Step  3900 | recon=0.0000 sem=0.0000 ant=0.0000 total=0.0001                                      
  Step  4000 | recon=0.0001 sem=0.0003 ant=0.0001 total=0.0005                                      
  ── VAL step 4000 ── accuracy=1.000 recon_loss=0.000 CE=0.000 semantic=10/10                       
  Step  4100 | recon=0.0182 sem=0.0002 ant=0.0002 total=0.0185                                      
  Step  4200 | recon=0.0001 sem=0.0003 ant=0.0004 total=0.0006                                      
  Step  4300 | recon=0.0001 sem=0.0003 ant=0.0004 total=0.0007                                      
  Step  4600 | recon=0.0002 sem=0.0002 ant=0.0001 total=0.0004                                      
  Step  4700 | recon=0.0001 sem=0.0001 ant=0.0002 total=0.0003                                      
  Step  4800 | recon=0.0000 sem=0.0002 ant=0.0000 total=0.0002                                      
  Step  4900 | recon=0.0000 sem=0.0000 ant=0.0000 total=0.0001                                      
  Step  5000 | recon=0.0000 sem=0.0001 ant=0.0001 total=0.0002                                      
  ── VAL step 5000 ── accuracy=1.000 recon_loss=0.000 CE=0.001 semantic=10/10                       
Training: 100%|████████████████| 5000/5000 [44:46<00:00,  1.86it/s, recon=0.000, sem=0.000, sw=1.00]

── Training Complete ──
  Duration: 0:44:46.070150
  Best validation accuracy: 1.000

── Final Validation ──
  accuracy: 0.9999263677196083
  recon_loss: 7.36322803917e-05
  cross_entropy: 0.00025408325906482786
  semantic_ordering: 10/10

── Saving Checkpoint ──
✓ Phase 1 checkpoint saved: /kaggle/working/FLUX/checkpoints/phase1.phase.pt (7.0 MB)

============================================================
Phase 1 training complete!
  Reconstruction accuracy: 1.000
  Semantic ordering:       10/10
  Next: run test scripts to verify acceptance criteria
============================================================
[23:32:54]   ✓ Checkpoint saved: /kaggle/working/FLUX/checkpoints/phase1.phase.pt (7.0 MB)
[23:32:54]   ◼ CELL Cell 5 — Training — OK


[23:32:59] 
▶ CELL: Cell 7 — Test 1: Reconstruction
[23:32:59]   Started: 2026-03-20 23:32:59
============================================================
FLUX Phase 1 Test 1: Reconstruction Quality
============================================================

  Loading Phase 1 checkpoint...
✓ Phase 1 checkpoint loaded (local, 7.0 MB)
  Loading validation sentences...
`trust_remote_code` is not supported anymore.
Please check that the Hugging Face dataset 'wikitext' isn't based on a loading script and remove `trust_remote_code`.
If the dataset is based on a loading script, please ask the dataset author to remove it and convert it to a standard format like Parquet.
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
  ✓ 950 sentences loaded

  Evaluating reconstruction quality...
    [200/950] running accuracy: 1.000
    [400/950] running accuracy: 1.000
    [600/950] running accuracy: 1.000
    [800/950] running accuracy: 1.000

  ── Results ──
  Average byte accuracy:       0.9998
  Average reconstruction loss: 0.0002
  Max single-sentence loss:    0.0889
  Failed sentences (loss>0.5): 0/950
  Time elapsed:                9.4s
  ✓ Average Reconstruction Loss: 0.0002 (threshold: < 0.1)
  ✓ No Complete Failures: 0 failures (threshold: 0 sentences with loss > 0.5)
  ✓ Test Runtime: 9.4s (threshold: < 60s)

==================================================
Phase 1 Results saved to: /kaggle/working/FLUX/phases/phase1/RESULTS_PHASE_1.md
All tests passed: True
Ready for Phase 2: True
==================================================

  All tests passed: True
  ✓ RECONSTRUCTION TEST PASSED
[23:33:11]   ◼ CELL Cell 7 — Test 1: Reconstruction — OK


[23:33:11] 
▶ CELL: Cell 8 — Test 2: Language Agnostic
[23:33:11]   Started: 2026-03-20 23:33:11
============================================================
FLUX Phase 1 Test 2: Language-Agnostic Encoding
============================================================

  Loading Phase 1 checkpoint...
✓ Phase 1 checkpoint loaded (local, 7.0 MB)

  ── Test: Encode diverse UTF-8 inputs ──
    ✓ "Hello, world!" → [13, 432]
    ✓ "Bonjour le monde!" → [17, 432]
    ✓ "¡Hola mundo!" → [13, 432]
    ✓ "你好世界" → [12, 432]
    ✓ "こんにちは世界" → [21, 432]
    ✓ "def hello(): print('hi')" → [24, 432]
    ✓ "E = mc² + ∫f(x)dx" → [20, 432]
    ✓ "🎉🚀💡" → [12, 432]
    ✓ "Mixed: hello 你好 🌍" → [24, 432]
    ✓ "" → [0, 432]
    ✓ "a" → [1, 432]
  ✓ UTF-8 Encoding (no errors): 11/11 success (threshold: all inputs encode)

  ── Test: Cross-Language Similarity ──
    ✓ EN/FR cat: sim("the cat", "le chat") = 0.9280
    ✓ EN/FR hello: sim("hello", "bonjour") = 0.8480
    ✓ EN/FR water: sim("water", "eau") = 0.8517
    ✓ EN/FR sentence: sim("the cat is sleeping", "le chat dort") = 0.8503
    ✓ EN/FR thanks: sim("thank you", "merci") = 0.8464
  ✓ Cross-Language Similarity: 5/5 pairs > 0.4 (threshold: ≥ 4/5 pairs pass)

  ── Test: Deterministic Encoding ──
    Max difference between two encodings: 0.00e+00
    ✓ Encoding is deterministic
  ✓ Deterministic Encoding: max_diff=0.00e+00 (threshold: < 1e-6)

==================================================
Phase 1 Results saved to: /kaggle/working/FLUX/phases/phase1/RESULTS_PHASE_1.md
All tests passed: True
Ready for Phase 2: True
==================================================

  All tests passed: True
  ✓ LANGUAGE AGNOSTIC TEST PASSED
[23:33:14]   ◼ CELL Cell 8 — Test 2: Language Agnostic — OK


[23:33:14] 
▶ CELL: Cell 9 — Test 3: Semantic Proximity
[23:33:14]   Started: 2026-03-20 23:33:14
============================================================
FLUX Phase 1 Test 3: Semantic Proximity Ordering
============================================================

  Loading Phase 1 checkpoint...
✓ Phase 1 checkpoint loaded (local, 7.0 MB)

  ── Evaluating semantic proximity ──
    #  Anchor       Closer         Farther        sim(A,B) sim(A,C)  Pass
  ----------------------------------------------------------------------
    1  dog          cat            democracy        0.8467  -0.0029     ✓
    2  Paris        France         banana           0.8557  -0.0053     ✓
    3  running      jogging        purple           0.8396  -0.0177     ✓
    4  happy        joyful         concrete         0.8512   0.0001     ✓
    5  king         queen          triangle         0.8365  -0.0153     ✓
    6  car          vehicle        philosophy       0.8501   0.0057     ✓
    7  ocean        sea            keyboard         0.8528  -0.0087     ✓
    8  doctor       physician      algebra          0.8349  -0.0214     ✓
    9  big          large          grammar          0.8514  -0.0085     ✓
   10  fast         quick          geology          0.8503  -0.0001     ✓
   11  house        home           electron         0.8474   0.0019     ✓
   12  road         street         symphony         0.8475  -0.0141     ✓
   13  smart        intelligent    pencil           0.8508  -0.0155     ✓
   14  old          ancient        software         0.8421  -0.0013     ✓
   15  dark         dim            onion            0.8421  -0.0031     ✓
   16  love         adore          thermometer      0.8494  -0.0077     ✓
   17  strong       powerful       butterfly        0.8339  -0.0081     ✓
   18  beautiful    pretty         protocol         0.8459  -0.0189     ✓
   19  sad          unhappy        politics         0.8571  -0.0097     ✓
   20  begin        start          geology          0.8526  -0.0132     ✓

  ── Score: 20/20 ──
  ✓ Semantic Proximity Ordering: 20/20 (threshold: ≥ 18/20)

  ── Similarity Levels (informational) ──
    Average similarity (related words):   0.8469
    Average similarity (unrelated words):  -0.0082
    Gap (related - unrelated):             0.8551
  ✓ Similar Words > 0.5 avg: 0.8469 (threshold: > 0.5 average)

==================================================
Phase 1 Results saved to: /kaggle/working/FLUX/phases/phase1/RESULTS_PHASE_1.md
All tests passed: True
Ready for Phase 2: True
==================================================

  All tests passed: True
  ✓ SEMANTIC PROXIMITY TEST PASSED
[23:33:18]   ◼ CELL Cell 9 — Test 3: Semantic Proximity — OK



[23:33:23]   ✓ Wave visualization generated
[23:33:23]   ◼ CELL Cell 10 — Demo 1: Wave Visualization — OK



[23:33:23] 
▶ CELL: Cell 11 — Demo 2: Interference Patterns
[23:33:23]   Started: 2026-03-20 23:33:23

=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
FLUX Phase 1 Demo 2: Interference Patterns
============================================================

  Loading Phase 1 checkpoint...
✓ Phase 1 checkpoint loaded (local, 7.0 MB)

── Constructive Interference (Similar Words) ──
      Similar Words → High Similarity (Constructive)       
┏━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Word 1 ┃ Word 2    ┃ Cosine Similarity ┃     Status     ┃
┡━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ king   │ queen     │            0.8365 │ ✓ constructive │
│ dog    │ cat       │            0.8467 │ ✓ constructive │
│ happy  │ joyful    │            0.8512 │ ✓ constructive │
│ car    │ vehicle   │            0.8501 │ ✓ constructive │
│ ocean  │ sea       │            0.8528 │ ✓ constructive │
│ doctor │ physician │            0.8349 │ ✓ constructive │
└────────┴───────────┴───────────────────┴────────────────┘

── Destructive Interference (Opposite Words) ──
     Opposite Words → Low Similarity (Destructive)     
┏━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Word 1 ┃ Word 2 ┃ Cosine Similarity ┃    Status     ┃
┡━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ hot    │ cold   │            0.0971 │ ✓ destructive │
│ happy  │ sad    │            0.1043 │ ✓ destructive │
│ love   │ hate   │            0.0999 │ ✓ destructive │
│ big    │ small  │            0.0926 │ ✓ destructive │
│ light  │ dark   │            0.1051 │ ✓ destructive │
└────────┴────────┴───────────────────┴───────────────┘

── Neutral (Unrelated Words) ──
   Unrelated Words → Near-Zero Similarity (Neutral)   
┏━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Word 1 ┃ Word 2    ┃ Cosine Similarity ┃  Status   ┃
┡━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ cat    │ democracy │            0.1945 │ ✓ neutral │
│ Paris  │ banana    │           -0.0053 │ ✓ neutral │
│ happy  │ concrete  │            0.0001 │ ✓ neutral │
│ ocean  │ keyboard  │           -0.0087 │ ✓ neutral │
│ doctor │ algebra   │           -0.0214 │ ✓ neutral │
└────────┴───────────┴───────────────────┴───────────┘

── Summary ──
  Average similarity (similar words):   0.8454
  Average similarity (opposite words):  0.0998
  Average similarity (unrelated words): 0.0319

  Separation (similar - opposite): 0.7455
  ✓ Demo 2 complete

[23:33:26]   ◼ CELL Cell 11 — Demo 2: Interference Patterns — OK


[23:33:26] 
▶ CELL: Cell 12 — Interactive Exploration
[23:33:26]   Started: 2026-03-20 23:33:26
✓ Phase 1 checkpoint loaded (local, 7.0 MB)
Text 1                              Text 2                              Similarity
──────────────────────────────────────────────────────────────────────────────────
[23:33:26]   📊 sim("dog", "cat"): 0.8467
dog                                 cat                                     0.8467
[23:33:26]   📊 sim("dog", "democracy"): -0.0029
dog                                 democracy                              -0.0029
[23:33:26]   📊 sim("king", "queen"): 0.8365
king                                queen                                   0.8365
[23:33:26]   📊 sim("hot", "cold"): 0.0971
hot                                 cold                                    0.0971
[23:33:26]   📊 sim("hello", "bonjour"): 0.8480
hello                               bonjour                                 0.8480
[23:33:26]   📊 sim("python code", "javascript code"): 0.7842
python code                         javascript code                         0.7842
[23:33:26]   📊 sim("I love pizza", "Pizza is my favorite food"): 0.7836
I love pizza                        Pizza is my favorite food               0.7836
[23:33:26]   📊 sim("the sun is a star", "le soleil est une étoile"): 0.7990
the sun is a star                   le soleil est une étoile                0.7990
[23:33:26]   ◼ CELL Cell 12 — Interactive Exploration — OK



[23:33:26]   ✓ Results displayed
[23:33:26]   ◼ CELL Cell 13 — Results Summary — OK




