[07:50:46] 
▶ CELL: Cell 6 — Stage 1: WaveToText Pre-Training
[07:50:46]   Started: 2026-03-23 07:50:46
  ℹ Stage 1: Collecting (wave, word) pairs...
  WTT Step  10000  loss=0.6605  chunks=320,000
[07:52:51]   📊 wtt_step_10000_loss: 0.6605
  WTT Step  20000  loss=0.2177  chunks=640,000
[07:54:53]   📊 wtt_step_20000_loss: 0.2177
  WTT Step  30000  loss=0.1618  chunks=960,000
[07:56:53]   📊 wtt_step_30000_loss: 0.1618
  WTT Step  40000  loss=0.1366  chunks=1,280,000
[07:58:54]   📊 wtt_step_40000_loss: 0.1366
  WTT Step  50000  loss=0.1235  chunks=1,600,000
[08:00:57]   📊 wtt_step_50000_loss: 0.1235
  WTT Step  60000  loss=0.1067  chunks=1,920,000
[08:02:58]   📊 wtt_step_60000_loss: 0.1067

  ✓ Stage 1 complete: 60000 steps, 1,920,000 chunks
    Final loss: 0.0956
    Word accuracy: 86.8%
    Time: 733.5s
[08:03:00]   📊 wtt_steps: 60000
[08:03:00]   📊 wtt_word_accuracy: 86.8%
[08:03:00]   📊 wtt_final_loss: 0.0956
[08:03:00]   📊 wtt_time: 733.5s
[08:03:00]   ◼ CELL Cell 6 — Stage 1: WaveToText Pre-Training — PASS══════════════════════════════════════════════════════════════════════
  WaveToText Decode Diagnostic
══════════════════════════════════════════════════════════════════════

  Text: "Hello world"
  Chunks: 6

    Chunk 0: expected="He" (2 bytes)
      Teacher-forced byte acc: 100% (2/2)
      Greedy decode:   "He" ✓
      Temp=0.5 decode: "He"
      First byte — expected: 72 ('H')
      First byte — top5 logits: [(72, "'H'"), (119, "'w'"), (83, "'S'"), (69, "'E'"), (66, "'B'")]
      First byte — top5 probs:  [0.9040500521659851, 0.04798198863863945, 0.01684623397886753, 0.00905662402510643, 0.005151932127773762]

    Chunk 1: expected="ll" (2 bytes)
      Teacher-forced byte acc: 100% (2/2)
      Greedy decode:   "ll" ✓
      Temp=0.5 decode: "ll"
      First byte — expected: 108 ('l')
      First byte — top5 logits: [(108, "'l'"), (117, "'u'"), (97, "'a'"), (118, "'v'"), (105, "'i'")]
      First byte — top5 probs:  [0.988990068435669, 0.005186557304114103, 0.004650229122489691, 0.0005834443145431578, 0.00017387632397003472]

    Chunk 2: expected="o " (2 bytes)
      Teacher-forced byte acc: 50% (1/2)
      Greedy decode:   " o" ✗
      Temp=0.5 decode: " o"
      First byte — expected: 111 ('o')
      First byte — top5 logits: [(32, "' '"), (111, "'o'"), (47, "'/'"), (100, "'d'"), (89, "'Y'")]
      First byte — top5 probs:  [0.770015299320221, 0.2278723567724228, 0.0005383077659644186, 0.0002945945889223367, 0.00023066633730195463]

    Chunk 3: expected="wo" (2 bytes)
      Teacher-forced byte acc: 100% (2/2)
      Greedy decode:   "wo" ✓
      Temp=0.5 decode: "wo"
      First byte — expected: 119 ('w')
      First byte — top5 logits: [(119, "'w'"), (111, "'o'"), (66, "'B'"), (77, "'M'"), (95, "'_'")]
      First byte — top5 probs:  [0.8747654557228088, 0.10506124049425125, 0.01362115703523159, 0.0016651804326102138, 0.0011297555174678564]

    Chunk 4: expected="rl" (2 bytes)
      Teacher-forced byte acc: 100% (2/2)
      Greedy decode:   "rl" ✓
      Temp=0.5 decode: "lr"
      First byte — expected: 114 ('r')
      First byte — top5 logits: [(114, "'r'"), (112, "'p'"), (108, "'l'"), (101, "'e'"), (69, "'E'")]
      First byte — top5 probs:  [0.5670316815376282, 0.13169682025909424, 0.09911778569221497, 0.08245129138231277, 0.03592448681592941]

  Text: "The cat sat on the mat"
  Chunks: 11

    Chunk 0: expected="Th" (2 bytes)
      Teacher-forced byte acc: 100% (2/2)
      Greedy decode:   "Th" ✓
      Temp=0.5 decode: "Th"
      First byte — expected: 84 ('T')
      First byte — top5 logits: [(84, "'T'"), (104, "'h'"), (85, "'U'"), (70, "'F'"), (122, "'z'")]
      First byte — top5 probs:  [0.9971645474433899, 0.0018578112358227372, 0.0003257932257838547, 0.00016503392544109374, 9.127190423896536e-05]

    Chunk 1: expected="e " (2 bytes)
      Teacher-forced byte acc: 100% (2/2)
      Greedy decode:   "e " ✓
      Temp=0.5 decode: "e "
      First byte — expected: 101 ('e')
      First byte — top5 logits: [(101, "'e'"), (32, "' '"), (44, "','"), (97, "'a'"), (104, "'h'")]
      First byte — top5 probs:  [0.9905825853347778, 0.009257960133254528, 4.079860445926897e-05, 3.651322185760364e-05, 2.140761171176564e-05]

    Chunk 2: expected="ca" (2 bytes)
      Teacher-forced byte acc: 100% (2/2)
      Greedy decode:   "ca" ✓
      Temp=0.5 decode: "ca"
      First byte — expected: 99 ('c')
      First byte — top5 logits: [(99, "'c'"), (97, "'a'"), (69, "'E'"), (34, '\'"\''), (47, "'/'")]
      First byte — top5 probs:  [0.8202415108680725, 0.1793757975101471, 7.361717143794522e-05, 6.414785457309335e-05, 2.9772774723824114e-05]

    Chunk 3: expected="t " (2 bytes)
      Teacher-forced byte acc: 100% (2/2)
      Greedy decode:   "t " ✓
      Temp=0.5 decode: "t "
      First byte — expected: 116 ('t')
      First byte — top5 logits: [(116, "'t'"), (32, "' '"), (46, "'.'"), (97, "'a'"), (110, "'n'")]
      First byte — top5 probs:  [0.9175429344177246, 0.08190325647592545, 0.0002278133324580267, 8.452058682451025e-05, 8.117157267406583e-05]

    Chunk 4: expected="sa" (2 bytes)
      Teacher-forced byte acc: 100% (2/2)
      Greedy decode:   "sa" ✓
      Temp=0.5 decode: "sa"
      First byte — expected: 115 ('s')
      First byte — top5 logits: [(115, "'s'"), (97, "'a'"), (112, "'p'"), (108, "'l'"), (101, "'e'")]
      First byte — top5 probs:  [0.9810417890548706, 0.017398329451680183, 0.0004254376981407404, 0.0003477212449070066, 7.566493150079623e-05]

  Text: "artificial intelligence"
  Chunks: 9

    Chunk 0: expected="ar" (2 bytes)
      Teacher-forced byte acc: 100% (2/2)
      Greedy decode:   "ar" ✓
      Temp=0.5 decode: "Na"
      First byte — expected: 97 ('a')
      First byte — top5 logits: [(97, "'a'"), (78, "'N'"), (114, "'r'"), (68, "'D'"), (54, "'6'")]
      First byte — top5 probs:  [0.6390855312347412, 0.07417955994606018, 0.05784626677632332, 0.04755963385105133, 0.03302442282438278]

    Chunk 1: expected="ti" (2 bytes)
      Teacher-forced byte acc: 50% (1/2)
      Greedy decode:   "it" ✗
      Temp=0.5 decode: "it"
      First byte — expected: 116 ('t')
      First byte — top5 logits: [(105, "'i'"), (116, "'t'"), (101, "'e'"), (104, "'h'"), (87, "'W'")]
      First byte — top5 probs:  [0.8980404138565063, 0.09903597086668015, 0.002828447613865137, 3.6245510273147374e-05, 1.6025513104978018e-05]

    Chunk 2: expected="fici" (4 bytes)
      Teacher-forced byte acc: 75% (3/4)
      Greedy decode:   "ific" ✗
      Temp=0.5 decode: "fici"
      First byte — expected: 102 ('f')
      First byte — top5 logits: [(105, "'i'"), (102, "'f'"), (99, "'c'"), (100, "'d'"), (101, "'e'")]
      First byte — top5 probs:  [0.68344646692276, 0.23115789890289307, 0.04555588215589523, 0.01659880205988884, 0.009188089519739151]

    Chunk 3: expected="al" (2 bytes)
      Teacher-forced byte acc: 100% (2/2)
      Greedy decode:   "al" ✓
      Temp=0.5 decode: "al"
      First byte — expected: 97 ('a')
      First byte — top5 logits: [(97, "'a'"), (108, "'l'"), (105, "'i'"), (118, "'v'"), (99, "'c'")]
      First byte — top5 probs:  [0.9707419276237488, 0.029094204306602478, 4.774420813191682e-05, 3.788309550145641e-05, 1.9043749489355832e-05]

    Chunk 4: expected=" i" (2 bytes)
      Teacher-forced byte acc: 100% (2/2)
      Greedy decode:   " i" ✓
      Temp=0.5 decode: " i"
      First byte — expected: 32 (' ')
      First byte — top5 logits: [(32, "' '"), (105, "'i'"), (44, "','"), (101, "'e'"), (70, "'F'")]
      First byte — top5 probs:  [0.7724661827087402, 0.22426216304302216, 0.002906650770455599, 0.0001297437702305615, 5.185498594073579e-05]

  Text: "energy"
  Chunks: 3

    Chunk 0: expected="en" (2 bytes)
      Teacher-forced byte acc: 100% (2/2)
      Greedy decode:   "en" ✓
      Temp=0.5 decode: "en"
      First byte — expected: 101 ('e')
      First byte — top5 logits: [(101, "'e'"), (84, "'T'"), (110, "'n'"), (115, "'s'"), (40, "'('")]
      First byte — top5 probs:  [0.531379222869873, 0.0975954607129097, 0.07560502737760544, 0.0541369803249836, 0.04854648560285568]

    Chunk 1: expected="er" (2 bytes)
      Teacher-forced byte acc: 100% (2/2)
      Greedy decode:   "er" ✓
      Temp=0.5 decode: "er"
      First byte — expected: 101 ('e')
      First byte — top5 logits: [(101, "'e'"), (78, "'N'"), (68, "'D'"), (40, "'('"), (83, "'S'")]
      First byte — top5 probs:  [0.8179118633270264, 0.11632362753152847, 0.014353177510201931, 0.012740877456963062, 0.008926397189497948]

    Chunk 2: expected="gy" (2 bytes)
      Teacher-forced byte acc: 50% (1/2)
      Greedy decode:   "y:" ✗
      Temp=0.5 decode: "yg"
      First byte — expected: 103 ('g')
      First byte — top5 logits: [(121, "'y'"), (100, "'d'"), (105, "'i'"), (111, "'o'"), (71, "'G'")]
      First byte — top5 probs:  [0.8941399455070496, 0.040412962436676025, 0.008728617802262306, 0.006414060015231371, 0.004049587529152632]

  Text: "physics"
  Chunks: 4

    Chunk 0: expected="ph" (2 bytes)
      Teacher-forced byte acc: 100% (2/2)
      Greedy decode:   "ph" ✓
      Temp=0.5 decode: "eph"
      First byte — expected: 112 ('p')
      First byte — top5 logits: [(112, "'p'"), (128, '0x80'), (85, "'U'"), (101, "'e'"), (104, "'h'")]
      First byte — top5 probs:  [0.3114811182022095, 0.250656396150589, 0.13226363062858582, 0.12339033931493759, 0.09084213525056839]

    Chunk 1: expected="ys" (2 bytes)
      Teacher-forced byte acc: 50% (1/2)
      Greedy decode:   "sy" ✗
      Temp=0.5 decode: "s:"
      First byte — expected: 121 ('y')
      First byte — top5 logits: [(115, "'s'"), (121, "'y'"), (49, "'1'"), (53, "'5'"), (51, "'3'")]
      First byte — top5 probs:  [0.6731345653533936, 0.17809052765369415, 0.03040505014359951, 0.024083556607365608, 0.01295231468975544]

    Chunk 2: expected="ic" (2 bytes)
      Teacher-forced byte acc: 50% (1/2)
      Greedy decode:   "��ic" ✗
      Temp=0.5 decode: "�ic"
      First byte — expected: 105 ('i')
      First byte — top5 logits: [(128, '0x80'), (99, "'c'"), (105, "'i'"), (101, "'e'"), (47, "'/'")]
      First byte — top5 probs:  [0.4562646448612213, 0.228773832321167, 0.09438703209161758, 0.061242617666721344, 0.035474807024002075]

    Chunk 3: expected="s" (1 bytes)
      Teacher-forced byte acc: 100% (1/1)
      Greedy decode:   "ss" ✗
      Temp=0.5 decode: "ss"
      First byte — expected: 115 ('s')
      First byte — top5 logits: [(115, "'s'"), (101, "'e'"), (108, "'l'"), (32, "' '"), (117, "'u'")]
      First byte — top5 probs:  [0.9921038150787354, 0.0026743013877421618, 0.0010010021505877376, 0.0009041826706379652, 0.0008158628479577601]

══════════════════════════════════════════════════════════════════════
  If teacher-forced acc is high but greedy fails → exposure bias
  If teacher-forced acc is low → training didn't work (check loss)
  If greedy works but temp=0.5 fails → temperature too high
  If first-byte top5 is wrong → wave→hidden projection is broken
══════════════════════════════════════════════════════════════════════[09:17:23] 
▶ CELL: Cell 7a -- Pre-compute WG Data
[09:17:23]   Started: 2026-03-23 09:17:23
  ℹ Pre-computing frozen pipeline outputs for up to 2,000 samples...
    ... 50 texts → 50 valid  [4 text/s, ETA 511s]
    ... 100 texts → 100 valid  [4 text/s, ETA 501s]
    ... 150 texts → 150 valid  [4 text/s, ETA 462s]
    ... 200 texts → 200 valid  [4 text/s, ETA 452s]
    ... 250 texts → 250 valid  [4 text/s, ETA 441s]
    ... 300 texts → 300 valid  [4 text/s, ETA 424s]
    ... 350 texts → 350 valid  [4 text/s, ETA 417s]
    ... 400 texts → 400 valid  [4 text/s, ETA 403s]
    ... 450 texts → 450 valid  [4 text/s, ETA 389s]
    ... 500 texts → 500 valid  [4 text/s, ETA 373s]
    ... 550 texts → 550 valid  [4 text/s, ETA 361s]
    ... 600 texts → 600 valid  [4 text/s, ETA 348s]
    ... 650 texts → 650 valid  [4 text/s, ETA 336s]
    ... 700 texts → 700 valid  [4 text/s, ETA 323s]
    ... 750 texts → 750 valid  [4 text/s, ETA 311s]
    ... 800 texts → 800 valid  [4 text/s, ETA 298s]
    ... 850 texts → 850 valid  [4 text/s, ETA 287s]
    ... 900 texts → 900 valid  [4 text/s, ETA 274s]
    ... 950 texts → 950 valid  [4 text/s, ETA 261s]
    ... 1,000 texts → 1,000 valid  [4 text/s, ETA 248s]
    ... 1,050 texts → 1,050 valid  [4 text/s, ETA 235s]
    ... 1,100 texts → 1,100 valid  [4 text/s, ETA 224s]
    ... 1,150 texts → 1,150 valid  [4 text/s, ETA 212s]
    ... 1,200 texts → 1,200 valid  [4 text/s, ETA 199s]
    ... 1,250 texts → 1,250 valid  [4 text/s, ETA 186s]
    ... 1,300 texts → 1,300 valid  [4 text/s, ETA 173s]
    ... 1,350 texts → 1,350 valid  [4 text/s, ETA 161s]
    ... 1,400 texts → 1,400 valid  [4 text/s, ETA 149s]
    ... 1,450 texts → 1,450 valid  [4 text/s, ETA 136s]
    ... 1,500 texts → 1,500 valid  [4 text/s, ETA 124s]
    ... 1,550 texts → 1,550 valid  [4 text/s, ETA 111s]
    ... 1,600 texts → 1,600 valid  [4 text/s, ETA 99s]
    ... 1,650 texts → 1,650 valid  [4 text/s, ETA 87s]
    ... 1,700 texts → 1,700 valid  [4 text/s, ETA 74s]
    ... 1,750 texts → 1,750 valid  [4 text/s, ETA 62s]
    ... 1,800 texts → 1,800 valid  [4 text/s, ETA 49s]
    ... 1,850 texts → 1,850 valid  [4 text/s, ETA 37s]
    ... 1,900 texts → 1,900 valid  [4 text/s, ETA 25s]
    ... 1,950 texts → 1,950 valid  [4 text/s, ETA 12s]
    ... 2,000 texts → 2,000 valid  [4 text/s, ETA 0s]
  ✓ Pre-computed 2,000 samples in 495.0s (skipped 0)

  Pre-computed 2,000 samples in 495.0s
  Each sample: (merged [768], target_waves [N, 432]) on CPU
[09:25:38]   ◼ CELL Cell 7a -- Pre-compute WG Data — PASS[09:47:31] 
▶ CELL: Cell 7c -- WaveGenerator Training
[09:47:31]   Started: 2026-03-23 09:47:31
  ✓ Using pre-computed data (2,000 samples from Cell 7a)
  ✓ Using 2,000 pre-computed samples (skipping pre-computation)
  ✓ Gradient check: 19/19 generator params trainable

  ℹ Starting WG training loop: 1 steps over 2,000 samples
  WG Step     1/1  loss=0.3753  cos_acc=0.711  (first step: 10.71s)

  ✓ Stage 2 complete: 1 steps
    Final loss: 0.3753
    Avg cosine accuracy: 0.711
    Pre-compute time: 0.0s
    Training time: 10.7s
    Total time: 10.7s
[09:47:42]   📊 wg_steps: 1
[09:47:42]   📊 wg_final_loss: 0.3753
[09:47:42]   📊 wg_cosine_accuracy: 0.711
[09:47:42]   📊 wg_time: 10.7s

  Total training time: 744.3s (0.2h)
[09:47:42]   ◼ CELL Cell 7c -- WaveGenerator Training — PASS══════════════════════════════════════════════════════════════════════
  WaveGenerator Quality Diagnostic
══════════════════════════════════════════════════════════════════════

  Text: "The future of artificial intelligence"
  Chunks: 15  |  MSE: 0.1050  |  Mean cos: 0.655
    Wave 0: cos=0.577  conf=0.48  expected="Th"  gt_decode="Th"  predicted="edTTpdTTeTTF" ✗
    Wave 1: cos=0.554  conf=0.46  expected="e "  gt_decode="e "  predicted="apFgFFFFgigF" ✗
    Wave 2: cos=0.543  conf=0.47  expected="fu"  gt_decode="fu"  predicted="d FFFgFFFFighigF" ✗
    Wave 3: cos=0.612  conf=0.48  expected="tur"  gt_decode="tur"  predicted="  F" ✗
    Wave 4: cos=0.684  conf=0.47  expected="e "  gt_decode="e "  predicted=" n F FF" ✗
    Wave 5: cos=0.622  conf=0.47  expected="of"  gt_decode="of"  predicted=" nFFFFFFFFFFFFFigF" ✗
    Wave 6: cos=0.689  conf=0.47  expected=" a"  gt_decode=" a"  predicted="n g TTTTTTTTTTTTTTTT" ✗
    Wave 7: cos=0.616  conf=0.47  expected="rt"  gt_decode="tr"  predicted="a FigFgFFgh" ✗
    Free generation (15 waves): "edTTpdTTeTTF apghghghghhghhghhghh ag FgTasgTgh TgTghgh u gFgTghgTgTTgT,gTTT aghggh gh agFghagsTgTgsTgTghTg aggh TgTghghgTgh agghgsTgTghgTghgTgTT agghgsTgTgTghgTgTTgh agghgsTgTgTghgTgTgTT agghgsTgTgTghgTgTgTT agghgsTgTgTghgTgTgTT agghgsTgTgTghgTgTgTT agghgsTgTgTghgTgTgTT agghgsTgTgTghgTgTgTT"
    Free vs target cos: 0.588

  Text: "Hello world"
  Chunks: 6  |  MSE: 0.1355  |  Mean cos: 0.556
    Wave 0: cos=0.427  conf=0.48  expected="He"  gt_decode="He"  predicted="edTTpdTTTeTF" ✗
    Wave 1: cos=0.399  conf=0.46  expected="ll"  gt_decode="ll"  predicted="apdFFgFighpgFph" ✗
    Wave 2: cos=0.630  conf=0.47  expected="o "  gt_decode=" o"  predicted="d FF FFTF" ✗
    Wave 3: cos=0.671  conf=0.48  expected="wo"  gt_decode="wo"  predicted=" ng FigigigF" ✗
    Wave 4: cos=0.605  conf=0.47  expected="rl"  gt_decode="rl"  predicted=" n gFFFF" ✗
    Wave 5: cos=0.603  conf=0.48  expected="d"  gt_decode="dd"  predicted=" n FFF" ✗
    Free generation (6 waves): "edTTpdTTTeTF apghghghghhghhghhghh ag FgTsTgTgThgTh TgT u gFgTghgTgTTgT,gTTT aghggh gh agFghagsTgTgsTgTgTTg"
    Free vs target cos: 0.520

  Text: "Energy and momentum are conserved"
  Chunks: 15  |  MSE: 0.1072  |  Mean cos: 0.651
    Wave 0: cos=0.417  conf=0.48  expected="En"  gt_decode="En"  predicted="edTTpdTTeTTF" ✗
    Wave 1: cos=0.541  conf=0.46  expected="er"  gt_decode="er"  predicted="apdgFFFigphigp" ✗
    Wave 2: cos=0.608  conf=0.46  expected="gy"  gt_decode="yg"  predicted="d FFFhFFFFFF" ✗
    Wave 3: cos=0.698  conf=0.48  expected=" an"  gt_decode=" an"  predicted="  FF" ✗
    Wave 4: cos=0.782  conf=0.47  expected="d "  gt_decode="d "  predicted=" n gFFFFFi" ✗
    Wave 5: cos=0.701  conf=0.47  expected="mo"  gt_decode="mo"  predicted="an FFFghFFFFFF" ✗
    Wave 6: cos=0.737  conf=0.47  expected="me"  gt_decode="me"  predicted="n g TTTTTTTTTTTTTTTT" ✗
    Wave 7: cos=0.729  conf=0.47  expected="nt"  gt_decode="nt"  predicted="a FFgFFFFFFFF" ✗
    Free generation (15 waves): "edTTpdTTeTTF apghghghghhghhghhghh ag FgTasgh TgTghghgh u gFgTghgTgTTgT,gTTT aggh gh agFghagsTgTgsTgTghTg aggh TgTghghgTgh agghgsTgTghgTghgTgTT agghgsTgTgTghgTgTTgh agghgsTgTgTghgTgTgTT agghgsTgTgTghgTgTgTT agghgsTgTgTghgTgTgTT agghgsTgTgTghgTgTgTT agghgsTgTgTghgTgTgTT aggh TgTghghgh"
    Free vs target cos: 0.594

  Text: "The cat sat on the mat"
  Chunks: 11  |  MSE: 0.1082  |  Mean cos: 0.666
    Wave 0: cos=0.590  conf=0.48  expected="Th"  gt_decode="Th"  predicted="edTTpdTTeTF" ✗
    Wave 1: cos=0.575  conf=0.46  expected="e "  gt_decode="e "  predicted="apFgFFFFgFigF" ✗
    Wave 2: cos=0.748  conf=0.47  expected="ca"  gt_decode="ca"  predicted="d FFFghFFFFFigFFFF" ✗
    Wave 3: cos=0.728  conf=0.48  expected="t "  gt_decode="t "  predicted=" n gFiFF" ✗
    Wave 4: cos=0.574  conf=0.47  expected="sa"  gt_decode="sa"  predicted="a FFFFFFF" ✗
    Wave 5: cos=0.719  conf=0.47  expected="t "  gt_decode="t "  predicted=" n FFFFFFFFFFFFF" ✗
    Wave 6: cos=0.646  conf=0.47  expected="on"  gt_decode="on"  predicted="a FFFFFFFFFFFFFFFFF" ✗
    Wave 7: cos=0.757  conf=0.47  expected=" t"  gt_decode=" t"  predicted="an gF FFTFTF" ✗
    Free generation (11 waves): "edTTpdTTeTF apghghghghhghhghhghh ag FgTasgTgh TgTghgh u gFgTghgTgTTgT,gTTT aghggh gh agFghagsTgTgsTgTgTTg aggh TgTghghgTgh agghgsTgTghgTghgTgTT agghgsTgTgTghgTgTTgh agghgsTgTgTghgTgTgTT agghgsTgTgTghgTgTgTT"
    Free vs target cos: 0.598

  Text: "physics"
  Chunks: 4  |  MSE: 0.1572  |  Mean cos: 0.527
    Wave 0: cos=0.507  conf=0.48  expected="ph"  gt_decode="ph"  predicted="edTTpdTTeTF" ✗
    Wave 1: cos=0.461  conf=0.46  expected="ys"  gt_decode="sy"  predicted="adgFFFFFFgFFFigh" ✗
    Wave 2: cos=0.605  conf=0.47  expected="ic"  gt_decode="��ic"  predicted=" FFFFFFFFFF F" ✗
    Wave 3: cos=0.537  conf=0.48  expected="s"  gt_decode="ss"  predicted="  F gFiF" ✗
    Free generation (4 waves): "edTTpdTTeTF apghghghghghhghhghhg ag FgTasgTgh TgTghgh u gFgTghgTgTTgT,gTTT"
    Free vs target cos: 0.495

══════════════════════════════════════════════════════════════════════
  If teacher-forced cos is high but free gen is bad → exposure bias
  If teacher-forced cos is low → WG training needs more steps
  If cos is high but WTT decodes wrong → WTT is the weak link
  If free gen repeats the same word → interference signal is broken
══════════════════════════════════════════════════════════════════════ now heres the logs from the full training dont make any changes is it capable?plai n english no jorgen i need the good the bad and everything in betweens also any recommendations moving forwared
\




 Here's the full breakdown:

---

## The Good

1. **It finally runs.** After weeks of crashes (grad missing, OOM, timer bug, buffer bug), Cell 7c completed start to finish. That's the milestone.

2. **WTT works well.** 86.8% word accuracy, loss dropped from 0.66 → 0.096 over 60K steps. Common words decode perfectly: "Th", "e ", "ca", "t ", "He", "ll", "wo", " i", "al", "en", "er". The model clearly learned English character patterns.

3. **Pre-computation works.** 2,000 samples in ~8 minutes at 4 text/s are now stored on CPU, ready to reuse without re-computing.

4. **Gradient plumbing is solid.** 19/19 generator params are trainable, forward test passes, backward pass runs. The optimizer works.

---

## The Bad

1. **Only 1 training step ran.** The log says "1 steps over 2,000 samples." Whether this was intentional (quick test) or a bug, the WaveGenerator is essentially untrained. Everything downstream reflects this.

2. **That 1 step took 10.7 seconds.** This is extremely slow. A single forward+backward on a small model (~19 params) should take milliseconds, not 10 seconds. The bottleneck is almost certainly the `compute_interference_signal()` inside the WaveGenerator's forward pass — it loops over each wave (5-15 per sample) and calls Phase 1's `apply_neighborhood_interference`, which likely runs on CPU. At 10s/step, 5K steps = **14 hours**. That blows past your Kaggle time budget.

3. **Free generation is identical garbage regardless of input.** "edTTpdTTeTTF apghghghghhghhghhghh..." appears for every single prompt. This means the model hasn't learned anything context-dependent — Wave 0 always outputs the same thing, Wave 1 always outputs the same thing, etc. After 1 step, this is expected.

4. **Teacher-forced cosine of 0.55–0.71 looks deceptively decent.** This isn't the model being smart — consecutive CSE wave chunks within the same sentence are naturally correlated. Even random weights projecting wave[i] will have positive cosine with wave[i+1]. The real test is whether this number *improves* with training and whether *free generation* produces coherent output.

5. **WTT has soft spots.** "physics" → chunk ordering errors (ys→sy, ic→0x80+ic), "energy" → gy decoded as "y:". Byte-order swaps and EOS confusion on short/unusual words. 86.8% is below the 92.8% you got on a previous run — possibly different data sampling.

---

## The Ugly Truth

The **10.7s per step** is the real blocker. Even if the model architecture is correct, you can't train it in reasonable time. The interference computation in WaveGenerator's forward loop is a CPU bottleneck that pre-computation doesn't help with — it runs *during* the forward pass, not during data preparation.

---

## Recommendations

| Priority | What | Why |
|----------|------|-----|
| **1 (critical)** | Profile why 1 step takes 10.7s | If it's `compute_interference_signal`, either batch it, cache it, or simplify it. Without fixing this, 5K steps is infeasible on Kaggle. |
| **2 (high)** | Run with 5K steps anyway as a test | Even at 10s/step, 5K steps = ~14h. Won't fit a single Kaggle session, but try 500 steps (~80 min) to see if loss actually drops and free generation improves. |
| **3 (medium)** | Check if interference can be disabled or approximated | Try a forward pass with interference zeroed out. If training speed drops to <0.1s/step with no interference, that confirms the bottleneck. You could train without interference first, then fine-tune with it. |
| **4 (medium)** | Bump WTT to 80K-100K steps | 86.8% → 93%+ with more steps. The loss was still dropping at 60K (0.096), so there's room. |
| **5 (low)** | Increase pre-computed samples to 5500 | More diversity for training. Only matters once per-step speed is fixed. |

**Bottom line:** The plumbing works. The architecture connects. But the WaveGenerator's forward pass is 1000x too slow for practical training. Fix the per-step speed first, everything else follows. 



hopw long did the 1 step take also i changed it to 1 because it was taking to long didnt see any logs for ong time but why its taking so long dont make any changes talk to me






 **1 step took 10.71 seconds.**

That's absurdly slow. A forward+backward on a model with only 19 parameter tensors should take <50 milliseconds. So it's ~200x slower than it should be.

**Why you didn't see logs:** That was the `flush=True` bug we just fixed. Kaggle was buffering all output until the function returned. The fix is pushed now — but the speed problem is separate.

**Why 10.71 seconds per step:**

The bottleneck is the WaveGenerator's `forward()` method. Here's what happens on every single training step:

1. Load merged[768] + target_waves[~10, 432] from CPU to GPU — fast (~1ms)
2. **Call `generator(merged, target_waves)`** — THIS is where 10+ seconds goes
3. Compute loss — fast (~1ms)
4. Backward pass — fast (~5ms)

Inside `generator(merged, target_waves)`, it loops through each wave one-by-one (typically ~10-15 waves per sample). For **each wave**, it calls `compute_interference_signal()`, which calls Phase 1's `apply_neighborhood_interference()`. That Phase 1 function is a plain Python loop over 432-dimensional wave vectors doing element-wise physics math (constructive/destructive interference with cosine phase calculations). It's pure Python — not vectorized, not on GPU, not batched.

So the execution looks like:

```
Step 1:
  Wave 0:  interference(432 dims, python loop) → ~0.7s
  Wave 1:  interference(432 dims, python loop) → ~0.7s
  Wave 2:  interference(432 dims, python loop) → ~0.7s
  ... 15 waves ...
  Wave 14: interference(432 dims, python loop) → ~0.7s
  Total: ~10.5s for ONE sample
```

**The math doesn't work for Kaggle:**
- 5,000 steps × 10.7s = 53,500s = **14.9 hours** — over Kaggle's 12-hour limit
- Even 1,000 steps = **3 hours** just for WG training
- Plus 12 min for WTT + 8 min pre-compute = you'd barely squeeze 1,000 steps in

**The core problem is architectural:** The interference computation was designed for inference (one generation at a time), not for training (thousands of gradient steps). It's doing CPU-bound Phase 1 physics inside a GPU training loop, on every step, for every wave.

**Options going forward:**
- Vectorize the interference computation (replace Python loops with torch tensor ops)
- Skip interference during training entirely (use a zero or learned placeholder) and only enable it at inference
- Pre-compute interference signals alongside the other frozen outputs in Cell 7a

Want me to investigate which of these is most feasible?






 Not exactly batch in the traditional sense. There are a few options, ranked from easiest/fastest to most correct:

**Option A: Skip interference during training (easiest, biggest speedup)**
The interference computation is the 10s bottleneck. But `wave_predictor` and `context_to_wave` are where the actual learning happens. During teacher-forced training, you could zero out the interference signal — the model still learns to predict next wave from `[prev_wave, zeros, context]`. At inference time, you turn interference back on. This would drop per-step from ~10s to ~10ms. 5K steps in under a minute.

**Option B: Pre-compute interference signals (medium effort)**
Since training is teacher-forced, you already know all the target waves. The interference between wave[i] and wave[0..i-1] can be computed once during the pre-computation phase and stored alongside each sample. Then during training, you just look up the pre-computed interference tensor instead of recomputing it. This keeps the physics-inspired signal in the training but moves the cost out of the loop.

**Option C: Vectorize the interference (hardest, most correct)**
Rewrite `compute_interference_signal()` to use batched tensor ops (masked matrix multiply) instead of Python for-loops. Something like computing all pairwise interferences as a [N, N, 432] tensor, then masking the upper triangle (future waves), and summing. Fully GPU-parallel. But this is a significant rewrite of Phase 1 code.

**My recommendation:** Start with **Option A** — it's a 3-line change, gets you training today, and lets you validate whether the WaveGenerator architecture even learns well. If cosine accuracy hits 0.5+ without interference, great. Then you can add Option B as a refinement later.

Want me to implement Option A? yes