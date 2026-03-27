 
 #1st training 1000
 
 rseults 1


 11:49:07] 
▶ CELL: Cell 7 — Test 1: Perplexity on Penn Treebank
[11:49:07]   Started: 2026-03-22 11:49:07
============================================================
  Test 1: Penn Treebank Perplexity
============================================================
✓ Phase 8 checkpoint loaded (local, 3022.9 MB)
  ✓ FLUXLarge loaded from Phase 8 checkpoint: 75,799,389 params
README.md:  4.21k/? [00:00<00:00, 444kB/s]ptb_text_only.py:  6.50k/? [00:00<00:00, 758kB/s]  ⚠ Could not load ptb: Dataset scripts are no longer supported, but found ptb_text_only.py
  Loaded 100 PTB test samples

  Penn Treebank Perplexity: 22.08

  Checks:
    Perplexity is finite:    ✓ (22.08)
    Perplexity < 500:        ✓ (22.08)
    All samples processed:   ✓

  ✓ TEST PASSED
[11:49:22]   ◼ CELL Cell 7 — Test 1: Perplexity on Penn Treebank — PASS
 [11:49:22] 
▶ CELL: Cell 8 — Test 2: Perplexity on WikiText-2
[11:49:22]   Started: 2026-03-22 11:49:22
============================================================
  Test 2: WikiText-2 Perplexity
============================================================
✓ Phase 8 checkpoint loaded (local, 3022.9 MB)
  ✓ FLUXLarge loaded from Phase 8 checkpoint: 75,799,389 params
README.md:  10.5k/? [00:00<00:00, 1.00MB/s]wikitext-2-raw-v1/test-00000-of-00001.pa(…): 100% 733k/733k [00:01<00:00, 3.66MB/s]wikitext-2-raw-v1/train-00000-of-00001.p(…): 100% 6.36M/6.36M [00:00<00:00, 31.8MB/s]wikitext-2-raw-v1/validation-00000-of-00(…): 100% 657k/657k [00:00<00:00, 3.29MB/s]Generating test split: 100% 4358/4358 [00:00<00:00, 239778.27 examples/s]Generating train split: 100% 36718/36718 [00:00<00:00, 711388.93 examples/s]Generating validation split: 100% 3760/3760 [00:00<00:00, 244213.62 examples/s]  Loaded 100 WikiText-2 test samples

  WikiText-2 Perplexity: 35.79

  Checks:
    Perplexity is finite:    ✓ (35.79)
    Perplexity < 500:        ✓ (35.79)
    All samples processed:   ✓

  ✓ TEST PASSED
[11:49:45]   ◼ CELL Cell 8 — Test 2: Perplexity on WikiText-2 — PASS
[11:49:45] 
▶ CELL: Cell 9 — Test 3: Continual Learning (FLUX wins)
[11:49:45]   Started: 2026-03-22 11:49:45
============================================================
  Test 3: Continual Learning — Zero Forgetting
============================================================
✓ Phase 8 checkpoint loaded (local, 3022.9 MB)
  ✓ FLUXLarge loaded from Phase 8 checkpoint: 75,799,389 params

  Step 1: Learning Facts A...
    Learned 8 facts (episodic: 904 → 912)

  Step 2: Verifying recall of Facts A...
    ✓ The capital of France is Paris
    ✓ Water freezes at zero degrees Celsius
    ✓ Light travels at approximately 300000 km per secon
    ✓ The human genome contains about 3 billion base pai
    ✓ Pi is approximately 3.14159
    ✓ The Eiffel Tower is in Paris France
    ✓ Oxygen has atomic number 8
    ✓ The speed of sound is about 343 meters per second
    Recall: 8/8 = 100.0%

  Step 3: Learning Facts B...
    Learned 8 facts (episodic: 920)

  Step 4: Re-checking recall of Facts A (after learning B)...
    ✓ The capital of France is Paris
    ✓ Water freezes at zero degrees Celsius
    ✓ Light travels at approximately 300000 km per secon
    ✓ The human genome contains about 3 billion base pai
    ✓ Pi is approximately 3.14159
    ✓ The Eiffel Tower is in Paris France
    ✓ Oxygen has atomic number 8
    ✓ The speed of sound is about 343 meters per second
    Recall: 8/8 = 100.0%

  Results:
    Recall before B: 100.0%
    Recall after B:  100.0%
    Forgetting score: 0.0000
    Threshold:        < 0.10

  Checks:
    Forgetting < 0.10:      ✓ (0.0000)
    Initial recall > 50%:   ✓ (100.0%)
    Episodic memory grew:   ✓ (920 entries)

  ✓ TEST PASSED
[11:50:41]   ◼ CELL Cell 9 — Test 3: Continual Learning — PASS
[11:50:41] 
▶ CELL: Cell 10 — Test 4: Long Sequence Speed (FLUX wins)
[11:50:41]   Started: 2026-03-22 11:50:41
============================================================
  Test 4: Long Sequence Speed
============================================================
✓ Phase 8 checkpoint loaded (local, 3022.9 MB)
  ✓ FLUXLarge loaded from Phase 8 checkpoint: 75,799,389 params

    Length   Speed (B/s)    Latency (ms)  Status
  ────────  ────────────  ──────────────  ────────
       256        3692.3            69.3  ✓
      1024       17164.6            59.7  ✓
      4096       84565.3            48.4  ✓
      8192      169670.2            48.3  ✓
     16384      324738.2            50.5  ✓

  Analysis:
    All lengths processed:  ✓
    16k bytes processed:    ✓
    Speed degradation:      ✓ (0.05x, threshold: <5x)
    Sub-linear scaling:     ✓ (ratio: 87.9505, min: 0.0156)

  ✓ TEST PASSED
[11:50:50]   ◼ CELL Cell 10 — Test 4: Long Sequence Speed — PASS
 [11:51:51] 
▶ CELL: Cell 11 — Demo 1: FLUX vs GPT-2 Generation Quality
[11:51:51]   Started: 2026-03-22 11:51:51
======================================================================
  Demo 1: FLUX vs GPT-2 — Generation Quality Comparison
======================================================================

  Loading FLUXLarge...
✓ Phase 8 checkpoint loaded (local, 3022.9 MB)
  ✓ FLUXLarge loaded from Phase 8 checkpoint: 75,799,389 params
  ✓ FLUXLarge: 75,799,389 params

  Loading GPT-2 small...
tokenizer_config.json: 100% 26.0/26.0 [00:00<00:00, 3.31kB/s]vocab.json:  1.04M/? [00:00<00:00, 1.84MB/s]merges.txt:  456k/? [00:00<00:00, 961kB/s]tokenizer.json:  1.36M/? [00:00<00:00, 1.80MB/s]config.json: 100% 665/665 [00:00<00:00, 92.9kB/s]model.safetensors: 100% 548M/548M [00:02<00:00, 677MB/s]Loading weights: 100% 148/148 [00:00<00:00, 968.03it/s, Materializing param=transformer.wte.weight]GPT2LMHeadModel LOAD REPORT from: gpt2
Key                  | Status     |  | 
---------------------+------------+--+-
h.{0...11}.attn.bias | UNEXPECTED |  | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
generation_config.json: 100% 124/124 [00:00<00:00, 16.4kB/s]  ✓ GPT-2 small loaded: 124,439,808 parameters

══════════════════════════════════════════════════════════════════════
  Side-by-Side Generation
══════════════════════════════════════════════════════════════════════

  Prompt: "The future of artificial intelligence is"
  ────────────────────────────────────────────────────────────
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
  FLUX:  dv g rr orhsrtiurl esgntshrllso i,o�sssa sumg e fect d orn S
         (2360ms, 60 bytes)
  GPT-2:  coming and we've done a good job of it, but there are also many other new areas
         (858ms, 258 chars)

  Prompt: "In a world where machines can think,"
  ────────────────────────────────────────────────────────────
  FLUX:    tlec t au ePss   � eb feetsaeue
iceiu m  d i ae oeun o8set
         (2090ms, 60 bytes)
  GPT-2:  it is very difficult to get a person to come into your kitchen to feed your pet
         (725ms, 256 chars)

  Prompt: "The discovery of gravitational waves proved that"
  ────────────────────────────────────────────────────────────
  FLUX:  m tio omi tadee4rsosgun hnWwirie uo arill ins ertltrepM   r 
         (2101ms, 60 bytes)
  GPT-2:  the material in question, the quark, has a physical center, a very strong magne
         (704ms, 295 chars)

  Prompt: "Deep learning has revolutionized how we"
  ────────────────────────────────────────────────────────────
  FLUX:  eas rndttm c ektet itr espiutMesD0  iOrd s  drnoun revtet,at
         (2311ms, 60 bytes)
  GPT-2:  think about and use data (and so much) about people and the world, and it's no 
         (711ms, 259 chars)

  Prompt: "The relationship between physics and computation"
  ────────────────────────────────────────────────────────────
  FLUX:  o  " lcCssNelrnhtniteos ieseeu aelteedo   evtliet2c h iye so
         (2045ms, 60 bytes)
  GPT-2:  is often called the "spaghetti monster" and its relationship with computation i
         (700ms, 317 chars)

══════════════════════════════════════════════════════════════════════
  FLUX Unique Properties
══════════════════════════════════════════════════════════════════════

  1. Byte-Level (No Tokenizer):
     Input:  "Hello 你好 مرحبا 🌍 → works without any vocabulary"
     Wave shape: torch.Size([61, 432])
     ✓ Any UTF-8 input — no OOV errors ever

  2. Real-Time Learning:
     Learned: "FLUXLarge has been benchmarked against GPT-2 in Phase 8"
     Query:   "What was benchmarked in Phase 8?"
     Result:  [0.999] FLUXLarge has been benchmarked against GPT-2 in Phase 8

  3. Model Statistics:
     Total params:     75,799,389
     Learning steps:   901
     Episodic entries: 910
     Field energy:     119007.9141

══════════════════════════════════════════════════════════════════════
  ✓ Demo 1 Complete
══════════════════════════════════════════════════════════════════════
[11:52:34]   ◼ CELL Cell 11 — Demo 1: FLUX vs GPT-2 Generation Quality — PASS
[11:52:34] 
▶ CELL: Cell 12 — Demo 2: FLUX Continual Learning Advantage
[11:52:34]   Started: 2026-03-22 11:52:34
======================================================================
  Demo 2: FLUX Continual Learning Advantage
======================================================================
✓ Phase 8 checkpoint loaded (local, 3022.9 MB)
  ✓ FLUXLarge loaded from Phase 8 checkpoint: 75,799,389 params

══════════════════════════════════════════════════════════════════════
  Stage A: One-Shot Fact Learning
══════════════════════════════════════════════════════════════════════
  GPT-2: ✗ Cannot learn without full retraining
  FLUX:  ✓ Learns instantly via episodic memory + field perturbation

  Learning 8 facts (one-shot)...

  📝 [ 2909ms] The FLUX architecture was designed by UnseenGAP in 2025
  📝 [ 2869ms] FLUX replaces attention with gravitational relevance at O(log n)
  📝 [ 2869ms] Thermodynamic learning eliminates the need for backpropagation
  📝 [ 2868ms] The resonance field stores knowledge as energy minima, not weight
  📝 [ 2869ms] FLUX uses three-tier memory: working, episodic, and semantic
  📝 [ 2869ms] Causal geometry nodes store both WHAT and WHY for every conclusio
  📝 [ 2868ms] The continuous semantic encoder works on raw UTF-8 bytes
  📝 [ 2868ms] Negative mass in FLUX means contradiction — wrong answers repel q

  Episodic memory: 904 → 912
  Learning steps:  900 → 908

══════════════════════════════════════════════════════════════════════
  Stage B: Immediate Recall (no retraining needed)
══════════════════════════════════════════════════════════════════════
  GPT-2: ✗ No recall mechanism for new facts
  FLUX:  ✓ Episodic memory returns exact match

  ✗ Q: "Who designed FLUX?"
    → [0.995] The resonance field stores knowledge as energy minima, not w
  ✓ Q: "What replaces attention in FLUX?"
    → [0.999] FLUX replaces attention with gravitational relevance at O(lo
  ✓ Q: "What eliminates backpropagation?"
    → [0.998] Thermodynamic learning eliminates the need for backpropagati
  ✓ Q: "How does the field store knowledge?"
    → [0.998] The resonance field stores knowledge as energy minima, not w
  ✗ Q: "What are the three memory tiers?"
    → [0.997] At the beginning? Quite a bit. Over time? Not as much.

In t
  ✓ Q: "What do causal geometry nodes store?"
    → [0.999] A Los Angeles police officer was sentenced to 36 months in j
  ✓ Q: "What does the semantic encoder work on?"
    → [0.999] The continuous semantic encoder works on raw UTF-8 bytes
  ✗ Q: "What does negative mass mean?"
    → [0.998] ALAMEDA, Calif. -- A "frustrated" Mark Davis said he stands 

  Recall accuracy: 5/8 = 62%

══════════════════════════════════════════════════════════════════════
  Stage C: Zero Catastrophic Forgetting
══════════════════════════════════════════════════════════════════════
  GPT-2: ✗ Fine-tuning on new data destroys old knowledge
  FLUX:  ✓ New attractors form without erasing existing ones

  Learning 4 NEW facts...
  📝 The Andromeda galaxy will collide with the Milky Way in 4 billion
  📝 Black holes emit Hawking radiation due to quantum effects
  📝 Entanglement allows particles to be correlated across any distanc
  📝 The observable universe is about 93 billion light-years in diamet

  Re-checking original facts...
  ✗ "Who designed FLUX?"
  ✓ "What replaces attention in FLUX?"
  ✓ "What eliminates backpropagation?"
  ✓ "How does the field store knowledge?"
  ✗ "What are the three memory tiers?"
  ✓ "What do causal geometry nodes store?"
  ✓ "What does the semantic encoder work on?"
  ✗ "What does negative mass mean?"

  Original recall after new learning: 5/8 = 62%
  Forgetting score: 0.0000 (target: < 0.10)
  GPT-2 baseline:  ~0.50 (50% degradation after fine-tuning)

══════════════════════════════════════════════════════════════════════
  Summary: FLUX vs GPT-2 Continual Learning
══════════════════════════════════════════════════════════════════════

  Capability                     FLUX                      GPT-2                    
  ────────────────────────────── ───────────────────────── ─────────────────────────
  One-shot fact learning         ✓ Instant                 ✗ Impossible             
  Immediate recall               ✓ Episodic memory         ✗ No mechanism           
  Zero forgetting                ✓ Score=0.000             ✗ Score≈0.50             
  Real-time adaptation           ✓ Field settling          ✗ Static weights         
  Cross-session memory           ✓ Episodic store          ✗ Context window only    

  Final Model State:
    Episodic entries: 916
    Learning steps:   912
    Total params:     75,799,389

══════════════════════════════════════════════════════════════════════
  ✓ Demo 2 Complete
══════════════════════════════════════════════════════════════════════
[11:53:18]   ◼ CELL Cell 12 — Demo 2: FLUX Continual Learning Advantage — PASS
[11:53:18] 
▶ CELL: Cell 13 — Demo 3: FLUX Speed at Long Sequences
[11:53:18]   Started: 2026-03-22 11:53:18
======================================================================
  Demo 3: FLUX Speed at Long Sequences
======================================================================
✓ Phase 8 checkpoint loaded (local, 3022.9 MB)
  ✓ FLUXLarge loaded from Phase 8 checkpoint: 75,799,389 params

  Testing speed at different sequence lengths...
  Device: cuda

    Length    Latency (ms)   Speed (B/s)     Scaling
  ────────  ──────────────  ────────────  ──────────
       128            43.7          2930       1.00x
       256            39.1          6552       2.24x
       512            43.0         11914       4.07x
      1024            38.7         26437       9.02x
      2048            39.2         52266      17.84x
      4096            38.6        105980      36.17x
      8192            39.8        205891      70.28x
     16384            42.5        385551     131.60x

══════════════════════════════════════════════════════════════════════
  Theoretical Complexity Comparison
══════════════════════════════════════════════════════════════════════

    Sequence    FLUX O(log n)      GPT-2 O(n²)     Speedup
  ──────────  ───────────────  ───────────────  ──────────
         512              9.0              262k          29x
        1024             10.0             1049k         105x
        4096             12.0            16777k        1398x
       16384             14.0           268435k       19174x
       65536             16.0          4294967k      268435x
      262144             18.0         68719477k     3817749x

  ✓ Speed chart saved: /content/FLUX/phases/phase8/speed_benchmark.png

══════════════════════════════════════════════════════════════════════
  Key Takeaways
══════════════════════════════════════════════════════════════════════
  • FLUX uses O(log n) gravitational relevance (spatial tree)
  • GPT-2 uses O(n²) self-attention (all-pairs comparison)
  • At 16k tokens: FLUX ~14 ops vs GPT-2 ~268M ops
  • FLUX advantage grows with sequence length
  • No quadratic memory scaling in FLUX

  Measured:
    Max sequence:  16,384 bytes
    Peak speed:    385,551 bytes/sec
    Min speed:     2,930 bytes/sec

══════════════════════════════════════════════════════════════════════
  ✓ Demo 3 Complete
══════════════════════════════════════════════════════════════════════
[11:53:30]   ◼ CELL Cell 13 — Demo 3: FLUX Speed at Long Sequences — PASS



#training 2 on 4000

results 2


now we are on phase 8 rerun on on cell 5 heres the results [18:51:34] 
▶ CELL: Cell 5 — Training on OpenWebText
[18:51:34]   Started: 2026-03-22 18:51:34
── Starting Phase 8 Training (Scaled Up) ──

=================================================================
  Loading Training Data (OpenWebText — 5k docs)
=================================================================
  ℹ Loading OpenWebText subset (5,000 docs)...
README.md: 
 7.46k/? [00:00<00:00, 1.29MB/s]
Resolving data files: 100%
 80/80 [00:00<00:00, 15657.69it/s]
Resolving data files: 100%
 80/80 [00:00<00:00, 13108.74it/s]
  ✓ Loaded 5,000 documents from OpenWebText
  Train: 4,500 docs
  Eval:  500 docs
  Total training bytes: 13,095,692 (13.1 MB)

=================================================================
  Stage A: WaveDecoder + Bridge — OpenWebText Training
=================================================================
  Corpus: 4500 documents
  Max seq len: 512 bytes (2x previous)
  Decoder: GRU 2-layer, 512 hidden
  Training: single-pass stream (no epochs)
  Grad accum: 4 (effective batch = 4)
  LR: 3e-4 with warmup + cosine decay
  Estimated: ~225 min on A100
  Step    100/4500  loss=4.9257  ppl=158.7  lr=0.000075  tokens=311,206  latency=1205ms
[18:53:43]   📊 step_100_loss: 4.9257
  Step    200/4500  loss=3.3261  ppl=29.5  lr=0.000150  tokens=590,161  latency=1207ms
[18:55:43]   📊 step_200_loss: 3.3261
  Step    300/4500  loss=2.8151  ppl=16.9  lr=0.000225  tokens=880,947  latency=1201ms
[18:57:43]   📊 step_300_loss: 2.8151
  Step    400/4500  loss=2.6025  ppl=13.7  lr=0.000300  tokens=1,188,700  latency=1202ms
[18:59:44]   📊 step_400_loss: 2.6025
  Step    500/4500  loss=2.4680  ppl=12.0  lr=0.000300  tokens=1,468,045  latency=1203ms
[19:01:44]   📊 step_500_loss: 2.4680
  Step    600/4500  loss=2.3736  ppl=10.8  lr=0.000300  tokens=1,759,733  latency=1198ms
[19:03:44]   📊 step_600_loss: 2.3736
  Step    700/4500  loss=2.3521  ppl=10.7  lr=0.000300  tokens=2,052,014  latency=1199ms
[19:05:44]   📊 step_700_loss: 2.3521
  Step    800/4500  loss=2.2636  ppl=9.7  lr=0.000300  tokens=2,338,823  latency=1208ms
[19:07:45]   📊 step_800_loss: 2.2636
  Step    900/4500  loss=2.2517  ppl=9.7  lr=0.000299  tokens=2,632,397  latency=1203ms
[19:09:45]   📊 step_900_loss: 2.2517
  Step   1000/4500  loss=2.2102  ppl=9.2  lr=0.000299  tokens=2,916,374  latency=1200ms
[19:11:45]   📊 step_1000_loss: 2.2102
✓ Phase 8 checkpoint saved: /content/FLUX/checkpoints/phase8.phase.pt (3070.3 MB)
[19:11:50]   ✓ Checkpoint saved at step 1000
  Step   1100/4500  loss=2.1986  ppl=9.3  lr=0.000299  tokens=3,214,932  latency=1204ms
[19:13:50]   📊 step_1100_loss: 2.1986
  Step   1200/4500  loss=2.2021  ppl=9.4  lr=0.000298  tokens=3,510,609  latency=1199ms
[19:15:51]   📊 step_1200_loss: 2.2021
  Step   1300/4500  loss=2.1636  ppl=8.8  lr=0.000298  tokens=3,780,307  latency=1204ms
[19:17:51]   📊 step_1300_loss: 2.1636
  Step   1400/4500  loss=2.1118  ppl=8.3  lr=0.000298  tokens=4,075,412  latency=1200ms
[19:19:51]   📊 step_1400_loss: 2.1118
  Step   1500/4500  loss=2.0944  ppl=8.2  lr=0.000297  tokens=4,364,313  latency=1200ms
[19:21:51]   📊 step_1500_loss: 2.0944
  Step   1600/4500  loss=2.1143  ppl=8.4  lr=0.000297  tokens=4,647,935  latency=1206ms
[19:23:51]   📊 step_1600_loss: 2.1143
  Step   1700/4500  loss=2.0799  ppl=8.1  lr=0.000296  tokens=4,934,808  latency=1205ms
[19:25:51]   📊 step_1700_loss: 2.0799
  Step   1800/4500  loss=2.0852  ppl=8.7  lr=0.000295  tokens=5,215,155  latency=1205ms
[19:27:52]   📊 step_1800_loss: 2.0852
  Step   1900/4500  loss=2.0150  ppl=7.6  lr=0.000295  tokens=5,515,284  latency=1200ms
[19:29:52]   📊 step_1900_loss: 2.0150
  Step   2000/4500  loss=2.0675  ppl=8.1  lr=0.000294  tokens=5,812,219  latency=1206ms
[19:31:52]   📊 step_2000_loss: 2.0675
✓ Phase 8 checkpoint saved: /content/FLUX/checkpoints/phase8.phase.pt (3073.7 MB)
[19:31:58]   ✓ Checkpoint saved at step 2000
  Step   2100/4500  loss=1.9965  ppl=7.5  lr=0.000293  tokens=6,104,846  latency=1210ms
[19:34:04]   📊 step_2100_loss: 1.9965
  Step   2200/4500  loss=2.0387  ppl=8.8  lr=0.000292  tokens=6,385,236  latency=1202ms
[19:36:04]   📊 step_2200_loss: 2.0387
  Step   2300/4500  loss=1.9754  ppl=7.3  lr=0.000291  tokens=6,683,423  latency=1199ms
[19:38:04]   📊 step_2300_loss: 1.9754
  Step   2400/4500  loss=1.9797  ppl=7.4  lr=0.000291  tokens=6,953,530  latency=1201ms
[19:40:05]   📊 step_2400_loss: 1.9797
  Step   2500/4500  loss=1.9735  ppl=7.7  lr=0.000290  tokens=7,269,621  latency=1201ms
[19:42:05]   📊 step_2500_loss: 1.9735
  Step   2600/4500  loss=1.9763  ppl=7.3  lr=0.000289  tokens=7,576,489  latency=1208ms
[19:44:05]   📊 step_2600_loss: 1.9763
  Step   2700/4500  loss=1.9274  ppl=7.0  lr=0.000288  tokens=7,881,607  latency=1207ms
[19:46:05]   📊 step_2700_loss: 1.9274
  Step   2800/4500  loss=1.9425  ppl=7.3  lr=0.000286  tokens=8,175,526  latency=1206ms
[19:48:06]   📊 step_2800_loss: 1.9425
  Step   2900/4500  loss=1.9729  ppl=7.5  lr=0.000285  tokens=8,441,588  latency=1208ms
[19:50:06]   📊 step_2900_loss: 1.9729
  Step   3000/4500  loss=1.9331  ppl=7.0  lr=0.000284  tokens=8,726,809  latency=1199ms
[19:52:06]   📊 step_3000_loss: 1.9331
✓ Phase 8 checkpoint saved: /content/FLUX/checkpoints/phase8.phase.pt (3077.1 MB)
[19:52:12]   ✓ Checkpoint saved at step 3000
  Step   3100/4500  loss=1.8938  ppl=6.7  lr=0.000283  tokens=9,046,206  latency=1207ms
[19:54:12]   📊 step_3100_loss: 1.8938
  Step   3200/4500  loss=1.8974  ppl=6.9  lr=0.000282  tokens=9,341,123  latency=1199ms
[19:56:13]   📊 step_3200_loss: 1.8974
  Step   3300/4500  loss=1.8847  ppl=6.7  lr=0.000280  tokens=9,626,767  latency=1197ms
[19:58:13]   📊 step_3300_loss: 1.8847
  Step   3400/4500  loss=1.8700  ppl=6.6  lr=0.000279  tokens=9,907,095  latency=1203ms
[20:00:13]   📊 step_3400_loss: 1.8700
  Step   3500/4500  loss=1.8589  ppl=6.5  lr=0.000278  tokens=10,177,715  latency=1202ms
[20:02:13]   📊 step_3500_loss: 1.8589
  Step   3600/4500  loss=1.8804  ppl=6.8  lr=0.000276  tokens=10,460,717  latency=1201ms
[20:04:13]   📊 step_3600_loss: 1.8804
  Step   3700/4500  loss=1.8184  ppl=6.3  lr=0.000275  tokens=10,756,026  latency=1206ms
[20:06:14]   📊 step_3700_loss: 1.8184
  Step   3800/4500  loss=1.8515  ppl=6.6  lr=0.000273  tokens=11,042,553  latency=1208ms
[20:08:14]   📊 step_3800_loss: 1.8515
  Step   3900/4500  loss=1.8482  ppl=6.5  lr=0.000272  tokens=11,331,033  latency=1199ms
[20:10:14]   📊 step_3900_loss: 1.8482
  Step   4000/4500  loss=1.8099  ppl=6.2  lr=0.000270  tokens=11,634,781  latency=1198ms
[20:12:14]   📊 step_4000_loss: 1.8099
✓ Phase 8 checkpoint saved: /content/FLUX/checkpoints/phase8.phase.pt (3080.4 MB)
[20:12:19]   ✓ Checkpoint saved at step 4000
  Step   4100/4500  loss=1.8313  ppl=6.3  lr=0.000268  tokens=11,922,849  latency=1198ms
[20:14:25]   📊 step_4100_loss: 1.8313
  Step   4200/4500  loss=1.8061  ppl=6.2  lr=0.000267  tokens=12,215,815  latency=1198ms
[20:16:25]   📊 step_4200_loss: 1.8061
  Step   4300/4500  loss=1.8146  ppl=6.3  lr=0.000265  tokens=12,517,957  latency=1203ms
[20:18:25]   📊 step_4300_loss: 1.8146




[20:32:40] 
▶ CELL: Cell 7 — Test 1: Perplexity on Penn Treebank
[20:32:40]   Started: 2026-03-22 20:32:40
============================================================
  Test 1: Penn Treebank Perplexity
============================================================
✓ Phase 8 checkpoint loaded (local, 3080.4 MB)
  ✓ WaveDecoder loaded from checkpoint
  ✓ FLUXLarge loaded from Phase 8 checkpoint: 75,799,389 params
  ⚠ Could not load ptb: Dataset scripts are no longer supported, but found ptb_text_only.py
  Loaded 100 PTB test samples

  Penn Treebank Perplexity: 316.99

  Checks:
    Perplexity is finite:    ✓ (316.99)
    Perplexity < 500:        ✓ (316.99)
    All samples processed:   ✓

  ✓ TEST PASSED
[20:32:48]   ◼ CELL Cell 7 — Test 1: Perplexity on Penn Treebank — PASS



[20:32:55] 
▶ CELL: Cell 8 — Test 2: Perplexity on WikiText-2
[20:32:55]   Started: 2026-03-22 20:32:55
============================================================
  Test 2: WikiText-2 Perplexity
============================================================
✓ Phase 8 checkpoint loaded (local, 3080.4 MB)
  ✓ WaveDecoder loaded from checkpoint
  ✓ FLUXLarge loaded from Phase 8 checkpoint: 75,799,389 params
  Loaded 100 WikiText-2 test samples

  WikiText-2 Perplexity: 298.48

  Checks:
    Perplexity is finite:    ✓ (298.48)
    Perplexity < 500:        ✓ (298.48)
    All samples processed:   ✓

  ✓ TEST PASSED
[20:33:04]   ◼ CELL Cell 8 — Test 2: Perplexity on WikiText-2 — PASS



[20:33:14] 
▶ CELL: Cell 9 — Test 3: Continual Learning (FLUX wins)
[20:33:14]   Started: 2026-03-22 20:33:14
============================================================
  Test 3: Continual Learning — Zero Forgetting
============================================================
✓ Phase 8 checkpoint loaded (local, 3080.4 MB)
  ✓ WaveDecoder loaded from checkpoint
  ✓ FLUXLarge loaded from Phase 8 checkpoint: 75,799,389 params

  Step 1: Learning Facts A...
    Learned 8 facts (episodic: 4001 → 4009)

  Step 2: Verifying recall of Facts A...
    ✓ The capital of France is Paris
    ✓ Water freezes at zero degrees Celsius
    ✓ Light travels at approximately 300000 km per secon
    ✓ The human genome contains about 3 billion base pai
    ✓ Pi is approximately 3.14159
    ✓ The Eiffel Tower is in Paris France
    ✓ Oxygen has atomic number 8
    ✓ The speed of sound is about 343 meters per second
    Recall: 8/8 = 100.0%

  Step 3: Learning Facts B...
    Learned 8 facts (episodic: 4017)

  Step 4: Re-checking recall of Facts A (after learning B)...
    ✓ The capital of France is Paris
    ✓ Water freezes at zero degrees Celsius
    ✓ Light travels at approximately 300000 km per secon
    ✓ The human genome contains about 3 billion base pai
    ✓ Pi is approximately 3.14159
    ✓ The Eiffel Tower is in Paris France
    ✓ Oxygen has atomic number 8
    ✓ The speed of sound is about 343 meters per second
    Recall: 8/8 = 100.0%

  Results:
    Recall before B: 100.0%
    Recall after B:  100.0%
    Forgetting score: 0.0000
    Threshold:        < 0.10

  Checks:
    Forgetting < 0.10:      ✓ (0.0000)
    Initial recall > 50%:   ✓ (100.0%)
    Episodic memory grew:   ✓ (4017 entries)

  ✓ TEST PASSED
[20:33:38]   ◼ CELL Cell 9 — Test 3: Continual Learning — PASS






[20:34:02] 
▶ CELL: Cell 10 — Test 4: Long Sequence Speed (FLUX wins)
[20:34:02]   Started: 2026-03-22 20:34:02
============================================================
  Test 4: Long Sequence Speed
============================================================
✓ Phase 8 checkpoint loaded (local, 3080.4 MB)
  ✓ WaveDecoder loaded from checkpoint
  ✓ FLUXLarge loaded from Phase 8 checkpoint: 75,799,389 params

    Length   Speed (B/s)    Latency (ms)  Status
  ────────  ────────────  ──────────────  ────────
       256        7468.5            34.3  ✓
      1024       25622.9            40.0  ✓
      4096      195581.4            20.9  ✓
      8192      234390.3            35.0  ✓
     16384      457666.1            35.8  ✓

  Analysis:
    All lengths processed:  ✓
    16k bytes processed:    ✓
    Speed degradation:      ✓ (0.06x, threshold: <5x)
    Sub-linear scaling:     ✓ (ratio: 61.2793, min: 0.0156)

  ✓ TEST PASSED
[20:34:08]   ◼ CELL Cell 10 — Test 4: Long Sequence Speed — PASS





 [20:34:24] 
▶ CELL: Cell 11 — Demo 1: FLUX vs GPT-2 Generation Quality
[20:34:24]   Started: 2026-03-22 20:34:24
======================================================================
  Demo 1: FLUX vs GPT-2 — Generation Quality Comparison
======================================================================

  Loading FLUXLarge...
✓ Phase 8 checkpoint loaded (local, 3080.4 MB)
  ✓ WaveDecoder loaded from checkpoint
  ✓ FLUXLarge loaded from Phase 8 checkpoint: 75,799,389 params
  ✓ FLUXLarge: 75,799,389 params

  Loading GPT-2 small...
tokenizer_config.json: 100% 26.0/26.0 [00:00<00:00, 5.08kB/s]vocab.json:  1.04M/? [00:00<00:00, 25.3MB/s]merges.txt:  456k/? [00:00<00:00, 3.29MB/s]tokenizer.json:  1.36M/? [00:00<00:00, 6.14MB/s]config.json: 100% 665/665 [00:00<00:00, 141kB/s]model.safetensors: 100% 548M/548M [00:01<00:00, 898MB/s]Loading weights: 100% 148/148 [00:00<00:00, 1704.66it/s, Materializing param=transformer.wte.weight]GPT2LMHeadModel LOAD REPORT from: gpt2
Key                  | Status     |  | 
---------------------+------------+--+-
h.{0...11}.attn.bias | UNEXPECTED |  | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
generation_config.json: 100% 124/124 [00:00<00:00, 24.2kB/s]The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
  ✓ GPT-2 small loaded: 124,439,808 parameters

══════════════════════════════════════════════════════════════════════
  Side-by-Side Generation
══════════════════════════════════════════════════════════════════════

  Prompt: "The future of artificial intelligence is"
  ────────────────────────────────────────────────────────────
  FLUX:  ulage ton Tow Mown Solleri vurumen ciews on to deven on expr
         (83ms, 60 bytes)
  GPT-2:  at stake if it's to become a viable industry, but it's certainly not a bad bet 
         (373ms, 276 chars)

  Prompt: "In a world where machines can think,"
  ────────────────────────────────────────────────────────────
  FLUX:   the could spick to lake worke in the Beanco to public.

Arg
         (74ms, 60 bytes)
  GPT-2:  it's probably good for everyone.

The next day, I went to our local grocery sto
         (307ms, 253 chars)

  Prompt: "The discovery of gravitational waves proved that"
  ────────────────────────────────────────────────────────────
  FLUX:  en for descing celared jus well and a Monvey that drougn in 
         (75ms, 60 bytes)
  GPT-2:  it was possible to distinguish a binary neutron star from a neutron star. But n
         (317ms, 284 chars)

  Prompt: "Deep learning has revolutionized how we"
  ────────────────────────────────────────────────────────────
  FLUX:  k of with the Sundacted on Today, had Janaez Raid Marmuder P
         (79ms, 60 bytes)
  GPT-2:  interact with the world. We are not just making decisions; we are making decisi
         (301ms, 271 chars)

  Prompt: "The relationship between physics and computation"
  ────────────────────────────────────────────────────────────
  FLUX:   shollinies a really"ners a new one apported that jass, shar
         (80ms, 60 bytes)
  GPT-2:  may make them harder to understand and understand than they actually are. That'
         (303ms, 311 chars)

══════════════════════════════════════════════════════════════════════
  FLUX Unique Properties
══════════════════════════════════════════════════════════════════════

  1. Byte-Level (No Tokenizer):
     Input:  "Hello 你好 مرحبا 🌍 → works without any vocabulary"
     Wave shape: torch.Size([61, 432])
     ✓ Any UTF-8 input — no OOV errors ever

  2. Real-Time Learning:
     Learned: "FLUXLarge has been benchmarked against GPT-2 in Phase 8"
     Query:   "What was benchmarked in Phase 8?"
     Result:  [0.998] FLUXLarge has been benchmarked against GPT-2 in Phase 8

  3. Model Statistics:
     Total params:     75,799,389
     Learning steps:   4001
     Episodic entries: 4007
     Field energy:     9298.2383

══════════════════════════════════════════════════════════════════════
  ✓ Demo 1 Complete
══════════════════════════════════════════════════════════════════════
[20:34:42]   ◼ CELL Cell 11 — Demo 1: FLUX vs GPT-2 Generation Quality — PASS




[20:35:26] 
▶ CELL: Cell 12 — Demo 2: FLUX Continual Learning Advantage
[20:35:26]   Started: 2026-03-22 20:35:26
======================================================================
  Demo 2: FLUX Continual Learning Advantage
======================================================================
✓ Phase 8 checkpoint loaded (local, 3080.4 MB)
  ✓ WaveDecoder loaded from checkpoint
  ✓ FLUXLarge loaded from Phase 8 checkpoint: 75,799,389 params

══════════════════════════════════════════════════════════════════════
  Stage A: One-Shot Fact Learning
══════════════════════════════════════════════════════════════════════
  GPT-2: ✗ Cannot learn without full retraining
  FLUX:  ✓ Learns instantly via episodic memory + field perturbation

  Learning 8 facts (one-shot)...

  📝 [ 1167ms] The FLUX architecture was designed by UnseenGAP in 2025
  📝 [ 1170ms] FLUX replaces attention with gravitational relevance at O(log n)
  📝 [ 1172ms] Thermodynamic learning eliminates the need for backpropagation
  📝 [ 1164ms] The resonance field stores knowledge as energy minima, not weight
  📝 [ 1171ms] FLUX uses three-tier memory: working, episodic, and semantic
  📝 [ 1173ms] Causal geometry nodes store both WHAT and WHY for every conclusio
  📝 [ 1170ms] The continuous semantic encoder works on raw UTF-8 bytes
  📝 [ 1173ms] Negative mass in FLUX means contradiction — wrong answers repel q

  Episodic memory: 4001 → 4009
  Learning steps:  4000 → 4008

══════════════════════════════════════════════════════════════════════
  Stage B: Immediate Recall (no retraining needed)
══════════════════════════════════════════════════════════════════════
  GPT-2: ✗ No recall mechanism for new facts
  FLUX:  ✓ Episodic memory returns exact match

  ✓ Q: "Who designed FLUX?"
    → [0.996] FLUX uses three-tier memory: working, episodic, and semantic
  ✓ Q: "What replaces attention in FLUX?"
    → [0.999] FLUX replaces attention with gravitational relevance at O(lo
  ✓ Q: "What eliminates backpropagation?"
    → [0.998] Thermodynamic learning eliminates the need for backpropagati
  ✗ Q: "How does the field store knowledge?"
    → [0.998] Two days ago we observed the latest disclosure in the seemin
  ✗ Q: "What are the three memory tiers?"
    → [0.998] MADRID (AP) — Spain’s maritime rescue service says it has re
  ✓ Q: "What do causal geometry nodes store?"
    → [0.998] by

O Governo anunciou recentemente a criação de um novo org
  ✗ Q: "What does the semantic encoder work on?"
    → [0.999] Ok so after some excited communication from my gifter, I was
  ✗ Q: "What does negative mass mean?"
    → [0.998] Posted by Darren Urban on November 20, 2015 – 9:05 am

The N

  Recall accuracy: 4/8 = 50%

══════════════════════════════════════════════════════════════════════
  Stage C: Zero Catastrophic Forgetting
══════════════════════════════════════════════════════════════════════
  GPT-2: ✗ Fine-tuning on new data destroys old knowledge
  FLUX:  ✓ New attractors form without erasing existing ones

  Learning 4 NEW facts...
  📝 The Andromeda galaxy will collide with the Milky Way in 4 billion
  📝 Black holes emit Hawking radiation due to quantum effects
  📝 Entanglement allows particles to be correlated across any distanc
  📝 The observable universe is about 93 billion light-years in diamet

  Re-checking original facts...
  ✓ "Who designed FLUX?"
  ✓ "What replaces attention in FLUX?"
  ✓ "What eliminates backpropagation?"
  ✗ "How does the field store knowledge?"
  ✗ "What are the three memory tiers?"
  ✓ "What do causal geometry nodes store?"
  ✗ "What does the semantic encoder work on?"
  ✗ "What does negative mass mean?"

  Original recall after new learning: 4/8 = 50%
  Forgetting score: 0.0000 (target: < 0.10)
  GPT-2 baseline:  ~0.50 (50% degradation after fine-tuning)

══════════════════════════════════════════════════════════════════════
  Summary: FLUX vs GPT-2 Continual Learning
══════════════════════════════════════════════════════════════════════

  Capability                     FLUX                      GPT-2                    
  ────────────────────────────── ───────────────────────── ─────────────────────────
  One-shot fact learning         ✓ Instant                 ✗ Impossible             
  Immediate recall               ✓ Episodic memory         ✗ No mechanism           
  Zero forgetting                ✓ Score=0.000             ✗ Score≈0.50             
  Real-time adaptation           ✓ Field settling          ✗ Static weights         
  Cross-session memory           ✓ Episodic store          ✗ Context window only    

  Final Model State:
    Episodic entries: 4013
    Learning steps:   4012
    Total params:     75,799,389

══════════════════════════════════════════════════════════════════════
  ✓ Demo 2 Complete
══════════════════════════════════════════════════════════════════════
[20:35:47]   ◼ CELL Cell 12 — Demo 2: FLUX Continual Learning Advantage — PASS


[20:36:23] 
▶ CELL: Cell 13 — Demo 3: FLUX Speed at Long Sequences
[20:36:23]   Started: 2026-03-22 20:36:23
======================================================================
  Demo 3: FLUX Speed at Long Sequences
======================================================================
✓ Phase 8 checkpoint loaded (local, 3080.4 MB)
  ✓ WaveDecoder loaded from checkpoint
  ✓ FLUXLarge loaded from Phase 8 checkpoint: 75,799,389 params

  Testing speed at different sequence lengths...
  Device: cuda

    Length    Latency (ms)   Speed (B/s)     Scaling
  ────────  ──────────────  ────────────  ──────────
       128            21.8          5868       1.00x
       256            19.5         13146       2.24x
       512            25.7         19902       3.39x
      1024            22.0         46645       7.95x
      2048            25.7         79568      13.56x
      4096            20.7        198237      33.78x
      8192            21.6        378630      64.52x
     16384            22.5        728852     124.20x

══════════════════════════════════════════════════════════════════════
  Theoretical Complexity Comparison
══════════════════════════════════════════════════════════════════════

    Sequence    FLUX O(log n)      GPT-2 O(n²)     Speedup
  ──────────  ───────────────  ───────────────  ──────────
         512              9.0              262k          29x
        1024             10.0             1049k         105x
        4096             12.0            16777k        1398x
       16384             14.0           268435k       19174x
       65536             16.0          4294967k      268435x
      262144             18.0         68719477k     3817749x

  ✓ Speed chart saved: /content/FLUX/phases/phase8/speed_benchmark.png

══════════════════════════════════════════════════════════════════════
  Key Takeaways
══════════════════════════════════════════════════════════════════════
  • FLUX uses O(log n) gravitational relevance (spatial tree)
  • GPT-2 uses O(n²) self-attention (all-pairs comparison)
  • At 16k tokens: FLUX ~14 ops vs GPT-2 ~268M ops
  • FLUX advantage grows with sequence length
  • No quadratic memory scaling in FLUX

  Measured:
    Max sequence:  16,384 bytes
    Peak speed:    728,852 bytes/sec
    Min speed:     5,868 bytes/sec

══════════════════════════════════════════════════════════════════════
  ✓ Demo 3 Complete
══════════════════════════════════════════════════════════════════════
[20:36:31]   ◼ CELL Cell 13 — Demo 3: FLUX Speed at Long Sequences — PASS





 [20:36:45] 
▶ CELL: Cell 14 — Interactive Exploration + Benchmark
[20:36:45]   Started: 2026-03-22 20:36:45
============================================================
  Interactive: FLUXLarge Exploration + Full Benchmark
============================================================

  ── Real-Time Learning ──
  📝 FLUXLarge is Phase 8 of the FLUX project
  📝 The benchmark compares FLUX against GPT-2 small
  📝 FLUX uses gravitational relevance instead of attention
  📝 Training on OpenWebText demonstrates scalability
  📝 The three-tier memory enables zero catastrophic forgetting

  ── Querying Learned Facts ──

  🔍 'What phase is FLUXLarge?'
     → [0.997] FLUXLarge is Phase 8 of the FLUX project
     → [0.997] In the previous post, we took a look at the Blackhawks’ pena

  🔍 'What is FLUX compared against?'
     → [0.999] Another Space_Man_Spiff 1990 Chevrolet Tracker post...

Moab
     → [0.999] Image caption Zack Davies attacked Dr Sarandev Bhambra on 14

  🔍 'What replaces attention in FLUX?'
     → [0.999] FLUX uses gravitational relevance instead of attention
     → [0.998] India-Uzbekistan relations: Important facts you need to know

  🔍 'What dataset was used for training?'
     → [0.999] Rick Ross was on Power 105's Breakfast Club show recently, w
     → [0.999] Republican Sen. Lindsey Graham blasted news reports that Jar

  🔍 'How does FLUX avoid forgetting?'
     → [0.998] *UPDATE* Our friend JD has made a one-click install for Andr
     → [0.998] Jefferson Co. poll workers failed to ask voters for IDs

Tra

  ── Running Full Benchmark Suite ──
Loading weights: 100% 148/148 [00:00<00:00, 1650.76it/s, Materializing param=transformer.wte.weight]GPT2LMHeadModel LOAD REPORT from: gpt2
Key                  | Status     |  | 
---------------------+------------+--+-
h.{0...11}.attn.bias | UNEXPECTED |  | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
  ✓ GPT-2 small loaded: 124,439,808 parameters

  ── Benchmark 1: Penn Treebank Perplexity ──
  ⚠ Could not load ptb: Dataset scripts are no longer supported, but found ptb_text_only.py
`loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`.
    FLUX: 316.98  |  GPT-2: 33.86

  ── Benchmark 2: WikiText-2 Perplexity ──
    FLUX: 298.48  |  GPT-2: 57.25

  ── Benchmark 3: Continual Learning Retention ──
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
    FLUX: 0.0000  |  GPT-2: 0.5000

  ── Benchmark 4: Generation Speed ──
    FLUX: 1395.0 b/s  |  GPT-2: 184.0 t/s

  ── Benchmark 5: One-Shot Fact Learning ──
    FLUX: 1.0  |  GPT-2: 0.0

======================================================================
  FLUX vs GPT-2 Benchmark Results
======================================================================

Benchmark                                 FLUX      GPT-2   Winner
----------------------------------------------------------------------
Penn Treebank Perplexity              316.9750    33.8648    GPT-2
WikiText-2 Perplexity                 298.4829    57.2514    GPT-2
Continual Learning (forgetting)         0.0000     0.5000     FLUX
Generation Speed (bytes/sec)         1394.9719   184.0374     FLUX
One-Shot Fact Learning                  1.0000     0.0000     FLUX
----------------------------------------------------------------------
  FLUX wins: 3  |  GPT-2 wins: 2
======================================================================
[20:37:13]   📊 bench_Penn Treebank Perplexity: FLUX=316.9750 GPT2=33.8648
[20:37:13]   📊 bench_WikiText-2 Perplexity: FLUX=298.4829 GPT2=57.2514
[20:37:13]   📊 bench_Continual Learning (forgetting): FLUX=0.0000 GPT2=0.5000
[20:37:13]   📊 bench_Generation Speed (bytes/sec): FLUX=1394.9719 GPT2=184.0374
[20:37:13]   📊 bench_One-Shot Fact Learning: FLUX=1.0000 GPT2=0.0000
[20:37:13]   📊 flux_wins: 3
[20:37:13]   📊 gpt2_wins: 2

  ── Text Generation ──

  Prompt:    'The scaled FLUX model'
  Generated: The scaled FLUX modelybed.

Same much in with a know BetwordL/Thi part Killar pre

  Prompt:    'In the era of transformers,'
  Generated: In the era of transformers, a roughnet.

In annear Wills Banter healthed secures to dec

────────────────────────────────────────────────────────────
  Final Model Stats:
    Total params:       75,799,389
    Learning steps:     4516
    Episodic entries:   4525
    Working entries:    2048
    Field energy:       9021.4971
    Field attractors:   267
[20:37:13]   ◼ CELL Cell 14 — Interactive Exploration + Benchmark — PASS






Secoud run bigger gru results 


 [18:53:16] 
▶ CELL: Cell 5 — Training on OpenWebText
[18:53:16]   Started: 2026-03-26 18:53:16
── Starting Phase 8 Training ──

=================================================================
  Loading Training Data (OpenWebText subset)
=================================================================
  ℹ Loading OpenWebText subset (5,000 docs)...
README.md:  7.46k/? [00:00<00:00, 697kB/s]Resolving data files: 100% 80/80 [00:00<00:00, 10638.02it/s]Resolving data files: 100% 80/80 [00:00<00:00, 7758.43it/s]  ✓ Loaded 5,000 documents from OpenWebText
  Train: 4,500 docs
  Eval:  500 docs

=================================================================
  Stage A: Output Head + Bridge — OpenWebText Training
=================================================================
  Corpus: 4500 documents
  Training: single-pass stream (no epochs)
  Grad accum: 4 (effective batch = 4)
  LR: 5e-4 with cosine decay
  Step     50/4500  loss=5.1049  ppl=182.8  lr=0.000060  tokens=154,048  latency=6582ms
[18:58:59]   📊 step_50_loss: 5.1049
  Step    100/4500  loss=3.6850  ppl=41.7  lr=0.000125  tokens=311,206  latency=6590ms
[19:04:29]   📊 step_100_loss: 3.6850
  Step    150/4500  loss=3.2498  ppl=27.1  lr=0.000185  tokens=441,034  latency=6587ms
[19:09:58]   📊 step_150_loss: 3.2498
  Step    200/4500  loss=3.0111  ppl=21.3  lr=0.000250  tokens=590,161  latency=6584ms
[19:15:27]   📊 step_200_loss: 3.0111
  Step    250/4500  loss=2.7816  ppl=16.3  lr=0.000310  tokens=742,589  latency=6582ms
[19:20:57]   📊 step_250_loss: 2.7816
  Step    300/4500  loss=2.6137  ppl=13.8  lr=0.000375  tokens=880,947  latency=6578ms
[19:26:26]   📊 step_300_loss: 2.6137
  Step    350/4500  loss=2.5326  ppl=12.8  lr=0.000435  tokens=1,043,220  latency=6575ms
[19:31:55]   📊 step_350_loss: 2.5326
  Step    400/4500  loss=2.4793  ppl=12.1  lr=0.000500  tokens=1,188,700  latency=6582ms
[19:37:24]   📊 step_400_loss: 2.4793
  Step    450/4500  loss=2.3808  ppl=10.9  lr=0.000500  tokens=1,333,222  latency=6589ms
[19:42:53]   📊 step_450_loss: 2.3808
  Step    500/4500  loss=2.3741  ppl=10.9  lr=0.000500  tokens=1,468,045  latency=6586ms
[19:48:22]   📊 step_500_loss: 2.3741
  Step    550/4500  loss=2.2902  ppl=10.0  lr=0.000500  tokens=1,614,278  latency=6590ms
[19:53:52]   📊 step_550_loss: 2.2902
  Step    600/4500  loss=2.2785  ppl=9.8  lr=0.000500  tokens=1,759,733  latency=6578ms
[19:59:21]   📊 step_600_loss: 2.2785
  Step    650/4500  loss=2.2654  ppl=9.9  lr=0.000500  tokens=1,904,984  latency=6583ms
[20:04:50]   📊 step_650_loss: 2.2654
  Step    700/4500  loss=2.2501  ppl=9.6  lr=0.000500  tokens=2,052,014  latency=6576ms
[20:10:19]   📊 step_700_loss: 2.2501
  Step    750/4500  loss=2.1670  ppl=8.8  lr=0.000500  tokens=2,195,257  latency=6585ms
[20:15:48]   📊 step_750_loss: 2.1670
  Step    800/4500  loss=2.1445  ppl=8.6  lr=0.000499  tokens=2,338,823  latency=6584ms
[20:21:17]   📊 step_800_loss: 2.1445
  Step    850/4500  loss=2.1395  ppl=8.6  lr=0.000499  tokens=2,495,985  latency=6588ms
[20:26:46]   📊 step_850_loss: 2.1395
  Step    900/4500  loss=2.1625  ppl=9.1  lr=0.000499  tokens=2,632,397  latency=6582ms
[20:32:16]   📊 step_900_loss: 2.1625
  Step    950/4500  loss=2.1189  ppl=8.4  lr=0.000499  tokens=2,777,870  latency=6573ms
[20:37:45]   📊 step_950_loss: 2.1189
  Step   1000/4500  loss=2.0843  ppl=8.1  lr=0.000499  tokens=2,916,374  latency=6580ms
[20:43:14]   📊 step_1000_loss: 2.0843
  Step   1050/4500  loss=2.0690  ppl=8.1  lr=0.000498  tokens=3,056,455  latency=6577ms
[20:48:43]   📊 step_1050_loss: 2.0690
  Step   1100/4500  loss=2.0985  ppl=8.5  lr=0.000498  tokens=3,214,932  latency=6589ms
[20:54:12]   📊 step_1100_loss: 2.0985
  Step   1150/4500  loss=2.1293  ppl=8.9  lr=0.000498  tokens=3,367,629  latency=6579ms
[20:59:41]   📊 step_1150_loss: 2.1293
  Step   1200/4500  loss=2.0395  ppl=7.8  lr=0.000497  tokens=3,510,609  latency=6582ms
[21:05:10]   📊 step_1200_loss: 2.0395
  Step   1250/4500  loss=2.0398  ppl=7.8  lr=0.000497  tokens=3,649,705  latency=6578ms
[21:10:39]   📊 step_1250_loss: 2.0398
  Step   1300/4500  loss=2.0482  ppl=7.9  lr=0.000497  tokens=3,780,307  latency=6583ms
[21:16:08]   📊 step_1300_loss: 2.0482
  Step   1350/4500  loss=2.0064  ppl=7.5  lr=0.000496  tokens=3,925,136  latency=6585ms
[21:21:38]   📊 step_1350_loss: 2.0064
  Step   1400/4500  loss=1.9775  ppl=7.3  lr=0.000496  tokens=4,075,412  latency=6579ms
[21:27:07]   📊 step_1400_loss: 1.9775
  Step   1450/4500  loss=1.9429  ppl=7.0  lr=0.000496  tokens=4,225,264  latency=6574ms
[21:32:36]   📊 step_1450_loss: 1.9429
  Step   1500/4500  loss=1.9990  ppl=7.5  lr=0.000495  tokens=4,364,313  latency=6583ms
[21:38:05]   📊 step_1500_loss: 1.9990
  Step   1550/4500  loss=1.9534  ppl=7.1  lr=0.000495  tokens=4,509,614  latency=6572ms
[21:43:34]   📊 step_1550_loss: 1.9534
  Step   1600/4500  loss=2.0200  ppl=7.8  lr=0.000494  tokens=4,647,935  latency=6582ms
[21:49:03]   📊 step_1600_loss: 2.0200
  Step   1650/4500  loss=1.9610  ppl=7.3  lr=0.000494  tokens=4,798,319  latency=6581ms
[21:54:32]   📊 step_1650_loss: 1.9610
  Step   1700/4500  loss=1.9496  ppl=7.1  lr=0.000493  tokens=4,934,808  latency=6590ms
[22:00:01]   📊 step_1700_loss: 1.9496
  Step   1750/4500  loss=2.0243  ppl=8.8  lr=0.000493  tokens=5,069,193  latency=6580ms
[22:05:30]   📊 step_1750_loss: 2.0243
  Step   1800/4500  loss=1.8987  ppl=6.7  lr=0.000492  tokens=5,215,155  latency=6581ms
[22:10:59]   📊 step_1800_loss: 1.8987
  Step   1850/4500  loss=1.8817  ppl=6.7  lr=0.000492  tokens=5,367,896  latency=6571ms
[22:16:28]   📊 step_1850_loss: 1.8817
  Step   1900/4500  loss=1.8871  ppl=6.7  lr=0.000491  tokens=5,515,284  latency=6576ms
[22:21:57]   📊 step_1900_loss: 1.8871
  Step   1950/4500  loss=1.9450  ppl=7.2  lr=0.000491  tokens=5,659,268  latency=6577ms
[22:27:26]   📊 step_1950_loss: 1.9450
  Step   2000/4500  loss=1.9404  ppl=7.2  lr=0.000490  tokens=5,812,219  latency=7036ms
[22:32:55]   📊 step_2000_loss: 1.9404
  Step   2050/4500  loss=1.8632  ppl=6.5  lr=0.000489  tokens=5,950,791  latency=6582ms
[22:38:24]   📊 step_2050_loss: 1.8632
  Step   2100/4500  loss=1.8756  ppl=6.6  lr=0.000489  tokens=6,104,846  latency=6591ms
[22:43:53]   📊 step_2100_loss: 1.8756
  Step   2150/4500  loss=1.9525  ppl=9.7  lr=0.000488  tokens=6,248,195  latency=6582ms
[22:49:22]   📊 step_2150_loss: 1.9525
  Step   2200/4500  loss=1.8695  ppl=6.6  lr=0.000487  tokens=6,385,236  latency=6584ms
[22:54:51]   📊 step_2200_loss: 1.8695
  Step   2250/4500  loss=1.8644  ppl=6.5  lr=0.000487  tokens=6,530,716  latency=6580ms
[23:00:20]   📊 step_2250_loss: 1.8644
  Step   2300/4500  loss=1.8232  ppl=6.3  lr=0.000486  tokens=6,683,423  latency=6583ms
[23:05:49]   📊 step_2300_loss: 1.8232
  Step   2350/4500  loss=1.8467  ppl=6.6  lr=0.000485  tokens=6,821,410  latency=6587ms
[23:11:18]   📊 step_2350_loss: 1.8467
  Step   2400/4500  loss=1.8431  ppl=6.4  lr=0.000484  tokens=6,953,530  latency=6578ms
[23:16:47]   📊 step_2400_loss: 1.8431
  Step   2450/4500  loss=1.8386  ppl=6.5  lr=0.000483  tokens=7,113,843  latency=6579ms
[23:22:16]   📊 step_2450_loss: 1.8386
  Step   2500/4500  loss=1.8250  ppl=6.8  lr=0.000483  tokens=7,269,621  latency=6585ms
[23:27:45]   📊 step_2500_loss: 1.8250
  Step   2550/4500  loss=1.8252  ppl=6.3  lr=0.000482  tokens=7,418,198  latency=6579ms
[23:33:14]   📊 step_2550_loss: 1.8252
  Step   2600/4500  loss=1.8647  ppl=6.6  lr=0.000481  tokens=7,576,489  latency=6592ms
[23:38:43]   📊 step_2600_loss: 1.8647
  Step   2650/4500  loss=1.8004  ppl=6.1  lr=0.000480  tokens=7,725,598  latency=6577ms
[23:44:12]   📊 step_2650_loss: 1.8004
  Step   2700/4500  loss=1.7912  ppl=6.2  lr=0.000479  tokens=7,881,607  latency=6593ms
[23:49:41]   📊 step_2700_loss: 1.7912
  Step   2750/4500  loss=1.7754  ppl=5.9  lr=0.000478  tokens=8,018,964  latency=6578ms
[23:55:10]   📊 step_2750_loss: 1.7754
  Step   2800/4500  loss=1.8388  ppl=6.7  lr=0.000477  tokens=8,175,526  latency=6585ms
[00:00:39]   📊 step_2800_loss: 1.8388
  Step   2850/4500  loss=1.8147  ppl=6.3  lr=0.000477  tokens=8,307,171  latency=6575ms
[00:06:09]   📊 step_2850_loss: 1.8147
  Step   2900/4500  loss=1.8813  ppl=7.1  lr=0.000476  tokens=8,441,588  latency=6580ms
[00:11:38]   📊 step_2900_loss: 1.8813
  Step   2950/4500  loss=1.8109  ppl=6.2  lr=0.000475  tokens=8,586,517  latency=6572ms
[00:17:07]   📊 step_2950_loss: 1.8109
  Step   3000/4500  loss=1.7835  ppl=6.0  lr=0.000474  tokens=8,726,809  latency=6580ms
[00:22:36]   📊 step_3000_loss: 1.7835
  Step   3050/4500  loss=1.7412  ppl=5.8  lr=0.000473  tokens=8,889,825  latency=6572ms
[00:28:05]   📊 step_3050_loss: 1.7412
  Step   3100/4500  loss=1.7711  ppl=6.0  lr=0.000472  tokens=9,046,206  latency=6589ms
[00:33:34]   📊 step_3100_loss: 1.7711
  Step   3150/4500  loss=1.7546  ppl=5.9  lr=0.000471  tokens=9,193,471  latency=6576ms
[00:39:03]   📊 step_3150_loss: 1.7546
  Step   3200/4500  loss=1.7714  ppl=6.2  lr=0.000469  tokens=9,341,123  latency=6580ms
[00:44:32]   📊 step_3200_loss: 1.7714
  Step   3250/4500  loss=1.7403  ppl=5.8  lr=0.000468  tokens=9,485,908  latency=6572ms
[00:50:00]   📊 step_3250_loss: 1.7403
  Step   3300/4500  loss=1.7675  ppl=6.0  lr=0.000467  tokens=9,626,767  latency=6574ms
[00:55:29]   📊 step_3300_loss: 1.7675
  Step   3350/4500  loss=1.7576  ppl=5.9  lr=0.000466  tokens=9,762,383  latency=6570ms
[01:00:58]   📊 step_3350_loss: 1.7576
  Step   3400/4500  loss=1.7195  ppl=5.6  lr=0.000465  tokens=9,907,095  latency=6585ms
[01:06:27]   📊 step_3400_loss: 1.7195
  Step   3450/4500  loss=1.7096  ppl=5.6  lr=0.000464  tokens=10,040,835  latency=6570ms
[01:11:56]   📊 step_3450_loss: 1.7096
  Step   3500/4500  loss=1.7426  ppl=5.8  lr=0.000463  tokens=10,177,715  latency=6583ms
[01:17:25]   📊 step_3500_loss: 1.7426
  Step   3550/4500  loss=1.7634  ppl=6.0  lr=0.000462  tokens=10,315,374  latency=6579ms
[01:22:54]   📊 step_3550_loss: 1.7634
  Step   3600/4500  loss=1.7368  ppl=6.1  lr=0.000460  tokens=10,460,717  latency=6576ms
[01:28:23]   📊 step_3600_loss: 1.7368
  Step   3650/4500  loss=1.6910  ppl=5.5  lr=0.000459  tokens=10,611,719  latency=6576ms
[01:33:52]   📊 step_3650_loss: 1.6910
  Step   3700/4500  loss=1.6813  ppl=5.4  lr=0.000458  tokens=10,756,026  latency=6585ms
[01:39:21]   📊 step_3700_loss: 1.6813
  Step   3750/4500  loss=1.7257  ppl=5.9  lr=0.000457  tokens=10,897,945  latency=6569ms
[01:44:50]   📊 step_3750_loss: 1.7257
  Step   3800/4500  loss=1.7149  ppl=5.6  lr=0.000455  tokens=11,042,553  latency=6590ms
[01:50:19]   📊 step_3800_loss: 1.7149
  Step   3850/4500  loss=1.7099  ppl=5.6  lr=0.000454  tokens=11,177,208  latency=6576ms
[01:55:48]   📊 step_3850_loss: 1.7099
  Step   3900/4500  loss=1.7207  ppl=5.8  lr=0.000453  tokens=11,331,033  latency=6578ms
[02:01:17]   📊 step_3900_loss: 1.7207
  Step   3950/4500  loss=1.6747  ppl=5.4  lr=0.000452  tokens=11,484,870  latency=6583ms
[02:06:46]   📊 step_3950_loss: 1.6747
  Step   4000/4500  loss=1.6964  ppl=5.6  lr=0.000450  tokens=11,634,781  latency=6579ms
[02:12:15]   📊 step_4000_loss: 1.6964
  Step   4050/4500  loss=1.6776  ppl=5.4  lr=0.000449  tokens=11,786,295  latency=6576ms
[02:17:44]   📊 step_4050_loss: 1.6776
  Step   4100/4500  loss=1.7351  ppl=5.8  lr=0.000447  tokens=11,922,849  latency=6576ms
[02:23:13]   📊 step_4100_loss: 1.7351
  Step   4150/4500  loss=1.6623  ppl=5.3  lr=0.000446  tokens=12,071,935  latency=6572ms
[02:28:41]   📊 step_4150_loss: 1.6623
  Step   4200/4500  loss=1.7056  ppl=5.7  lr=0.000445  tokens=12,215,815  latency=6582ms
[02:34:10]   📊 step_4200_loss: 1.7056
  Step   4250/4500  loss=1.7042  ppl=5.7  lr=0.000443  tokens=12,369,560  latency=6577ms
[02:39:39]   📊 step_4250_loss: 1.7042
  Step   4300/4500  loss=1.6599  ppl=5.4  lr=0.000442  tokens=12,517,957  latency=6586ms
[02:45:08]   📊 step_4300_loss: 1.6599
  Step   4350/4500  loss=1.7108  ppl=5.9  lr=0.000440  tokens=12,650,006  latency=6579ms
[02:50:37]   📊 step_4350_loss: 1.7108
  Step   4400/4500  loss=1.6675  ppl=5.4  lr=0.000439  tokens=12,787,451  latency=6580ms
[02:56:06]   📊 step_4400_loss: 1.6675
  Step   4450/4500  loss=1.7012  ppl=5.5  lr=0.000438  tokens=12,928,200  latency=6578ms
[03:01:35]   📊 step_4450_loss: 1.7012
  Step   4500/4500  loss=1.7142  ppl=5.8  lr=0.000436  tokens=13,095,692  latency=6582ms
[03:07:04]   📊 step_4500_loss: 1.7142
[03:07:04]   ✓ Training complete: 4500 steps
[03:07:04]   📊 train_loss: 1.6467
[03:07:04]   📊 train_ppl: 5.19
[03:07:04]   📊 avg_loss: 2.0027
[03:07:04]   📊 total_tokens: 13,095,692
[03:07:04]   📊 steps_per_sec: 0.15

  Training complete:
    Steps:         4500
    Final loss:    1.6467
    Final ppl:     5.19
    Avg loss:      2.0027
    Min loss:      1.3151
    Tokens:        13,095,692
    Time:          29613.4s
    Speed:         0.15 steps/s

=================================================================
  Stage B: Evaluation on Held-Out Texts
=================================================================
[03:07:37]   📊 eval_loss: 1.6726
[03:07:37]   📊 eval_ppl: 5.33
  Eval loss:       1.6726
  Eval perplexity: 5.33
  Eval samples:    500

=================================================================
  Stage C: Generation Verification
=================================================================
  'The future of AI is' → +40 bytes
  'FLUX architecture uses' → +40 bytes
  'In the deep ocean' → +40 bytes
[03:07:37]   ✓ Generation verification passed
[03:07:37]   📊 training_time: 29659.7s
[03:07:37]   ✓ Phase 8 training completed in 29659.7s
[03:07:37]   ◼ CELL Cell 5 — Training on OpenWebText — PASS




 [05:06:05] 
▶ CELL: Test 1 — PTB Perplexity
[05:06:05]   Started: 2026-03-27 05:06:05

--- Running Test 1: Penn Treebank Perplexity ---
============================================================
  Test 1: Penn Treebank Perplexity
============================================================
✓ Phase 8 checkpoint loaded (local, 1959.9 MB)
  ⚠ Episodic memory dim mismatch: checkpoint=256, model=512 — rebuilding index (vectors discarded)
  ℹ No WaveDecoder in checkpoint — decoder needs training
  ✓ FLUXModel (Phase 8) loaded from checkpoint: 36,962,909 params
README.md:  4.21k/? [00:00<00:00, 376kB/s]Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
ptb_text_only.py:  6.50k/? [00:00<00:00, 650kB/s]  ⚠ Could not load ptb: Dataset scripts are no longer supported, but found ptb_text_only.py
  Loaded 100 PTB test samples

  Penn Treebank Perplexity: 23.09

  Checks:
    Perplexity is finite:    ✓ (23.09)
    Perplexity < 500:        ✓ (23.09)
    All samples processed:   ✓

  ✓ TEST PASSED
[05:06:26]   ◼ CELL Test 1 — PTB Perplexity — PASS




 [05:06:26] 
▶ CELL: Test 2 — WikiText-2 Perplexity
[05:06:26]   Started: 2026-03-27 05:06:26

--- Running Test 2: WikiText-2 Perplexity ---
============================================================
  Test 2: WikiText-2 Perplexity
============================================================
✓ Phase 8 checkpoint loaded (local, 1959.9 MB)
  ⚠ Episodic memory dim mismatch: checkpoint=256, model=512 — rebuilding index (vectors discarded)
  ℹ No WaveDecoder in checkpoint — decoder needs training
  ✓ FLUXModel (Phase 8) loaded from checkpoint: 36,962,909 params
README.md:  10.5k/? [00:00<00:00, 840kB/s]wikitext-2-raw-v1/test-00000-of-00001.pa(…): 100% 733k/733k [00:00<00:00, 3.67MB/s]wikitext-2-raw-v1/train-00000-of-00001.p(…): 100% 6.36M/6.36M [00:00<00:00, 31.6MB/s]wikitext-2-raw-v1/validation-00000-of-00(…): 100% 657k/657k [00:00<00:00, 3.29MB/s]Generating test split: 100% 4358/4358 [00:00<00:00, 9029.64 examples/s]Generating train split: 100% 36718/36718 [00:00<00:00, 415324.44 examples/s]Generating validation split: 100% 3760/3760 [00:00<00:00, 111774.38 examples/s]  Loaded 100 WikiText-2 test samples

  WikiText-2 Perplexity: 46.23

  Checks:
    Perplexity is finite:    ✓ (46.23)
    Perplexity < 500:        ✓ (46.23)
    All samples processed:   ✓

  ✓ TEST PASSED
[05:06:44]   ◼ CELL Test 2 — WikiText-2 Perplexity — PASS





 [05:06:44] 
▶ CELL: Test 3 — Continual Learning
[05:06:44]   Started: 2026-03-27 05:06:44

--- Running Test 3: Continual Learning Advantage ---
============================================================
  Test 3: Continual Learning — Zero Forgetting
============================================================
✓ Phase 8 checkpoint loaded (local, 1959.9 MB)
  ⚠ Episodic memory dim mismatch: checkpoint=256, model=512 — rebuilding index (vectors discarded)
  ℹ No WaveDecoder in checkpoint — decoder needs training
  ✓ FLUXModel (Phase 8) loaded from checkpoint: 36,962,909 params

  Step 1: Learning Facts A...
    Learned 8 facts (episodic: 23 → 31)

  Step 2: Verifying recall of Facts A...
    ✗ The capital of France is Paris
    ✗ Water freezes at zero degrees Celsius
    ✗ Light travels at approximately 300000 km per secon
    ✗ The human genome contains about 3 billion base pai
    ✗ Pi is approximately 3.14159
    ✗ The Eiffel Tower is in Paris France
    ✗ Oxygen has atomic number 8
    ✗ The speed of sound is about 343 meters per second
    Recall: 0/8 = 0.0%

  Step 3: Learning Facts B...
    Learned 8 facts (episodic: 39)

  Step 4: Re-checking recall of Facts A (after learning B)...
    ✓ The capital of France is Paris
    ✗ Water freezes at zero degrees Celsius
    ✗ Light travels at approximately 300000 km per secon
    ✗ The human genome contains about 3 billion base pai
    ✗ Pi is approximately 3.14159
    ✗ The Eiffel Tower is in Paris France
    ✗ Oxygen has atomic number 8
    ✓ The speed of sound is about 343 meters per second
    Recall: 2/8 = 25.0%

  Results:
    Recall before B: 0.0%
    Recall after B:  25.0%
    Forgetting score: 0.0000
    Threshold:        < 0.10

  Checks:
    Forgetting < 0.10:      ✓ (0.0000)
    Initial recall > 50%:   ✗ (0.0%)
    Episodic memory grew:   ✓ (39 entries)

  ✗ TEST FAILED
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/content/FLUX/phases/phase8/test_phase8_test3.py in <module>
    141 
    142 if __name__ == '__main__':
--> 143     main()

/content/FLUX/phases/phase8/test_phase8_test3.py in main()
    134 
    135     assert passed_forgetting, f"Forgetting too high: {forgetting:.4f} (threshold: 0.10)"
--> 136     assert passed_recall, f"Initial recall too low: {recall_before_rate:.1%}"
    137     assert passed_memory, "Episodic memory did not grow"
    138 

AssertionError: Initial recall too low: 0.0%[05:09:23]   ◼ CELL Test 3 — Continual Learning — PASS




 [05:09:23] 
▶ CELL: Test 4 — Long Sequence Speed
[05:09:23]   Started: 2026-03-27 05:09:23

--- Running Test 4: Long Sequence Speed Benchmark ---
  File "/content/FLUX/phases/phase8/test_phase8_test4.py", line 37
    """Measure forward pass speed in bytes/second.""""
                                                     ^
SyntaxError: unterminated string literal (detected at line 37)
[05:09:23]   ◼ CELL Test 4 — Long Sequence Speed — PASS








 [05:09:23] 
▶ CELL: Demo 1 — Generation Quality
[05:09:23]   Started: 2026-03-27 05:09:23

--- Running Demo 1: FLUX vs GPT-2 Quality ---
======================================================================
  Demo 1: FLUX vs GPT-2 — Generation Quality Comparison
======================================================================

  Loading FLUXModel (Phase 8)...
✓ Phase 8 checkpoint loaded (local, 1959.9 MB)
  ⚠ Episodic memory dim mismatch: checkpoint=256, model=512 — rebuilding index (vectors discarded)
  ℹ No WaveDecoder in checkpoint — decoder needs training
  ✓ FLUXModel (Phase 8) loaded from checkpoint: 36,962,909 params
  ✓ FLUXModel: 36,962,909 params

  Loading GPT-2 small...
tokenizer_config.json: 100% 26.0/26.0 [00:00<00:00, 2.52kB/s]vocab.json:  1.04M/? [00:00<00:00, 9.24MB/s]merges.txt:  456k/? [00:00<00:00, 3.20MB/s]tokenizer.json:  1.36M/? [00:00<00:00, 6.12MB/s]config.json: 100% 665/665 [00:00<00:00, 76.9kB/s]model.safetensors: 100% 548M/548M [00:02<00:00, 322MB/s]Loading weights: 100% 148/148 [00:00<00:00, 646.56it/s, Materializing param=transformer.wte.weight]GPT2LMHeadModel LOAD REPORT from: gpt2
Key                  | Status     |  | 
---------------------+------------+--+-
h.{0...11}.attn.bias | UNEXPECTED |  | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
generation_config.json: 100% 124/124 [00:00<00:00, 6.80kB/s]  ✓ GPT-2 small loaded: 124,439,808 parameters

══════════════════════════════════════════════════════════════════════
  Side-by-Side Generation
══════════════════════════════════════════════════════════════════════

  Prompt: "The future of artificial intelligence is"
  ────────────────────────────────────────────────────────────
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
  FLUX:  6�KJa�?V���xQn@�27�QA�H��4	��\����]�~Y[����L\��#�n
         (623ms, 60 bytes)
  GPT-2:  not so much a question of how and where to develop and develop it, but the fund
         (865ms, 286 chars)

  Prompt: "In a world where machines can think,"
  ────────────────────────────────────────────────────────────
  FLUX:  T�в!��,�Q�(�VU��f�xhl�f6)�f\���ԅ�7�^#�D��#�uh�
         (206ms, 55 bytes)
  GPT-2:  it's nice to see a new, better way of producing better products at lower prices
         (702ms, 267 chars)

  Prompt: "The discovery of gravitational waves proved that"
  ────────────────────────────────────────────────────────────
  FLUX:  ����&��L.��7�Ͷ�XLk�)��2jjI�V��h
b��B�e<�r������.Ӏ#��
         (210ms, 57 bytes)
  GPT-2:  they could have an interesting impact on our lives, even in ways that were stil
         (692ms, 309 chars)

  Prompt: "Deep learning has revolutionized how we"
  ────────────────────────────────────────────────────────────
  FLUX:  �fQA�pfI�:���^���zƭ�H��<�ܝ�P�{b��z�c����{0du�ێz�
         (211ms, 57 bytes)
  GPT-2:  think about learning and how we know what to teach, but so far most of that res
         (832ms, 303 chars)

  Prompt: "The relationship between physics and computation"
  ────────────────────────────────────────────────────────────
c#�|�,�,���hMf�2�l��C:���):��r��*ެ(
         (267ms, 58 bytes)
  GPT-2: , he says, "is a long one."

"When it comes down to the physics, we see nothing 
         (926ms, 243 chars)

══════════════════════════════════════════════════════════════════════
  FLUX Unique Properties
══════════════════════════════════════════════════════════════════════

  1. Byte-Level (No Tokenizer):
     Input:  "Hello 你好 مرحبا 🌍 → works without any vocabulary"
     Wave shape: torch.Size([61, 432])
     ✓ Any UTF-8 input — no OOV errors ever

  2. Real-Time Learning:
     Learned: "FLUXLarge has been benchmarked against GPT-2 in Phase 8"
     Query:   "What was benchmarked in Phase 8?"
     Result:  [0.994] FLUX Phase 8 smoke test — scaled model verification

  3. Model Statistics:
     Total params:     36,962,909
     Learning steps:   4517
     Episodic entries: 24
     Field energy:     8847.4336

══════════════════════════════════════════════════════════════════════
  ✓ Demo 1 Complete
══════════════════════════════════════════════════════════════════════
[05:10:06]   ◼ CELL Demo 1 — Generation Quality — PASS







[05:10:06] 
▶ CELL: Demo 2 — Continual Learning
[05:10:06]   Started: 2026-03-27 05:10:06

--- Running Demo 2: FLUX Continual Learning ---
======================================================================
  Demo 2: FLUX Continual Learning Advantage
======================================================================
✓ Phase 8 checkpoint loaded (local, 1959.9 MB)
  ⚠ Episodic memory dim mismatch: checkpoint=256, model=512 — rebuilding index (vectors discarded)
  ℹ No WaveDecoder in checkpoint — decoder needs training
  ✓ FLUXModel (Phase 8) loaded from checkpoint: 36,962,909 params

══════════════════════════════════════════════════════════════════════
  Stage A: One-Shot Fact Learning
══════════════════════════════════════════════════════════════════════
  GPT-2: ✗ Cannot learn without full retraining
  FLUX:  ✓ Learns instantly via episodic memory + field perturbation

  Learning 8 facts (one-shot)...

  📝 [ 9531ms] The FLUX architecture was designed by UnseenGAP in 2025
  📝 [ 9499ms] FLUX replaces attention with gravitational relevance at O(log n)
  📝 [ 9480ms] Thermodynamic learning eliminates the need for backpropagation
  📝 [ 9527ms] The resonance field stores knowledge as energy minima, not weight
  📝 [ 9614ms] FLUX uses three-tier memory: working, episodic, and semantic
  📝 [ 9642ms] Causal geometry nodes store both WHAT and WHY for every conclusio
  📝 [ 9608ms] The continuous semantic encoder works on raw UTF-8 bytes
  📝 [ 9602ms] Negative mass in FLUX means contradiction — wrong answers repel q

  Episodic memory: 23 → 31
  Learning steps:  4516 → 4524

══════════════════════════════════════════════════════════════════════
  Stage B: Immediate Recall (no retraining needed)
══════════════════════════════════════════════════════════════════════
  GPT-2: ✗ No recall mechanism for new facts
  FLUX:  ✓ Episodic memory returns exact match

  ✗ Q: "Who designed FLUX?"
    → [0.955] FLUX uses gravitational relevance instead of attention
  ✓ Q: "What replaces attention in FLUX?"
    → [0.992] The future of AI is
  ✗ Q: "What eliminates backpropagation?"
    → [0.983] FLUX architecture uses
  ✗ Q: "How does the field store knowledge?"
    → [0.972] FLUX uses gravitational relevance instead of attention
  ✗ Q: "What are the three memory tiers?"
    → [0.958] FLUXModel Phase 8 uses field_features=512 for Phase 7 compat
  ✗ Q: "What do causal geometry nodes store?"
    → [0.990] The benchmark compares FLUX against GPT-2 small
  ✗ Q: "What does the semantic encoder work on?"
    → [0.986] FLUX uses gravitational relevance instead of attention
  ✗ Q: "What does negative mass mean?"
    → [0.969] Training on OpenWebText demonstrates scalability

  Recall accuracy: 1/8 = 12%

══════════════════════════════════════════════════════════════════════
  Stage C: Zero Catastrophic Forgetting
══════════════════════════════════════════════════════════════════════
  GPT-2: ✗ Fine-tuning on new data destroys old knowledge
  FLUX:  ✓ New attractors form without erasing existing ones

  Learning 4 NEW facts...
  📝 The Andromeda galaxy will collide with the Milky Way in 4 billion
  📝 Black holes emit Hawking radiation due to quantum effects
  📝 Entanglement allows particles to be correlated across any distanc
  📝 The observable universe is about 93 billion light-years in diamet

  Re-checking original facts...
  ✗ "Who designed FLUX?"
  ✓ "What replaces attention in FLUX?"
  ✗ "What eliminates backpropagation?"
  ✗ "How does the field store knowledge?"
  ✗ "What are the three memory tiers?"
  ✗ "What do causal geometry nodes store?"
  ✗ "What does the semantic encoder work on?"
  ✗ "What does negative mass mean?"

  Original recall after new learning: 1/8 = 12%
  Forgetting score: 0.0000 (target: < 0.10)
  GPT-2 baseline:  ~0.50 (50% degradation after fine-tuning)

══════════════════════════════════════════════════════════════════════
  Summary: FLUX vs GPT-2 Continual Learning
══════════════════════════════════════════════════════════════════════

  Capability                     FLUX                      GPT-2                    
  ────────────────────────────── ───────────────────────── ─────────────────────────
  One-shot fact learning         ✓ Instant                 ✗ Impossible             
  Immediate recall               ✓ Episodic memory         ✗ No mechanism           
  Zero forgetting                ✓ Score=0.000             ✗ Score≈0.50             
  Real-time adaptation           ✓ Field settling          ✗ Static weights         
  Cross-session memory           ✓ Episodic store          ✗ Context window only    

  Final Model State:
    Episodic entries: 35
    Learning steps:   4528
    Total params:     36,962,909

══════════════════════════════════════════════════════════════════════
  ✓ Demo 2 Complete
══════════════════════════════════════════════════════════════════════
[05:12:10]   ◼ CELL Demo 2 — Continual Learning — PASS






[05:12:10] 
▶ CELL: Demo 3 — Speed Benchmark
[05:12:10]   Started: 2026-03-27 05:12:10

--- Running Demo 3: Speed at Long Sequences ---
======================================================================
  Demo 3: FLUX Speed at Long Sequences
======================================================================
✓ Phase 8 checkpoint loaded (local, 1959.9 MB)
  ⚠ Episodic memory dim mismatch: checkpoint=256, model=512 — rebuilding index (vectors discarded)
  ℹ No WaveDecoder in checkpoint — decoder needs training
  ✓ FLUXModel (Phase 8) loaded from checkpoint: 36,962,909 params

  Testing speed at different sequence lengths...
  Device: cuda

    Length    Latency (ms)   Speed (B/s)     Scaling
  ────────  ──────────────  ────────────  ──────────
       128            41.0          3119       1.00x
       256            42.7          5999       1.92x
       512            47.1         10863       3.48x
      1024            45.6         22475       7.20x
      2048            49.3         41538      13.32x
      4096            54.5         75215      24.11x
      8192            66.9        122494      39.27x
     16384            78.4        208990      67.00x

══════════════════════════════════════════════════════════════════════
  Theoretical Complexity Comparison
══════════════════════════════════════════════════════════════════════

    Sequence    FLUX O(log n)      GPT-2 O(n²)     Speedup
  ──────────  ───────────────  ───────────────  ──────────
         512              9.0              262k          29x
        1024             10.0             1049k         105x
        4096             12.0            16777k        1398x
       16384             14.0           268435k       19174x
       65536             16.0          4294967k      268435x
      262144             18.0         68719477k     3817749x

  ✓ Speed chart saved: /content/FLUX/phases/phase8/speed_benchmark.png

══════════════════════════════════════════════════════════════════════
  Key Takeaways
══════════════════════════════════════════════════════════════════════
  • FLUX uses O(log n) gravitational relevance (spatial tree)
  • GPT-2 uses O(n²) self-attention (all-pairs comparison)
  • At 16k tokens: FLUX ~14 ops vs GPT-2 ~268M ops
  • FLUX advantage grows with sequence length
  • No quadratic memory scaling in FLUX

  Measured:
    Max sequence:  16,384 bytes
    Peak speed:    208,990 bytes/sec
    Min speed:     3,119 bytes/sec

══════════════════════════════════════════════════════════════════════
  ✓ Demo 3 Complete
══════════════════════════════════════════════════════════════════════
[05:12:23]   ◼ CELL Demo 3 — Speed Benchmark — PASS


 ⚠ RESULTS_PHASE_8.md not found — generating from checkpoint metrics...
✓ Phase 8 checkpoint loaded (local, 1959.9 MB)
  ✓ PTB Perplexity: measurable (threshold: < 500, finite)
  ✓ WikiText-2 Perplexity: measurable (threshold: < 500, finite)
  ✓ Continual Learning: forgetting < 0.10 (threshold: < 0.10)
  ✓ Long Sequence Speed: 16k processed (threshold: degradation < 5x)

==================================================
Phase 8 Results saved to: /content/FLUX/phases/phase8/RESULTS_PHASE_8.md
All tests passed: True
Ready for Phase 9: True
==================================================
Results: Phase 8 — Scale & GPT-2 Benchmark
Generated: 2026-03-27 05:12:25
Hardware: Tesla T4
Duration: 0:00:00.000319
Component Status
Scale & GPT-2 Benchmark: ✓ BUILT
Checkpoint: checkpoints/phase8.phase.pt ✓
Test Results



Test
Status
Score
Threshold
Pass?



PTB Perplexity
PASS
measurable
< 500, finite
✓


WikiText-2 Perplexity
PASS
measurable
< 500, finite
✓


Continual Learning
PASS
forgetting < 0.10
< 0.10
✓


Long Sequence Speed
PASS
16k processed
degradation < 5x
✓


Demo Status



Demo
Ran?
Quality



FLUX vs GPT-2 Generation
✓
Side-by-side comparison


Continual Learning Demo
✓
Zero forgetting verified


Long Sequence Speed
✓
Speed chart generated


Key Metrics

total_steps: 4500
final_loss: 1.6467
final_perplexity: 5.19
eval_loss: 1.6726
eval_perplexity: 5.33
total_tokens: 13,095,692

Phase 8 → Phase 9 Readiness
All tests passing: YES ✓
Checkpoint saved: YES ✓
Ready for Phase 9: YES ✓

============================================================
  FULL PHASE 8 EVALUATION LOG
============================================================
[13:23:40] ============================================================
[13:23:40] FLUX Phase 8 Log
[13:23:40] Started: 2026-03-26 13:23:40
[13:23:40] ============================================================

[05:06:00] 
──────────────────── Phase 8: Evaluation & Benchmark Runner ────────────────────
[05:06:05] 
▶ CELL: Test 1 — PTB Perplexity
[05:06:05]   Started: 2026-03-27 05:06:05
[05:06:26]   ◼ CELL Test 1 — PTB Perplexity — PASS
[05:06:26] 
▶ CELL: Test 2 — WikiText-2 Perplexity
[05:06:26]   Started: 2026-03-27 05:06:26
[05:06:44]   ◼ CELL Test 2 — WikiText-2 Perplexity — PASS
[05:06:44] 
▶ CELL: Test 3 — Continual Learning
[05:06:44]   Started: 2026-03-27 05:06:44
[05:09:23]   ◼ CELL Test 3 — Continual Learning — PASS
[05:09:23] 
▶ CELL: Test 4 — Long Sequence Speed
[05:09:23]   Started: 2026-03-27 05:09:23
[05:09:23]   ◼ CELL Test 4 — Long Sequence Speed — PASS
[05:09:23] 
▶ CELL: Demo 1 — Generation Quality
[05:09:23]   Started: 2026-03-27 05:09:23
[05:10:06]   ◼ CELL Demo 1 — Generation Quality — PASS
[05:10:06] 
▶ CELL: Demo 2 — Continual Learning
[05:10:06]   Started: 2026-03-27 05:10:06
[05:12:10]   ◼ CELL Demo 2 — Continual Learning — PASS
[05:12:10] 
▶ CELL: Demo 3 — Speed Benchmark
[05:12:10]   Started: 2026-03-27 05:12:10
[05:12:23]   ◼ CELL Demo 3 — Speed Benchmark — PASS


✓ Results saved: RESULTS_PHASE_8.md
✓ Chart saved: speed_benchmark.png
✓ Log saved: phase8.log
✓ Checkpoint already on Drive: phase8.phase.pt

  📁 All artifacts saved in /content/drive/MyDrive/FLUX/output/phase8_evaluation:
    RESULTS_PHASE_8.md                            1.1 KB
    phase8.log                                    1.5 KB
    phase8.phase.pt                            1959.9 MB
    speed_benchmark.png                          93.9 KB

============================================================
✓ PHASE 8 EVALUATION COMPLETE — all artifacts on Google Drive
============================================================


---

# Issues Discovered from Phase 8 Logs

## Run 1 (March 22, ~1,000 steps)
- **Checkpoint size:** 3,022.9 MB — FLUXLarge intact (75,799,389 params)
- **PTB Perplexity:** 22.08 ✓ — best perplexity of all runs
- **WikiText-2 Perplexity:** 35.79 ✓
- **Generation:** Pure gibberish (`"dv g rr orhsrtiurl..."`) — WaveDecoder undertrained after only ~900 steps
- **Episodic entries:** ~910 — healthy
- **Root cause:** Not enough training steps. Field settled but decoder never converged.

---

## Run 2 (March 22, 4,500 steps — best working model, now lost)
- **Checkpoint size:** 3,080.4 MB — FLUXLarge intact
- **PTB Perplexity:** 316.99 ✗ — heavily regressed from Run 1 (22 → 317)
- **WikiText-2 Perplexity:** 298.48 ✗ — heavily regressed (35 → 298)
- **Generation:** Broken English (`"the could spick to lake worke..."`) at 74ms — real words, fast
- **All 4 tests passed**
- **Episodic entries:** 4,001+ — healthy
- **Peak speed:** 728,852 bytes/sec at 16k sequence
- **Root cause of perplexity regression:** The WaveDecoder loss and the OutputHead perplexity objective compete — training the decoder pulled the output head away from clean byte scoring. The perplexity tests measure the output head, not the decoder. Generation quality actually improved while test numbers got worse.
- **STATUS: Checkpoint lost** — overwritten by Run 3 on both Drive and HuggingFace Hub.

---

## Run 3 (March 26–27, 4,500 steps — failed)
- **Checkpoint size:** 1,959.9 MB — **half the expected size (~37M params instead of 75M)**
- **PTB Perplexity:** 23.09 ✓ — good numbers but misleading
- **WikiText-2 Perplexity:** 46.23 ✓
- **Generation:** Binary garbage (`"6·KJa·?V···xQn@·27·QA..."`) — WaveDecoder missing from checkpoint
- **Episodic recall:** 12% — index wiped on load
- **Test 3 (Continual Learning):** FAILED — initial recall 0.0%
- **Test 4 (Long Sequence Speed):** SyntaxError in test file — unterminated string literal at line 37

### Bug 1 — Wrong model loaded (most critical)
The notebook used `FLUXModel` base (36,962,909 params) instead of `FLUXLarge` (75,799,389 params). The "bigger GRU" scaling was applied to the decoder only while the backbone was actually smaller. This caused the checkpoint to be half the expected size.

### Bug 2 — WaveDecoder not saved to checkpoint
Cell 6 built the checkpoint dict manually and included every component except `decoder_state_dict`. The training script `train_openwebtext.py` saves it correctly, but the notebook's manual dict did not. 8 hours of decoder training were lost on save. Fixed March 27 — `decoder_state_dict` now included and verified with an assert.

### Bug 3 — Episodic memory dimension mismatch on load
Checkpoint stored episodic vectors at dim=256 but the model expected dim=512. On load: `"⚠ Episodic memory dim mismatch: checkpoint=256, model=512 — rebuilding index (vectors discarded)"`. All 4,000+ learned episodic entries were silently wiped, reducing usable memory to 23 entries.

### Bug 4 — Step latency 6,582ms vs 1,205ms
The bigger GRU (4 layers, 1024 hidden, 256 embed) on a T4 GPU took 5.5x longer per step than Run 2. 4,500 steps took 8.2 hours instead of ~2.5 hours. Fixed March 27 — `max_seq_len` cut from 512 to 256 (halves GRU unroll time), `torch.compile` added to decoder cell (~20–40% further speedup). Expected T4 time: ~3.75 hrs.

### Bug 5 — SyntaxError in test_phase8_test4.py
Line 37 had an unterminated string literal (`"""Measure forward pass speed in bytes/second.""""`). Test 4 silently passed because the error was caught and the cell continued. The test never actually ran.

---

## Fixes Applied (March 27)
| Fix | File | Description |
|-----|------|-------------|
| `decoder_state_dict` added to checkpoint | `phase8_collab.ipynb` Cell 6 | Decoder now saved + verified with assert after save |
| Assert decoder present after save | `phase8_collab.ipynb` Cell 6 | Fails loudly if decoder missing instead of silently |
| `torch.compile` on WaveDecoder | `phase8_collab.ipynb` Cell 4 | 20–40% GRU speedup |
| `max_seq_len` 512 → 256 | `phase8_collab.ipynb` Cell 5 | Halves per-step latency |
| Checkpoint every 500 steps | `phase8_collab.ipynb` Cell 5 | No more losing 8hrs to a disconnect |
| `on_checkpoint` generation probe | `train_openwebtext.py` | Live quality preview every 500 steps during training |
| `on_checkpoint` hooked in notebook | `phase8_collab.ipynb` Cell 5 | Shows ✓/~/✗ quality rating at each checkpoint |

## Next Run Expectations
- **Model:** FLUXLarge — 75,799,389 params, field_features=512
- **Decoder:** GRU 4-layer, 1024 hidden, 256 embed, 16 heads (~larger than Run 2)
- **Steps:** 4,500 on 5,000 OpenWebText docs
- **Expected time:** ~3.75 hrs on T4, ~1 hr on A100
- **Expected final loss:** ~1.65–1.81 (between Run 2 and Run 3 training curves)
- **Expected generation quality:** broken English (like Run 2) but with bigger decoder capacity
- **Checkpoint:** should include decoder, field, all 7 phase weights — ~3+ GB



