[06:46:33] 
▶ CELL: Cell 5 — Training / Load from Checkpoint
[06:46:33]   Started: 2026-03-22 06:46:33
[06:46:34]   ✓ MultiTimescaleCoordinator built: 14,708,767 params
[06:46:34]   ✓ CausalGraph initialized

=================================================================
  Stage A: Causal Graph Formation
=================================================================
  ✓       bird → can_fly          w=0.80  birds can fly
  ✓    penguin → bird             w=1.00  penguin is a bird
  ✓    penguin → cannot_fly       w=0.90  penguin cannot fly
  ✓    sparrow → bird             w=1.00  sparrow is a bird
  ✓    sparrow → can_fly          w=0.95  sparrow can fly
  ✓       fish → can_swim         w=0.90  fish can swim
  ✓    dolphin → fish             w=0.30  dolphin resembles fish
  ✓    dolphin → mammal           w=1.00  dolphin is a mammal
  ✓     mammal → breathes_air     w=0.95  mammal breathes air
[06:46:34]   📊 causal_edges: 9
[06:46:34]   📊 causal_nodes: 10

  Trace 'can fly':    ['sparrow', 'can_fly']
  Trace 'cannot fly': ['penguin', 'cannot_fly']
  Penguin conflicts: [('can_fly', 'cannot_fly')]
  Sparrow conflicts: []  (none expected)

=================================================================
  Stage B: Multi-Timescale Separation
=================================================================
  Fast nodes activate at step:   2
  Medium nodes activate at step: 5
  Slow nodes activate at step:   31
[06:46:35]   📊 fast_activation_step: 2
[06:46:35]   📊 medium_activation_step: 5
[06:46:35]   📊 slow_activation_step: 31
  'Birds can fly through the air' → fast=0.0007  med=0.0042  slow=0.0097
  'Penguins are birds that cannot fly' → fast=0.0007  med=0.0042  slow=0.0096
  'The quick brown fox jumps over the lazy dog' → fast=0.0007  med=0.0040  slow=0.0091

=================================================================
  Stage C: Geometry Computation Correctness
=================================================================
  Bending magnitude: 0.8729
  Influence strength: 0.3291
  Mass=5.0 amplification: 5.00x
[06:46:36]   📊 mass_amplification: 5.00x

=================================================================
  Stage D: Causal Invalidation
=================================================================
  Before: 'supported' (net=1.00)
  After:  'contradicted' (net=-1.00)
  Affected downstream: [11, 12]
[06:46:36]   📊 invalidation_propagation: 2

=================================================================
  Stage E: Full Pipeline — CSE → Field → CGN
=================================================================
  ✓ 'The Earth revolves around the Sun' → cgn_norm=0.2681
  ✓ 'Water freezes at zero degrees Celsius' → cgn_norm=0.2724
  ✓ 'Gravity pulls objects toward the center of the Ear' → cgn_norm=0.2635
  ✓ 'Light travels at 300000 kilometers per second' → cgn_norm=0.2735
  ✓ 'Plants convert sunlight into energy through photos' → cgn_norm=0.2722
[06:46:36]   📊 training_time: 2.4s

  ✓ Training complete in 2.4s (0.0 min)
[06:46:36]   ◼ CELL Cell 5 — Training / Load from Checkpoint — PASS




[06:52:34] 
▶ CELL: Cell 7 — Test 1: Causal Trace Accuracy
[06:52:34]   Started: 2026-03-22 06:52:34
============================================================
Phase 5 Test 1: Causal Trace Accuracy
============================================================

  Trace for node 3: [1, 2, 3]
  Chain length: 3 (expected ≥ 3)
  Total weight: 2.00
  ✓ Simple chain trace: PASS

  Conflict summary for node 3:
    Supporting: 2 sources
    Opposing:   1 sources
    Conclusion: supported
    Net weight: 0.70
  ✓ Conflict detection: PASS

  Before invalidation: weight=1.00
  After invalidation:  weight=-1.00
  Affected downstream: [11, 12]
  ✓ Causal invalidation: PASS

  Deep trace (5 hops): [0, 1, 2, 3, 4, 5]
  ✓ Multi-hop trace: PASS
  ✓ Simple chain trace: chain_len=3 (threshold: ≥ 3)
  ✓ Conflict detection: opposing=1 (threshold: > 0)
  ✓ Causal invalidation: affected=2 (threshold: > 0)
  ✓ Multi-hop trace: depth=6 (threshold: ≥ 5)

==================================================
Phase 5 Results saved to: /kaggle/working/FLUX/phases/phase5/RESULTS_PHASE_5.md
All tests passed: True
Ready for Phase 6: True
==================================================

============================================================
Test 1 Overall: PASS
============================================================
[06:52:34]   ◼ CELL Cell 7 — Test 1: Causal Trace Accuracy — PASS




[06:58:37] 
▶ CELL: Cell 8 — Test 2: Multi-Timescale Separation
[06:58:37]   Started: 2026-03-22 06:58:37
============================================================
Phase 5 Test 2: Multi-Timescale Separation
============================================================

  Fast nodes activate at step:   1
  Medium nodes activate at step: 5
  Slow nodes activate at step:   31
  Threshold fraction: 0.8

  Fast < 5?         True  (1)
  Slow > 10?        True  (31)
  Fast ≤ Med ≤ Slow? True  (1 ≤ 5 ≤ 31)
  Separation?       True

  Signal evolution check:
    Diff(1 vs 10 steps):  0.6742
    Diff(10 vs 50 steps): 0.1062
    Diff(1 vs 50 steps):  0.7800
    Signal evolves over time: True

  Node counts:
    Fast:   16
    Medium: 8
    Slow:   4
    Total:  28
    Correct: True

  Activation curves:
    Fast early (steps 1-5):    0.0387
    Fast late  (steps 96-100): 0.0407
    Slow early (steps 1-5):    0.0311
    Slow late  (steps 96-100): 0.2201
  ✓ Fast nodes < 5 steps: step=1 (threshold: < 5)
  ✓ Slow nodes > 10 steps: step=31 (threshold: > 10)
  ✓ Timescale ordering: 1≤5≤31 (threshold: fast ≤ med ≤ slow)
  ✓ Signal evolution: diff=0.7800 (threshold: > 0)
  ✓ Node architecture: 16+8+4=28 (threshold: correct)

==================================================
Phase 5 Results saved to: /kaggle/working/FLUX/phases/phase5/RESULTS_PHASE_5.md
All tests passed: True
Ready for Phase 6: False
==================================================

============================================================
Test 2 Overall: PASS
============================================================
[06:58:46]   ◼ CELL Cell 8 — Test 2: Multi-Timescale Separation — PASS



[06:52:48] 
▶ CELL: Cell 9 — Test 3: Geometry Computation Correctness
[06:52:48]   Started: 2026-03-22 06:52:48
============================================================
Phase 5 Test 3: Geometry Computation Correctness
============================================================

  Test 3a: Signal bending via curvature
    Input shape:  torch.Size([512])
    Output shape: torch.Size([512])
    Bending magnitude: 17.8930
    Shapes match: True
    Signal bent:  True

  Test 3b: Numerical stability
    No NaN: True
    No Inf: True
    Zero input stable: True
    Large input stable: True

  Test 3c: Mass amplification
    Mass=1.0 output norm: 5.4329
    Mass=10.0 output norm: 54.3293
    Higher mass → larger output: True

  Test 3d: Learnable parameters
    Parameters: ['curvature', 'orientation', 'mass', 'manifold.metric_L', 'manifold.connection']...
    Has curvature: True
    Has orientation: True
    Has mass: True
    Total params per node: 525,313

  Test 3e: Manifold geodesic step
    Position moved: True
    Displacement: 0.2122

  Test 3f: Batch processing
    Batch input:  torch.Size([8, 512])
    Batch output: torch.Size([8, 512])
    Shapes match: True

  Test 3g: Forward with trace
    Trace node_id: 0
    Bending: 1.9767
    Influence: 0.2347
    Trace valid: True
  ✓ Signal bending: bending=17.8930 (threshold: > 0)
  ✓ Numerical stability: no NaN/Inf (threshold: all stable)
  ✓ Mass amplification: m1=5.4329 m10=54.3293 (threshold: m10 > m1)
  ✓ Learnable parameters: 525313 params (threshold: curvature+orientation+mass)
  ✓ Geodesic step: moved=True (threshold: position changes)
  ✓ Batch processing: torch.Size([8, 512]) (threshold: same shape)
  ✓ Causal trace: bending=1.9767 (threshold: valid trace)

==================================================
Phase 5 Results saved to: /kaggle/working/FLUX/phases/phase5/RESULTS_PHASE_5.md
All tests passed: True
Ready for Phase 6: True
==================================================

============================================================
Test 3 Overall: PASS
============================================================
[06:52:48]   ◼ CELL Cell 9 — Test 3: Geometry Computation Correctness — PASS






[06:58:53] 
▶ CELL: Cell 10 — Demo 1: Trace Why a Conclusion Was Reached
[06:58:53]   Started: 2026-03-22 06:58:53
=================================================================
  FLUX Phase 5 Demo 1: Why Did You Say That?
=================================================================

  Building Knowledge Graph:
  -------------------------------------------------------
               bird → can_fly          w=0.80  (birds can fly)
               bird → has_wings        w=0.95  (birds have wings)
            penguin → bird             w=1.00  (penguin is a bird)
            penguin → cannot_fly       w=0.90  (penguin cannot fly)
            penguin → lives_in_arctic  w=0.85  (penguin lives in arctic)
            penguin → is_heavy_bodied  w=0.70  (penguin is heavy-bodied)
    is_heavy_bodied → cannot_fly       w=0.60  (heavy body prevents flight)
            sparrow → bird             w=1.00  (sparrow is a bird)
            sparrow → can_fly          w=0.95  (sparrow can fly)

  =======================================================
  Query: Can penguins fly?
  =======================================================

  Path to 'can_fly':
    Chain:   sparrow → can_fly
    Weight:  0.95
    Conflict: False

  Path to 'cannot_fly':
    Chain:   penguin → cannot_fly
    Weight:  0.90

  Evidence Analysis:
    'can_fly' — 2 supporting, 0 opposing → supported
    'cannot_fly' — 2 supporting, 0 opposing → supported

  Conclusion:
    ✓ Birds generally can fly (strong evidence: w=1.75)
    ✓ Penguins specifically cannot fly (exception: w=1.50)
    ✓ Conflict detected and resolved via evidence weight
    ✓ Full causal chain stored — can explain WHY

  =======================================================
  Invalidation Demo: What if we disprove 'birds can fly'?
  =======================================================

  Invalidated: 'bird' as cause
  Affected downstream nodes: ['can_fly', 'has_wings']

  After invalidation:
    'can_fly' — now supported (net=0.15)
    ✓ Invalidation propagated correctly through causal graph

  =======================================================
  Graph stats: 8 nodes, 9 edges
  =======================================================
  ✓ Demo 1 complete
[06:58:53]   ◼ CELL Cell 10 — Demo 1: Trace Why a Conclusion Was Reached — PASS







[07:01:22] 
▶ CELL: Cell 11 — Demo 2: Fast vs Slow Node Activation
[07:01:22]   Started: 2026-03-22 07:01:22
=================================================================
  FLUX Phase 5 Demo 2: Fast vs Slow Node Activation
=================================================================

  Input signal: norm=11.1826
  Architecture: 16 fast + 8 medium + 4 slow = 28 nodes
  Total parameters: 14,708,767

  Activation Timeline (threshold_frac=0.8):
  ------------------------------------------------------------
    Step        Fast      Medium        Slow
  ------------------------------------------------------------
       1      0.0336      0.0392      0.0115  | ████████
       2      0.0404      0.0667      0.0225  | █████████
       3      0.0417      0.0859      0.0329  | █████████
       5      0.0420      0.1087      0.0522  | █████████
      10      0.0420      0.1270      0.0927  | █████████
      20      0.0420      0.1306      0.1482  | ██████████
      30      0.0420      0.1307      0.1814  | ██████████
      50      0.0420      0.1307      0.2132  | ██████████
      75      0.0420      0.1307      0.2260  | ██████████
     100      0.0420      0.1307      0.2296  | ██████████

  Activation Threshold Crossing (80% of converged value):
  ---------------------------------------------
    Fast nodes activate at step:   1
    Medium nodes activate at step: 5
    Slow nodes activate at step:   31

  ✓ Chart saved: demo5_timescale_activation.png

  Key Insights:
    • Fast nodes (syntax) respond almost immediately
    • Medium nodes (semantics) accumulate over ~10 steps
    • Slow nodes (concepts) build gradually over many steps
    • This mirrors human cognition: fast reflexes + slow reasoning
    • All timescales operate simultaneously — no layer separation

  Node Statistics:
    total_nodes              : 28
    fast_nodes               : 16
    medium_nodes             : 8
    slow_nodes               : 4
    total_params             : 14708767
    steps_processed          : 100
    fast_activation          : 0.0420
    medium_activation        : 0.1307
    slow_activation          : 0.2296
    mix_weights              : [0.3556267023086548, 0.33828258514404297, 0.30609074234962463]

  ✓ Demo 2 complete




[06:57:56] 
▶ CELL: Cell 12 — Interactive Exploration
[06:57:56]   Started: 2026-03-22 06:57:56
  Building Custom Knowledge Graph:
  -------------------------------------------------------
                FLUX → physics_compute   w=1.00
     physics_compute → no_weights        w=0.90
          no_weights → no_backprop       w=0.85
         no_backprop → realtime_learn    w=0.80
      realtime_learn → oneshot_facts     w=0.75
                FLUX → CGN               w=0.90
                 CGN → stores_WHY        w=0.85
          stores_WHY → explainable       w=0.80

  Reasoning Traces:
  -------------------------------------------------------
    Why 'oneshot_facts'?
      Chain: FLUX → physics_compute → no_weights → no_backprop → realtime_learn → oneshot_facts
      Total weight: 4.30
[06:57:56]   📊 trace(oneshot_facts): chain_len=6
    Why 'explainable'?
      Chain: FLUX → CGN → stores_WHY → explainable
      Total weight: 2.55
[06:57:56]   📊 trace(explainable): chain_len=4

  Full Pipeline Processing:
  -------------------------------------------------------
    'Neural networks compute via matrix multiplication'
      Output norm: 0.1403  Fast: 0.0007  Slow: 0.0095  Traces: 28
    'FLUX computes via geometric signal bending on manifolds'
      Output norm: 0.1520  Fast: 0.0008  Slow: 0.0102  Traces: 28
    'Causal reasoning requires knowing WHY not just WHAT'
      Output norm: 0.1358  Fast: 0.0007  Slow: 0.0091  Traces: 28
    'Temperature controls how much the system can change'
      Output norm: 0.1454  Fast: 0.0007  Slow: 0.0095  Traces: 28

  CGN Network Statistics:
  ---------------------------------------------
    total_nodes              : 28
[06:57:58]   📊 total_nodes: 28
    fast_nodes               : 16
[06:57:58]   📊 fast_nodes: 16
    medium_nodes             : 8
[06:57:58]   📊 medium_nodes: 8
    slow_nodes               : 4
[06:57:58]   📊 slow_nodes: 4
    total_params             : 14708767
[06:57:58]   📊 total_params: 14708767
    steps_processed          : 0
[06:57:58]   📊 steps_processed: 0
    fast_activation          : 0.0007
[06:57:58]   📊 fast_activation: 0.000711910251993686
    medium_activation        : 0.0041
[06:57:58]   📊 medium_activation: 0.0041296882554888725
    slow_activation          : 0.0095
[06:57:58]   📊 slow_activation: 0.009514817036688328
    mix_weights              : [0.356, 0.338, 0.306]
[06:57:58]   📊 mix_weights: [0.3556267023086548, 0.33828258514404297, 0.30609074234962463]

  ✓ Interactive exploration complete
[06:57:59]   ◼ CELL Cell 12 — Interactive Exploration — PASS