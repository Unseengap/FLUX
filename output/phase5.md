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




