
Training GridToWave for discriminative embeddings...
============================================================
Training: 100%
 300/300 [00:16<00:00, 21.52it/s, loss=0.3347, valid=8]

✓ Training complete!
  Final loss: 0.2625
  Initial loss: 0.5861
  Improvement: 55.2%

Testing wave discrimination...
  Red@5,5 vs Teal@5,5: 0.9999
  Red@5,5 vs Red@2,2:  0.9858
⚠ Waves too similar - but agent uses direct grid fallback


 [Agent] Reset at position (0, 0)
============================================================
FLUX-POWERED TREASURE HUNT
============================================================
Grid size: 10x10
Visible treasures: 4
Hidden treasures: 2
Total to find: 6

Treasure locations:
  (9, 4): color=1 → triggers hidden at (7, 7)
  (8, 5): color=2 → triggers hidden at (9, 1)
  (4, 2): color=3
  (5, 9): color=4

Using FLUX GridToWave for observation encoding!


Running FLUX-powered exploration...
============================================================
    [Agent] New treasure spotted at (4,2) color=3
    [Agent] New treasure spotted at (5,9) color=4
    [Agent] New treasure spotted at (8,5) color=2
    [Agent] New treasure spotted at (9,4) color=1
    [Agent] New target: (8, 5) (unexplored)
  Step 13: 💎 TREASURE at (8, 5)!
  Step 13: 🔮 HIDDEN revealed at (9, 1)!
    [Agent] Grid change at (8,5): 2 → 0
    [Agent] Grid change at (9,1): 0 → 6
    [Agent] New treasure spotted at (9,1) color=6
    [Agent] New target: (8, 5) (high_curiosity)
    [Agent] New target: (4, 2) (high_curiosity)
  Step 22: 💎 TREASURE at (4, 2)!
    [Agent] Grid change at (4,2): 3 → 0
    [Agent] New target: (4, 2) (high_curiosity)
    [Agent] New target: (4, 2) (high_curiosity)

─── Step 25 ───
  Position: (4, 1)
  Found: 2/6 treasures
  Hidden revealed: 1/2
  Coverage: 99.0%
  Treasures seen: 5
  Avg wave Δ: 0.0065
  Direct change rate: 8.0%
  Current target: (4, 2) (high_curiosity)
  Top curiosity: (4,2)=93.7 (9,4)=91.7 (5,9)=91.7 
    [Agent] New target: (5, 9) (high_curiosity)
  Step 34: 💎 TREASURE at (5, 9)!
    [Agent] Grid change at (5,9): 4 → 0
    [Agent] New target: (9, 4) (high_curiosity)
  Step 43: 💎 TREASURE at (9, 4)!
  Step 43: 🔮 HIDDEN revealed at (7, 7)!
    [Agent] Grid change at (7,7): 0 → 5
    [Agent] Grid change at (9,4): 1 → 0
    [Agent] New treasure spotted at (7,7) color=5
    [Agent] New target: (9, 4) (high_curiosity)
    [Agent] New target: (9, 4) (high_curiosity)
    [Agent] New target: (9, 4) (high_curiosity)
    [Agent] New target: (9, 1) (high_curiosity)
  Step 50: 💎 TREASURE at (9, 1)!

─── Step 50 ───
  Position: (9, 1)
  Found: 5/6 treasures
  Hidden revealed: 2/2
  Coverage: 100.0%
  Treasures seen: 6
  Avg wave Δ: 0.0010
  Direct change rate: 8.0%
  Current target: (9, 1) (high_curiosity)
  Top curiosity: (9,1)=111.1 (9,4)=91.8 (7,7)=54.1 
    [Agent] Grid change at (9,1): 6 → 0
    [Agent] New target: (9, 1) (high_curiosity)
    [Agent] New target: (9, 1) (high_curiosity)
    [Agent] New target: (9, 1) (high_curiosity)
    [Agent] New target: (9, 1) (high_curiosity)
    [Agent] New target: (7, 7) (high_curiosity)
  Step 65: 💎 TREASURE at (7, 7)!

─── Step 65 ───
  Position: (7, 7)
  Found: 6/6 treasures
  Hidden revealed: 2/2
  Coverage: 100.0%
  Treasures seen: 6
  Avg wave Δ: 0.0011
  Direct change rate: 8.0%
  Current target: (7, 7) (high_curiosity)
  Top curiosity: (7,7)=101.5 (9,1)=56.1 (9,4)=42.5 

============================================================
GAME OVER
============================================================

📊 FINAL STATS:
  Steps taken: 65
  Treasures found: 6/6
  Hidden revealed: 2/2
  Grid coverage: 100.0%
  Waves encoded: 65
  Treasures spotted: 6
  Result: ✓ SUCCESS

📜 KEY EVENTS (16 total):
  Step 13: 💎 COLLECTED at (8, 5)
  Step 13: 🔮 HIDDEN revealed at (9, 1)
  Step 14: 📍 Grid change detected
  Step 14: ⚡ Wave Δ=0.0677
  Step 22: 💎 COLLECTED at (4, 2)
  Step 23: 📍 Grid change detected
  Step 23: ⚡ Wave Δ=0.0958
  Step 34: 💎 COLLECTED at (5, 9)
  Step 35: 📍 Grid change detected
  Step 43: 💎 COLLECTED at (9, 4)
  Step 43: 🔮 HIDDEN revealed at (7, 7)
  Step 44: 📍 Grid change detected
  Step 44: ⚡ Wave Δ=0.0254
  Step 50: 💎 COLLECTED at (9, 1)
  Step 51: 📍 Grid change detected
  Step 65: 💎 COLLECTED at (7, 7)

📝 AGENT LOGS (last 15):
  Grid change at (5,9): 4 → 0
  New target: (9, 4) (high_curiosity)
  Grid change at (7,7): 0 → 5
  Grid change at (9,4): 1 → 0
  New treasure spotted at (7,7) color=5
  New target: (9, 4) (high_curiosity)
  New target: (9, 4) (high_curiosity)
  New target: (9, 4) (high_curiosity)
  New target: (9, 1) (high_curiosity)
  Grid change at (9,1): 6 → 0
  New target: (9, 1) (high_curiosity)
  New target: (9, 1) (high_curiosity)
  New target: (9, 1) (high_curiosity)
  New target: (9, 1) (high_curiosity)
  New target: (7, 7) (high_curiosity)
CPU times: user 2.94 s, sys: 140 ms, total: 3.08 s
Wall time: 3.21 s



Spatial Memory State:
┌─────────────────────┐
│ ● ● ● ● ● ● ● ● ● ● │
│ ● ● ● ● ● ● ● ● ● ● │
│ ● ● ● ● ● ● ● ● ● ● │
│ ❄ ● ● ● ● ● ● ● ● ● │
│ 🧊❄ 🧊● ● 🧊🧊🧊🧊🧊│
│ ❄ ❄ ❄ ● ● ● 🧊🧊🧊🧊│
│ ❄ ● ❄ ● ● ● ❄ ● ❄ 🧊│
│ ● 🧊🧊● ● ● @ 🧊● ❄ │
│ 🧊🧊🧊🧊🧊🧊🧊🧊🧊● │
│ ❄ 🧊🧊🧊🧊🧊● 🧊🧊● │
└─────────────────────┘

Legend: @ agent, 🧊 high ice, ❄ ice, ! change, ● explored, · visited