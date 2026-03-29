# Spatial Memory Specification: The Ice & Water Model

> **Phase:** 9 ARC / ARC-AGI-3 Agent
> **Component:** `SpatialMemory` (spatial_memory.py)
> **Purpose:** Dual-field navigation for curiosity-driven exploration

---

## Overview

Traditional AI agents treat space as text tokens or flat coordinates. FLUX agents have **physics-native spatial awareness** through two complementary fields:

| Field | Metaphor | Function |
|-------|----------|----------|
| **Exploration Mass** | Water | Where I've been, what I saw |
| **Curiosity Ice** | Ice floating in water | What's interesting, where to look |

Together, these fields enable:
- ✓ Spatial memory (knows where things were)
- ✓ Curiosity-driven exploration (pulled toward anomalies)
- ✓ Change detection (notices when reality differs from memory)
- ✓ Efficient navigation (uses mass trail to plan paths)

---

## The Ice & Water Metaphor

```
┌─────────────────────────────────────────┐
│  Water (Explorable Space)               │
│                                         │
│  ~~~~ 🧊 ~~~~ ~~~~ 🧊🧊 ~~~~            │
│  ~~~~ ~~~~ ~~~~ 🧊 ~~~~ ~~~~            │  🧊 = Ice (high curiosity)
│  ~~~~ ~~~~ 🧊 ~~~~ ~~~~ ~~~~            │  ~ = Water (ambient space)
│  ~~~~ ~~~~ ~~~~ ~~~~ ~~~~ 🧊            │
│                                         │
│  Ice floats UP because it's DIFFERENT   │
│  (lower density = higher signal/noise)  │
│                                         │
└─────────────────────────────────────────┘
```

**Water** = The explorable field, ambient, uniform density
**Ice** = Anomalies that *float up* as curiosity targets

The agent is gravitationally pulled toward ice. It's not about avoiding explored areas — it's about being **attracted to what's interesting**.

---

## Dual-Field Architecture

### Field 1: Exploration Mass

Tracks where the agent has been and what it observed.

```python
@dataclass
class ExplorationMassField:
    visit_count: Tensor      # [H, W] — times visited each cell
    exploration_mass: Tensor # [H, W] — accumulated mass (decays)
    last_observation: Tensor # [H, W, features] — what was there
    last_visit_time: Tensor  # [H, W] — when visited
```

**Properties:**
- Gains mass when visited
- Decays slowly over time (0.99 per step)
- Stores observation features for change detection
- Enables path planning via mass gradient

### Field 2: Curiosity Ice

Highlights interesting locations that deserve attention.

```python
@dataclass
class CuriosityIceField:
    curiosity_field: Tensor  # [H, W] — salience score
    change_detected: Tensor  # [H, W] — boolean flags
```

**Ice spawns when:**
- Color contrast detected (red cell in sea of blue)
- Isolated objects found
- Pattern breaks identified
- **Change detected** — previously visited area looks different

---

## Ice Formation Rules

| Trigger | Ice Amount | Priority |
|---------|------------|----------|
| Change in visited area | `delta * 5.0` | **Highest** |
| Color contrast | `contrast_score` | High |
| Isolated cell | `2.0` | High |
| Non-background cell | `0.5` | Medium |
| Unvisited territory | `0.5` | Low |

The **change detection** case is the killer feature:
- You KNOW what was there before (mass field stores history)
- You SEE ice where there wasn't ice (new anomaly)
- This is high-value signal — revisit immediately

---

## Navigation Algorithm

### Combined Behavior Matrix

| Exploration Mass | Curiosity Ice | Behavior |
|------------------|---------------|----------|
| Zero (never visited) | None | Low priority |
| Zero (never visited) | **Has ice** | **Go explore!** |
| High (visited) | None | Skip — known territory |
| High (visited) | **New ice** | **ALERT — go back!** |

### Path Planning via Mass Gradient

```
Goal: Get to ice at position (7, 3)
Current: Position (1, 1)

Without mass trail:
  → Random walk, might take 50 steps
  
With mass trail:
  → Follow your own gravitational breadcrumbs
  → You've been to (3,2) before, high mass there
  → From (3,2) you can see path to (5,3)
  → Efficient: (1,1) → (3,2) → (5,3) → (7,3)
  → 3 hops, not 50 random steps
```

Uses A* with mass as heuristic bonus — prefers paths through known (high-mass) territory.

---

## The Agent Loop

```
┌────────────────────────────────────────────────────────────┐
│                   SPATIAL AGENT LOOP                        │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  1. OBSERVE                                                 │
│     └─→ Encode current view into wave features              │
│                                                             │
│  2. UPDATE MASS                                             │
│     └─→ Current position gains exploration mass             │
│     └─→ Store what we observed here                         │
│                                                             │
│  3. CHECK FOR CHANGE                                        │
│     └─→ Compare current to last_observation                 │
│     └─→ If different: ICE FORMS (curiosity spike)           │
│                                                             │
│  4. DETECT ANOMALIES                                        │
│     └─→ Scan for color contrast, isolated cells             │
│     └─→ These also create ice                               │
│                                                             │
│  5. APPLY DECAY                                             │
│     └─→ Mass decays 0.99x, ice decays 0.95x                 │
│                                                             │
│  6. FIND TARGET                                             │
│     └─→ Get highest curiosity (ice) location                │
│                                                             │
│  7. PLAN PATH                                               │
│     └─→ A* with mass gradient bonus                         │
│                                                             │
│  8. ACT                                                     │
│     └─→ Take step toward target                             │
│                                                             │
│  9. REPEAT                                                  │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

---

## Configuration

```python
SPATIAL_MEMORY_CONFIG = {
    'max_grid_size': 30,          # Max ARC grid size
    'feature_dim': 64,            # Features per location
    'curiosity_threshold': 0.1,   # Change amount to trigger ice
    'mass_decay': 0.99,           # Exploration mass decay per step
    'ice_decay': 0.95,            # Curiosity ice decay per step
    'visit_mass_gain': 1.0,       # Mass gained per visit
    'change_ice_multiplier': 5.0, # Ice multiplier for changes
}
```

---

## Visualization

ASCII visualization shows both fields combined:

```
┌─────────────────────────┐
│ @ · · · ❄ · · · · ·    │  @ = Agent
│ ● ● · · · · · · · ·    │  🧊 = High ice (curiosity peak)
│ ● ○ · · · · · · 🧊·    │  ❄ = Medium ice
│ ○ · · · · · · · · ·    │  ! = Change detected
│ · · · ! · · · · · ·    │  ● = High mass (well explored)
│ · · · · · · · · · ·    │  ○ = Medium mass
│ · · · · · · · · · ·    │  · = Visited once
└─────────────────────────┘   (empty) = Unknown
```

---

## Why This Matters for ARC-AGI-3

ARC-AGI-3 is **blind exploration** — no instructions, just observe → act → observe.

The dual-field model solves key ARC-AGI-3 challenges:

| Challenge | Solution |
|-----------|----------|
| Where to explore first? | Go to ice (high curiosity) |
| How to navigate efficiently? | Follow mass gradient |
| How to notice important changes? | Ice spawns at changes |
| How to remember terrain? | Mass field stores observations |
| How to avoid wasted steps? | Plan paths through known territory |

---

## Integration with FLUX Components

| FLUX Component | Role in Spatial Memory |
|----------------|------------------------|
| `ResonanceField` | Could extend to encode spatial positions natively |
| `GravitationalRelevance` | Query "where to explore?" → pulled to low-mass/high-ice |
| `CausalGeometryNode` | Track WHY each area gained mass ("saw red block at step 5") |
| `EpisodicMemory` | Store exploration history as episodes |
| `GridToWave` | Encode grid observations to wave features |

---

## Files

```
phases/phase9_arc/
├── spatial_memory.py          # This implementation
├── SPATIAL_MEMORY_SPEC.md     # This document
├── arc_agent.py               # Uses SpatialMemory
└── notebooks/
    └── spatial_memory_demo.ipynb  # Interactive demo
```

---

## Future Extensions

1. **3D Mass Field** — extend to volumetric environments
2. **Multi-Agent Ice Sharing** — agents share curiosity maps
3. **Temporal Ice** — ice patterns over time predict dynamics
4. **Semantic Ice Labeling** — annotate what kind of anomaly
5. **Causal Ice Chains** — ice at B because of observation at A

---

*The physics doesn't lie. Curiosity pulls you toward the unknown. Mass anchors you to the known. Together, they create efficient exploration.*
