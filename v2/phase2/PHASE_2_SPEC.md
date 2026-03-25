# Phase 2 v2 — Resonance Field (with Decode Gate)

> **Roadmap:** ROADMAP_WAVE_FIRST.md  
> **Checkpoint:** `checkpoints/phase2_v2.phase.pt`  
> **Prerequisite:** Phase 1 v2 trained checkpoint (`phase1_v2.phase.pt`)

---

## Goal

Waves now live in a resonance field. The field stores wave patterns as
attractors. **Prove that: wave → field → wave preserves decodability.**

This phase introduces the critical fix that the original roadmap missed.

---

## The Original Bug (What Phase 9 Was Actually About)

```
Original Phase 2:
  wave_to_field/field_to_wave were added to the model
  but NEVER trained with a loss that required them to be invertible.
  
Original Phase 7 (FLUXModel):
  .detach() was used when passing features between components
  → gradients could NEVER flow back to these projections
  → they stayed at random init FOREVER

Original Phase 9 (WaveToText):
  WTT had to decode the output of field_to_wave(wave_to_field(wave))
  where field_to_wave was a RANDOM PROJECTION
  → mode collapse: WTT always predicted the same token
  → context collapse: all contexts looked the same to WTT
  
Time lost: 8 phases before catching the bug.
```

---

## The Fix in v2

```python
# Original Phase 2 (broken):
features = field.wave_to_feature(wave)   # Only direction trained
# field_to_wave: never trained, stays random

# Phase 2 v2 (fixed):
field_vec     = wave_to_field(wave)       # [432] → [512]
reconstructed = field_to_wave(field_vec)  # [512] → [432]

loss = mse(wave, reconstructed)           # Reconstruction loss
     + (1 - cos_sim(wave, reconstructed)) # Direction loss
     + wtt.forward_batch(reconstructed, target_bytes)  # DECODE LOSS ← THE FIX
```

The decode loss forces `wave_to_field` and `field_to_wave` to learn
**invertible projections** — not just any projections, but ones where
a frozen WTT can decode the reconstructed wave back to text.

---

## Components

### ResonanceField (`field.py`)
- 3D spatial field: `[64, 64, 64, 512]` — 16M parameters in the state
- Physics-based updates: perturbation + settling (no gradient on state)
- `wave_to_location`: SpatialProjection MLP — learns coordinate mapping
- `wave_to_feature`: Linear(432, 512) — learns feature mapping
- Both `wave_to_location` + `wave_to_feature` trained WITH gradients here

### WaveToField (`wave_to_field.py`) ← NEW in v2
- Input:  `[432]` wave vector
- Output: `[512]` field feature vector
- Architecture: LayerNorm → Linear(432, 512) → GELU → LayerNorm → Linear(512, 512)

### FieldToWave (`field_to_wave.py`) ← NEW in v2
- Input:  `[512]` field feature vector
- Output: `[432]` reconstructed wave vector
- Architecture: LayerNorm → Linear(512, 512) → GELU → LayerNorm → Linear(512, 432)
- Trained with decode loss so WTT can decode the output

### AttractorCatalog (`attractor.py`)
- Tracks stable attractors (mass > threshold)
- Verifies persistence (no-forgetting invariant)
- Serialized into checkpoint for analysis

---

## Training: What's Frozen vs Trained

| Component | Status | Reason |
|-----------|--------|--------|
| CSE (from Phase 1) | ✓ **FROZEN** | Trust phase 1 wave space |
| WaveChunker (from Phase 1) | ✓ **FROZEN** | Trust phase 1 chunking |
| WaveToText (from Phase 1) | ✓ **FROZEN** | WTT is the decode oracle |
| WaveToField | ← **TRAINED** | New component needs to learn |
| FieldToWave | ← **TRAINED** | New component needs to learn |
| field.wave_to_location | ← **TRAINED** | Spatial spread matters |
| field.wave_to_feature | ← **TRAINED** | Feature slot initialization |
| field.state/energy/mass | Physics update | Not gradient-based |

---

## Phase 2 Decode Gate (Harder Than Phase 1)

The Phase 2 decode gate tests the ENTIRE pipeline including the field bridge:

```
text → CSE (frozen) → Chunker (frozen) → WaveToField → FieldToWave → WTT (frozen) → text
```

This is exactly what failed in Phase 9 of the original roadmap.
The field bridge adds an extra hop. If WaveToField or FieldToWave are
not trained properly, the decode gate fails HERE — not 7 phases later.

**Threshold:** avg byte accuracy > 90%, min > 70% (same as Phase 1 gate)

---

## No-Forgetting Invariant

Physics-based updates are local by construction:

```python
# Only the neighborhood of wave_to_field_coords(wave) is updated
# Everything else in the field is untouched
field.perturb(wave_vector, strength=1.0)
```

Attractor survival rate after 100 interfering updates: **≥ 80%**  
Transformer baseline: 30-80% catastrophic forgetting (literature)

---

## Acceptance Criteria

- [ ] Field forms stable attractors from repeated wave patterns
- [ ] `WaveToField` and `FieldToWave` trained TOGETHER (not separately)
- [ ] Round-trip cosine: `wave → field → wave > 0.85`
- [ ] **Decode gate:** `field → wave → WTT → text > 90%` byte accuracy
- [ ] Old attractors survive 100 new learning steps (≥ 80% survival rate)
- [ ] Local update only — no global field change
- [ ] All 3 tests pass

---

## File Structure

```
v2/phase2/
├── __init__.py
├── field.py                 ← ResonanceField (adapted from phase2, updated imports)
├── attractor.py             ← AttractorCatalog
├── field_ops.py             ← Field analysis utilities
├── wave_to_field.py         ← NEW: wave [432] → field [512]
├── field_to_wave.py         ← NEW: field [512] → wave [432]
├── train_field.py           ← NEW: Joint training WITH decode loss
├── demo_phase2_demo1.py     ← Wave → field → wave → text round-trip
├── demo_phase2_demo2.py     ← No forgetting: old attractors survive
├── test_phase2_test1.py     ← Attractor formation
├── test_phase2_test2.py     ← wave→field→wave cosine > 0.85
├── test_phase2_test3.py     ← wave→field→wave→TEXT byte accuracy > 90%
└── PHASE_2_SPEC.md          ← This file
```

---

## Running

```bash
# Train (requires Phase 1 v2 checkpoint first)
python v2/phase1/train_codec.py --steps 30000  # Phase 1 first
python v2/phase2/train_field.py --steps 20000  # Then Phase 2

# Test
python v2/phase2/test_phase2_test1.py
python v2/phase2/test_phase2_test2.py
python v2/phase2/test_phase2_test3.py

# Demo
python v2/phase2/demo_phase2_demo1.py
python v2/phase2/demo_phase2_demo2.py
```

---

## Checkpoint Format

```python
{
    'phase': 2,
    'version': 'v2',
    'timestamp': str,
    'step': int,
    'config': {
        'field_h': 64, 'field_w': 64, 'field_d': 64,
        'field_features': 512,
        'wave_dim': 432,
    },
    'state_dict': {
        'field':       OrderedDict,   # ResonanceField full state
        'bridge_wtf':  OrderedDict,   # WaveToField weights
        'bridge_ftw':  OrderedDict,   # FieldToWave weights
    },
    'attractor_catalog': List[dict],  # Serialized attractors
    'metrics': {
        'best_recon_loss':  float,
        'best_decode_loss': float,
        'num_attractors':   int,
    },
}
```
