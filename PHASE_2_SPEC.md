# Phase 2 Specification: Resonance Field Core (RFC)
## Replacing Weight Matrices With a Living Field

> Prerequisites: Phase 1 checkpoint must exist and pass smoke test.
> Copilot: Open SPECIFICATION.md + PHASE_1_SPEC.md + this file while building.

---

## Goal

Build a dynamic field tensor that:
- Forms stable attractors when shown repeated patterns
- Maintains those attractors when shown new patterns
- Updates locally (never globally)
- Saves and loads as a .flx file

---

## Load Phase 1 First

```python
# Every Phase 2 script starts with this
from pathlib import Path
import sys
sys.path.append('../phase1')
from cse import ContinuousSemanticEncoder

def load_phase1():
    checkpoint = Path('../../checkpoints/phase1.phase.pt')
    assert checkpoint.exists(), "Phase 1 checkpoint missing — run Phase 1 first"
    cse = ContinuousSemanticEncoder.load(str(checkpoint))
    print("✓ Phase 1 (CSE) loaded successfully")
    return cse
```

---

## File Structure

```
phases/phase2/
├── PHASE_2_SPEC.md           ← This file
├── field.py                  ← ResonanceField class — build first
├── attractor.py              ← AttractorCatalog — build second
├── field_ops.py              ← Energy functions — build third
├── flux_format.py            ← .flx save/load — build fourth
├── train_field.py            ← Training script — build fifth
├── demo_phase2_demo1.py      ← Attractor formation live
├── demo_phase2_demo2.py      ← No forgetting demo
├── test_phase2_test1.py      ← Attractor formation test
├── test_phase2_test2.py      ← Local update test
├── test_phase2_test3.py      ← Field stability test
└── RESULTS_PHASE_2.md        ← Auto-generated
```

---

## field.py — Core Implementation

```python
"""
ResonanceField: Replaces all weight matrices in FLUX.

The field is a 3D spatial tensor where:
- Each location holds a feature vector (what's stored here)
- Each location has an energy scalar (how stable this location is)
- Each location has a mass scalar (how much evidence supports this)
- Each location has a velocity vector (current rate of change)

Learning = perturbation + settling (no backprop)
Memory   = stable attractor locations (energy minima)
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

FIELD_H = 64
FIELD_W = 64  
FIELD_D = 64
FIELD_FEATURES = 512

@dataclass
class FieldLocation:
    """A coordinate in the 3D field."""
    h: int
    w: int
    d: int
    
    def to_tensor(self) -> Tensor:
        return torch.tensor([self.h, self.w, self.d], dtype=torch.float)

class ResonanceField(nn.Module):
    """
    The living field that replaces all weight matrices.
    
    Stores knowledge as energy minima (attractors) in a 3D landscape.
    New information creates local perturbations.
    The field settling into new energy minimum = learning.
    
    Key invariant: Updates are LOCAL. Far regions never change unless
    the perturbation energy is physically large enough to reach them.
    This is how catastrophic forgetting is eliminated by physics.
    """
    
    def __init__(
        self,
        h: int = FIELD_H,
        w: int = FIELD_W,
        d: int = FIELD_D,
        features: int = FIELD_FEATURES,
        device: str = 'cuda'
    ):
        super().__init__()
        self.h = h
        self.w = w
        self.d = d
        self.features = features
        self.device = device
        
        # Main field state: features at each location
        # Initialized as near-zero Gaussian (empty field)
        self.state = nn.Parameter(
            torch.randn(h, w, d, features, device=device) * 0.01
        )
        
        # Energy landscape: scalar energy at each location
        # Higher energy = less stable = more likely to change
        self.energy = nn.Parameter(
            torch.ones(h, w, d, 1, device=device)
        )
        
        # Evidence mass: how much evidence supports each location
        # Starts at 0, grows as patterns are reinforced
        self.mass = torch.zeros(h, w, d, 1, device=device)
        
        # Projection: SemanticWave → field location
        # Maps 432-dim wave to 3D field coordinates
        self.wave_to_location = nn.Linear(432, 3)
        
        # Projection: SemanticWave → field feature
        # Maps 432-dim wave to 512-dim field feature vector
        self.wave_to_feature = nn.Linear(432, features)
    
    def wave_to_field_coords(self, wave_vector: Tensor) -> FieldLocation:
        """
        Map a semantic wave vector to a location in the 3D field.
        Similar waves should map to similar locations.
        
        Args:
            wave_vector: [432] semantic wave (mean pooled)
        Returns:
            FieldLocation in [0, H) x [0, W) x [0, D)
        """
        coords = torch.sigmoid(self.wave_to_location(wave_vector))
        h = int(coords[0].item() * (self.h - 1))
        w = int(coords[1].item() * (self.w - 1))
        d = int(coords[2].item() * (self.d - 1))
        return FieldLocation(h, w, d)
    
    def perturb(
        self, 
        wave_vector: Tensor,
        perturbation_strength: float = 1.0
    ) -> FieldLocation:
        """
        Apply a perturbation to the field at the location corresponding
        to the input wave. This is both the forward pass AND learning.
        
        Args:
            wave_vector: [432] input semantic wave (mean pooled)
            perturbation_strength: how strongly to perturb
        Returns:
            FieldLocation: where the perturbation was applied
        """
        location = self.wave_to_field_coords(wave_vector)
        target_feature = self.wave_to_feature(wave_vector)
        
        # Compute influence radius based on perturbation strength
        radius = max(1, int(perturbation_strength * 4))
        
        # Get neighborhood slice
        h_slice = slice(
            max(0, location.h - radius), 
            min(self.h, location.h + radius + 1)
        )
        w_slice = slice(
            max(0, location.w - radius),
            min(self.w, location.w + radius + 1)
        )
        d_slice = slice(
            max(0, location.d - radius),
            min(self.d, location.d + radius + 1)
        )
        
        # Compute distance-weighted update for neighborhood
        neighborhood = self.state[h_slice, w_slice, d_slice]
        h_size = h_slice.stop - h_slice.start
        w_size = w_slice.stop - w_slice.start
        d_size = d_slice.stop - d_slice.start
        
        # Distance weights — closer = stronger update
        distances = self._compute_distances(
            location, h_slice, w_slice, d_slice
        )
        weights = torch.exp(-distances / radius).unsqueeze(-1)
        
        # Update: move neighborhood toward target feature
        delta = (target_feature.unsqueeze(0).unsqueeze(0).unsqueeze(0) 
                 - neighborhood)
        update = weights * delta * perturbation_strength * 0.01
        
        # Apply LOCAL update only
        with torch.no_grad():
            self.state[h_slice, w_slice, d_slice] += update
            self.mass[h_slice, w_slice, d_slice] += weights * 0.01
        
        return location
    
    def query(self, wave_vector: Tensor, k: int = 8) -> Tensor:
        """
        Retrieve k nearest field features to a query wave.
        This is the "read" operation — like attention but field-based.
        
        Args:
            wave_vector: [432] query wave
            k: number of nearest neighbors
        Returns:
            [k, features] nearest field states
        """
        location = self.wave_to_field_coords(wave_vector)
        
        # Search neighborhood for k nearest features
        search_radius = 16  # Search wider than update radius
        h_slice = slice(
            max(0, location.h - search_radius),
            min(self.h, location.h + search_radius + 1)
        )
        w_slice = slice(
            max(0, location.w - search_radius),
            min(self.w, location.w + search_radius + 1)
        )
        d_slice = slice(
            max(0, location.d - search_radius),
            min(self.d, location.d + search_radius + 1)
        )
        
        neighborhood = self.state[h_slice, w_slice, d_slice]
        # [h_r, w_r, d_r, features]
        flat = neighborhood.view(-1, self.features)
        
        # Find k most similar to target feature
        target = self.wave_to_feature(wave_vector)
        similarities = F.cosine_similarity(
            flat, target.unsqueeze(0).expand_as(flat)
        )
        top_k = similarities.topk(min(k, flat.shape[0]))
        return flat[top_k.indices]
    
    def _compute_distances(
        self, location: FieldLocation,
        h_slice: slice, w_slice: slice, d_slice: slice
    ) -> Tensor:
        """Compute L2 distances from location to all points in slices."""
        # TODO: Copilot — build distance grid for the slice range
        pass
    
    def total_energy(self) -> float:
        """Total energy of the field — should be stable after settling."""
        return self.energy.sum().item()
    
    def num_attractors(self, threshold: float = 0.5) -> int:
        """Count approximate number of stable attractors (energy minima)."""
        # TODO: Copilot — count local energy minima in field
        pass
```

---

## attractor.py — Attractor Catalog

```python
"""
AttractorCatalog: Tracks and catalogs stable attractors in the field.

An attractor is a location in the field that has:
- High mass (reinforced by many observations)
- Low energy (stable, resists change)
- Strong feature vector (clear identity)

The catalog enables:
- Listing what the model "knows"
- Verifying that old attractors survive new learning
- Providing the causal arrow mechanism (Phase 5 prep)
"""

class AttractorCatalog:
    def __init__(self, field: 'ResonanceField'):
        self.field = field
        self.attractors = []  # List of known stable attractors
    
    def scan_and_update(self, mass_threshold: float = 0.3):
        """
        Scan the field for new stable attractors and add to catalog.
        Called periodically during training.
        """
        pass
    
    def verify_attractor_persistence(
        self, 
        attractor_id: int,
        similarity_threshold: float = 0.8
    ) -> bool:
        """
        Check that a previously registered attractor still exists.
        Returns True if attractor is intact, False if destroyed.
        Used in the no-forgetting test.
        """
        pass
    
    def get_all(self) -> list:
        """Return all known attractors."""
        return self.attractors
```

---

## flux_format.py — .flx File Format

```python
"""
FLUX model file format (.flx).

Unlike .pt files which are just weight dicts,
.flx files capture the full living state of the model:
- Field tensor (the "knowledge landscape")
- Attractor catalog (what stable things are known)
- Causal graph (why things are known) — placeholder until Phase 5
- Learning state (how many steps, temperature, etc.)

A .flx file can be loaded and the model continues from EXACTLY
where it left off — including mid-learning. There is no distinction
between "trained model" and "model in training" in FLUX.
"""

import torch
import json
from pathlib import Path

def save_flux(
    field: 'ResonanceField',
    cse: 'ContinuousSemanticEncoder', 
    path: str,
    metadata: dict = None
):
    """
    Save full FLUX state to .flx file.
    .flx is just a torch save with a specific schema.
    """
    state = {
        'format': 'FLUX',
        'version': '0.1',
        'phase': 2,
        
        # Component states
        'field_state': field.state_dict(),
        'cse_state': cse.state_dict(),
        
        # Field knowledge
        'field_mass': field.mass,
        'attractor_catalog': [],  # Populated in Phase 5+
        'causal_graph': {},       # Populated in Phase 5+
        
        # Learning metadata
        'learning_steps': metadata.get('steps', 0) if metadata else 0,
        'can_continue_learning': True,  # Always true in FLUX
        
        # Config
        'field_config': {
            'h': field.h, 'w': field.w, 'd': field.d,
            'features': field.features
        },
        'cse_config': {
            'wave_dims': cse.wave_dims,
            'byte_window': cse.byte_window,
        }
    }
    torch.save(state, path)
    print(f"✓ FLUX model saved: {path}")
    size_mb = Path(path).stat().st_size / 1e6
    print(f"  File size: {size_mb:.1f} MB")

def load_flux(path: str) -> dict:
    """
    Load FLUX state from .flx file.
    Returns dict with all components ready to initialize.
    """
    state = torch.load(path)
    assert state['format'] == 'FLUX', "Not a FLUX model file"
    print(f"✓ FLUX model loaded: {path}")
    print(f"  Version: {state['version']}")
    print(f"  Phase: {state['phase']}")
    print(f"  Learning steps: {state['learning_steps']}")
    return state
```

---

## Test Scripts

### test_phase2_test1.py — Attractor Formation
```python
"""
PHASE 2 TEST 1: Verify that repeated inputs create stable attractors.

Procedure:
1. Create fresh field
2. Feed pattern A 20 times
3. Check attractor_catalog.verify(A) == True
4. Feed pattern B 20 times  
5. Check attractor_catalog.verify(A) == True  ← CRITICAL
6. Check attractor_catalog.verify(B) == True

Pass criteria:
- Both attractors exist after sequential training
- A's feature similarity to original > 0.8 after B training
"""
```

### test_phase2_test2.py — Local Update Verification
```python
"""
PHASE 2 TEST 2: Verify that field updates are truly local.

Procedure:
1. Record field state at 100 random far-from-center locations
2. Apply perturbation at center of field
3. Record field state at same 100 locations
4. Assert: average change at far locations < 0.001

Pass criteria:
- Max change at distance > 20 units from perturbation: < 0.01
- Average change at distance > 20 units: < 0.001
- This proves no global gradient is propagating
"""
```

### test_phase2_test3.py — Field Stability
```python
"""
PHASE 2 TEST 3: Field remains stable over 1000 update steps.

Procedure:
1. Run 1000 random perturbations on the field
2. Track total_energy() at each step
3. Track num_attractors() at each 100 steps

Pass criteria:
- total_energy() never exceeds initial * 10 (no explosion)
- total_energy() never drops below initial * 0.1 (no collapse)
- num_attractors() grows monotonically (never loses attractors)
- Runs in < 120 seconds on GPU
"""
```

---

## Demo Scripts

### demo_phase2_demo1.py — Attractor Formation Live
```
Run: python demo_phase2_demo1.py

Shows:
- Field energy landscape as 2D slice (sum over D axis)
- Updates in real-time as patterns are fed in
- Energy wells forming and deepening with repetition
- New patterns creating new wells without disrupting old ones
- Printed stats: num_attractors, total_energy per step
```

### demo_phase2_demo2.py — The No-Forgetting Demo
```
Run: python demo_phase2_demo2.py

Script:
Step 1: "Training on Pattern A (20 repetitions)..."
        [shows energy landscape — well A forms]
Step 2: "Training on Pattern B (20 repetitions)..."
        [shows energy landscape — well B forms, well A intact]
Step 3: "Training on Pattern C (20 repetitions)..."
        [shows all three wells intact]
Step 4: "Querying for Pattern A..."
        Output: "Found Pattern A (similarity: 0.94)"
Step 5: "CATASTROPHIC FORGETTING SCORE: 0.00"
        "Transformer baseline: ~0.45"
        "FLUX advantage: COMPLETE RETENTION"
```

---

## Acceptance Criteria Checklist

Before marking Phase 2 complete:
- [ ] Phase 1 checkpoint loads without error
- [ ] Field forms stable attractor after 10 repetitions
- [ ] New pattern doesn't destroy old attractor (test2_test1)
- [ ] Updates are local — far regions unaffected (test2_test2)
- [ ] Field stable over 1000 steps (test2_test3)
- [ ] .flx file saves and loads correctly
- [ ] Loaded field state identical to saved state (bit-exact)
- [ ] Both demos run and produce meaningful output
- [ ] Checkpoint saved to checkpoints/phase2.phase.pt
