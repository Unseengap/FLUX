"""
ConceptSeeder: Ontological bootstrapping via ConceptNet.

Instead of discovering how the world works from raw text,
we directly seed the SparseResonanceField with the structural
map of human knowledge from ConceptNet.

Each ConceptNet triple (Dog, IsA, Animal) becomes:
  1. Two attractors in the sparse field (dog, animal)
  2. A directional arrow in the ImplicationChainStore
  3. For Antonym triples: negative mass applied → repulsion

Edge weight mapping:
    IsA:         +1.0  (strong taxonomy)
    Causes:      +0.9  (causal physics)
    HasProperty: +0.8  (adjectival geometry)
    PartOf:      +0.75 (compositional)
    UsedFor:     +0.7  (functional)
    CapableOf:   +0.7  (capability)
    AtLocation:  +0.6  (spatial)
    Antonym:     -1.0  (repulsion — negative mass)
"""

import sys
import json
import torch
import torch.nn.functional as F
from torch import Tensor
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1_5'))

from implication import ImplicationChainStore
from causal_types import CausalArrow


EDGE_WEIGHTS = {
    'IsA':         1.0,
    'Causes':      0.9,
    'HasProperty': 0.8,
    'PartOf':      0.75,
    'UsedFor':     0.7,
    'CapableOf':   0.7,
    'AtLocation':  0.6,
    'Antonym':    -1.0,
}

# Bundled minimal ConceptNet triples for offline / fallback use
FALLBACK_TRIPLES = [
    # Taxonomy
    ('dog',       'IsA',         'animal'),
    ('cat',       'IsA',         'animal'),
    ('bird',      'IsA',         'animal'),
    ('rose',      'IsA',         'flower'),
    ('flower',    'IsA',         'plant'),
    ('plant',     'IsA',         'living thing'),
    ('car',       'IsA',         'vehicle'),
    ('truck',     'IsA',         'vehicle'),
    ('apple',     'IsA',         'fruit'),
    ('banana',    'IsA',         'fruit'),
    ('doctor',    'IsA',         'person'),
    ('teacher',   'IsA',         'person'),
    ('paris',     'IsA',         'city'),
    ('france',    'IsA',         'country'),
    # Causal
    ('rain',      'Causes',      'wet ground'),
    ('fire',      'Causes',      'smoke'),
    ('eating',    'Causes',      'fullness'),
    ('exercise',  'Causes',      'fatigue'),
    ('sun',       'Causes',      'warmth'),
    ('study',     'Causes',      'knowledge'),
    # Properties
    ('fire',      'HasProperty', 'hot'),
    ('ice',       'HasProperty', 'cold'),
    ('sky',       'HasProperty', 'blue'),
    ('grass',     'HasProperty', 'green'),
    ('sugar',     'HasProperty', 'sweet'),
    ('lemon',     'HasProperty', 'sour'),
    # Part of
    ('wheel',     'PartOf',      'car'),
    ('branch',    'PartOf',      'tree'),
    ('finger',    'PartOf',      'hand'),
    ('page',      'PartOf',      'book'),
    # Used for
    ('knife',     'UsedFor',     'cutting'),
    ('pen',       'UsedFor',     'writing'),
    ('bed',       'UsedFor',     'sleeping'),
    ('key',       'UsedFor',     'opening locks'),
    # Antonyms
    ('hot',       'Antonym',     'cold'),
    ('big',       'Antonym',     'small'),
    ('happy',     'Antonym',     'sad'),
    ('fast',      'Antonym',     'slow'),
    ('dark',      'Antonym',     'light'),
    ('up',        'Antonym',     'down'),
    ('open',      'Antonym',     'closed'),
    ('alive',     'Antonym',     'dead'),
    ('love',      'Antonym',     'hate'),
    ('truth',     'Antonym',     'lie'),
    # Location
    ('fish',      'AtLocation',  'ocean'),
    ('books',     'AtLocation',  'library'),
    ('food',      'AtLocation',  'kitchen'),
    ('patients',  'AtLocation',  'hospital'),
]


def load_conceptnet_triples(
    path: Optional[str] = None,
    max_triples: int = 50000,
) -> List[Tuple[str, str, str]]:
    """
    Load ConceptNet triples from file or use fallback.

    Args:
        path: path to ConceptNet assertions CSV or JSON, or None for fallback
        max_triples: maximum triples to load
    Returns:
        List of (head, relation, tail) string triples
    """
    if path and Path(path).exists():
        triples = _load_from_file(path, max_triples)
        print(f"  ✓ Loaded {len(triples)} ConceptNet triples from {path}")
        return triples

    # Try downloading a small subset via datasets
    try:
        triples = _download_conceptnet_subset(max_triples)
        print(f"  ✓ Downloaded {len(triples)} ConceptNet triples")
        return triples
    except Exception as e:
        print(f"  ⚠ ConceptNet download failed ({e}), using {len(FALLBACK_TRIPLES)} fallback triples")
        return FALLBACK_TRIPLES


def _load_from_file(path: str, max_triples: int) -> List[Tuple[str,str,str]]:
    """Load triples from a ConceptNet assertions file."""
    triples = []
    p = Path(path)
    with open(p) as f:
        for line in f:
            if len(triples) >= max_triples:
                break
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                relation = parts[1].split('/')[-1]
                head     = parts[2].split('/')[-1].replace('_', ' ')
                tail     = parts[3].split('/')[-1].replace('_', ' ') if len(parts) > 3 else parts[2]
                if relation in EDGE_WEIGHTS:
                    triples.append((head, relation, tail))
    return triples


def _download_conceptnet_subset(max_triples: int) -> List[Tuple[str,str,str]]:
    """Download a subset of ConceptNet via HuggingFace datasets."""
    from datasets import load_dataset
    ds = load_dataset('conceptnet5', 'conceptnet5', split='train', streaming=True)
    triples = []
    for item in ds:
        if len(triples) >= max_triples:
            break
        rel  = item.get('rel', '').split('/')[-1]
        head = item.get('arg1', '').replace('_', ' ').lower()
        tail = item.get('arg2', '').replace('_', ' ').lower()
        if rel in EDGE_WEIGHTS and head and tail:
            triples.append((head, rel, tail))
    return triples


class OntologicalSeeder:
    """
    Seeds the SparseResonanceField with ConceptNet knowledge.

    For each triple:
    - Encodes head and tail through the full CSE → CWC pipeline
    - Perturbs the sparse field to create/reinforce attractors
    - Registers a directional arrow in the ImplicationChainStore
    - For Antonym triples: applies negative mass to repel attractors
    """

    def __init__(
        self,
        cse,
        cwc,
        field: 'SparseResonanceField',
        impl_store: ImplicationChainStore,
        device: str = 'cuda',
    ):
        self.cse        = cse
        self.cwc        = cwc
        self.field      = field
        self.impl_store = impl_store
        self.device     = device

        self.stats = {
            'total_seeded':   0,
            'attractors_created': 0,
            'arrows_registered':  0,
            'antonyms_repelled':  0,
            'growth_events':      0,
            'start_time':         datetime.now().isoformat(),
        }

    def _encode_concept(self, text: str) -> Tensor:
        """Encode a concept through CSE → CWC → mean pool."""
        with torch.no_grad():
            wave   = self.cse.encode(text)
            causal = self.cwc.forward(wave)
        return causal.full.mean(dim=0)

    def seed_triple(
        self,
        head:     str,
        relation: str,
        tail:     str,
    ) -> bool:
        """
        Seed a single ConceptNet triple into the field.

        Returns True if successful.
        """
        weight = EDGE_WEIGHTS.get(relation, 0.0)
        if weight == 0.0:
            return False

        try:
            vec_head = self._encode_concept(head)
            vec_tail = self._encode_concept(tail)

            # Use the 432-dim base wave for field perturbation
            wave_head = self.cse.encode(head).full.mean(dim=0)
            wave_tail = self.cse.encode(tail).full.mean(dim=0)

            if weight > 0:
                # Positive relationship: create/reinforce both attractors
                self.field.perturb(wave_head, strength=abs(weight))
                self.field.perturb(wave_tail, strength=abs(weight))

                # Register implication arrow
                self.impl_store.arrows.append(CausalArrow(
                    source_vector  = vec_head.cpu(),
                    target_vector  = vec_tail.cpu(),
                    strength       = abs(weight),
                    evidence_count = 1,
                    arrow_type     = relation.lower(),
                ))
                self.stats['arrows_registered'] += 1

            else:
                # Antonym: create attractors but apply negative mass
                self.field.perturb(wave_head, strength=0.5)
                self.field.perturb(wave_tail, strength=0.5)

                # Apply negative mass to head location (repels tail)
                h, w, d = self.field.wave_to_field_coords(wave_head)
                current_mass = self.field.registry.get_mass(h, w, d)
                # Drive mass negative → repulsion
                self.field.registry._mass[
                    self.field.registry._key(h, w, d)
                ] = max(-1.0, current_mass - abs(weight) * 0.3)

                self.stats['antonyms_repelled'] += 1

            self.stats['total_seeded']      += 1
            self.stats['attractors_created'] += 2
            return True

        except Exception as e:
            return False

    def seed_batch(
        self,
        triples:      List[Tuple[str,str,str]],
        log_every:    int = 100,
        check_growth: bool = True,
        extra_state:  Dict = None,
    ) -> Dict:
        """
        Seed a batch of triples with progress logging and growth monitoring.

        Args:
            triples:      list of (head, relation, tail)
            log_every:    log capacity status every N triples
            check_growth: whether to check for field growth after each batch
            extra_state:  passed to GrowthManager if growth occurs
        Returns:
            stats dict
        """
        print(f"\n  Seeding {len(triples)} ConceptNet triples...")
        print(f"  Field starting capacity: {self.field.registry.capacity_fraction():.1%}")

        for i, (head, rel, tail) in enumerate(triples):
            self.seed_triple(head, rel, tail)

            if (i + 1) % log_every == 0:
                self.field.growth_manager.log_capacity_status(self.field.registry)

                if check_growth:
                    grew = self.field.check_and_grow(extra_state or {})
                    if grew:
                        self.stats['growth_events'] += 1
                        print(
                            f"  Field grew at triple {i+1} — "
                            f"now at Tier {self.field.growth_manager.current_tier}"
                        )

        # Final growth check
        if check_growth:
            grew = self.field.check_and_grow(extra_state or {})
            if grew:
                self.stats['growth_events'] += 1

        self.stats['end_time'] = datetime.now().isoformat()
        print(f"\n  Seeding complete:")
        print(f"    Triples seeded:    {self.stats['total_seeded']}")
        print(f"    Arrows registered: {self.stats['arrows_registered']}")
        print(f"    Antonyms repelled: {self.stats['antonyms_repelled']}")
        print(f"    Growth events:     {self.stats['growth_events']}")
        print(f"    Active locations:  {self.field.registry.active_count():,}")

        return self.stats

    def build_hierarchical_abstraction(
        self,
        child:  str,
        parent: str,
    ):
        """
        Force the child concept to nest physically within the
        gravitational radius of the parent in the sparse field.

        We achieve this by encoding both, computing the child's
        target location as a small perturbation of the parent's
        location, and writing it directly.
        """
        with torch.no_grad():
            wave_parent = self.cse.encode(parent).full.mean(dim=0)
            wave_child  = self.cse.encode(child).full.mean(dim=0)

        ph, pw, pd = self.field.wave_to_field_coords(wave_parent)

        # Place child in a small radius around parent
        offset = 2
        ch = min(self.field.h - 1, ph + offset)
        cw = min(self.field.w - 1, pw + offset)
        cd = min(self.field.d - 1, pd + offset)

        with torch.no_grad():
            child_feature = self.field.wave_to_feature(wave_child)

        self.field.registry.set(
            ch, cw, cd,
            child_feature,
            mass   = 0.3,
            energy = 0.7,
        )
