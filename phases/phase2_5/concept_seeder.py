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

Fix v1.1:
    The HuggingFace conceptnet5 dataset schema (confirmed from dataset viewer):
        rel:  '/r/IsA'          — full URI, split('/')[-1] gives 'IsA' ✓
        arg1: '/c/en/dog/n'     — URI where parts[2] is the concept, NOT parts[-1]
                                  parts[-1] is the POS tag ('n','v','a') ✗
        arg2: '/c/en/animal'    — same
        lang: 'en'              — dedicated field for language filtering

    Two bugs fixed:
    1. No lang=='en' filter. Dataset is sorted alphabetically by language code
       (ab, adx, ae, af, ..., en). Without filtering, max_triples is exhausted
       on non-English Antonym rows before any English IsA/Causes rows appear.
       This caused arrows_registered=0 in all previous runs.

    2. arg1/arg2 parsed with split('/')[-1] which returns the POS tag ('n')
       for URIs like /c/en/dog/n instead of the concept word 'dog'.
       Fixed: take parts[2] for /c/lang/concept[/pos] URIs.
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
    ('dog',        'IsA',         'animal'),
    ('cat',        'IsA',         'animal'),
    ('bird',       'IsA',         'animal'),
    ('rose',       'IsA',         'flower'),
    ('flower',     'IsA',         'plant'),
    ('plant',      'IsA',         'living thing'),
    ('car',        'IsA',         'vehicle'),
    ('truck',      'IsA',         'vehicle'),
    ('apple',      'IsA',         'fruit'),
    ('banana',     'IsA',         'fruit'),
    ('doctor',     'IsA',         'person'),
    ('teacher',    'IsA',         'person'),
    ('paris',      'IsA',         'city'),
    ('france',     'IsA',         'country'),
    ('dog',        'IsA',         'mammal'),
    ('cat',        'IsA',         'mammal'),
    ('whale',      'IsA',         'mammal'),
    ('salmon',     'IsA',         'fish'),
    ('eagle',      'IsA',         'bird'),
    ('oak',        'IsA',         'tree'),
    ('table',      'IsA',         'furniture'),
    ('chair',      'IsA',         'furniture'),
    ('bus',        'IsA',         'vehicle'),
    ('train',      'IsA',         'vehicle'),
    ('red',        'IsA',         'color'),
    ('monday',     'IsA',         'day'),
    # Causal
    ('rain',       'Causes',      'wet ground'),
    ('fire',       'Causes',      'smoke'),
    ('eating',     'Causes',      'fullness'),
    ('exercise',   'Causes',      'fatigue'),
    ('sun',        'Causes',      'warmth'),
    ('study',      'Causes',      'knowledge'),
    ('reading',    'Causes',      'knowledge'),
    ('infection',  'Causes',      'fever'),
    ('drought',    'Causes',      'famine'),
    ('practice',   'Causes',      'improvement'),
    ('wind',       'Causes',      'waves'),
    ('gravity',    'Causes',      'falling'),
    # Properties
    ('fire',       'HasProperty', 'hot'),
    ('ice',        'HasProperty', 'cold'),
    ('sky',        'HasProperty', 'blue'),
    ('grass',      'HasProperty', 'green'),
    ('sugar',      'HasProperty', 'sweet'),
    ('lemon',      'HasProperty', 'sour'),
    ('rock',       'HasProperty', 'hard'),
    ('cotton',     'HasProperty', 'soft'),
    ('gold',       'HasProperty', 'shiny'),
    ('ocean',      'HasProperty', 'vast'),
    ('diamond',    'HasProperty', 'hard'),
    # Part of
    ('wheel',      'PartOf',      'car'),
    ('branch',     'PartOf',      'tree'),
    ('finger',     'PartOf',      'hand'),
    ('page',       'PartOf',      'book'),
    ('engine',     'PartOf',      'car'),
    ('roof',       'PartOf',      'house'),
    ('chapter',    'PartOf',      'book'),
    ('petal',      'PartOf',      'flower'),
    ('wing',       'PartOf',      'bird'),
    ('nucleus',    'PartOf',      'cell'),
    # Used for
    ('knife',      'UsedFor',     'cutting'),
    ('pen',        'UsedFor',     'writing'),
    ('bed',        'UsedFor',     'sleeping'),
    ('key',        'UsedFor',     'opening locks'),
    ('hammer',     'UsedFor',     'hitting nails'),
    ('glasses',    'UsedFor',     'seeing clearly'),
    ('umbrella',   'UsedFor',     'blocking rain'),
    ('telescope',  'UsedFor',     'viewing distant objects'),
    ('microscope', 'UsedFor',     'viewing tiny objects'),
    ('stove',      'UsedFor',     'cooking food'),
    # Capable of
    ('dog',        'CapableOf',   'barking'),
    ('bird',       'CapableOf',   'flying'),
    ('fish',       'CapableOf',   'swimming'),
    ('human',      'CapableOf',   'thinking'),
    ('computer',   'CapableOf',   'calculating'),
    ('plant',      'CapableOf',   'photosynthesis'),
    ('fire',       'CapableOf',   'burning'),
    # Antonyms
    ('hot',        'Antonym',     'cold'),
    ('big',        'Antonym',     'small'),
    ('happy',      'Antonym',     'sad'),
    ('fast',       'Antonym',     'slow'),
    ('dark',       'Antonym',     'light'),
    ('up',         'Antonym',     'down'),
    ('open',       'Antonym',     'closed'),
    ('alive',      'Antonym',     'dead'),
    ('love',       'Antonym',     'hate'),
    ('truth',      'Antonym',     'lie'),
    ('hard',       'Antonym',     'soft'),
    ('rich',       'Antonym',     'poor'),
    ('old',        'Antonym',     'young'),
    ('strong',     'Antonym',     'weak'),
    ('empty',      'Antonym',     'full'),
    # Location
    ('fish',       'AtLocation',  'ocean'),
    ('books',      'AtLocation',  'library'),
    ('food',       'AtLocation',  'kitchen'),
    ('patients',   'AtLocation',  'hospital'),
    ('students',   'AtLocation',  'school'),
    ('athletes',   'AtLocation',  'stadium'),
    ('penguins',   'AtLocation',  'antarctica'),
]


def _parse_concept_uri(uri: str) -> str:
    """
    Parse a ConceptNet concept URI to a clean word/phrase.

    URI formats seen in the dataset:
        /c/en/dog/n        → 'dog'    (take parts[2], not parts[-1]='n')
        /c/en/hot_dog/n    → 'hot dog'
        /c/en/be_on_fire   → 'be on fire'
        /r/IsA             → 'IsA'   (relation URI)
        dog                → 'dog'   (already clean)
    """
    if not uri:
        return ''
    parts = uri.strip('/').split('/')
    # Concept URI: /c/lang/concept[/pos[/sense]]
    if len(parts) >= 3 and parts[0] == 'c':
        return parts[2].replace('_', ' ').lower()
    # Relation URI: /r/IsA
    if len(parts) >= 2 and parts[0] == 'r':
        return parts[-1]
    # Already clean string
    return uri.replace('_', ' ').lower()


def load_conceptnet_triples(
    path: Optional[str] = None,
    max_triples: int = 50000,
) -> List[Tuple[str, str, str]]:
    """
    Load ConceptNet triples from file or download, English-only.

    Args:
        path: path to ConceptNet assertions TSV file, or None to download
        max_triples: maximum English triples to collect
    Returns:
        List of (head, relation, tail) clean string triples
    """
    if path and Path(path).exists():
        triples = _load_from_file(path, max_triples)
        print(f"  ✓ Loaded {len(triples)} ConceptNet triples from {path}")
        return triples

    try:
        triples = _download_conceptnet_subset(max_triples)
        print(f"  ✓ Downloaded {len(triples)} ConceptNet triples")
        return triples
    except Exception as e:
        print(f"  ⚠ ConceptNet download failed ({e}), using {len(FALLBACK_TRIPLES)} fallback triples")
        return FALLBACK_TRIPLES


def _load_from_file(path: str, max_triples: int) -> List[Tuple[str, str, str]]:
    """
    Load from a ConceptNet assertions TSV file.
    Columns: URI  full_rel  arg1_uri  arg2_uri  weight  ...
    Filters to English-only by checking /en/ in arg1 URI.
    """
    triples = []
    with open(path) as f:
        for line in f:
            if len(triples) >= max_triples:
                break
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            # Filter English by URI
            if '/en/' not in parts[2]:
                continue
            rel  = _parse_concept_uri(parts[1])
            head = _parse_concept_uri(parts[2])
            tail = _parse_concept_uri(parts[3])
            if rel in EDGE_WEIGHTS and head and tail:
                triples.append((head, rel, tail))
    return triples


def _download_conceptnet_subset(max_triples: int) -> List[Tuple[str, str, str]]:
    """
    Download English ConceptNet triples via HuggingFace datasets.

    Key fixes vs original:
    1. Filter on lang == 'en' FIRST — the dataset is sorted alphabetically
       by language so non-English rows dominate the start of the stream.
    2. Parse arg1/arg2 with _parse_concept_uri which takes parts[2]
       (the concept slot) rather than parts[-1] (the POS tag).
    """
    from datasets import load_dataset

    ds = load_dataset('conceptnet5', 'conceptnet5', split='train', streaming=True)
    triples      = []
    skipped_lang = 0
    skipped_rel  = 0

    for item in ds:
        if len(triples) >= max_triples:
            break

        # ── English-only filter using the dedicated 'lang' field ──
        lang = item.get('lang', '')
        if lang != 'en':
            skipped_lang += 1
            continue

        # ── Relation: /r/IsA → IsA ──
        rel = _parse_concept_uri(item.get('rel', ''))
        if rel not in EDGE_WEIGHTS:
            skipped_rel += 1
            continue

        # ── Concepts: /c/en/dog/n → dog ──
        head = _parse_concept_uri(item.get('arg1', ''))
        tail = _parse_concept_uri(item.get('arg2', ''))

        if head and tail:
            triples.append((head, rel, tail))

        # Progress every 500k items scanned
        total_seen = len(triples) + skipped_lang + skipped_rel
        if total_seen % 500_000 == 0 and total_seen > 0:
            print(
                f"    Scanned {total_seen:,} → "
                f"{len(triples):,} EN triples, "
                f"{skipped_lang:,} non-EN skipped"
            )

    print(
        f"  Scan complete: {len(triples):,} EN triples, "
        f"{skipped_lang:,} non-EN rows skipped"
    )
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
            'total_seeded':       0,
            'attractors_created': 0,
            'arrows_registered':  0,
            'antonyms_repelled':  0,
            'growth_events':      0,
            'start_time':         datetime.now().isoformat(),
        }

    def _encode_concept(self, text: str) -> Tensor:
        """Encode a concept through CSE → CWC → mean pool → [608]."""
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
            with torch.no_grad():
                wave_head = self.cse.encode(head).full.mean(dim=0)
                wave_tail = self.cse.encode(tail).full.mean(dim=0)

            if weight > 0:
                # Positive relationship: create/reinforce both attractors
                self.field.perturb(wave_head, strength=abs(weight))
                self.field.perturb(wave_tail, strength=abs(weight))

                # Register implication arrow head → tail
                self.impl_store.arrows.append(CausalArrow(
                    source_vector  = vec_head.cpu(),
                    target_vector  = vec_tail.cpu(),
                    strength       = abs(weight),
                    evidence_count = 1,
                    arrow_type     = relation.lower(),
                ))
                self.stats['arrows_registered'] += 1

            else:
                # Antonym: create attractors but drive mass negative
                self.field.perturb(wave_head, strength=0.5)
                self.field.perturb(wave_tail, strength=0.5)

                h, w, d = self.field.wave_to_field_coords(wave_head)
                current_mass = self.field.registry.get_mass(h, w, d)
                self.field.registry._mass[
                    self.field.registry._key(h, w, d)
                ] = max(-1.0, current_mass - abs(weight) * 0.3)

                self.stats['antonyms_repelled'] += 1

            self.stats['total_seeded']       += 1
            self.stats['attractors_created'] += 2
            return True

        except Exception:
            return False

    def seed_batch(
        self,
        triples:      List[Tuple[str, str, str]],
        log_every:    int = 100,
        check_growth: bool = True,
        extra_state:  Dict = None,
    ) -> Dict:
        """
        Seed a batch of triples with progress logging and growth monitoring.
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
        """
        with torch.no_grad():
            wave_parent = self.cse.encode(parent).full.mean(dim=0)
            wave_child  = self.cse.encode(child).full.mean(dim=0)

        ph, pw, pd = self.field.wave_to_field_coords(wave_parent)

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