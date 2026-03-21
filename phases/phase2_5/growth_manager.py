"""
GrowthManager: Dynamic field tier management.

Handles the full lifecycle of field growth:
    - Detects available VRAM
    - Decides when to grow (capacity threshold)
    - Migrates the FieldRegistry to new dimensions
    - Saves a tier checkpoint with full metadata
    - Logs every growth event for replay at inference time

Growth tiers:
    Tier 0:  64³   (~0.5 GB dense equivalent)  — Phase 2 starting point
    Tier 1:  96³   (~1.8 GB dense equivalent)
    Tier 2:  128³  (~4.3 GB dense equivalent)
    Tier 3:  160³  (~8.4 GB dense equivalent)
    Tier 4:  192³  (~17  GB dense equivalent)
    Tier 5:  256³  (~34  GB dense equivalent)  — max

Sparse memory usage is a fraction of the dense equivalent —
only allocated locations consume VRAM.

This architecture supports inference-time real-time learning:
when the field fills up during deployment, it grows automatically,
and the growth history becomes a log of what the model learned and when.
"""

import torch
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from field_registry import FieldRegistry


# ─────────────────────────────────────────────
# Tier Definitions
# ─────────────────────────────────────────────

GROWTH_TIERS = [
    # (tier_id, h,   w,   d,   label)
    (0,  64,  64,  64,  'Phase2-inherited'),
    (1,  96,  96,  96,  'Tier1-expanded'),
    (2,  128, 128, 128, 'Tier2-medium'),
    (3,  160, 160, 160, 'Tier3-large'),
    (4,  192, 192, 192, 'Tier4-xlarge'),
    (5,  256, 256, 256, 'Tier5-maximum'),
]

# Trigger growth when active locations exceed this fraction of
# the MEANINGFUL capacity (we use 60% of theoretical max as the
# point where retrieval quality starts degrading)
GROWTH_TRIGGER_FRACTION = 0.60

# Minimum VRAM headroom to attempt growth (MB)
MIN_VRAM_HEADROOM_MB = 1500

# Feature dim (matches Phase 2)
FIELD_FEATURES = 512


class GrowthEvent:
    """Record of a single growth transition."""
    def __init__(
        self,
        from_tier:   int,
        to_tier:     int,
        from_dims:   Tuple[int,int,int],
        to_dims:     Tuple[int,int,int],
        active_locs: int,
        cap_frac:    float,
        vram_used:   float,
        vram_avail:  float,
        scale:       float,
        ckpt_path:   str,
        timestamp:   str,
    ):
        self.from_tier   = from_tier
        self.to_tier     = to_tier
        self.from_dims   = from_dims
        self.to_dims     = to_dims
        self.active_locs = active_locs
        self.cap_frac    = cap_frac
        self.vram_used   = vram_used
        self.vram_avail  = vram_avail
        self.scale       = scale
        self.ckpt_path   = ckpt_path
        self.timestamp   = timestamp

    def as_dict(self) -> Dict:
        return {
            'from_tier':   self.from_tier,
            'to_tier':     self.to_tier,
            'from_dims':   self.from_dims,
            'to_dims':     self.to_dims,
            'active_locs': self.active_locs,
            'cap_frac':    self.cap_frac,
            'vram_used':   self.vram_used,
            'vram_avail':  self.vram_avail,
            'scale':       self.scale,
            'ckpt_path':   self.ckpt_path,
            'timestamp':   self.timestamp,
        }

    def log_str(self) -> str:
        return (
            f"\n{'='*60}\n"
            f"[GROWTH EVENT] Tier {self.from_tier} → Tier {self.to_tier}\n"
            f"  Previous dimensions:  {self.from_dims[0]} × {self.from_dims[1]} × {self.from_dims[2]}\n"
            f"  New dimensions:       {self.to_dims[0]} × {self.to_dims[1]} × {self.to_dims[2]}\n"
            f"  Active locations:     {self.active_locs:,} ({self.cap_frac:.1%})\n"
            f"  Scale factor:         {self.scale:.2f}×\n"
            f"  VRAM used:            {self.vram_used:.0f} MB / {self.vram_used + self.vram_avail:.0f} MB total\n"
            f"  VRAM available:       {self.vram_avail:.0f} MB\n"
            f"  Checkpoint:           {self.ckpt_path}\n"
            f"  Timestamp:            {self.timestamp}\n"
            f"{'='*60}"
        )


class GrowthManager:
    """
    Manages dynamic growth of the SparseResonanceField.

    Usage:
        gm = GrowthManager(checkpoint_dir='checkpoints/', device='cuda')
        gm.attach(sparse_field)

        # During training, call after each batch:
        if gm.should_grow():
            gm.grow(sparse_field, extra_state)
    """

    def __init__(
        self,
        checkpoint_dir:        str   = 'checkpoints/',
        device:                str   = 'cuda',
        growth_trigger:        float = GROWTH_TRIGGER_FRACTION,
        min_vram_headroom_mb:  float = MIN_VRAM_HEADROOM_MB,
        max_tier:              int   = 5,
    ):
        self.checkpoint_dir       = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device               = device
        self.growth_trigger       = growth_trigger
        self.min_vram_headroom_mb = min_vram_headroom_mb
        self.max_tier             = max_tier

        self.current_tier  = 0
        self.growth_events: List[GrowthEvent] = []

    # ─────────────────────────────────────────
    # VRAM detection
    # ─────────────────────────────────────────

    def get_vram_info(self) -> Tuple[float, float]:
        """
        Returns (used_mb, available_mb).
        Falls back to (0, 999999) on CPU so growth always proceeds.
        """
        if not torch.cuda.is_available():
            return 0.0, 999_999.0
        try:
            free, total = torch.cuda.mem_get_info()
            used_mb  = (total - free) / 1e6
            free_mb  = free / 1e6
            return used_mb, free_mb
        except Exception:
            return 0.0, 999_999.0

    def estimate_tier_memory_mb(self, tier_id: int, active_locs: int) -> float:
        """
        Estimate memory needed for a tier given current active locations.
        Sparse: only active locations cost memory.
        We project that active_locs * 2.5 will be used at the new tier
        (growth headroom).
        """
        projected_locs = int(active_locs * 2.5)
        return projected_locs * FIELD_FEATURES * 4 / 1e6  # float32

    # ─────────────────────────────────────────
    # Growth decision
    # ─────────────────────────────────────────

    def should_grow(self, registry: FieldRegistry) -> bool:
        """
        Returns True if the field should attempt to grow.

        Conditions:
        1. Not already at max tier
        2. Capacity fraction exceeds trigger threshold
        3. Enough VRAM for next tier
        """
        if self.current_tier >= self.max_tier:
            return False

        cap = registry.capacity_fraction()
        if cap < self.growth_trigger:
            return False

        # Check VRAM headroom for next tier
        next_tier_id = self.current_tier + 1
        _, avail_mb = self.get_vram_info()
        needed_mb   = self.estimate_tier_memory_mb(next_tier_id, registry.active_count())

        if avail_mb < needed_mb + self.min_vram_headroom_mb:
            print(
                f"  ⚠ Growth trigger hit but VRAM insufficient: "
                f"need {needed_mb:.0f} MB + {self.min_vram_headroom_mb:.0f} MB headroom, "
                f"have {avail_mb:.0f} MB free. Staying at Tier {self.current_tier}."
            )
            return False

        return True

    # ─────────────────────────────────────────
    # Growth execution
    # ─────────────────────────────────────────

    def grow(
        self,
        registry:    FieldRegistry,
        extra_state: Dict[str, Any],
    ) -> Tuple[FieldRegistry, GrowthEvent]:
        """
        Execute a tier growth:
        1. Migrate registry to next tier dimensions
        2. Save tier checkpoint with full metadata
        3. Log growth event
        4. Return new registry and event

        Args:
            registry:    current FieldRegistry
            extra_state: additional state to include in checkpoint
                         (e.g. spatial_projection weights, cse state)
        Returns:
            (new_registry, growth_event)
        """
        from_tier_id = self.current_tier
        to_tier_id   = self.current_tier + 1

        _, _, _, _, from_label = GROWTH_TIERS[from_tier_id]
        to_tid, new_h, new_w, new_d, to_label = GROWTH_TIERS[to_tier_id]

        from_dims = (registry.max_h, registry.max_w, registry.max_d)
        to_dims   = (new_h, new_w, new_d)

        scale = new_h / registry.max_h

        used_mb, avail_mb = self.get_vram_info()
        active_locs = registry.active_count()
        cap_frac    = registry.capacity_fraction()

        print(f"\n  Growing field: {from_label} → {to_label}")
        print(f"    Capacity trigger: {cap_frac:.1%} of {from_dims}")
        print(f"    VRAM: {used_mb:.0f} MB used, {avail_mb:.0f} MB free")

        t0 = time.time()

        # Migrate registry
        new_registry = registry.resize(new_h, new_w, new_d)
        self.current_tier = to_tier_id

        elapsed = time.time() - t0

        # Save tier checkpoint
        ckpt_name = f"phase2_5_tier{to_tier_id}_{new_h}cube.pt"
        ckpt_path = self.checkpoint_dir / ckpt_name

        tier_checkpoint = {
            'format':       'FLUX_SPARSE',
            'version':      '0.1',
            'phase':        '2.5',
            'tier':         to_tier_id,
            'tier_label':   to_label,
            'dimensions':   to_dims,
            'active_locs':  new_registry.active_count(),
            'cap_frac':     new_registry.capacity_fraction(),
            'vram_used_mb': used_mb,
            'vram_avail_mb':avail_mb,
            'scale_factor': scale,
            'migration_seconds': elapsed,
            'can_continue_learning': True,
            'timestamp':    datetime.now().isoformat(),
            'registry':     new_registry.serialize(),
            'growth_history': [e.as_dict() for e in self.growth_events],
        }
        tier_checkpoint.update(extra_state)

        torch.save(tier_checkpoint, str(ckpt_path))
        size_mb = ckpt_path.stat().st_size / 1e6

        event = GrowthEvent(
            from_tier   = from_tier_id,
            to_tier     = to_tier_id,
            from_dims   = from_dims,
            to_dims     = to_dims,
            active_locs = active_locs,
            cap_frac    = cap_frac,
            vram_used   = used_mb,
            vram_avail  = avail_mb,
            scale       = scale,
            ckpt_path   = str(ckpt_path),
            timestamp   = datetime.now().isoformat(),
        )
        self.growth_events.append(event)

        print(event.log_str())
        print(f"  ✓ Tier checkpoint saved: {ckpt_path} ({size_mb:.1f} MB)")
        print(f"  ✓ Migration took {elapsed:.1f}s")

        return new_registry, event

    # ─────────────────────────────────────────
    # Status logging
    # ─────────────────────────────────────────

    def log_capacity_status(self, registry: FieldRegistry):
        """Print a clean capacity status line."""
        tier_id, h, w, d, label = GROWTH_TIERS[self.current_tier]
        used_mb, avail_mb = self.get_vram_info()
        cap = registry.capacity_fraction()
        active = registry.active_count()
        theoretical = registry.theoretical_max()
        mem = registry.memory_mb()

        print(
            f"  [{label}] "
            f"Active: {active:,}/{theoretical:,} ({cap:.1%}) | "
            f"Sparse mem: {mem:.1f} MB | "
            f"VRAM: {used_mb:.0f}/{used_mb+avail_mb:.0f} MB"
        )

    def full_status(self, registry: FieldRegistry) -> Dict[str, Any]:
        """Return complete status dict for logging/checkpointing."""
        tier_id, h, w, d, label = GROWTH_TIERS[self.current_tier]
        used_mb, avail_mb = self.get_vram_info()
        return {
            'current_tier':      self.current_tier,
            'tier_label':        label,
            'dimensions':        (h, w, d),
            'active_locations':  registry.active_count(),
            'theoretical_max':   registry.theoretical_max(),
            'capacity_fraction': registry.capacity_fraction(),
            'sparse_memory_mb':  registry.memory_mb(),
            'vram_used_mb':      used_mb,
            'vram_available_mb': avail_mb,
            'growth_events':     len(self.growth_events),
            'timestamp':         datetime.now().isoformat(),
        }

    # ─────────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────────

    def save_state(self) -> Dict[str, Any]:
        return {
            'current_tier':  self.current_tier,
            'growth_trigger': self.growth_trigger,
            'min_vram_headroom_mb': self.min_vram_headroom_mb,
            'max_tier':      self.max_tier,
            'growth_events': [e.as_dict() for e in self.growth_events],
        }

    def load_state(self, state: Dict[str, Any]):
        self.current_tier          = state['current_tier']
        self.growth_trigger        = state.get('growth_trigger', GROWTH_TRIGGER_FRACTION)
        self.min_vram_headroom_mb  = state.get('min_vram_headroom_mb', MIN_VRAM_HEADROOM_MB)
        self.max_tier              = state.get('max_tier', 5)
        self.growth_events = [
            GrowthEvent(**e) for e in state.get('growth_events', [])
        ]
