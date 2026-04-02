from __future__ import annotations

try:
    from phases.phase_claw.cost_tracker import CostTracker
except ImportError:
    from .cost_tracker import CostTracker


def apply_cost_hook(tracker: CostTracker, label: str, units: int) -> CostTracker:
    tracker.record(label, units)
    return tracker
