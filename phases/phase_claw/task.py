from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PortingTask:
    """A task in the porting backlog."""
    name: str
    description: str


__all__ = ['PortingTask']
