"""Environment snapshots where rollback is legitimate."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnvironmentSnapshot:
    snapshot_id: str
    environment_id: str
    run_id: str
    sequence_id: int
    adapter_state_ref: str | None
    belief_state_ref: str
    trace_ref: str
    reversible: bool


__all__ = ["EnvironmentSnapshot"]
