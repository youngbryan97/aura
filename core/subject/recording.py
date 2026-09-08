"""The matrix everything else is computed from, and the honesty checks on it.

A recording is X in R^{T x D} with a domain map, a condition per row, and the
environment reading that went with each row. The environment matters because
half the battery is about telling apart what the core did from what was done
to it, and a recording that does not carry E_t cannot support that separation.

Two guards live here rather than downstream. A column that never moved has no
scale, so a distance divided by its spread is a division by nothing; those
columns are named and excluded, and the count of them is reported rather than
hidden. And a recording whose rows arrive in a fixed cycle of conditions can
manufacture apparent structure out of the cycle alone, so the condition column
is kept for the nulls to shuffle against.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.subject.state import (
    DOMAINS,
    CoreState,
    domain_slices,
    feature_names,
)

__all__ = ["Recording", "build_recording", "load_recording"]

#: A column whose standard deviation across the whole recording is at or below
#: this is treated as constant. It is not zero because a float that is written
#: and rewritten with the same arithmetic can wander in the last bits, and a
#: 1e-17 spread is not a scale anything should be divided by.
FLAT_EPS: float = 1e-9


@dataclass(frozen=True, slots=True)
class Recording:
    """X, plus what each row was and what the world was doing at the time."""

    x: np.ndarray  # (T, D)
    conditions: tuple[str, ...]
    tags: tuple[str, ...]
    times: np.ndarray  # (T,)
    env: np.ndarray  # (T, E)
    env_names: tuple[str, ...]
    columns: tuple[str, ...]
    slices: dict[str, slice]
    notes: dict[str, Any]

    @property
    def frames(self) -> int:
        return int(self.x.shape[0])

    @property
    def width(self) -> int:
        return int(self.x.shape[1])

    def domain(self, key: str) -> np.ndarray:
        return self.x[:, self.slices[key]]

    def scale(self) -> np.ndarray:
        """Per-column spread, used everywhere a distance needs a unit."""
        return self.x.std(axis=0)

    def live_columns(self) -> np.ndarray:
        return self.scale() > FLAT_EPS

    def flat_columns(self) -> tuple[str, ...]:
        mask = ~self.live_columns()
        return tuple(name for name, dead in zip(self.columns, mask, strict=True) if dead)

    def live_domains(self) -> tuple[str, ...]:
        """Domains with at least one column that moved. The rest are unmeasured."""
        live = self.live_columns()
        return tuple(key for key in DOMAINS if bool(live[self.slices[key]].any()))

    def condition_rows(self, condition: str) -> np.ndarray:
        return np.array(
            [index for index, name in enumerate(self.conditions) if name == condition],
            dtype=np.int64,
        )

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for name in self.conditions:
            counts[name] = counts.get(name, 0) + 1
        return {
            "frames": self.frames,
            "width": self.width,
            "conditions": counts,
            "live_domains": list(self.live_domains()),
            "flat_columns": len(self.flat_columns()),
            "flat_column_names": list(self.flat_columns()),
            "env_names": list(self.env_names),
            "notes": self.notes,
        }

    def save(self, directory: Path) -> Path:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        gateway = get_file_write_gateway()
        with local_internal_governed_scope("subject_core.recording"):
            gateway.ensure_directory(directory, source="subject_core.recording")
        np.savez_compressed(
            directory / "core_state.npz",
            x=self.x,
            times=self.times,
            env=self.env,
        )
        manifest = json.dumps(
            {
                "conditions": list(self.conditions),
                "tags": list(self.tags),
                "columns": list(self.columns),
                "env_names": list(self.env_names),
                "notes": self.notes,
                "summary": self.summary(),
            },
            indent=2,
        )
        with local_internal_governed_scope("subject_core.recording"):
            gateway.write_text(
                directory / "core_state_manifest.json",
                manifest,
                source="subject_core.recording",
            )
        return directory / "core_state.npz"


def build_recording(
    states: Sequence[CoreState],
    *,
    notes: dict[str, Any] | None = None,
) -> Recording:
    if not states:
        raise ValueError("a recording of nothing has no dynamics to measure")
    env_names = tuple(sorted({key for item in states for key in item.env}))
    env = np.array(
        [[float(item.env.get(name, 0.0)) for name in env_names] for item in states],
        dtype=np.float64,
    ).reshape(len(states), len(env_names))
    return Recording(
        x=np.vstack([item.vector() for item in states]),
        conditions=tuple(item.condition for item in states),
        tags=tuple(item.tag for item in states),
        times=np.array([item.t for item in states], dtype=np.float64),
        env=env,
        env_names=env_names,
        columns=feature_names(),
        slices=domain_slices(),
        notes=dict(notes or {}),
    )


def load_recording(directory: Path) -> Recording:
    blob = np.load(directory / "core_state.npz")
    manifest = json.loads((directory / "core_state_manifest.json").read_text())
    return Recording(
        x=blob["x"],
        conditions=tuple(manifest["conditions"]),
        tags=tuple(manifest["tags"]),
        times=blob["times"],
        env=blob["env"],
        env_names=tuple(manifest["env_names"]),
        columns=tuple(manifest["columns"]),
        slices=domain_slices(),
        notes=dict(manifest.get("notes", {})),
    )
