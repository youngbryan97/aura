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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.subject.state import (
    DOMAINS,
    CoreState,
    domain_slices,
    feature_names,
)

__all__ = ["Recording", "build_recording", "load_recording", "slices_from_columns"]

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
    #: How many frames failed to read each source, and why. A column whose
    #: source is in here is a default rather than a measurement for that many
    #: frames, and a criterion resting on it is reporting on the harness.
    misses: dict[str, dict[str, Any]] = field(default_factory=dict)

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

    def turn_rows(self, marker: str = "ontogeny") -> np.ndarray:
        """One row per turn: the last frame of each cognitive cycle.

        A frame-to-frame step inside a turn is not a transition of the system,
        it is one line of the transition function — the phases run in a fixed
        order and most domains do not move at all between two of them. Every
        measure that fits K_{t+1} from K_t belongs on this series, and the
        frame series belongs to the perturbation measures, which are asking
        about propagation inside a cycle rather than about the law between
        cycles.
        """
        return np.array(
            [index for index, tag in enumerate(self.tags) if tag == marker],
            dtype=np.int64,
        )

    def by_turn(self, marker: str = "ontogeny") -> Recording:
        """The same recording sampled once per turn."""
        rows = self.turn_rows(marker)
        if rows.size == 0:
            return self
        return Recording(
            x=self.x[rows],
            conditions=tuple(self.conditions[index] for index in rows),
            tags=tuple(self.tags[index] for index in rows),
            times=self.times[rows],
            env=self.env[rows],
            env_names=self.env_names,
            columns=self.columns,
            slices=self.slices,
            notes={**self.notes, "sampled": "one frame per turn"},
            misses=self.misses,
        )

    def monotone_columns(self, tolerance: float = 0.99) -> np.ndarray:
        """Columns that only ever go one way. Clocks, not state.

        A running total changes what the system computes — a version counter
        gates a phase every twentieth turn — so a counter is state by the
        definition the schema opens with. It is also a trend, and a trend in
        the inputs lets a fitted model extrapolate elapsed time across a
        contiguous train/test split and then fail on rows beyond the range it
        saw.

        Reported rather than removed. Differencing them was tried and made the
        partition score worse, not better, so the fix that worked was in what
        the measures predict rather than in what the recording holds: the
        transition measures target the change, where a level's trend cancels.
        The list stays in the summary because a run whose count of these jumps
        has grown a clock somewhere, and that is worth seeing.
        """
        if self.frames < 8:
            return np.zeros(self.width, dtype=bool)
        steps = np.diff(self.x, axis=0)
        moving = np.abs(steps) > FLAT_EPS
        counts = moving.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            up = np.where(counts > 0, (steps > FLAT_EPS).sum(axis=0) / np.maximum(counts, 1), 0.0)
            down = np.where(counts > 0, (steps < -FLAT_EPS).sum(axis=0) / np.maximum(counts, 1), 0.0)
        one_way = (up >= tolerance) | (down >= tolerance)
        return one_way & (counts >= 4)

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
            "one_way_columns": [
                name
                for name, flag in zip(self.columns, self.monotone_columns(), strict=True)
                if flag
            ],
            "flat_column_names": list(self.flat_columns()),
            "env_names": list(self.env_names),
            # The validity matrix: what could not be read, how often, and why.
            # A run where this is not empty for a required organ is a run that
            # measured the harness, and the verdict says so rather than letting
            # a default stand in for a reading.
            "misses": self.misses,
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


def _missingness(states: Sequence[CoreState]) -> dict[str, dict[str, Any]]:
    """One row per source that ever failed to read, over the whole recording.

    A single missed read is a moment; the same source missing on every frame is
    a subsystem that was never running, and the two cannot be told apart from
    the values, which are 0.0 either way.
    """
    total = len(states)
    counts: dict[str, dict[str, Any]] = {}
    for item in states:
        for source, reason in (item.misses or {}).items():
            row = counts.setdefault(source, {"frames": 0, "reasons": {}})
            row["frames"] += 1
            row["reasons"][reason] = row["reasons"].get(reason, 0) + 1
    for row in counts.values():
        row["share"] = round(row["frames"] / max(1, total), 4)
    return dict(sorted(counts.items(), key=lambda pair: -pair[1]["frames"]))


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
        misses=_missingness(states),
    )


def slices_from_columns(columns: Sequence[str]) -> dict[str, slice]:
    """Where each domain sits, read from the column names that were recorded.

    Not from the current schema. A recording saved before a feature was added
    or removed has its own widths, and rebuilding the slices from today's
    schema silently reads the wrong columns for every domain after the one that
    changed — or indexes past the end, which is the lucky case because it says
    so.
    """
    out: dict[str, slice] = {}
    start = 0
    for key in DOMAINS:
        width = sum(1 for name in columns if name.startswith(f"{key}."))
        out[key] = slice(start, start + width)
        start += width
    return out


def load_recording(directory: Path) -> Recording:
    blob = np.load(directory / "core_state.npz")
    manifest = json.loads((directory / "core_state_manifest.json").read_text())
    columns = tuple(manifest["columns"])
    return Recording(
        x=blob["x"],
        conditions=tuple(manifest["conditions"]),
        tags=tuple(manifest["tags"]),
        times=blob["times"],
        env=blob["env"],
        env_names=tuple(manifest["env_names"]),
        columns=columns,
        slices=slices_from_columns(columns),
        notes=dict(manifest.get("notes", {})),
    )
