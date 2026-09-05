"""core/consciousness/inner_light/activity.py — the real activity source.

To run the discriminators on Aura's *actual* mind, we need a spatiotemporal
activity matrix (channels × time) drawn from real subsystem activity. The live
source is the ``ConsequenceBus`` stream: every consequential action publishes its
source subsystem, so binning the recent stream over time yields, for each
subsystem, an activity trace — exactly the (region × time) matrix the measures
expect, at the granularity of the organism's own causal events.

This is honest about thinness: with too few events, channels, or bins, the
matrix cannot support the measures, and the battery is told so rather than
fabricating a signal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


@dataclass
class ActivitySample:
    matrix: np.ndarray
    channels: list[str] = field(default_factory=list)
    n_events: int = 0
    timespan_s: float = 0.0
    sufficient: bool = False
    reason: str = ""

    def meta(self) -> dict[str, Any]:
        return {
            "n_channels": int(self.matrix.shape[0]) if self.matrix.size else 0,
            "n_timesteps": int(self.matrix.shape[1]) if self.matrix.size else 0,
            "n_events": self.n_events,
            "timespan_s": round(self.timespan_s, 3),
            "channels": list(self.channels),
            "sufficient": self.sufficient,
            "reason": self.reason,
        }


def build_activity_matrix(
    events: Iterable[tuple],
    *,
    n_bins: int = 64,
    min_channels: int = 3,
    min_events: int = 40,
    min_active_bins: int = 8,
    smooth: bool = True,
) -> ActivitySample:
    """Bin ``(timestamp, source[, magnitude])`` events into a channel×time matrix."""
    evs = []
    for e in events:
        ts = float(e[0])
        src = str(e[1])
        mag = float(e[2]) if len(e) > 2 and e[2] is not None else 1.0
        evs.append((ts, src, mag))
    if not evs:
        return ActivitySample(np.zeros((0, 0)), reason="no events")

    channels = sorted({src for _, src, _ in evs})
    ch_index = {c: i for i, c in enumerate(channels)}
    t0 = min(ts for ts, _, _ in evs)
    t1 = max(ts for ts, _, _ in evs)
    timespan = t1 - t0
    n_events = len(evs)

    if len(channels) < min_channels:
        return ActivitySample(np.zeros((0, 0)), channels=channels, n_events=n_events,
                              timespan_s=timespan, reason=f"too few channels ({len(channels)}<{min_channels})")
    if n_events < min_events:
        return ActivitySample(np.zeros((0, 0)), channels=channels, n_events=n_events,
                              timespan_s=timespan, reason=f"too few events ({n_events}<{min_events})")
    if timespan <= 0:
        return ActivitySample(np.zeros((0, 0)), channels=channels, n_events=n_events,
                              timespan_s=timespan, reason="zero timespan")

    M = np.zeros((len(channels), n_bins))
    for ts, src, mag in evs:
        b = int((ts - t0) / timespan * (n_bins - 1))
        b = min(max(b, 0), n_bins - 1)
        M[ch_index[src], b] += mag

    active_bins = int(np.count_nonzero(M.sum(axis=0)))
    if active_bins < min_active_bins:
        return ActivitySample(M, channels=channels, n_events=n_events, timespan_s=timespan,
                              reason=f"too few active time bins ({active_bins}<{min_active_bins})")

    if smooth and n_bins >= 8:
        kernel = np.array([0.25, 0.5, 0.25])
        M = np.stack([np.convolve(row, kernel, mode="same") for row in M])

    return ActivitySample(M, channels=channels, n_events=n_events, timespan_s=timespan,
                          sufficient=True, reason="ok")


def from_consequence_bus(bus: Any = None, *, n_bins: int = 64, window: int = 500) -> ActivitySample:
    """Build an activity sample from the live ConsequenceBus history."""
    if bus is None:
        from core.runtime.consequence_bus import ConsequenceBus
        bus = ConsequenceBus.get()
    try:
        events = [(float(e.timestamp), str(e.source)) for e in bus.recent_events(window)]
    except (AttributeError, TypeError, ValueError):
        events = []
    return build_activity_matrix(events, n_bins=n_bins)


def _consequence_events(bus: Any, window: int) -> list[tuple]:
    try:
        return [(float(e.timestamp), f"bus:{e.source}") for e in bus.recent_events(window)]
    except (AttributeError, TypeError, ValueError):
        return []


def _workspace_events(workspace: Any, window: int) -> list[tuple]:
    """Broadcast winners from the global workspace: each competition win is an
    ignition event attributed to the winning subsystem, weighted by its priority."""
    try:
        records = list(getattr(workspace, "history", []) or [])[-window:]
    except (AttributeError, TypeError):
        return []
    out: list[tuple] = []
    for r in records:
        try:
            winner = getattr(r, "winner", None)
            src = str(getattr(winner, "source", "") or "")
            ts = float(getattr(r, "timestamp", 0.0) or 0.0)
            if not src or ts <= 0:
                continue
            prio = float(getattr(winner, "priority", 1.0) or 1.0)
            out.append((ts, f"gw:{src}", max(0.1, prio)))
        except (AttributeError, TypeError, ValueError):
            continue
    return out


def from_live_streams(
    *,
    bus: Any = None,
    workspace: Any = None,
    n_bins: int = 96,
    window: int = 500,
) -> ActivitySample:
    """The richest live source: merge the ConsequenceBus stream with the global
    workspace's broadcast history into one channel space.

    Channels are namespaced per stream (``bus:affect`` vs ``gw:affect_engine``)
    so a subsystem appearing on both streams is two genuinely different signals
    (its consequential actions vs its workspace wins), not an accidental merge.
    Each stream is fault-isolated: if one is unavailable the other still counts,
    and thinness is still reported honestly by the matrix builder.
    """
    events: list[tuple] = []
    try:
        if bus is None:
            from core.runtime.consequence_bus import ConsequenceBus
            bus = ConsequenceBus.get()
        events.extend(_consequence_events(bus, window))
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        pass
    try:
        if workspace is None:
            from core.runtime.service_access import resolve_global_workspace
            workspace = resolve_global_workspace(default=None)
        if workspace is not None:
            events.extend(_workspace_events(workspace, window))
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        pass
    return build_activity_matrix(events, n_bins=n_bins)


__all__ = [
    "ActivitySample",
    "build_activity_matrix",
    "from_consequence_bus",
    "from_live_streams",
]
