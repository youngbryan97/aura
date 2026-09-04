"""core/interiority/interoception.py — Aura's own readings, from live services.

A faculty reads two things: the appraisal frame, which is about the
world, and the interior, which is about her. The interior is not a bag
the caller fills in. It is pulled from services that were already
running and already producing these numbers, so a faculty that reads
``load`` is reading the same load the scheduler is reacting to.

Every channel here names where it comes from. A channel whose source is
unavailable is *absent* from the mapping rather than zero, because a
faculty that requires it must decline rather than treat a missing sensor
as a calm reading — which is the failure mode that makes a system report
serenity while it is on fire.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Mapping

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Interiority.Interoception")

#: Rolling affect history, which item 32 reads to detect a transition.
_TRACE_LEN = 64


class Interoception:
    """Collects Aura's own interior readings from the services that produce them."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._trace: deque[float] = deque(maxlen=_TRACE_LEN)
        self._extra: dict[str, Any] = {}
        self._reads = 0
        self._sources_missing: set[str] = set()

    def offer(self, channel: str, value: Any) -> None:
        """Supply a reading a service produced but does not publish anywhere."""
        with self._lock:
            self._extra[channel] = value

    def note_affect(self, valence: float) -> None:
        with self._lock:
            try:
                self._trace.append(float(valence))
            except (TypeError, ValueError):
                return

    def read(self) -> Mapping[str, Any]:
        """The current interior. Missing sources are absent, never zero."""
        with self._lock:
            self._reads += 1
            reading: dict[str, Any] = dict(self._extra)
            missing: set[str] = set()

        affect = self._affect_state()
        if affect is not None:
            reading.setdefault("valence", affect.get("valence", 0.0))
            reading.setdefault("arousal", affect.get("arousal", 0.0))
            reading.setdefault("engagement", affect.get("engagement", 0.0))
            with self._lock:
                self._trace.append(float(affect.get("valence", 0.0)))
        else:
            missing.add("affect_engine")

        load = self._system_load()
        if load is not None:
            reading.setdefault("load", load)
        else:
            missing.add("system_load")

        with self._lock:
            if len(self._trace) >= 2:
                reading.setdefault("affect_trace", list(self._trace))
            self._sources_missing = missing
        return reading

    # ── sources ───────────────────────────────────────────────────────
    def _affect_state(self) -> dict[str, float] | None:
        try:
            from core.container import ServiceContainer

            engine = ServiceContainer.get("affect_engine", default=None)
            if engine is None or not hasattr(engine, "get_state_sync"):
                return None
            state = engine.get_state_sync()
            if not isinstance(state, dict):
                return None
            return {
                "valence": float(state.get("valence", 0.0)),
                "arousal": float(state.get("arousal", 0.0)),
                "engagement": float(state.get("engagement", 0.0)),
            }
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "interiority.interoception", exc, action="affect reading unavailable"
            )
            return None

    def _system_load(self) -> float | None:
        try:
            import os

            one, _, _ = os.getloadavg()
            cpus = os.cpu_count() or 1
            return max(0.0, min(1.0, one / float(cpus)))
        except (OSError, AttributeError, ValueError):
            return None

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "reads": self._reads,
                "trace_length": len(self._trace),
                "offered_channels": sorted(self._extra),
                "sources_missing": sorted(self._sources_missing),
            }


_INTEROCEPTION: Interoception | None = None
_LOCK = threading.Lock()


def get_interoception() -> Interoception:
    global _INTEROCEPTION
    with _LOCK:
        if _INTEROCEPTION is None:
            _INTEROCEPTION = Interoception()
        return _INTEROCEPTION


__all__ = ["Interoception", "get_interoception"]
