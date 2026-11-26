"""core/observability/trace_events.py — Chrome trace-event emission.

Clean-room adoption of Chromium's trace-event format (the one
`chrome://tracing` and `ui.perfetto.dev` read).

Logs answer "what happened". They are bad at "what happened *at the same
time as* what, and which of those was on the critical path". That second
question is the one that matters when a turn takes nine seconds and the
logs show eleven things each claiming to take under a second — the answer
is in the overlap, the gaps, and the handoffs, none of which a linear log
shows.

The format is a JSON array of events with a phase character. Four kinds
carry almost all of the value:

* **Complete slices** (`X`): a named span with a duration, on a named
  thread. Nesting is implied by containment, so a flamegraph falls out.
* **Async slices** (`b`/`e`): a span that starts on one thread and ends on
  another, or outlives its caller. Every model call, every detached
  worker, every awaited lane acquisition is one of these, and they are
  invisible in a thread-local view.
* **Flow events** (`s`/`f`): an arrow from the moment work was *requested*
  to the moment it was *done*, across threads. This is what makes queue
  latency visible — the gap between the arrow's tail and head is time the
  work existed and nothing was working on it.
* **Counters** (`C`): a value over time on the same timeline as the
  slices, so memory pressure and lane depth can be read against the spans
  they explain.

No dependencies, no daemon: events accumulate in a bounded ring and are
written as a JSON file on demand. The file opens in any Perfetto UI.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("Aura.Trace")

#: Bounded so tracing can be left on. ~100k events is a few tens of MB.
DEFAULT_CAPACITY = 100_000


@dataclass(frozen=True)
class TraceEvent:
    name: str
    category: str
    phase: str
    timestamp_us: float
    pid: int
    tid: int
    duration_us: float | None = None
    args: dict[str, Any] | None = None
    ident: str | None = None
    scope: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "cat": self.category,
            "ph": self.phase,
            "ts": round(self.timestamp_us, 3),
            "pid": self.pid,
            "tid": self.tid,
        }
        if self.duration_us is not None:
            payload["dur"] = round(self.duration_us, 3)
        if self.args:
            payload["args"] = self.args
        if self.ident is not None:
            payload["id"] = self.ident
        if self.scope is not None:
            payload["s"] = self.scope
        return payload


class Tracer:
    """Bounded, thread-safe, always-on trace-event ring."""

    def __init__(self, *, capacity: int = DEFAULT_CAPACITY) -> None:
        self._lock = threading.Lock()
        self._events: deque[TraceEvent] = deque(maxlen=capacity)
        self._enabled = True
        self._categories: set[str] = set()
        self._disabled_categories: set[str] = set()
        self._thread_names: dict[int, str] = {}
        self._pid = os.getpid()
        self.emitted = 0
        self.dropped = 0

    # ── configuration ─────────────────────────────────────────────────
    def enable(self, enabled: bool = True) -> None:
        self._enabled = enabled

    def disable_category(self, *categories: str) -> None:
        with self._lock:
            self._disabled_categories.update(categories)

    def name_thread(self, name: str) -> None:
        """Label the current thread so the timeline reads as a system."""
        tid = threading.get_ident()
        with self._lock:
            self._thread_names[tid] = name

    def _enabled_for(self, category: str) -> bool:
        return self._enabled and category not in self._disabled_categories

    @staticmethod
    def _now_us() -> float:
        return time.perf_counter() * 1e6

    def _tid(self) -> int:
        # Trace viewers want small integers; the ident is fine and stable.
        return threading.get_ident() % 1_000_000

    def _emit(self, event: TraceEvent) -> None:
        with self._lock:
            if len(self._events) == self._events.maxlen:
                self.dropped += 1
            self._events.append(event)
            self._categories.add(event.category)
            self.emitted += 1

    # ── the four useful phases ────────────────────────────────────────
    @contextmanager
    def slice(
        self, name: str, *, category: str = "aura", args: dict[str, Any] | None = None
    ) -> Iterator[dict[str, Any]]:
        """A complete slice. Nesting gives you a flamegraph for free."""
        if not self._enabled_for(category):
            yield {}
            return
        started = self._now_us()
        payload: dict[str, Any] = dict(args or {})
        try:
            yield payload
        finally:
            self._emit(
                TraceEvent(
                    name=name,
                    category=category,
                    phase="X",
                    timestamp_us=started,
                    duration_us=self._now_us() - started,
                    pid=self._pid,
                    tid=self._tid(),
                    args=payload or None,
                )
            )

    def async_begin(
        self,
        name: str,
        ident: str,
        *,
        category: str = "aura",
        args: dict[str, Any] | None = None,
    ) -> None:
        """Start a span that may end on a different thread."""
        if not self._enabled_for(category):
            return
        self._emit(
            TraceEvent(
                name=name,
                category=category,
                phase="b",
                timestamp_us=self._now_us(),
                pid=self._pid,
                tid=self._tid(),
                ident=ident,
                args=args,
            )
        )

    def async_end(
        self,
        name: str,
        ident: str,
        *,
        category: str = "aura",
        args: dict[str, Any] | None = None,
    ) -> None:
        if not self._enabled_for(category):
            return
        self._emit(
            TraceEvent(
                name=name,
                category=category,
                phase="e",
                timestamp_us=self._now_us(),
                pid=self._pid,
                tid=self._tid(),
                ident=ident,
                args=args,
            )
        )

    @contextmanager
    def async_slice(
        self, name: str, ident: str, *, category: str = "aura", args: dict[str, Any] | None = None
    ) -> Iterator[None]:
        self.async_begin(name, ident, category=category, args=args)
        try:
            yield
        finally:
            self.async_end(name, ident, category=category)

    def flow_out(self, name: str, ident: str, *, category: str = "aura") -> None:
        """Mark where work was requested. Pair with :meth:`flow_in`.

        The gap between the two is time the work existed and nothing was
        working on it — queue latency, which no thread-local view shows.
        """
        if not self._enabled_for(category):
            return
        self._emit(
            TraceEvent(
                name=name,
                category=category,
                phase="s",
                timestamp_us=self._now_us(),
                pid=self._pid,
                tid=self._tid(),
                ident=ident,
                scope="t",
            )
        )

    def flow_in(self, name: str, ident: str, *, category: str = "aura") -> None:
        if not self._enabled_for(category):
            return
        self._emit(
            TraceEvent(
                name=name,
                category=category,
                phase="f",
                timestamp_us=self._now_us(),
                pid=self._pid,
                tid=self._tid(),
                ident=ident,
                scope="t",
            )
        )

    def counter(self, name: str, values: dict[str, float], *, category: str = "aura") -> None:
        """A value on the same timeline as the slices it explains."""
        if not self._enabled_for(category):
            return
        self._emit(
            TraceEvent(
                name=name,
                category=category,
                phase="C",
                timestamp_us=self._now_us(),
                pid=self._pid,
                tid=self._tid(),
                args={k: float(v) for k, v in values.items()},
            )
        )

    def instant(
        self, name: str, *, category: str = "aura", args: dict[str, Any] | None = None
    ) -> None:
        if not self._enabled_for(category):
            return
        self._emit(
            TraceEvent(
                name=name,
                category=category,
                phase="i",
                timestamp_us=self._now_us(),
                pid=self._pid,
                tid=self._tid(),
                args=args,
                scope="t",
            )
        )

    # ── output ────────────────────────────────────────────────────────
    def to_trace_json(self) -> dict[str, Any]:
        """The object a Perfetto/chrome://tracing UI loads directly."""
        with self._lock:
            events = [e.to_dict() for e in self._events]
            names = dict(self._thread_names)
        metadata = [
            {
                "name": "thread_name",
                "ph": "M",
                "pid": self._pid,
                "tid": tid % 1_000_000,
                "args": {"name": label},
            }
            for tid, label in names.items()
        ]
        metadata.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": self._pid,
                "tid": 0,
                "args": {"name": "aura_runtime"},
            }
        )
        return {
            "traceEvents": metadata + events,
            "displayTimeUnit": "ms",
            "otherData": {
                "runtime": "aura",
                "written_at": time.time(),
                "emitted": self.emitted,
                "dropped": self.dropped,
            },
        }

    async def write(self, path: Path | None = None, *, reason: str = "manual") -> Path | None:
        with self._lock:
            empty = not self._events
        if empty:
            return None
        target = path or (_trace_dir() / f"trace_{int(time.time())}_{_slug(reason)}.json")
        body = json.dumps(self.to_trace_json(), separators=(",", ":"))
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            with local_internal_governed_scope("tracer.write"):
                gateway = get_file_write_gateway()
                await gateway.write_text_async(
                    target,
                    body,
                    durable=False,
                    source="tracer.write",
                )
        except Exception:  # noqa: BLE001 — a trace is evidence, never a dependency
            logger.warning("trace write failed", exc_info=True)
            return None
        logger.info("trace written: %s (%d events)", target, self.emitted)
        return target

    def report(self) -> dict[str, Any]:
        with self._lock:
            events = list(self._events)
            categories = sorted(self._categories)
            disabled = sorted(self._disabled_categories)
        span_us = (events[-1].timestamp_us - events[0].timestamp_us) if len(events) > 1 else 0.0
        by_phase: dict[str, int] = {}
        for event in events:
            by_phase[event.phase] = by_phase.get(event.phase, 0) + 1
        return {
            "enabled": self._enabled,
            "buffered": len(events),
            "capacity": self._events.maxlen,
            "emitted": self.emitted,
            "dropped": self.dropped,
            "span_s": round(span_us / 1e6, 3),
            "categories": categories,
            "disabled_categories": disabled,
            "by_phase": by_phase,
        }

    def reset_for_test(self) -> None:
        with self._lock:
            self._events.clear()
            self._categories.clear()
            self._disabled_categories.clear()
            self._thread_names.clear()
            self._enabled = True
            self.emitted = 0
            self.dropped = 0


def _trace_dir() -> Path:
    from core.config import config

    return Path(config.paths.data_dir) / "error_logs" / "traces"


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text)[:48].strip("_") or "trace"


_TRACER = Tracer()


def get_tracer() -> Tracer:
    return _TRACER


@contextmanager
def trace_slice(
    name: str, *, category: str = "aura", args: dict[str, Any] | None = None
) -> Iterator[dict[str, Any]]:
    with _TRACER.slice(name, category=category, args=args) as payload:
        yield payload


def trace_counter(name: str, values: dict[str, float], *, category: str = "aura") -> None:
    _TRACER.counter(name, values, category=category)


def trace_instant(name: str, *, category: str = "aura", args: dict[str, Any] | None = None) -> None:
    _TRACER.instant(name, category=category, args=args)


def tracer_report() -> dict[str, Any]:
    return _TRACER.report()


def reset_tracer_for_test() -> None:
    _TRACER.reset_for_test()


def install_pass_tracing() -> bool:
    """Emit a slice and a histogram sample for every cognitive pass.

    Both hang off the pass instrumentation, which every pipeline already
    announces itself through — so the flamegraph covers the live tick loop
    without the tick loop knowing tracing exists.
    """
    from core.pipeline.pass_manager import PassRecord, get_instrumentation

    instrumentation = get_instrumentation()
    if getattr(instrumentation, "_tracing_installed", False):
        return False

    def emit(record: PassRecord) -> None:
        if record.skipped:
            _TRACER.instant(
                f"skip:{record.name}", category="pass", args={"reason": record.reason}
            )
            return
        # Reconstruct the slice from the recorded duration: the pass has
        # already finished, so this is a complete event placed backwards.
        now_us = _TRACER._now_us()  # noqa: SLF001 — same-module clock
        duration_us = record.duration_s * 1e6
        _TRACER._emit(  # noqa: SLF001
            TraceEvent(
                name=record.name,
                category="pass",
                phase="X",
                timestamp_us=now_us - duration_us,
                duration_us=duration_us,
                pid=os.getpid(),
                tid=threading.get_ident() % 1_000_000,
                args={"ordinal": record.ordinal, "error": record.error} if record.error else None,
            )
        )
        from core.observability.histograms import record as record_histogram

        record_histogram("Aura.Pass.DurationMs", record.duration_s * 1000.0)

    instrumentation.add_after_hook(emit)
    instrumentation.add_analysis_hook(
        lambda name, seconds: _import_record()("Aura.Analysis.DurationMs", seconds * 1000.0)
    )
    instrumentation._tracing_installed = True  # type: ignore[attr-defined]
    return True


def _import_record():
    from core.observability.histograms import record

    return record


__all__ = [
    "DEFAULT_CAPACITY",
    "TraceEvent",
    "Tracer",
    "get_tracer",
    "install_pass_tracing",
    "reset_tracer_for_test",
    "trace_counter",
    "trace_instant",
    "trace_slice",
    "tracer_report",
]
