"""core/observability/tracing.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lightweight distributed tracing layer — OpenTelemetry-compatible in export
format but with zero external dependencies.

Provides structured trace spans across the entire cognitive pipeline:
inference, memory, tool execution, Will decisions, and governance checks.

Design goals:
- Zero overhead when tracing is disabled (AURA_TRACING_ENABLED=0)
- No new pip dependencies — pure Python implementation
- Export format compatible with OTLP JSON (can be piped to Jaeger/Zipkin)
- Context propagation across async boundaries via contextvars
- Head-based sampling for normal traffic, always-on for errors

Usage:
    from core.observability.tracing import Tracer, get_tracer

    tracer = get_tracer()
    with tracer.span("inference_request", attributes={"model": "32b"}) as span:
        result = await run_inference(prompt)
        span.set_attribute("tokens", result.token_count)
        span.set_status("OK")
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import random
import sys
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Generator

logger = logging.getLogger("Aura.Tracing")

_TRACING_ENABLED = os.environ.get("AURA_TRACING_ENABLED", "1") == "1"
_SAMPLE_RATE = float(os.environ.get("AURA_TRACE_SAMPLE_RATE", "0.1"))

# Context variable for propagating the current span across async boundaries
_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "current_span", default=None,
)


class SpanStatus(StrEnum):
    UNSET = "UNSET"
    OK = "OK"
    ERROR = "ERROR"


@dataclass
class SpanEvent:
    """An event within a span (like a log entry)."""
    name: str
    timestamp: float = field(default_factory=time.time)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """A single trace span.

    Compatible with OpenTelemetry span model.
    """
    trace_id: str
    span_id: str
    name: str
    parent_span_id: str | None = None
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    status: SpanStatus = SpanStatus.UNSET
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)
    _token: contextvars.Token | None = field(default=None, repr=False)

    @property
    def duration_ms(self) -> float | None:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_status(self, status: str, description: str = "") -> None:
        self.status = SpanStatus(status) if status in SpanStatus.__members__ else SpanStatus.UNSET
        if description:
            self.attributes["status_description"] = description

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append(SpanEvent(name=name, attributes=attributes or {}))

    def end(self) -> None:
        if self.end_time is None:
            self.end_time = time.time()

    def to_otlp_dict(self) -> dict[str, Any]:
        """Export in OTLP-compatible JSON format."""
        return {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "parentSpanId": self.parent_span_id or "",
            "name": self.name,
            "startTimeUnixNano": int(self.start_time * 1e9),
            "endTimeUnixNano": int((self.end_time or time.time()) * 1e9),
            "status": {"code": self.status.value},
            "attributes": [
                {"key": k, "value": {"stringValue": str(v)}}
                for k, v in self.attributes.items()
            ],
            "events": [
                {
                    "name": e.name,
                    "timeUnixNano": int(e.timestamp * 1e9),
                    "attributes": [
                        {"key": k, "value": {"stringValue": str(v)}}
                        for k, v in e.attributes.items()
                    ],
                }
                for e in self.events
            ],
        }


def _generate_id(length: int = 16) -> str:
    return uuid.uuid4().hex[:length * 2]


class Tracer:
    """Lightweight tracer that produces OpenTelemetry-compatible spans.

    Thread-safe. Spans are stored in a bounded buffer and can be
    exported to stdout or a file.
    """

    def __init__(
        self,
        service_name: str = "aura",
        enabled: bool = _TRACING_ENABLED,
        sample_rate: float = _SAMPLE_RATE,
        max_spans: int = 5000,
    ) -> None:
        self.service_name = service_name
        self.enabled = enabled
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        self._spans: deque[Span] = deque(maxlen=max_spans)
        self._active_spans: dict[str, Span] = {}

    @contextmanager
    def span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        force_sample: bool = False,
    ) -> Generator[Span, None, None]:
        """Create a trace span context manager.

        Automatically propagates parent context and handles sampling.
        """
        if not self.enabled:
            yield _NOOP_SPAN
            return

        # Head-based sampling: the decision is made once at the trace root
        # and propagated to every descendant. A child under a sampled parent
        # is always recorded (no re-roll → no orphaned children); a child
        # under an unsampled root is always dropped (no parentless spans) —
        # unless force_sample promotes it (errors/forensics stay visible).
        parent = _current_span.get()
        if parent is _NOOP_SPAN and not force_sample:
            token = _current_span.set(_NOOP_SPAN)
            try:
                yield _NOOP_SPAN
            finally:
                _current_span.reset(token)
            return
        if parent is None and not force_sample and random.random() > self.sample_rate:
            token = _current_span.set(_NOOP_SPAN)
            try:
                yield _NOOP_SPAN
            finally:
                _current_span.reset(token)
            return

        if parent is _NOOP_SPAN:
            parent = None  # force-sampled span under an unsampled root
        trace_id = parent.trace_id if parent else _generate_id(16)
        parent_span_id = parent.span_id if parent else None

        span = Span(
            trace_id=trace_id,
            span_id=_generate_id(8),
            name=name,
            parent_span_id=parent_span_id,
            attributes=dict(attributes or {}),
        )
        span.set_attribute("service.name", self.service_name)

        # Set as current span
        token = _current_span.set(span)
        span._token = token

        with self._lock:
            self._active_spans[span.span_id] = span

        # A span opened inside an except block sees that outer exception in
        # sys.exc_info(); remember it so only exceptions raised BY the span
        # body mark the span as ERROR.
        pre_existing_exc = sys.exc_info()[1]
        try:
            yield span
        finally:
            # sys.exc_info() sees any in-flight exception here — including
            # BaseException subclasses like CancelledError — so error spans
            # are recorded without a broad except/re-raise handler.
            in_flight = sys.exc_info()[1]
            if in_flight is not None and in_flight is not pre_existing_exc:
                span.set_status("ERROR", str(in_flight))
                span.add_event("exception", {
                    "exception.type": type(in_flight).__name__,
                    "exception.message": str(in_flight)[:200],
                })
            span.end()
            if span.status == SpanStatus.UNSET:
                span.set_status("OK")

            # Restore parent context
            _current_span.reset(token)

            with self._lock:
                self._active_spans.pop(span.span_id, None)
                self._spans.append(span)

    def export_json(self, limit: int = 100) -> str:
        """Export recent spans as OTLP-compatible JSON."""
        with self._lock:
            spans = list(self._spans)[-limit:]
        return json.dumps({
            "resourceSpans": [{
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": self.service_name}},
                    ],
                },
                "scopeSpans": [{
                    "scope": {"name": "aura.tracing"},
                    "spans": [s.to_otlp_dict() for s in spans],
                }],
            }],
        }, indent=2)

    def recent_spans(self, limit: int = 20) -> list[Span]:
        with self._lock:
            return list(self._spans)[-limit:]

    def active_span_count(self) -> int:
        with self._lock:
            return len(self._active_spans)

    def status(self) -> dict[str, Any]:
        with self._lock:
            total = len(self._spans)
            errors = sum(1 for s in self._spans if s.status == SpanStatus.ERROR)
            return {
                "enabled": self.enabled,
                "sample_rate": self.sample_rate,
                "total_spans": total,
                "error_spans": errors,
                "active_spans": len(self._active_spans),
                "error_rate": round(errors / max(total, 1), 4),
            }


# Noop span for when tracing is disabled or not sampled
class _NoopSpan(Span):
    def __init__(self) -> None:
        super().__init__(trace_id="0", span_id="0", name="noop")

    def set_attribute(self, key: str, value: Any) -> None:
        return None  # dropped span: attributes are discarded

    def set_status(self, status: str, description: str = "") -> None:
        return None  # dropped span: status is discarded

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        return None  # dropped span: events are discarded

    def end(self) -> None:
        return None  # dropped span: nothing to finalize

_NOOP_SPAN = _NoopSpan()


# ── Module singleton ─────────────────────────────────────────────────

_tracer: Tracer | None = None
_tracer_lock = threading.Lock()


def get_tracer() -> Tracer:
    global _tracer
    if _tracer is None:
        with _tracer_lock:
            if _tracer is None:
                _tracer = Tracer()
    return _tracer
