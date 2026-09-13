
"""Lightweight distributed causal tracing for Aura.

Tracebacks show where an exception surfaced.  They do not show the async chain
that caused it.  This module carries a trace/span context across tasks, actor
messages, queues, repair packets, and receipts without requiring OpenTelemetry
at runtime.  Exporters can consume the JSONL ledger later.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("core.runtime.causal_trace")
import asyncio
import contextvars
import json
import os
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.state_ownership import state_root

_active_span: contextvars.ContextVar[TraceSpanContext | None] = contextvars.ContextVar(
    "aura_active_causal_span",
    default=None,
)


@dataclass(frozen=True)
class TraceSpanContext:
    trace_id: str
    span_id: str
    parent_span_id: str = ""
    name: str = ""
    origin: str = "runtime"
    started_at: float = field(default_factory=lambda: time.time())
    attrs: dict[str, Any] = field(default_factory=dict)

    def child(self, name: str, **attrs: Any) -> TraceSpanContext:
        merged = dict(self.attrs)
        merged.update(attrs)
        return TraceSpanContext(
            trace_id=self.trace_id,
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=self.span_id,
            name=name,
            origin=str(attrs.get("origin") or self.origin),
            attrs=merged,
        )

    def to_carrier(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "trace_name": self.name,
            "trace_origin": self.origin,
            # W3C Trace Context, so a carrier that leaves Aura is readable by
            # any standard tool without a translation layer. Aura's own ids are
            # the authority; this is a rendering of them, not a second identity.
            "traceparent": self.traceparent(),
        }

    def traceparent(self) -> str:
        """Render this span as a W3C ``traceparent`` header value.

        Format: version-trace_id-parent_id-flags, with trace_id 32 hex chars
        and span_id 16. Aura's ids are already hex but not always those exact
        widths, so they are normalised here rather than emitting a header that
        a conformant parser would reject.
        """
        trace = _as_hex(self.trace_id, 32)
        span = _as_hex(self.span_id, 16)
        # 01 = sampled. Aura records every span locally, so every one is.
        return f"00-{trace}-{span}-01"


def _as_hex(value: str, width: int) -> str:
    """Normalise an id to exactly ``width`` lowercase hex characters.

    Truncating or padding keeps the value stable and collision-resistant while
    satisfying parsers that reject the wrong length. A non-hex id is hashed
    rather than mangled, so an unusual id still yields a valid, stable header.
    """
    text = str(value or "").lower()
    cleaned = "".join(ch for ch in text if ch in "0123456789abcdef")
    if len(cleaned) < width:
        import hashlib

        cleaned = hashlib.blake2b(
            str(value or "").encode("utf-8", "ignore"), digest_size=width
        ).hexdigest()
    return cleaned[:width]


def new_trace(name: str, *, origin: str = "runtime", **attrs: Any) -> TraceSpanContext:
    return TraceSpanContext(
        trace_id=uuid.uuid4().hex,
        span_id=uuid.uuid4().hex[:16],
        name=name,
        origin=origin,
        attrs=dict(attrs),
    )


def current_span() -> TraceSpanContext | None:
    return _active_span.get()


def current_trace_id(default: str = "") -> str:
    span = current_span()
    return span.trace_id if span is not None else default


@contextmanager
def trace_scope(span_or_name: TraceSpanContext | str, *, origin: str = "runtime", **attrs: Any) -> Iterator[TraceSpanContext]:
    if isinstance(span_or_name, TraceSpanContext):
        span = span_or_name
    else:
        parent = current_span()
        span = parent.child(span_or_name, **attrs) if parent else new_trace(span_or_name, origin=origin, **attrs)
    token = _active_span.set(span)
    record_trace_event("span_start", span=span)
    try:
        yield span
        record_trace_event("span_end", span=span, status="ok")
    except asyncio.CancelledError as exc:
        record_trace_event("span_end", span=span, status="cancelled", error_type=type(exc).__name__, error=str(exc))
        raise
    except (AttributeError, LookupError, OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
        record_trace_event("span_end", span=span, status="error", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        _active_span.reset(token)


def inject_trace_carrier(payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    out = dict(payload or {})
    span = current_span()
    if span is not None:
        out.setdefault("_causal_trace", span.to_carrier())
        out.setdefault("trace_id", span.trace_id)
    return out


def extract_trace_carrier(payload: Mapping[str, Any] | None, *, fallback_name: str = "extracted") -> TraceSpanContext | None:
    data = dict(payload or {})
    carrier = data.get("_causal_trace")
    if not isinstance(carrier, Mapping):
        trace_id = data.get("trace_id")
        if not trace_id:
            return None
        carrier = {"trace_id": trace_id, "span_id": "", "trace_name": fallback_name}
    trace_id = str(carrier.get("trace_id") or "")
    if not trace_id:
        return None
    return TraceSpanContext(
        trace_id=trace_id,
        span_id=str(carrier.get("span_id") or uuid.uuid4().hex[:16]),
        parent_span_id=str(carrier.get("parent_span_id") or ""),
        name=str(carrier.get("trace_name") or fallback_name),
        origin=str(carrier.get("trace_origin") or data.get("origin") or "runtime"),
        attrs={k: v for k, v in data.items() if k not in {"_causal_trace"}},
    )


def _ledger_path() -> Path:
    root = Path(os.environ.get("AURA_TRACE_LEDGER", "")) if os.environ.get("AURA_TRACE_LEDGER") else state_root() / "data" / "traces"
    root.mkdir(parents=True, exist_ok=True)
    return root / "causal_trace.jsonl"


def record_trace_event(event: str, *, span: TraceSpanContext | None = None, **payload: Any) -> None:
    span = span or current_span()
    body = {
        "timestamp": time.time(),
        "event": event,
        "span": asdict(span) if span is not None else None,
        "payload": payload,
    }
    try:
        with _ledger_path().open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(body, sort_keys=True, default=str) + "\n")
    except (json.JSONDecodeError, TypeError, ValueError) as _exc:
        logger.debug("Suppressed %s in core.runtime.causal_trace: %s", type(_exc).__name__, _exc)


def trace_task_factory(name: str, coro: Any, *, origin: str = "asyncio", **attrs: Any) -> Any:
    """Wrap a coroutine so it runs inside a child causal span."""

    async def _runner() -> Any:
        parent = current_span()
        span = parent.child(name, origin=origin, **attrs) if parent else new_trace(name, origin=origin, **attrs)
        with trace_scope(span):
            return await coro

    return _runner()


__all__ = [
    "TraceSpanContext",
    "new_trace",
    "current_span",
    "current_trace_id",
    "trace_scope",
    "inject_trace_carrier",
    "extract_trace_carrier",
    "record_trace_event",
    "trace_task_factory",
]
