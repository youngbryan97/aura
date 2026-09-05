"""Telemetry exporter — OpenTelemetry / Prometheus integration shell.

The audit calls for a real exporter pipeline, but bundling the OTel/
Prom SDKs is a separate concern. This module gives the *contract* a
real exporter would implement, plus a NullExporter so tests can prove
the call graph without a backend.

Real adapters can register a concrete exporter via ``set_exporter()``.
"""
from __future__ import annotations


import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class MetricSample:
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    at: float = field(default_factory=time.time)


@dataclass
class TraceSpan:
    trace_id: str
    span_id: str
    name: str
    parent_id: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    attrs: Dict[str, Any] = field(default_factory=dict)


class TelemetryExporter(Protocol):
    def emit_metric(self, sample: MetricSample) -> None: ...
    def emit_span(self, span: TraceSpan) -> None: ...
    def flush(self) -> None: ...


class NullExporter:
    def __init__(self):
        self.metrics: List[MetricSample] = []
        self.spans: List[TraceSpan] = []

    def emit_metric(self, sample: MetricSample) -> None:
        self.metrics.append(sample)

    def emit_span(self, span: TraceSpan) -> None:
        self.spans.append(span)

    def flush(self) -> None:
        return None


_exporter: TelemetryExporter = NullExporter()


def set_exporter(exporter: TelemetryExporter) -> None:
    global _exporter
    _exporter = exporter


def get_exporter() -> TelemetryExporter:
    return _exporter


def metric(name: str, value: float, **labels: str) -> None:
    _exporter.emit_metric(MetricSample(name=name, value=value, labels=labels))


def span(name: str, *, trace_id: str, span_id: str, parent_id: Optional[str] = None, **attrs: Any) -> TraceSpan:
    sp = TraceSpan(trace_id=trace_id, span_id=span_id, name=name, parent_id=parent_id, attrs=dict(attrs))
    _exporter.emit_span(sp)
    return sp
