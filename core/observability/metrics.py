"""core/observability/metrics.py
================================
Prometheus-compatible metrics and health endpoints for Aura.

Exposes /metrics in OpenMetrics/Prometheus format and /healthz + /readyz
for orchestrator liveness and readiness probes.

All metrics are collected from existing subsystem status methods —
no new data collection is introduced. This is a read-only observation layer.
"""
from __future__ import annotations

import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional

logger = logging.getLogger("Aura.Observability.Metrics")


@dataclass
class MetricSample:
    """A single metric data point."""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: str = "gauge"  # gauge, counter, histogram
    help_text: str = ""
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """Centralized metrics collector for Aura runtime.

    Collects metrics from all subsystems and exposes them in
    Prometheus text format.
    """

    def __init__(self) -> None:
        self._boot_time = time.time()
        self._tick_count = 0
        self._tick_durations: Deque[float] = deque(maxlen=100)
        self._last_tick_time = 0.0
        self._substrate_resets = 0
        self._will_decisions: Dict[str, int] = {
            "proceed": 0,
            "constrain": 0,
            "defer": 0,
            "refuse": 0,
            "critical": 0,
        }
        self._process_restarts: Dict[str, int] = {}
        self._receipt_count = 0
        self._initiative_queue_length = 0
        self._initiative_overflow_count = 0
        self._db_size_bytes = 0
        self._custom_gauges: Dict[str, float] = {}
        self._custom_counters: Dict[str, int] = {}

    # ── Recording methods ─────────────────────────────────────────

    def record_tick(self, duration_ms: float) -> None:
        self._tick_count += 1
        self._tick_durations.append(duration_ms)
        self._last_tick_time = time.time()

    def record_will_decision(self, outcome: str) -> None:
        outcome_key = outcome.lower()
        if outcome_key in self._will_decisions:
            self._will_decisions[outcome_key] += 1

    def record_substrate_reset(self) -> None:
        self._substrate_resets += 1

    def record_process_restart(self, process_name: str) -> None:
        self._process_restarts[process_name] = (
            self._process_restarts.get(process_name, 0) + 1
        )

    def record_receipt(self) -> None:
        self._receipt_count += 1

    def set_initiative_queue_length(self, length: int) -> None:
        self._initiative_queue_length = length

    def record_initiative_overflow(self) -> None:
        self._initiative_overflow_count += 1

    def set_gauge(self, name: str, value: float) -> None:
        self._custom_gauges[name] = value

    def increment_counter(self, name: str, amount: int = 1) -> None:
        self._custom_counters[name] = self._custom_counters.get(name, 0) + amount

    # ── Collection ────────────────────────────────────────────────

    def collect(self) -> list[MetricSample]:
        """Collect all current metrics."""
        samples: list[MetricSample] = []

        # Uptime
        samples.append(MetricSample(
            name="aura_uptime_seconds",
            value=time.time() - self._boot_time,
            help_text="Aura process uptime in seconds",
        ))

        # Ticks
        samples.append(MetricSample(
            name="aura_ticks_total",
            value=float(self._tick_count),
            metric_type="counter",
            help_text="Total mind ticks completed",
        ))

        if self._tick_durations:
            sorted_durations = sorted(self._tick_durations)
            samples.append(MetricSample(
                name="aura_tick_duration_ms_p50",
                value=sorted_durations[len(sorted_durations) // 2],
                help_text="Tick duration p50 in milliseconds",
            ))
            samples.append(MetricSample(
                name="aura_tick_duration_ms_p95",
                value=sorted_durations[int(len(sorted_durations) * 0.95)],
                help_text="Tick duration p95 in milliseconds",
            ))
            samples.append(MetricSample(
                name="aura_tick_duration_ms_p99",
                value=sorted_durations[int(len(sorted_durations) * 0.99)],
                help_text="Tick duration p99 in milliseconds",
            ))

        # Last tick age
        if self._last_tick_time > 0:
            samples.append(MetricSample(
                name="aura_last_tick_age_seconds",
                value=time.time() - self._last_tick_time,
                help_text="Seconds since last completed tick",
            ))

        # Will decisions
        for outcome, count in self._will_decisions.items():
            samples.append(MetricSample(
                name="aura_will_decisions_total",
                value=float(count),
                labels={"outcome": outcome},
                metric_type="counter",
                help_text="Total Will decisions by outcome",
            ))

        # Substrate
        samples.append(MetricSample(
            name="aura_substrate_resets_total",
            value=float(self._substrate_resets),
            metric_type="counter",
            help_text="Total substrate ODE resets due to NaN/Inf",
        ))

        # Process restarts
        for proc_name, count in self._process_restarts.items():
            samples.append(MetricSample(
                name="aura_process_restarts_total",
                value=float(count),
                labels={"process": proc_name},
                metric_type="counter",
                help_text="Total process restarts",
            ))

        # Receipts
        samples.append(MetricSample(
            name="aura_receipts_total",
            value=float(self._receipt_count),
            metric_type="counter",
            help_text="Total receipts emitted",
        ))

        # Initiative queue
        samples.append(MetricSample(
            name="aura_initiative_queue_length",
            value=float(self._initiative_queue_length),
            help_text="Current initiative queue length",
        ))
        samples.append(MetricSample(
            name="aura_initiative_overflow_total",
            value=float(self._initiative_overflow_count),
            metric_type="counter",
            help_text="Total initiative overflow events",
        ))

        # Memory (RSS)
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            samples.append(MetricSample(
                name="aura_memory_rss_bytes",
                value=float(mem_info.rss),
                help_text="Resident set size in bytes",
            ))
            samples.append(MetricSample(
                name="aura_memory_vms_bytes",
                value=float(mem_info.vms),
                help_text="Virtual memory size in bytes",
            ))
            # System memory
            vm = psutil.virtual_memory()
            samples.append(MetricSample(
                name="aura_system_memory_percent",
                value=float(vm.percent),
                help_text="System memory usage percentage",
            ))
        except Exception:
            pass

        # CPU
        try:
            import psutil
            samples.append(MetricSample(
                name="aura_cpu_percent",
                value=float(psutil.cpu_percent(interval=0)),
                help_text="Current CPU usage percentage",
            ))
        except Exception:
            pass

        # Substrate state
        try:
            from core.container import ServiceContainer
            substrate = ServiceContainer.get("continuous_substrate", default=None)
            if substrate and hasattr(substrate, "get_state_summary"):
                summary = substrate.get_state_summary()
                for key in ("valence", "arousal", "dominance", "phi", "curiosity", "energy"):
                    if key in summary:
                        samples.append(MetricSample(
                            name=f"aura_substrate_{key}",
                            value=float(summary[key]),
                            help_text=f"Substrate {key} value",
                        ))
                samples.append(MetricSample(
                    name="aura_substrate_step_count",
                    value=float(summary.get("step_count", 0)),
                    metric_type="counter",
                    help_text="Substrate ODE step count",
                ))
        except Exception:
            pass

        # Will status
        try:
            from core.will import get_will
            will = get_will()
            status = will.get_status()
            samples.append(MetricSample(
                name="aura_will_assertiveness",
                value=float(status.get("assertiveness", 0)),
                help_text="Will assertiveness level",
            ))
            samples.append(MetricSample(
                name="aura_will_confidence",
                value=float(status.get("confidence", 0)),
                help_text="Will confidence level",
            ))
            samples.append(MetricSample(
                name="aura_will_refuse_rate",
                value=float(status.get("refuse_rate", 0)),
                help_text="Will refuse rate (0-1)",
            ))
        except Exception:
            pass

        # Drive levels
        try:
            from core.container import ServiceContainer
            drive_engine = ServiceContainer.get("drive_engine", default=None)
            if drive_engine and hasattr(drive_engine, "get_state"):
                drive_state = drive_engine.get_state()
                drives = drive_state.get("drives", {})
                for drive_name, drive_val in drives.items():
                    if isinstance(drive_val, (int, float)):
                        samples.append(MetricSample(
                            name="aura_drive_level",
                            value=float(drive_val),
                            labels={"drive": str(drive_name)},
                            help_text="Current drive level",
                        ))
        except Exception:
            pass

        # DB size
        try:
            from pathlib import Path
            db_path = Path(os.environ.get(
                "AURA_ENV_RUNTIME_DIR",
                str(Path.home() / ".aura" / "live-source" / "data"),
            )) / "aura_state.db"
            if db_path.exists():
                self._db_size_bytes = db_path.stat().st_size
                samples.append(MetricSample(
                    name="aura_db_size_bytes",
                    value=float(self._db_size_bytes),
                    help_text="SQLite database file size in bytes",
                ))
        except Exception:
            pass

        # Custom gauges
        for name, value in self._custom_gauges.items():
            samples.append(MetricSample(
                name=f"aura_{name}",
                value=value,
                help_text=f"Custom gauge: {name}",
            ))

        # Custom counters
        for name, value in self._custom_counters.items():
            samples.append(MetricSample(
                name=f"aura_{name}",
                value=float(value),
                metric_type="counter",
                help_text=f"Custom counter: {name}",
            ))

        return samples

    def render_prometheus(self) -> str:
        """Render all metrics in Prometheus text exposition format."""
        samples = self.collect()
        lines: list[str] = []
        seen_help: set[str] = set()

        for sample in samples:
            if sample.name not in seen_help:
                if sample.help_text:
                    lines.append(f"# HELP {sample.name} {sample.help_text}")
                lines.append(f"# TYPE {sample.name} {sample.metric_type}")
                seen_help.add(sample.name)

            if sample.labels:
                label_str = ",".join(
                    f'{k}="{v}"' for k, v in sample.labels.items()
                )
                lines.append(f"{sample.name}{{{label_str}}} {sample.value}")
            else:
                lines.append(f"{sample.name} {sample.value}")

        lines.append("")
        return "\n".join(lines)


# Singleton
_metrics_instance: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get the singleton MetricsCollector instance."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = MetricsCollector()
    return _metrics_instance


# ---------------------------------------------------------------------------
# Health check functions
# ---------------------------------------------------------------------------


def check_liveness() -> Dict[str, Any]:
    """Liveness probe: is the process alive and the event loop responsive?
    Returns 200-compatible dict if alive.
    """
    return {
        "status": "alive",
        "uptime_s": round(time.time() - get_metrics()._boot_time, 1),
        "pid": os.getpid(),
    }


def check_readiness() -> Dict[str, Any]:
    """Readiness probe: can Aura accept and process requests?

    Checks:
    1. Last tick completed < 30s ago (or no ticks yet within 60s of boot)
    2. Substrate state is finite (no NaN/Inf)
    3. Database is accessible
    """
    metrics = get_metrics()
    issues: list[str] = []
    ready = True

    # Check tick recency
    now = time.time()
    age_since_boot = now - metrics._boot_time
    if metrics._last_tick_time > 0:
        tick_age = now - metrics._last_tick_time
        if tick_age > 30.0:
            issues.append(f"last_tick_stale: {tick_age:.1f}s ago")
            ready = False
    elif age_since_boot > 120.0:
        issues.append("no_ticks_completed_after_120s")
        ready = False

    # Check substrate
    try:
        from core.container import ServiceContainer
        substrate = ServiceContainer.get("continuous_substrate", default=None)
        if substrate and hasattr(substrate, "get_state_vector"):
            import numpy as np
            state = substrate.get_state_vector()
            if not np.isfinite(state).all():
                issues.append("substrate_nan_inf")
                ready = False
    except Exception:
        pass  # Substrate not loaded yet is OK during boot

    # Check DB
    try:
        from pathlib import Path
        db_path = Path(os.environ.get(
            "AURA_ENV_RUNTIME_DIR",
            str(Path.home() / ".aura" / "live-source" / "data"),
        )) / "aura_state.db"
        if db_path.exists():
            # Quick integrity check
            import sqlite3
            conn = sqlite3.connect(str(db_path), timeout=2.0)
            conn.execute("SELECT 1")
            conn.close()
        else:
            # DB not existing during first boot is OK
            pass
    except Exception as e:
        issues.append(f"db_inaccessible: {e}")
        ready = False

    return {
        "status": "ready" if ready else "not_ready",
        "ready": ready,
        "issues": issues,
        "uptime_s": round(age_since_boot, 1),
        "tick_count": metrics._tick_count,
    }
