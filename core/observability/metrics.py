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
from core.runtime.state_ownership import state_root

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
        self._custom_timers: Dict[str, list[float]] = {}

    # ── Recording methods ─────────────────────────────────────────

    def record_tick(self, duration_ms: float) -> None:
        self._tick_count += 1
        self._tick_durations.append(duration_ms)
        self._last_tick_time = time.time()
        # Reliability: feed tick duration into SLO monitor for live burn-rate tracking.
        try:
            from slo.slo_monitor import get_slo_monitor
            get_slo_monitor().record("tick_duration_p95_ms", duration_ms)
        except (ImportError, AttributeError, RuntimeError):
            pass

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

    def gauge(self, name: str, value: float) -> None:
        """Compatibility alias for runtime services that emit gauge samples."""
        self.set_gauge(name, value)

    def increment_counter(self, name: str, amount: int = 1) -> None:
        self._custom_counters[name] = self._custom_counters.get(name, 0) + amount

    def increment(self, name: str, amount: int = 1) -> None:
        self.increment_counter(name, amount)

    def timer(self, name: str) -> MetricTimer:
        return MetricTimer(self, name)

    def record_duration(self, name: str, duration: float) -> None:
        if name not in self._custom_timers:
            self._custom_timers[name] = []
        self._custom_timers[name].append(duration)

    def get_snapshot(self, name: str) -> Dict[str, Any]:
        durations = self._custom_timers.get(name, [])
        if not durations:
            return {"count": 0, "sum": 0.0, "avg": 0.0, "max": 0.0, "min": 0.0}
        return {
            "count": len(durations),
            "sum": sum(durations),
            "avg": sum(durations) / len(durations),
            "max": max(durations),
            "min": min(durations),
        }


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
            from core.runtime import resource_psutil as psutil
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
        except (ImportError, OSError, AttributeError):
            pass  # psutil unavailable or process gone

        # CPU
        try:
            from core.runtime import resource_psutil as psutil
            samples.append(MetricSample(
                name="aura_cpu_percent",
                value=float(psutil.cpu_percent(interval=0)),
                help_text="Current CPU usage percentage",
            ))
        except (ImportError, OSError, AttributeError):
            pass  # psutil unavailable

        # Substrate state
        try:
            from core.runtime.service_registry import get_runtime_service
            substrate = get_runtime_service("continuous_substrate", default=None)
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
        except (ImportError, AttributeError, TypeError, ValueError):
            pass  # Substrate not available

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
        except (ImportError, AttributeError, TypeError, ValueError):
            pass  # Will not available

        # Drive levels
        try:
            from core.runtime.service_registry import get_runtime_service
            drive_engine = get_runtime_service("drive_engine", default=None)
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
        except (ImportError, AttributeError, TypeError, ValueError):
            pass  # Drive engine not available

        # DB size
        try:
            from pathlib import Path
            db_path = Path(os.environ.get(
                "AURA_ENV_RUNTIME_DIR",
                str(state_root() / "live-source" / "data"),
            )) / "aura_state.db"
            if db_path.exists():
                self._db_size_bytes = db_path.stat().st_size
                samples.append(MetricSample(
                    name="aura_db_size_bytes",
                    value=float(self._db_size_bytes),
                    help_text="SQLite database file size in bytes",
                ))
        except (OSError, AttributeError):
            pass  # DB path not accessible

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


class MetricTimer:
    def __init__(self, collector: MetricsCollector, name: str) -> None:
        self.collector = collector
        self.name = name
        self.start_time = 0.0

    def __enter__(self) -> MetricTimer:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        duration = time.perf_counter() - self.start_time
        self.collector.record_duration(self.name, duration)


# Singleton
_metrics_instance: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get the singleton MetricsCollector instance."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = MetricsCollector()
    return _metrics_instance


def _increment_runtime_counter_sink(name: str, amount: int = 1) -> None:
    get_metrics().increment_counter(name, amount)


try:
    from core.runtime.service_registry import install_metric_counter_sink

    install_metric_counter_sink(_increment_runtime_counter_sink)
except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
    logger.debug("Runtime metric counter bridge unavailable: %s", exc)


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
    1. The canonical MindTick service is alive and making supervised progress
    2. Substrate state is finite (no NaN/Inf)
    3. Database is accessible
    4. The canonical runtime health contract is operational
    """
    metrics = get_metrics()
    issues: list[str] = []
    ready = True

    now = time.time()
    age_since_boot = now - metrics._boot_time
    tick_count = 0
    tick_age_s: float | None = None
    tick_source = "mind_tick"
    try:
        from core.runtime.service_registry import get_runtime_service

        mind_tick = get_runtime_service("mind_tick", default=None)
        if mind_tick is None:
            if age_since_boot > 120.0:
                issues.append("mind_tick_unavailable_after_120s")
                ready = False
        else:
            status_getter = getattr(mind_tick, "get_health_status", None)
            status = status_getter() if callable(status_getter) else {}
            if not isinstance(status, dict):
                status = {}
            alive = status.get("healthy")
            if alive is not True:
                is_alive = getattr(mind_tick, "is_alive", None)
                alive = is_alive() if callable(is_alive) else False
            tick_count = int(
                status.get("tick_count", getattr(mind_tick, "_tick_count", 0)) or 0
            )
            freshest_progress = max(
                float(status.get("last_successful_tick_at", 0.0) or 0.0),
                float(status.get("last_loop_progress_at", 0.0) or 0.0),
            )
            if freshest_progress > 0.0:
                tick_age_s = max(0.0, now - freshest_progress)
            if alive is not True:
                stage = str(status.get("active_tick_stage", "unknown") or "unknown")
                issues.append(f"mind_tick_unhealthy: stage={stage}")
                ready = False
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        issues.append(f"mind_tick_probe_unavailable: {type(exc).__name__}")
        ready = False

    # ``record_tick`` is currently emitted by the 60-second metabolic pulse.
    # Preserve it as telemetry, but never use a 30-second freshness threshold
    # against that slower scheduler and call the result cognitive readiness.
    metabolic_tick_count = metrics._tick_count
    metabolic_tick_age_s = None
    if metrics._last_tick_time > 0:
        metabolic_tick_age_s = max(0.0, now - metrics._last_tick_time)

    # Check substrate
    try:
        from core.runtime.service_registry import get_runtime_service
        substrate = get_runtime_service("continuous_substrate", default=None)
        if substrate and hasattr(substrate, "get_state_vector"):
            import numpy as np
            state = substrate.get_state_vector()
            if not np.isfinite(state).all():
                issues.append("substrate_nan_inf")
                ready = False
    except (ImportError, AttributeError, TypeError, ValueError):
        pass  # Substrate not loaded yet is OK during boot

    # Check DB
    try:
        from pathlib import Path
        db_path = Path(os.environ.get(
            "AURA_ENV_RUNTIME_DIR",
            str(state_root() / "live-source" / "data"),
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
    except (ImportError, sqlite3.Error, OSError) as e:
        issues.append(f"db_inaccessible: {e}")
        ready = False

    # Canonical runtime contract: readiness must include the same probes the UI
    # uses to prevent heartbeat-only false health.
    try:
        from core.runtime.health_contract import runtime_health_report

        contract = runtime_health_report()
        required_probes = contract.get("required_probes", {})
        if not bool(contract.get("operational", False)):
            issues.append(f"runtime_contract:{contract.get('status', 'unknown')}")
            ready = False
        if not bool(required_probes.get("all_passed", False)):
            failed = [
                name
                for name, probe in required_probes.items()
                if isinstance(probe, dict) and not bool(probe.get("ok", False))
            ]
            issues.append("required_probes:" + ",".join(failed))
            ready = False
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
        issues.append(f"runtime_contract_unavailable: {e}")
        ready = False

    return {
        "status": "ready" if ready else "not_ready",
        "ready": ready,
        "issues": issues,
        "uptime_s": round(age_since_boot, 1),
        "tick_source": tick_source,
        "tick_count": tick_count,
        "tick_age_s": round(tick_age_s, 3) if tick_age_s is not None else None,
        "metabolic_tick_count": metabolic_tick_count,
        "metabolic_tick_age_s": (
            round(metabolic_tick_age_s, 3)
            if metabolic_tick_age_s is not None
            else None
        ),
    }
