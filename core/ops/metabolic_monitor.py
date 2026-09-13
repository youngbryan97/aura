"""Runtime metabolic telemetry and compute-cost persistence.

The metabolic monitor is a low-level survival loop: it samples process and
system pressure from a background thread, reports health to shared state, and
triggers resource mitigation when pressure crosses operational thresholds.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime import resource_psutil as psutil
from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import FallbackClassification, Severity, record_degradation

logger = logging.getLogger("Aura.MetabolicMonitor")

_BYTES_PER_MB = 1024 * 1024
_METABOLIC_ERRORS = (
    AttributeError,
    ImportError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    psutil.Error,
)


def _record_metabolic_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    try:
        record_degradation(
            "metabolic_monitor",
            error,
            severity=severity,
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            receipt_required=True,
            extra=extra,
        )
    except TypeError as signature_exc:
        try:
            record_degradation("metabolic_monitor", error, severity=severity, action=action)
        except TypeError:
            logger.debug("Metabolic degradation could not be recorded: %s", signature_exc)


def _clamp(value: float, low: float, high: float) -> float:
    if not math.isfinite(value):
        return low
    return max(low, min(high, value))


@dataclass
class MetabolismSnapshot:
    cpu_percent: float
    ram_rss_mb: float
    ram_percent: float
    disk_usage_percent: float
    llm_latency_avg: float
    health_score: float
    pressure_state: str = "nominal"
    sample_valid: bool = True
    fault: str = ""
    timestamp: float = field(default_factory=lambda: time.time())


class MetabolicMonitor:
    """Tracks physical system resources and keeps runtime health causal."""

    def __init__(
        self,
        ram_threshold_mb: int | None = None,
        cpu_threshold: float = 80.0,
        disk_threshold: float = 92.0,
    ) -> None:
        self.process: psutil.Process | None = None
        self.ram_threshold_mb = self._resolve_ram_threshold_mb(ram_threshold_mb)
        self.cpu_threshold = max(1.0, float(cpu_threshold))
        self.disk_threshold = _clamp(float(disk_threshold), 50.0, 99.0)

        self.latency_history: list[float] = []
        self.max_latency_history = 10

        self._last_snapshot: MetabolismSnapshot | None = None
        self._lock = threading.RLock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._interval = 5.0
        self._last_error = ""
        self._last_error_at = 0.0
        self._consecutive_failures = 0
        self._last_pressure_action_at = 0.0
        self._pressure_action_cooldown_s = 30.0
        self._pressure_actions_total = 0

        self._refresh_process()

    def _resolve_ram_threshold_mb(self, configured: int | None) -> int:
        """Choose a process-RSS threshold that matches the actual host.

        Aura's primary local model lane can legitimately occupy tens of GB RSS on
        large-memory Apple Silicon systems. A fixed 8 GB stress threshold, or
        even half of physical RAM, can misclassify healthy 32B runtime as
        pressure and trigger repeated GC during proof/baseline runs. Keep
        explicit constructor/env overrides exact, and otherwise scale the default
        to 70% of physical RAM while leaving at least 8 GB for the OS.
        """
        if configured is not None:
            return max(256, int(configured))
        override = os.getenv("AURA_METABOLIC_RAM_THRESHOLD_MB", "").strip()
        if override:
            try:
                return max(256, int(float(override)))
            except ValueError as exc:
                _record_metabolic_degradation(
                    exc,
                    action="ignored invalid AURA_METABOLIC_RAM_THRESHOLD_MB override",
                    severity="warning",
                    extra={"override": override},
                )
        try:
            total_ram_mb = float(psutil.virtual_memory().total) / _BYTES_PER_MB
        except psutil.Error as exc:
            _record_metabolic_degradation(
                exc,
                action="using conservative metabolic RAM threshold after host RAM probe failed",
                severity="warning",
            )
            total_ram_mb = 12288.0
        # Scale to 85% of physical RAM while leaving at least 4 GB for the OS (with a baseline minimum of 12 GB).
        # This prevents false alarms on M-series Macs running local models with large weight footprints.
        return max(12288, int(min(total_ram_mb * 0.85, max(12288.0, total_ram_mb - 4096.0))))

    def start(self, interval: float = 5.0) -> None:
        """Start the background monitoring thread."""
        if self._running:
            return
        self._interval = max(0.5, float(interval))
        self._running = True
        self._stop_event.clear()
        try:
            self._thread = threading.Thread(
                target=self._run_loop,
                daemon=True,
                name="Aura-ANS-Metabolism",
            )
            self._thread.start()
        except _METABOLIC_ERRORS as exc:
            self._running = False
            self._remember_error(exc)
            _record_metabolic_degradation(
                exc,
                action="metabolic monitor thread did not start; runtime health marked degraded",
                severity="degraded",
            )
            return
        logger.info("Autonomic metabolism monitor active (interval %.1fs)", self._interval)

    def stop(self) -> None:
        """Stop the background monitoring thread without waiting on a full sample interval."""
        self._running = False
        self._stop_event.set()
        thread = self._thread
        if thread:
            thread.join(timeout=max(2.0, min(10.0, self._interval + 1.0)))
            if thread.is_alive():
                _record_metabolic_degradation(
                    TimeoutError("metabolic monitor thread did not stop within timeout"),
                    action="left metabolic monitor thread daemonized after bounded shutdown wait",
                    severity="warning",
                )
            self._thread = None

    def is_alive(self) -> bool:
        return bool(self._running and self._thread is not None and self._thread.is_alive())

    def _run_loop(self) -> None:
        while self._running:
            sleep_s = self._interval
            try:
                self.get_current_metabolism()
                self._consecutive_failures = 0
            except _METABOLIC_ERRORS as exc:
                self._remember_error(exc)
                sleep_s = min(self._interval * (1 + self._consecutive_failures), self._interval * 6)
                _record_metabolic_degradation(
                    exc,
                    action="kept metabolic monitor alive and backed off after sample failure",
                    severity="degraded",
                    extra={"consecutive_failures": self._consecutive_failures},
                )
                logger.error("Metabolic background loop error: %s", exc)
            self._stop_event.wait(sleep_s)

    def record_latency(self, seconds: float) -> None:
        """Track LLM response latency with bounded, sanitized history."""
        try:
            value = float(seconds)
        except (TypeError, ValueError) as exc:
            _record_metabolic_degradation(
                exc,
                action="ignored invalid latency sample",
                severity="warning",
                extra={"sample": repr(seconds)[:80]},
            )
            return
        if not math.isfinite(value) or value < 0:
            _record_metabolic_degradation(
                ValueError(f"invalid latency sample: {seconds!r}"),
                action="ignored invalid latency sample",
                severity="warning",
            )
            return
        with self._lock:
            self.latency_history.append(min(value, 600.0))
            del self.latency_history[:-self.max_latency_history]

    def get_current_metabolism(self) -> MetabolismSnapshot:
        """Collect current resource stats and calculate health score."""
        try:
            self._refresh_process()
            if self.process is None:
                raise RuntimeError("process handle unavailable")
            cpu = float(self.process.cpu_percent())
            mem_info = self.process.memory_info()
            rss_mb = float(mem_info.rss) / _BYTES_PER_MB

            cpu, rss_mb = self._sanitize_metrics(cpu, rss_mb)
            system_ram_percent = _clamp(float(psutil.virtual_memory().percent), 0.0, 100.0)
            from core.runtime.disk_budget import state_volume_percent

            disk_percent = _clamp(float(state_volume_percent()), 0.0, 100.0)

            with self._lock:
                avg_latency = (
                    sum(self.latency_history) / len(self.latency_history)
                    if self.latency_history
                    else 0.5
                )

            health_score = self._compute_health_score(
                cpu_percent=cpu,
                rss_mb=rss_mb,
                ram_percent=system_ram_percent,
                disk_percent=disk_percent,
                avg_latency=avg_latency,
            )
            pressure_state = self._classify_pressure(
                cpu_percent=cpu,
                rss_mb=rss_mb,
                ram_percent=system_ram_percent,
                disk_percent=disk_percent,
                health_score=health_score,
            )

            snapshot = MetabolismSnapshot(
                cpu_percent=cpu,
                ram_rss_mb=rss_mb,
                ram_percent=system_ram_percent,
                disk_usage_percent=disk_percent,
                llm_latency_avg=avg_latency,
                health_score=health_score,
                pressure_state=pressure_state,
            )

            with self._lock:
                self._last_snapshot = snapshot

            self._sync_registry(snapshot)
            self._apply_pressure_controls(snapshot)
            return snapshot
        except _METABOLIC_ERRORS as exc:
            self._remember_error(exc)
            _record_metabolic_degradation(
                exc,
                action="returned bounded metabolic snapshot after telemetry collection failed",
                severity="degraded",
            )
            return self._fallback_snapshot(exc)

    def _refresh_process(self) -> None:
        if self.process is not None:
            return
        try:
            self.process = psutil.Process(os.getpid())
            self.process.cpu_percent()
        except psutil.Error as exc:
            self.process = None
            self._remember_error(exc)
            _record_metabolic_degradation(
                exc,
                action="process handle unavailable; metabolic sampling will retry",
                severity="degraded",
            )

    def _sanitize_metrics(self, cpu: float, rss_mb: float) -> tuple[float, float]:
        try:
            total_ram_mb = psutil.virtual_memory().total / _BYTES_PER_MB
        except psutil.Error:
            total_ram_mb = max(float(self.ram_threshold_mb), 1024.0)
        max_reasonable_rss = max(total_ram_mb * 1.25, float(self.ram_threshold_mb) * 2)
        if cpu < 0 or cpu > 1000 or rss_mb < 0 or rss_mb > max_reasonable_rss:
            _record_metabolic_degradation(
                ValueError(f"cpu={cpu} rss_mb={rss_mb} outside sane bounds"),
                action="clamped anomalous metabolic metrics before health calculation",
                severity="warning",
            )
        return _clamp(cpu, 0.0, 1000.0), _clamp(rss_mb, 0.0, max_reasonable_rss)

    def _compute_health_score(
        self,
        *,
        cpu_percent: float,
        rss_mb: float,
        ram_percent: float,
        disk_percent: float,
        avg_latency: float,
    ) -> float:
        process_ram_factor = 1.0 - _clamp(
            (rss_mb - (self.ram_threshold_mb * 0.5)) / max(self.ram_threshold_mb * 0.75, 1.0),
            0.0,
            1.0,
        )
        system_ram_factor = 1.0 - _clamp((ram_percent - 70.0) / 28.0, 0.0, 1.0)
        cpu_factor = 1.0 - _clamp((cpu_percent - (self.cpu_threshold * 0.5)) / self.cpu_threshold, 0.0, 1.0)
        latency_factor = 1.0 - _clamp(avg_latency / 20.0, 0.0, 1.0)
        disk_factor = 1.0 - _clamp((disk_percent - 80.0) / 18.0, 0.0, 1.0)
        score = (
            process_ram_factor * 0.30
            + system_ram_factor * 0.20
            + cpu_factor * 0.25
            + latency_factor * 0.15
            + disk_factor * 0.10
        )
        return _clamp(score, 0.0, 1.0)

    def _classify_pressure(
        self,
        *,
        cpu_percent: float,
        rss_mb: float,
        ram_percent: float,
        disk_percent: float,
        health_score: float,
    ) -> str:
        if (
            health_score <= 0.20
            or ram_percent >= 96.0
            or disk_percent >= 98.0
            or rss_mb >= self.ram_threshold_mb * 1.50
            or cpu_percent >= self.cpu_threshold * 1.75
        ):
            return "critical"
        if (
            health_score <= 0.55
            or ram_percent >= 88.0
            or disk_percent >= self.disk_threshold
            or rss_mb >= self.ram_threshold_mb
            or cpu_percent >= self.cpu_threshold
        ):
            return "stressed"
        return "nominal"

    def _eviction_pressure_present(self, snapshot: MetabolismSnapshot) -> bool:
        """Return True when pressure can be relieved by eviction/GC.

        CPU-only spikes are real pressure, but running GC in response to them
        adds work and does not address the cause. Eviction is reserved for
        memory, disk, or process-RSS pressure.
        """

        return (
            snapshot.ram_percent >= 88.0
            or snapshot.disk_usage_percent >= self.disk_threshold
            or snapshot.ram_rss_mb >= self.ram_threshold_mb
        )

    def _sync_registry(self, snapshot: MetabolismSnapshot) -> None:
        try:
            from core.state.state_registry import get_registry

            get_registry().sync_update(
                health_score=snapshot.health_score,
                cpu_load=snapshot.cpu_percent,
                memory_usage=snapshot.ram_rss_mb,
            )
        except _METABOLIC_ERRORS as exc:
            _record_metabolic_degradation(
                exc,
                action="metabolic registry sync skipped after local snapshot was preserved",
                severity="warning",
            )
            logger.debug("Registry sync failed in metabolism: %s", exc)

    def _apply_pressure_controls(self, snapshot: MetabolismSnapshot) -> None:
        if snapshot.pressure_state == "nominal":
            return
        if not self._eviction_pressure_present(snapshot):
            logger.debug(
                "Metabolic pressure observed without eviction pressure: state=%s cpu=%.1f ram=%.1f rss=%.1f disk=%.1f",
                snapshot.pressure_state,
                snapshot.cpu_percent,
                snapshot.ram_percent,
                snapshot.ram_rss_mb,
                snapshot.disk_usage_percent,
            )
            return
        now = time.monotonic()
        if now - self._last_pressure_action_at < self._pressure_action_cooldown_s:
            return
        try:
            from core.resource.resource_governor import EvictionTier, get_resource_governor

            tier = EvictionTier.AGGRESSIVE if snapshot.pressure_state == "critical" else EvictionTier.MODERATE
            invoked = get_resource_governor().execute_eviction(tier)
            self._last_pressure_action_at = now
            self._pressure_actions_total += 1
            logger.warning(
                "Metabolic pressure mitigation executed: state=%s tier=%s callbacks=%d",
                snapshot.pressure_state,
                tier.value,
                invoked,
            )
        except _METABOLIC_ERRORS as exc:
            _record_metabolic_degradation(
                exc,
                action="resource pressure detected but mitigation dispatch failed",
                severity="degraded",
                extra={"pressure_state": snapshot.pressure_state},
            )

    def _fallback_snapshot(self, exc: BaseException) -> MetabolismSnapshot:
        fault = f"{type(exc).__name__}: {exc}"
        with self._lock:
            last = self._last_snapshot
            if last is not None:
                snapshot = MetabolismSnapshot(
                    cpu_percent=last.cpu_percent,
                    ram_rss_mb=last.ram_rss_mb,
                    ram_percent=last.ram_percent,
                    disk_usage_percent=last.disk_usage_percent,
                    llm_latency_avg=last.llm_latency_avg,
                    health_score=min(last.health_score, 0.5),
                    pressure_state="degraded",
                    sample_valid=False,
                    fault=fault[:240],
                )
            else:
                snapshot = MetabolismSnapshot(
                    cpu_percent=0.0,
                    ram_rss_mb=0.0,
                    ram_percent=0.0,
                    disk_usage_percent=0.0,
                    llm_latency_avg=0.0,
                    health_score=0.25,
                    pressure_state="degraded",
                    sample_valid=False,
                    fault=fault[:240],
                )
            self._last_snapshot = snapshot
            return snapshot

    def _remember_error(self, exc: BaseException) -> None:
        self._consecutive_failures += 1
        self._last_error = f"{type(exc).__name__}: {exc}"
        self._last_error_at = time.time()

    def get_status_report(self) -> dict[str, Any]:
        """Friendly dictionary for telemetry and UI."""
        with self._lock:
            snapshot = self._last_snapshot
        if snapshot is None:
            snapshot = self.get_current_metabolism()
        status = "OPTIMAL" if snapshot.health_score > 0.8 else "STRESSED" if snapshot.health_score > 0.4 else "CRITICAL"
        return {
            "health": round(snapshot.health_score * 100),
            "cpu": f"{snapshot.cpu_percent:.1f}%",
            "ram": f"{snapshot.ram_rss_mb:.0f}MB",
            "latency": f"{snapshot.llm_latency_avg:.2f}s",
            "status": status,
            "pressure_state": snapshot.pressure_state,
            "sample_valid": snapshot.sample_valid,
            "fault": snapshot.fault,
            "alive": self.is_alive(),
            "consecutive_failures": self._consecutive_failures,
            "pressure_actions_total": self._pressure_actions_total,
        }


class PersistentComputeCostTracker:
    """Tracks the metabolic cost of cognitive operations with disk persistence."""

    def __init__(self, state_path: Path | None = None) -> None:
        if state_path is None:
            from core.config import config

            state_path = config.paths.data_dir / "metabolic_state.json"
        self.state_path = Path(state_path)
        self.total_ergs = 0.0
        self.session_start = time.time()
        self.cost_history: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self._ops_since_save = 0
        self._last_saved_total = 0.0
        self._load_state()

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text())
            total = float(data.get("total_ergs", 0.0))
            if not math.isfinite(total) or total < 0:
                raise ValueError(f"invalid total_ergs: {total!r}")
            self.total_ergs = total
            self._last_saved_total = total
            logger.info("Loaded %.2f persistent ergs.", self.total_ergs)
        except _METABOLIC_ERRORS as exc:
            self.total_ergs = 0.0
            self._last_saved_total = 0.0
            _record_metabolic_degradation(
                exc,
                action="started compute cost tracker with zeroed durable state after load failed",
                severity="warning",
                extra={"state_path": str(self.state_path)[:240]},
            )

    def _save_state(self) -> None:
        try:
            payload = json.dumps(
                {
                    "total_ergs": round(self.total_ergs, 6),
                    "last_updated": time.time(),
                },
                sort_keys=True,
            )
            atomic_write_text(self.state_path, payload)
            self._last_saved_total = self.total_ergs
            self._ops_since_save = 0
        except _METABOLIC_ERRORS as exc:
            _record_metabolic_degradation(
                exc,
                action="kept compute cost in memory after durable save failed",
                severity="warning",
                extra={"state_path": str(self.state_path)[:240]},
            )

    def record_operation(
        self,
        op_type: str,
        tokens: int,
        duration_s: float,
        model_tier: str = "primary",
    ) -> float:
        """Record the cost of a cognitive operation."""
        safe_tokens = self._sanitize_count(tokens, "tokens")
        safe_duration = self._sanitize_duration(duration_s)
        tier_mult = {"primary": 1.5, "secondary": 1.0, "tertiary": 0.5}.get(model_tier, 1.0)
        ergs = (safe_tokens * 0.1) + (safe_duration * 10.0 * tier_mult)

        with self._lock:
            self.total_ergs += ergs
            entry = {
                "timestamp": time.time(),
                "op_type": str(op_type)[:120],
                "ergs": ergs,
                "tokens": safe_tokens,
                "duration": safe_duration,
                "model_tier": model_tier,
            }
            self.cost_history.append(entry)
            del self.cost_history[:-100]
            self._ops_since_save += 1
            if self.total_ergs - self._last_saved_total >= 100.0 or self._ops_since_save >= 10:
                self._save_state()

        logger.debug("Metabolic Cost: %.2f ergs (%s)", ergs, op_type)
        return ergs

    def _sanitize_count(self, value: int, field_name: str) -> int:
        try:
            safe_value = int(value)
        except (TypeError, ValueError) as exc:
            _record_metabolic_degradation(
                exc,
                action=f"clamped invalid compute cost field {field_name}",
                severity="warning",
            )
            return 0
        if safe_value < 0:
            _record_metabolic_degradation(
                ValueError(f"{field_name} cannot be negative"),
                action=f"clamped invalid compute cost field {field_name}",
                severity="warning",
            )
            return 0
        return min(safe_value, 10_000_000)

    def _sanitize_duration(self, duration_s: float) -> float:
        try:
            safe_duration = float(duration_s)
        except (TypeError, ValueError) as exc:
            _record_metabolic_degradation(
                exc,
                action="clamped invalid compute duration",
                severity="warning",
            )
            return 0.0
        if not math.isfinite(safe_duration) or safe_duration < 0:
            _record_metabolic_degradation(
                ValueError(f"duration cannot be negative or non-finite: {duration_s!r}"),
                action="clamped invalid compute duration",
                severity="warning",
            )
            return 0.0
        return min(safe_duration, 86_400.0)

    def get_metabolic_rate(self, window_s: int = 60) -> float:
        """Calculate average ergs per second over a bounded window."""
        safe_window = max(1, int(window_s))
        now = time.time()
        cutoff = now - safe_window
        with self._lock:
            recent = [entry["ergs"] for entry in self.cost_history if entry["timestamp"] > cutoff]
        if not recent:
            return 0.0
        return sum(recent) / safe_window

    def get_burn_report(self) -> dict[str, Any]:
        """Summary of energy consumption."""
        with self._lock:
            return {
                "total_ergs": f"{self.total_ergs:.2f}",
                "avg_rate": f"{self.get_metabolic_rate():.3f}",
                "uptimes_s": round(time.time() - self.session_start),
                "history_len": len(self.cost_history),
                "ops_since_save": self._ops_since_save,
            }


_cost_tracker: PersistentComputeCostTracker | None = None


def get_cost_tracker() -> PersistentComputeCostTracker:
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = PersistentComputeCostTracker()
    return _cost_tracker
