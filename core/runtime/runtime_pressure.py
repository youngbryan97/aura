"""Unified runtime pressure — the pull-based provider for a real contract.

The health contract requires ``unified_runtime_pressure`` ("Aura must not
claim healthy when scheduling lag or substrate survival pressure is high"),
but no provider ever existed: the requirement was a phantom and the runtime
could sit DEGRADED forever against a service nobody could start or heal
(observed live 2026-07-05, 84 minutes of DEGRADED with this entry dead).

This provider is deliberately pull-based: no thread, no loop, no task —
nothing that can die and pin the contract. Every snapshot is computed on
demand from organs that already exist:

* event-loop lag   — the registered EventLoopMonitor's status
* memory pressure  — psutil RSS / system percent
* thermal pressure — core/runtime/thermal (NSProcessInfo on macOS)

``is_alive`` is True exactly when a fresh snapshot can be produced and no
pressure dimension is in its red zone — so the contract entry now measures
real pressure instead of the liveness of a nonexistent loop.
"""
from __future__ import annotations

import logging
import sys
import time
from typing import Any

from core.runtime.resource_observation import ResourceObserver, get_resource_observer

logger = logging.getLogger("Aura.Runtime.Pressure")

# Red-zone thresholds: past these, the runtime should not claim healthy.
_LOOP_LAG_RED_S = 5.0
_MEMORY_RED_PCT = 92.0
_THERMAL_RED_LEVEL = 3  # critical


def _registry_snapshot(module: Any) -> dict[str, Any]:
    """The MLX client registry as an atomic dict, or {} when unavailable.

    Through the registry's own lock, via the accessor the module exposes.
    Reading ``_CLIENTS`` and copying it here would iterate a dict another
    thread registers into. Kept duck-typed rather than imported: core/runtime
    may not depend on core/brain, and this probe stays passive by design —
    it looks only at an already-loaded module.
    """

    if module is None:
        return {}
    snapshot = getattr(module, "clients_snapshot", None)
    if callable(snapshot):
        try:
            return dict(snapshot())
        except (RuntimeError, TypeError, ValueError):
            return {}
    return {}


def _model_resource_lifecycle_snapshot() -> dict[str, Any]:
    """Observe resident MLX allocation state without importing or waking it.

    Importing ``mlx_client`` from the pressure thread would create a new
    dependency edge during boot.  Looking only at an already-loaded module
    keeps this probe passive while still distinguishing a model allocation
    ramp from steady-state memory growth.
    """
    module = sys.modules.get("core.brain.llm.mlx_client")
    # Through the registry's own lock: copying it directly iterates a dict
    # another thread registers into, which raises mid-copy.
    clients = _registry_snapshot(module)
    if not isinstance(clients, dict) or not clients:
        return {
            "state": "cold",
            "load_active": False,
            "lane_count": 0,
            "states": [],
        }

    lane_states: list[str] = []
    load_active = False
    live_lanes = 0
    for client in list(clients.values()):
        if client is None:
            continue
        state = str(getattr(client, "_lane_state", "cold") or "cold").lower()
        lane_states.append(state)
        process = getattr(client, "_process", None)
        try:
            alive = bool(process is not None and process.is_alive())
        except (AttributeError, OSError, RuntimeError, ValueError):
            # A closed handle means the process is gone, which is an answer.
            #
            # multiprocessing raises ValueError("process object is closed")
            # once a worker has been reaped and its handle closed — and that
            # was the one exception this did not catch. It escaped into every
            # reader of the pressure snapshot at once: allostasis skipped its
            # vitals pulse, the stability guardian reported DEGRADED, and
            # inference_gate escalated it to CRITICAL SERVICE FAILURE because
            # a required subsystem is fail-closed. Measured live, three
            # subsystems failing on one stale handle while she was mid-task.
            alive = False
        if alive:
            live_lanes += 1
        initialized = bool(getattr(client, "_init_done", False))
        warming = bool(getattr(client, "_warmup_in_flight", False))
        if warming or (alive and not initialized) or state in {
            "spawning",
            "handshaking",
            "warming",
            "recovering",
        }:
            load_active = True

    if load_active:
        lifecycle = "model_loading"
    elif live_lanes and any(state == "ready" for state in lane_states):
        lifecycle = "steady"
    elif live_lanes:
        lifecycle = "model_loading"
        load_active = True
    else:
        lifecycle = "cold"
    return {
        "state": lifecycle,
        "load_active": load_active,
        "lane_count": live_lanes,
        "states": sorted(set(lane_states)),
    }


class UnifiedRuntimePressure:
    """On-demand pressure snapshot over existing organs. Nothing to crash."""

    def __init__(self, *, observer: ResourceObserver | None = None) -> None:
        self._observer = observer
        self._last_snapshot: dict[str, Any] = {}
        self._last_snapshot_at = 0.0

    def runtime_pressure_snapshot(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {"at_unix": time.time()}
        observer = self._observer or get_resource_observer()

        loop_lag_s = 0.0
        observed_loop_lag_s = 0.0
        monitor_alive = None
        monitor_running = None
        monitor_healthy = None
        monitor_incident_active = False
        loop_sample_at = 0.0
        loop_sample_age_s: float | None = None
        loop_sample_fresh = True
        loop_sample_metadata_available = False
        try:
            from core.runtime.service_registry import get_runtime_service

            monitor = get_runtime_service("event_loop_monitor", default=None)
            status = monitor.get_status() if monitor is not None else {}
            if isinstance(status, dict):
                observed_loop_lag_s = max(
                    0.0,
                    float(status.get("last_lag_s", 0.0) or 0.0),
                )
                if status:
                    monitor_alive = bool(status.get("alive", False))
                    monitor_running = bool(
                        status.get("running", status.get("alive", False))
                    )
                    monitor_healthy = bool(
                        status.get("healthy", status.get("alive", False))
                    )
                    monitor_incident_active = bool(status.get("incident_active", False))
                if "sample_fresh" in status:
                    loop_sample_metadata_available = True
                    loop_sample_fresh = bool(status.get("sample_fresh", False))
                    loop_sample_at = float(status.get("last_sample_at_unix", 0.0) or 0.0)
                    raw_age = status.get("sample_age_s")
                    loop_sample_age_s = (
                        max(0.0, float(raw_age)) if raw_age is not None else None
                    )
                loop_lag_s = observed_loop_lag_s if loop_sample_fresh else 0.0
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Pressure snapshot: loop-lag source unavailable: %s", exc)
        snapshot["loop_lag_s"] = round(loop_lag_s, 4)
        snapshot["last_observed_loop_lag_s"] = round(observed_loop_lag_s, 4)
        snapshot["loop_monitor_alive"] = monitor_alive
        snapshot["loop_monitor_running"] = monitor_running
        snapshot["loop_monitor_healthy"] = monitor_healthy
        snapshot["loop_monitor_incident_active"] = monitor_incident_active
        snapshot["loop_lag_sample_at_unix"] = loop_sample_at
        snapshot["loop_lag_sample_age_s"] = (
            round(loop_sample_age_s, 4) if loop_sample_age_s is not None else None
        )
        snapshot["loop_lag_sample_fresh"] = loop_sample_fresh
        snapshot["loop_lag_sample_metadata_available"] = loop_sample_metadata_available

        memory_pct = 0.0
        memory_rss_mb = 0.0
        process_tree_rss_mb = 0.0
        observation_available = False
        try:
            memory = observer.memory()
            memory_pct = float(memory.percent)
            memory_rss_mb = float(memory.process_rss_bytes) / float(1024**2)
            process_tree_rss_mb = float(memory.process_tree_rss_bytes) / float(1024**2)
            observation_available = bool(memory.available)
        except (AttributeError, RuntimeError, OSError, TypeError, ValueError) as exc:
            logger.debug("Pressure snapshot: memory source unavailable: %s", exc)
        snapshot["memory_pct"] = memory_pct
        snapshot["memory_rss_mb"] = round(memory_rss_mb, 3)
        snapshot["process_tree_rss_mb"] = round(process_tree_rss_mb, 3)
        model_lifecycle = _model_resource_lifecycle_snapshot()
        snapshot["model_resource_lifecycle"] = model_lifecycle["state"]
        snapshot["model_load_active"] = model_lifecycle["load_active"]
        snapshot["model_lane_count"] = model_lifecycle["lane_count"]
        snapshot["model_lane_states"] = model_lifecycle["states"]

        thermal_level = 0
        thermal_provider = "blind"
        thermal_available = False
        try:
            thermal = observer.thermal()
            thermal_level = int(thermal.level)
            thermal_provider = str(thermal.provider)
            thermal_available = bool(thermal.available)
        except (RuntimeError, AttributeError, OSError, TypeError, ValueError) as exc:
            logger.debug("Pressure snapshot: thermal source unavailable: %s", exc)
        snapshot["thermal_level"] = thermal_level
        snapshot["thermal_provider"] = thermal_provider
        snapshot["thermal_available"] = thermal_available

        disk_percent = 0.0
        disk_free_bytes = 0
        disk_available = False
        try:
            disk = observer.disk("/")
            disk_percent = float(disk.percent)
            disk_free_bytes = int(disk.free_bytes)
            disk_available = bool(disk.available)
        except (RuntimeError, AttributeError, OSError, TypeError, ValueError) as exc:
            logger.debug("Pressure snapshot: disk source unavailable: %s", exc)
        snapshot["disk_percent"] = disk_percent
        snapshot["disk_free_bytes"] = disk_free_bytes
        snapshot["disk_available"] = disk_available

        accelerator_error = ""
        try:
            accelerator = observer.accelerator()
            snapshot["accelerator"] = accelerator.to_dict()
        except (AttributeError, RuntimeError, OSError, TypeError, ValueError) as exc:
            accelerator_error = f"{type(exc).__name__}:{exc}"
            logger.debug("Pressure snapshot: accelerator source unavailable: %s", exc)
            snapshot["accelerator"] = {
                "provenance": observer.provenance.to_dict(),
                "available": False,
                "error": accelerator_error,
            }

        provenance = observer.provenance
        snapshot["observation_source"] = provenance.source.value
        snapshot["observation_scenario_id"] = provenance.scenario_id
        snapshot["host_observed"] = provenance.host_observed
        snapshot["qualifies_as_live_pressure"] = provenance.qualifies_as_live_pressure
        snapshot["resource_observation_available"] = bool(
            observation_available and thermal_available and disk_available
        )

        red_zones = []
        if loop_lag_s >= _LOOP_LAG_RED_S:
            red_zones.append(f"loop_lag_{loop_lag_s:.1f}s")
        # Availability is a current lifecycle/freshness property. A retained
        # hard-lag incident is reported separately and must not masquerade as a
        # dead signal source after healthy samples resume; doing so denied all
        # background model recovery for the full incident hold window.
        if monitor_running is False:
            red_zones.append("loop_monitor_unavailable")
        elif loop_sample_metadata_available and not loop_sample_fresh:
            red_zones.append("loop_lag_observation_stale")
        if memory_pct >= _MEMORY_RED_PCT:
            red_zones.append(f"memory_{memory_pct:.0f}pct")
        if thermal_level >= _THERMAL_RED_LEVEL:
            red_zones.append(f"thermal_level_{thermal_level}")
        if not observation_available:
            red_zones.append("memory_observation_unavailable")
        if not thermal_available:
            red_zones.append("thermal_observation_unavailable")
        if not disk_available:
            red_zones.append("disk_observation_unavailable")
        if accelerator_error:
            red_zones.append("accelerator_observation_failed")
        snapshot["red_zones"] = red_zones
        snapshot["pressure_ok"] = not red_zones

        self._last_snapshot = snapshot
        self._last_snapshot_at = time.monotonic()
        return snapshot

    def is_alive(self) -> bool:
        """Fresh snapshot succeeds and no pressure dimension is red."""
        try:
            return bool(self.runtime_pressure_snapshot().get("pressure_ok", False))
        except Exception as exc:  # noqa: BLE001 — liveness must never raise
            logger.warning("Runtime pressure snapshot failed: %s", exc)
            return False

    def get_status(self) -> dict[str, Any]:
        return dict(self._last_snapshot)


_instance: UnifiedRuntimePressure | None = None


def get_unified_runtime_pressure() -> UnifiedRuntimePressure:
    global _instance
    if _instance is None:
        _instance = UnifiedRuntimePressure()
    return _instance


def reset_unified_runtime_pressure_for_test() -> None:
    global _instance
    _instance = None


__all__ = [
    "UnifiedRuntimePressure",
    "get_unified_runtime_pressure",
    "reset_unified_runtime_pressure_for_test",
]
