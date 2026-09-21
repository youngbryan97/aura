"""What the host is doing, and what admission says about it: the pressure snapshot, its default provider, and the reasons admission refuses for.

Lifted whole out of `control_plane`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PressureSnapshot:
    captured_at: float = field(default_factory=lambda: time.time())
    observation_source: str = "unavailable"
    observation_scenario_id: str = ""
    host_observed: bool = False
    qualifies_as_live_pressure: bool = False
    resource_observation_available: bool = False
    memory_percent: float = 0.0
    memory_rss_mb: float = 0.0
    process_tree_rss_mb: float = 0.0
    thermal_level: int = 0
    thermal_provider: str = "blind"
    disk_percent: float = 0.0
    disk_free_bytes: int = 0
    loop_lag_s: float = 0.0
    loop_lag_sample_age_s: float = 0.0
    loop_lag_sample_fresh: bool = True
    loop_monitor_alive: bool | None = None
    loop_monitor_running: bool | None = None
    loop_monitor_healthy: bool | None = None
    loop_monitor_incident_active: bool = False
    shutdown_requested: bool = False
    red_zones: tuple[str, ...] = ()
    suspended_capabilities: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> PressureSnapshot:
        raw = dict(value or {})
        return cls(
            captured_at=float(raw.get("captured_at") or raw.get("at_unix") or time.time()),
            observation_source=str(raw.get("observation_source") or "unavailable"),
            observation_scenario_id=str(raw.get("observation_scenario_id") or ""),
            host_observed=bool(raw.get("host_observed", False)),
            qualifies_as_live_pressure=bool(raw.get("qualifies_as_live_pressure", False)),
            resource_observation_available=bool(
                raw.get("resource_observation_available", False)
            ),
            memory_percent=max(
                0.0,
                float(raw.get("memory_percent") or raw.get("memory_pct") or 0.0),
            ),
            memory_rss_mb=max(0.0, float(raw.get("memory_rss_mb") or 0.0)),
            process_tree_rss_mb=max(
                0.0,
                float(raw.get("process_tree_rss_mb") or 0.0),
            ),
            thermal_level=max(0, int(raw.get("thermal_level") or 0)),
            thermal_provider=str(raw.get("thermal_provider") or "blind"),
            disk_percent=max(0.0, float(raw.get("disk_percent") or 0.0)),
            disk_free_bytes=max(0, int(raw.get("disk_free_bytes") or 0)),
            loop_lag_s=max(0.0, float(raw.get("loop_lag_s") or 0.0)),
            loop_lag_sample_age_s=max(
                0.0,
                float(raw.get("loop_lag_sample_age_s") or 0.0),
            ),
            loop_lag_sample_fresh=bool(
                raw.get("loop_lag_sample_fresh", True)
            ),
            loop_monitor_alive=(
                bool(raw.get("loop_monitor_alive"))
                if raw.get("loop_monitor_alive") is not None
                else None
            ),
            loop_monitor_running=(
                bool(raw.get("loop_monitor_running"))
                if raw.get("loop_monitor_running") is not None
                else (
                    bool(raw.get("loop_monitor_alive"))
                    if raw.get("loop_monitor_alive") is not None
                    else None
                )
            ),
            loop_monitor_healthy=(
                bool(raw.get("loop_monitor_healthy"))
                if raw.get("loop_monitor_healthy") is not None
                else (
                    bool(raw.get("loop_monitor_alive"))
                    if raw.get("loop_monitor_alive") is not None
                    else None
                )
            ),
            loop_monitor_incident_active=bool(
                raw.get("loop_monitor_incident_active", False)
            ),
            shutdown_requested=bool(raw.get("shutdown_requested", False)),
            red_zones=tuple(str(item) for item in raw.get("red_zones") or ()),
            suspended_capabilities=tuple(
                str(item) for item in raw.get("suspended_capabilities") or ()
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["red_zones"] = list(self.red_zones)
        payload["suspended_capabilities"] = list(self.suspended_capabilities)
        return payload


def _default_pressure_provider() -> PressureSnapshot:
    from .control_plane import (
        logger,
    )

    raw: dict[str, Any] = {}
    try:
        from core.runtime.runtime_pressure import get_unified_runtime_pressure

        raw.update(get_unified_runtime_pressure().runtime_pressure_snapshot())
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("runtime pressure provider unavailable: %s", exc)

    try:
        from core.resource.resource_governor import get_resource_governor

        resource = get_resource_governor().get_snapshot()
        if resource is not None:
            raw["memory_percent"] = float(resource.memory_percent)
            raw["memory_rss_mb"] = float(resource.memory_rss_mb)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("resource sampler unavailable: %s", exc)

    try:
        from core.runtime.shutdown_coordinator import is_shutdown_requested

        raw["shutdown_requested"] = bool(is_shutdown_requested())
    except (ImportError, RuntimeError) as exc:
        logger.debug("shutdown state unavailable: %s", exc)

    try:
        from core.runtime.service_registry import get_runtime_service

        stakes = get_runtime_service("resource_stakes", default=None)
        if stakes is not None and hasattr(stakes, "state"):
            state = stakes.state()
            raw["suspended_capabilities"] = tuple(
                getattr(state, "suspended_capabilities", ()) or ()
            )
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("resource stakes unavailable: %s", exc)

    return PressureSnapshot.from_mapping(raw)


#: Every reason this controller refuses or defers work for, by prefix. A
#: caller that sees one of these was told "not now" by admission; nothing
#: about the endpoint it asked for is in the answer. The router counted an
#: endpoint failure for anything else, and opened the brainstem's circuit
#: on `event_loop_lag_1.0s` three times in one uptime (2026-09-20).
ADMISSION_REASON_PREFIXES: tuple[str, ...] = (
    "critical_memory_pressure_",
    "moderate_memory_pressure_",
    "critical_thermal_pressure_",
    "serious_thermal_pressure_",
    "event_loop_lag_",
    "event_loop_signal_unavailable",
    "pressure_provider_unavailable",
    "runtime_shutdown_requested",
    "background_capability_suspended",
    "large_model_capability_suspended",
    "resource_busy",
    "resource_admission_",
    "candidate_worker_not_ready",
    "fairness_wait",
)


def is_an_admission_reason(reason: object) -> bool:
    """Whether ``reason`` is admission saying "not now", by its own spelling."""
    said = str(reason or "").strip().lower()
    return bool(said) and said.startswith(ADMISSION_REASON_PREFIXES)
