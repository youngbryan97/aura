from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class RestartEventSpec:
    day: int
    hour: int
    kind: str
    downtime_minutes: int
    label: str = ""


@dataclass(frozen=True)
class SimulationProfile:
    name: str
    description: str
    active_hours_per_day: int
    foreground_turns_per_day: int
    tool_executions_per_day: int
    autonomous_idle_windows_per_day: int
    websocket_active_during_active_hours: bool
    planned_restarts: Tuple[RestartEventSpec, ...]
    activity_start_hour_local: int = 8
    idle_window_minutes: int = 10


PROFILES: Dict[str, SimulationProfile] = {
    "stress_load": SimulationProfile(
        name="stress_load",
        description="High interaction density, continuous UI presence, and enough overnight silence for background loops to fire.",
        active_hours_per_day=12,
        foreground_turns_per_day=240,
        tool_executions_per_day=48,
        autonomous_idle_windows_per_day=12,
        websocket_active_during_active_hours=True,
        planned_restarts=(
            RestartEventSpec(day=10, hour=3, kind="graceful", downtime_minutes=5, label="graceful_maintenance_restart"),
            RestartEventSpec(day=21, hour=14, kind="abrupt", downtime_minutes=30, label="abrupt_recovery_restart"),
        ),
        activity_start_hour_local=8,
        idle_window_minutes=10,
    ),
    "mixed_daily": SimulationProfile(
        name="mixed_daily",
        description="Regular everyday use with moderate tool activity and a single graceful restart.",
        active_hours_per_day=6,
        foreground_turns_per_day=90,
        tool_executions_per_day=16,
        autonomous_idle_windows_per_day=8,
        websocket_active_during_active_hours=True,
        planned_restarts=(
            RestartEventSpec(day=15, hour=3, kind="graceful", downtime_minutes=5, label="graceful_restart"),
        ),
        activity_start_hour_local=9,
        idle_window_minutes=12,
    ),
    "idle_heavy": SimulationProfile(
        name="idle_heavy",
        description="Mostly passive uptime with light daily interaction and no planned restarts.",
        active_hours_per_day=1,
        foreground_turns_per_day=12,
        tool_executions_per_day=4,
        autonomous_idle_windows_per_day=4,
        websocket_active_during_active_hours=False,
        planned_restarts=(),
        activity_start_hour_local=10,
        idle_window_minutes=15,
    ),
}


def get_profile(name: str) -> SimulationProfile:
    key = str(name or "").strip().lower()
    if key not in PROFILES:
        raise KeyError(f"Unknown profile: {name}")
    return PROFILES[key]
