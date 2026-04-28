"""Operational health monitor compatibility shim."""

from __future__ import annotations

from core.managers.health_monitor import HealthMonitor as _ManagerHealthMonitor


class HealthMonitor(_ManagerHealthMonitor):
    """Ops-facing health monitor with the interface expected by providers."""

    @property
    def error_rate(self) -> float:
        minutes = max(self.uptime / 60.0, 1.0)
        return min(1.0, float(self.total_errors) / minutes / max(self.max_consecutive_errors, 1))

    def record_success(self) -> None:
        self.reset_failure_counter()

    def is_healthy(self) -> bool:
        return bool(self.healthy)


__all__ = ["HealthMonitor"]
