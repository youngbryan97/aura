"""core/audit/failure_injector.py — Fault/Failure Injector."""
from __future__ import annotations

import logging

logger = logging.getLogger("Aura.FailureInjector")


class FailureInjector:
    """Injects simulated network lags, timeouts, and OS errors to verify resilience."""

    def __init__(self) -> None:
        self.active_faults: dict[str, str] = {}

    def inject_fault(self, system_name: str, fault_type: str) -> None:
        """Configures a temporary fault for the given system."""
        self.active_faults[system_name] = fault_type
        logger.warning("💉 Injected fault '%s' into system '%s'", fault_type, system_name)

    def remove_fault(self, system_name: str) -> None:
        if system_name in self.active_faults:
            del self.active_faults[system_name]
            logger.info("💉 Removed fault from system '%s'", system_name)

    def check_fault(self, system_name: str) -> bool:
        """Returns True if the system currently has a simulated fault."""
        return system_name in self.active_faults
