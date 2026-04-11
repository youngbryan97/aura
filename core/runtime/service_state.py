"""core/runtime/service_state.py

Canonical service lifecycle states for all Aura subsystems.

Every major subsystem should track its state using this enum and expose it
via a `get_service_state() -> ServiceState` method.  The orchestrator,
health endpoints, and UI all read this to provide consistent status.

State transitions:
    UNINITIALIZED -> INITIALIZING -> READY
    READY -> DEGRADED (recoverable fault)
    READY -> BLOCKED (waiting on dependency)
    READY | DEGRADED | BLOCKED -> STOPPING -> STOPPED
    Any -> FAILED (unrecoverable)
"""

from enum import Enum


class ServiceState(Enum):
    """Lifecycle state for any Aura subsystem."""

    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    DEGRADED = "degraded"      # running but impaired
    BLOCKED = "blocked"        # waiting on external dependency
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"          # unrecoverable — needs restart

    @property
    def is_operational(self) -> bool:
        """True if the service can accept work (possibly degraded)."""
        return self in (ServiceState.READY, ServiceState.DEGRADED)

    @property
    def is_terminal(self) -> bool:
        """True if the service will not recover without restart."""
        return self in (ServiceState.STOPPED, ServiceState.FAILED)
