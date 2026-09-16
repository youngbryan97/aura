"""Lifecycle owner for the organs required by Aura's live desktop speech path."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from core.runtime.live_mind_snapshot import (
    REQUIRED_LIVE_MIND_SERVICES,
    assess_live_mind_snapshot,
    collect_live_mind_snapshot,
)
from core.runtime.service_registry import get_runtime_service

logger = logging.getLogger("Aura.LiveMindRuntime")

_ACTIVATION_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


class LiveMindRuntime:
    """Materializes, probes, and reports the causal organs used by live speech.

    Snapshot collection deliberately uses the non-instantiating runtime registry.
    This owner is the only place allowed to turn the corresponding lazy factories
    into live instances, and boot calls it before registry lock.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._activated_at = 0.0
        self._activation_errors: dict[str, str] = {}
        self._last_probe: dict[str, Any] = {}
        #: What materialize has completed so far, for a waiter that bounds
        #: the activation by its progress rather than by the wall clock.
        self._materialized = ""

    def materialized(self) -> str:
        """The last organ (or the snapshot) that materialize completed."""
        return self._materialized

    def materialize(self, container: Any | None = None) -> dict[str, Any]:
        if container is None:
            from core.container import ServiceContainer

            container = ServiceContainer

        errors: dict[str, str] = {}
        for name in REQUIRED_LIVE_MIND_SERVICES:
            try:
                instance = container.get(name, default=None)
                if instance is None:
                    errors[name] = "registered service resolved to None"
            except _ACTIVATION_ERRORS as exc:
                errors[name] = f"{type(exc).__name__}: {exc}"
            self._materialized = name

        snapshot = collect_live_mind_snapshot(lane={"origin": "boot_activation"})
        self._materialized = "snapshot"
        quality = assess_live_mind_snapshot(snapshot)
        with self._lock:
            self._activated_at = time.time()
            self._activation_errors = errors
            self._last_probe = dict(quality)

        report = self.get_status()
        if report["ready"]:
            logger.info(
                "Live-mind runtime ready: %d/%d required organs materialized.",
                len(REQUIRED_LIVE_MIND_SERVICES),
                len(REQUIRED_LIVE_MIND_SERVICES),
            )
        else:
            logger.error(
                "Live-mind runtime activation incomplete: missing=%s errors=%s quality=%s",
                report["missing_services"],
                report["activation_errors"],
                quality,
            )
        return report

    def probe(self, *, lane: dict[str, Any] | None = None) -> dict[str, Any]:
        quality = assess_live_mind_snapshot(collect_live_mind_snapshot(lane=lane))
        with self._lock:
            self._last_probe = dict(quality)
        return self.get_status()

    def get_status(self) -> dict[str, Any]:
        present = {
            name: get_runtime_service(name, default=None) is not None
            for name in REQUIRED_LIVE_MIND_SERVICES
        }
        missing = [name for name, available in present.items() if not available]
        with self._lock:
            activation_errors = dict(self._activation_errors)
            last_probe = dict(self._last_probe)
            activated_at = self._activated_at
        snapshot_ready = bool(last_probe.get("ready"))
        return {
            "schema": "aura.live_mind_runtime.v1",
            "ready": bool(activated_at and not missing and not activation_errors and snapshot_ready),
            "activated_at": activated_at,
            "required_services": list(REQUIRED_LIVE_MIND_SERVICES),
            "services_present": present,
            "missing_services": missing,
            "activation_errors": activation_errors,
            "snapshot_quality": last_probe,
        }

    def is_ready(self) -> bool:
        return bool(self.get_status()["ready"])


_instance: LiveMindRuntime | None = None
_instance_lock = threading.Lock()


def get_live_mind_runtime() -> LiveMindRuntime:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = LiveMindRuntime()
    return _instance


def activate_live_mind_runtime(container: Any | None = None) -> dict[str, Any]:
    """Resolve the registered owner, then materialize its required organs.

    Runtime health deliberately observes initialized services without invoking
    their factories. Resolving the owner through the container here is therefore
    part of activation, not an optional implementation detail.
    """

    if container is None:
        from core.container import ServiceContainer

        container = ServiceContainer

    runtime = container.get("live_mind_runtime", default=None)
    if runtime is None:
        raise RuntimeError("Live-mind runtime owner is not registered")
    materialize = getattr(runtime, "materialize", None)
    if not callable(materialize):
        raise RuntimeError("Live-mind runtime owner has no materialize() contract")
    return materialize(container)
