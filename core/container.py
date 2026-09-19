import asyncio
import contextvars
import functools
import hashlib
import hmac
import importlib
import inspect
import json
import logging
import os
import sys
import threading
import time
from collections.abc import Callable, Collection
from enum import Enum
from pathlib import Path
from types import FrameType
from typing import Any, Optional

from core.exceptions import (
    CircularDependencyError,
    ContainerError,
    LifecycleError,
    ServiceNotFoundError,
)
from core.governance_context import local_internal_governed_scope
from core.health.degraded_events import record_degraded_event
from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.shutdown_execution import run_sync_shutdown_callable
from core.runtime.state_ownership import state_root
from core.utils.concurrency import RobustLock

logger = logging.getLogger("Aura.Container")

_LATE_CAUSAL_SERVICES = frozenset(
    {
        "orchestrator",
        "aura_kernel",
        "kernel_interface",
        "cognitive_engine",
        "capability_engine",
        "llm_router",
        "inference_gate",
        "agency_core",
        "agency_facade",
        "memory_facade",
        "runtime_control_plane",
        "resource_admission",
        "lane_admission",
        "lane_reconciler",
        "actor_supervision",
        "inhibition_manager",
        "global_workspace",
        "attention_schema",
        "affect_facade",
        "cognitive_loop",
        "swarm",
        "agent_delegator",
        "self_modifier",
        "healing_swarm",
        "meta_cognition_shard",
    }
)

_PROTECTED_CORE_SERVICES = frozenset(
    {
        "orchestrator",
        "aura_kernel",
        "kernel_interface",
        "executive_core",
        "executive_authority",
        "constitution",
        "identity",
        "identity_guard",
        "capability_engine",
        "llm_router",
        "inference_gate",
        "runtime_control_plane",
        "resource_admission",
        "lane_admission",
        "lane_reconciler",
        "actor_supervision",
        "inhibition_manager",
        "global_workspace",
        "attention_schema",
    }
)

_CONTAINER_RECOVERABLE_ERRORS = (
    AttributeError,
    TypeError,
    ValueError,
    RuntimeError,
    OSError,
    ImportError,
    LookupError,
    TimeoutError,
)
_SERVICE_INIT_ERRORS = (*_CONTAINER_RECOVERABLE_ERRORS, LifecycleError)
_SEAL_IO_ERRORS = (OSError, json.JSONDecodeError, TypeError, ValueError)


class ServiceLifetime(Enum):
    SINGLETON = "singleton"
    TRANSIENT = "transient"

_CALLER_DISPLAY_CACHE: dict[str, str] = {}


def _caller_display(filename: str) -> str:
    """Repo-relative display for a frame filename — pure string work.

    No Path.resolve(), no stat(): this runs on the EVENT LOOP for every
    service registration, and registrations happen on hot paths (a live
    SIGUSR1 sample caught the loop inside pathlib.stat here, 8.3s of
    accumulated lag failing the health contract and pinning boot at 48%).
    """
    cached = _CALLER_DISPLAY_CACHE.get(filename)
    if cached is not None:
        return cached
    marker = "live-source"
    idx = filename.find(marker)
    if idx >= 0:
        display = filename[idx + len(marker):].lstrip("/\\")
    else:
        display = filename.rsplit("/", 1)[-1]
    if len(_CALLER_DISPLAY_CACHE) < 4096:
        _CALLER_DISPLAY_CACHE[filename] = display
    return display


def _determine_caller() -> str:
    """Name the registering module WITHOUT touching the filesystem.

    The previous implementation used traceback.extract_stack (which reads
    source lines through linecache — file I/O) plus Path.resolve()/stat per
    registration, all synchronously on the loop. Raw frame walking + string
    slicing carries the same provenance for free.
    """
    frame: FrameType | None = sys._getframe(1)
    depth = 0
    while frame is not None and depth < 12:
        filename = frame.f_code.co_filename
        if "core/container.py" not in filename and "traceback.py" not in filename:
            return _caller_display(filename)
        frame = frame.f_back
        depth += 1
    return "unknown"

class ServiceDescriptor:
    """Describes how to create and manage a service."""
    def __init__(self, name: str, factory: Callable[..., Any], lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
                 instance: Any = None, required: bool = True, initialized: bool = False,
                 dependencies: list[str] | None = None, owner: str | None = None,
                 registered_by: str | None = None, required_for: str | None = None,
                 failure_policy: str | None = None) -> None:
        self.name = name
        self.factory = factory
        self.lifetime = lifetime
        self.instance = instance
        self.required = required
        self.initialized = initialized
        self._async_initialized = False
        self.dependencies = list(dependencies or [])
        caller = _determine_caller()
        self.owner = owner or caller
        self.registered_by = registered_by or caller
        self.required_for = required_for or ("boot" if required else "optional features")
        self.failure_policy = failure_policy or ("fail-closed" if required else "degrade_with_receipt")


def _callable_attr(instance: Any, attr_name: str) -> Callable[..., Any] | None:
    """Return a callable instance attribute without treating absence as failure."""
    try:
        inspect.getattr_static(instance, attr_name)
    except (AttributeError, TypeError):
        return None
    try:
        attr = getattr(instance, attr_name)
    except _CONTAINER_RECOVERABLE_ERRORS as exc:
        record_degradation("container", exc)
        logger.debug("Unable to resolve %s on %s: %s", attr_name, type(instance).__name__, exc)
        return None
    return attr if callable(attr) else None


def _status_from_result(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("status", "active"))
    return str(result)


def _read_instance_status(name: str, instance: Any) -> str:
    for attr_name in ("get_status", "status"):
        status_fn = _callable_attr(instance, attr_name)
        if status_fn is None:
            continue
        if inspect.iscoroutinefunction(status_fn):
            return "async_status_unread"
        try:
            result = status_fn()
            if inspect.isawaitable(result):
                close = getattr(result, "close", None)
                if callable(close):
                    close()
                return "async_status_unread"
            return _status_from_result(result)
        except _CONTAINER_RECOVERABLE_ERRORS as exc:
            record_degradation("container", exc)
            logger.debug("Status read failed for %s via %s: %s", name, attr_name, exc)
            return "status_error"
    return "active_unverified"


def zero_sync_guard(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to ensure async methods do not perform synchronous blocking calls."""
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        # In the future, this could monitor thread status or event loop lag
        return await func(*args, **kwargs)
    return wrapper


ZeroSyncGuard = zero_sync_guard

class _BootRegistrationLease:
    __slots__ = ("_container", "_token", "active", "name")

    #: The ServiceContainer class this lease belongs to. Named rather than
    #: bare `type`, which has no `_boot_lease`, so the three reads of it
    #: below could not be checked at all.
    _container: "type[ServiceContainer]"

    def __init__(self, container: "type[ServiceContainer]", name: str) -> None:
        self._container = container
        self._token: contextvars.Token[Any] | None = None
        self.active = False
        self.name = name

    def __enter__(self) -> "_BootRegistrationLease":
        self.active = True
        self._token = self._container._boot_lease.set(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        self.active = False
        if self._token is not None:
            try:
                self._container._boot_lease.reset(self._token)
            except ValueError:
                # Reset from a different context than set: the lease was
                # entered in one task and left in another. The flag is what
                # gates registration; the var only carries it.
                self._container._boot_lease.set(None)
            self._token = None


def _offer_to_the_shed_order(name: str, instance: Any) -> None:
    """Tell the OOM policy about an organ that can free memory.

    A singleton becomes real here and nowhere else, which makes this the
    only place the shed order can be complete. It used to be built by one
    sweep of the container in boot wave one, when the container is nearly
    empty by design, so the ladder had no rungs and the only answer to
    memory pressure was a restart.
    """

    try:
        from core.runtime.oom_policy import offer_organ

        offer_organ(name, instance)
    except Exception:  # noqa: BLE001 — a shed offer never fails a service
        logger.debug("shed-order offer skipped for %r", name, exc_info=True)


class ServiceContainer:
    """Aura 3.0 Static ServiceContainer.
    
    Zenith Protocol: 
    - Zero dynamic imports in get()
    - Registration frozen after wake()
    - Atomic per-service initialization locks
    """
    _instance: Optional["ServiceContainer"] = None
    _lock = threading.RLock()
    # NOTE: _services and _aliases are class-level singletons.  This is
    # intentional — ServiceContainer itself is a singleton (__new__) and all
    # access goes through classmethods.  Keep them here so that callers can
    # interact with the class directly without needing an instance.
    _services: dict[str, ServiceDescriptor] = {}
    _aliases: dict[str, str] = {}
    _registration_locked = False
    #: The deferred boot initialisers register services after the lock can
    #: fall (on a loaded host they landed 65s after it, 2026-09-15). A task
    #: running under a boot lease is boot, and registers through the lock
    #: for as long as the task that holds the lease is running.
    _boot_lease: contextvars.ContextVar["_BootRegistrationLease | None"] = (
        contextvars.ContextVar("service_container_boot_lease", default=None)
    )
    _resolving_var: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar('resolving', default=frozenset())
    _wake_lock = RobustLock("ServiceContainer.Wake")
    _start_time: float | None = None
    _init_locks: dict[str, threading.Lock] = {}
    _last_seal_hash: str | None = None
    _optional_absent_breadcrumbs: set[str] = set()
    _shutdown_reports: list[dict[str, Any]] = []

    def __new__(cls) -> "ServiceContainer":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    #: Import results for check_package, keyed by module name. A failed
    #: import is expensive to repeat and requirement validation runs across
    #: the whole skill catalog.
    _package_availability: dict[str, bool] = {}

    @classmethod
    def get_all_subsystem_statuses(cls) -> dict[str, str]:
        """Return the active/degraded/missing status of all registered subsystems."""
        with cls._lock:
            statuses = {}
            for name, desc in cls._services.items():
                if desc.instance is not None:
                    statuses[name] = _read_instance_status(name, desc.instance)
                else:
                    if desc.required:
                        statuses[name] = "missing"
                    else:
                        statuses[name] = "optional_missing"
            return statuses

    @classmethod
    def _runtime_registration_suppressed(cls, name: str) -> bool:
        """True when the shutdown latch suppresses this registration.

        Suppression is SOFT — record the admission event and let the caller
        no-op. The previous hard raise detonated whatever path touched the
        container mid-teardown: the aura_now telemetry republish (fired per
        Will sample) raised ContainerError INSIDE kernel.shutdown(), which
        aborted teardown before the vault closed — leaking non-daemon
        aiosqlite worker threads that held interpreter exit until the chunk
        runner's 2400s timeout, twice, with zero failure evidence. Shutdown
        philosophy is flush-as-much-as-possible; refusing new work must
        never abort the teardown that refuses it.
        """
        try:
            from core.runtime.shutdown_coordinator import (
                is_shutdown_requested,
                record_shutdown_admission_event,
            )

            if not is_shutdown_requested():
                return False
            record_shutdown_admission_event(
                f"service_container.register:{name}",
                resource_kind="service",
                outcome="suppressed",
                detail="shutdown_latch",
            )
            logger.debug(
                "Service registration '%s' suppressed: runtime shutdown is active.",
                name,
            )
            return True
        except ImportError:
            return False

    @classmethod
    def _absent_service_error(cls, name: str) -> ContainerError:
        """The right exception for a name that has no descriptor.

        LIVE DEFECT, 2026-08-09 (found in desktop-launch.log 2026-08-10).
        ``register`` suppresses SOFTLY under the shutdown latch — it returns
        without registering — while ``get`` raised HARD. A name can therefore
        be absent for two very different reasons: nobody ever registered it (a
        wiring bug), or the latch refused the registration microseconds ago
        (ordinary teardown). Reporting both as "not found in static registry"
        turned a mid-boot SIGTERM into what looked like a wiring bug:

            ServiceNotFoundError: Service 'defensive_runtime' not found in
            static registry.
            ERROR:    Application startup failed. Exiting.

        The runtime had been told to quit while it was still booting; the API
        server's lifespan went on to register the whole service graph anyway,
        every registration was suppressed, and the first register-then-get
        pair detonated. Any such pair is a landmine during shutdown, and the
        traceback names the wrong cause.
        """
        if cls._shutdown_latch_active():
            return ContainerError(
                f"runtime shutdown is active: service '{name}' was not registered "
                "because the shutdown latch suppressed its registration"
            )
        return ServiceNotFoundError(f"Service '{name}' not found in static registry.")

    @classmethod
    def _shutdown_latch_active(cls) -> bool:
        try:
            from core.runtime.shutdown_coordinator import is_shutdown_requested

            return bool(is_shutdown_requested())
        except (ImportError, RuntimeError):
            return False

    @classmethod
    def _assert_runtime_initialization_allowed(cls, name: str) -> None:
        try:
            from core.runtime.shutdown_coordinator import (
                is_shutdown_requested,
                record_shutdown_admission_event,
            )

            if not is_shutdown_requested():
                return
            record_shutdown_admission_event(
                f"service_container.initialize:{name}",
                resource_kind="service",
                outcome="suppressed",
                detail="shutdown_latch",
            )
            raise ContainerError(
                f"runtime shutdown is active: cannot initialize service '{name}'"
            )
        except ImportError:
            return

    @classmethod
    def _begin_new_lifecycle_epoch_if_needed(cls) -> None:
        with cls._lock:
            if cls._shutdown_reports and not any(
                descriptor.instance is not None
                for descriptor in cls._services.values()
            ):
                cls._shutdown_reports.clear()
            
    @classmethod
    def register(
        cls,
        name: str,
        factory: Callable[..., Any],
        lifetime: ServiceLifetime | str = ServiceLifetime.SINGLETON,
        required: bool = True,
        dependencies: list[str] | None = None,
        owner: str | None = None,
        registered_by: str | None = None,
        required_for: str | None = None,
        failure_policy: str | None = None,
    ) -> None:
        """Register a service factory."""
        if cls._runtime_registration_suppressed(name):
            return
        cls._begin_new_lifecycle_epoch_if_needed()
        if cls._registration_locked and not cls._boot_lease_active():
            raise ContainerError(f"Registration locked: Cannot register '{name}'")
        if isinstance(lifetime, str):
            try:
                lifetime = ServiceLifetime(lifetime)
            except ValueError:
                logger.warning(
                    "Unknown service lifetime '%s' for '%s'; defaulting to singleton.",
                    lifetime,
                    name,
                )
                lifetime = ServiceLifetime.SINGLETON
        if not callable(factory):
            logger.debug(
                "Normalizing legacy non-callable registration for '%s' into a pre-built instance.",
                name,
            )
            with cls._lock:
                cls._services[name] = ServiceDescriptor(
                    name=name,
                    factory=lambda: factory,
                    lifetime=ServiceLifetime.SINGLETON,
                    instance=factory,
                    required=required,
                    initialized=True,
                    dependencies=[],
                    owner=owner,
                    registered_by=registered_by,
                    required_for=required_for,
                    failure_policy=failure_policy,
                )
                logger.debug("Registered legacy pre-built instance via register(): %s", name)
            return
        with cls._lock:
            cls._services[name] = ServiceDescriptor(
                name,
                factory,
                lifetime,
                required=required,
                dependencies=dependencies,
                owner=owner,
                registered_by=registered_by,
                required_for=required_for,
                failure_policy=failure_policy,
            )
            logger.debug("Registered static service: %s", name)
    @classmethod
    def unlock_registration(cls, *, caller: str = "", reason: str = "") -> None:
        """Unlock registration to allow dynamic service updates.

        AUDIT: Every unlock is logged at WARNING level with the caller identity
        and reason. This ensures that any post-boot service injection is visible
        in logs and can be traced to a specific subsystem. The audit trail is
        the primary defense against unauthorized runtime service replacement.

        Args:
            caller: Name of the subsystem requesting the unlock. Should be
                    a module path or class name, not "unknown".
            reason: Human-readable reason for the unlock (e.g., "late boot
                    service registration for affective_circumplex").
        """
        # CP126 (critical): "Any caller can unlock service registration.
        # unlock_registration accepts descriptive caller and reason strings
        # but authenticates neither."
        #
        # This cannot be authenticated from inside the process — anything
        # with import access can call it, and a signature check would be
        # verified by the same code an attacker already controls. What CAN
        # be fixed is the audit claim: the docstring calls the audit trail
        # "the primary defense", and a defence that accepts caller="unknown"
        # by default and emits only a log line is not one.
        #
        # Attribution is now required, and every unlock is recorded as a
        # degradation so it reaches the incident ledger rather than only a
        # log stream nobody greps.
        import traceback
        attributed_caller = str(caller or "").strip()
        if not attributed_caller:
            raise ValueError(
                "unlock_registration requires an explicit caller identity: "
                "reopening the sealed service registry must be attributable. "
                "Pass caller='<module or class>' and reason='<why>'."
            )
        with cls._lock:
            cls._registration_locked = False
            # Log at WARNING to ensure visibility in production logs
            frame_info = ""
            stack = traceback.extract_stack(limit=3)
            if len(stack) >= 2:
                frame = stack[-2]
                frame_info = f" (from {frame.filename}:{frame.lineno})"
            logger.warning(
                "ServiceContainer registration UNLOCKED by '%s'%s%s",
                attributed_caller,
                frame_info,
                f" — reason: {reason}" if reason else "",
            )
        try:
            from core.runtime.errors import record_degradation

            record_degradation(
                "service_container",
                RuntimeError(
                    f"service registry unlocked by {attributed_caller!r}"
                    f"{frame_info} reason={reason or 'unstated'}"
                ),
                severity="warning",
                action="reopened the sealed service registry",
                enforce_failure_policy=False,
            )
        except (ImportError, RuntimeError, TypeError, ValueError):
            # The log line above is still emitted; never let audit plumbing
            # stop an unlock a booting runtime may depend on.
            pass

    @classmethod
    def _boot_lease_active(cls) -> bool:
        lease = cls._boot_lease.get()
        return bool(lease is not None and lease.active)

    @classmethod
    def boot_registration_lease(cls, name: str) -> "_BootRegistrationLease":
        """A lease for one deferred boot task to register through the lock.

        Enter it inside the task: the lease lives in the task's context, so
        everything the task awaits sees it, and it is deactivated on exit —
        a task the boot task spawned inherits the same lease object and loses
        the right when its parent finishes.
        """
        return _BootRegistrationLease(cls, str(name))

    @classmethod
    def lock_registration(cls) -> None:
        """Standard locking interface."""
        with cls._lock:
            cls._registration_locked = True
            logger.info("🔒 ServiceContainer registration LOCKED")

    @classmethod
    def register_instance(
        cls,
        name: str,
        instance: Any,
        required: bool = True,
        owner: str | None = None,
        registered_by: str | None = None,
        required_for: str | None = None,
        failure_policy: str | None = None,
    ) -> None:
        """Register a pre-built instance.

        Unlike factory-based ``register()``, pre-built instances are safe to
        add after the lock because they carry no lazy-init or circular-dep
        risk.  Late registrations are permitted with a warning so that
        subsystems that boot asynchronously (final_engines, affective_circumplex,
        architecture_index, etc.) can complete their setup without crashing.
        """
        if cls._runtime_registration_suppressed(name):
            return
        cls._begin_new_lifecycle_epoch_if_needed()
        with cls._lock:
            resolved_name = cls._resolve_name(name)
            desc = cls._services.get(resolved_name)
            if desc and desc.instance is not None:
                existing_instance = desc.instance
            elif desc and not callable(desc.factory):
                existing_instance = desc.factory
            else:
                existing_instance = None

            existing = desc is not None

            # HOT-PATH UPSERT: re-publishing a live value under an existing
            # non-protected name (aura_now on every Will decision, telemetry
            # snapshots, etc.) must not rebuild a ServiceDescriptor — the
            # constructor walks frames for provenance, and a live SIGUSR1
            # sample caught exactly that churn lagging the event loop 8.3s
            # and failing the health contract. Swap the instance in place;
            # provenance from first registration stands.
            # CP126 (critical): "Alias registration can bypass protected-
            # service checks. Instance registration resolves aliases for
            # descriptor replacement while protection decisions use the raw
            # incoming name."
            #
            # `resolved_name` finds the descriptor; every protection check
            # below used `name`. So an alias pointing at a protected service
            # resolved to that service's descriptor while the guard compared
            # the ALIAS against _PROTECTED_CORE_SERVICES, found no match, and
            # took the hot-path upsert — silently swapping the instance of a
            # service the container is meant to refuse to overwrite.
            # Reproduced: registering under an alias replaced a protected
            # core service after the registry was locked.
            #
            # Protection is a property of the service, not of the name the
            # caller happened to use to reach it.
            if (
                desc is not None
                and resolved_name not in _PROTECTED_CORE_SERVICES
                and desc.lifetime == ServiceLifetime.SINGLETON
            ):
                desc.instance = instance
                desc.factory = lambda: instance
                desc.initialized = True
                _offer_to_the_shed_order(resolved_name, instance)
                return
        if cls._registration_locked:
            logger.debug("⚠️ Late instance registration (post-lock): '%s' — allowed for pre-built instances.", name)
            if (
                existing
                and resolved_name in _PROTECTED_CORE_SERVICES
                and existing_instance is not instance
            ):
                logger.error(
                    "🚫 Protected core service overwrite blocked after lock: '%s'%s",
                    resolved_name,
                    f" (via alias '{name}')" if resolved_name != name else "",
                )
                record_degraded_event(
                    "service_container",
                    "protected_service_overwrite_blocked",
                    detail=resolved_name,
                    severity="error",
                    classification="foreground_blocking",
                    context={"service": resolved_name, "requested_as": name},
                )
                return
            if not existing and name in _LATE_CAUSAL_SERVICES:
                logger.warning("⚠️ Late CAUSAL instance registration after lock: '%s'", name)
                record_degraded_event(
                    "service_container",
                    "late_causal_registration",
                    detail=name,
                    severity="warning",
                    classification="background_degraded",
                    context={"service": name},
                )
        with cls._lock:
            cls._services[name] = ServiceDescriptor(
                name=name,
                factory=lambda: instance,
                lifetime=ServiceLifetime.SINGLETON,
                instance=instance,
                required=required,
                initialized=True,
                owner=owner,
                registered_by=registered_by,
                required_for=required_for,
                failure_policy=failure_policy,
            )
            logger.debug("Registered pre-built instance: %s", name)

    @classmethod
    def set(cls, name: str, instance: Any, required: bool = True) -> Any:
        """Legacy compatibility alias for replacing a singleton instance.

        A large portion of Aura's older runtime expects ``ServiceContainer.set``
        to behave like an upsert for already-built singleton instances.
        """
        cls.register_instance(name, instance, required=required)
        return instance

    @classmethod
    def register_alias(cls, alias: str, target: str) -> None:
        """Register a legacy service alias that resolves to another service name."""
        if cls._runtime_registration_suppressed(alias):
            return
        if cls._registration_locked and not cls._boot_lease_active():
            raise ContainerError(f"Registration locked: Cannot register alias '{alias}'")
        with cls._lock:
            cls._aliases[alias] = target
            logger.debug("Registered service alias: %s -> %s", alias, target)

    @classmethod
    def register_aliases(cls, aliases: dict[str, str]) -> None:
        """Bulk-register legacy aliases."""
        for alias, target in aliases.items():
            cls.register_alias(alias, target)

    @classmethod
    def unregister_instance(
        cls,
        name: str,
        *,
        expected_instance: Any | None = None,
    ) -> bool:
        """Remove one registration without deleting a replacement instance.

        Runtime owners call this during teardown.  The identity precondition is
        important: a stale owner that closes after a replacement was published
        must not remove the replacement from the process-wide service graph.
        """

        with cls._lock:
            resolved_name = cls._resolve_name(name)
            descriptor = cls._services.get(resolved_name)
            if descriptor is None:
                return False
            current = descriptor.instance
            if current is None and not callable(descriptor.factory):
                current = descriptor.factory
            if expected_instance is not None and current is not expected_instance:
                return False
            cls._services.pop(resolved_name, None)
            return True

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Compatibility wrapper for callers that own the named registration."""

        return cls.unregister_instance(name)

    @classmethod
    def clear(cls) -> None:
        """Reset the static registry to a pristine state for tests and warm reboots."""
        with cls._lock:
            cls._services.clear()
            cls._aliases.clear()
            cls._init_locks.clear()
            cls._optional_absent_breadcrumbs.clear()
            cls._shutdown_reports.clear()
            cls._registration_locked = False
            cls._start_time = None
        cls._resolving_var.set(frozenset())
        try:
            from core.service_registration import register_all_services
            if hasattr(register_all_services, "_full_run"):
                register_all_services._full_run = False
        except (ImportError, AttributeError) as _exc:
            record_degradation('container', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

    @classmethod
    def reset(cls) -> None:
        """Legacy compatibility alias used by older tests and boot flows."""
        cls.clear()

    @classmethod
    def _resolve_name(cls, name: str) -> str:
        """Resolve an alias chain to its canonical service name."""
        seen: set[str] = set()
        current = name
        for _ in range(len(cls._aliases) + 1):
            with cls._lock:
                target = cls._aliases.get(current)
            if not target:
                return current
            if current in seen:
                raise CircularDependencyError(f"Circular dependency detected in aliases for '{name}'")
            seen.add(current)
            current = target
        raise CircularDependencyError(f"Circular dependency detected in aliases for '{name}'")

    @classmethod
    def _infer_dependency_names(cls, desc: ServiceDescriptor) -> list[str]:
        """Infer dependencies from explicit metadata or required factory parameters."""
        if desc.dependencies:
            return list(desc.dependencies)

        try:
            signature = inspect.signature(desc.factory)
        except (TypeError, ValueError):
            return []

        dependencies: list[str] = []
        for param in signature.parameters.values():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if param.default is not inspect._empty:
                continue
            dependencies.append(param.name)
        return dependencies

    @classmethod
    def _build_factory_call(cls, name: str, desc: ServiceDescriptor) -> tuple[list[Any], dict[str, Any]]:
        """Resolve service dependencies and map them onto the target factory signature."""
        dependency_names = cls._infer_dependency_names(desc)
        if not dependency_names:
            return [], {}

        resolved = {dep_name: cls.get(dep_name) for dep_name in dependency_names}

        try:
            signature = inspect.signature(desc.factory)
        except (TypeError, ValueError):
            return [resolved[dep_name] for dep_name in dependency_names], {}

        args: list[Any] = []
        kwargs: dict[str, Any] = {}
        unresolved = dict(resolved)

        for param in signature.parameters.values():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if param.default is not inspect._empty and param.name not in unresolved:
                continue
            if param.name not in unresolved:
                raise ServiceNotFoundError(f"Service '{name}' missing dependency '{param.name}'")

            value = unresolved.pop(param.name)
            if param.kind is inspect.Parameter.POSITIONAL_ONLY:
                args.append(value)
            else:
                kwargs[param.name] = value

        if unresolved:
            if not signature.parameters:
                # A zero-parameter factory cannot receive injected values.
                # Its declared dependencies are ordering/existence
                # constraints — already satisfied by resolving them above.
                # Appending them positionally guaranteed a boot-time
                # TypeError (lived once: actor_supervision took the whole
                # runtime down at the first health evaluation).
                return [], {}
            # Fall back to positional ordering for legacy factories that specify dependencies
            # separately from their parameter names.
            args.extend(unresolved[dep_name] for dep_name in dependency_names if dep_name in unresolved)

        return args, kwargs

    @classmethod
    def has(cls, name: str) -> bool:
        """Check if a service is registered."""
        resolved_name = cls._resolve_name(name)
        with cls._lock:
            return resolved_name in cls._services

    @classmethod
    def check_package(cls, package_name: str, *, auto_install: bool = False) -> bool:
        """Is this Python package actually importable right now?

        CP126 (critical): "SkillRequirements and the engine proxy call
        ServiceContainer.check_package, but ServiceContainer defines no such
        method. Any skill declaring Python package requirements can fail
        during catalog validation instead of receiving a truthful answer."

        Three call sites invoked this — per-skill requirement validation,
        the engine's async proxy, and the boot dependency check — and every
        one of them raised AttributeError instead of getting an answer. A
        skill that honestly declared its dependencies was punished for it.

        The check performs the real import rather than ``find_spec``.
        ``core/runtime/integration_liveness.py`` documents why at length:
        ``find_spec`` answers "is this module present on disk", which is not
        the question. A package whose ``__init__`` raises on a moved API or
        a missing transitive dependency is present and unimportable — so a
        find_spec check reports available and the feature is dead. Nineteen
        call sites in this codebase already make that mistake; this is not
        the twentieth.

        Results are cached because requirement validation runs per skill
        across the whole catalog, and a failed import is expensive to repeat.

        ``auto_install`` is accepted and deliberately NOT honoured — see
        below.
        """
        name = str(package_name or "").strip()
        if not name:
            return False

        if auto_install:
            # Installing into the interpreter's environment at runtime is
            # not a dependency check, it is an unreviewed mutation of the
            # machine — network egress, arbitrary setup code, and a venv
            # that no longer matches any lockfile. This venv is also shared
            # with long-running training processes, where a package
            # changing underneath a run invalidates it silently.
            #
            # The parameter is kept so the existing callers keep working and
            # so the refusal is visible at the call site rather than being a
            # silently ignored argument.
            logger.warning(
                "check_package(%s, auto_install=True): refusing to install at "
                "runtime; reporting availability only.",
                name,
            )

        cached = cls._package_availability.get(name)
        if cached is not None:
            return cached

        try:
            importlib.import_module(name)
            available = True
        except ImportError:
            available = False
        except Exception as exc:  # noqa: BLE001 - a package __init__ may raise anything
            # Installed but unimportable: the silent-decay state. It is a
            # negative answer, and it is worth saying out loud, because
            # "missing" and "broken" call for different fixes.
            logger.warning(
                "Package %r is installed but failed to import: %s: %s",
                name,
                type(exc).__name__,
                exc,
            )
            available = False

        cls._package_availability[name] = available
        return available

    @classmethod
    def get(cls, name: str, default: Any = "_SENTINEL") -> Any:
        """Resolve a service. Static only — no auto-wiring, no dynamic imports."""
        resolved_name = cls._resolve_name(name)

        # Recursion Guard
        resolving = cls._resolving_var.get()
        if resolved_name in resolving:
            logger.warning("🔄 Circular check hit for '%s' in static registry. Returning None/Default.", resolved_name)
            if default != "_SENTINEL":
                return default
            raise CircularDependencyError(f"Circular dependency detected while resolving '{resolved_name}'")

        # 1. Fast Path (Already Initialized)
        with cls._lock:
            desc = cls._services.get(resolved_name)
            if desc and desc.lifetime == ServiceLifetime.SINGLETON and desc.instance is not None and desc.initialized:
                return desc.instance
            if not desc:
                if default != "_SENTINEL":
                    cls._emit_absent_event_once(resolved_name)
                    return default
                raise cls._absent_service_error(resolved_name)

        try:
            cls._assert_runtime_initialization_allowed(resolved_name)
        except ContainerError:
            if default != "_SENTINEL":
                return default
            raise

        # 2. Initialization Path (Per-service Lock)
        with cls._lock:
            if resolved_name not in cls._init_locks:
                cls._init_locks[resolved_name] = threading.Lock()
            service_lock = cls._init_locks[resolved_name]

        with service_lock:
            # Double-check after lock
            with cls._lock:
                desc = cls._services.get(resolved_name)

            if desc is None:
                if default != "_SENTINEL":
                    cls._emit_absent_event_once(resolved_name)
                    return default
                raise cls._absent_service_error(resolved_name)

            if desc.lifetime == ServiceLifetime.SINGLETON and desc.instance is not None and desc.initialized:
                return desc.instance

            try:
                cls._assert_runtime_initialization_allowed(resolved_name)
            except ContainerError:
                if default != "_SENTINEL":
                    return default
                raise

            # Circular Dependency Check
            resolving = cls._resolving_var.get()
            if resolved_name in resolving:
                raise CircularDependencyError(f"Circular dependency detected while resolving '{resolved_name}'")
            token = cls._resolving_var.set(resolving | {resolved_name})

            try:
                # Zenith Protocol: No Mycelium pulses, no inspect.signature auto-wiring.
                # We still support legacy dependency injection contracts for compatibility.
                args, kwargs = cls._build_factory_call(resolved_name, desc)
                instance = desc.factory(*args, **kwargs)

                # Sync on_start hook (Zenith prefers async, but support for legacy)
                start_hook = _callable_attr(instance, "on_start")
                if start_hook is not None and not desc.initialized:
                    start_result = start_hook()
                    if inspect.isawaitable(start_result):
                        close = getattr(start_result, "close", None)
                        if callable(close):
                            close()
                        raise LifecycleError(
                            f"Service '{resolved_name}' on_start returned an awaitable; use on_start_async"
                        )

                if desc.lifetime == ServiceLifetime.SINGLETON:
                    desc.instance = instance
                    desc.initialized = True
                    _offer_to_the_shed_order(resolved_name, instance)

                return instance
            except (CircularDependencyError, ServiceNotFoundError):
                raise
            except _SERVICE_INIT_ERRORS as exc:
                record_degradation('container', exc)
                raise LifecycleError(f"Service '{resolved_name}' failed to initialize: {exc}") from exc
            finally:
                cls._resolving_var.reset(token)

    @classmethod
    def peek(cls, name: str, default: Any = "_SENTINEL") -> Any:
        """Return an initialized singleton instance without triggering factory creation."""
        resolved_name = cls._resolve_name(name)
        with cls._lock:
            desc = cls._services.get(resolved_name)
            if desc and desc.lifetime == ServiceLifetime.SINGLETON and desc.instance is not None and desc.initialized:
                return desc.instance
        if default != "_SENTINEL":
            return default
        raise ServiceNotFoundError(
            f"Service '{resolved_name}' has no initialized singleton instance."
        )

    @classmethod
    def get_service(cls, name: str, default: Any = "_SENTINEL") -> Any:
        """Legacy alias for get()."""
        return cls.get(name, default=default)

    @classmethod
    def require(cls, name: str) -> Any:
        """Resolve a service and fail loudly if it is missing or unavailable."""
        service = cls.get(name)
        if service is None:
            resolved_name = cls._resolve_name(name)
            raise ServiceNotFoundError(
                f"Service '{resolved_name}' resolved to None in static registry."
            )
        return service

    @classmethod
    def validate(cls) -> tuple[bool, list[str]]:
        """Check that required dependencies are registered without instantiating services."""
        errors: list[str] = []
        with cls._lock:
            descriptors = list(cls._services.items())

        for name, desc in descriptors:
            for dep_name in cls._infer_dependency_names(desc):
                resolved_dep = cls._resolve_name(dep_name)
                if not cls.has(resolved_dep):
                    errors.append(f"Service '{name}' missing dependency '{dep_name}'")

        return not errors, errors

    @classmethod
    async def wake(cls) -> list[str]:
        """EAGER WAKE: Lock registration and initialize all required services."""
        if cls._wake_lock is None:
            cls._wake_lock = RobustLock("ServiceContainer.WakeLock")
        
        await cls._wake_lock.acquire_robust(timeout=10.0)
        try:
            cls._registration_locked = True
            cls._start_time = time.monotonic()
            logger.info("🔒 ServiceContainer registration LOCKED (Zenith static mode)")
            
            for name, desc in cls._services.items():
                if desc.lifetime == ServiceLifetime.SINGLETON:
                    try:
                        instance = cls.get(name)
                        start_async = _callable_attr(instance, "on_start_async")
                        if start_async is not None and not desc._async_initialized:
                            result = start_async()
                            if inspect.isawaitable(result):
                                await result
                            elif result is not None:
                                logger.debug(
                                    "on_start_async for %s returned non-awaitable %r",
                                    name,
                                    type(result).__name__,
                                )
                            desc._async_initialized = True
                        logger.info("   [✓] %s online.", name)
                    except _SERVICE_INIT_ERRORS as e:
                        record_degradation('container', e)
                        logger.critical("   [!] %s FAILED: %s", name, e)
                        raise ContainerError(f"Wake failed for {name}: {e}") from e

            try:
                seal = cls.write_sovereignty_seal()
                logger.info("🔒 ServiceContainer sovereignty seal written — %s", seal.get("hash", "")[:12])
            except _SEAL_IO_ERRORS as seal_exc:
                record_degradation('container', seal_exc)
                logger.warning("ServiceContainer sovereignty seal write failed: %s", seal_exc)
            
            return list(cls._services.keys())
        finally:
            if cls._wake_lock:
                cls._wake_lock.release()

    @classmethod
    async def shutdown(
        cls,
        *,
        hook_timeout_s: float = 5.0,
        total_timeout_s: float = 45.0,
        exclude: Collection[str] | None = None,
    ) -> dict[str, Any]:
        """Cleanup singleton services in reverse order and return durable evidence."""
        def _resolve_hook(instance: Any, hook_name: str) -> Callable[..., Any] | None:
            return _callable_attr(instance, hook_name)

        async def _invoke_bounded(
            name: str,
            hook_name: str,
            hook: Callable[..., Any],
            remaining: float,
            *,
            hook_timeout_override_s: float | None = None,
        ) -> str | None:
            effective_hook_timeout_s = hook_timeout_s
            if hook_timeout_override_s is not None:
                effective_hook_timeout_s = max(float(hook_timeout_override_s), float(hook_timeout_s))
            timeout_s = max(0.05, min(float(effective_hook_timeout_s), remaining))
            started = time.monotonic()
            from core.utils.task_tracker import (
                begin_shutdown_task_creation_scope,
                end_shutdown_task_creation_scope,
            )

            shutdown_scope_token = begin_shutdown_task_creation_scope()
            try:
                if inspect.iscoroutinefunction(hook):
                    result = hook()
                else:
                    result = await run_sync_shutdown_callable(
                        hook,
                        timeout_s=timeout_s,
                        name=f"container:{name}:{hook_name}",
                    )
                if inspect.isawaitable(result):
                    awaitable_budget = max(
                        0.05,
                        min(timeout_s - (time.monotonic() - started), remaining),
                    )
                    await asyncio.wait_for(result, timeout=awaitable_budget)
            except TimeoutError as exc:
                record_degradation(
                    "container",
                    exc,
                    action=f"bounded {hook_name} timeout for service '{name}' during shutdown",
                    severity="degraded",
                )
                logger.warning("%s timed out for %s after %.2fs", hook_name, name, timeout_s)
                return f"timeout_after_{timeout_s:.3f}s"
            except asyncio.CancelledError as exc:
                current_task = asyncio.current_task()
                if current_task is None or current_task.cancelling():
                    raise
                failure = "hook_cancelled_without_container_cancellation"
                record_degradation(
                    "container",
                    exc,
                    action=(
                        f"continued service teardown after {hook_name} for '{name}' "
                        "propagated cancellation from owned work"
                    ),
                    severity="degraded",
                )
                logger.error(
                    "%s for %s propagated child cancellation; continuing container shutdown",
                    hook_name,
                    name,
                )
                return failure
            except Exception as exc:  # noqa: BLE001 - final service teardown boundary
                record_degradation('container', exc)
                logger.error("%s failed for %s: %s", hook_name, name, exc)
                return repr(exc)
            finally:
                end_shutdown_task_creation_scope(shutdown_scope_token)
            return None

        excluded = {str(name) for name in (exclude or ())}
        with cls._lock:
            names = list(reversed(list(cls._services.keys())))
            descriptors = [
                (n, cls._services.get(n)) for n in names if n not in excluded
            ]
            # Runtime hygiene observes threads/processes owned by other
            # services. Stop it after owners have had their own shutdown hooks
            # so it audits true leftovers instead of racing live subsystems.
            runtime_hygiene_descriptors = [
                item for item in descriptors if item[0] == "runtime_hygiene"
            ]
            if runtime_hygiene_descriptors:
                descriptors = [
                    item for item in descriptors if item[0] != "runtime_hygiene"
                ] + runtime_hygiene_descriptors

        shutdown_started = time.monotonic()
        completed_services: list[str] = []
        failures: dict[str, str] = {}
        skipped_services: list[str] = []
        services_without_shutdown_hook: list[str] = []
        cleanup_owner_by_instance: dict[int, str] = {}
        coalesced_aliases: dict[str, str] = {}
        for descriptor_index, (name, desc) in enumerate(descriptors):
            if not desc or not desc.instance:
                continue
            remaining_total = total_timeout_s - (time.monotonic() - shutdown_started)
            if remaining_total <= 0:
                record_degradation(
                    "container",
                    TimeoutError("ServiceContainer shutdown budget exhausted"),
                    action="stopped container shutdown after bounded total budget was exhausted",
                    severity="degraded",
                )
                logger.warning("ServiceContainer shutdown budget exhausted; remaining services left cold by process exit.")
                skipped_services.extend(
                    pending_name
                    for pending_name, pending_desc in descriptors[descriptor_index:]
                    if pending_desc is not None and pending_desc.instance is not None
                )
                break
            instance = desc.instance
            instance_key = id(instance)
            existing_owner = cleanup_owner_by_instance.get(instance_key)
            if existing_owner is not None:
                coalesced_aliases[name] = existing_owner
                desc.instance = None
                desc.initialized = False
                desc._async_initialized = False
                completed_services.append(name)
                continue
            cleanup_owner_by_instance[instance_key] = name
            service_shutdown_timeout_s = hook_timeout_s
            timeout_attr = getattr(instance, "shutdown_timeout_s", None)
            if timeout_attr is not None:
                try:
                    service_shutdown_timeout_s = max(float(timeout_attr), float(hook_timeout_s))
                except (TypeError, ValueError):
                    service_shutdown_timeout_s = hook_timeout_s
            service_failed = False
            selected_hook: tuple[str, Callable[..., Any]] | None = None
            incompatible_hooks: list[str] = []
            for candidate_name in (
                "on_stop_async",
                "on_stop",
                "cleanup",
                "stop",
                "close",
            ):
                candidate = _resolve_hook(instance, candidate_name)
                if candidate is None:
                    continue
                try:
                    inspect.signature(candidate).bind()
                except TypeError:
                    incompatible_hooks.append(candidate_name)
                    continue
                except (ValueError, AttributeError):
                    pass
                selected_hook = (candidate_name, candidate)
                break
            if selected_hook is not None:
                hook_name, hook_fn = selected_hook
                remaining_total = total_timeout_s - (
                    time.monotonic() - shutdown_started
                )
                failure: str | None
                if remaining_total <= 0:
                    failure = "container_total_budget_exhausted"
                else:
                    failure = await _invoke_bounded(
                        name,
                        hook_name,
                        hook_fn,
                        remaining_total,
                        hook_timeout_override_s=service_shutdown_timeout_s,
                    )
                if failure is not None:
                    failures[f"{name}:{hook_name}"] = failure
                    service_failed = True
            elif incompatible_hooks:
                failure_key = f"{name}:container"
                failures[failure_key] = (
                    "no_zero_argument_shutdown_hook:"
                    + ",".join(incompatible_hooks)
                )
                service_failed = True
            else:
                services_without_shutdown_hook.append(name)

            desc.instance = None
            desc.initialized = False
            desc._async_initialized = False
            if not service_failed:
                completed_services.append(name)

        for name in skipped_services:
            failures.setdefault(f"{name}:container", "container_total_budget_exhausted")
        current_report: dict[str, Any] = {
            "clean": not failures,
            "completed_services": completed_services,
            "failed_hooks": failures,
            "skipped_services": skipped_services,
            "services_without_shutdown_hook": services_without_shutdown_hook,
            "coalesced_aliases": coalesced_aliases,
            "excluded_services": sorted(excluded),
            "duration_seconds": round(time.monotonic() - shutdown_started, 6),
            "total_timeout_seconds": float(total_timeout_s),
            "hook_timeout_seconds": float(hook_timeout_s),
        }
        with cls._lock:
            cls._shutdown_reports.append(dict(current_report))
            cls._shutdown_reports = cls._shutdown_reports[-8:]
            reports = list(cls._shutdown_reports)
        aggregate_failures: dict[str, str] = {}
        aggregate_skipped: list[str] = []
        aggregate_completed: list[str] = []
        aggregate_deferred: list[str] = []
        aggregate_without_hook: list[str] = []
        aggregate_coalesced: dict[str, str] = {}
        for report in reports:
            aggregate_failures.update(
                {
                    str(key): str(value)
                    for key, value in dict(report.get("failed_hooks", {})).items()
                }
            )
            aggregate_skipped.extend(
                str(name) for name in report.get("skipped_services", [])
            )
            aggregate_completed.extend(
                str(name) for name in report.get("completed_services", [])
            )
            aggregate_deferred.extend(
                str(name) for name in report.get("excluded_services", [])
            )
            aggregate_without_hook.extend(
                str(name)
                for name in report.get("services_without_shutdown_hook", [])
            )
            aggregate_coalesced.update(
                {
                    str(alias): str(owner)
                    for alias, owner in dict(
                        report.get("coalesced_aliases", {})
                    ).items()
                }
            )
        current_report["clean"] = not aggregate_failures
        current_report["failed_hooks"] = aggregate_failures
        current_report["skipped_services"] = sorted(set(aggregate_skipped))
        current_report["completed_services"] = sorted(set(aggregate_completed))
        current_report["deferred_services"] = sorted(set(aggregate_deferred))
        current_report["services_without_shutdown_hook"] = sorted(
            set(aggregate_without_hook)
        )
        current_report["coalesced_aliases"] = aggregate_coalesced
        current_report["shutdown_pass_count"] = len(reports)
        with cls._lock:
            cls._shutdown_reports[-1] = dict(current_report)
        return current_report

    @classmethod
    def get_shutdown_report(cls) -> dict[str, Any] | None:
        with cls._lock:
            if not cls._shutdown_reports:
                return None
            return dict(cls._shutdown_reports[-1])

    @classmethod
    def get_health_report(cls) -> dict[str, Any]:
        """Generate a health report for all registered services."""
        report: dict[str, Any] = {
            "status": "operational",
            "uptime_seconds": round(time.monotonic() - (cls._start_time or time.monotonic()), 2),
            "services": {},
            "sovereignty_seal": {
                "present": False,
                "valid": True,
                "hash": cls._last_seal_hash,
            },
        }
        with cls._lock:
            for name, d in cls._services.items():
                report["services"][name] = {
                    "status": "online" if d.initialized else "offline",
                    "required": d.required,
                    "lifetime": d.lifetime.value
                }
                if d.required and not d.initialized:
                    report["status"] = "degraded"

        try:
            seal_path = cls._seal_path()
            seal_valid = cls.verify_sovereignty_seal()
            seal_state = cls.sovereignty_seal_state()
            report["sovereignty_seal"] = {
                "present": seal_path.exists(),
                # None, not True, when there is no seal: nothing was verified.
                "valid": seal_valid if seal_state != "unsealed" else None,
                "state": seal_state,
                "covers": "registry_shape_only",
                "hash": cls._last_seal_hash,
            }
            if not seal_valid:
                report["status"] = "degraded"
        except _SEAL_IO_ERRORS as exc:
            record_degradation('container', exc)
            logger.debug("ServiceContainer health seal verification failed: %s", exc)
        return report

    @classmethod
    def _seal_path(cls) -> Path:
        try:
            from core.config import config

            return Path(config.paths.data_dir) / "sovereignty_seal.json"
        except (ImportError, AttributeError, RuntimeError, OSError) as exc:
            record_degradation("container", exc)
            logger.debug("Falling back to default sovereignty seal path after config lookup failed: %s", exc)
            return Path(state_root()) / "data" / "sovereignty_seal.json"

    @classmethod
    def _manifest_snapshot(cls) -> dict[str, str]:
        with cls._lock:
            descriptors = dict(cls._services)
        manifest: dict[str, str] = {}
        for name, desc in descriptors.items():
            instance = desc.instance
            if instance is not None:
                manifest[name] = instance.__class__.__name__
            else:
                manifest[name] = getattr(desc.factory, "__qualname__", repr(desc.factory))
        return manifest

    @classmethod
    def _seal_key(cls) -> bytes | None:
        """Local HMAC key for the sovereignty seal, created on first use.

        None when unavailable, which the verifier treats as unsigned rather
        than valid.
        """
        path = cls._seal_path().with_name(".sovereignty_seal.key")
        try:
            if path.exists():
                key: bytes | None = path.read_bytes()
                return key if key is not None and len(key) == 32 else None
            candidate = os.urandom(32)
            with local_internal_governed_scope(
                "service_container.sovereignty_seal_key",
                domain="file_write",
            ):
                # Annotated above because the gateway is not followed by the
                # ratchet's mypy, so its return reads as Any here.
                key = get_file_write_gateway().provision_private_bytes(
                    path,
                    candidate,
                    expected_size=32,
                    mode=0o600,
                    source="service_container.sovereignty_seal_key",
                )
            return key
        except (OSError, ValueError):
            return None

    @classmethod
    def _seal_signature(cls, digest: str, service_count: int) -> str:
        key = cls._seal_key()
        if key is None:
            return ""
        body = f"{digest}:{service_count}".encode()
        return hmac.new(key, body, hashlib.sha256).hexdigest()

    @classmethod
    def write_sovereignty_seal(cls) -> dict[str, Any]:
        # CP126 (critical): "Sovereignty seal is an unsigned registry-name
        # hash. The seal covers descriptor/factory names rather than source
        # bytes, dependencies, configuration, policies, or runtime identity
        # and is stored beside its manifest without a signature."
        #
        # The unsigned part is the half that can be fixed here: anything able
        # to write the seal file could recompute the hash for a tampered
        # registry and the seal would verify. It now carries an HMAC over the
        # digest, so forging it needs the local key rather than a hashlib
        # call.
        #
        # The other half stands as stated: this seals registry SHAPE — which
        # service names map to which class names — not source bytes,
        # dependencies or policy. A same-named impostor class still passes.
        # That is a real limit and it is documented rather than implied away.
        manifest = cls._manifest_snapshot()
        digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()
        payload = {
            "hash": digest,
            "timestamp": time.time(),
            "service_count": len(manifest),
            "manifest": manifest,
            "signature": cls._seal_signature(digest, len(manifest)),
            "covers": "registry_shape_only",
        }
        seal_path = cls._seal_path()
        atomic_write_text(seal_path, json.dumps(payload, sort_keys=True, indent=2))
        cls._last_seal_hash = digest
        return payload

    @classmethod
    def verify_sovereignty_seal(cls) -> bool:
        seal_path = cls._seal_path()
        if not seal_path.exists():
            return True
        try:
            stored = json.loads(seal_path.read_text())
        except _SEAL_IO_ERRORS as exc:
            record_degradation("container", exc)
            logger.debug("Sovereignty seal read failed: %s", exc)
            return False
        manifest = cls._manifest_snapshot()
        current = hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode("utf-8")
        ).hexdigest()
        cls._last_seal_hash = current
        if str(stored.get("hash", "")) != current:
            return False
        # A matching hash proves the registry is unchanged; the signature
        # proves the seal itself was not rewritten to match a tampered one.
        stored_signature = str(stored.get("signature", "") or "")
        expected_signature = cls._seal_signature(
            current, int(stored.get("service_count") or len(manifest))
        )
        if not expected_signature:
            logger.warning("Sovereignty seal cannot be verified: no local key.")
            return False
        return bool(stored_signature) and hmac.compare_digest(
            stored_signature, expected_signature
        )

    @classmethod
    def sovereignty_seal_state(cls) -> str:
        """'valid' | 'invalid' | 'unsealed'.

        verify_sovereignty_seal() returns True when no seal exists — there is
        nothing to contradict, and first boot must not be degraded. But the
        health report published that as valid=True, so an absent seal read as
        a verified one: absence of a check reported as a passed check. The
        report now uses this instead.
        """
        if not cls._seal_path().exists():
            return "unsealed"
        return "valid" if cls.verify_sovereignty_seal() else "invalid"

    @classmethod
    def write_service_ownership_manifest(cls, project_root: Path) -> Path:
        with cls._lock:
            items = sorted(list(cls._services.items()))
        
        lines = [
            "# Aura Subsystem and Service Ownership Manifest",
            "",
            "This file outlines every registered service, its source code location, registration origin, failure policy, and operational requirements.",
            "",
            "| Service | Owner File | Registered By | Required For | Failure Policy |",
            "|---|---|---|---|---|",
        ]
        for name, desc in items:
            lines.append(
                f"| `{name}` | `{getattr(desc, 'owner', 'unknown')}` | `{getattr(desc, 'registered_by', 'unknown')}` | {getattr(desc, 'required_for', 'general utility')} | `{getattr(desc, 'failure_policy', 'degrade_with_receipt')}` |"
            )
        
        project_root.mkdir(parents=True, exist_ok=True)
        path = project_root / "SERVICE_OWNERSHIP.md"
        atomic_write_text(path, "\n".join(lines) + "\n")
        return path

    @classmethod
    def _emit_absent_event(cls, service_name: str) -> None:
        """Emit a quiet breadcrumb when an explicitly optional service is absent.

        Callers that pass a default are declaring the lookup optional. Treat that
        as diagnostic context, not a live degradation, so UI/status probes do not
        pollute the neural feed with false subsystem failures.
        """
        logger.debug("Optional service absent from static registry: %s", service_name)

    @classmethod
    def _emit_absent_event_once(cls, service_name: str) -> None:
        """Record one optional-absence breadcrumb per service per process."""

        with cls._lock:
            if service_name in cls._optional_absent_breadcrumbs:
                return
            cls._optional_absent_breadcrumbs.add(service_name)
        cls._emit_absent_event(service_name)


def _install_runtime_service_registry_bridge() -> None:
    """Expose read-only service policy metadata to low-level runtime modules."""

    try:
        from core.runtime.service_registry import (
            install_container_health_report_resolver,
            install_failure_policy_resolver,
            install_registration_locked_resolver,
            install_service_factory_registration_sink,
            install_service_presence_resolver,
            install_service_registration_sink,
            install_service_resolver,
        )

        def _failure_policy_for(service_name: str) -> str | None:
            resolved = ServiceContainer._resolve_name(service_name)
            with ServiceContainer._lock:
                desc = ServiceContainer._services.get(resolved)
            if desc is None:
                return None
            return getattr(desc, "failure_policy", None)

        def _service_for(service_name: str, default: object | None = None) -> object | None:
            # Low-level observers and error sinks may inspect runtime services,
            # but they are not lifecycle owners. A read here must never invoke
            # a factory and recursively boot an organ during diagnostics.
            result: object | None = ServiceContainer.peek(service_name, default=default)
            return result

        def _has_service(service_name: str) -> bool:
            return ServiceContainer.has(service_name)

        def _registration_locked() -> bool:
            return bool(getattr(ServiceContainer, "_registration_locked", False))

        def _health_report() -> dict[str, object]:
            return ServiceContainer.get_health_report()

        def _register_service(
            service_name: str,
            instance: object,
            required: bool,
            metadata: dict[str, str | None],
        ) -> None:
            ServiceContainer.register_instance(
                service_name,
                instance,
                required=required,
                owner=metadata.get("owner"),
                registered_by=metadata.get("registered_by"),
                required_for=metadata.get("required_for"),
                failure_policy=metadata.get("failure_policy"),
            )

        def _register_factory(
            service_name: str,
            factory: Callable[..., object],
            lifetime: object | None,
            required: bool,
            metadata: dict[str, str | None],
        ) -> None:
            resolved_lifetime: ServiceLifetime
            if isinstance(lifetime, ServiceLifetime):
                resolved_lifetime = lifetime
            elif lifetime is None:
                resolved_lifetime = ServiceLifetime.SINGLETON
            elif isinstance(lifetime, str):
                try:
                    resolved_lifetime = ServiceLifetime(lifetime)
                except ValueError:
                    resolved_lifetime = ServiceLifetime.SINGLETON
            else:
                resolved_lifetime = ServiceLifetime.SINGLETON

            ServiceContainer.register(
                service_name,
                factory=factory,
                lifetime=resolved_lifetime,
                required=required,
                owner=metadata.get("owner"),
                registered_by=metadata.get("registered_by"),
                required_for=metadata.get("required_for"),
                failure_policy=metadata.get("failure_policy"),
            )

        install_failure_policy_resolver(_failure_policy_for)
        install_service_resolver(_service_for)
        install_service_presence_resolver(_has_service)
        install_service_registration_sink(_register_service)
        install_service_factory_registration_sink(_register_factory)
        install_container_health_report_resolver(_health_report)
        install_registration_locked_resolver(_registration_locked)
    except (ImportError, RuntimeError, AttributeError) as exc:
        logger.debug("Runtime service registry bridge unavailable: %s", exc)


_install_runtime_service_registry_bridge()


def get_container() -> type[ServiceContainer]:
    return ServiceContainer
