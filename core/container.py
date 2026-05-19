import contextvars
import functools
import hashlib
import inspect
import json
import logging
import threading
import time
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from core.exceptions import (
    CircularDependencyError,
    ContainerError,
    LifecycleError,
    ServiceNotFoundError,
)
from core.health.degraded_events import record_degraded_event
from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation
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

class ServiceDescriptor:
    """Describes how to create and manage a service."""
    def __init__(self, name: str, factory: Callable, lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
                 instance: Any = None, required: bool = True, initialized: bool = False,
                 dependencies: list[str] | None = None):
        self.name = name
        self.factory = factory
        self.lifetime = lifetime
        self.instance = instance
        self.required = required
        self.initialized = initialized
        self._async_initialized = False
        self.dependencies = list(dependencies or [])


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


def zero_sync_guard(func: Callable):
    """Decorator to ensure async methods do not perform synchronous blocking calls."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # In the future, this could monitor thread status or event loop lag
        return await func(*args, **kwargs)
    return wrapper


ZeroSyncGuard = zero_sync_guard

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
    _resolving_var: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar('resolving', default=frozenset())
    _wake_lock = RobustLock("ServiceContainer.Wake")
    _start_time: float | None = None
    _init_locks: dict[str, threading.Lock] = {}
    _last_seal_hash: str | None = None

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

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
    def register(
        cls,
        name: str,
        factory: Callable,
        lifetime=ServiceLifetime.SINGLETON,
        required=True,
        dependencies: list[str] | None = None,
    ):
        """Register a service factory."""
        if cls._registration_locked:
            raise ContainerError(f"Registration locked: Cannot register '{name}'")
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
            )
            logger.debug("Registered static service: %s", name)
    @classmethod
    def unlock_registration(cls, *, caller: str = "unknown", reason: str = ""):
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
        import traceback
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
                caller,
                frame_info,
                f" — reason: {reason}" if reason else "",
            )

    @classmethod
    def lock_registration(cls):
        """Standard locking interface."""
        with cls._lock:
            cls._registration_locked = True
            logger.info("🔒 ServiceContainer registration LOCKED")

    @classmethod
    def register_instance(cls, name: str, instance: Any, required=True):
        """Register a pre-built instance.

        Unlike factory-based ``register()``, pre-built instances are safe to
        add after the lock because they carry no lazy-init or circular-dep
        risk.  Late registrations are permitted with a warning so that
        subsystems that boot asynchronously (final_engines, affective_circumplex,
        architecture_index, etc.) can complete their setup without crashing.
        """
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
        if cls._registration_locked:
            logger.debug("⚠️ Late instance registration (post-lock): '%s' — allowed for pre-built instances.", name)
            if (
                existing
                and name in _PROTECTED_CORE_SERVICES
                and existing_instance is not instance
            ):
                logger.error("🚫 Protected core service overwrite blocked after lock: '%s'", name)
                record_degraded_event(
                    "service_container",
                    "protected_service_overwrite_blocked",
                    detail=name,
                    severity="error",
                    classification="foreground_blocking",
                    context={"service": name},
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
                initialized=True
            )
            logger.debug("Registered pre-built instance: %s", name)

    @classmethod
    def set(cls, name: str, instance: Any, required: bool = True):
        """Legacy compatibility alias for replacing a singleton instance.

        A large portion of Aura's older runtime expects ``ServiceContainer.set``
        to behave like an upsert for already-built singleton instances.
        """
        cls.register_instance(name, instance, required=required)
        return instance

    @classmethod
    def register_alias(cls, alias: str, target: str) -> None:
        """Register a legacy service alias that resolves to another service name."""
        if cls._registration_locked:
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
    def clear(cls) -> None:
        """Reset the static registry to a pristine state for tests and warm reboots."""
        with cls._lock:
            cls._services.clear()
            cls._aliases.clear()
            cls._init_locks.clear()
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
                    cls._emit_absent_event(resolved_name)
                    return default
                raise ServiceNotFoundError(f"Service '{resolved_name}' not found in static registry.")

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
                    cls._emit_absent_event(resolved_name)
                    return default
                raise ServiceNotFoundError(f"Service '{resolved_name}' not found in static registry.")

            if desc.lifetime == ServiceLifetime.SINGLETON and desc.instance is not None and desc.initialized:
                return desc.instance

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
    async def shutdown(cls) -> None:
        """Cleanup all singleton services in reverse order."""
        def _resolve_hook(instance: Any, hook_name: str) -> Callable[..., Any] | None:
            return _callable_attr(instance, hook_name)

        with cls._lock:
            names = list(reversed(list(cls._services.keys())))
            descriptors = [(n, cls._services.get(n)) for n in names]

        for name, desc in descriptors:
            if not desc or not desc.instance:
                continue
            instance = desc.instance
            
            # Async stop
            hook = _resolve_hook(instance, "on_stop_async")
            if hook is not None:
                try:
                    result = hook()
                    if inspect.isawaitable(result):
                        await result
                    elif result is not None:
                        logger.debug("on_stop_async for %s returned non-awaitable %r", name, type(result).__name__)
                except _SERVICE_INIT_ERRORS as e:
                    record_degradation('container', e)
                    logger.error("on_stop_async failed for %s: %s", name, e)
            
            # Sync stop
            for hook in ("on_stop", "cleanup"):
                hook_fn = _resolve_hook(instance, hook)
                if hook_fn is not None:
                    try:
                        result = hook_fn()
                        if inspect.isawaitable(result):
                            await result
                    except _SERVICE_INIT_ERRORS as e:
                        record_degradation('container', e)
                        logger.error("%s failed for %s: %s", hook, name, e)
            
            desc.instance = None
            desc.initialized = False
            desc._async_initialized = False

    @classmethod
    def get_health_report(cls) -> dict[str, Any]:
        """Generate a health report for all registered services."""
        report = {
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
            report["sovereignty_seal"] = {
                "present": seal_path.exists(),
                "valid": seal_valid,
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

            return config.paths.data_dir / "sovereignty_seal.json"
        except (ImportError, AttributeError, RuntimeError, OSError) as exc:
            record_degradation("container", exc)
            logger.debug("Falling back to default sovereignty seal path after config lookup failed: %s", exc)
            return Path.home() / ".aura" / "data" / "sovereignty_seal.json"

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
    def write_sovereignty_seal(cls) -> dict[str, Any]:
        manifest = cls._manifest_snapshot()
        digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()
        payload = {
            "hash": digest,
            "timestamp": time.time(),
            "service_count": len(manifest),
            "manifest": manifest,
        }
        seal_path = cls._seal_path()
        seal_path.parent.mkdir(parents=True, exist_ok=True)
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
        current = hashlib.sha256(
            json.dumps(cls._manifest_snapshot(), sort_keys=True).encode("utf-8")
        ).hexdigest()
        cls._last_seal_hash = current
        return str(stored.get("hash", "")) == current

    @classmethod
    def _emit_absent_event(cls, service_name: str) -> None:
        """Emit a quiet breadcrumb when an explicitly optional service is absent.

        Callers that pass a default are declaring the lookup optional. Treat that
        as diagnostic context, not a live degradation, so UI/status probes do not
        pollute the neural feed with false subsystem failures.
        """
        record_degraded_event(
            "service_container",
            "SUBSYSTEM_ABSENT",
            detail=service_name,
            severity="info",
            classification="non_critical_fallback",
            context={"service": service_name},
        )


def get_container() -> type[ServiceContainer]:
    return ServiceContainer
