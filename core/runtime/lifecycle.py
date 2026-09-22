"""core/runtime/lifecycle.py — managed lifecycles for every organ.

Clean-room adoption of ROS 2's managed (lifecycle) nodes.

The problem ROS 2 solved with this is exactly Aura's boot problem. A
system of many independent components, each of which constructs itself,
finds its dependencies, and starts doing work in its constructor, has no
moment at which the system is *configured but not yet running*. So there
is no way to bring everything up, check that it is all there, and only
then let it act. Every start is a race, every failure is partial, and
"is the system ready" has no answer better than "probably".

Managed nodes give components a small, explicit state machine:

    unconfigured --configure--> inactive --activate--> active
         ^                          ^                    |
         |                          +----deactivate------+
         +---------cleanup----------+
    (any state) --shutdown--> finalized

The two properties that make it worth adopting:

1. **`inactive` is a real state.** A configured organ has allocated what
   it needs, resolved its dependencies, and published nothing. The system
   can verify every organ reached `inactive` before activating any of
   them — which converts a boot race into a boot checkpoint.
2. **Transitions are the only place work happens, and they can fail.**
   `on_configure` returning failure leaves the organ `unconfigured` and
   says so, rather than leaving a half-built object that looks fine until
   its first call. A transition that raises goes to `errorprocessing`,
   which is a state, not an exception nobody caught.

Aura already had the ingredients scattered — boot phases, a health
contract, an activation audit, an organ supervisor — but no organ could
say "I am configured and deliberately not running". Deactivation was
indistinguishable from failure. This makes that distinction real, which
is what lets the runtime pause an organ (under memory pressure, during a
degraded mode, while a model reloads) and later resume it without a
restart, and lets `nothing orphaned` be checked rather than hoped for.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.Lifecycle")


class State(StrEnum):
    UNCONFIGURED = "unconfigured"
    INACTIVE = "inactive"
    ACTIVE = "active"
    FINALIZED = "finalized"
    #: A transition is in flight. Visible so a wedged transition is
    #: distinguishable from a slow one.
    CONFIGURING = "configuring"
    ACTIVATING = "activating"
    DEACTIVATING = "deactivating"
    CLEANING_UP = "cleaning_up"
    SHUTTING_DOWN = "shutting_down"
    ERROR_PROCESSING = "error_processing"


class Transition(StrEnum):
    CONFIGURE = "configure"
    ACTIVATE = "activate"
    DEACTIVATE = "deactivate"
    CLEANUP = "cleanup"
    SHUTDOWN = "shutdown"


#: (from-state, transition) -> (in-flight state, success state)
_TRANSITION_TABLE: dict[tuple[State, Transition], tuple[State, State]] = {
    (State.UNCONFIGURED, Transition.CONFIGURE): (State.CONFIGURING, State.INACTIVE),
    (State.INACTIVE, Transition.ACTIVATE): (State.ACTIVATING, State.ACTIVE),
    (State.ACTIVE, Transition.DEACTIVATE): (State.DEACTIVATING, State.INACTIVE),
    (State.INACTIVE, Transition.CLEANUP): (State.CLEANING_UP, State.UNCONFIGURED),
    (State.UNCONFIGURED, Transition.SHUTDOWN): (State.SHUTTING_DOWN, State.FINALIZED),
    (State.INACTIVE, Transition.SHUTDOWN): (State.SHUTTING_DOWN, State.FINALIZED),
    (State.ACTIVE, Transition.SHUTDOWN): (State.SHUTTING_DOWN, State.FINALIZED),
}

#: Where a failed transition lands. Failure is a *result*, not an
#: exception: the organ stays in a state the system can reason about.
_FAILURE_STATE: dict[Transition, State] = {
    Transition.CONFIGURE: State.UNCONFIGURED,
    Transition.ACTIVATE: State.INACTIVE,
    Transition.DEACTIVATE: State.ACTIVE,
    Transition.CLEANUP: State.INACTIVE,
    Transition.SHUTDOWN: State.FINALIZED,
}


@dataclass(frozen=True)
class TransitionRecord:
    at: float
    organ: str
    transition: Transition
    from_state: State
    to_state: State
    ok: bool
    duration_s: float
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "at": self.at,
            "organ": self.organ,
            "transition": str(self.transition),
            "from": str(self.from_state),
            "to": str(self.to_state),
            "ok": self.ok,
            "duration_s": round(self.duration_s, 4),
            "detail": self.detail,
        }


class TransitionError(RuntimeError):
    """Raised only for transitions that are not legal from the current state."""


Hook = Callable[[], Any]


class ManagedOrgan:
    """One component with a declared lifecycle.

    Subclass and override the ``on_*`` hooks, or pass them in. A hook may
    be sync or async; returning ``False`` means the transition failed.
    """

    def __init__(
        self,
        name: str,
        *,
        on_configure: Hook | None = None,
        on_activate: Hook | None = None,
        on_deactivate: Hook | None = None,
        on_cleanup: Hook | None = None,
        on_shutdown: Hook | None = None,
        on_error: Hook | None = None,
        transition_timeout_s: float = 30.0,
        critical: bool = False,
    ) -> None:
        self.name = name
        self.transition_timeout_s = transition_timeout_s
        #: A critical organ that cannot reach `active` blocks system readiness.
        self.critical = critical
        self._state = State.UNCONFIGURED
        self._lock = asyncio.Lock()
        self._state_lock = threading.Lock()
        self._history: list[TransitionRecord] = []
        self._hooks: dict[Transition, Hook | None] = {
            Transition.CONFIGURE: on_configure,
            Transition.ACTIVATE: on_activate,
            Transition.DEACTIVATE: on_deactivate,
            Transition.CLEANUP: on_cleanup,
            Transition.SHUTDOWN: on_shutdown,
        }
        self._on_error = on_error
        self.last_error = ""
        self.entered_state_at = time.time()

    # ── overridable hooks ─────────────────────────────────────────────
    async def on_configure(self) -> bool:
        """Allocate, resolve dependencies, publish nothing."""
        return True

    async def on_activate(self) -> bool:
        """Start doing work."""
        return True

    async def on_deactivate(self) -> bool:
        """Stop doing work, keep everything allocated."""
        return True

    async def on_cleanup(self) -> bool:
        """Release what configure allocated."""
        return True

    async def on_shutdown(self) -> bool:
        """Final teardown. Never called twice."""
        return True

    async def on_error(self) -> bool:
        """Attempt recovery from the error state. False finalizes the organ.

        The default records the error transition and declines to recover:
        a base class cannot know how to repair a subclass's state, and
        guessing is how a half-repaired organ gets presented as healthy.
        An organ that CAN recover overrides this.
        """
        from core.runtime.errors import record_degradation

        record_degradation(
            f"lifecycle.{self.name}",
            RuntimeError(self.last_error or "organ entered the error state"),
            severity="warning",
            action=(
                "declined the default recovery; the organ finalizes. Override "
                "on_error() if this organ knows how to repair itself"
            ),
            enforce_failure_policy=False,
        )
        return False

    # ── state ─────────────────────────────────────────────────────────
    @property
    def state(self) -> State:
        with self._state_lock:
            return self._state

    @property
    def active(self) -> bool:
        return self.state is State.ACTIVE

    def _set_state(self, state: State) -> None:
        with self._state_lock:
            self._state = state
            self.entered_state_at = time.time()

    def can(self, transition: Transition) -> bool:
        return (self.state, transition) in _TRANSITION_TABLE

    def available_transitions(self) -> list[Transition]:
        current = self.state
        return [t for (s, t) in _TRANSITION_TABLE if s is current]

    # ── the machine ───────────────────────────────────────────────────
    async def trigger(self, transition: Transition) -> bool:
        """Run a transition. Returns success; raises only if illegal."""
        async with self._lock:
            start_state = self.state
            entry = _TRANSITION_TABLE.get((start_state, transition))
            if entry is None:
                raise TransitionError(
                    f"{self.name}: {transition} is not legal from {start_state}; "
                    f"available: {[str(t) for t in self.available_transitions()]}"
                )
            in_flight, success_state = entry
            self._set_state(in_flight)
            started = time.perf_counter()
            detail = ""
            ok = False
            try:
                ok = await asyncio.wait_for(
                    self._run_hook(transition), timeout=self.transition_timeout_s
                )
            except TimeoutError:
                detail = f"transition timed out after {self.transition_timeout_s:.0f}s"
                ok = False
            except asyncio.CancelledError:
                self._set_state(start_state)
                raise
            except Exception as exc:  # noqa: BLE001 — a raise is errorprocessing, not a crash
                detail = f"{type(exc).__name__}: {exc}"
                self.last_error = detail
                elapsed = time.perf_counter() - started
                self._set_state(State.ERROR_PROCESSING)
                self._record(transition, start_state, State.ERROR_PROCESSING, False, elapsed, detail)
                await self._handle_error(transition)
                return False

            elapsed = time.perf_counter() - started
            final = success_state if ok else _FAILURE_STATE[transition]
            if not ok and not detail:
                detail = "hook returned failure"
            if not ok:
                self.last_error = detail
            self._set_state(final)
            self._record(transition, start_state, final, ok, elapsed, detail)
            return ok

    async def _run_hook(self, transition: Transition) -> bool:
        injected = self._hooks.get(transition)
        if injected is not None:
            outcome = injected()
        else:
            method = {
                Transition.CONFIGURE: self.on_configure,
                Transition.ACTIVATE: self.on_activate,
                Transition.DEACTIVATE: self.on_deactivate,
                Transition.CLEANUP: self.on_cleanup,
                Transition.SHUTDOWN: self.on_shutdown,
            }[transition]
            outcome = method()
        if asyncio.iscoroutine(outcome):
            outcome = await outcome
        return outcome is not False

    async def _handle_error(self, transition: Transition) -> None:
        handler = self._on_error or self.on_error
        recovered = False
        try:
            outcome = handler()
            if asyncio.iscoroutine(outcome):
                outcome = await outcome
            recovered = bool(outcome)
        except Exception:  # noqa: BLE001
            logger.debug("%s error handler raised", self.name, exc_info=True)
        target = State.UNCONFIGURED if recovered else State.FINALIZED
        self._set_state(target)
        logger.warning(
            "🔄 %s error during %s: %s → %s",
            self.name,
            transition,
            self.last_error,
            target,
        )
        from core.runtime.errors import record_degradation

        with contextlib.suppress(Exception):
            record_degradation(
                f"lifecycle.{self.name}",
                RuntimeError(self.last_error or "transition error"),
                severity="degraded" if recovered else "critical",
                action=f"organ moved to {target} after failed {transition}",
                enforce_failure_policy=False,
            )

    def _record(
        self,
        transition: Transition,
        from_state: State,
        to_state: State,
        ok: bool,
        duration_s: float,
        detail: str,
    ) -> None:
        record = TransitionRecord(
            at=time.time(),
            organ=self.name,
            transition=transition,
            from_state=from_state,
            to_state=to_state,
            ok=ok,
            duration_s=duration_s,
            detail=detail,
        )
        self._history.append(record)
        if len(self._history) > 64:
            del self._history[:-64]
        # Aura.Lifecycle.TransitionMs was declared in
        # core/observability/histograms.py naming THIS file as its owner, and
        # nothing ever wrote to it. A declared histogram with no writer reads
        # empty forever, which is indistinguishable from a system that never
        # transitions — so the one question it exists to answer ("are organ
        # transitions getting slower?") could not be asked. Recorded here rather
        # than at the three call sites because this is the chokepoint they all
        # reach, including the error path.
        try:
            from core.observability.histograms import record as record_histogram

            record_histogram("Aura.Lifecycle.TransitionMs", max(0.0, duration_s) * 1000.0)
        except (ImportError, RuntimeError, ValueError, TypeError) as exc:
            # The comment above is about this histogram reading empty forever.
            # A silent except is the second way to arrive there.
            logger.debug(
                "Aura.Lifecycle.TransitionMs not recorded (%s: %s)",
                type(exc).__name__,
                exc,
            )
        logger.log(
            logging.INFO if ok else logging.WARNING,
            "%s %s: %s --%s--> %s (%.3fs)%s",
            "🟢" if ok else "🔴",
            self.name,
            from_state,
            transition,
            to_state,
            duration_s,
            f" — {detail}" if detail else "",
        )
        with contextlib.suppress(Exception):
            from core.runtime.conditions import ConditionType, get_component_conditions

            conditions = get_component_conditions(self.name)
            conditions.set(
                ConditionType.READY,
                to_state is State.ACTIVE,
                reason=str(to_state),
                message=detail or f"after {transition}",
            )

    # ── convenience ───────────────────────────────────────────────────
    async def configure(self) -> bool:
        return await self.trigger(Transition.CONFIGURE)

    async def activate(self) -> bool:
        return await self.trigger(Transition.ACTIVATE)

    async def deactivate(self) -> bool:
        return await self.trigger(Transition.DEACTIVATE)

    async def cleanup(self) -> bool:
        return await self.trigger(Transition.CLEANUP)

    async def shutdown(self) -> bool:
        if self.state is State.FINALIZED:
            return True
        return await self.trigger(Transition.SHUTDOWN)

    def report(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": str(self.state),
            "critical": self.critical,
            "active": self.active,
            "in_state_for_s": round(time.time() - self.entered_state_at, 2),
            "available_transitions": [str(t) for t in self.available_transitions()],
            "last_error": self.last_error,
            "history": [r.to_dict() for r in self._history[-8:]],
        }


class LifecycleManager:
    """Brings every organ to `inactive`, verifies, then activates.

    The two-phase bring-up is the whole point: a boot that activates each
    organ as it is constructed can only discover a missing dependency
    after something already started acting on it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._organs: dict[str, ManagedOrgan] = {}

    def register(self, organ: ManagedOrgan) -> ManagedOrgan:
        with self._lock:
            existing = self._organs.get(organ.name)
            if existing is not None:
                return existing
            self._organs[organ.name] = organ
            return organ

    def get(self, name: str) -> ManagedOrgan | None:
        with self._lock:
            return self._organs.get(name)

    def organs(self) -> list[ManagedOrgan]:
        with self._lock:
            return list(self._organs.values())

    async def configure_all(self) -> dict[str, bool]:
        return await self._run_all(Transition.CONFIGURE)

    async def activate_all(self) -> dict[str, bool]:
        return await self._run_all(Transition.ACTIVATE)

    async def deactivate_all(self) -> dict[str, bool]:
        return await self._run_all(Transition.DEACTIVATE)

    async def shutdown_all(self) -> dict[str, bool]:
        return await self._run_all(Transition.SHUTDOWN)

    async def _run_all(self, transition: Transition) -> dict[str, bool]:
        results: dict[str, bool] = {}
        for organ in self.organs():
            if not organ.can(transition):
                continue
            try:
                results[organ.name] = await organ.trigger(transition)
            except TransitionError as exc:
                logger.debug("skipping %s: %s", organ.name, exc)
        return results

    async def bring_up(self) -> dict[str, Any]:
        """Configure everything, check, then activate. The boot checkpoint."""
        configured = await self.configure_all()
        not_inactive = [
            organ.name
            for organ in self.organs()
            if organ.state not in (State.INACTIVE, State.ACTIVE, State.FINALIZED)
        ]
        critical_blocked = [
            organ.name
            for organ in self.organs()
            if organ.critical and organ.state not in (State.INACTIVE, State.ACTIVE)
        ]
        activated: dict[str, bool] = {}
        if not critical_blocked:
            activated = await self.activate_all()
        return {
            "configured": configured,
            "not_inactive": not_inactive,
            "critical_blocked": critical_blocked,
            "activated": activated,
            "ready": not critical_blocked and all(activated.values()) if activated else False,
        }

    def report(self) -> dict[str, Any]:
        organs = self.organs()
        by_state: dict[str, list[str]] = {}
        for organ in organs:
            by_state.setdefault(str(organ.state), []).append(organ.name)
        return {
            "count": len(organs),
            "by_state": by_state,
            "active": sorted(o.name for o in organs if o.active),
            "orphaned": sorted(
                o.name for o in organs if o.state is State.UNCONFIGURED
            ),
            "errored": sorted(o.name for o in organs if o.state is State.ERROR_PROCESSING),
            "critical_inactive": sorted(
                o.name for o in organs if o.critical and not o.active
            ),
            "organs": [o.report() for o in organs],
        }

    def reset_for_test(self) -> None:
        with self._lock:
            self._organs.clear()


_MANAGER = LifecycleManager()


def get_lifecycle_manager() -> LifecycleManager:
    return _MANAGER


def managed(
    name: str,
    *,
    critical: bool = False,
    transition_timeout_s: float = 30.0,
    **hooks: Hook,
) -> ManagedOrgan:
    """Register an organ's lifecycle from plain callables::

        managed(
            "vector_index",
            on_configure=index.open,
            on_activate=index.start_ingest,
            on_deactivate=index.pause_ingest,
            on_cleanup=index.close,
        )
    """
    return _MANAGER.register(
        ManagedOrgan(
            name, critical=critical, transition_timeout_s=transition_timeout_s, **hooks
        )
    )


@dataclass
class _AdoptedOrgan:
    """Bookkeeping for organs adopted from existing start/stop objects."""

    target: Any
    started: bool = False
    extras: dict[str, Any] = field(default_factory=dict)


def adopt(name: str, target: Any, *, critical: bool = False) -> ManagedOrgan | None:
    """Give an existing start/stop object a managed lifecycle.

    Most of Aura's organs already have ``start``/``stop`` (or ``run``/
    ``shutdown``). Adopting them costs one call and immediately makes
    their state visible and their deactivation distinguishable from their
    failure — without rewriting them.
    """
    start = _first_callable(target, "start", "run", "begin")
    stop = _first_callable(target, "stop", "shutdown", "close")
    if start is None and stop is None:
        return None
    state = _AdoptedOrgan(target=target)

    async def _activate() -> bool:
        if start is None:
            return True
        outcome = start()
        if asyncio.iscoroutine(outcome):
            outcome = await outcome
        state.started = True
        return outcome is not False

    async def _deactivate() -> bool:
        if stop is None:
            return True
        outcome = stop()
        if asyncio.iscoroutine(outcome):
            outcome = await outcome
        state.started = False
        return outcome is not False

    return _MANAGER.register(
        ManagedOrgan(
            name,
            critical=critical,
            on_activate=_activate,
            on_deactivate=_deactivate,
            on_shutdown=_deactivate,
        )
    )


def _first_callable(target: Any, *names: str) -> Callable[[], Any] | None:
    for candidate in names:
        fn = getattr(target, candidate, None)
        if callable(fn):
            return fn
    return None


def lifecycle_report() -> dict[str, Any]:
    return _MANAGER.report()


def reset_lifecycle_for_test() -> None:
    _MANAGER.reset_for_test()


__all__ = [
    "LifecycleManager",
    "ManagedOrgan",
    "State",
    "Transition",
    "TransitionError",
    "TransitionRecord",
    "adopt",
    "get_lifecycle_manager",
    "lifecycle_report",
    "managed",
    "reset_lifecycle_for_test",
]
