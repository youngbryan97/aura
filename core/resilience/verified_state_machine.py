"""core/resilience/verified_state_machine.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Formally verified finite state machine framework for reliability-grade
lifecycle management.

Features:
- Explicit state graph with allowed transitions
- Transition guards (preconditions that must hold)
- Transition actions (side effects on transition)
- Unreachable state detection at definition time
- Deadlock detection (no outgoing transitions from non-terminal states)
- Runtime enforcement: illegal transitions raise IllegalTransitionError
  and record FAULT-F17

Satisfies production-grade formal state machine verification practices.

Usage:
    from core.resilience.verified_state_machine import VerifiedStateMachine

    sm = VerifiedStateMachine(
        name="component_health",
        states={"HEALTHY", "DEGRADED", "FAILING", "FAILED", "RECOVERING"},
        initial="HEALTHY",
        terminal={"FAILED"},
        transitions={
            ("HEALTHY", "DEGRADED"): None,
            ("DEGRADED", "FAILING"): None,
            ("DEGRADED", "RECOVERING"): None,
            ("FAILING", "FAILED"): None,
            ("FAILED", "RECOVERING"): None,
            ("RECOVERING", "HEALTHY"): None,
        },
    )
    sm.verify()  # Checks for deadlocks and unreachable states
    sm.transition("DEGRADED")  # HEALTHY → DEGRADED
"""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("Aura.StateMachine")


# Guarded-callable failure envelope: user-supplied callables (checks, guards,
# actions, hooks, channels) may raise anything. House discipline forbids broad
# `except Exception`; this tuple names the realistic failure universe
# explicitly. Exotic escapes — custom Exception subtypes outside these bases,
# SystemExit, KeyboardInterrupt — propagate loudly by design.
_GUARDED_CALLABLE_ERRORS = (
    RuntimeError, AttributeError, TypeError, ValueError,
    LookupError, ArithmeticError, OSError, ImportError,
)


class IllegalTransitionError(RuntimeError):
    """Raised when an illegal state transition is attempted."""

    def __init__(self, machine: str, from_state: str, to_state: str) -> None:
        self.machine = machine
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(
            f"Illegal transition in {machine}: {from_state} → {to_state}"
        )


class DeadlockDetectedError(RuntimeError):
    """Raised during verification if a non-terminal state has no outgoing transitions."""


class UnreachableStateError(RuntimeError):
    """Raised during verification if a state cannot be reached from the initial state."""


@dataclass
class TransitionRecord:
    """Record of a state transition."""
    from_state: str
    to_state: str
    timestamp: float = field(default_factory=time.time)
    guard_passed: bool = True
    action_executed: bool = False


TransitionGuard = Callable[[], bool]
TransitionAction = Callable[[str, str], None]  # (from_state, to_state)


class VerifiedStateMachine:
    """Formally verified finite state machine.

    Thread-safe. All state mutations are lock-protected.
    """

    def __init__(
        self,
        name: str,
        states: set[str],
        initial: str,
        transitions: dict[tuple[str, str], TransitionGuard | None],
        terminal: set[str] | None = None,
        actions: dict[tuple[str, str], TransitionAction] | None = None,
        max_history: int = 200,
    ) -> None:
        if initial not in states:
            raise ValueError(f"Initial state '{initial}' not in states: {states}")

        self.name = name
        self.states = frozenset(states)
        self.initial = initial
        self.terminal = frozenset(terminal or set())
        self._transitions = dict(transitions)
        self._actions = dict(actions or {})

        # RLock: transition() runs guards and actions while holding the lock
        # (transitions must serialize), and a guard/action may legitimately
        # call back into this machine (current, can_transition, status). A
        # non-reentrant lock would deadlock on that first callback.
        self._lock = threading.RLock()
        self._current = initial
        self._history: deque[TransitionRecord] = deque(maxlen=max_history)
        self._transition_counts: dict[tuple[str, str], int] = defaultdict(int)

        # Validate terminal states are in states set
        unknown_terminal = self.terminal - self.states
        if unknown_terminal:
            raise ValueError(f"Terminal states not in states set: {unknown_terminal}")

        # Validate transition endpoints
        for from_s, to_s in self._transitions:
            if from_s not in self.states:
                raise ValueError(f"Transition source '{from_s}' not in states")
            if to_s not in self.states:
                raise ValueError(f"Transition target '{to_s}' not in states")

    @property
    def current(self) -> str:
        with self._lock:
            return self._current

    def verify(self) -> list[str]:
        """Verify the state machine for structural correctness.

        Checks:
        1. No deadlocks: every non-terminal state has outgoing transitions
        2. No unreachable states: every state is reachable from initial
        3. Terminal states have no outgoing transitions (warning only)

        Returns a list of warnings (empty if clean).
        Raises on critical issues (deadlocks, unreachable states).
        """
        warnings: list[str] = []

        # Build adjacency list
        adj: dict[str, set[str]] = {s: set() for s in self.states}
        for from_s, to_s in self._transitions:
            adj[from_s].add(to_s)

        # Check 1: Deadlocks — non-terminal states with no outgoing transitions
        for state in self.states:
            if state in self.terminal:
                continue
            if not adj[state]:
                raise DeadlockDetectedError(
                    f"Non-terminal state '{state}' in {self.name} has no outgoing "
                    f"transitions (deadlock). Add transitions or mark as terminal."
                )

        # Check 2: Reachability — BFS from initial state
        visited: set[str] = set()
        queue = [self.initial]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            queue.extend(adj.get(current, set()) - visited)

        unreachable = self.states - visited
        if unreachable:
            raise UnreachableStateError(
                f"States unreachable from '{self.initial}' in {self.name}: "
                f"{unreachable}. Add transitions or remove these states."
            )

        # Check 3: Terminal states with outgoing transitions (warning)
        for state in self.terminal:
            if adj[state]:
                warnings.append(
                    f"Terminal state '{state}' has outgoing transitions: "
                    f"{adj[state]}. This is unusual."
                )

        if not warnings:
            logger.info("State machine '%s' verified: %d states, %d transitions, clean",
                        self.name, len(self.states), len(self._transitions))
        else:
            logger.warning("State machine '%s' verified with %d warnings: %s",
                           self.name, len(warnings), warnings)

        return warnings

    def transition(self, to_state: str) -> TransitionRecord:
        """Attempt a state transition.

        Returns a TransitionRecord on success.
        Raises IllegalTransitionError if the transition is not allowed.
        """
        with self._lock:
            from_state = self._current
            key = (from_state, to_state)

            # Check transition is defined
            if key not in self._transitions:
                self._record_illegal_transition(from_state, to_state)
                raise IllegalTransitionError(self.name, from_state, to_state)

            # Check guard
            guard = self._transitions[key]
            guard_passed = True
            if guard is not None:
                try:
                    guard_passed = guard()
                except _GUARDED_CALLABLE_ERRORS as exc:
                    logger.error(
                        "Guard for %s → %s in %s raised: %s",
                        from_state, to_state, self.name, exc,
                    )
                    guard_passed = False

            if not guard_passed:
                logger.warning(
                    "Guard rejected transition %s → %s in %s",
                    from_state, to_state, self.name,
                )
                return TransitionRecord(
                    from_state=from_state, to_state=to_state,
                    guard_passed=False,
                )

            # Execute transition
            self._current = to_state
            self._transition_counts[key] += 1

            record = TransitionRecord(
                from_state=from_state, to_state=to_state,
                guard_passed=True,
            )

            # Execute action
            action = self._actions.get(key)
            if action is not None:
                try:
                    action(from_state, to_state)
                    record.action_executed = True
                except _GUARDED_CALLABLE_ERRORS as exc:
                    logger.error(
                        "Action for %s → %s in %s raised: %s",
                        from_state, to_state, self.name, exc,
                    )

            self._history.append(record)

            logger.debug("%s: %s → %s", self.name, from_state, to_state)
            return record

    def can_transition(self, to_state: str) -> bool:
        """Check if a transition to the given state is allowed."""
        with self._lock:
            return (self._current, to_state) in self._transitions

    def allowed_transitions(self) -> list[str]:
        """Return all states reachable from the current state."""
        with self._lock:
            return [to_s for (from_s, to_s) in self._transitions
                    if from_s == self._current]

    def reset(self) -> None:
        """Reset to initial state (for recovery)."""
        with self._lock:
            old = self._current
            self._current = self.initial
            self._history.append(TransitionRecord(
                from_state=old, to_state=self.initial,
            ))
            logger.warning("%s: RESET %s → %s", self.name, old, self.initial)

    def history(self, limit: int = 20) -> list[TransitionRecord]:
        with self._lock:
            items = list(self._history)
        return items[-limit:]

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "current": self._current,
                "states": sorted(self.states),
                "terminal": sorted(self.terminal),
                "transitions_defined": len(self._transitions),
                "transition_counts": {
                    f"{f}→{t}": c for (f, t), c in self._transition_counts.items()
                },
                "history_length": len(self._history),
            }

    def _record_illegal_transition(self, from_state: str, to_state: str) -> None:
        """Record an illegal transition attempt in the fault registry."""
        logger.error(
            "ILLEGAL TRANSITION in %s: %s → %s (allowed: %s)",
            self.name, from_state, to_state,
            [t for (f, t) in self._transitions if f == from_state],
        )
        try:
            from core.resilience.fault_taxonomy import get_fault_registry
            get_fault_registry().record_fault(
                "F17", subsystem=f"state_machine.{self.name}",
                details=f"Illegal: {from_state} → {to_state}",
            )
        except (ImportError, AttributeError, RuntimeError) as exc:
            logger.debug("Fault registry unavailable for illegal transition: %s", exc)


# ── Pre-built state machines for Aura subsystems ─────────────────────

def create_component_health_machine(name: str = "component_health") -> VerifiedStateMachine:
    """Standard component health state machine.

    Enforces the correct recovery path:
        HEALTHY → DEGRADED → FAILING → FAILED → RECOVERING → HEALTHY
    No direct FAILED→HEALTHY jumps.
    """
    sm = VerifiedStateMachine(
        name=name,
        states={"HEALTHY", "DEGRADED", "FAILING", "FAILED", "RECOVERING"},
        initial="HEALTHY",
        terminal=set(),  # No terminal states — health is cyclical
        transitions={
            ("HEALTHY", "DEGRADED"): None,
            ("DEGRADED", "FAILING"): None,
            ("DEGRADED", "RECOVERING"): None,
            ("DEGRADED", "HEALTHY"): None,  # Minor degradation self-heals
            ("FAILING", "FAILED"): None,
            ("FAILING", "RECOVERING"): None,  # Catch before full failure
            ("FAILED", "RECOVERING"): None,
            ("RECOVERING", "HEALTHY"): None,
            ("RECOVERING", "DEGRADED"): None,  # Partial recovery
        },
    )
    sm.verify()
    return sm


def create_boot_lifecycle_machine(name: str = "boot_lifecycle") -> VerifiedStateMachine:
    """Boot lifecycle state machine."""
    sm = VerifiedStateMachine(
        name=name,
        states={
            "INIT", "LOADING_CONFIG", "LOADING_MODEL", "STARTING_SERVICES",
            "HEALTH_CHECK", "READY", "BOOT_FAILED",
        },
        initial="INIT",
        terminal={"BOOT_FAILED", "READY"},
        transitions={
            ("INIT", "LOADING_CONFIG"): None,
            ("LOADING_CONFIG", "LOADING_MODEL"): None,
            ("LOADING_CONFIG", "BOOT_FAILED"): None,
            ("LOADING_MODEL", "STARTING_SERVICES"): None,
            ("LOADING_MODEL", "BOOT_FAILED"): None,
            ("STARTING_SERVICES", "HEALTH_CHECK"): None,
            ("STARTING_SERVICES", "BOOT_FAILED"): None,
            ("HEALTH_CHECK", "READY"): None,
            ("HEALTH_CHECK", "BOOT_FAILED"): None,
        },
    )
    sm.verify()
    return sm


def create_shutdown_lifecycle_machine(name: str = "shutdown_lifecycle") -> VerifiedStateMachine:
    """Shutdown lifecycle state machine."""
    sm = VerifiedStateMachine(
        name=name,
        states={
            "RUNNING", "DRAINING", "STOPPING_SERVICES",
            "FLUSHING_STATE", "TERMINATED",
        },
        initial="RUNNING",
        terminal={"TERMINATED"},
        transitions={
            ("RUNNING", "DRAINING"): None,
            ("DRAINING", "STOPPING_SERVICES"): None,
            ("STOPPING_SERVICES", "FLUSHING_STATE"): None,
            ("FLUSHING_STATE", "TERMINATED"): None,
            # Emergency shortcuts
            ("RUNNING", "TERMINATED"): None,  # SIGKILL
            ("DRAINING", "TERMINATED"): None,  # Timeout
            ("STOPPING_SERVICES", "TERMINATED"): None,  # Timeout
        },
    )
    sm.verify()
    return sm
