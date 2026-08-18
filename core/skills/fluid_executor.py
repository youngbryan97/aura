"""Fluid execution loop — the closed perceive→act→verify→recover cycle.

What makes acting in the world feel *fluid* instead of brittle is not the individual
actions — Aura already has rich action primitives (``computer_use``), effect
verification (``PostActionVerifier``), learned affordances
(``AffordanceKnowledgeBase``) and an action-governance gate
(``EnvironmentActionGateway``). What was missing is the tight loop that *composes*
them so that every action is governed, executed, **verified against its expected
effect**, and — when it fails — **autonomously recovered** rather than silently
dropped or left to stall.

This module is that loop. Each :class:`Step` pairs an action with the verification
predicate that proves it worked. :class:`FluidExecutor` governs the action, runs it,
verifies the effect, and on failure runs a recovery hook + bounded backoff retry. A
run aborts cleanly on a stall (no verified progress over a window) instead of
grinding forever, and returns a full receipt of what actually happened — the
provenance the rest of the system (governance, memory, autonomy) consumes.

The verifier / gateway / sleep are injected, so the loop is deterministically
testable and wires to the real subsystems in production.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.FluidExecutor")

ActionFn = Callable[[], Awaitable[Any]]
RecoveryFn = Callable[["StepResult"], Awaitable[Any]]


@dataclass
class Step:
    """One unit of fluid action: do ``action``, then prove it with ``verify``."""

    name: str
    action: ActionFn
    verify: str = "always_true"                 # PostActionVerifier predicate
    verify_args: dict[str, Any] = field(default_factory=dict)
    max_retries: int = 2
    recovery: RecoveryFn | None = None          # run before each retry
    optional: bool = False                      # a failed optional step doesn't abort the run
    backoff_base_s: float = 0.5


@dataclass
class StepResult:
    name: str
    ok: bool
    attempts: int = 0
    verified: bool = False
    recovered: bool = False
    blocked: bool = False
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "attempts": self.attempts,
            "verified": self.verified,
            "recovered": self.recovered,
            "blocked": self.blocked,
            "detail": self.detail,
        }


@dataclass
class ExecutionReceipt:
    goal: str
    completed: bool
    steps: list[StepResult] = field(default_factory=list)
    verified_progress: int = 0
    stalled: bool = False
    elapsed_s: float = 0.0
    #: Why a goal-directed run ended. "" for a plan-shaped run, which ends
    #: when its list does.
    outcome: str = ""
    #: Iterations of the perceive-decide-act cycle, which is not the same as
    #: len(steps): a cycle can decide to do nothing and still be a cycle.
    cycles: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "completed": self.completed,
            "verified_progress": self.verified_progress,
            "stalled": self.stalled,
            "elapsed_s": round(self.elapsed_s, 3),
            "outcome": self.outcome,
            "cycles": self.cycles,
            "steps": [s.to_dict() for s in self.steps],
        }


def _read_decision(decision: Any) -> tuple[bool, str]:
    """Read an approval verdict in either of the two shapes gateways return.

    `bool(getattr(decision, "allowed", decision))` handled the attribute shape
    and silently inverted the mapping one: `core/security/conscience.py`
    returns `{"allowed": False, "reason": ...}`, and a non-empty dict is
    truthy, so a refusal from it read as approval. A mapping is checked for the
    key before the object is fallen back to.
    """
    if isinstance(decision, Mapping):
        allowed = bool(decision.get("allowed", False))
        return allowed, str(decision.get("reason", "") or "")
    allowed_attr = getattr(decision, "allowed", None)
    if allowed_attr is not None:
        return bool(allowed_attr), str(getattr(decision, "reason", "") or "")
    return bool(decision), str(getattr(decision, "reason", "") or "")


class FluidExecutor:
    """Run governed, verified, self-recovering action sequences."""

    def __init__(
        self,
        *,
        verifier: Any | None = None,
        gateway: Any | None = None,
        stall_window: int = 3,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._verifier = verifier
        self._gateway = gateway
        # Abort if this many consecutive steps make no verified progress.
        self.stall_window = max(1, int(stall_window))
        self._sleep = sleep or asyncio.sleep

    async def _get_verifier(self) -> Any | None:
        if self._verifier is not None:
            return self._verifier
        try:
            from core.capabilities.post_action_verifier import get_post_action_verifier

            self._verifier = get_post_action_verifier()
            return self._verifier
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("fluid_executor", exc)
            return None

    async def _verify(self, predicate: str, args: dict[str, Any]) -> tuple[bool, str]:
        if predicate in ("always_true", "", None):
            return True, "no verification required"
        verifier = await self._get_verifier()
        if verifier is None:
            # No verifier available → trust a clean action dispatch rather than
            # blocking the loop (matches the desktop effect-verified convention).
            return True, "verifier unavailable; trusting clean dispatch"
        try:
            result = await verifier.verify(predicate, args)
            ok = bool(getattr(result, "success", False))
            return ok, str(getattr(result, "detail", "") or getattr(result, "reason", ""))
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("fluid_executor", exc)
            return False, f"verification error: {exc}"

    async def _resolve_gateway(self) -> Any | None:
        """The injected gateway, or the canonical one, or None.

        `gateway=None` used to mean "allowed", and `DesktopPlanner` builds this
        executor without passing one — so the default construction of the
        desktop lane approved every step it was asked about. A default that
        means "no opinion" has to resolve the real gateway before concluding
        anything, which is what this does.
        """
        if self._gateway is not None:
            return self._gateway
        try:
            from core.skills.action_gateway import get_action_gateway

            self._gateway = get_action_gateway()
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("fluid_executor", exc)
            return None
        return self._gateway

    async def _approved(self, step: Step) -> tuple[bool, str]:
        gateway = await self._resolve_gateway()
        if gateway is None:
            # No gateway anywhere. Refuse: an action lane whose governance is
            # missing is not an ungoverned lane that may proceed, it is a lane
            # that cannot say whether the step is allowed.
            return False, "no action gateway available to approve this step"
        try:
            decision = gateway.approve(step.name)
            return _read_decision(decision)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            # A gateway that raised did not approve. Returning True here read
            # "governance has its own hard gates elsewhere", which is the
            # assumption this whole lane exists to stop relying on.
            record_degradation(
                "fluid_executor",
                exc,
                action="refused a step because the action gateway raised",
            )
            return False, f"action gateway error: {exc}"

    async def run_step(self, step: Step) -> StepResult:
        """Govern → act → verify → (recover+retry). Returns the step outcome."""
        approved, reason = await self._approved(step)
        if not approved:
            logger.info("🛡️ [Fluid] step '%s' blocked by governance: %s", step.name, reason)
            return StepResult(step.name, ok=False, blocked=True, detail=f"blocked: {reason}")

        recovered = False
        last_detail = ""
        for attempt in range(1, step.max_retries + 2):
            if attempt > 1 and step.recovery is not None:
                try:
                    await step.recovery(StepResult(step.name, ok=False, attempts=attempt - 1, detail=last_detail))
                    recovered = True
                except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    record_degradation("fluid_executor", exc)
            try:
                await step.action()
            except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
                record_degradation("fluid_executor", exc)
                last_detail = f"action error: {exc}"
                await self._sleep(step.backoff_base_s * attempt)
                continue

            verified, detail = await self._verify(step.verify, step.verify_args)
            last_detail = detail
            if verified:
                return StepResult(
                    step.name, ok=True, attempts=attempt, verified=True,
                    recovered=recovered, detail=detail,
                )
            await self._sleep(step.backoff_base_s * attempt)

        logger.warning("🌀 [Fluid] step '%s' failed after %d attempts: %s",
                       step.name, step.max_retries + 1, last_detail)
        return StepResult(
            step.name, ok=False, attempts=step.max_retries + 1, verified=False,
            recovered=recovered, detail=last_detail,
        )

    async def pursue(
        self,
        goal: str,
        *,
        observe: Callable[[], Awaitable[Any]],
        decide: Callable[[Any], Awaitable[Step | None]],
        is_satisfied: Callable[[Any], Awaitable[bool]] | Callable[[Any], bool],
        max_cycles: int = 200,
        max_seconds: float = 600.0,
        perception_reason: str = "",
    ) -> ExecutionReceipt:
        """Pursue a goal by looking, deciding, acting and looking again.

        ``run`` executes a list someone wrote in advance. That is the right
        shape when the steps are known — open this app, click that button —
        and the wrong shape for anything whose next move depends on what just
        happened. A board that changes, a page that loads at its own pace, a
        drag that has to be corrected mid-flight: none of them can be written
        down as a list beforehand, so none of them were reachable through the
        executor even though every part needed to reach them already existed.

        This is the same loop with the plan removed. Everything it runs still
        goes through run_step, so governance, effect verification, recovery
        and receipts are unchanged — the only new thing is that ``decide``
        gets to see the world before choosing, and the run ends on a PREDICATE
        rather than on running out of list.

        Bounded three ways, because a loop with a goal and no bound is how a
        process eats a machine: cycles, wall-clock, and the same stall
        detection ``run`` uses. ``decide`` returning None means "nothing worth
        doing from here", which counts as a cycle without progress and stalls
        out honestly rather than spinning.

        Perception is held open for the whole run. A loop that acts on what it
        sees is exactly the case where sight was being throttled to one frame
        every ten seconds — see core/runtime/perception_demand.py.
        """
        started = time.monotonic()
        receipt = ExecutionReceipt(goal=goal, completed=False)
        consecutive_no_progress = 0
        cycles = max(1, int(max_cycles))
        deadline = started + max(0.1, float(max_seconds))

        token = None
        try:
            from core.runtime.perception_demand import (
                claim_perception,
                release_perception,
                renew_perception,
            )

            token = claim_perception(perception_reason or f"pursuing: {goal}")
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "fluid_executor",
                exc,
                severity="info",
                action="pursued a goal without raising perception cadence",
            )
            renew_perception = release_perception = None  # type: ignore[assignment]

        try:
            for _ in range(cycles):
                receipt.cycles += 1
                if time.monotonic() >= deadline:
                    receipt.outcome = "out_of_time"
                    break
                if token is not None and renew_perception is not None:
                    renew_perception(token)

                observation = await observe()

                satisfied = is_satisfied(observation)
                if inspect.isawaitable(satisfied):
                    satisfied = await satisfied
                if satisfied:
                    receipt.completed = True
                    receipt.outcome = "goal_reached"
                    break

                step = await decide(observation)
                if step is None:
                    consecutive_no_progress += 1
                    if consecutive_no_progress >= self.stall_window:
                        receipt.stalled = True
                        receipt.outcome = "no_move_available"
                        break
                    continue

                result = await self.run_step(step)
                receipt.steps.append(result)
                if result.ok:
                    receipt.verified_progress += 1
                    consecutive_no_progress = 0
                    continue
                if result.blocked:
                    receipt.outcome = "blocked_by_governance"
                    logger.info(
                        "🛡️ [Fluid] pursuit of '%s' halted: step blocked by governance.",
                        goal,
                    )
                    break
                consecutive_no_progress += 1
                if consecutive_no_progress >= self.stall_window:
                    receipt.stalled = True
                    receipt.outcome = "stalled"
                    logger.warning(
                        "🌀 [Fluid] pursuit of '%s' stalled after %d cycles with no "
                        "verified progress.",
                        goal,
                        consecutive_no_progress,
                    )
                    break
            else:
                receipt.outcome = "out_of_cycles"
        finally:
            if token is not None and release_perception is not None:
                release_perception(token)

        receipt.elapsed_s = time.monotonic() - started
        return receipt

    async def run(self, goal: str, steps: list[Step]) -> ExecutionReceipt:
        """Execute a sequence, aborting on a stall, returning a full receipt."""
        started = time.monotonic()
        receipt = ExecutionReceipt(goal=goal, completed=False)
        consecutive_no_progress = 0
        for step in steps:
            result = await self.run_step(step)
            receipt.steps.append(result)
            if result.ok:
                receipt.verified_progress += 1
                consecutive_no_progress = 0
                continue
            if step.optional:
                consecutive_no_progress = 0
                continue
            consecutive_no_progress += 1
            if result.blocked:
                receipt.elapsed_s = time.monotonic() - started
                logger.info("🛡️ [Fluid] run '%s' halted: step blocked by governance.", goal)
                return receipt
            if consecutive_no_progress >= self.stall_window:
                receipt.stalled = True
                receipt.elapsed_s = time.monotonic() - started
                logger.warning(
                    "🌀 [Fluid] run '%s' stalled after %d steps with no verified progress.",
                    goal, consecutive_no_progress,
                )
                return receipt
            # a non-optional, non-blocking failure that hasn't stalled yet: stop here
            # (the sequence's contract is broken), but mark not-stalled so callers can
            # distinguish "one step failed" from "loop spun without progress".
            receipt.elapsed_s = time.monotonic() - started
            return receipt
        receipt.completed = True
        receipt.elapsed_s = time.monotonic() - started
        return receipt
