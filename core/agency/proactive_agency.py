"""Proactive agency — the bridge from 'a goal was selected' to 'pursue it to completion'.

The autonomous-initiative loop already *selects* goals (InitiativeArbiter) and advances
missions, but advancing only marks progress — it never drove a goal through the
capability stack. This is the connective tissue: given a goal, it builds a plan (an
injected planner) and pursues it to a verified finish via :class:`GoalPursuitEngine`
(fluid + parallel execution), strictly gated so autonomous action only happens when it
is *allowed* (background policy) and *appropriate* (timing / user-presence).

Safe by construction: no planner ⇒ no autonomous execution (returns ``None``), and every
pursuit passes the background-allowed and timing gates first. The planner is injected, so
this is testable and pluggable — a computational/reasoning goal can plan to a deliberation
step, a desktop goal to verified UI steps, etc.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.ProactiveAgency")

Planner = Callable[[str], Awaitable[Any]]   # goal -> list[Step] | list[ParallelTask]


class ProactiveAgency:
    def __init__(
        self,
        *,
        pursuit: Any | None = None,
        planner: Planner | None = None,
        background_allowed: Callable[[], bool] | None = None,
        timing_ok: Callable[[], Any] | None = None,
        default_planner_enabled: bool | None = None,
    ) -> None:
        self._pursuit = pursuit
        self._planner = planner
        self._background_allowed = background_allowed
        self._timing_ok = timing_ok
        # Proactive autonomy is ON by default — Aura is always self-directed. It is made
        # safe not by disabling it but by being non-blocking (fire-and-forget),
        # single-flight (one pursuit at a time), and running on the cheap BACKGROUND lane,
        # so it never stalls the event loop or competes with the foreground conversation.
        # AURA_PROACTIVE_AUTONOMY=0 is a kill-switch.
        self._default_planner_enabled = (
            default_planner_enabled
            if default_planner_enabled is not None
            else os.getenv("AURA_PROACTIVE_AUTONOMY", "1") != "0"
        )
        self._pursued = 0
        self._completed = 0
        self._running = False   # single-flight guard

    @property
    def enabled(self) -> bool:
        """True if proactive pursuit can run (an explicit planner, or env opt-in)."""
        return self._planner is not None or self._default_planner_enabled

    def register_planner(self, planner: Planner) -> None:
        self._planner = planner

    def _get_planner(self) -> Planner | None:
        """Resolve a planner — default to the GoalPlanner so open-ended goals are plannable."""
        if self._planner is not None:
            return self._planner
        if self._default_planner_enabled:
            try:
                from core.agency.goal_planner import GoalPlanner

                # Cheap background planner: a single deliberation sample (not 3-5) so
                # proactive autonomous thinking is light on the background lane.
                self._planner = GoalPlanner(deliberate_samples=1)
                return self._planner
            except (ImportError, AttributeError, RuntimeError) as exc:
                record_degradation("proactive_agency", exc)
        return None

    def _engine(self) -> Any | None:
        if self._pursuit is not None:
            return self._pursuit
        try:
            from core.agency.goal_pursuit import get_goal_pursuit_engine

            self._pursuit = get_goal_pursuit_engine()
            return self._pursuit
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("proactive_agency", exc)
            return None

    async def pursue_goal(self, goal: str, *, parallel: bool = False) -> Any | None:
        """Plan and pursue ``goal`` to completion — or ``None`` if not allowed/plannable.

        Single-flight: only one proactive pursuit runs at a time, so background
        deliberation can never pile up and saturate the model lane.
        """
        if not goal or not str(goal).strip():
            return None
        if self._running:
            return None   # a pursuit is already in flight — don't stack another
        if self._background_allowed is not None:
            try:
                if not self._background_allowed():
                    logger.debug("⏸️ [Proactive] background action not allowed; skipping '%s'.", goal[:50])
                    return None
            except (RuntimeError, AttributeError, TypeError) as exc:
                # Fail CLOSED. The exception was recorded and execution fell
                # through to planning and pursuit, so the gate raising had
                # exactly the same effect as the gate approving. A caller
                # reading this code sees a check; what ran was no check at
                # all whenever it mattered most.
                record_degradation(
                    "proactive_agency",
                    exc,
                    severity="warning",
                    action="skipped a proactive pursuit because the background gate raised",
                    extra={"goal": str(goal)[:120]},
                )
                return None
        planner = self._get_planner()
        if planner is None:
            return None
        self._running = True
        try:
            try:
                plan = await planner(goal)
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("proactive_agency", exc)
                return None
            if not plan:
                return None
            engine = self._engine()
            if engine is None:
                return None
            self._pursued += 1
            outcome = await engine.pursue(
                goal,
                plan,
                parallel=parallel,
                timing_ok=self._timing_ok,
                replan=self._replanner(goal, planner, plan),
            )
            if getattr(outcome, "completed", False):
                self._completed += 1
                logger.info("✅ [Proactive] autonomously completed goal: %s", goal[:60])
            elif getattr(outcome, "deferred", False):
                logger.debug("⏸️ [Proactive] goal deferred (timing): %s", goal[:50])
            return outcome
        finally:
            self._running = False

    @staticmethod
    def _failed_approaches(receipt: Any) -> set[str]:
        """Which approaches the last attempt actually tried and did not finish.

        Read from the receipt's steps rather than from the plan, because what
        matters is what RAN. A step that never got to run is not evidence
        against the approach it belonged to.
        """
        failed: set[str] = set()
        for step in list(getattr(receipt, "steps", None) or []):
            if bool(getattr(step, "ok", False)) and not bool(getattr(step, "blocked", False)):
                continue
            # The approach is carried on the step itself, stamped by whoever
            # planned it. Reconstructing it from the step NAME needs a
            # name-to-approach table, which is a second vocabulary that drifts
            # from the planner's — measured: a stalled desktop step reported
            # "desktop_open", matched no approach, and the retry was abandoned
            # even though a different approach was available.
            approach = str(getattr(step, "approach", "") or "").strip().lower()
            if approach:
                failed.add(approach)
        return failed

    def _replanner(self, goal: str, planner: Planner, first_plan: Any) -> Callable[[Any], Any]:
        """A second attempt that is actually a different attempt.

        The engine has carried a replan budget from the start and nothing ever
        passed a replanner, so `max_replans` was dead: every stall ended the
        pursuit for good. Supplying one is only half of it — a goal classifies
        the same way every time it is read, so a naive re-plan hands back the
        plan that just stalled and it stalls again identically.

        So the failure is fed back structurally: the approaches that ran and did
        not finish are excluded, and the planner routes to the next most direct
        way of reaching the same goal. Nothing is phrased into a prompt; the
        evidence is which steps failed.
        """

        async def replan(receipt: Any) -> Any:
            avoid = self._failed_approaches(receipt)
            try:
                fresh = await planner(goal, avoid=tuple(sorted(avoid)))
            except TypeError:
                # A planner that predates failure-aware routing. Re-planning it
                # can only reproduce the same plan, so there is nothing to try.
                return None
            except (RuntimeError, AttributeError, ValueError) as exc:
                record_degradation(
                    "proactive_agency",
                    exc,
                    severity="warning",
                    action="abandoned a stalled pursuit because re-planning failed",
                    extra={"goal": str(goal)[:120]},
                )
                return None
            if not fresh:
                return None
            if self._same_plan(fresh, first_plan):
                # Running it again would stall in the same place. Stopping is
                # the honest outcome; a retry that cannot differ is theatre.
                logger.debug("[Proactive] re-plan matched the stalled plan; not retrying '%s'.", goal[:50])
                return None
            return fresh

        return replan

    @staticmethod
    def _same_plan(left: Any, right: Any) -> bool:
        def signature(plan: Any) -> tuple[str, ...]:
            return tuple(
                str(getattr(step, "name", "") or "").strip().lower()
                for step in list(plan or [])
            )

        return signature(left) == signature(right)

    def status(self) -> dict[str, Any]:
        return {"pursued": self._pursued, "completed": self._completed, "has_planner": self._planner is not None}


_instance: ProactiveAgency | None = None


def get_proactive_agency() -> ProactiveAgency:
    global _instance
    if _instance is None:
        _instance = ProactiveAgency()
    return _instance
