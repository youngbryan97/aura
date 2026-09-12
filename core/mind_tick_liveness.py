"""Whether the tick is still running, and what to do when it is not.

A loop that has stopped and a loop that is merely slow look the same from
outside, so liveness here is measured against the last progress mark rather
than against a clock. The repair is bounded and says when it gives up, because
a restart that retries forever is the same outage with more log lines.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from core.health.degraded_events import record_degraded_event
from core.runtime.shutdown_coordinator import is_shutdown_requested


class _KnowsWhetherItIsStillAlive:
    """Lifted whole from MindTick; see mind_tick.py."""

    def is_alive(self) -> bool:
        """Report whether the supervised loop is running and progressing.

        A QUERY. It records what it observes and changes nothing: a health
        probe that restarts the service it is inspecting turns every reader —
        a dashboard, a contract sweep, a status endpoint — into an actuator,
        and the reader has no idea it did anything. ``ensure_alive`` is the
        one that repairs, and callers that want self-healing ask for it.
        """
        return self._liveness(repair=False)

    def ensure_alive(self) -> bool:
        """Report liveness AND repair a lost loop, returning the result.

        This is what the old ``is_alive`` did to every caller. It is a
        deliberate action now, taken by whoever wants it.
        """
        return self._liveness(repair=True)

    def _liveness(self, *, repair: bool) -> bool:
        # Imported here rather than at module level: the module these
        # came from imports this one to build the class. A call-time
        # import also still sees a test's patch of the original.
        from .mind_tick import (
            DEFAULT_HARD_STALL_S,
            DEFAULT_STALE_PROGRESS_S,
        )

        task_alive = bool(self._task and not self._task.done())
        if not self._running or not task_alive:
            if not repair or not self._attempt_liveness_repair():
                return False
            task_alive = bool(self._task and not self._task.done())
        if (
            float(getattr(self, "_last_successful_tick_at", 0.0) or 0.0) <= 0.0
            and float(getattr(self, "_last_loop_progress_at", 0.0) or 0.0) <= 0.0
        ):
            if int(getattr(self, "_consecutive_loop_failures", 0) or 0) >= 3:
                if not repair or not self._attempt_liveness_repair(
                    reason="repeated loop failures before first progress"
                ):
                    return False
                return bool(self._task and not self._task.done())
            return bool(self._started_at and (time.time() - self._started_at) <= 180.0)
        now = time.time()
        freshest_progress = max(
            float(getattr(self, "_last_successful_tick_at", 0.0) or 0.0),
            float(getattr(self, "_last_loop_progress_at", 0.0) or 0.0),
        )
        stale_progress_s = self._float_env(
            "AURA_MIND_TICK_STALE_PROGRESS_S", DEFAULT_STALE_PROGRESS_S
        )
        if freshest_progress > 0.0 and (now - freshest_progress) <= stale_progress_s:
            return True
        active_started = float(getattr(self, "_active_tick_started_at", 0.0) or 0.0)
        hard_stall_s = self._float_env(
            "AURA_MIND_TICK_HARD_STALL_S", DEFAULT_HARD_STALL_S
        )
        if (
            task_alive
            and active_started > 0.0
            and (now - active_started) <= hard_stall_s
            and int(getattr(self, "_consecutive_loop_failures", 0) or 0) < 3
        ):
            return True
        if int(getattr(self, "_consecutive_loop_failures", 0) or 0) >= 3:
            if not repair or not self._attempt_liveness_repair(
                reason="repeated loop failures without fresh progress"
            ):
                return False
            return bool(self._task and not self._task.done())
        if task_alive:
            age = now - freshest_progress if freshest_progress > 0.0 else now - float(getattr(self, "_started_at", now) or now)
            # Name the wedge: the contract line 'is_alive() returned False'
            # told an operator nothing for two hours. Every stale-progress
            # verdict now records WHERE the rhythm is stuck.
            record_degraded_event(
                "mind_tick",
                "rhythm_stale",
                detail=(
                    f"stage={getattr(self, '_active_tick_stage', '?')} "
                    f"progress_age={age:.0f}s"
                ),
                severity="warning",
                classification="background_degraded",
                context={
                    "stage": str(getattr(self, "_active_tick_stage", "") or ""),
                    "progress_age_s": round(age, 1),
                },
            )
            if repair:
                self._attempt_liveness_repair(
                    reason=f"stale progress for {age:.1f}s",
                    cancel_existing=True,
                )
        return False

    def _schedule_restart_after(self, old_task: Any, *, reason: str = "") -> None:
        """Start the replacement loop only once the old one has actually ended.

        The whole point of the fix for CP126 e98446be: there must never be
        two ``_run_loop`` coroutines alive at the same time. A done-callback
        is the only way to know the cancelled one has really unwound,
        because ``cancel()`` returns long before that happens.
        """
        from .mind_tick import (
            _MIND_BOUNDARY_ERRORS,
            _record_mind_degradation,
            _schedule_mind_task,
            logger,
        )


        def _restart(finished: Any) -> None:
            try:
                if is_shutdown_requested():
                    self._repair_pending = False
                    return
                self._consecutive_loop_failures = 0
                self._running = True
                self._started_at = time.time()
                self._active_tick_started_at = 0.0
                self._active_tick_stage = "repair_after_cancel"
                self._last_successful_tick_at = 0.0
                self._mark_loop_progress("repair_after_cancel")
                count = int(getattr(self, "_liveness_repair_count", 0) or 0) + 1
                name = f"mind_tick.run_loop.recovered.{count}"
                self._task = _schedule_mind_task(self._run_loop(), name=name)
                if self._task is None:
                    self._running = False
                    self._repair_pending = False
                    record_degraded_event(
                        "mind_tick",
                        "liveness_repair_failed",
                        detail="replacement loop could not be scheduled after cancel",
                        severity="error",
                        classification="background_degraded",
                    )
                    return
                self._install_loop_done_callback(self._task, name=name)
                self._liveness_repair_count = count
                self._repair_pending = False
                record_degraded_event(
                    "mind_tick",
                    "liveness_repair",
                    detail=reason or "stale loop cancelled and replaced after it unwound",
                    severity="warning",
                    classification="runtime_recovered",
                    context={"repair_count": count},
                )
                logger.warning(
                    "💓 MindTick: stale loop unwound; replacement started (repair %d).",
                    count,
                )
            except _MIND_BOUNDARY_ERRORS as exc:
                self._repair_pending = False
                _record_mind_degradation(
                    exc,
                    action="MindTick replacement loop could not start after the stale loop ended",
                    severity="critical",
                )

        try:
            old_task.add_done_callback(_restart)
        except _MIND_BOUNDARY_ERRORS as exc:
            self._repair_pending = False
            _record_mind_degradation(
                exc,
                action="could not chain the MindTick restart to the cancelled loop",
                severity="critical",
            )

    def _attempt_liveness_repair(self, *, reason: str = "", cancel_existing: bool = False) -> bool:
        """Restart the supervised cognitive loop when a live runtime loses it.

        This is deliberately narrow: it only repairs a runtime that already
        marked MindTick running or has a finished task, it rate-limits repair
        attempts, and it never runs during coordinated shutdown. The goal is to
        turn the health-contract failure into an internal recovery attempt
        instead of leaving the desktop path degraded until the next user prompt.
        """
        from .mind_tick import (
            _MIND_BOUNDARY_ERRORS,
            _record_mind_degradation,
            _schedule_mind_task,
            logger,
        )

        if is_shutdown_requested():
            return False
        if not self._running and self._task is None:
            return False
        if not callable(getattr(self, "_run_loop", None)):
            return False
        now = time.monotonic()
        if now - float(getattr(self, "_last_liveness_repair_at", 0.0) or 0.0) < 10.0:
            return False
        self._last_liveness_repair_at = now

        existing_task = getattr(self, "_task", None)
        if cancel_existing and existing_task is not None and not existing_task.done():
            # CP126 e98446be: this cancelled and immediately started a new
            # loop. `cancel()` only REQUESTS cancellation — the old
            # coroutine keeps running until it next reaches an await point,
            # so both loops ran concurrently, each mutating the same state
            # object and committing over the other. A repair that produces
            # two minds is worse than the stall it was repairing.
            #
            # The restart is now chained to the old task's completion, so
            # there is exactly one loop at every instant. If the old task
            # will not die, no new one is started and that is recorded —
            # a stuck loop must not be masked by a second one.
            try:
                existing_task.cancel()
            except _MIND_BOUNDARY_ERRORS as exc:
                _record_mind_degradation(
                    exc,
                    action="continued MindTick liveness repair after stale loop cancel failed",
                    severity="warning",
                )
            else:
                self._repair_pending = True
                self._schedule_restart_after(existing_task, reason=reason)
                return True

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # Health checks call is_alive() from plain threads, where task
            # creation is impossible — so a dead loop could never be revived
            # by the pulse that detected it (observed live 2026-07-05: the
            # runtime sat DEGRADED for 84 minutes with repair machinery
            # present). Hand the repair to the owning loop instead.
            owner_loop = getattr(self, "_owner_loop", None)
            if owner_loop is not None and not owner_loop.is_closed():
                def _threadsafe_repair() -> None:
                    self._consecutive_loop_failures = 0
                    self._running = True
                    self._started_at = time.time()
                    self._active_tick_started_at = 0.0
                    self._active_tick_stage = "threadsafe_repair"
                    self._last_successful_tick_at = 0.0
                    self._mark_loop_progress("threadsafe_repair")
                    self._task = _schedule_mind_task(
                        self._run_loop(), name="mind_tick.run_loop.recovered.threadsafe"
                    )
                    self._install_loop_done_callback(
                        self._task,
                        name="mind_tick.run_loop.recovered.threadsafe",
                    )
                    self._liveness_repair_count = (
                        int(getattr(self, "_liveness_repair_count", 0) or 0) + 1
                    )
                    self._repair_pending = False
                    logger.warning("💓 MindTick: loop revived via owning-loop repair.")

                self._repair_pending = True
                owner_loop.call_soon_threadsafe(_threadsafe_repair)
                logger.info("MindTick repair scheduled onto owning loop from thread.")
                # CP126 c76abf56: this returned False immediately, so health
                # read "unhealthy" while a recovery was already in flight and
                # nothing recorded that one had been started. The pending flag
                # is the receipt; get_health_status reports it.
                record_degraded_event(
                    "mind_tick",
                    "liveness_repair_scheduled",
                    detail=reason or "repair handed to the owning loop from a probe thread",
                    severity="warning",
                    classification="runtime_recovering",
                    context={"stage": str(getattr(self, "_active_tick_stage", "") or "")},
                )
            else:
                # A repair that silently cannot run is how a runtime sits
                # DEGRADED for hours with 'repair machinery present'. Say so.
                record_degraded_event(
                    "mind_tick",
                    "liveness_repair_unreachable",
                    detail=(
                        "no usable owner loop from thread context"
                        if owner_loop is None
                        else "owner loop closed"
                    ),
                    severity="error",
                    classification="background_degraded",
                    context={"stage": str(getattr(self, "_active_tick_stage", "") or "")},
                )
            return False

        finished_error = None
        if self._task and self._task.done():
            try:
                finished_error = self._task.exception()
            except (asyncio.CancelledError, RuntimeError, AttributeError) as exc:
                finished_error = exc

        self._consecutive_loop_failures = 0
        self._running = True
        self._started_at = time.time()
        self._active_tick_started_at = 0.0
        self._active_tick_stage = "repair"
        self._last_successful_tick_at = 0.0
        self._mark_loop_progress("repair")
        self._task = _schedule_mind_task(
            self._run_loop(),
            name=f"mind_tick.run_loop.recovered.{int(getattr(self, '_liveness_repair_count', 0) or 0) + 1}",
        )
        if self._task is None:
            self._running = False
            return False
        self._install_loop_done_callback(
            self._task,
            name=f"mind_tick.run_loop.recovered.{int(getattr(self, '_liveness_repair_count', 0) or 0) + 1}",
        )

        self._liveness_repair_count = int(getattr(self, "_liveness_repair_count", 0) or 0) + 1
        detail = (
            f"{type(finished_error).__name__}: {finished_error}"
            if finished_error is not None
            else (reason or "liveness probe found MindTick loop missing, stopped, or stale")
        )
        logger.warning("💓 MindTick: Repaired stalled cognitive rhythm (%s).", detail)
        record_degraded_event(
            "mind_tick",
            "liveness_repair",
            detail=detail,
            severity="warning",
            classification="runtime_recovered",
            context={"repair_count": self._liveness_repair_count},
            exc=finished_error if isinstance(finished_error, BaseException) else None,
        )
        return True

    def get_health_status(self) -> dict[str, Any]:
        """Expose causal loop progress without treating heartbeat transport as health."""
        return {
            "healthy": self.is_alive(),
            "running": self._running,
            "task_alive": bool(self._task and not self._task.done()),
            "tick_count": int(getattr(self, "_tick_count", 0) or 0),
            "consecutive_failures": int(getattr(self, "_consecutive_loop_failures", 0) or 0),
            "last_successful_tick_at": float(getattr(self, "_last_successful_tick_at", 0.0) or 0.0),
            "last_loop_progress_at": float(getattr(self, "_last_loop_progress_at", 0.0) or 0.0),
            "last_progress_label": str(getattr(self, "_last_progress_label", "") or ""),
            "active_tick_started_at": float(getattr(self, "_active_tick_started_at", 0.0) or 0.0),
            "active_tick_stage": str(getattr(self, "_active_tick_stage", "") or ""),
            "liveness_repair_count": int(getattr(self, "_liveness_repair_count", 0) or 0),
            # CP126 c76abf56: a recovery already in flight used to be
            # invisible, so the surface said "unhealthy" with no indication
            # that anything was being done about it. "Recovering" and
            # "broken and unattended" are different operational states.
            "repair_pending": bool(getattr(self, "_repair_pending", False)),
        }
