"""Her resting rhythm: one quiet cycle of her own when nothing else has thought for a while.

Her state log advances only when a thought commits. On an unattended day nothing asks her anything, so on the third day
run (6 November 2026) the only thoughts were her journal's, and those came every five minutes only because the journal
kept failing (it was refused its text for want of a user anchor, 66 times by 03:15) and retried. Those retries ran her
phases and committed a row each: 121 steps in 6 h 12 m, her record of the day. With her journal working it writes
hourly, as designed, and an unattended day would leave her record nearly empty while the rest of her (her affect, her
drives, her workspace, her initiatives) went on.

So when no thought has committed her state for ``RESTING_INTERVAL_S`` (the five minutes of the day she was measured
on), she runs one cycle of her own: her phases over her current state, origin ``resting_cycle``, with no reply asked of
her model (the response step says nothing for this origin). It is a cycle, not a message: nothing reaches anyone and
nothing is written but her state. It is admitted as her other background thought is (memory and failure pressure, a
person's turn in flight), except that it needs no person present. It never runs in a proof or evaluation runtime, and
only where her journal runs: the live desktop runtime, not the subject harnesses experiments drive her through.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger("Cognition.RestingRhythm")

RESTING_ORIGIN = "resting_cycle"
#: The cadence her record had on the day she was measured unattended.
RESTING_INTERVAL_S = 300.0
#: How often the rhythm looks.
_LOOK_EVERY_S = 60.0
RESTING_OBJECTIVE = (
    "A quiet moment with nothing asked of you. Nothing needs saying; let what has happened since your last "
    "thought settle."
)


class RestingRhythm:
    """Runs a resting cycle when her own thinking has been quiet for ``RESTING_INTERVAL_S``."""

    def __init__(self, orchestrator: Any, *, interval_s: float = RESTING_INTERVAL_S) -> None:
        self.orchestrator = orchestrator
        self.interval_s = float(interval_s)
        self.running = False
        self._task: asyncio.Task | None = None
        self._started_at = time.time()
        self.cycles = 0
        self.last_reason = ""

    async def start(self) -> None:
        if self.running:
            return
        from core.utils.task_tracker import get_task_tracker

        self.running = True
        self._started_at = time.time()
        self._task = get_task_tracker().create_task(self._loop(), name="resting_rhythm")
        logger.info("Resting rhythm active: a quiet cycle after %.0fs with no thought committed.", self.interval_s)

    async def stop(self) -> None:
        self.running = False
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                if asyncio.current_task() is not None and asyncio.current_task().cancelling():
                    raise

    async def _loop(self) -> None:
        while self.running:
            await asyncio.sleep(_LOOK_EVERY_S)
            try:
                await self.maybe_rest()
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                from core.runtime.errors import record_degradation

                record_degradation("resting_rhythm", exc, severity="warning", action="skipped one resting cycle")

    def why_not(self, now: float | None = None) -> str:
        """Why a resting cycle should not run now, or "" when it may."""
        now = time.time() if now is None else now
        from core.brain.cognitive_engine import cognition_liveness
        from core.runtime.proof_policy import proof_run_active

        if proof_run_active():
            return "proof_run_active"
        last = float(cognition_liveness().get("last_committed_at") or 0.0)
        quiet_since = max(last, self._started_at)
        if now - quiet_since < self.interval_s:
            return "a thought committed recently"
        orch = self.orchestrator
        status = getattr(orch, "status", None)
        if status is not None and getattr(status, "is_processing", False):
            return "a turn is in flight"
        from core.runtime import background_policy

        return str(
            background_policy.background_activity_reason(
                orch,
                profile=background_policy.THOUGHT_BACKGROUND_POLICY,
                allow_no_user_anchor=True,
            )
            or ""
        )

    async def maybe_rest(self) -> bool:
        """Run one resting cycle if her own thinking has been quiet long enough. True when it ran."""
        reason = self.why_not()
        self.last_reason = reason
        if reason:
            return False
        brain = getattr(self.orchestrator, "cognitive_engine", None)
        if brain is None:
            self.last_reason = "no cognitive engine"
            return False
        from core.brain.types import ThinkingMode

        await brain.think(
            objective=RESTING_OBJECTIVE,
            context={"mode": "resting"},
            mode=ThinkingMode.FAST,
            origin=RESTING_ORIGIN,
        )
        self.cycles += 1
        return True
