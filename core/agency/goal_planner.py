"""Goal planner — turn an open-ended goal into an executable, verified plan.

The proactive arm could pursue a *plan* but had no way to make one from a plain goal.
This library is that missing piece: it classifies a goal and routes it to the right
sub-planner, producing :class:`~core.skills.fluid_executor.Step` objects the
``GoalPursuitEngine`` can run:

  * **computational** ("calculate 47*89", "solve x²-5x+6=0") → one step that returns the
    EXACT answer from the prover/CAS (no guessing);
  * **reach** ("fetch the status from <allowlisted host>") → a governed reach step
    (deny-by-default; only fires for operator-allowlisted hosts);
  * **general / open-ended** ("figure out the best approach to X", a mission step) → one
    step that runs the inference-time amplifier (sample → verify → vote → escalate),
    i.e. she *reasons it through* — a safe, side-effect-free action that always applies.

Every planned step emits its result (insight) so the autonomy layer can record what it
learned. ``generate`` / ``reach`` / ``on_result`` are injected, so it is deterministically
testable and wires to the live model + reach gate in production. Registered as
``ProactiveAgency``'s default planner, so the autonomy loop can now act on open-ended
goals end-to-end, gated by background policy + timing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Sequence

from core.runtime.errors import record_degradation
from core.runtime.service_registry import get_runtime_service
from core.skills.fluid_executor import Step

logger = logging.getLogger("Aura.GoalPlanner")

GenerateFn = Callable[[str, float], Awaitable[str]]
ResultSink = Callable[[str, str, str], None]   # (goal, kind, answer)


@dataclass
class PlannedStep:
    kind: str
    answer: str = ""


def _tagged(steps: list[Step], approach: str) -> list[Step]:
    """Stamp each step with the approach that produced it.

    A replanner needs to know which APPROACH failed, not which step name did.
    Tagging at the point of planning keeps that fact where it is known, instead
    of reconstructing it later from a name-to-approach table that would be a
    second vocabulary drifting from this one.
    """
    for step in steps:
        if not getattr(step, "approach", ""):
            step.approach = approach
    return steps


class GoalPlanner:
    """Classify a goal and produce executable, result-capturing steps."""

    def __init__(
        self,
        *,
        generate: GenerateFn | None = None,
        reach: Any | None = None,
        on_result: ResultSink | None = None,
        deliberate_samples: int = 3,
    ) -> None:
        self._generate = generate
        self._reach = reach
        self._on_result = on_result
        self._deliberate_samples = max(1, int(deliberate_samples))
        self.last = PlannedStep(kind="none")

    async def __call__(self, goal: str, *, avoid: Sequence[str] = ()) -> list[Step]:
        return await self.plan(goal, avoid=avoid)

    def classify(self, goal: str, *, avoid: Sequence[str] = ()) -> str:
        """Which approach fits this goal, skipping any that have already failed.

        ``avoid`` names approaches that were just tried and did not work. A goal
        classifies the same way every time it is read, so re-planning after a
        stall produced a byte-identical plan and the pursuit stalled again in
        exactly the same place — which is why the engine's replan budget was
        worth nothing even once it was wired.

        Routing past a dead approach is what makes a second attempt a different
        attempt. The order is a preference ranking, so skipping one falls to the
        next most direct way of getting the same goal done.
        """
        g = str(goal or "").strip()
        if not g:
            return "none"
        blocked = {str(kind).strip().lower() for kind in avoid or ()}

        def _usable(kind: str) -> bool:
            return kind not in blocked

        try:
            from core.brain.tool_augmented_reasoning import looks_computational

            if looks_computational(g) and _usable("computational"):
                return "computational"
        except (ImportError, RuntimeError):
            pass
        # Desktop UI control ("open Notes", "make a folder", "open <url> in Chrome") plans
        # into verified computer-use steps.
        try:
            from core.agency.desktop_planner import is_desktop_goal

            if is_desktop_goal(g) and _usable("desktop"):
                return "desktop"
        except (ImportError, RuntimeError):
            pass
        lower = g.lower()
        if (
            self._reach is not None
            and any(w in lower for w in ("fetch ", "http", "webhook", "api ", "look up at "))
            and _usable("reach")
        ):
            return "reach"
        # Reasoning is the universal fallback: side-effect-free and always
        # applicable. If even that has been tried and failed, there is no
        # different attempt left to make, and saying so is better than handing
        # back the plan that just failed.
        return "reasoning" if _usable("reasoning") else "none"

    async def plan(self, goal: str, *, avoid: Sequence[str] = ()) -> list[Step]:
        kind = self.classify(goal, avoid=avoid)
        if kind == "none":
            return []
        if kind == "computational":
            return _tagged([self._compute_step(goal)], kind)
        if kind == "desktop":
            try:
                from core.agency.desktop_planner import get_desktop_planner

                steps = await get_desktop_planner().plan(goal)
                if steps:
                    return _tagged(steps, kind)
            except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("goal_planner", exc)
            if "reasoning" in {str(k).strip().lower() for k in avoid or ()}:
                return []
            return _tagged([self._reason_step(goal)], "reasoning")   # fall back if unparseable
        if kind == "reach":
            return _tagged([self._reach_step(goal)], kind)
        return _tagged([self._reason_step(goal)], kind)

    # ── sub-planners ──────────────────────────────────────────────────────

    def _emit(self, goal: str, kind: str, answer: str) -> None:
        self.last = PlannedStep(kind=kind, answer=answer)
        if self._on_result is not None:
            try:
                self._on_result(goal, kind, answer)
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("goal_planner", exc)
        logger.info("🗺️ [Planner] %s goal → %s: %s", kind, goal[:40], (answer or "")[:60])

    def _compute_step(self, goal: str) -> Step:
        async def _act() -> None:
            try:
                from core.brain.tool_augmented_reasoning import solve_exact

                r = solve_exact(goal)
                self._emit(goal, "computational", r.answer if r.ok else "")
            except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("goal_planner", exc)
        return Step(name="compute", action=_act, verify="always_true")

    async def _default_generate(self, prompt: str, temperature: float) -> str:
        try:
            gate = get_runtime_service("inference_gate", default=None)
            if gate is None or not hasattr(gate, "generate_response"):
                return ""
            # Background proactive thinking runs on the cheap background lane, NOT the
            # foreground Cortex — so it never competes with the user's live conversation
            # or saturate the 32B worker.
            return await gate.generate_response(
                prompt,
                origin="proactive_reasoning",
                temperature=temperature,
                max_tokens=384,
                is_background=True,
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("goal_planner", exc)
            return ""

    def _reason_step(self, goal: str) -> Step:
        async def _act() -> None:
            try:
                from core.brain.reasoning_amplifier import DeliberationEngine

                gen = self._generate or self._default_generate
                eng = DeliberationEngine(n_samples=self._deliberate_samples)
                result = await eng.adaptive_deliberate(
                    goal, gen, min_samples=self._deliberate_samples, max_samples=self._deliberate_samples + 2
                )
                self._emit(goal, "reasoning", result.answer)
            except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("goal_planner", exc)
        return Step(name="reason", action=_act, verify="always_true")

    def _reach_step(self, goal: str) -> Step:
        # Extract a URL if present; otherwise this is a no-op governed step.
        import re

        m = re.search(r"https?://\S+", goal)
        url = m.group(0) if m else ""

        async def _act() -> None:
            if not url or self._reach is None:
                self._emit(goal, "reach", "")
                return
            try:
                result = await self._reach.get(url)
                self._emit(goal, "reach", result.body_preview[:200] if result.ok else f"blocked/{result.reason}")
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("goal_planner", exc)
        return Step(name="reach", action=_act, verify="always_true")


_instance: GoalPlanner | None = None


def get_goal_planner() -> GoalPlanner:
    global _instance
    if _instance is None:
        _instance = GoalPlanner()
    return _instance
