"""Pursue a goal on screen: look, act, look again, until it is done.

Every part of this already existed and none of them were connected.

``FluidExecutor.pursue`` closes a perceive-decide-act loop with governance and
verification. ``host_automation.get_screen_text`` reads the screen and now
returns WHERE each run of text sat. ``perception_demand`` keeps her eyes open
at task cadence while she acts, instead of the 0.1Hz a foreground generation
used to impose. ``hotkey`` and ``click_at`` press keys and click points.

What was missing was any way to ASK for the combination. Every skill in the
registry performs ONE act and returns; nothing could watch something change
and keep going. So tasks that are trivially describable — "wait for that build
to finish and tell me if it fails", "keep pressing next until the form is
done", "play this until you win" — had no path through the system at all, and
the shape of the failure was always the same: she did one step and stopped.

This is deliberately not about any particular screen. It takes a goal, a way
to recognise success in what is read back, and a bound; it decides each move
from the CURRENT reading rather than from a plan written in advance. A build
log, a wizard, an installer and a game are the same problem.

The decision itself is delegated to a policy callable, so the loop does not
own any judgement about what to press. That keeps the cognition where
cognition lives and leaves this module as what it is: a way to keep looking.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from core.runtime.errors import record_degradation
from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Aura.ScreenPursuit")

#: Keys the loop is allowed to press, by the name a person would use.
#: Bounded on purpose — a loop that can press anything can press ⌘Q.
PRESSABLE_KEYS = (
    "up", "down", "left", "right",
    "return", "enter", "tab", "space", "escape",
)

#: How long a single observation may take before the cycle is abandoned. A
#: screen read is ~1-2s; anything far past that means the capture is wedged
#: and waiting longer only makes the loop less responsive.
OBSERVE_TIMEOUT_S = 8.0


class ScreenPursuitInput(BaseModel):
    goal: str = Field(..., min_length=1, max_length=400)
    #: Text that appearing on screen means the goal is reached. Matched
    #: case-insensitively against the reading, as a regular expression when it
    #: is one and as plain text otherwise.
    success_when: str = Field(..., min_length=1, max_length=200)
    max_cycles: int = Field(default=200, ge=1, le=2000)
    max_seconds: float = Field(default=600.0, ge=1.0, le=3600.0)
    narrate: bool = Field(default=True)


ObservationPolicy = Callable[[dict[str, Any]], Awaitable[dict[str, Any] | None]]


async def read_screen() -> dict[str, Any]:
    """One reading: the words, and where they were."""
    from core.capabilities.host_automation import get_host_automation

    receipt = await get_host_automation().get_screen_text(retain_screenshot=False)
    return {
        "ok": bool(getattr(receipt, "success", False)),
        "text": str(getattr(receipt, "result", "") or ""),
        "layout": list(getattr(receipt, "layout", []) or []),
        "error": str(getattr(receipt, "error", "") or ""),
        "at": time.time(),
    }


def goal_reached(observation: dict[str, Any], success_when: str) -> bool:
    """Whether this reading shows the goal met.

    Tested against what was actually read, not against a belief about what the
    action should have done. That distinction is the reason to look again at
    all: an action that ran is not an action that worked.
    """
    text = str(observation.get("text") or "")
    if not text:
        return False
    pattern = str(success_when or "").strip()
    if not pattern:
        return False
    try:
        return re.search(pattern, text, re.IGNORECASE) is not None
    except re.error:
        return pattern.lower() in text.lower()


async def press(key: str) -> bool:
    """Press one of the allowed keys. False if it is not one of them."""
    name = str(key or "").strip().lower()
    if name not in PRESSABLE_KEYS:
        return False
    from core.capabilities.host_automation import get_host_automation

    receipt = await get_host_automation().hotkey(name)
    return bool(getattr(receipt, "success", False))


class ScreenPursuitSkill(BaseSkill):
    """Keep looking at the screen and acting until a goal is reached."""

    name = "pursue_on_screen"
    description = (
        "Pursue a goal on screen over many steps: read the screen, decide one "
        "action, do it, read again, and repeat until a described condition "
        "appears or a bound is reached. Use for anything that needs watching "
        "rather than a single action — waiting for a long job and reacting to "
        "how it ends, stepping through a wizard, or playing something. Give "
        "the goal and the text that means it is done."
    )
    input_model: type[BaseModel] | None = ScreenPursuitInput
    timeout_seconds: float = 900.0
    metabolic_cost: int = 6
    requires_approval = False
    # Same authority as computer_use: this presses keys on the foreground
    # desktop, and calling it anything gentler would be understating it.
    effect_scope = "foreground_desktop_control"

    async def execute(self, params: Any, context: dict[str, Any]) -> dict[str, Any]:
        if isinstance(params, dict):
            params = ScreenPursuitInput(**params)
        policy = (context or {}).get("screen_policy")
        return await pursue_on_screen(
            goal=params.goal,
            success_when=params.success_when,
            policy=policy,
            max_cycles=params.max_cycles,
            max_seconds=params.max_seconds,
            narrate=params.narrate,
        )


async def pursue_on_screen(
    *,
    goal: str,
    success_when: str,
    policy: ObservationPolicy | None = None,
    max_cycles: int = 200,
    max_seconds: float = 600.0,
    narrate: bool = True,
) -> dict[str, Any]:
    """Run the loop. Returns the receipt the executor produced.

    ``policy`` decides the next move from a reading and returns either a
    ``{"key": ...}`` intent or None for "nothing worth doing". It is injected
    rather than imported so the judgement can come from cognition, from a
    cheap local heuristic, or from a test — the loop does not care, and that
    is what keeps it general.
    """
    from core.skills.fluid_executor import FluidExecutor, Step

    moves: list[dict[str, Any]] = []

    async def observe() -> dict[str, Any]:
        try:
            return await asyncio.wait_for(read_screen(), timeout=OBSERVE_TIMEOUT_S)
        except (TimeoutError, asyncio.TimeoutError):
            # A wedged capture is not a reason to keep acting blind.
            return {"ok": False, "text": "", "layout": [], "error": "observe_timeout"}

    def satisfied(observation: dict[str, Any]) -> bool:
        return goal_reached(observation, success_when)

    async def decide(observation: dict[str, Any]) -> Step | None:
        if policy is None or not observation.get("ok"):
            return None
        try:
            intent = await policy(observation)
        except (RuntimeError, TypeError, ValueError, KeyError) as exc:
            record_degradation(
                "screen_pursuit",
                exc,
                severity="info",
                action="ended a screen pursuit cycle without a move",
            )
            return None
        if not intent:
            return None
        key = str(intent.get("key") or "").strip().lower()
        if key not in PRESSABLE_KEYS:
            return None
        because = str(intent.get("because") or "").strip()
        moves.append({"key": key, "because": because, "at": time.time()})
        if narrate:
            # Said out loud, per move, because a loop that acts silently for
            # ten minutes is indistinguishable from one that has hung.
            await _narrate(f"Board: {key.capitalize()}", because)

        async def act() -> bool:
            return await press(key)

        return Step(name=f"press {key}", action=act)

    executor = FluidExecutor(verifier=None, gateway=None)
    receipt = await executor.pursue(
        goal,
        observe=observe,
        decide=decide,
        is_satisfied=satisfied,
        max_cycles=max_cycles,
        max_seconds=max_seconds,
        perception_reason=f"pursuing on screen: {goal[:60]}",
    )
    result = receipt.to_dict()
    result["moves"] = moves
    result["success_when"] = success_when
    return result


async def _narrate(line: str, because: str = "") -> None:
    """Offer one line to whatever surface is listening. Never raises."""
    try:
        from core.perception.ambient_presence import get_ambient_presence

        get_ambient_presence().offer_utterance(
            f"{line} — {because}" if because else line
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        logger.debug("screen pursuit narration unavailable: %s", line)


__all__ = [
    "OBSERVE_TIMEOUT_S",
    "PRESSABLE_KEYS",
    "ScreenPursuitInput",
    "ScreenPursuitSkill",
    "goal_reached",
    "press",
    "pursue_on_screen",
    "read_screen",
]
