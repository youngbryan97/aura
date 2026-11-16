"""Does she have something worth saying, unprompted, about what she saw, when she is not keeping him company?

In the bubble she is keeping the person company, and her remarks are a with's
(core/perception/ambient_presence.KEEPING_COMPANY, core/together/remarks.py):
Bryan, 15 November 2026, "unprompted advice and commentary. that's agency".
What follows is the gate for the rest, with her window up and no with open.

The default is no.

An assistant that comments on whatever is on screen is an assistant you
close. So this is not "generate a remark about the screen" — it is a filter
that starts at silence and requires a reason to leave it. Three gates, in
increasing cost, and the cheap ones exist so the expensive one runs rarely:

  1. is there anything here that would matter to HIM?  A heuristic over the
     observation, no model call. Most ticks stop here.
  2. is it something he already knows?  Telling someone what they are
     currently looking at is not insight, and it is the single most common
     way this kind of feature becomes noise.
  3. what would she actually say?  Her own cognition, in her own voice —
     not a template. A canned "I see you're working on X!" is worse than
     silence, because it is silence plus an interruption.

And then the Will decides, because unprompted speech is an action.

WHAT SHE IS ALLOWED TO NOTICE
─────────────────────────────
Deliberately narrow, and all of it is "something is wrong or about to be",
never "here is what you are doing":

  * a visible error, traceback, or failing test she can see the cause of;
  * a contradiction with something established earlier in the session;
  * something she was ASKED to watch for and has now seen.

Not: summarising the screen, praising, greeting, offering help generically,
or observing that he is working. Those are the failure modes this file
exists to prevent, and they are listed here so the next person adding a
category has to argue past them.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("Aura.Perception.AmbientUtterance")

UTTERANCE_SCHEMA = "aura.perception.ambient_utterance.v1"

#: Signals that something is WRONG on screen. The bar is deliberately a
#: visible fault, not an opportunity — "I could help with that" is the
#: category that makes these features intolerable.
_FAULT_MARKERS: tuple[str, ...] = (
    "traceback (most recent call last)",
    "assertionerror",
    "segmentation fault",
    "fatal error",
    "build failed",
    "tests failed",
    "failed to compile",
    "unhandled exception",
    "connection refused",
    "permission denied",
    "out of memory",
    "killed: 9",
    "merge conflict",
    "panic:",
    "error:",
    "failed:",
)

#: A fault marker in these contexts is almost always noise — a log viewer
#: showing historical errors, documentation about errors, a test file whose
#: job is to contain the word.
_NOISE_CONTEXTS: tuple[str, ...] = (
    "documentation",
    "stack overflow",
    "docs.",
    "man page",
    "changelog",
)

_MIN_CAPTURE_CHARS = 40
_MAX_CAPTURE_CHARS = 6000

#: The least time she has to decide whether to say something unprompted; a longer prompt gets what it needs.
#: Past this the screen has moved on and the remark would be about a window
#: the person already left.
_COMPOSE_TIMEOUT_S = 25.0


async def consider_utterance(tick_result: Any) -> str:
    """Return what she would say, or "" — which is the normal answer.

    Each step that keeps her quiet is named in her log, once per observation she considers. LIVE 14 November 2026,
    with her in the bubble and a traceback in front of her for six minutes, she said nothing, and nothing anywhere said
    which step had held her.
    """
    observation = _latest_observation()
    if observation is None:
        logger.info("🫧 remark: quiet, no screen reading to consider")
        return ""
    capture = str(getattr(observation, "capture", "") or "")
    if not (_MIN_CAPTURE_CHARS <= len(capture) <= _MAX_CAPTURE_CHARS):
        # Too little to be anything; too much to be one screen worth
        # reasoning about without a request.
        logger.info("🫧 remark: quiet, the screen reading is %d characters", len(capture))
        return ""

    trigger = _worth_noticing(capture)
    if not trigger:
        logger.info("🫧 remark: quiet, nothing in the %d characters read is a fault", len(capture))
        return ""
    if _already_said_about(observation):
        logger.info("🫧 remark: quiet, she already spoke about this screen")
        return ""
    said = await _compose(observation, trigger)
    logger.info("🫧 remark: %s (%r seen)", f"{len(said)} characters formed" if said else "nothing formed", trigger)
    return said


def _worth_noticing(capture: str) -> str:
    """The cheap gate. Returns the trigger phrase, or "".

    No model call. Most ticks end here, which is the point: the expensive
    judgment must be rare or the loop is not affordable.
    """
    lowered = capture.lower()
    if any(context in lowered for context in _NOISE_CONTEXTS):
        return ""
    for marker in _FAULT_MARKERS:
        if marker in lowered:
            # A bare "error:" needs corroboration — the word appears in
            # ordinary prose, in code that handles errors, and in every log
            # viewer. A traceback does not.
            if marker in {"error:", "failed:"} and lowered.count(marker) < 2:
                continue
            return marker
    return ""


def _already_said_about(observation: Any) -> bool:
    """Has she already spoken about this exact screen?

    Repeating herself about an unchanged screen is the fastest way to become
    something you mute.
    """
    try:
        from core.perception.ambient_presence import get_ambient_presence

        state = get_ambient_presence().state()
    # not a failure: no ambient presence means there is no utterance in the room.
    except (ImportError, AttributeError, RuntimeError):
        return False
    if state.get("has_utterance"):
        return True
    return False


#: What her generation of a remark is admitted as, and named in her log.
_ORIGIN = "ambient_remark"


async def _compose(observation: Any, trigger: str) -> str:
    """Her next turn in her conversation with him, with what the screen shows as a reading; "" when nothing formed.

    This asked her brainstem, under "You are Aura. You are quiet by default ...", to speak "If - and only if - you can
    say something SPECIFIC and USEFUL", to reply "exactly: NOTHING" otherwise, and not to greet him: instruction on the
    path to her words (F06; Bryan's rule, no prompt engineering, ever), on a model that is not hers. Now her cortex gets
    what her remarks while watching together get (core/together/remarks.py): her recent conversation and the screen,
    and the sentence is hers (core/brain/her_unprompted_turn.py). Whether a remark is made at all is decided before
    this and by her Will after it.
    """
    del trigger  # what made her look is the gate's; what she says of it is hers
    from core.brain.her_unprompted_turn import her_turn_on
    from core.together.remarks import SCREEN_CHARS

    capture = str(getattr(observation, "capture", "") or "")
    source = str(getattr(observation, "source", "") or "")
    screen = "\n".join(part for part in (source, capture[:SCREEN_CHARS]) if part)
    return await her_turn_on("WHAT IS ON THE SCREEN", screen, origin=_ORIGIN, least_s=_COMPOSE_TIMEOUT_S)


async def compose_at_the_desk(observation: Any) -> str:
    """Her turn while she keeps a person company at the desk: the screen as a reading, and her desk tools offered.

    Whether to point at something there, and at what, is her call in that turn (core/brain/her_unprompted_turn.py,
    her_turn_at_the_desk). What she says is her words after it.
    """
    from core.brain.her_unprompted_turn import her_turn_at_the_desk
    from core.together.remarks import SCREEN_CHARS

    capture = str(getattr(observation, "capture", "") or "")
    source = str(getattr(observation, "source", "") or "")
    screen = "\n".join(part for part in (source, capture[:SCREEN_CHARS]) if part)
    return await her_turn_at_the_desk("WHAT IS ON THE SCREEN", screen, least_s=_COMPOSE_TIMEOUT_S)


def _latest_observation() -> Any:
    try:
        from core.perception.observation_evidence import (
            ObservationKind,
            get_observation_memory,
        )

        return get_observation_memory().latest(ObservationKind.SCREEN_TEXT)
    # not a failure: no observation memory means there is no latest screen text.
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return None


def latest_screen() -> Any:
    """The newest screen reading her screen sense took, or None."""
    return _latest_observation()


__all__ = ["UTTERANCE_SCHEMA", "compose_at_the_desk", "consider_utterance", "latest_screen"]
