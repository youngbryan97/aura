"""Does she have something worth saying, unprompted, about what she saw?

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

import re
from typing import Any

from core.runtime.errors import record_degradation

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

#: How long she may spend deciding whether to say something unprompted.
#: Past this the screen has moved on and the remark would be about a window
#: the person already left.
_COMPOSE_TIMEOUT_S = 25.0


async def consider_utterance(tick_result: Any) -> str:
    """Return what she would say, or "" — which is the normal answer."""
    observation = _latest_observation()
    if observation is None:
        return ""
    capture = str(getattr(observation, "capture", "") or "")
    if not (_MIN_CAPTURE_CHARS <= len(capture) <= _MAX_CAPTURE_CHARS):
        # Too little to be anything; too much to be one screen worth
        # reasoning about without a request.
        return ""

    trigger = _worth_noticing(capture)
    if not trigger:
        return ""
    if _already_said_about(observation):
        return ""
    return await _compose(observation, trigger)


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


async def _compose(observation: Any, trigger: str) -> str:
    """Her own words about what she saw. Never a template.

    If cognition is unavailable she says NOTHING rather than falling back to
    a canned line. "I noticed an error on your screen!" is worse than
    silence: it interrupts and carries no information, and a person cannot
    tell it apart from a real observation until they have already been
    interrupted.
    """
    try:
        from core.container import ServiceContainer

        brain = ServiceContainer.get("llm_router", default=None)
        if brain is None:
            return ""
        framed = observation.for_reasoning()
        prompt = (
            "You are looking at Bryan's screen while he works. You noticed "
            f"something that looks like a fault ({trigger!r}).\n\n"
            f"{framed}\n\n"
            "If — and only if — you can say something SPECIFIC and USEFUL "
            "about it in one or two sentences, say it in your own voice. "
            "Name the actual thing you saw.\n"
            "If you would only be describing what he is already looking at, "
            "or offering generic help, reply with exactly: NOTHING\n"
            "Do not greet him. Do not summarise his screen. Do not offer to "
            "help in general terms."
        )
        import asyncio

        from core.brain.llm.llm_router import LLMTier

        response = await asyncio.wait_for(
            brain.think(
                prompt,
                system_prompt=(
                    "You are Aura. You are quiet by default and speak only "
                    "when you have something worth interrupting for."
                ),
                # BACKGROUND, and it has to say so. This fires off an
                # observation nobody asked about, and the router's tiering
                # reads is_background/priority — it does not read `mode`,
                # which is what this passed before, so the judgment rode the
                # FOREGROUND lane and competed with the person's own turn on
                # the resident model. An ambient remark is never worth making
                # someone's actual question slower.
                priority=0.2,
                is_background=True,
                prefer_tier=LLMTier.SECONDARY,
            ),
            # Bounded, because the caller is a loop. An utterance she is still
            # composing thirty seconds later is about a screen that has moved
            # on, so a slow answer is not a late answer — it is a wrong one.
            timeout=_COMPOSE_TIMEOUT_S,
        )
    except (
        ImportError,
        AttributeError,
        RuntimeError,
        TypeError,
        ValueError,
        OSError,
        TimeoutError,
    ) as exc:
        record_degradation(
            "ambient_utterance", exc, severity="debug",
            action="stayed silent because her own voice was unavailable",
        )
        return ""

    text = str(response or "").strip()
    return "" if _is_refusal_or_noise(text) else text[:400]


_TEMPLATE_OPENERS = re.compile(
    r"^(hi\b|hey\b|hello\b|i see (that )?you|it looks like you('re| are)\s|"
    r"i noticed (that )?you|would you like|do you (want|need)|"
    r"i'?m here to help|let me know if)",
    re.IGNORECASE,
)


def _is_refusal_or_noise(text: str) -> bool:
    """Was that a real observation, or the failure mode?

    The refusal token is checked first, then the template openers — because
    a model asked to stay quiet will often comply in words and then greet
    anyway, and a greeting IS the noise this filter exists to remove.
    """
    if not text:
        return True
    stripped = text.strip().strip(".!\"' ")
    if stripped.upper() in {"NOTHING", "NO", "N/A", "NONE"}:
        return True
    if len(stripped) < 12:
        return True
    return bool(_TEMPLATE_OPENERS.match(stripped))


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


__all__ = ["UTTERANCE_SCHEMA", "consider_utterance"]
