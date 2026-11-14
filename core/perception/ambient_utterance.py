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

import logging
from typing import Any

from core.runtime.errors import record_degradation

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


#: How much of her conversation with him her model reads before what the screen shows: the last few turns.
_CONVERSATION_TURNS = 6


async def _compose(observation: Any, trigger: str) -> str:
    """Her next turn in her conversation with him, with what the screen shows as a reading; "" when nothing formed.

    This asked her brainstem, under "You are Aura. You are quiet by default ...", to speak "If - and only if - you can
    say something SPECIFIC and USEFUL", to reply "exactly: NOTHING" otherwise, and not to greet him: instruction on the
    path to her words (F06; Bryan's rule, no prompt engineering, ever), on a model that is not hers. Now her cortex gets
    what her remarks while watching together get (core/together/remarks.py): her recent conversation and the screen,
    and the sentence is hers. Whether a remark is made at all is decided before this and by her Will after it.
    """
    del trigger  # what made her look is the gate's; what she says of it is hers
    from core.together.remarks import SCREEN_CHARS, said_since
    from core.utils.injected_blocks import RUNTIME_EVIDENCE_ROLE, stamp_grounding

    messages = [{"role": role, "content": content} for _at, role, content in said_since(0.0)[-_CONVERSATION_TURNS:]]
    if not any(message["role"] == "user" for message in messages):
        # Her model's template has no turn of hers without somebody's before it.
        return ""
    capture = str(getattr(observation, "capture", "") or "")
    source = str(getattr(observation, "source", "") or "")
    screen = "\n".join(part for part in (source, capture[:SCREEN_CHARS]) if part)
    messages.append(stamp_grounding({"role": RUNTIME_EVIDENCE_ROLE, "content": f"## WHAT IS ON THE SCREEN\n{screen}"}))
    try:
        text = await _on_her_cortex(messages)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, OSError, TimeoutError) as exc:
        record_degradation(
            "ambient_utterance", exc, severity="debug",
            action="stayed silent because her own voice was unavailable",
        )
        return ""
    return str(text or "").strip()[:_REMARK_CHARS]


#: What her generation of a remark is admitted as, and named in her log.
_ORIGIN = "ambient_remark"

#: The most of a remark the bubble shows.
_REMARK_CHARS = 400


async def _on_her_cortex(messages: list[dict[str, Any]]) -> str:
    """Her cortex's next turn on these messages, admitted as her goal pass is; "" when her model is not free for it.

    Asked of her router as background work, a remark never reached her cortex: the router sends background work to its
    tertiary tier. LIVE 14 November 2026, 14:22 to 14:28, a traceback on the screen and her in the bubble: eleven times
    "Tier Lock: Background task requested 'primary'; using the governed tertiary tier", then "no endpoints matched"
    (her brainstem was out), and nothing formed. Her goal pass reaches her cortex through its client under the
    generation lane's lease (core/agency/her_goal_at_work.py). The lease is a background one, so a person's turn asks
    it to yield, and it is not asked for while a person is waiting for her model. The time it may take is what this
    prompt needs on this machine, and at least as long as the screen is worth a remark.
    """
    import asyncio

    from core.brain.llm.mlx_client import time_a_prompt_needs
    from core.brain.llm_health_router import (
        _BACKGROUND_GENERATION_GATE_WAIT_S,
        a_person_is_waiting_for_her_model,
        acquire_external_generation_gate_lease,
        release_external_generation_gate_lease,
    )
    from core.runtime.service_access import resolve_inference_gate
    from core.runtime.structured_input import A_CLOSED_QUESTIONS_FLOOR

    if a_person_is_waiting_for_her_model():
        logger.info("🫧 remark: her model is someone's now")
        return ""
    client = getattr(resolve_inference_gate(default=None), "_mlx_client", None)
    if client is None or not hasattr(client, "generate_text_async"):
        return ""
    prompt_chars = sum(len(str(message.get("content") or "")) for message in messages)
    seconds = max(_COMPOSE_TIMEOUT_S, time_a_prompt_needs(prompt_chars, A_CLOSED_QUESTIONS_FLOOR))
    lease = await acquire_external_generation_gate_lease(
        owner=f"{_ORIGIN}:her_own_turn", timeout_s=seconds, wait_s=_BACKGROUND_GENERATION_GATE_WAIT_S)
    if lease is None:
        logger.info("🫧 remark: her model is held for something else")
        return ""
    try:
        return await asyncio.wait_for(
            client.generate_text_async(
                "", messages=messages, system_prompt="", origin=_ORIGIN, is_background=True,
                max_tokens=A_CLOSED_QUESTIONS_FLOOR, timeout=seconds),
            timeout=seconds,
        ) or ""
    finally:
        release_external_generation_gate_lease(lease)


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


__all__ = ["UTTERANCE_SCHEMA", "consider_utterance", "latest_screen"]
