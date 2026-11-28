"""Saying something while watching together, when it is worth saying now.

Unless she is hidden she is in a with beside the person, keeping them
company at the desk if they asked her into nothing else
(core/perception/ambient_presence.py), and a remark is part of what they are
doing (core/together/a_with.py). Each look her screen sense takes goes here:

    what the screen shows becomes the with's shared field, with how much it
    changed since the with last saw it;
    how much it interests her is read off her record: the strongest like or
    dislike among the things it names, her stake in what it is about, and how
    new it is;
    whether to say something now is the with's own expected utility
    (core/together/when_to_say_something.py).

If she speaks, her model is handed what was said in the with and what the
screen shows now as a reading, and nothing else. It is not told to be brief,
to be useful, or to say nothing; it is her next turn in a conversation in
which something is on the screen.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from core.runtime.errors import record_degradation
from core.together.a_with import HER, With
from core.together.when_to_say_something import worth_saying_now

logger = logging.getLogger("Aura.Together")

__all__ = ["interest_in", "remark_in_a_with", "said_since"]

#: How much of the screen her model is shown: the bound her own screen
#: reading uses (core/brain/observable_registry.py, `_read_screen`).
SCREEN_CHARS = 2000

#: How long a remark may take to compose before the screen has moved on; the
#: bound the ambient remark already holds to (core/perception/ambient_utterance.py).
_COMPOSE_TIMEOUT_S = 25.0


def _interest(capture: str, change: float | None) -> tuple[float, str]:
    """How much what is on the screen interests her, and what it is most about."""
    try:
        from core.conation.what_she_wants import get_what_she_wants

        record = get_what_she_wants()
        # What it is about, where it has been read; where not, read beside her for the next look, which on a still
        # screen sees the same words.
        about = [f"topic:{subject.title}" for subject in record.about.read_soon(capture) or []]
        regard = abs(record.regard_for_text(capture))
        stakes = {key: record.care.stake(key) for key in about}
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("together", exc, severity="warning", action="what is on the screen was weighed as new only")
        about, regard, stakes = [], 0.0, {}
    thing = max(stakes, key=lambda key: (stakes[key], key)) if stakes else (about[0] if about else "")
    return max([regard, *stakes.values(), change or 0.0]), thing


def said_since(began: float) -> list[tuple[float, str, str]]:
    """The conversation since ``began``, from the one she keeps: when, ``user`` or ``assistant``, and what was said."""
    try:
        from core.conversation.persistence import get_persistence

        stored = get_persistence().get_recent_history(limit=256)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("together", exc, severity="warning", action="a remark was composed without the with's talk")
        return []
    turns: list[tuple[float, str, str]] = []
    for turn in stored:
        role = str(turn.get("role", "")).lower()
        role = "assistant" if role in {"aura", "assistant"} else "user" if role == "user" else ""
        content = str(turn.get("content", "") or "").strip()
        try:
            at = float(turn.get("created_at") or 0.0)
        except (TypeError, ValueError):
            at = 0.0  # not a failure: a turn with no readable time is not one since the start
        if role and content and at >= began:
            turns.append((at, role, content))
    return turns


def interest_in(seen: str, change: float | None) -> tuple[float, str]:
    """How much what is in front of her interests her, and what it is most about: her record's strongest like or
    dislike among what it names, her stake in what it is about, and how new it is."""
    return _interest(seen, change)


def _what_was_said_since(began: float) -> list[dict[str, str]]:
    """The conversation since the with began, as her model reads a conversation."""
    return [{"role": role, "content": content} for _at, role, content in said_since(began)]


async def _her_remark(held: With, capture: str, source: str) -> str:
    from core.utils.injected_blocks import RUNTIME_EVIDENCE_ROLE, stamp_grounding

    messages = _what_was_said_since(held.began)
    screen = "\n".join(part for part in (source, capture[:SCREEN_CHARS]) if part)
    if any(message["role"] == "user" for message in messages):
        messages.append(stamp_grounding({"role": RUNTIME_EVIDENCE_ROLE,
                                         "content": f"## WHAT IS ON THE SCREEN\n{screen}"}))
    else:
        # A with begun without a word carries on from the sitting going on now. Carried on from her conversation
        # however old, LIVE 16 November 2026 it answered a question of eight hours before
        # (core/brain/her_unprompted_turn.py, her_conversation_now_with). With nobody's words in the with or the
        # sitting there is no turn of hers to write here; at the desk her own turn speaks from the screen alone.
        from core.brain.her_unprompted_turn import A_LOOK_AT_THE_SCREEN, her_conversation_now_with

        messages = her_conversation_now_with("WHAT IS ON THE SCREEN", screen, looked_with=A_LOOK_AT_THE_SCREEN)
        if not any(message["role"] == "user" and str(message.get("content") or "").strip() for message in messages):
            return ""
    try:
        from core.brain.llm.llm_tier import LLMTier
        from core.container import ServiceContainer

        brain = ServiceContainer.get("llm_router", default=None)
        if brain is None:
            return ""
        response = await asyncio.wait_for(
            brain.think(
                "", messages=messages, system_prompt="", priority=0.2, is_background=True,
                prefer_tier=LLMTier.SECONDARY,
            ),
            timeout=_COMPOSE_TIMEOUT_S,
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, OSError, TimeoutError) as exc:
        record_degradation("together", exc, severity="warning", action="she had something to say and it was not composed")
        return ""
    text = response[1] if isinstance(response, tuple) and len(response) > 1 else response
    return str(text or "").strip()


def _said_already(text: str) -> bool:
    """Whether these words are one of her own recent turns, said again.

    LIVE 15 November 2026, her remark at the desk was her chat reply of an hour before, word for word ("I'm sorry
    you're waking up on so little rest ..."), composed on a background tier from a conversation that ended with it.
    """
    from core.brain.her_unprompted_turn import CONVERSATION_TURNS

    said = " ".join(str(text or "").split()).casefold()
    return any(role != "user" and " ".join(str(content or "").split()).casefold() == said
               for _at, role, content in said_since(0.0)[-CONVERSATION_TURNS:])


async def remark_in_a_with(
    held: With,
    observation: Any,
    *,
    now: float | None = None,
    compose: Any = None,
    against_usual: bool = False,
) -> str:
    """What she says about what the screen shows now, or nothing.

    ``compose`` is who writes it, given the observation; by default her remark in the with (``_her_remark``).
    ``against_usual`` weighs a look by how much more it interests her than the looks before it in this with, for a
    field nobody chose to share, a person's own screen, where most of what crosses it is ordinary. LIVE 15 November
    2026 at the desk, a still screen held no attention, so the cost of a remark was nothing and one worth 0.058 was
    made.
    """
    capture = str(getattr(observation, "capture", "") or "")
    if not capture.strip() or not held.open:
        return ""
    from core.together.taking_it_up import _how_much_changed, _what_is_in_front

    now = time.time() if now is None else now
    source = str(getattr(observation, "source", "") or "")
    change = _how_much_changed(held, capture)
    shown = _what_is_in_front(capture) or source
    held.shown(shown, now, change=change)
    interest, thing = _interest(capture, change)
    before = list(held.interests)
    held.interests.append(interest)
    if against_usual:
        beaten = sum(1 for seen in before if seen < interest) / len(before) if before else 0.5
        interest *= beaten
    say = worth_saying_now(held, now, interest=interest, thing=thing or shown)
    logger.info("Watching with %s: %s (%s).", held.who, "says something" if say.speak else "stays quiet", say.why)
    if not say.speak:
        return ""
    text = await (compose(observation) if compose is not None else _her_remark(held, capture, source))
    if text and _said_already(text):
        logger.info("Watching with %s: what formed repeats what she said, so it is not said again.", held.who)
        return ""
    if text:
        said_at = time.time()
        held.said(HER, said_at, about=[thing or shown], until=said_at, unprompted=True)
    return text
