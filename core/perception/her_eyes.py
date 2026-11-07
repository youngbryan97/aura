"""Her eyes, one look at a time: the vision model asked of a picture, by whatever wants to know what is in it.

What a thing in play is (core/perception/what_things_look_like.py) and what a screen is of at a glance
(core/perception/what_a_scene_is.py) both ask the same eyes (core/brain/llm/mlx_vision_client.py). Asked at once, each
started the vision worker for itself, and the second start was refused its model lane and stopped the worker the first
had just brought up (LIVE 2026-10-09). A person looks at one thing at a time too: here the looks wait their turn, and
each is timed, so how long her eyes take is known rather than guessed.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

__all__ = ["look_with_her_eyes", "how_long_a_look_takes", "plainly"]

logger = logging.getLogger("Aura.HerEyes")

_TURN: dict[int, Any] = {}
_TOOK: list[float] = []


def _turn() -> Any:
    """The one turn at her eyes for the running event loop."""
    from core.runtime.lockdep import checked_async_lock

    loop = asyncio.get_running_loop()
    lock = _TURN.get(id(loop))
    if lock is None:
        lock = _TURN[id(loop)] = checked_async_lock("her_eyes.turn")
    return lock


#: The vision model answers a look as a chat reply to whoever sent the picture. LIVE 20 October 2026, every look of a
#: watching opened "Based on the image you provided, here's what is happening: You are watching a short film".
_TO_WHOEVER_SENT_IT = re.compile(
    r"^\s*(?:(?:based on|from|in|looking at) (?:the|this) (?:image|picture|screenshot|frame)"
    r"(?: (?:you |you've |you have )?(?:provided|sent|shared))?(?:,? here(?:'|’)?s what(?:'s| is) happening)?\s*[:,.]\s*)?"
    r"(?:you are (?:watching|looking at|seeing|viewing) )?"
    r"(?:(?:this|the) (?:picture|image|screenshot|frame) (?:shows|depicts|is|appears to (?:be|show)) )?", re.IGNORECASE)


#: Where a look says which work a picture is from, not what is in it: a guess the picture does not show. LIVE 7 November
#: 2026, watching Rick Astley's "Never Gonna Give You Up", her eyes called frames of it a still from "the 1972 music video
#: for 'A Full Commitment'", from "the 1987 Sesame Street episode 'The Song of the Glass'", and from a-ha's "Take On Me";
#: the last moved her most of anything in it. What a frame shows (a man singing at a microphone in a tunnel) holds.
_WHICH_WORK_IT_IS_FROM = re.compile(
    r"(?:^|(?<=[.!?]\s))(?:\*\*)?(?:a|an|this is an?|the)\s+(?:\*\*)?(?:still|scene|screenshot|frame|clip|shot|image|moment)"
    r"(?:\*\*)?\s+(?:from|of)\s+[^.!?\n]{0,240}?(?:music video|film|movie|episode|show|series|sketch|trailer|video|song)"
    r"[^.!?\n]{0,240}[.!?:;]\s*", re.IGNORECASE)
_A_YEAR_IT_GUESSES = re.compile(r",?\s*(?:released|from|filmed|recorded)\s+(?:in\s+)?(?:19|20)\d\d\b", re.IGNORECASE)


def plainly(answer: str) -> str:
    """What a look saw, without the reply's address to whoever sent the picture, and without its guess at which work
    the picture is from."""
    seen = _TO_WHOEVER_SENT_IT.sub("", str(answer or "").strip(), count=1).strip()
    seen = _A_YEAR_IT_GUESSES.sub("", _WHICH_WORK_IT_IS_FROM.sub("", seen)).strip()
    # A look cut at its budget ends at its last whole sentence.
    whole = re.match(r"(?s)^(.*[.!?])(?:\s|$)", seen)
    seen = whole.group(1) if whole and not re.search(r"[.!?][\"'\u201d\u2019)*_]*$", seen) else seen
    return seen[:1].upper() + seen[1:] if seen else ""


def how_long_a_look_takes() -> float:
    """The middle of the last looks' lengths in seconds, 0.0 before any."""
    took = sorted(_TOOK[-9:])
    return took[len(took) // 2] if took else 0.0


async def look_with_her_eyes(prompt: str, image_b64: str, *, max_tokens: int, timeout_s: float) -> str:
    """Her eyes' answer to ``prompt`` about the picture, once it is this look's turn."""
    from core.brain.llm.mlx_vision_client import get_vision_client

    async with _turn():
        began = time.monotonic()
        answer = await get_vision_client().see_async(prompt, image_b64, max_tokens=max_tokens, temp=0.0,
                                                     timeout_s=timeout_s)
        took = time.monotonic() - began
        _TOOK.append(took)
        del _TOOK[:-20]
        logger.info("her eyes took %.1fs for a look", took)
        return answer
