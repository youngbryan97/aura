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
import time
from typing import Any

__all__ = ["look_with_her_eyes", "how_long_a_look_takes"]

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
