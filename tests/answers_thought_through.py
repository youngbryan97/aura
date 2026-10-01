"""A screen of answers with every one thought about, the way the loop makes them.

`_answer_each_question` hands back each answer with its thinking still to be
done, because the loop asks for it at the moment that answer is made. A test of
what she says for each one runs the thinking in the same order.
"""
from __future__ import annotations

from typing import Any


async def answered_and_thought(skill: Any, *args: Any, **kwargs: Any) -> Any:
    decision = await skill._answer_each_question(*args, **kwargs)
    for item in (decision or {}).get("resolved_actions") or []:
        think = item.get("think")
        if callable(think):
            await think()
    return decision
