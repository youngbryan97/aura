"""Her model, asked for typed data between people's turns: what work a person asked for is done with.

The work goes on beside the conversation, and a person's turn comes before any part of it. So an ask waits while anybody
is being answered, is made under a lease of her model that gives way when somebody speaks (core/brain/
llm_health_router_gate_owners.py), and is made again after them when it had to. It goes to her cortex's client as her own
goal pass does, with the words of the ask and nothing beside them.

LIVE 23 November 2026, the first build begun from a turn asked through her router while she was answering that turn. Her
reply waited 98 s behind it; the ask was stopped for her, and the gate took an ask stopped on purpose for one that had
failed and made it again at once, with 384 tokens in place of 1,536, so what came back was cut off in the middle.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from pydantic import BaseModel

from core.understanding.workbench import PutDown, this_work

logger = logging.getLogger("Aura.Understanding")

__all__ = ["ORIGIN", "ask", "nobody_waiting", "write"]

#: What names this work to her lane, her log and her conversation.
ORIGIN = "work_asked_for"
#: The least time an ask is given, and how long it waits for the lane behind her own background work.
LEAST_S, WAITS_FOR_THE_LANE_S = 120.0, 600.0


async def nobody_waiting() -> None:
    """Until nobody is being answered or waiting to be."""
    from core.brain.llm_health_router_gate_owners import a_person_is_waiting_for_her_model

    while a_person_is_waiting_for_her_model():  # noqa: ASYNC110 - the turns are counted in a module of their own, with no event to wait on
        await asyncio.sleep(0.5)


def _somebody_is_waiting() -> bool:
    from core.brain.llm_health_router_gate_owners import a_person_is_waiting_for_her_model

    return bool(a_person_is_waiting_for_her_model())


def _going_on(asked: str, so_far: str) -> str:
    """The exchange with what she had written of her answer in her turn: what she writes next goes on from it."""
    return f"<|im_start|>user\n{asked}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{so_far}"


async def _on_the_lane(asked: str, max_tokens: int, *, so_far: str = "", held: list[str] | None = None, **how: Any) -> str | None:
    """Her cortex's answer under one lease; "" where it wrote nothing, None where the lane was not to be had. ``so_far``
    is what she had written of it before she gave way, which she goes on from; ``held`` is given what she has written."""
    from core.brain.llm.mlx_client import time_a_prompt_needs
    from core.brain.llm.what_is_being_written import watched
    from core.brain.llm_health_router import (
        acquire_external_generation_gate_lease,
        release_external_generation_gate_lease,
    )
    from core.runtime.service_access import resolve_inference_gate

    client = getattr(resolve_inference_gate(default=None), "_mlx_client", None)
    if client is None or not hasattr(client, "generate_text_async"):
        return None
    work = this_work()
    seconds = max(LEAST_S, 2.0 * time_a_prompt_needs(len(asked), max_tokens))
    if work is not None:
        work.waits("her model, which is busy with something else of hers")
    lease = await acquire_external_generation_gate_lease(owner=f"{ORIGIN}:between_turns", timeout_s=seconds,
                                                         wait_s=min(seconds, WAITS_FOR_THE_LANE_S))
    if lease is None:
        return None
    if work is not None:
        work.waits("her model to read what she is given")

    async def written() -> str:
        with_it = {"origin": ORIGIN, "max_tokens": int(max_tokens), "internal_inference": True, "foreground_request": True,
                   "timeout": seconds, **how}
        if so_far:
            return str(await client.generate(_going_on(asked, so_far), **with_it) or "")
        return str(await client.generate_text_async(asked, messages=[{"role": "user", "content": asked}], system_prompt="", **with_it) or "")

    def seen(text: str, tokens: int) -> None:
        if held is not None:
            held[:] = [text]
        if work is not None:
            work.writing(so_far + text, tokens)

    try:
        if work is None:
            return await asyncio.wait_for(written(), timeout=seconds)
        with watched(seen):
            task = asyncio.ensure_future(written())
            put_down = False
            while not task.done():
                await asyncio.wait({task}, timeout=0.5)
                if not put_down and not task.done() and (work.stopped or work.somebody_spoke()):
                    put_down = True
                    from core.brain.llm.mlx_client import soft_cancel_active_generations

                    soft_cancel_active_generations(reason="somebody spoke to the work it is for")
            return task.result()
    finally:
        if work is not None:
            work.wrote_it()
            work.waits("")
        release_external_generation_gate_lease(lease)


async def _asked_until_answered(asked: str, max_tokens: int, *, goes_on: bool = False, **how: Any) -> str:
    """The answer to an ask, made again after anybody it gave way to; or gone on with, where it is plain writing: a word
    in the conversation does not cost her what she had written."""
    from core.rebuilding.her_model import HerModelIsAwayError

    work = this_work()
    so_far = ""
    while True:
        held: list[str] = []
        if work is not None:
            await work.may_go_on()
            if _somebody_is_waiting():
                work.waits("the conversation, which comes first")
        await nobody_waiting()
        try:
            text = await _on_the_lane(asked, max_tokens, so_far=so_far, held=held, **how)
        except (RuntimeError, TimeoutError, OSError) as why:
            if _somebody_is_waiting():
                so_far += held[0] if goes_on and held else ""
                continue
            raise HerModelIsAwayError(f"it could not be asked: {type(why).__name__}: {str(why)[:160]}") from why
        if work is not None and (work.stopped or work.somebody_spoke()):
            raise PutDown("it was stopped" if work.stopped else "somebody spoke to it")
        if text and re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip():
            return so_far + text
        if _somebody_is_waiting():
            so_far += held[0] if goes_on and held else ""
            logger.info("🧰 An ask for work gave way to somebody speaking, and %s after them.",
                        f"goes on from {len(so_far)} characters" if so_far else "is made again")
            continue
        raise HerModelIsAwayError("an empty answer" if text is not None else "its lane is held for something else")


async def ask(prompt: str, schema: type[BaseModel], max_tokens: int) -> BaseModel | None:
    """Her cortex's answer to ``prompt`` as an instance of ``schema``, or None when it gave none that fits."""
    from core.rebuilding.her_model import as_answered

    asked = f"{prompt}\n\nAnswer with one JSON object that follows this JSON schema:\n{json.dumps(schema.model_json_schema())}"
    text = await _asked_until_answered(asked, max_tokens, schema=schema.model_json_schema(), output_shape="json_object",
                                       temperature=0.2)
    return as_answered(text, schema)


async def write(prompt: str, max_tokens: int, *, warmth: float = 0.2) -> str:
    """What her cortex writes for ``prompt``, as it wrote it: prose, code, or the one around the other."""
    text = await _asked_until_answered(prompt, max_tokens, goes_on=True, temperature=warmth, cognitive_mode="fast")
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
