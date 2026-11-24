"""Her model, asked for typed data between people's turns: what work a person asked for is done with.

The work goes on beside the conversation, and a person's turn comes before any part of it. So an ask waits while anybody
is being answered, is made under a lease of her model that gives way when somebody speaks (core/brain/
llm_health_router_gate_owners.py), and is made again after them when it had to. It goes to her cortex's client as her own
goal pass does, with the words of the ask and nothing beside them.

LIVE 23 November 2026, the first build begun from a turn asked through her router while she was answering that turn. Her
reply waited 98 s behind it; the ask was stopped for her, and the gate took an ask stopped on purpose for one that had
failed and made it again at once, with 384 tokens in place of 1,536, so what came back was cut off in the middle.

LIVE 24 November 2026, the first program made this way while she was open: every write was given 91 tokens of the 4,600
asked for, because the machine's memory was at its warning line (384) and her vitality took the rest, and each of her
typed answers thought for 5,000 tokens first. What she writes here is read by a machine: it is sized by what it has to
hold, written straight, and not steered by how she feels or what she remembers. Where it was cut all the same, by room or
by somebody speaking, it goes on from where it was.
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
#: What is read by a machine is written so: no thinking aloud, no token held back because it was used before.
READ_BY_A_MACHINE: dict[str, Any] = {"read_by_a_machine": True, "cognitive_mode": "fast", "repetition_penalty": 1.0,
                                     "presence_penalty": 0.0, "prompt_cache_scope": "work_asked_for"}
#: How generation ends where something stopped it, and where its room ran out; and how often a write goes on past its room.
STOPPED, NO_ROOM, GOES_ON = {"soft_cancelled", "deadline_exceeded"}, {"max_tokens", "hard_token_limit"}, 2
#: How often a write is taken up again where something of hers, and nobody, stopped it.
TAKEN_UP_AGAIN = 12


async def nobody_waiting() -> None:
    """Until nobody is being answered or waiting to be."""
    from core.brain.llm_health_router_gate_owners import a_person_is_waiting_for_her_model

    while a_person_is_waiting_for_her_model():  # noqa: ASYNC110 - the turns are counted in a module of their own, with no event to wait on
        await asyncio.sleep(0.5)


def _somebody_is_waiting() -> bool:
    from core.brain.llm_health_router_gate_owners import a_person_is_waiting_for_her_model

    return bool(a_person_is_waiting_for_her_model())


def _messages(asked: str, given: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text} for text in (given, asked) if text]


def _going_on(asked: str, so_far: str, given: str = "") -> str:
    """The exchange with what she had written of her answer in her turn: what she writes next goes on from it."""
    before = "".join(f"<|im_start|>user\n{message['content']}<|im_end|>\n" for message in _messages(asked, given))
    return f"{before}<|im_start|>assistant\n<think>\n\n</think>\n\n{so_far}"


async def _memory_to_write_in(work: Any) -> None:
    """Until the machine's memory is off its critical line, past which her model is given 64 tokens whatever was asked."""
    from core.utils.memory_monitor import get_memory_pressure_snapshot

    waited = 0.0
    while (await asyncio.to_thread(get_memory_pressure_snapshot)).critical and waited < WAITS_FOR_THE_LANE_S:
        if work is not None:
            work.waits("the machine to have memory free")
            await work.may_go_on()
        await asyncio.sleep(5.0)
        waited += 5.0


async def _on_the_lane(asked: str, max_tokens: int, *, so_far: str = "", held: list[str] | None = None, given: str = "",
                       **how: Any) -> tuple[str | None, str]:
    """Her cortex's answer under one lease and how it ended; "" where it wrote nothing, None where the lane was not to be
    had. ``so_far`` is what she had written of it before she gave way, which she goes on from; ``held`` is given what she
    has written."""
    from core.brain.llm.mlx_client import time_a_prompt_needs
    from core.brain.llm.what_is_being_written import watched
    from core.brain.llm_health_router import (
        acquire_external_generation_gate_lease,
        release_external_generation_gate_lease,
    )
    from core.runtime.service_access import resolve_inference_gate

    client = getattr(resolve_inference_gate(default=None), "_mlx_client", None)
    if client is None or not hasattr(client, "generate_text_async"):
        return None, ""
    work = this_work()
    ended: dict[str, Any] = {}
    await _memory_to_write_in(work)
    seconds = max(LEAST_S, 2.0 * time_a_prompt_needs(len(given) + len(asked), max_tokens))
    if work is not None:
        work.waits("her model, which is busy with something else of hers")
    lease = await acquire_external_generation_gate_lease(owner=f"{ORIGIN}:between_turns", timeout_s=seconds,
                                                         wait_s=min(seconds, WAITS_FOR_THE_LANE_S))
    if lease is None:
        return None, ""
    if work is not None:
        work.waits("her model to read what she is given")

    async def written() -> str:
        with_it = {"origin": ORIGIN, "max_tokens": int(max_tokens), "internal_inference": True, "foreground_request": True,
                   "timeout": seconds, "_generation_result_sink": ended, **READ_BY_A_MACHINE, **how}
        if so_far:
            return str(await client.generate(_going_on(asked, so_far, given), **with_it) or "")
        return str(await client.generate_text_async(asked, messages=_messages(asked, given), system_prompt="", **with_it) or "")

    def seen(text: str, tokens: int) -> None:
        if held is not None:
            held[:] = [text]
        if work is not None:
            work.writing(so_far + text, tokens)

    def how_it_ended() -> str:
        return str((ended.get("surface_control_receipt") or {}).get("generation_stop_reason") or "")

    try:
        if work is None:
            return await asyncio.wait_for(written(), timeout=seconds), how_it_ended()
        with watched(seen):
            task = asyncio.ensure_future(written())
            put_down = False
            while not task.done():
                await asyncio.wait({task}, timeout=0.5)
                if not put_down and not task.done() and (work.stopped or work.somebody_spoke()):
                    put_down = True
                    from core.brain.llm.mlx_client import soft_cancel_active_generations

                    soft_cancel_active_generations(reason="somebody spoke to the work it is for")
            return task.result(), how_it_ended()
    finally:
        if work is not None:
            work.wrote_it()
            work.waits("")
        release_external_generation_gate_lease(lease)


async def _asked_until_answered(asked: str, max_tokens: int, *, goes_on: bool = False, **how: Any) -> str:
    """The answer to an ask, made again after anybody it gave way to; or gone on with, where it is plain writing: a word
    in the conversation does not cost her what she had written, and neither does running out of room."""
    from core.rebuilding.her_model import HerModelIsAwayError

    work = this_work()
    so_far = ""
    out_of_room = stopped = 0
    while True:
        held: list[str] = []
        if work is not None:
            await work.may_go_on()
            if _somebody_is_waiting():
                work.waits("the conversation, which comes first")
        await nobody_waiting()
        try:
            text, ended = await _on_the_lane(asked, max_tokens, so_far=so_far, held=held, **how)
        except (RuntimeError, TimeoutError, OSError) as why:
            if _somebody_is_waiting():
                so_far += held[0] if goes_on and held else ""
                continue
            raise HerModelIsAwayError(f"it could not be asked: {type(why).__name__}: {str(why)[:160]}") from why
        if work is not None and (work.stopped or work.somebody_spoke()):
            raise PutDown("it was stopped" if work.stopped else "somebody spoke to it")
        if ended in STOPPED and (_somebody_is_waiting() or stopped < TAKEN_UP_AGAIN):
            # Part of an answer, stopped for somebody or by the clock: not the answer.
            stopped += not _somebody_is_waiting()
            so_far += max([text or "", *held], key=len) if goes_on else ""
            logger.info("🧰 An ask for work was stopped (%s), and %s.", ended,
                        f"goes on from {len(so_far)} characters" if so_far else "is made again")
            continue
        if text and re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip():
            # Past its room it goes on where what it is in the middle of is code: prose that fills its room is not getting there.
            if goes_on and ended in NO_ROOM and out_of_room < GOES_ON and (so_far + text).count("```") % 2:
                out_of_room += 1
                so_far += text
                logger.info("🧰 A write ran out of its room of %d tokens, and goes on from %d characters.", max_tokens, len(so_far))
                continue
            return so_far + text
        if _somebody_is_waiting():
            so_far += held[0] if goes_on and held else ""
            logger.info("🧰 An ask for work gave way to somebody speaking, and %s after them.",
                        f"goes on from {len(so_far)} characters" if so_far else "is made again")
            continue
        raise HerModelIsAwayError("an empty answer" if text is not None else "its lane is held for something else")


async def ask(prompt: str, schema: type[BaseModel], max_tokens: int, *, given: str = "") -> BaseModel | None:
    """Her cortex's answer to ``prompt`` as an instance of ``schema``, or None when it gave none that fits. ``given`` is
    what comes before it and before the asks after it alike, which she reads once."""
    from core.rebuilding.her_model import as_answered

    asked = f"{prompt}\n\nAnswer with one JSON object that follows this JSON schema:\n{json.dumps(schema.model_json_schema())}"
    # The decoder holds it to one JSON object; whether it fits is read here, where what only overran a bound is cut to it.
    # LIVE 24 November 2026 a design with one list too long was written three times over in her worker, the same 2,484
    # tokens each time, for 200 s of it.
    text = await _asked_until_answered(asked, max_tokens, output_shape="json_object", temperature=0.2, given=given)
    return as_answered(text, schema)


async def write(prompt: str, max_tokens: int, *, warmth: float = 0.2, given: str = "") -> str:
    """What her cortex writes for ``prompt``, as it wrote it: prose, code, or the one around the other."""
    text = await _asked_until_answered(prompt, max_tokens, goes_on=True, temperature=warmth, given=given)
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
