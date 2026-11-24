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

from pydantic import BaseModel

logger = logging.getLogger("Aura.Understanding")

__all__ = ["ORIGIN", "ask", "nobody_waiting"]

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


async def _on_the_lane(asked: str, schema: type[BaseModel], max_tokens: int) -> str | None:
    """Her cortex's answer under one lease; "" where it wrote nothing, None where the lane was not to be had."""
    from core.brain.llm.mlx_client import time_a_prompt_needs
    from core.brain.llm_health_router import (
        acquire_external_generation_gate_lease,
        release_external_generation_gate_lease,
    )
    from core.runtime.service_access import resolve_inference_gate

    client = getattr(resolve_inference_gate(default=None), "_mlx_client", None)
    if client is None or not hasattr(client, "generate_text_async"):
        return None
    seconds = max(LEAST_S, 2.0 * time_a_prompt_needs(len(asked), max_tokens))
    lease = await acquire_external_generation_gate_lease(owner=f"{ORIGIN}:between_turns", timeout_s=seconds,
                                                         wait_s=min(seconds, WAITS_FOR_THE_LANE_S))
    if lease is None:
        return None
    try:
        return str(await asyncio.wait_for(client.generate_text_async(
            asked, messages=[{"role": "user", "content": asked}], system_prompt="", origin=ORIGIN,
            schema=schema.model_json_schema(), output_shape="json_object", max_tokens=int(max_tokens), temperature=0.2,
            internal_inference=True, foreground_request=True, timeout=seconds), timeout=seconds) or "")
    finally:
        release_external_generation_gate_lease(lease)


async def ask(prompt: str, schema: type[BaseModel], max_tokens: int) -> BaseModel | None:
    """Her cortex's answer to ``prompt`` as an instance of ``schema``, or None when it gave none that fits."""
    from core.rebuilding.her_model import HerModelIsAwayError, as_answered

    asked = f"{prompt}\n\nAnswer with one JSON object that follows this JSON schema:\n{json.dumps(schema.model_json_schema())}"
    while True:
        await nobody_waiting()
        try:
            text = await _on_the_lane(asked, schema, max_tokens)
        except (RuntimeError, TimeoutError, OSError) as why:
            if _somebody_is_waiting():
                continue
            raise HerModelIsAwayError(f"it could not be asked: {type(why).__name__}: {str(why)[:160]}") from why
        if text and re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip():
            return as_answered(text, schema)
        if _somebody_is_waiting():
            logger.info("🧰 An ask for work gave way to somebody speaking, and is made again after them.")
            continue
        raise HerModelIsAwayError("an empty answer" if text is not None else "its lane is held for something else")
