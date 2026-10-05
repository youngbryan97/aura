"""Asking her own model for typed data, with room enough for a part of a program.

The same way in as every structured ask she makes (the router, the shape the
decoder holds to a JSON object), with a token budget and a deadline that fit
writing code: a part of a program runs to a few thousand tokens, and the
person is waiting on the turn that asked for it.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, ValidationError

logger = logging.getLogger("Rebuilding.HerModel")

__all__ = ["ask_her_model", "the_object_in", "what_was_finished"]

#: The longest one ask may take, in seconds.
ASK_S = 900.0


def the_object_in(text: str) -> str | None:
    """The first whole JSON object in ``text``, past any thinking aloud before it."""
    text = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.S)
    start = text.find("{")
    if start < 0:
        return None
    depth, in_string, escaped = 0, False, False
    for at in range(start, len(text)):
        ch = text[at]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : at + 1]
    return None


def what_was_finished(text: str) -> str | None:
    """An answer cut off by its token budget, closed after its last finished item.

    A list of thirty features stopped inside the twenty-ninth is twenty-eight
    features, not nothing: everything before the last whole value is kept and
    the brackets still open are closed.
    """
    text = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.S)
    start = text.find("{")
    if start < 0:
        return None
    ends = [at for at in range(len(text) - 1, start, -1) if text[at] in "}]\""][:400]
    for end in ends:
        cut = text[start : end + 1]
        opened: list[str] = []
        in_string = escaped = False
        for ch in cut:
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch in "{[":
                opened.append("}" if ch == "{" else "]")
            elif ch in "}]" and opened:
                opened.pop()
        if in_string:
            continue
        candidate = cut + "".join(reversed(opened))
        try:
            json.loads(candidate)
        except ValueError:
            continue
        return candidate
    return None


async def ask_her_model(prompt: str, schema: type[BaseModel], max_tokens: int, *, router: Any = None) -> BaseModel | None:
    """Her model's answer to ``prompt`` as an instance of ``schema``, or None when it gave none that fits."""
    if router is None:
        from core.container import ServiceContainer

        router = ServiceContainer.get("llm_router", default=None)
    if router is None:
        logger.info("no model to ask")
        return None
    asked = f"{prompt}\n\nAnswer with one JSON object that follows this JSON schema:\n{json.dumps(schema.model_json_schema())}"
    try:
        reply = await router.generate_with_metadata(
            asked,
            None,
            ASK_S,
            prefer_tier="primary",
            schema=schema.model_json_schema(),
            output_shape="json_object",
            origin="rebuilding_a_program",
            purpose="rebuilding_a_program",
            is_background=False,
            foreground_request=True,
            max_tokens=int(max_tokens),
            temperature=0.2,
        )
    except (RuntimeError, TimeoutError, ValueError, TypeError, OSError) as why:
        logger.info("her model could not be asked: %s", why)
        return None
    text = str(reply.get("text") or "") if isinstance(reply, dict) else str(reply or "")
    found = the_object_in(text) or what_was_finished(text)
    if found is None:
        logger.info("her model answered without an object (%d chars)", len(text))
        return None
    try:
        return schema.model_validate_json(found)
    except ValidationError:
        try:
            return schema.model_validate(json.loads(found))
        except (ValidationError, ValueError) as why:
            logger.info("her model's answer did not fit %s: %s", schema.__name__, str(why)[:300])
            return None
