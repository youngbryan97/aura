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

__all__ = ["HerModelIsAwayError", "ask_her_model", "patiently", "the_object_in", "what_was_finished"]

#: The longest one ask may take, in seconds.
ASK_S = 900.0

#: How long a build waits for her model to come back from one absence, in all, and the first and longest pause.
PATIENCE_S = 900.0
FIRST_PAUSE_S = 15.0
LONGEST_PAUSE_S = 120.0


class HerModelIsAwayError(RuntimeError):
    """Her model gave no answer at all: not there, refused, or empty. Not the same as an answer that does not fit.

    LIVE 2026-10-05 the event loop stalled, the model's endpoint was marked
    stale, and every ask came back empty in milliseconds; a build read thirteen
    empty answers as thirteen features nothing could be written for, in two
    seconds. An absence is waited out; a bad answer is tried again.
    """


def patiently(ask: Any, tell: Any = None, *, patience_s: float | None = None) -> Any:
    """``ask``, waiting her model's absences out (pauses doubling up to a bound) before trying again; still away past ``patience_s``, it says so."""
    import asyncio

    async def asking(prompt: str, schema: type[BaseModel], max_tokens: int) -> Any:
        waited, pause, limit = 0.0, FIRST_PAUSE_S, PATIENCE_S if patience_s is None else patience_s
        while True:
            try:
                return await ask(prompt, schema, max_tokens)
            except HerModelIsAwayError as why:
                if waited >= limit:
                    raise
                if waited == 0.0:
                    logger.info("her model is away (%s); waiting for it", why)
                    if tell is not None:
                        said = tell(f"My model is not answering ({why}); waiting for it to come back. What I have built so far is kept.")
                        if asyncio.iscoroutine(said):
                            await said
                await asyncio.sleep(pause)
                waited += pause
                pause = min(pause * 2.0, LONGEST_PAUSE_S)

    return asking


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


def _the_code_in(text: str) -> str:
    """The code an answer gives as code: its fenced block, or the whole answer when it reads as code."""
    text = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.S).strip()
    fenced = re.findall(r"```[\w+-]*\n(.*?)```", text, flags=re.S)
    if fenced:
        return max(fenced, key=len).strip()
    looks_like_code = sum(text.count(c) for c in "{};()=") > len(text) / 40
    return text if looks_like_code else ""


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
        # No model in this runtime at all is not one that is away: nothing will come back to wait for.
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
            # Read by code, not a reply to anybody: its shape's parser judges it.
            internal_inference=True,
            max_tokens=int(max_tokens),
            temperature=0.2,
        )
    except (RuntimeError, TimeoutError, OSError) as why:
        raise HerModelIsAwayError(f"it could not be asked: {str(why)[:200]}") from why
    except (ValueError, TypeError) as why:
        logger.info("her model could not be asked this: %s", why)
        return None
    text = str(reply.get("text") or "") if isinstance(reply, dict) else str(reply or "")
    if not re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip():
        raise HerModelIsAwayError("an empty answer")
    found = the_object_in(text) or what_was_finished(text)
    data: Any = None
    if found is not None:
        try:
            data = json.loads(found)
        except ValueError:
            data = None  # braces in code are not an object
    if not isinstance(data, dict):
        # Asked for code, a model may answer with the code itself.
        written = _the_code_in(text) if "code" in schema.model_fields else ""
        if written:
            return schema.model_validate({"code": written})
        logger.info("her model answered without an object (%d chars): %r", len(text), text[:300])
        return None
    try:
        return schema.model_validate(data)
    except ValidationError as why:
        logger.info("her model's answer did not fit %s: %s", schema.__name__, str(why)[:300])
        return None
