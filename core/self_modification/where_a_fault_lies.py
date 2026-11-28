"""Where a fault she heard lies, when the function raised it itself: her model reads the function and the call, and is
asked before her loop changes either.

A function that refuses what it was given (``raise ValueError("argv entries must not be empty")``), or reports what it
found (``record_degradation(RuntimeError("subsystem ... is unhealthy"))``), does what it was written to do when it
raises. Where it raised for a call like the one that failed, the fault is in what called it or in what it depends on,
and a change to the function only stops it saying so. Repair by selection cannot tell the two apart: a change that stops
the raise passes the fault's test either way. LIVE 26 and 27 November 2026 her own Python offered two such changes, each
held back only by a gate that failed for reasons of its own:

- ``and not coerced`` added after a check that already holds, so ``_coerce_argv`` could never refuse an empty word. The
  call that failed had passed an empty subtitle to a notification; the refusal was right.
- the report of an unhealthy subsystem dropped from ``DistributedResilienceCore._request_recovery``. The subsystem was
  unhealthy: the host was refusing a worker for want of memory.

Her Python finds whether the function raised it itself (the statement at the fault's line builds the exception it
raised, by name). Only then is her model asked, once for each version of the function: the function, the call that
failed with its arguments, where it went below, and what her sources said, as data. What it answers is kept on her goal
with its reasons; where it reads the raise as done on purpose and the fault as lying elsewhere, her loop leaves the
function as it is. Her model away, nothing is judged and her loop goes on as it did.
"""
from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = ["HerCortexReads", "Judged", "NotReadNow", "reader_of", "ToLookUp", "WhereItLies", "judge", "raised_here", "the_function", "version_of", "what_to_look_up"]

logger = logging.getLogger("SelfModification.WhereAFaultLies")

#: How long her model is given to read one fault, in seconds: as long as it is given to propose edits to a function
#: (edits_her_model_proposes).
ASK_S = 120.0
#: How much of each argument and of what her sources said is shown, in characters: an argument is shown as the test that
#: makes the call again writes it, and the fault's own test caps nothing, so this is only what fits beside a function.
SHOWN = 1200

THIS_FUNCTION = "the function"


class NotReadNow(RuntimeError):
    """Her model put the reading off (her background policy: the quiet window after a start, someone talking with her):
    not now, which is not the same as no reading. LIVE 27 November 2026, put off in the quiet window after her start,
    each fault went on unread, was made again, and stalled for good."""


class WhereItLies(BaseModel):
    """Where what should change lies, as her model reads the function and the call that failed."""

    on_purpose: bool = Field(description="whether the function raised or reported this as it was written to, for a call "
                                         "like the one that failed")
    lies_in: Literal["the function", "what called it", "what it depends on"] = Field(
        description="where what should change is")
    why: str = Field(description="why, from the function and the call", max_length=900)


@dataclass(frozen=True)
class Judged:
    """Her model's reading of one fault, and the version of the function it read."""

    on_purpose: bool
    lies_in: str
    why: str
    function_digest: str
    #: Which of her models read it: "cortex", or "brainstem" for her router's background tier. Readings kept before
    #: this was kept were her brainstem's.
    reader: str = "brainstem"

    @property
    def leaves_the_function(self) -> bool:
        """Whether her loop leaves the function as it is: raised on purpose, and what should change is elsewhere."""
        return self.on_purpose and self.lies_in != THIS_FUNCTION

    def as_kept(self) -> dict[str, Any]:
        return {"on_purpose": self.on_purpose, "lies_in": self.lies_in, "why": self.why,
                "function_digest": self.function_digest, "reader": self.reader}

    @classmethod
    def from_kept(cls, kept: Any) -> Judged | None:
        if not isinstance(kept, dict) or not kept.get("function_digest"):
            return None
        return cls(bool(kept.get("on_purpose")), str(kept.get("lies_in") or ""), str(kept.get("why") or ""),
                   str(kept["function_digest"]), str(kept.get("reader") or "brainstem"))

    def stands_for(self, reader: str) -> bool:
        """Whether this reading stands where ``reader`` would read now: her cortex's always, her brainstem's only
        where her cortex is not there to read it again."""
        return self.reader == "cortex" or reader != "cortex"


def _name_called(node: ast.AST) -> str:
    """The name a call or a raise names: ``ValueError`` for ``ValueError(...)`` and ``errors.ValueError(...)``."""
    target = node.func if isinstance(node, ast.Call) else node
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return ""


def _the_statement_at(tree: ast.AST, line: int) -> ast.stmt | None:
    """The innermost statement holding ``line`` that is not a block (an if, a loop, a def): what ran there."""
    found = None
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt) and not hasattr(node, "body") \
                and node.lineno <= line <= (node.end_lineno or node.lineno):
            if found is None or node.lineno >= found.lineno:
                found = node
    return found


def raised_here(source: str, line: int, error_type: str) -> bool:
    """Whether the statement at ``line`` builds the exception the fault raised: a raise of it, or a call that hands one
    it builds to something that records it. A bare ``raise`` hands on what was raised below, and an operation that
    failed (``items[key]``) built nothing; neither is the function's own doing."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    statement = _the_statement_at(tree, int(line or 0))
    name = str(error_type or "").rsplit(".", 1)[-1]
    if statement is None or not name:
        return False
    if isinstance(statement, ast.Raise):
        return statement.exc is not None and _name_called(statement.exc) == name
    return any(isinstance(node, ast.Call) and _name_called(node) == name for node in ast.walk(statement))


def the_function(source: str, qualname: str) -> tuple[int, str] | None:
    """The function ``qualname`` names in ``source`` (``Class.method``), as its first line and its text; None if absent."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    wanted = str(qualname or "").split(".")
    lines = source.splitlines()

    def search(body: list[ast.stmt], path: list[str]) -> tuple[int, str] | None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                here = [*path, node.name]
                if here == wanted and not isinstance(node, ast.ClassDef):
                    first = min([node.lineno, *(item.lineno for item in node.decorator_list)])
                    return first, "\n".join(lines[first - 1:node.end_lineno or node.lineno])
                if wanted[:len(here)] == here:
                    found = search(node.body, here)
                    if found is not None:
                        return found
        return None

    return search(tree.body, [])


def version_of(source: str, qualname: str) -> str:
    """A digest of the function ``qualname`` as ``source`` has it, "" when it is not there: a reading of the function
    holds for as long as this does."""
    found = the_function(source, qualname)
    return hashlib.sha256(found[1].encode("utf-8")).hexdigest()[:16] if found is not None else ""


def _numbered(first: int, text: str) -> str:
    return "\n".join(f"{n:04d}: {line}" for n, line in enumerate(text.splitlines(), start=first))


def _shown(text: Any) -> str:
    text = str(text or "")
    return text if len(text) <= SHOWN else text[:SHOWN] + " ..."


def the_call(call: Any) -> dict[str, str]:
    """The call that failed, each argument as the test that makes it again writes it."""
    shown = dict(zip(call.positional_names, call.positional, strict=False))
    shown.update(dict(call.keywords))
    if call.star_args:
        shown["*args"] = call.star_args
    if call.star_kwargs:
        shown["**kwargs"] = call.star_kwargs
    return {name: _shown(value) for name, value in shown.items()}


class ToLookUp(BaseModel):
    """What is worth looking up before the function is changed."""

    searches: list[str] = Field(default_factory=list,
                                description="what is worth looking up before changing it, each as the words to search by")


#: Where her readings of faults come from, by name in her log and to her generation lane.
ORIGIN = "her_reading_of_a_fault"


class HerCortexReads:
    """A typed reading asked of her cortex as background work, under her generation lane's lease, as her goal passes and
    her unprompted turns reach it (core/brain/her_unprompted_turn.on_her_lane): it gives way to a person and is not
    begun while someone waits for her model. Asked of her router as background work, a reading went to her brainstem,
    a 27B at two bits: LIVE 27 November 2026 it read her watchdog's report of a 59 s lag as a fault of the watchdog's to
    change. Shaped as StructuredLLM is, ``generate`` and ``last_defer_reason``."""

    def __init__(self, schema: type[BaseModel]) -> None:
        self.schema = schema
        self.last_defer_reason = ""

    async def generate(self, prompt: str, *, is_background: bool = True, deadline_s: float = ASK_S) -> BaseModel | None:
        from core.brain.her_unprompted_turn import on_her_lane
        from core.brain.llm.token_budget_evidence import chars_per_token
        from core.runtime.structured_input import A_CLOSED_QUESTIONS_FLOOR

        from core.brain.llm.mlx_client import MLXLocalClient

        self.last_defer_reason = ""
        messages = [{"role": "user", "content": prompt}]
        # The reading is a call of hers whose arguments are the answer, in her template's own tool-call form, as her
        # model writes every call of her goal passes. Held to a JSON object instead, her cortex answered in prose
        # (LIVE 27 November 2026, 16:33, "no_json_value_found" twice), the decoder's shape waiting on a channel close.
        name = re.sub(r"(?<!^)(?=[A-Z])", "_", self.schema.__name__).lower()
        properties = {key: {k: v for k, v in spec.items() if k != "title"}
                      for key, spec in self.schema.model_json_schema().get("properties", {}).items()}
        definitions = {name: {"description": " ".join(str(self.schema.__doc__ or name).split()),
                              "parameters": {"type": "object", "properties": properties,
                                             "required": list(self.schema.model_json_schema().get("required", []))}}}
        offered = MLXLocalClient._normalize_tool_definitions_for_template(definitions)
        # One closed question's answer, and the longest text a field of the schema may hold, in her measured tokens.
        longest = max((int(field.metadata[0].max_length) for field in self.schema.model_fields.values()
                       if field.metadata and getattr(field.metadata[0], "max_length", None)), default=0)
        ratio = max(chars_per_token().ratio, 1.0)
        answer_tokens = A_CLOSED_QUESTIONS_FLOOR + int(longest / ratio)

        async def read(client: Any, seconds: float) -> str:
            return str(await client.generate_text_async(
                "", messages=messages, system_prompt="", origin=ORIGIN, is_background=is_background, tools=offered,
                tool_budget_definitions=offered, max_tokens=answer_tokens, timeout=seconds) or "")

        text = await on_her_lane(messages, origin=ORIGIN, least_s=max(deadline_s, self._time_it_needs(messages,
                                                                                                    answer_tokens)),
                                 work=read)
        if text is None:
            self.last_defer_reason = "her cortex is held for something else"
            return None
        if not text.strip():
            # Nothing came back: her generation was stopped for a turn that came first (LIVE 27 November 2026, 15:33,
            # soft-cancelled 986 tokens in), which is a reading put off and not one that found nothing.
            self.last_defer_reason = "her cortex's generation was stopped before it answered"
            return None
        call = MLXLocalClient._extract_tool_call_payload(text, allowed_tools={name}, tool_definitions=definitions)
        for found in ([call["args"]] if call else []) + _the_objects_in(text):
            try:
                return self.schema.model_validate(found)
            except ValueError:
                continue
        logger.info("her cortex's answer held no %s: %r", self.schema.__name__, " ".join(text.split())[-300:])
        return None


    @staticmethod
    def _time_it_needs(messages: list[dict[str, Any]], answer_tokens: int) -> float:
        """What reading the question, thinking and answering take on this machine: the prompt, the thinking her
        cortex's private channel has been measured to cost on top of the answer, and the answer. Given 120 s, a
        reading thought until its time ran out and came back as thinking with no answer (LIVE 27 November 2026, 15:4x,
        "the function deliberately raises RuntimeError(\"warmup_precompile_worker_dead\")")."""
        from core.brain.llm.mlx_client import time_a_prompt_needs
        from core.brain.llm.thinking_reserve import reserve_tokens
        from core.runtime.service_access import resolve_inference_gate

        client = getattr(resolve_inference_gate(default=None), "_mlx_client", None)
        thinking = reserve_tokens(str(getattr(client, "model_path", "") or ""))
        chars = sum(len(str(message.get("content") or "")) for message in messages)
        return time_a_prompt_needs(chars, answer_tokens + thinking)


def _the_objects_in(text: str) -> list[Any]:
    """Each whole JSON object in what her cortex wrote, past what it thought before it, in order."""
    said = str(text or "")
    if "</think>" in said:
        said = said.rsplit("</think>", 1)[1]
    found, at, decoder = [], said.find("{"), json.JSONDecoder()
    while at >= 0:
        try:
            value, end = decoder.raw_decode(said[at:])
        except ValueError:
            at = said.find("{", at + 1)
            continue
        if isinstance(value, dict):
            found.append(value)
        at = said.find("{", at + end)
    return found


def reader_of(advisor: Any) -> str:
    """Which of her models ``advisor`` asks."""
    return "cortex" if isinstance(advisor, HerCortexReads) else "brainstem"


def her_model(schema: type[BaseModel] = WhereItLies) -> Any:
    """Her model, asked for a typed reading: her cortex where it is loaded (HerCortexReads), else her router's
    background tier; None where there is neither (a test, or a process still starting), which is no reading, as her
    model away is."""
    from core.container import ServiceContainer
    from core.runtime.service_access import resolve_inference_gate

    if hasattr(getattr(resolve_inference_gate(default=None), "_mlx_client", None), "generate_text_async"):
        return HerCortexReads(schema)
    if ServiceContainer.get("llm_router", default=None) is None:
        return None
    try:
        from core.brain.llm.structured_llm import StructuredLLM

        # Her mending loop runs with nobody yet spoken to her after a restart (adeff4bde), and so does its reading of a
        # fault: LIVE 27 November 2026 every reading after her restart was put off as "no_user_anchor".
        return StructuredLLM(schema, max_retries=1, before_anyone_has_spoken=True)
    except (ImportError, RuntimeError) as why:
        logger.info("no model to ask where a fault lies: %s", why)
        return None


def _put_off(advisor: Any, read: Any) -> None:
    """Raise NotReadNow where her model gave nothing because it put the asking off."""
    reason = str(getattr(advisor, "last_defer_reason", "") or "")
    if read is None and reason:
        raise NotReadNow(reason)


def _what_was_called(call: Any, source: str) -> dict[str, Any] | None:
    """The function and the call that failed, as data her model reads; None where the function is not in ``source``."""
    found = the_function(source, call.qualname)
    if found is None:
        return None
    data: dict[str, Any] = {
        "the function": _numbered(*found),
        "it raised": f"{call.error_type.rsplit('.', 1)[-1]}: {call.error_message}",
        "at line": call.line,
        "the call that failed": the_call(call),
        "where it went below before it raised": list(call.below),
        "what it was raised from": list(call.causes),
    }
    if call.receiver_class:
        data["the instance it was called on"] = {name: _shown(value) for name, value in call.receiver_state}
    return data


async def what_to_look_up(call: Any, source: str, *, advisor: Any = None) -> list[str]:
    """What her model reads as worth looking up before the function is changed, as the words to search by; none where
    the function is not there or her model gave no answer. Searched by the error's words alone, her library answered an
    empty argument to a program with a passage on LL parsers (26 November 2026)."""
    data = _what_was_called(call, source)
    advisor = advisor if advisor is not None else her_model(ToLookUp) if data is not None else None
    if data is None or advisor is None:
        return []
    try:
        read = await advisor.generate("This function raised while she ran. What is worth looking up before changing "
                                      "it? Data: " + json.dumps(data, ensure_ascii=False),
                                      is_background=True, deadline_s=ASK_S)
        read = read if read is None or isinstance(read, ToLookUp) else ToLookUp.model_validate(read)
    except (RuntimeError, TimeoutError, ValueError, TypeError, OSError) as why:
        logger.info("her model could not be asked what to look up: %s", why)
        return []
    _put_off(advisor, read)
    searches = [" ".join(str(each).split()) for each in (read.searches if read is not None else [])]
    return list(dict.fromkeys(each for each in searches if each))


async def judge(call: Any, source: str, *, consulted: str = "", advisor: Any = None) -> Judged | None:
    """Her model's reading of where the fault ``call`` raised lies; None where her Python found the function did not
    raise it itself, or her model gave no reading."""
    if not raised_here(source, call.line, call.error_type):
        return None
    data = _what_was_called(call, source)
    if data is None:
        return None
    digest = version_of(source, call.qualname)
    advisor = advisor if advisor is not None else her_model()
    if advisor is None:
        return None
    data["what her sources say of it"] = _shown(consulted)
    try:
        read = await advisor.generate(
            "This function raised while she ran. Did it raise this on purpose, as it was written to for a call like "
            "this one, and where is what should change: in the function, in what called it, or in what it depends on? "
            "Data: " + json.dumps(data, ensure_ascii=False),
            is_background=True, deadline_s=ASK_S,
        )
    except (RuntimeError, TimeoutError, ValueError, TypeError, OSError) as why:
        logger.info("her model could not be asked where a fault lies: %s", why)
        return None
    _put_off(advisor, read)
    if read is None:
        logger.info("her model gave no reading of %s", call.qualname)
        return None
    if not isinstance(read, WhereItLies):
        read = WhereItLies.model_validate(read)
    judged = Judged(read.on_purpose, read.lies_in, " ".join(read.why.split()), digest, reader_of(advisor))
    logger.info("her model read %s: %s, and what should change is in %s", call.qualname,
                "raised on purpose" if judged.on_purpose else "not raised on purpose", judged.lies_in)
    return judged
