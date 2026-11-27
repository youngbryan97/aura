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
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = ["Judged", "WhereItLies", "judge", "raised_here", "the_function", "version_of"]

logger = logging.getLogger("SelfModification.WhereAFaultLies")

#: How long her model is given to read one fault, in seconds: as long as it is given to propose edits to a function
#: (edits_her_model_proposes).
ASK_S = 120.0
#: How much of each argument and of what her sources said is shown, in characters: an argument is shown as the test that
#: makes the call again writes it, and the fault's own test caps nothing, so this is only what fits beside a function.
SHOWN = 1200

THIS_FUNCTION = "the function"


class WhereItLies(BaseModel):
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

    @property
    def leaves_the_function(self) -> bool:
        """Whether her loop leaves the function as it is: raised on purpose, and what should change is elsewhere."""
        return self.on_purpose and self.lies_in != THIS_FUNCTION

    def as_kept(self) -> dict[str, Any]:
        return {"on_purpose": self.on_purpose, "lies_in": self.lies_in, "why": self.why,
                "function_digest": self.function_digest}

    @classmethod
    def from_kept(cls, kept: Any) -> Judged | None:
        if not isinstance(kept, dict) or not kept.get("function_digest"):
            return None
        return cls(bool(kept.get("on_purpose")), str(kept.get("lies_in") or ""), str(kept.get("why") or ""),
                   str(kept["function_digest"]))


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


def her_model() -> Any:
    """Her model, asked for a typed reading through her router; None where there is none to ask: no router to reach it
    by (a test, or a process still starting) is no reading, as her model away is."""
    from core.container import ServiceContainer

    if ServiceContainer.get("llm_router", default=None) is None:
        return None
    try:
        from core.brain.llm.structured_llm import StructuredLLM

        return StructuredLLM(WhereItLies, max_retries=1)
    except (ImportError, RuntimeError) as why:
        logger.info("no model to ask where a fault lies: %s", why)
        return None


async def judge(call: Any, source: str, *, consulted: str = "", advisor: Any = None) -> Judged | None:
    """Her model's reading of where the fault ``call`` raised lies; None where her Python found the function did not
    raise it itself, or her model gave no reading."""
    if not raised_here(source, call.line, call.error_type):
        return None
    found = the_function(source, call.qualname)
    if found is None:
        return None
    first, text = found
    digest = version_of(source, call.qualname)
    advisor = advisor if advisor is not None else her_model()
    if advisor is None:
        return None
    data = {
        "the function": _numbered(first, text),
        "it raised": f"{call.error_type.rsplit('.', 1)[-1]}: {call.error_message}",
        "at line": call.line,
        "the call that failed": the_call(call),
        "where it went below before it raised": list(call.below),
        "what it was raised from": list(call.causes),
        "what her sources say of it": _shown(consulted),
    }
    if call.receiver_class:
        data["the instance it was called on"] = {name: _shown(value) for name, value in call.receiver_state}
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
    if read is None:
        return None
    if not isinstance(read, WhereItLies):
        read = WhereItLies.model_validate(read)
    judged = Judged(read.on_purpose, read.lies_in, " ".join(read.why.split()), digest)
    logger.info("her model read %s: %s, and what should change is in %s", call.qualname,
                "raised on purpose" if judged.on_purpose else "not raised on purpose", judged.lies_in)
    return judged
