"""Programs that are code rather than windows: a library, a command-line tool, checked by calling them.

Not every program is used through a window. A word counter, a date parser, a
CSV cleaner or a tool run in a terminal is used by calling it: give it this,
it gives back that. Its checks are written the same way, before any code,
from what it is described to do: a call (a function's name and its
arguments) and what comes back (a value it equals, text it contains, or an
error it raises). They are done by calling the code in a kernel sandbox
(core/sandbox/untrusted_python.py) with no network and no access to this
machine's files, because the code is written by a model and nothing it writes
is trusted.

The program is written part by part as an application is
(writing_it_part_by_part.py): each feature's functions, kept only when that
feature's calls come back right and every call that came back right before
still does. A command-line entry point is written last, from the functions
that were kept.
"""
from __future__ import annotations

import ast
import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from core.rebuilding.what_a_program_does import Asker, Feature, Genome

logger = logging.getLogger("Rebuilding.Code")

__all__ = ["CodeCall", "CodeRun", "ProgramInCode", "calls_for", "run_calls", "write_code"]

#: Tries at one feature before it is left out.
TRIES = 3

#: The longest one call may take in the sandbox, in seconds.
CALL_S = 15.0


class CodeCall(BaseModel):
    feature: str = Field(max_length=120)
    function: str = Field(max_length=80, description="the function called")
    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    expect: Literal["equals", "contains", "raises"] = "equals"
    value: Any = Field(default=None, description="what it returns (equals), text the result contains (contains), or the error's name (raises)")

    def said(self) -> str:
        shown = ", ".join([json.dumps(a)[:80] for a in self.args] + [f"{k}={json.dumps(v)[:60]}" for k, v in self.kwargs.items()])
        return f"{self.function}({shown}) {self.expect} {json.dumps(self.value)[:160]}"


class _Calls(BaseModel):
    calls: list[CodeCall] = Field(default_factory=list)


@dataclass
class CodeRun:
    call: CodeCall
    held: bool
    why: str = ""


@dataclass
class _CodePart:
    name: str
    code: str
    serves: list[str] = field(default_factory=list)


@dataclass
class ProgramInCode:
    """A program as a Python module: its parts in order."""

    title: str
    parts: list[_CodePart] = field(default_factory=list)
    accent: str = ""
    style: str = ""

    def source(self, *, leaving_out: str = "") -> str:
        body = "\n\n\n".join(p.code.strip() for p in self.parts if p.name != leaving_out)
        return f'"""{self.title}: written part by part, each part kept for what calling it showed."""\n\n{body}\n'

    def with_part(self, name: str, code: str) -> ProgramInCode:
        parts = [_CodePart(name, code, [name]) if p.name == name else p for p in self.parts]
        if not any(p.name == name for p in self.parts):
            parts.append(_CodePart(name, code, [name]))
        return ProgramInCode(self.title, parts)

    def write(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.source(), "utf-8")
        return out

    def keep(self, folder: str | Path) -> Path:
        out = Path(folder) / "program.json"
        out.write_text(json.dumps({"kind": "code", **asdict(self)}, indent=1), "utf-8")
        return out


def _comes_back_right(call: CodeCall, outcome: Any) -> tuple[bool, str]:
    """Whether what came back from one call is what the check says."""
    if outcome.status != "ok" or not outcome.results:
        if call.expect == "raises":
            raised = f"{outcome.error} {outcome.stderr}"
            return str(call.value or "") in raised, f"it raised: {raised[-200:]}"
        return False, f"it did not run: {(outcome.error or outcome.stderr)[-300:]}"
    got = outcome.results[0]
    if call.expect == "equals":
        return got == call.value, f"it gave back {json.dumps(got)[:200]}"
    if call.expect == "contains":
        return str(call.value) in (got if isinstance(got, str) else json.dumps(got)), f"it gave back {json.dumps(got)[:200]}"
    return False, f"it gave back {json.dumps(got)[:200]} where an error was expected"


async def run_calls(source: str, calls: list[CodeCall]) -> list[CodeRun]:
    """Each call made on the module in the sandbox, one child per call so one cannot spoil another."""
    from core.sandbox.untrusted_python import call_untrusted_function

    async def one(call: CodeCall) -> CodeRun:
        outcome = await asyncio.to_thread(
            call_untrusted_function, source, call.function, [{"args": call.args, "kwargs": call.kwargs}],
            timeout_s=CALL_S, source="rebuilding",
        )
        held, why = _comes_back_right(call, outcome)
        return CodeRun(call, held, "" if held else why)

    gate = asyncio.Semaphore(4)

    async def bounded(call: CodeCall) -> CodeRun:
        async with gate:
            return await one(call)

    return list(await asyncio.gather(*(bounded(c) for c in calls)))


_HOW_CALLS_ARE_WRITTEN = """\
A check is one call of a Python function and what comes back:
  function  the function's name; args / kwargs  its arguments (JSON values)
  expect    "equals" (it returns exactly value), "contains" (its result, as text, contains value),
            or "raises" (it raises an error whose name or message contains value)
Name the functions as a person using the library would expect them named. Each check calls the program fresh."""


async def calls_for(genome: Genome, features: list[Feature], ask: Asker) -> list[CodeCall]:
    """The calls a person would make of these features, written from their description alone."""
    listed = "\n".join(f"- {f.name}: {f.how} -> {f.shows}" for f in features)
    got = await ask(
        f"{genome.name}: {genome.what_it_is}\nWrite two or three checks for each of these features, naming the feature exactly:\n"
        f"{listed}\n\n{_HOW_CALLS_ARE_WRITTEN}",
        _Calls, 2048,
    )
    if not isinstance(got, _Calls):
        return []
    named = {f.name.lower(): f.name for f in features}
    return [c.model_copy(update={"feature": named[c.feature.lower()]}) for c in got.calls if c.feature.lower() in named]


class WrittenCode(BaseModel):
    code: str = Field(default="", description="Python: the functions this feature needs, with imports from the standard library only")


def _does_not_parse(code: str) -> str:
    try:
        ast.parse(code)
    except SyntaxError as why:
        return f"line {why.lineno}: {why.msg}"
    return ""


def _a_feature_prompt(genome: Genome, program: ProgramInCode, feature: Feature, calls: list[CodeCall], wrong: str, before: str) -> str:
    text = (
        f"{genome.name}: {genome.what_it_is}\n"
        f"Write the Python for the feature \"{feature.name}\": {feature.how} -> {feature.shows}.\n"
        "These calls must come back as shown:\n" + "\n".join(f"- {c.said()}" for c in calls) + "\n"
        "Standard library only; no input(), no network, no files outside what the arguments name. "
        "Give the functions docstrings. Do not repeat the code already written; use it.\n\n"
        f"The code already written:\n```python\n{program.source()[:8000]}\n```\n"
    )
    if wrong:
        text += f"\nYour last version:\n```python\n{before[:5000]}\n```\nCalling it showed: {wrong}\nWrite it again so the calls come back right."
    return text


@dataclass
class CodeOutcome:
    feature: Feature
    held: int = 0
    of: int = 0
    tries: int = 0
    kept: bool = False
    why_not: str = ""


async def write_code(genome: Genome, calls: list[CodeCall], ask: Asker, out: Path) -> tuple[ProgramInCode, list[CodeOutcome], list[CodeCall]]:
    """The program as code, feature by feature, each kept for what calling it showed."""
    program = ProgramInCode(genome.name)
    holding: list[CodeCall] = []
    outcomes: list[CodeOutcome] = []
    for feature in genome.features:
        own = [c for c in calls if c.feature == feature.name]
        outcome = CodeOutcome(feature, of=len(own))
        outcomes.append(outcome)
        if not own:
            outcome.why_not = "no check could be written for it"
            continue
        wrong, before, best = "", "", None
        for outcome.tries in range(1, TRIES + 1):
            written = await ask(_a_feature_prompt(genome, program, feature, own, wrong, before), WrittenCode, 2048)
            code = written.code.strip() if isinstance(written, WrittenCode) else ""
            if code.startswith("```"):
                code = code.split("\n", 1)[-1].rsplit("```", 1)[0]
            if not code:
                wrong = "nothing was written"
                continue
            before = code
            unparsed = _does_not_parse(code)
            if unparsed:
                wrong = f"it does not parse: {unparsed}"
                continue
            candidate = program.with_part(feature.name, code)
            runs = await run_calls(candidate.source(), [*own, *holding])
            mine, others = runs[: len(own)], runs[len(own) :]
            held = [r for r in mine if r.held]
            broke = [r for r in others if not r.held]
            if held and not broke and (best is None or len(held) > len(best[1])):
                best = (candidate, held)
            if len(held) == len(own) and not broke:
                break
            wrong = "; ".join([f"{r.call.said()}: {r.why}" for r in mine if not r.held][:4] + [f"it broke {r.call.said()}: {r.why}" for r in broke][:3])
        if best is None:
            outcome.why_not = wrong
            continue
        program, held = best
        holding.extend(r.call for r in held)
        outcome.kept, outcome.held = True, len(held)
    await asyncio.to_thread(program.write, out)
    return program, outcomes, holding
