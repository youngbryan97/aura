"""What a forged skill must be, and what has to happen before it counts as one.

The defect this closes
----------------------
:mod:`core.skill_management.hephaestus` writes model-authored Python into the
live ``skills/`` directory and registers it as a capability. Between generation
and registration it checks two things: that the source parses, and that an AST
guard finds no banned import. **It never runs the code.** The first execution of
a forged skill is the first time anybody finds out whether it works, and by then
it is already advertised in the catalog as something Aura can do.

Voyager — the Minecraft agent whose skill library is the reference design for
this — does not have that hole. Its loop is: propose a task, write an executable
program, *run it in the environment*, take the interpreter's errors as feedback,
retry, and add the program to the library **only after a critic confirms the
attempt actually succeeded**. The library is a record of things that worked. The
retention gate is the whole design.

Three things made the same gate impossible here until now:

* The generated artifact subclassed ``BaseSkill`` and opened with
  ``from core.skills.base_skill import BaseSkill``. The sandbox denies reads of
  user data, so the repository is not on the child's disk and that import cannot
  resolve. The artifact could not be executed under the boundary at all.
* ``execute`` was ``async``. Coroutines do not cross the JSON boundary the
  sandbox marshals results over.
* ``Sandbox2.execute`` was handed the *class* name as the callable, so the child
  constructed the class and never reached the method.

So the contract changed rather than the checks. A forged skill is now a plain
module with one synchronous entrypoint that takes a JSON object and returns a
JSON object. That form is executable under the boundary, which is what makes
verification possible; :class:`~core.skill_management.skill_adapter` wraps it
back into a ``BaseSkill`` in-process so nothing downstream notices.

Precommitted probes, not self-grading
-------------------------------------
Voyager's critic is asked, after the fact, whether the attempt succeeded. That
exact shape has already cost this codebase once: a reasoner took the caller's
word as ground truth and wrote it into a durable corpus. A model grading its own
output after seeing it is not evidence.

The distinction this module rests on is *when* the claim was made. A probe is
declared **before** the code runs: these inputs, that expected shape, that
expected value. Then the code runs and reality either matches or does not. A
precommitted expectation the code then violates is a failing test. A post-hoc
"yes, that looks right" is a second opinion from the same source that produced
the error. Only the first kind is admitted here, and
:attr:`VerificationReport.evidence` records what was executed so the claim
"verified" can be checked later rather than believed.

What passing means, precisely
-----------------------------
Passing is not a claim that the skill is correct. It is exactly this list:

* the module executed under a real kernel boundary,
* the declared entrypoint existed and was callable,
* every probe ran to completion without raising,
* every probe returned a JSON object carrying the declared result contract,
* every probe that precommitted an expected value matched it,
* and if the draft declared itself deterministic, repeated calls agreed.

That is a floor, not a proof, and :attr:`VerificationReport.summary` says so in
those terms. It is also six things more than the previous gate checked.
"""

from __future__ import annotations

import ast
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.sandbox.untrusted_python import (
    DEFAULT_MEM_BYTES,
    DEFAULT_TIMEOUT_S,
    SandboxOutcome,
    call_untrusted_function,
)

__all__ = [
    "ENTRYPOINT",
    "SKILL_MODULE_TEMPLATE",
    "Probe",
    "ProbeOutcome",
    "SkillDraft",
    "VerificationReport",
    "ContractError",
    "screen_source",
    "verify_draft",
    "render_skill_module",
]

#: The one function a forged skill must expose. Fixed rather than declared by
#: the draft: a name the model chooses is a name the loader has to trust, and
#: the loader would then be calling whatever the model pointed it at.
ENTRYPOINT = "run"

#: Modules a forged skill may import. Deliberately smaller than the sandbox's
#: own allowance, because this is the set a *pure computation* needs — anything
#: reaching outside the process is the job of a real skill with a real gateway,
#: not of generated code. The kernel boundary is what enforces it; this list is
#: the earlier, clearer refusal.
ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "base64",
        "binascii",
        "bisect",
        "calendar",
        "cmath",
        "collections",
        "colorsys",
        "copy",
        "csv",
        "dataclasses",
        "datetime",
        "decimal",
        "difflib",
        "enum",
        "fnmatch",
        "fractions",
        "functools",
        "hashlib",
        "heapq",
        "hmac",
        "html",
        "io",
        "ipaddress",
        "itertools",
        "json",
        "math",
        "numbers",
        "operator",
        "pprint",
        "random",
        "re",
        "statistics",
        "string",
        "textwrap",
        "time",
        "types",
        "typing",
        "unicodedata",
        "urllib",
        "uuid",
    }
)

#: Attribute names that reach the interpreter's own machinery. An allowlist of
#: imports does not stop ``().__class__.__mro__[1].__subclasses__()``, which is
#: the standard way out of a restricted namespace and needs no import at all.
_ESCAPE_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "__subclasses__",
        "__bases__",
        "__mro__",
        "__globals__",
        "__code__",
        "__closure__",
        "__builtins__",
        "__loader__",
        "__spec__",
        "__import__",
        "__getattribute__",
        "__reduce__",
        "__reduce_ex__",
        "gi_frame",
        "cr_frame",
        "f_globals",
        "f_builtins",
        "f_back",
    }
)

_BANNED_CALLS: frozenset[str] = frozenset(
    {"eval", "exec", "compile", "open", "input", "breakpoint", "globals", "vars"}
)

SKILL_MODULE_TEMPLATE = '''"""{description}

Forged by Aura for: {objective}
Gap: {gap}
Contract: run(params: dict) -> dict, pure computation, no host access.
"""
{imports}

def {entrypoint}(params):
{body}
'''


class ContractError(ValueError):
    """The candidate cannot be a forged skill, whatever else is true of it."""


@dataclass(frozen=True)
class Probe:
    """One call the candidate must survive, declared before it runs.

    ``expect`` is optional and, when given, is compared to the returned object
    with plain equality after JSON normalisation. It is the precommitment that
    makes this a test: the drafter says what the answer should be, then the
    interpreter says what it is.

    ``expect_keys`` is the weaker form for skills whose exact output nobody can
    state up front — a timestamp, a random draw. It asserts structure without
    asserting values, which is worth strictly more than asserting nothing.
    """

    params: dict[str, Any] = field(default_factory=dict)
    expect: Any = None
    expect_keys: tuple[str, ...] = ()
    has_expectation: bool = False
    label: str = ""

    @staticmethod
    def of(
        params: Mapping[str, Any] | None = None,
        *,
        expect_keys: Sequence[str] = (),
        label: str = "",
        **expectation: Any,
    ) -> "Probe":
        """Build a probe, optionally precommitting an exact result.

        The expectation is passed as ``expect=<value>`` and its *presence* is
        what marks the probe as precommitted. A plain default would make
        "expects ``None``" and "expects nothing" the same probe, and a skill
        that returns ``None`` would then pass a test nobody wrote.
        """
        unknown = set(expectation) - {"expect"}
        if unknown:
            raise TypeError(f"unexpected keyword argument(s): {', '.join(sorted(unknown))}")
        return Probe(
            params=dict(params or {}),
            expect=expectation.get("expect"),
            expect_keys=tuple(str(k) for k in expect_keys),
            has_expectation="expect" in expectation,
            label=str(label),
        )

    def describe(self) -> str:
        return self.label or f"run({json.dumps(self.params, sort_keys=True, default=repr)})"


@dataclass(frozen=True)
class ProbeOutcome:
    """What actually happened when a probe ran."""

    probe: Probe
    passed: bool
    reason: str
    returned: Any = None
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    duration_s: float = 0.0
    sandboxed: bool = False
    boundary: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe": self.probe.describe(),
            "passed": self.passed,
            "reason": self.reason,
            "returned": self.returned,
            "error": self.error,
            "duration_s": round(self.duration_s, 6),
            "sandboxed": self.sandboxed,
            "boundary": self.boundary,
        }


@dataclass(frozen=True)
class SkillDraft:
    """A candidate skill, before anything is known about whether it works."""

    name: str
    description: str
    source: str
    probes: tuple[Probe, ...] = ()
    objective: str = ""
    gap: str = ""
    deterministic: bool = True
    #: Keys the returned object must carry. ``ok`` is always required and is
    #: added here rather than left to the drafter, because the callers of a
    #: skill branch on it and a skill that omits it reads as a failure.
    result_contract: tuple[str, ...] = ("ok",)

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ContractError("a draft must be named")
        if not str(self.source).strip():
            raise ContractError(f"draft {self.name!r} has no source")
        if "ok" not in self.result_contract:
            object.__setattr__(self, "result_contract", ("ok", *self.result_contract))

    @property
    def digest(self) -> str:
        """Content address of the source. Two drafts with the same digest are
        the same code, whatever the surrounding metadata says."""
        return hashlib.blake2b(self.source.encode("utf-8"), digest_size=16).hexdigest()


@dataclass(frozen=True)
class VerificationReport:
    """The evidence, and the verdict that follows from it."""

    draft: SkillDraft
    passed: bool
    stage: str
    reason: str
    outcomes: tuple[ProbeOutcome, ...] = ()
    verified_at: float = field(default_factory=time.time)
    boundary: str = "none"

    @property
    def executed(self) -> int:
        return sum(1 for o in self.outcomes if o.reason != "not run")

    @property
    def precommitted(self) -> int:
        return sum(1 for o in self.outcomes if o.probe.has_expectation or o.probe.expect_keys)

    @property
    def summary(self) -> str:
        if not self.passed:
            return f"rejected at {self.stage}: {self.reason}"
        return (
            f"ran {self.executed} probe(s) under {self.boundary}, "
            f"{self.precommitted} with a precommitted expectation; "
            "this is evidence the skill executes as declared, not that it is correct"
        )

    def feedback(self) -> str:
        """What went wrong, in the interpreter's words rather than a paraphrase.

        Handed back to the drafter for the next attempt. The value is that it is
        a real traceback from a real execution: the retry is responding to
        evidence, not to a rephrased instruction.
        """
        if self.passed:
            return ""
        lines = [f"{self.stage}: {self.reason}"]
        for outcome in self.outcomes:
            if outcome.passed:
                continue
            detail = outcome.error or outcome.stderr or outcome.reason
            if detail:
                lines.append(f"{outcome.probe.describe()} -> {detail.strip()}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.draft.name,
            "digest": self.draft.digest,
            "passed": self.passed,
            "stage": self.stage,
            "reason": self.reason,
            "summary": self.summary,
            "boundary": self.boundary,
            "verified_at": self.verified_at,
            "probes_executed": self.executed,
            "probes_precommitted": self.precommitted,
            "outcomes": [o.to_dict() for o in self.outcomes],
        }


def render_skill_module(
    *,
    name: str,
    description: str,
    body: str,
    imports: Sequence[str] = (),
    objective: str = "",
    gap: str = "",
) -> str:
    """Assemble a contract-shaped module from a generated function body.

    The body is indented into ``run``; imports are hoisted to module scope so
    :func:`screen_source` sees them as imports rather than having to reason
    about a call to ``__import__`` inside a function.
    """
    import textwrap

    text = textwrap.dedent(str(body or "")).strip("\n")
    if not text.strip():
        raise ContractError(f"draft {name!r} has an empty body")
    import_lines = "\n".join(f"import {m}" for m in dict.fromkeys(imports) if m)
    return SKILL_MODULE_TEMPLATE.format(
        description=str(description or name).replace('"""', "'''"),
        objective=str(objective or "unstated").replace('"""', "'''"),
        gap=str(gap or "unstated").replace('"""', "'''"),
        imports=import_lines,
        entrypoint=ENTRYPOINT,
        body=textwrap.indent(text, "    "),
    )


def screen_source(source: str, *, entrypoint: str = ENTRYPOINT) -> None:
    """Reject source that cannot be a forged skill. Raises :class:`ContractError`.

    Runs before the sandbox, and is not a substitute for it. Two things are
    being separated: the boundary stops code from reaching the machine, while
    this stops code that would reach the boundary and fail there anyway, or that
    is shaped wrongly for the loader. Getting a clear reason at draft time is
    worth more to the retry loop than a subprocess death signal.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ContractError(f"source does not parse: {exc}") from exc

    entry = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == entrypoint:
            entry = node
        elif isinstance(node, ast.AsyncFunctionDef) and node.name == entrypoint:
            raise ContractError(
                f"{entrypoint}() is async; the sandbox marshals results over JSON "
                "and cannot await a coroutine"
            )
    if entry is None:
        raise ContractError(f"source defines no {entrypoint}() at module level")

    positional = [a.arg for a in entry.args.posonlyargs + entry.args.args]
    if len(positional) != 1:
        raise ContractError(
            f"{entrypoint}() takes {len(positional)} positional arguments; "
            "the contract is exactly one, the params object"
        )

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    violations.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level or root not in ALLOWED_IMPORTS:
                violations.append(f"from {node.module or '.'} import ...")
        elif isinstance(node, ast.Attribute):
            if node.attr in _ESCAPE_ATTRIBUTES:
                violations.append(f"attribute {node.attr}")
        elif isinstance(node, ast.Name):
            if node.id in _BANNED_CALLS and isinstance(node.ctx, ast.Load):
                violations.append(f"reference to {node.id}")
    if violations:
        raise ContractError(
            "source reaches outside the skill contract: " + ", ".join(sorted(set(violations)))
        )


def _normalise(value: Any) -> Any:
    """Round-trip through JSON so comparisons match what crossed the boundary.

    Without this a probe expecting ``{"n": 1}`` fails against a returned
    ``{"n": 1.0}`` on one host and passes on another, because the difference
    never survives the marshalling anyway.
    """
    try:
        return json.loads(json.dumps(value, sort_keys=True, default=repr))
    except (TypeError, ValueError):
        return repr(value)


def _reject(
    draft: SkillDraft, stage: str, reason: str, outcomes: Sequence[ProbeOutcome] = ()
) -> VerificationReport:
    return VerificationReport(
        draft=draft, passed=False, stage=stage, reason=reason, outcomes=tuple(outcomes)
    )


def verify_draft(
    draft: SkillDraft,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    mem_bytes: int = DEFAULT_MEM_BYTES,
    require_boundary: bool = True,
    source_label: str = "skill_forge",
) -> VerificationReport:
    """Run the draft and report whether it earned retention.

    ``require_boundary`` defaults to refusing rather than running unconfined.
    A verification that ran the code outside a sandbox has proved something
    about a different situation than the one the skill will live in, and would
    report "verified" for it.
    """
    if not draft.probes:
        # A skill with no probe cannot be verified, and verifying nothing is
        # exactly the state this module exists to end. The draft is rejected
        # rather than passed with a caveat, because a caveat in a report nobody
        # reads is how the previous gate came to be trusted.
        return _reject(
            draft,
            "contract",
            "draft declares no probes, so retention would rest on nothing executed",
        )

    try:
        screen_source(draft.source)
    except ContractError as exc:
        return _reject(draft, "screen", str(exc))

    calls = [{"args": [dict(p.params)], "kwargs": {}} for p in draft.probes]
    if draft.deterministic:
        # Repeat every probe once. A skill that answers differently to identical
        # input is not something a library can hand out later and expect the same
        # behaviour from, and the drafter said it would not do that.
        calls.extend({"args": [dict(p.params)], "kwargs": {}} for p in draft.probes)

    started = time.perf_counter()
    outcome: SandboxOutcome = call_untrusted_function(
        draft.source,
        ENTRYPOINT,
        calls,
        timeout_s=timeout_s,
        mem_bytes=mem_bytes,
        require_boundary=require_boundary,
        source=source_label,
    )
    elapsed = time.perf_counter() - started

    if outcome.status == "no_boundary":
        return _reject(
            draft,
            "boundary",
            f"refusing to verify without a kernel boundary: {outcome.error}",
        )
    if outcome.status != "ok":
        # One failure aborts the whole child, so which probe died is not
        # recoverable from the payload. The traceback is, and that is what the
        # retry needs.
        detail = (outcome.error or outcome.stderr or outcome.status).strip()
        return _reject(
            draft,
            "execution",
            f"the module did not run to completion ({outcome.status}): {detail}",
            [
                ProbeOutcome(
                    probe=p,
                    passed=False,
                    reason="not run" if i else "aborted the run",
                    error=detail if not i else "",
                    stdout=outcome.stdout,
                    stderr=outcome.stderr,
                    sandboxed=outcome.sandboxed,
                    boundary=outcome.boundary,
                )
                for i, p in enumerate(draft.probes)
            ],
        )

    n = len(draft.probes)
    results = list(outcome.results)
    if len(results) < n:
        return _reject(
            draft,
            "execution",
            f"expected {n} results and the child returned {len(results)}",
        )
    repeats = results[n : 2 * n] if draft.deterministic and len(results) >= 2 * n else []

    per_probe = elapsed / max(1, len(calls))
    outcomes: list[ProbeOutcome] = []
    for i, probe in enumerate(draft.probes):
        returned = _normalise(results[i])
        base = {
            "probe": probe,
            "returned": returned,
            "stdout": outcome.stdout,
            "stderr": outcome.stderr,
            "duration_s": per_probe,
            "sandboxed": outcome.sandboxed,
            "boundary": outcome.boundary,
        }

        if not isinstance(returned, dict):
            outcomes.append(
                ProbeOutcome(
                    **base,
                    passed=False,
                    reason=f"returned {type(results[i]).__name__}, and the contract is a JSON object",
                )
            )
            continue

        missing = [k for k in draft.result_contract if k not in returned]
        if missing:
            outcomes.append(
                ProbeOutcome(
                    **base,
                    passed=False,
                    reason=f"result is missing declared key(s): {', '.join(missing)}",
                )
            )
            continue

        if probe.expect_keys:
            absent = [k for k in probe.expect_keys if k not in returned]
            if absent:
                outcomes.append(
                    ProbeOutcome(
                        **base,
                        passed=False,
                        reason=f"precommitted key(s) absent: {', '.join(absent)}",
                    )
                )
                continue

        if probe.has_expectation and returned != _normalise(probe.expect):
            outcomes.append(
                ProbeOutcome(
                    **base,
                    passed=False,
                    reason=(
                        "precommitted expectation not met: expected "
                        f"{json.dumps(_normalise(probe.expect), sort_keys=True, default=repr)}"
                    ),
                )
            )
            continue

        if repeats and _normalise(repeats[i]) != returned:
            outcomes.append(
                ProbeOutcome(
                    **base,
                    passed=False,
                    reason="declared deterministic but two identical calls disagreed",
                )
            )
            continue

        outcomes.append(ProbeOutcome(**base, passed=True, reason="ran and met its contract"))

    failed = [o for o in outcomes if not o.passed]
    if failed:
        return VerificationReport(
            draft=draft,
            passed=False,
            stage="probe",
            reason=f"{len(failed)} of {n} probe(s) failed",
            outcomes=tuple(outcomes),
            boundary=outcome.boundary,
        )

    return VerificationReport(
        draft=draft,
        passed=True,
        stage="verified",
        reason=f"all {n} probe(s) ran and met their declared contract",
        outcomes=tuple(outcomes),
        boundary=outcome.boundary,
    )
