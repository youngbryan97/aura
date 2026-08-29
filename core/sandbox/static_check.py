"""Everything wrong with a piece of code that can be known without running it.

Three things are decidable by reading, and each one currently costs an
execution to discover:

* it does not parse;
* it uses a name that is not defined anywhere it can see;
* it calls into a named library in a way that library does not support.

All three are what invention looks like when a model writes code, and all
three come back from a run as a traceback — which says what broke, at the
price of a full attempt, and on this machine an attempt is a generation from
a resident 27B plus however long the code takes to fail.

Reading costs milliseconds and can say more: not "AttributeError: add_entry"
but "Ledger offers balance, post, reverse, trial_balance". The next attempt
starts from the real API rather than from a second guess.

Undefined names come from Ruff, which is the tool that does this properly and
is already installed here. When it is not on the machine that check is simply
absent — a missing checker reports nothing, never a clean bill.

Silence means nothing was decidable. Code can pass everything here and still
be wrong, because whether a ledger entry has its debit and credit the right
way round is not a fact about syntax.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from core.sandbox.api_check import ApiFinding, check_code_against_library

__all__ = ["StaticFinding", "what_will_not_work", "describe_findings"]

#: Undefined name, and "local variable referenced before assignment". Both are
#: a name the code expects to exist that does not. Ruff's other rules are
#: style, and style is not this check's business.
_NAME_RULES = ("F821", "F823")

#: A checker that hangs is worse than one that is absent.
_RUFF_SECONDS = 10.0


@dataclass(frozen=True)
class StaticFinding:
    """One thing that is wrong with the code as written."""

    line: int
    said: str
    problem: str

    def describe(self) -> str:
        where = f"line {self.line}: " if self.line else ""
        return f"{where}{self.said} — {self.problem}"


def _from_api(finding: ApiFinding) -> StaticFinding:
    return StaticFinding(line=finding.line, said=finding.said, problem=finding.problem)


def _does_not_parse(code: str) -> StaticFinding | None:
    try:
        compile(code, "<the code>", "exec")
    except SyntaxError as exc:
        return StaticFinding(
            line=int(exc.lineno or 0),
            said=(exc.text or "").strip() or "the code",
            problem=f"does not parse: {exc.msg}",
        )
    except ValueError as exc:  # embedded nulls and the like
        return StaticFinding(line=0, said="the code", problem=str(exc))
    return None


def _ruff() -> str | None:
    """Ruff as this interpreter can reach it, or nothing."""

    beside_python = Path(sys.executable).with_name("ruff")
    if beside_python.is_file():
        return str(beside_python)
    return shutil.which("ruff")


def _names_that_are_not_defined(code: str) -> list[StaticFinding]:
    ruff = _ruff()
    if not ruff:
        return []
    with tempfile.TemporaryDirectory(prefix="aura-static-check-") as directory:
        written = Path(directory) / "candidate.py"
        written.write_text(code, encoding="utf-8")
        try:
            finished = subprocess.run(  # noqa: S603 - fixed argv, no shell
                [
                    ruff,
                    "check",
                    "--no-cache",
                    "--isolated",
                    "--output-format",
                    "json",
                    "--select",
                    ",".join(_NAME_RULES),
                    str(written),
                ],
                capture_output=True,
                text=True,
                timeout=_RUFF_SECONDS,
            )
            reported = json.loads(finished.stdout or "[]")
        except (OSError, subprocess.SubprocessError, ValueError):
            return []  # a checker that did not run has found nothing
    findings: list[StaticFinding] = []
    for item in reported if isinstance(reported, list) else []:
        if not isinstance(item, dict):
            continue
        message = str(item.get("message") or "").strip()
        location = item.get("location") or {}
        findings.append(
            StaticFinding(
                line=int(location.get("row") or 0) if isinstance(location, dict) else 0,
                said=message,
                problem="this name is not defined anywhere the code can see it",
            )
        )
    return findings


def _names_one_of(finding: StaticFinding, provided: frozenset[str]) -> bool:
    """Whether this undefined-name finding is about a name the runtime supplies."""

    if not provided:
        return False
    said = finding.said
    start = said.find("`")
    end = said.find("`", start + 1)
    if start < 0 or end <= start:
        return False
    return said[start + 1 : end] in provided


def what_will_not_work(
    code: str, library_root: str = "", *, already_defined: frozenset[str] | None = None
) -> list[StaticFinding]:
    """What is decidably wrong with this code, in the order it will bite.

    Empty when nothing is decidable, which is not a claim that the code is
    correct. Never raises on the caller's code.
    """

    text = str(code or "")
    if not text.strip():
        return []

    unparseable = _does_not_parse(text)
    if unparseable is not None:
        # Nothing after this is meaningful: every other check reads the tree.
        return [unparseable]

    # Names the runtime lays in before the code starts are defined, whatever
    # the file looks like on its own. ``Q("3 m")`` has no import above it and
    # needs none; a checker that does not know refuses working code.
    provided = already_defined
    if provided is None:
        try:
            from core.sandbox.runner import names_the_sandbox_provides

            provided = names_the_sandbox_provides(library_root)
        except (ImportError, OSError, RuntimeError, TypeError, ValueError):
            provided = frozenset()
    findings = [
        finding
        for finding in _names_that_are_not_defined(text)
        if not _names_one_of(finding, provided)
    ]
    if library_root:
        findings.extend(
            _from_api(f) for f in check_code_against_library(text, library_root)
        )
    return sorted(findings, key=lambda f: (f.line, f.said))


def describe_findings(findings: list[StaticFinding]) -> str:
    """The findings as something the writer of the code can act on."""

    if not findings:
        return ""
    return "This code was not run, because:\n" + "\n".join(
        f"  {f.describe()}" for f in findings
    )
