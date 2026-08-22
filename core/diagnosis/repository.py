"""Diagnose a repository by running it, not by reading it aloud.

LIVE, 2026-08-22. Asked to look at a small Python project whose test failed,
the turn routed to `os_automation` — desktop control — and timed out having
completed nothing. A question about code went to the mouse-and-keyboard lane.

There is a lot of code machinery in this tree and all of it is pointed at her
own source: repair, refactor, health, the AST analyser, the error intelligence.
None of it can be aimed at a directory somebody names.

This can. It finds how the project runs its tests, runs them through the
governed subprocess gateway, reads the failure the runner reports, and pulls
the source and the stated intent around the line that failed. Everything it
reports is something it observed. The language model's part comes afterwards
and is to explain the finding, not to find it.
"""

from __future__ import annotations

import ast
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "Failure",
    "RepositoryDiagnosis",
    "diagnose_repository",
    "describe_diagnosis",
]

#: A project that takes longer than this to fail is not one a chat turn waits on.
_RUN_TIMEOUT_S = 120.0

#: How much source to quote around the line that failed.
_CONTEXT_LINES = 6

#: Files that state what a project is supposed to do.
_INTENT_FILES = ("README.md", "README.rst", "README.txt", "readme.md")


@dataclass(frozen=True, slots=True)
class Failure:
    """One failing test, as the runner reported it."""

    test: str = ""
    message: str = ""
    file: str = ""
    line: int = 0
    assertion: str = ""


@dataclass(frozen=True, slots=True)
class RepositoryDiagnosis:
    """What running the project actually showed."""

    root: str
    ran: str = ""
    ok: bool = False
    exit_code: int = 0
    passed: int = 0
    failed: int = 0
    failures: tuple[Failure, ...] = ()
    source: str = ""
    called_functions: tuple[str, ...] = ()
    stated_intent: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)
    error: str = ""


def _find_runner(root: Path) -> tuple[list[str], str]:
    """How this project runs its tests, discovered rather than assumed."""
    if (root / "tests").is_dir() or list(root.glob("test_*.py")) or list(root.glob("*_test.py")):
        target = "tests" if (root / "tests").is_dir() else "."
        return (["-m", "pytest", target, "-q", "--no-header", "-p", "no:cacheprovider"], f"pytest {target}")
    if (root / "manage.py").is_file():
        return (["manage.py", "test"], "manage.py test")
    return ([], "")


_FAIL_LINE = re.compile(r"^(?P<file>[^\s:]+\.py):(?P<line>\d+):\s*(?P<kind>\w+Error|assert.*)$")
_SUMMARY = re.compile(r"(?P<failed>\d+)\s+failed(?:,\s*(?P<passed>\d+)\s+passed)?")
_PASSED_ONLY = re.compile(r"(?P<passed>\d+)\s+passed")
_FAILED_TEST = re.compile(r"^FAILED\s+(?P<test>\S+)\s*-?\s*(?P<message>.*)$", re.MULTILINE)
_ASSERT_LINE = re.compile(r"^E\s+(?P<text>assert .+|\w+Error:.+)$", re.MULTILINE)


def _parse_failures(output: str) -> tuple[list[Failure], int, int]:
    """Read the runner's own report of what failed."""
    failures: list[Failure] = []
    assertions = [match.group("text").strip() for match in _ASSERT_LINE.finditer(output)]
    locations = [
        (match.group("file"), int(match.group("line")))
        for line in output.splitlines()
        if (match := _FAIL_LINE.match(line.strip()))
    ]
    for index, match in enumerate(_FAILED_TEST.finditer(output)):
        file_name, line_number = locations[index] if index < len(locations) else ("", 0)
        failures.append(
            Failure(
                test=match.group("test").strip(),
                message=" ".join(match.group("message").split())[:400],
                file=file_name,
                line=line_number,
                assertion=assertions[index] if index < len(assertions) else "",
            )
        )
    summary = _SUMMARY.search(output)
    if summary:
        failed = int(summary.group("failed"))
        passed = int(summary.group("passed") or 0)
    else:
        only = _PASSED_ONLY.search(output)
        failed, passed = 0, int(only.group("passed")) if only else 0
    return failures, passed, failed


def _quote_source(root: Path, failure: Failure) -> tuple[str, tuple[str, ...]]:
    """The lines around the failure, and what the failing line calls."""
    if not failure.file:
        return "", ()
    target = (root / failure.file).resolve()
    if not target.is_file():
        candidates = list(root.rglob(Path(failure.file).name))
        if not candidates:
            return "", ()
        target = candidates[0]
    try:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "", ()
    start = max(0, failure.line - 1 - _CONTEXT_LINES)
    end = min(len(lines), failure.line + _CONTEXT_LINES)
    quoted = "\n".join(
        f"{number + 1:>5}{' >' if number + 1 == failure.line else '  '} {lines[number]}"
        for number in range(start, end)
    )
    called = _functions_called_on(lines, failure.line)
    return quoted, called


def _functions_called_on(lines: list[str], line_number: int) -> tuple[str, ...]:
    """Names the failing line calls, so the reader knows where to look next."""
    if not 1 <= line_number <= len(lines):
        return ()
    try:
        tree = ast.parse("\n".join(lines))
    except SyntaxError:
        return ()
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or getattr(node, "lineno", 0) != line_number:
            continue
        target = node.func
        if isinstance(target, ast.Attribute):
            names.append(target.attr)
        elif isinstance(target, ast.Name):
            names.append(target.id)
    return tuple(dict.fromkeys(names))


def _stated_intent(root: Path, names: tuple[str, ...]) -> str:
    """What the project says it is supposed to do, where it mentions these names."""
    for candidate in _INTENT_FILES:
        readme = root / candidate
        if not readme.is_file():
            continue
        try:
            text = readme.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        wanted = [name.lower() for name in names]
        for paragraph in re.split(r"\n\s*\n", text):
            lowered = paragraph.lower()
            if any(name in lowered for name in wanted):
                return " ".join(paragraph.split())[:600]
        return " ".join(text.split())[:400]
    return ""


def _run(root: Path, argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Run the project's tests through the governed gateway."""
    import sys

    from core.governance_context import local_internal_governed_scope
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    with local_internal_governed_scope("diagnosis.repository"):
        return get_subprocess_gateway().run(
            [sys.executable, *argv],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=_RUN_TIMEOUT_S,
            read_only=False,
            check=False,
            source="diagnosis.repository",
            accelerator_capability="none",
        )


def diagnose_repository(path: str | Path, *, argv: list[str] | None = None) -> RepositoryDiagnosis:
    """Run the project and report what failed, with the code around it."""
    root = Path(str(path)).expanduser()
    if not root.is_dir():
        return RepositoryDiagnosis(root=str(root), error=f"{root} is not a directory")
    command, described = (argv, " ".join(argv)) if argv else _find_runner(root)
    if not command:
        return RepositoryDiagnosis(
            root=str(root),
            error="no test runner was found in this project",
        )
    try:
        done = _run(root, command)
    except (OSError, subprocess.SubprocessError, RuntimeError, ImportError) as exc:
        return RepositoryDiagnosis(
            root=str(root), ran=described, error=f"{type(exc).__name__}: {exc}"[:300]
        )

    output = f"{done.stdout or ''}\n{done.stderr or ''}"
    failures, passed, failed = _parse_failures(output)
    source, called = ("", ())
    intent = ""
    if failures:
        source, called = _quote_source(root, failures[0])
        intent = _stated_intent(root, called or (failures[0].test,))
    return RepositoryDiagnosis(
        root=str(root),
        ran=described,
        ok=done.returncode == 0,
        exit_code=done.returncode,
        passed=passed,
        failed=failed,
        failures=tuple(failures),
        source=source,
        called_functions=called,
        stated_intent=intent,
    )


def describe_diagnosis(diagnosis: RepositoryDiagnosis) -> str:
    """The finding as text, or "" when there is nothing to report."""
    if diagnosis.error:
        return f"I could not run {diagnosis.root}: {diagnosis.error}"
    if diagnosis.ok:
        return f"I ran {diagnosis.ran} in {diagnosis.root}: {diagnosis.passed} passed, nothing failed."
    if not diagnosis.failures:
        return (
            f"I ran {diagnosis.ran} in {diagnosis.root} and it exited "
            f"{diagnosis.exit_code} without naming a failing test."
        )
    first = diagnosis.failures[0]
    lines = [
        f"I ran {diagnosis.ran}: {diagnosis.passed} passed, {diagnosis.failed} failed.",
        f"The failure is {first.test}.",
    ]
    if first.assertion:
        lines.append(f"What the runner reported: {first.assertion}")
    if first.file and first.line:
        lines.append(f"It fails at {first.file}:{first.line}.")
    if diagnosis.source:
        lines.append("Around that line:\n" + diagnosis.source)
    if diagnosis.called_functions:
        lines.append("The failing line calls: " + ", ".join(diagnosis.called_functions) + ".")
    if diagnosis.stated_intent:
        lines.append("What the project says it should do: " + diagnosis.stated_intent)
    return "\n".join(lines)
