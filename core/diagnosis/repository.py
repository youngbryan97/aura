"""Diagnose a repository by running it, not by reading it aloud.

LIVE, 2026-08-22. Asked to look at a small Python project whose test failed,
the turn routed to `os_automation` — desktop control — and timed out having
completed nothing. A question about code went to the mouse-and-keyboard lane.

There is a lot of code machinery in this tree and all of it is pointed at her
own source: repair, refactor, health, the AST analyser, the error intelligence.
None of it can be aimed at a directory somebody names.

This can. It asks the project how it can be run — a test suite, a script —
runs it through the governed subprocess gateway, and keeps what came back.

LIVE, later the same day: a project with no tests and no traceback, handed over
with a symptom, got "no test runner was found" and nothing else. A failing test
is one kind of evidence, not the only kind. So the diagnosis now gathers three,
each computed: what running the project produced, what the source says survives
a call (`core/diagnosis/carried_state.py`), and what the project's own README
claims that contradicts it. Findings are filed as hypotheses with the
scientific engine, so a diagnosis made in one session can be confirmed in
another and the belief moves.

Everything it reports is something it observed. The language model's part comes
afterwards and is to explain the finding, not to find it.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "Failure",
    "confirm_diagnosis",
    "RepositoryDiagnosis",
    "diagnose_repository",
    "describe_diagnosis",
]

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
    observations: tuple[object, ...] = field(default_factory=tuple)
    carried: tuple[object, ...] = field(default_factory=tuple)
    hypothesis_id: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)
    error: str = ""

    def evidence_count(self) -> int:
        """How many independent things agree that something is wrong."""
        return sum(
            (
                bool(self.failures),
                bool(self.carried),
                bool(self.stated_intent),
                any(not getattr(item, "ok", True) for item in self.observations),
            )
        )


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


def _file_of(finding: object) -> str:
    """Which file a piece of carried state is in."""
    return str(getattr(finding, "file", ""))


def _record_hypothesis(diagnosis: RepositoryDiagnosis, claim: str) -> str:
    """File the finding with the scientific engine so it outlives this turn.

    A diagnosis is a claim that can be wrong. Filed here it carries an expected
    observable — that acting on it fixes the symptom — which somebody can
    resolve later, and the belief moves with the answer rather than with how
    confidently it was said.
    """
    try:
        from core.cognition.scientific_engine import get_scientific_engine
    except (ImportError, RuntimeError):
        return ""
    agreeing = diagnosis.evidence_count()
    try:
        engine = get_scientific_engine()
        hypothesis_id = engine.form_hypothesis(
            claim,
            predicted_observable="symptom_resolved",
            expected=1.0,
            prior_confidence=min(0.35 + 0.2 * agreeing, 0.95),
        )
        # The receipt is what makes it an experiment rather than a note: the
        # ledger scores expected against observed when somebody resolves it.
        engine.run_experiment(hypothesis_id)
    except (TypeError, ValueError, RuntimeError, OSError) as exc:
        from core.runtime.errors import record_degradation

        record_degradation("diagnosis.repository", exc, action="skipped filing the hypothesis")
        return ""
    return hypothesis_id


def confirm_diagnosis(hypothesis_id: str, *, worked: bool) -> None:
    """Resolve a filed diagnosis with what actually happened to the symptom."""
    if not hypothesis_id:
        return
    try:
        from core.cognition.scientific_engine import get_scientific_engine

        get_scientific_engine().observe(hypothesis_id, 1.0 if worked else 0.0)
    except (ImportError, TypeError, ValueError, RuntimeError, OSError) as exc:
        from core.runtime.errors import record_degradation

        record_degradation("diagnosis.repository", exc, action="left the hypothesis open")


def _claim_of(diagnosis: RepositoryDiagnosis) -> str:
    """The finding stated as something that can turn out to be false."""
    if diagnosis.carried:
        first = diagnosis.carried[0]
        return (
            f"in {Path(diagnosis.root).name}, {getattr(first, 'name', 'state')} surviving "
            f"{getattr(first, 'function', 'a call')} explains the symptom"
        )
    if diagnosis.failures:
        return f"in {Path(diagnosis.root).name}, {diagnosis.failures[0].test} fails for the reason reported"
    return f"something in {Path(diagnosis.root).name} is wrong"


def diagnose_repository(path: str | Path, *, argv: list[str] | None = None) -> RepositoryDiagnosis:
    """Run the project every way it affords, and report what that showed."""
    from core.diagnosis.carried_state import carried_state
    from core.diagnosis.experiment import Affordance, affordances, observe

    root = Path(str(path)).expanduser()
    if not root.is_dir():
        return RepositoryDiagnosis(root=str(root), error=f"{root} is not a directory")

    if argv:
        ways: tuple[Affordance, ...] = (
            Affordance(kind="given", argv=tuple(argv), described=" ".join(argv),
                       why="you named this command"),
        )
    else:
        ways = affordances(root)
    if not ways:
        return RepositoryDiagnosis(
            root=str(root),
            error="nothing in this project runs: no tests, and no script that does work when loaded",
        )

    observations = [observe(root, ways[0])]
    failures, passed, failed = _parse_failures(observations[0].output)
    # A suite that passes has not accounted for a symptom, so the project is
    # also run the way a person runs it.
    if not failures:
        for way in ways[1:]:
            if way.kind == "entry point":
                observations.append(observe(root, way))
                break

    carried = carried_state(root)
    source, called = ("", ())
    intent = ""
    if failures:
        source, called = _quote_source(root, failures[0])
        intent = _stated_intent(root, called or (failures[0].test,))
    elif carried:
        first = carried[0]
        source, called = _quote_source(
            root, Failure(file=_file_of(first), line=int(getattr(first, "line", 0)))
        )
        intent = _stated_intent(root, (str(getattr(first, "name", "")), str(getattr(first, "function", ""))))

    diagnosis = RepositoryDiagnosis(
        root=str(root),
        ran=" and ".join(item.ran for item in observations),
        ok=all(item.ok for item in observations) and not failures and not carried,
        exit_code=observations[0].exit_code,
        passed=passed,
        failed=failed,
        failures=tuple(failures),
        source=source,
        called_functions=called,
        stated_intent=intent,
        observations=tuple(observations),
        carried=carried,
    )
    if not diagnosis.ok:
        object.__setattr__(diagnosis, "hypothesis_id", _record_hypothesis(diagnosis, _claim_of(diagnosis)))
    return diagnosis


def describe_diagnosis(diagnosis: RepositoryDiagnosis) -> str:
    """The finding as text, or "" when there is nothing to report."""
    from core.diagnosis.carried_state import describe_carried_state

    if diagnosis.error:
        return f"I could not run {diagnosis.root}: {diagnosis.error}"

    if diagnosis.ok and not diagnosis.carried:
        counted = f": {diagnosis.passed} passed, nothing failed" if diagnosis.passed else ""
        return f"I ran {diagnosis.ran} in {diagnosis.root}{counted}."

    lines: list[str] = []
    if diagnosis.failures:
        first = diagnosis.failures[0]
        lines.append(f"I ran {diagnosis.ran}: {diagnosis.passed} passed, {diagnosis.failed} failed.")
        lines.append(f"The failure is {first.test}.")
        if first.assertion:
            lines.append(f"What the runner reported: {first.assertion}")
        if first.file and first.line:
            lines.append(f"It fails at {first.file}:{first.line}.")
    else:
        for item in diagnosis.observations:
            ran, output = getattr(item, "ran", ""), getattr(item, "output", "")
            if getattr(item, "error", ""):
                lines.append(f"I could not run {ran}: {item.error}")
            elif output:
                lines.append(f"I ran {ran} and it printed:\n{output[:1200]}")
            else:
                lines.append(f"I ran {ran}; it printed nothing and exited {getattr(item, 'exit_code', 0)}.")

    if diagnosis.carried:
        lines.append(describe_carried_state(tuple(diagnosis.carried)))
    if diagnosis.source:
        lines.append("Around that line:\n" + diagnosis.source)
    if diagnosis.called_functions:
        lines.append("The failing line calls: " + ", ".join(diagnosis.called_functions) + ".")
    if diagnosis.stated_intent:
        lines.append("What the project says it should do: " + diagnosis.stated_intent)
    return "\n".join(line for line in lines if line)
