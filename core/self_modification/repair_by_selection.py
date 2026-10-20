"""Repairing her own code by selection: a failing test is the defect, and tests choose what is kept.

Her self-modification loop never staged a fix. Its "bugs" were runtime
degradations read from her log (a timeout in the inference gate), which no
edit to a line can mend, and its fixes were written by the 1-bit Brainstem,
whose code did not parse ("unterminated string literal"), because her 27B was
reserved for conversation (her log, 10-18 October 2026).

Here a defect is a test that fails: something a change can be checked
against. Each candidate is a set of edits to the files the defect names. Its
fitness is measured by running tests on a copy: how many of the failing tests
now pass, and how many of the tests that passed before the change fail after
it. A child is written from the best candidate so far, shown what still fails
under it, so a line of descent improves on its parent or is dropped. A
candidate that fixes every failing test and breaks nothing ends the search.
Nothing here applies a change to her running code; the winner is written out
for the gates and a person to promote.

The proposals come from a callable, so her model writes them in use and a
test can supply them.
"""
from __future__ import annotations

import ast
import hashlib
import logging
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("SelfModification.RepairBySelection")

__all__ = [
    "Candidate",
    "Defect",
    "Edit",
    "Fitness",
    "Lineage",
    "applied",
    "edits_in",
    "repair_by_selection",
    "the_code_a_failure_passes_through",
]


@dataclass(frozen=True)
class Defect:
    """Tests that fail, the files a change may touch, and what the failure said."""

    tests: tuple[str, ...]
    files: tuple[str, ...]
    seen: str = ""


@dataclass(frozen=True)
class Edit:
    """Replace the one place ``old`` occurs in ``path`` with ``new``."""

    path: str
    old: str
    new: str


@dataclass(frozen=True)
class Fitness:
    """What running the tests on a candidate showed."""

    fixed: int
    of: int
    broken: int
    changed_lines: int
    failures: str = ""
    #: Whether the tests that passed before were run on it: only once it fixes something.
    guarded: bool = True
    #: Findings her code standards report in the changed files that the original did not have.
    below_standard: int = 0

    @property
    def mends(self) -> bool:
        return (self.of > 0 and self.fixed == self.of and self.broken == 0 and self.guarded
                and self.below_standard == 0)

    def key(self) -> tuple[int, int, int, int]:
        """Larger is fitter: more fixed, then fewer broken, then fewer findings, then the smaller change."""
        return (self.fixed, -self.broken, -self.below_standard, -self.changed_lines)


@dataclass
class Candidate:
    id: str
    parent: str
    edits: tuple[Edit, ...]
    origin: str
    fitness: Fitness | None = None
    refused: str = ""


@dataclass
class Lineage:
    """Every candidate tried, the baseline it was measured against, and the fittest."""

    defect: Defect
    baseline: dict[str, bool] = field(default_factory=dict)
    candidates: list[Candidate] = field(default_factory=list)
    best: Candidate | None = None
    stopped: str = ""

    def to_json(self) -> dict[str, Any]:
        def one(c: Candidate) -> dict[str, Any]:
            return {"id": c.id, "parent": c.parent, "origin": c.origin, "refused": c.refused,
                    "edits": [vars(e) for e in c.edits],
                    "fitness": None if c.fitness is None else {k: v for k, v in vars(c.fitness).items()
                                                               if k != "failures"}}

        return {"defect": vars(self.defect), "baseline": self.baseline, "stopped": self.stopped,
                "best": self.best.id if self.best else "", "candidates": [one(c) for c in self.candidates]}


_FENCE = re.compile(r"^```[\w+-]*[ \t]*$", re.MULTILINE)
_BLOCK = re.compile(
    r"^(?P<path>[^\n<>=]+?\.\w+)[ \t]*\n(?:[ \t]*\n)*<<<<<<< SEARCH\n(?P<old>.*?)\n?=======\n(?P<new>.*?)\n?>>>>>>> REPLACE",
    re.DOTALL | re.MULTILINE,
)


def edits_in(text: str, files: Sequence[str]) -> tuple[Edit, ...]:
    """The SEARCH/REPLACE blocks in a proposal, for the files the defect allows; others are left out."""
    allowed = {str(f) for f in files}
    found = []
    for m in _BLOCK.finditer(_FENCE.sub("", str(text or ""))):
        path = m.group("path").strip().strip("`*").strip()
        if path in allowed:
            found.append(Edit(path, m.group("old"), m.group("new")))
    return tuple(found)


def applied(sources: Mapping[str, str], edits: Sequence[Edit]) -> tuple[dict[str, str], str]:
    """The sources with every edit made, or why not: an edit whose text is absent or not unique is refused."""
    out = dict(sources)
    for edit in edits:
        current = out.get(edit.path)
        if current is None:
            return {}, f"{edit.path} is not one of the files this defect allows"
        count = current.count(edit.old) if edit.old else 0
        if count != 1:
            return {}, f"the text to replace occurs {count} times in {edit.path}"
        out[edit.path] = current.replace(edit.old, edit.new, 1)
    for path, text in out.items():
        if path.endswith(".py") and text != sources.get(path):
            try:
                ast.parse(text)
            except SyntaxError as exc:
                return {}, f"{path} does not parse after the edit: {exc.msg} at line {exc.lineno}"
    return out, ""


def _changed_lines(before: Mapping[str, str], after: Mapping[str, str]) -> int:
    import difflib

    return sum(
        1
        for path in after
        for line in difflib.unified_diff(before.get(path, "").splitlines(), after[path].splitlines(), lineterm="")
        if line[:1] in "+-" and not line.startswith(("+++", "---"))
    )


def the_code_a_failure_passes_through(failure: str, sources: Mapping[str, str]) -> dict[str, list[str]]:
    """For each allowed file, the whole functions a traceback's lines fall in, or the file when none do."""
    out: dict[str, list[str]] = {}
    for path, text in sources.items():
        lines = {int(n) for n in re.findall(rf"{re.escape(path)}:(\d+)", failure)}
        lines |= {int(n) for n in re.findall(rf'File "[^"]*{re.escape(path)}", line (\d+)', failure)}
        try:
            tree = ast.parse(text)
        except SyntaxError:
            out[path] = [text]
            continue
        spans = [(node.lineno, node.end_lineno or node.lineno) for node in ast.walk(tree)
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        chosen = sorted({min((s for s in spans if s[0] <= n <= s[1]), key=lambda s: s[1] - s[0])
                         for n in lines if any(s[0] <= n <= s[1] for s in spans)})
        rows = text.splitlines()
        out[path] = ["\n".join(rows[a - 1:b]) for a, b in chosen] or [text]
    return out


Run = Callable[[Sequence[str], Mapping[str, str]], Awaitable[tuple[dict[str, bool], str]]]
Propose = Callable[[Defect, Mapping[str, str], "Candidate | None", int], Awaitable[list[tuple[str, str]]]]


async def repair_by_selection(
    defect: Defect,
    *,
    sources: Mapping[str, str],
    neighbours: Sequence[str] | Callable[[], Awaitable[Sequence[str]]],
    run: Run,
    propose: Propose,
    proposals: int,
    deadline: float,
    record: Callable[[Candidate], None] | None = None,
    standards: Callable[[Mapping[str, str]], Awaitable[int]] | None = None,
) -> Lineage:
    """Search for a change that makes the defect's tests pass and breaks none of ``neighbours``.

    ``run(tests, sources)`` runs ``tests`` on a copy holding ``sources`` and
    returns each test's pass or fail and the failure text. ``propose(defect,
    start, parent, n)`` returns up to ``n`` (origin, text) proposals, each text
    holding SEARCH/REPLACE blocks against ``start``: the original sources, or
    with a ``parent`` (the fittest candidate so far, whose failures it shows)
    the parent's code, whose edits the child inherits. ``proposals`` is how many
    may be asked for in all and ``deadline`` (time.time()) when to stop asking.

    The tests that must not break are run only on a candidate that fixes
    something, and are found the first time one does: ``neighbours`` is either
    those tests or a callable that returns the ones passing before the change.
    For a module many tests import, running them all on every candidate spent a
    defect's whole budget before its first candidate (18 October 2026).

    ``standards(changed)`` counts the findings her code standards report in the
    changed files beyond the original's. Passing the tests is not enough: her
    27B's only passing answer to a planted one-line fault left a dead loop,
    unused names and "Placeholder" comments in the file (18 October 2026).
    """
    lineage = Lineage(defect)
    lineage.baseline, seen = await run(defect.tests, sources)
    if all(lineage.baseline.get(t, False) for t in defect.tests):
        lineage.stopped = "the defect's tests already pass"
        return lineage
    if not defect.seen:
        defect = Defect(defect.tests, defect.files, seen)
        lineage.defect = defect
    standing: list[str] | None = None
    looked = False

    async def guarding() -> list[str] | None:
        """The tests that passed before, or None when they could not be run: then nothing is guarded."""
        nonlocal standing, looked
        if not looked:
            looked = True
            if callable(neighbours):
                found = await neighbours()
            else:
                before, _ = await run(tuple(neighbours), sources)
                found = [t for t in neighbours if before.get(t)]
                lineage.baseline.update(before)
            standing = None if found is None else [t for t in found if t not in defect.tests]
        return standing

    asked = 0
    while asked < proposals:
        if time.time() >= deadline:
            lineage.stopped = "deadline"
            break
        # A candidate is a parent once it mends something and meets her standards. Built on one that mends
        # nothing, a child would inherit a change with no evidence for it and lose the original lines its own
        # edits are written to; built on one below her standards, it would inherit what fell below them.
        best = lineage.best.fitness if lineage.best is not None else None
        parent = lineage.best if best is not None and best.fixed > 0 and best.below_standard == 0 else None
        # A child starts from its parent's code and inherits its edits; what it proposes is added to them.
        inherited = parent.edits if parent else ()
        start = applied(sources, inherited)[0] if parent else dict(sources)
        offered = await propose(defect, start, parent, proposals - asked)
        if not offered:
            lineage.stopped = "nothing proposed"
            break
        for origin, text in offered:
            asked += 1
            own = edits_in(text, defect.files)
            edits = (*inherited, *own)
            identity = hashlib.sha256(repr((parent.id if parent else "", edits)).encode()).hexdigest()[:12]
            candidate = Candidate(identity, parent.id if parent else "", edits, origin)
            changed, refused = applied(sources, edits) if own else ({}, "no edit to an allowed file")
            if refused:
                candidate.refused = refused
            else:
                results, failures = await run(defect.tests, changed)
                fixed = sum(1 for t in defect.tests if results.get(t))
                broken, guarded = 0, False
                guard = await guarding() if fixed else None
                # Guarded means tests that passed before were run on it and held. With none to run, nothing was
                # checked: on 19 October a change to a tool no test imports counted as mended with no guard at all.
                if guard:
                    held, broke = await run(guard, changed)
                    broken, guarded = sum(1 for t in guard if not held.get(t)), True
                    failures = "\n".join(part for part in (failures, broke) if part)
                candidate.fitness = Fitness(
                    fixed=fixed,
                    of=len(defect.tests),
                    broken=broken,
                    changed_lines=_changed_lines(sources, changed),
                    failures=failures,
                    guarded=guarded,
                    below_standard=await standards(changed) if standards is not None and fixed else 0,
                )
                if lineage.best is None or candidate.fitness.key() > lineage.best.fitness.key():
                    lineage.best = candidate
            lineage.candidates.append(candidate)
            if record is not None:
                record(candidate)
            logger.info("candidate %s from %s (%s): %s", candidate.id, candidate.parent or "the defect", origin,
                        candidate.refused or (vars(candidate.fitness) if candidate.fitness else ""))
            if lineage.best is not None and lineage.best.fitness.mends:
                lineage.stopped = "mended"
                return lineage
    if not lineage.stopped:
        lineage.stopped = "proposals spent"
    return lineage
