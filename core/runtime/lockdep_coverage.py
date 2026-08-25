"""How much of the locking in this codebase can lockdep actually see?

``core/runtime/lockdep.py`` finds ABBA deadlocks without the deadlock having
to happen — but only for locks it wraps. CLAUDE.md says so plainly: "it only
sees locks it wraps." That sentence is a coverage claim with no number
behind it, and an ABBA detector covering an unknown fraction of the system
gives an assurance nobody can size.

This measures it. Every ``threading.Lock()``, ``threading.RLock()``,
``asyncio.Lock()`` and friend constructed in ``core/`` and ``interface/`` is
invisible to lockdep unless it is wrapped by ``checked_lock`` /
``checked_async_lock``, or its critical section is adopted with
``instrument(name)``.

The honest first reading is 721 raw constructions. That is not a number to
fix in one pass — migrating a lock carelessly causes exactly the deadlock
the migration is meant to prevent, and 721 careless migrations would be a
catastrophe. So this exists to do two things a mass edit cannot:

* publish the coverage fraction, so "lockdep protects us" stops being an
  unquantified claim and becomes a number that can be driven up;
* support a ratchet, so the debt stops growing while it is paid down.

Reports rather than judges: a module-private lock guarding one dict is a
very different risk from a lock held across an await in the inference path,
and this cannot tell them apart. The caller decides.
"""
from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Constructors that create a lock primitive lockdep does not know about.
_RAW_LOCK_ATTRS = frozenset({
    "Lock", "RLock", "Condition", "Semaphore", "BoundedSemaphore",
})
_RAW_LOCK_MODULES = frozenset({"threading", "asyncio"})

#: The wrapped lane. Constructing through these IS lockdep coverage.
_CHECKED_CONSTRUCTORS = frozenset({
    "checked_lock", "checked_async_lock", "checked_async_condition",
    "checked_semaphore", "instrument",
})

_DEFAULT_ROOTS = ("core", "interface")

_SKIP_PARTS = frozenset({
    "__pycache__", ".venv", "node_modules", ".git", "build", "dist",
    ".claude", "artifacts", "tests",
})

#: lockdep's own module. The raw primitives inside CheckedLock,
#: CheckedAsyncLock and CheckedSemaphore ARE the wrappers — reporting them as
#: "locks lockdep cannot see" is the abstraction accusing its own
#: implementation. They were carried as four baseline entries instead, which
#: meant every new wrapper had to grow a baseline that is only allowed to
#: shrink. Excluded structurally, so the baseline shrank by those four.
_SKIP_FILES = frozenset({"core/runtime/lockdep.py"})


@dataclass(frozen=True)
class RawLock:
    """One lock primitive lockdep cannot see."""

    path: str
    line: int
    kind: str
    function: str = ""

    def key(self) -> str:
        """Identity for a baseline entry, insensitive to line drift."""
        return f"{self.path}::{self.function}::{self.kind}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "line": self.line,
            "kind": self.kind,
            "function": self.function,
        }


@dataclass
class LockdepCoverageReport:
    raw: list[RawLock] = field(default_factory=list)
    checked: int = 0

    @property
    def keys(self) -> set[str]:
        return {lock.key() for lock in self.raw}

    @property
    def total(self) -> int:
        return len(self.raw) + self.checked

    @property
    def coverage(self) -> float:
        """Fraction of lock primitives lockdep can reason about.

        Zero total reports 0.0 rather than 1.0: a codebase with no locks
        found has not achieved full coverage, it has failed to measure.
        """
        return (self.checked / self.total) if self.total else 0.0

    def new_since(self, baseline: Iterable[str]) -> list[RawLock]:
        known = set(baseline)
        return sorted(
            (lock for lock in self.raw if lock.key() not in known),
            key=lambda lock: (lock.path, lock.line),
        )

    def fixed_since(self, baseline: Iterable[str]) -> list[str]:
        """Baseline entries that are gone — the half that makes it a ratchet.

        A baseline that only grows is a list of excuses; entries that have
        been migrated must leave it or the next regression hides behind a
        stale allowance.
        """
        return sorted(set(baseline) - self.keys)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "aura.lockdep_coverage.v1",
            "raw": len(self.raw),
            "checked": self.checked,
            "total": self.total,
            "coverage": round(self.coverage, 4),
        }


#: Counting primitives that register no lock ordering. CheckedSemaphore says
#: the same thing: N simultaneous holders have no ordering to violate, so the
#: only property the wrapper adds is a BOUNDED acquire.
_COUNTING_PRIMITIVES = frozenset({"Semaphore", "BoundedSemaphore"})


class _LockVisitor(ast.NodeVisitor):
    def __init__(self, rel_path: str) -> None:
        self.rel_path = rel_path
        self.raw: list[RawLock] = []
        self.checked = 0
        self._scope: list[str] = []
        #: name -> RawLock, for counting primitives only. Resolved against the
        #: acquires below once the whole module has been walked.
        self._counting: dict[str, RawLock] = {}
        #: Names acquired at least once WITHOUT blocking=False.
        self._waited_on: set[str] = set()
        #: Names acquired at all.
        self._acquired: set[str] = set()

    def _enter(self, node: Any) -> None:
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    visit_FunctionDef = _enter          # noqa: N815 - ast API casing
    visit_AsyncFunctionDef = _enter     # noqa: N815
    visit_ClassDef = _enter             # noqa: N815

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802 - ast API
        """Remember the NAME a counting primitive was assigned to.

        A semaphore acquired only with ``blocking=False`` cannot wedge — the
        call returns immediately either way — so it is not what this gate is
        looking for. Deciding that needs the name, and the name is only here.
        """
        name = _assigned_name(node)
        if name:
            for raw in _counting_constructions(node.value, self.rel_path, self._scope):
                self._counting[name] = raw
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 - ast API
        func = node.func
        if isinstance(func, ast.Attribute):
            module = getattr(func.value, "id", "")
            if module in _RAW_LOCK_MODULES and func.attr in _RAW_LOCK_ATTRS:
                self.raw.append(
                    RawLock(
                        path=self.rel_path,
                        line=node.lineno,
                        kind=f"{module}.{func.attr}",
                        function=".".join(self._scope),
                    )
                )
            elif func.attr == "acquire":
                target = _receiver_name(func.value)
                if target:
                    self._acquired.add(target)
                    if not _is_non_blocking_acquire(node):
                        self._waited_on.add(target)
        elif isinstance(func, ast.Name) and func.id in _CHECKED_CONSTRUCTORS:
            self.checked += 1
        self.generic_visit(node)

    def resolve(self) -> None:
        """Drop counting primitives this module never waits on."""
        exempt = {
            raw
            for name, raw in self._counting.items()
            if name in self._acquired and name not in self._waited_on
        }
        if exempt:
            self.raw = [raw for raw in self.raw if raw not in exempt]


def _assigned_name(node: ast.Assign) -> str:
    for target in node.targets:
        if isinstance(target, ast.Name):
            return target.id
        if isinstance(target, ast.Attribute):
            return target.attr
    return ""


def _receiver_name(value: ast.expr) -> str:
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        return value.attr
    return ""


def _counting_constructions(
    value: ast.expr, rel_path: str, scope: list[str]
) -> list[RawLock]:
    if not isinstance(value, ast.Call):
        return []
    func = value.func
    if not isinstance(func, ast.Attribute):
        return []
    module = getattr(func.value, "id", "")
    if module not in _RAW_LOCK_MODULES or func.attr not in _COUNTING_PRIMITIVES:
        return []
    return [
        RawLock(
            path=rel_path,
            line=value.lineno,
            kind=f"{module}.{func.attr}",
            function=".".join(scope),
        )
    ]


def _is_non_blocking_acquire(node: ast.Call) -> bool:
    for keyword in node.keywords:
        if keyword.arg == "blocking":
            return isinstance(keyword.value, ast.Constant) and keyword.value.value is False
    if node.args:
        first = node.args[0]
        return isinstance(first, ast.Constant) and first.value is False
    return False


def scan_lock_coverage(
    *,
    roots: Iterable[str] = _DEFAULT_ROOTS,
    repo_root: Path | None = None,
) -> LockdepCoverageReport:
    """Measure how many lock primitives lockdep can see."""
    root = repo_root or Path(__file__).resolve().parent.parent.parent
    report = LockdepCoverageReport()
    for name in roots:
        base = root / name
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            rel = path.relative_to(root)
            # Match the skip list against the path RELATIVE to the root. The
            # absolute path is the wrong thing to test: a worktree lives under
            # ``.claude/worktrees/`` — a skipped part — so every file in it
            # would be skipped, the scan would find nothing, and the ratchet
            # would read that as "the whole baseline was migrated".
            if _SKIP_PARTS.intersection(rel.parts):
                continue
            if rel.as_posix() in _SKIP_FILES:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
            visitor = _LockVisitor(str(rel))
            visitor.visit(tree)
            visitor.resolve()
            report.raw.extend(visitor.raw)
            report.checked += visitor.checked
    report.raw.sort(key=lambda lock: (lock.path, lock.line))
    return report


__all__ = [
    "LockdepCoverageReport",
    "RawLock",
    "scan_lock_coverage",
]
