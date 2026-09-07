"""core/connectome/dataflow.py — a call count is not a weight.

The reconstruction counts call sites and treats a pair joined by twenty of them
as twenty times connected. That is right for control — twenty places really do
hand control over — and wrong for information. One function can call another ten
thousand times and use nothing it returns, and two functions can be tightly
coupled through a single call whose value decides everything that follows.

So the graph needs a second weight on the same edges: what happens to what comes
back. Six things can, and they are not the same:

``discarded``
    The call is a statement. Nothing comes back that anyone reads. However many
    times this happens, no information crossed.
``logged``
    The value reaches a logger or a degradation record and stops. It left the
    function only to be written down.
``local``
    Bound to a name and used inside the function, and no further.
``branch``
    The value decides which way the caller goes. This is the strongest kind of
    short coupling: the callee is choosing the caller's control flow.
``returned``
    The value leaves through the caller's own return, so the coupling is not
    between these two cells at all — it reaches whatever called the caller.
``escapes``
    Stored on an attribute or a global, so it outlives the call and is readable
    by cells that were never in this path.

The analysis is a small def-use walk inside one function: bind the call's value
to a name, find every use of that name, and take the strongest consequence any
of them has. It does not follow values across functions, and it says so — a
value that is returned is marked ``returned`` rather than traced into the
caller's caller, because doing that properly is interprocedural analysis and
guessing at it would produce a weight that looks measured and is not.
"""

from __future__ import annotations

import ast
import logging
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from .types import ConnectomeSnapshot

logger = logging.getLogger("Aura.Connectome.Dataflow")

__all__ = [
    "Consequence",
    "CONSEQUENCE_WEIGHT",
    "DataflowMap",
    "extract_dataflow",
    "weight_edges",
    "dataflow_report",
]


class Consequence(StrEnum):
    """What became of the value, worst to best for information transfer."""

    DISCARDED = "discarded"
    LOGGED = "logged"
    LOCAL = "local"
    BRANCH = "branch"
    RETURNED = "returned"
    ESCAPES = "escapes"


#: How much information an edge carries, by what its value does. Discarded is
#: zero on purpose: the control edge is still there and this weight is not about
#: control. The rest are ordered by how far the value travels from the call, and
#: the numbers are a ranking made explicit rather than a measurement — anything
#: derived from them says so.
CONSEQUENCE_WEIGHT: dict[Consequence, float] = {
    Consequence.DISCARDED: 0.0,
    Consequence.LOGGED: 0.1,
    Consequence.LOCAL: 0.5,
    Consequence.BRANCH: 0.8,
    Consequence.RETURNED: 1.0,
    Consequence.ESCAPES: 1.0,
}

_RANK: dict[Consequence, int] = {
    Consequence.DISCARDED: 0,
    Consequence.LOGGED: 1,
    Consequence.LOCAL: 2,
    Consequence.BRANCH: 3,
    Consequence.RETURNED: 4,
    Consequence.ESCAPES: 5,
}

#: Calls whose only job is to write the value down somewhere a person reads.
_SINK_NAMES: frozenset[str] = frozenset(
    {
        "debug",
        "info",
        "warning",
        "warn",
        "error",
        "exception",
        "critical",
        "log",
        "record_degradation",
        "print",
        "repr",
        "format",
    }
)


@dataclass
class DataflowMap:
    """Per call site, what happened to the value it returned."""

    by_locus: dict[str, Consequence] = field(default_factory=dict)
    files_read: int = 0
    parse_failures: int = 0

    def counts(self) -> dict[str, int]:
        out = {str(c): 0 for c in Consequence}
        for consequence in self.by_locus.values():
            out[str(consequence)] += 1
        return out

    def summary(self) -> dict[str, Any]:
        counts = self.counts()
        total = sum(counts.values()) or 1
        return {
            "call_sites": sum(counts.values()),
            "files_read": self.files_read,
            "parse_failures": self.parse_failures,
            "counts": counts,
            "shares": {name: round(value / total, 4) for name, value in counts.items()},
            "carries_nothing_share": round(
                (counts[str(Consequence.DISCARDED)] + counts[str(Consequence.LOGGED)]) / total,
                4,
            ),
        }


class _FlowVisitor(ast.NodeVisitor):
    """Bind each call's value to a name and find the strongest use of it.

    Two passes over one function body. The first records where each call is and
    what name its value was bound to; the second walks the body again looking
    for those names. Doing it in one pass would miss a use that appears before
    the binding in the tree but after it in execution, which happens in loops.
    """

    def __init__(self, module: str) -> None:
        self.module = module
        self.calls: dict[str, Consequence] = {}
        self._bindings: dict[str, list[str]] = {}
        self._pending: list[tuple[str, ast.expr]] = []

    # -- first pass: where the calls are and what they are bound to ------

    def _locus(self, node: ast.AST) -> str:
        return f"{self.module}:{getattr(node, 'lineno', 0)}:{getattr(node, 'col_offset', 0)}"

    def _record(self, node: ast.Call, consequence: Consequence) -> None:
        locus = self._locus(node)
        current = self.calls.get(locus)
        if current is None or _RANK[consequence] > _RANK[current]:
            self.calls[locus] = consequence

    def visit_Expr(self, node: ast.Expr) -> None:  # noqa: N802
        if isinstance(node.value, ast.Call):
            name, _ = _target(node.value.func)
            self._record(
                node.value,
                Consequence.LOGGED if name in _SINK_NAMES else Consequence.DISCARDED,
            )
            for argument in [*node.value.args, *(kw.value for kw in node.value.keywords)]:
                self.visit(argument)
            self.visit(node.value.func)
            return
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:  # noqa: N802
        if isinstance(node.value, ast.Call):
            self._record(node.value, Consequence.RETURNED)
        elif isinstance(node.value, ast.Name):
            self._pending.append((node.value.id, node))
        if node.value is not None:
            for inner in ast.walk(node.value):
                if isinstance(inner, ast.Call) and inner is not node.value:
                    self._record(inner, Consequence.RETURNED)
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:  # noqa: N802
        self._mark_condition(node.test)
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:  # noqa: N802
        self._mark_condition(node.test)
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:  # noqa: N802
        self._mark_condition(node.test)
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:  # noqa: N802
        self._mark_condition(node.test)
        self.generic_visit(node)

    def _mark_condition(self, test: ast.expr) -> None:
        for inner in ast.walk(test):
            if isinstance(inner, ast.Call):
                self._record(inner, Consequence.BRANCH)
            elif isinstance(inner, ast.Name):
                self._pending.append((inner.id, test))

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        escapes = any(isinstance(target, ast.Attribute) for target in node.targets)
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if isinstance(node.value, ast.Call):
            locus = self._locus(node.value)
            self._record(node.value, Consequence.ESCAPES if escapes else Consequence.LOCAL)
            for name in names:
                self._bindings.setdefault(name, []).append(locus)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:  # noqa: N802
        if isinstance(node.value, ast.Call):
            self._record(
                node.value,
                Consequence.ESCAPES
                if isinstance(node.target, ast.Attribute)
                else Consequence.LOCAL,
            )
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:  # noqa: N802
        for name in node.names:
            for locus in self._bindings.get(name, []):
                self._promote(locus, Consequence.ESCAPES)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        locus = self._locus(node)
        if locus not in self.calls:
            self.calls[locus] = Consequence.LOCAL
        for argument in [*node.args, *(kw.value for kw in node.keywords)]:
            if isinstance(argument, ast.Name):
                self._pending.append((argument.id, node))
            elif isinstance(argument, ast.Call):
                self._record(argument, Consequence.LOCAL)
        self.generic_visit(node)

    # -- second pass ----------------------------------------------------

    def _promote(self, locus: str, consequence: Consequence) -> None:
        current = self.calls.get(locus)
        if current is None or _RANK[consequence] > _RANK[current]:
            self.calls[locus] = consequence

    def resolve(self) -> None:
        """Push each use of a bound name back onto the call that produced it."""
        for name, context in self._pending:
            loci = self._bindings.get(name)
            if not loci:
                continue
            if isinstance(context, ast.Return):
                consequence = Consequence.RETURNED
            elif isinstance(context, ast.Call):
                target, _ = _target(context.func)
                consequence = (
                    Consequence.LOGGED if target in _SINK_NAMES else Consequence.LOCAL
                )
            else:
                consequence = Consequence.BRANCH
            for locus in loci:
                self._promote(locus, consequence)


def _target(func: ast.expr) -> tuple[str, str | None]:
    if isinstance(func, ast.Name):
        return func.id, None
    if isinstance(func, ast.Attribute):
        base = func.value
        return func.attr, base.id if isinstance(base, ast.Name) else None
    return "", None


def _module_name(path: Path, repo: Path) -> str:
    try:
        rel = path.relative_to(repo).with_suffix("")
    except ValueError:
        return ""
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _iter_files(root: Path, roots: Sequence[str]) -> Iterator[Path]:
    skip = {"__pycache__", ".git", ".venv", "node_modules", ".claude", "artifacts", "data"}
    for name in roots:
        base = root / name
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in skip)
            for filename in sorted(filenames):
                if filename.endswith(".py") and not filename.startswith("test_"):
                    yield Path(dirpath) / filename


def extract_dataflow(
    repo: str | Path,
    *,
    roots: Sequence[str] = ("core", "interface", "skills", "security", "llm", "executors"),
) -> DataflowMap:
    """Walk the source and classify every call site by what its value does."""
    root = Path(repo)
    result = DataflowMap()
    for path in _iter_files(root, roots):
        try:
            tree = ast.parse(path.read_bytes().decode("utf-8", "replace"), filename=str(path))
        except (SyntaxError, OSError):
            result.parse_failures += 1
            continue
        result.files_read += 1
        module = _module_name(path, root)
        for _qualname, node in _iter_functions(tree):
            visitor = _FlowVisitor(module)
            for statement in node.body:
                visitor.visit(statement)
            visitor.resolve()
            result.by_locus.update(visitor.calls)
    return result


def _iter_functions(
    tree: ast.AST, prefix: str = ""
) -> Iterator[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    for child in ast.iter_child_nodes(tree):
        if isinstance(child, ast.ClassDef):
            yield from _iter_functions(child, prefix=f"{prefix}{child.name}.")
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield f"{prefix}{child.name}", child


def weight_edges(
    snapshot: ConnectomeSnapshot,
    flow: DataflowMap,
    contacts_by_edge: Mapping[tuple[str, str], Sequence[str]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Give each drive edge an information weight beside its contact count.

    ``contacts_by_edge`` maps a pair to the loci of its call sites, which is what
    joins the two analyses: the reconstruction knows which cells a locus joins
    and this module knows what happened to the value at that locus.
    """
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for pair, loci in contacts_by_edge.items():
        consequences = [flow.by_locus.get(locus) for locus in loci]
        present = [c for c in consequences if c is not None]
        if not present:
            continue
        strongest = max(present, key=lambda c: _RANK[c])
        weight = sum(CONSEQUENCE_WEIGHT[c] for c in present) / len(present)
        out[pair] = {
            "contacts": len(loci),
            "classified": len(present),
            "strongest": str(strongest),
            "mean_weight": round(weight, 4),
            "carries_nothing": all(
                c in (Consequence.DISCARDED, Consequence.LOGGED) for c in present
            ),
        }
    return out


def dataflow_report(
    snapshot: ConnectomeSnapshot,
    weighted: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    limit: int = 20,
) -> dict[str, Any]:
    """The edges a call count overstates, and the ones it understates.

    The interesting pairs are at both ends. A pair joined by many calls that
    carry nothing is a coupling the control graph reports and the information
    graph does not. A pair joined once by a value the caller branches on is a
    coupling the control graph barely notices.
    """
    heavy_empty: list[dict[str, Any]] = []
    light_decisive: list[dict[str, Any]] = []
    for (pre, post), entry in weighted.items():
        pre_unit = snapshot.units.get(pre)
        post_unit = snapshot.units.get(post)
        if pre_unit is None or post_unit is None:
            continue
        row = {
            "pair": f"{pre_unit.name} -> {post_unit.name}",
            "contacts": entry["contacts"],
            "strongest": entry["strongest"],
            "mean_weight": entry["mean_weight"],
        }
        if entry["carries_nothing"] and entry["contacts"] >= 4:
            heavy_empty.append(row)
        elif entry["contacts"] == 1 and entry["strongest"] in {
            str(Consequence.BRANCH),
            str(Consequence.ESCAPES),
        }:
            light_decisive.append(row)
    heavy_empty.sort(key=lambda row: -row["contacts"])
    light_decisive.sort(key=lambda row: row["pair"])
    total = len(weighted) or 1
    carries_nothing = sum(1 for entry in weighted.values() if entry["carries_nothing"])
    return {
        "edges_weighted": len(weighted),
        "edges_carrying_nothing": carries_nothing,
        "carries_nothing_share": round(carries_nothing / total, 4),
        "mean_information_weight": round(
            sum(float(entry["mean_weight"]) for entry in weighted.values()) / total, 4
        ),
        "heaviest_carrying_nothing": heavy_empty[:limit],
        "single_call_decisive": light_decisive[:limit],
    }
