"""A caller that always passes an empty argument has switched the code off.

``does_this_govern_anything`` asks whether a module has a production caller.
This asks the finer question one level down: the caller exists, and it hands
the mechanism a literal empty value every single time, so the body behind that
argument has never run.

Three of these were found by hand on 2026-09-07, all in code that was correct,
complete and covered by tests:

* ``IntentionLoop.revise`` pushes into the belief engine and writes a ledger
  transition, both guarded by ``if rec.belief_updates``. Its only production
  caller passed ``belief_updates=[]``, hardcoded, so every governed tool
  execution logged "0 belief updates, 0 self-model updates".
* ``record_verified_effects`` puts step receipts on the turn's effect ledger
  for any lane. One lane called it.
* The conversation resume handle was created, accepted and never consumed.

The shape is always the same and it is worth naming: passing a literal empty
value is a DECISION, written down, that this mechanism does not run here. That
is sometimes right — a caller can genuinely have nothing to contribute — which
is why this reports rather than refuses, and why a reviewed instance is
recorded in the baseline with its reason rather than argued about again.

The opposite failure is not better. A guard nobody can satisfy and a guard
nobody can fail are the same defect seen from two sides: a decision point that
only ever answers one way is not deciding anything. This finds the first kind
statically; the second kind needs the runtime and is not claimed here.
"""

from __future__ import annotations

import ast
import json
import pathlib
from dataclasses import dataclass, field

#: An EMPTY COLLECTION, and nothing else.
#:
#: ``False`` and ``0`` are excluded on purpose. A boolean flag set off is a
#: choice a caller is entitled to make and says nothing about whether the
#: mechanism can fire elsewhere; reading them as findings buried the real ones
#: a hundred and fifty deep. An empty list handed to a parameter the body
#: iterates or tests for truth is different: it says "there is nothing here to
#: work with", every time, from every caller.
#:
#: ``None`` is excluded too. It usually means "you decide", which is the
#: opposite of switching something off — and after the 2026-09-07 repair it is
#: exactly what ``revise`` reads as "work it out yourself".
def _is_empty_literal(node: ast.AST) -> bool:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return not node.elts
    if isinstance(node, ast.Dict):
        return not node.keys
    return False


@dataclass(frozen=True)
class ASwitchedOffArgument:
    """One parameter every caller hands an empty value."""

    module: str
    function: str
    parameter: str
    call_sites: tuple[str, ...] = field(default_factory=tuple)

    @property
    def identity(self) -> str:
        return f"{self.module}:{self.function}:{self.parameter}"


def _module_name(path: pathlib.Path, root: pathlib.Path) -> str:
    return str(path.relative_to(root).with_suffix("")).replace("/", ".")


def _production_files(root: pathlib.Path) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for package in ("core", "interface", "skills", "security", "llm", "executors"):
        base = root / package
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            parts = set(path.relative_to(root).parts)
            if parts & {"__pycache__", "tests", "test"}:
                continue
            files.append(path)
    return files


def _keyword_arguments_by_callee(
    trees: dict[str, ast.Module],
) -> dict[str, dict[str, list[tuple[str, ast.AST]]]]:
    """callee name -> parameter -> [(where it was called, what was passed)]."""

    passed: dict[str, dict[str, list[tuple[str, ast.AST]]]] = {}
    for module, tree in trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = node.func
            name = ""
            if isinstance(target, ast.Attribute):
                name = target.attr
            elif isinstance(target, ast.Name):
                name = target.id
            if not name:
                continue
            for keyword in node.keywords:
                if keyword.arg is None:
                    continue
                where = f"{module}:{getattr(node, 'lineno', 0)}"
                passed.setdefault(name, {}).setdefault(keyword.arg, []).append(
                    (where, keyword.value)
                )
    return passed


def switched_off_arguments(root: str = "") -> tuple[ASwitchedOffArgument, ...]:
    """Every parameter whose production callers all pass an empty literal."""

    base = pathlib.Path(root or ".").resolve()
    trees: dict[str, ast.Module] = {}
    for path in _production_files(base):
        try:
            trees[_module_name(path, base)] = ast.parse(path.read_text("utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue

    passed = _keyword_arguments_by_callee(trees)
    found: list[ASwitchedOffArgument] = []
    for module, tree in trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body_names = {
                inner.id
                for inner in ast.walk(node)
                if isinstance(inner, ast.Name) and isinstance(inner.ctx, ast.Load)
            }
            parameters = [
                argument.arg
                for argument in (*node.args.args, *node.args.kwonlyargs)
                if argument.arg not in {"self", "cls"}
            ]
            for parameter in parameters:
                if parameter not in body_names:
                    # Unused parameters are a different finding and a noisier
                    # one; ruff already has an opinion about them.
                    continue
                sites = passed.get(node.name, {}).get(parameter, [])
                if not sites:
                    continue
                if not all(_is_empty_literal(value) for _where, value in sites):
                    continue
                found.append(
                    ASwitchedOffArgument(
                        module=module,
                        function=node.name,
                        parameter=parameter,
                        call_sites=tuple(sorted(where for where, _v in sites)),
                    )
                )
    return tuple(sorted(found, key=lambda item: item.identity))


def load_baseline(path: pathlib.Path) -> dict[str, str]:
    try:
        payload = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    reviewed = payload.get("reviewed")
    return dict(reviewed) if isinstance(reviewed, dict) else {}


def load_baseline_reasons(path: pathlib.Path) -> tuple[dict[str, str], dict[str, str]]:
    """Reviewed entries and recorded-but-unreviewed ones, kept apart.

    Being listed is not approval. Only a reason is approval, and the two
    buckets exist so a finding cannot be settled by appearing in a file.
    """

    try:
        payload = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return {}, {}
    reviewed = payload.get("reviewed")
    recorded = payload.get("recorded_not_yet_reviewed")
    return (
        dict(reviewed) if isinstance(reviewed, dict) else {},
        dict(recorded) if isinstance(recorded, dict) else {},
    )


def report(root: str = "", baseline: pathlib.Path | None = None) -> dict[str, object]:
    """What can never fire, split into new, recorded, and reviewed."""

    base = pathlib.Path(root or ".").resolve()
    reviewed, recorded = load_baseline_reasons(
        baseline or (base / "config/switched_off_arguments.json")
    )
    found = switched_off_arguments(str(base))
    identities = {item.identity for item in found}
    return {
        "found": [item.identity for item in found],
        "new": sorted(identities - set(reviewed) - set(recorded)),
        "reviewed_still_present": sorted(identities & set(reviewed)),
        "outstanding": sorted(identities & set(recorded)),
        "reviewed_gone": sorted(set(reviewed) - identities),
        "recorded_gone": sorted(set(recorded) - identities),
    }
