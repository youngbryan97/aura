#!/usr/bin/env python3
"""Find success criteria derived from the result they judge.

An adaptive mechanism that decides what counts as success after seeing how it
did has measured nothing. The shape is always the same and never looks wrong:

    baseline = mean(scores)          # the same scores about to be judged
    improved = score > baseline      # so about half of anything "improves"

    self.target = max(self.target, achieved)
    met = achieved >= self.target    # met by construction

    threshold = observed * 0.9
    passed = observed > threshold    # passes for every positive observed

This gate finds comparisons whose threshold side is derived, in the same
function, from the value being compared. It is a narrow rule on purpose: it
reports only where the dataflow is visible in one function body, so what it
reports is real. A mechanism can still choose a bad threshold in advance —
that is a different problem, and one a reader can at least see.

The baseline may only shrink.
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASELINE = ROOT / "config" / "epistemic_independence_baseline.json"

SKIP_PARTS = {"tests", "archive", "__pycache__", ".venv", "node_modules", "tools"}

#: Comparisons that judge. `==` is excluded: equality against a derived value
#: is usually a lookup rather than a verdict.
_JUDGING_OPS = (ast.Gt, ast.GtE, ast.Lt, ast.LtE)


def _names(node: ast.AST) -> set[str]:
    """Every name and simple attribute an expression reads."""
    found: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            found.add(child.id)
        elif isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
            found.add(f"{child.value.id}.{child.attr}")
    return found


def _targets(node: ast.AST) -> list[str]:
    out: list[str] = []
    targets: list[ast.expr] = []
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        targets = [node.target]
    for target in targets:
        if isinstance(target, ast.Name):
            out.append(target.id)
        elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
            out.append(f"{target.value.id}.{target.attr}")
    return out


def _derivation(func: ast.AST) -> dict[str, set[str]]:
    """name -> every name it transitively derives from, within this function."""
    direct: dict[str, set[str]] = {}
    for node in ast.walk(func):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            continue
        value = node.value if not isinstance(node, ast.AnnAssign) else node.value
        if value is None:
            continue
        sources = _names(value)
        for target in _targets(node):
            if isinstance(node, ast.AugAssign):
                sources = sources | {target}
            direct.setdefault(target, set()).update(sources - {target})

    closure: dict[str, set[str]] = {}
    for name in direct:
        seen: set[str] = set()
        frontier = set(direct[name])
        while frontier:
            current = frontier.pop()
            if current in seen:
                continue
            seen.add(current)
            frontier |= direct.get(current, set()) - seen
        closure[name] = seen
    return closure


#: Calls that turn the judged collection into a number to judge it against.
#: A threshold that is an aggregate of the data it judges is the shape this
#: gate exists for: about half of anything beats its own mean.
_AGGREGATES = frozenset(
    {
        "mean", "average", "median", "percentile", "quantile", "nanmean",
        "nanmedian", "stdev", "std", "pstdev", "variance", "var",
        "fmean", "geometric_mean", "harmonic_mean",
    }
)


#: Names that are not a measurement of anything, so a threshold built from
#: one is not a criterion built from its own result. A deadline is a clock, a
#: length is a size, and `self` names an object rather than an observation.
_NOT_A_MEASUREMENT = frozenset(
    {
        "self", "cls", "time", "time.time", "perf_counter", "time.perf_counter",
        "monotonic", "time.monotonic", "len", "float", "int", "round", "abs",
        "datetime", "now", "os", "sys", "math", "np", "numpy",
    }
)


def _aggregate_calls(node: ast.AST) -> set[str]:
    """Names that an aggregate in this expression was computed over."""
    found: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        if str(name) in _AGGREGATES:
            for argument in child.args:
                found |= _names(argument)
    return found


def _scaled_from(node: ast.AST) -> set[str]:
    """Names this expression is a plain arithmetic rescaling of.

    `threshold = observed * 0.9` passes for every positive observed, and
    `threshold = observed - 0.1` for every observed at all. Both are the same
    move: setting the bar from the thing it is meant to test.
    """
    if not isinstance(node, ast.BinOp):
        return set()
    if not isinstance(node.op, (ast.Mult, ast.Sub, ast.Add, ast.Div)):
        return set()
    sides = (node.left, node.right)
    constants = [s for s in sides if isinstance(s, ast.Constant)]
    if len(constants) != 1:
        return set()
    if isinstance(node.op, (ast.Add, ast.Sub)):
        # An offset is a criterion only when it is a small one. `deadline =
        # clock + 300` and `version = current + 1` are a clock and a counter;
        # `bar = observed - 0.05` is setting the bar from the result.
        offset = constants[0].value
        if not isinstance(offset, (int, float)) or abs(float(offset)) >= 1.0:
            return set()
    other = sides[0] if isinstance(sides[1], ast.Constant) else sides[1]
    return _names(other)


def _accumulator_targets(func: ast.AST) -> set[str]:
    """Names assigned inside the body of a comparison that guards them.

    `if score > best_score: best_score = score` is the running-maximum idiom.
    The threshold does derive from the judged value, and it is not a criterion
    anybody chose — it is a loop finding the largest element.
    """
    guarded: set[str] = set()
    for node in ast.walk(func):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        for statement in node.body:
            guarded.update(_targets(statement))
    for node in ast.walk(func):
        if isinstance(node, (ast.For, ast.While)):
            for statement in ast.walk(node):
                if isinstance(statement, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                    guarded.update(_targets(statement))
    return guarded


def _criterion_sources(func: ast.AST) -> dict[str, set[str]]:
    """name -> judged names it was built from, by aggregate or by rescaling.

    Narrower than "derives from" on purpose. Any long enough dataflow chain
    connects almost everything to almost everything, and a gate that reports
    that teaches people to ignore it.
    """
    sources: dict[str, set[str]] = {}
    for node in ast.walk(func):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            continue
        value = node.value
        if value is None:
            continue
        judged = _aggregate_calls(value) | _scaled_from(value)
        if not judged:
            continue
        judged -= _NOT_A_MEASUREMENT
        if not judged:
            continue
        for target in _targets(node):
            sources.setdefault(target, set()).update(judged - {target})
    return sources


def _findings_in(func: ast.AST, rel: str, class_name: str) -> list[str]:
    sources = _criterion_sources(func)
    if not sources:
        return []
    accumulators = _accumulator_targets(func)
    # The judged side is expanded through its full derivation, the threshold
    # side is not. `latest = scores[-1]` beside `baseline = mean(scores)` is
    # the commonest form of this defect and the narrow rule missed it: latest
    # and baseline share no name until you follow latest back to scores.
    lineage = _derivation(func)
    out: list[str] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Compare) or not node.ops:
            continue
        if not isinstance(node.ops[0], _JUDGING_OPS):
            continue
        left_names = _names(node.left)
        right_names = _names(node.comparators[0])
        for judged, threshold in ((left_names, right_names), (right_names, left_names)):
            wider = set(judged)
            for name in judged:
                wider |= lineage.get(name, set())
            wider -= _NOT_A_MEASUREMENT
            for candidate in threshold:
                if candidate in accumulators or candidate in judged:
                    continue
                overlap = sources.get(candidate, set()) & wider
                if not overlap:
                    continue
                name = getattr(func, "name", "?")
                where = f"{class_name}.{name}" if class_name else name
                out.append(
                    f"{rel}::{where}:{node.lineno}"
                    f" [{candidate} is an aggregate or rescaling of {sorted(overlap)[0]}]"
                )
                break
    return out


def scan() -> list[str]:
    findings: list[str] = []
    for path in sorted((ROOT / "core").rglob("*.py")):
        rel = str(path.relative_to(ROOT))
        if SKIP_PARTS.intersection(pathlib.Path(rel).parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        stack: list[tuple[ast.AST, str]] = [(tree, "")]
        while stack:
            node, class_name = stack.pop()
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.ClassDef):
                    stack.append((child, child.name))
                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    findings.extend(_findings_in(child, rel, class_name))
                else:
                    stack.append((child, class_name))
    return sorted(set(findings))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()

    findings = scan()
    baseline = {"count": 0, "sites": []}
    if BASELINE.exists():
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    limit = int(baseline.get("count", 0))

    print(f"🎯 Criteria derived from what they judge: {len(findings)} (baseline {limit})")

    if args.write_baseline:
        if len(findings) > limit and BASELINE.exists():
            print(f"❌ refusing to raise the baseline from {limit} to {len(findings)}.")
            return 1
        BASELINE.write_text(
            json.dumps(
                {
                    "description": (
                        "Comparisons whose threshold is derived, in the same "
                        "function, from the value being compared. May only "
                        "SHRINK. Each one is a mechanism that cannot fail."
                    ),
                    "count": len(findings),
                    "sites": findings,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"✅ baseline written at {len(findings)}")
        return 0

    if len(findings) > limit:
        new = sorted(set(findings) - set(baseline.get("sites", [])))
        print(f"❌ {len(findings) - limit} new self-referential criteria:")
        for item in new[:20]:
            print(f"   • {item}")
        print(
            "\nDeclare the criterion before the run with "
            "core/verify/epistemic_independence.py, or compute the threshold "
            "from data the result is not part of."
        )
        return 1

    print(f"✅ at or below baseline ({len(findings)} <= {limit})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
