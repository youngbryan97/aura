#!/usr/bin/env python3
"""Count the subsystems that own a private copy of one canonical variable.

Aura's largest architectural risk is ontological duplication inside her own
software: many semi-independent mental organs, each maintaining its own
interpretation of how she is, with nothing deciding between them. The fix is
not a rewrite. It is a direction — each of those subsystems becomes an
estimator contributing to one canonical variable, and the count of private
copies goes down and never up.

This is that ratchet. It finds attributes whose names say they hold a
canonical quantity, in modules that are not the canonical package and are not
declared estimators, and it fails if there are more than the baseline. The
baseline may only shrink, exactly like the layering and writing ones.

What counts as a private copy is a judgement, so the rule is deliberately
narrow: an assignment to ``self.<name>`` where ``<name>`` is one of the
canonical quantities, in a module that does not estimate into the canonical
state. A module that estimates has declared what it is — an estimator with a
working copy — and a module that does not has a second answer.

Run with ``--write-baseline`` only when the count went DOWN. Refreshing a
baseline upward is how debt gets laundered into a green gate.
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASELINE = ROOT / "config" / "state_ownership_baseline.json"

#: Attribute names that name a canonical quantity. Short and specific: a name
#: here has to be one where a private copy is genuinely a second answer, not
#: an unrelated use of a common word.
OWNED_NAMES = frozenset(
    {
        # affect
        "valence", "_valence", "current_valence", "valence_index",
        "arousal", "_arousal", "current_arousal", "arousal_index",
        "engagement", "_engagement", "engagement_level", "engagement_index",
        "current_mood", "mood",
        # self
        "coherence", "_coherence", "self_coherence", "coherence_score",
        "continuity", "_continuity", "continuity_score", "temporal_continuity",
        # epistemic
        "uncertainty", "_uncertainty", "uncertainty_level",
        "current_uncertainty", "epistemic_uncertainty",
        # world
        "prediction_error", "_prediction_error",
        # body
        "fatigue", "_fatigue", "fatigue_level", "current_fatigue",
    }
)

#: Attributes matched by name that are NOT a private copy. A history, a lock,
#: a decay rate and a threshold all contain a canonical word and none of them
#: is a second answer to how she is.
NOT_A_COPY_SUFFIXES = (
    "_history", "_lock", "_task", "_rate", "_threshold", "_cache", "_events",
    "_decay", "_charges", "_backend", "_override", "_window", "_buffer",
    "_count", "_at", "_ts", "_path", "_key", "_id",
)

#: Directories that are not production state owners.
SKIP_PARTS = {"tests", "archive", "__pycache__", ".venv", "node_modules", "tools"}

CANONICAL_MARKER = "core.canonical.state"


def _is_estimator(source: str) -> bool:
    """Whether this module contributes to the canonical state."""
    return CANONICAL_MARKER in source


def _is_container(node: ast.AST | None) -> bool:
    """Whether an assignment is to a collection rather than to one number.

    `self._fatigue: dict[str, float] = {}` in the global workspace is
    per-bidder competition fatigue, not the organism's. Counting it would put
    a false positive in a ratchet, and a ratchet with false positives is one
    people learn to ignore.
    """
    if node is None:
        return False
    if isinstance(node, (ast.Dict, ast.List, ast.Set, ast.Tuple, ast.DictComp, ast.ListComp, ast.SetComp)):
        return True
    if isinstance(node, ast.Subscript):
        base = node.value
        name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
        return str(name).lower() in {"dict", "list", "set", "tuple", "deque", "mapping", "sequence"}
    if isinstance(node, ast.Name):
        return node.id.lower() in {"dict", "list", "set", "tuple", "deque"}
    if isinstance(node, ast.Call):
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        return str(name).lower() in {"dict", "list", "set", "tuple", "deque", "defaultdict", "counter"}
    return False


def _owned_assignments(tree: ast.AST) -> set[str]:
    found: set[str] = set()
    container_named: set[str] = set()
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        holder: ast.AST | None = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            holder = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            holder = node.annotation
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and _is_container(holder)
            ):
                container_named.add(target.attr)
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr in OWNED_NAMES
                and not target.attr.endswith(NOT_A_COPY_SUFFIXES)
            ):
                found.add(target.attr)
    return found - container_named


def scan() -> list[str]:
    """Every ``module::attribute`` that is a private copy of a canonical value."""
    findings: list[str] = []
    for path in sorted((ROOT / "core").rglob("*.py")):
        rel = path.relative_to(ROOT)
        if SKIP_PARTS.intersection(rel.parts):
            continue
        if rel.parts[:2] == ("core", "canonical"):
            continue
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        if _is_estimator(source):
            continue
        for attribute in sorted(_owned_assignments(tree)):
            findings.append(f"{rel}::{attribute}")
    return sorted(findings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="record the current count. Only valid when it went DOWN.",
    )
    args = parser.parse_args()

    findings = scan()
    baseline = {"count": 0, "owners": []}
    if BASELINE.exists():
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    limit = int(baseline.get("count", 0))

    print(f"🧭 Private copies of canonical state: {len(findings)} (baseline {limit})")

    if args.write_baseline:
        if len(findings) > limit and BASELINE.exists():
            print(
                f"❌ refusing to raise the baseline from {limit} to {len(findings)}. "
                "A baseline that goes up is debt laundered into a green gate."
            )
            return 1
        BASELINE.write_text(
            json.dumps(
                {
                    "description": (
                        "Subsystems holding a private copy of a canonical "
                        "variable. May only SHRINK. Each one is a second "
                        "answer to how Aura is, with nothing deciding between "
                        "it and the canonical value."
                    ),
                    "count": len(findings),
                    "owners": findings,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"✅ baseline written at {len(findings)}")
        return 0

    if len(findings) > limit:
        new = sorted(set(findings) - set(baseline.get("owners", [])))
        print(f"❌ {len(findings) - limit} new private copies:")
        for item in new[:20]:
            print(f"   • {item}")
        print(
            "\nEach is a subsystem keeping its own answer to how Aura is. "
            "Make it an estimator: contribute to the canonical channel and "
            "read the fused value back."
        )
        return 1

    print(f"✅ at or below baseline ({len(findings)} <= {limit})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
