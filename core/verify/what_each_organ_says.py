"""What each organ owns, consumes, promises, and does when it fails.

The blind comparison ended on a recommendation that is not "simplify Aura":

    You do not necessarily need fewer organs. You need organs that know
    exactly what they own, what they consume, what they promise and how
    failure propagates.

Aura has all four answers and has them in four different places. DEPS says
what a package may consume. The state-ownership registry says what owns a
field. The promise suites say what a store, a graph, a provider and a tool
give back. ``record_degradation`` is how failure travels. Nothing put them
side by side, so "does this organ know what it is" had no answer.

This asks all four of every package under ``core``, and the gate is coverage
rather than quality: a package that answers none of the four is not an organ,
it is a folder. The count of those only goes down.

Deliberately mechanical. Every answer is read from the tree — a DEPS file, an
import, a call to ``record_degradation`` — so nothing here is a claim about an
organ that the organ does not itself make.
"""
from __future__ import annotations

import ast
import functools
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("Aura.WhatEachOrganSays")

__all__ = [
    "THE_FOUR_QUESTIONS",
    "how_the_organs_answer",
    "what_an_organ_says",
    "which_organs_answer_nothing",
]

BASELINE = Path(__file__).resolve().parents[2] / "config" / "organ_answers_baseline.json"

#: The four, in the words the review used.
THE_FOUR_QUESTIONS: tuple[str, ...] = (
    "what it owns",
    "what it consumes",
    "what it promises",
    "how failure propagates",
)

#: Words that mean a module is declaring a promise about its own behaviour.
#: A promise suite, a protocol, or a contract someone else can run.
_A_PROMISE = (
    "THE_PROMISES",
    "Protocol",
    "runtime_checkable",
    "_promises(",
    "result_schema",
    "@invariant",
)


def _packages(root: Path) -> list[Path]:
    """Every package under ``core``, including the namespace ones.

    Requiring ``__init__.py`` left 42 directories out of this audit, among
    them ``state``, ``kernel``, ``learning``, ``ethics``, ``health``,
    ``organism`` and ``sovereign``. So the number everyone was reading —
    32 of 120 — was computed over three quarters of the tree, and the
    packages missing from it were not a random three quarters: a namespace
    package is what a directory becomes when nobody wrote its ``__init__``,
    which correlates with nobody writing its DEPS or its promises either.

    A directory of importable modules is a package whether or not it has an
    ``__init__``, and an organ that cannot be seen by the audit is the one
    worth seeing.
    """
    return sorted(
        one
        for one in (root / "core").iterdir()
        if one.is_dir()
        and not one.name.startswith("__")
        and any(one.glob("*.py"))
    )


def what_an_organ_says(where: Path) -> dict[str, Any]:
    """The four answers for one package, read from its own files."""
    consumes: list[str] = []
    deps = where / "DEPS"
    if deps.exists():
        for line in deps.read_text("utf-8", errors="ignore").splitlines():
            stripped = line.strip().strip(",").strip('"')
            if stripped.startswith("+core.") and stripped != f"+{where.name}":
                consumes.append(stripped[1:])

    owns: list[str] = []
    promises: list[str] = []
    degrades: list[str] = []
    catches: list[str] = []
    for path in sorted(where.rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        try:
            text = path.read_text("utf-8", errors="ignore")
        except OSError:
            continue
        rel = str(path.relative_to(where.parent.parent))
        if "record_degradation(" in text:
            degrades.append(rel)
        try:
            caught = sum(
                1
                for node in ast.walk(ast.parse(text))
                if isinstance(node, ast.ExceptHandler)
            )
        except (SyntaxError, ValueError):
            caught = 0
        if caught:
            catches.append(rel)
        if any(one in text for one in _A_PROMISE):
            promises.append(rel)
        # Owning something means holding module-level state others read.
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            continue
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        owns.append(f"{rel}:{target.id}")

    return {
        "organ": where.name,
        "what it owns": sorted(owns)[:20],
        "owns": len(owns),
        "what it consumes": sorted(set(consumes)),
        "consumes": len(set(consumes)),
        "what it promises": sorted(set(promises))[:20],
        "promises": len(set(promises)),
        "how failure propagates": sorted(set(degrades))[:20],
        "degrades": len(set(degrades)),
        #: Files that catch an exception. A package that catches nothing
        #: propagates every failure to its caller, which is an answer to
        #: "how does failure propagate" — the strongest one there is.
        "catches": len(set(catches)),
        "catches_and_records_nothing": bool(catches) and not degrades,
        "declares_its_edges": deps.exists(),
    }


@functools.lru_cache(maxsize=2)
def _all_organs(root: Path) -> tuple[tuple[str, Any], ...]:
    answered: dict[str, Any] = {}
    for where in _packages(root):
        answered[where.name] = what_an_organ_says(where)
    return tuple(answered.items())


def how_the_organs_answer(root: Path | None = None) -> dict[str, Any]:
    """Every organ, and how many of the four it answers."""
    here = root or Path(__file__).resolve().parents[2]
    organs = dict(_all_organs(here))

    # "Nothing" is an answer. A package that consumes no other package has
    # answered what it consumes, and scoring that as a gap would push every
    # leaf towards importing something.
    def answered(said: dict[str, Any]) -> set[str]:
        given = set()
        if said["owns"] > 0:
            given.add("what it owns")
        if said["declares_its_edges"]:
            given.add("what it consumes")
        if said["promises"] > 0:
            given.add("what it promises")
        # Two ways to answer this, and only one of them was being counted.
        # A package that records degradations says where failure goes. A
        # package that catches nothing says it too: every failure reaches the
        # caller, which is the strongest answer available and was scoring as
        # silence. Twelve packages with no `except` anywhere were being
        # counted as not knowing how their own failures travel.
        if said["degrades"] > 0 or said["catches"] == 0:
            given.add("how failure propagates")
        return given

    scored = {name: answered(said) for name, said in organs.items()}
    silent = sorted(name for name, given in scored.items() if not given)
    missing: dict[str, list[str]] = {}
    for question in THE_FOUR_QUESTIONS:
        missing[question] = sorted(
            name for name, given in scored.items() if question not in given
        )
    return {
        "organs": len(organs),
        "answer_all_four": sum(1 for given in scored.values() if len(given) == 4),
        "answer_some": sum(1 for given in scored.values() if 0 < len(given) < 4),
        "answer_nothing": len(silent),
        "silent": silent,
        "who_does_not_say": {
            question: len(names) for question, names in missing.items()
        },
        "say_nothing_about_what_they_promise": missing["what it promises"][:40],
        "say_nothing_about_failure": missing["how failure propagates"][:40],
        #: The dangerous middle: it catches exceptions and records none of
        #: them, so a failure is neither raised nor written down anywhere.
        "catches_and_records_nothing": sorted(
            name
            for name, said in organs.items()
            if said.get("catches_and_records_nothing")
        ),
        "without_a_deps_file": sorted(
            name for name, said in organs.items() if not said["declares_its_edges"]
        ),
        "the_four": list(THE_FOUR_QUESTIONS),
    }


def which_organs_answer_nothing(root: Path | None = None) -> list[str]:
    """Packages that answer none of the four. A folder, not an organ."""
    return list(how_the_organs_answer(root)["silent"])


def the_baseline() -> dict[str, Any]:
    try:
        return json.loads(BASELINE.read_text("utf-8"))
    except (OSError, ValueError) as exc:
        logger.debug("no organ-answers baseline: %s", exc)
        return {}
