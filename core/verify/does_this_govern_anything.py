"""A primitive with no caller outside its own tests is a proposal.

The maturity review's sharpest point was not about any one module. It was that
a beautiful 150-line module reads as though a system-wide invariant now exists,
and it does not — so the question worth asking of every abstraction is not
whether it is implemented correctly but whether it governs the organism.

That question has a mechanical form. For each primitive: who imports it, and
are any of them production code rather than its own test?

Three states, and the middle one is the one that matters:

* **governing** — production modules import it, so the behaviour it defines is
  the behaviour the runtime has.
* **reachable** — something can call it: it is exported, wired into a report or
  a tool, and a caller who knew about it would find it. Nothing does yet.
* **a proposal** — only its tests import it. It is correct and it decides
  nothing.

A proposal is not a failure. It is the honest state of something built before
the call sites that need it, and the number that matters is how long it stays
one. What would be dishonest is a report that counts it as an invariant.
"""
from __future__ import annotations

import ast
import functools
import logging
import pathlib
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.DoesThisGovernAnything")

__all__ = [
    "HowFarItReaches",
    "APrimitive",
    "who_imports",
    "how_far_it_reaches",
    "what_governs_and_what_does_not",
]

ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Where production code lives. Anything else importing a module is a caller
#: that does not make it govern anything: a test, a tool, a script.
_PRODUCTION = ("core", "interface", "skills", "llm", "executors", "security")

#: Places that make a module reachable without making it govern: a report a
#: person reads, an inspector, a registry of names.
_REACHABLE_FROM = ("tools", "docs")


class HowFarItReaches(StrEnum):
    """What a module actually decides."""

    GOVERNING = "governing"
    #: Imported only by modules that are themselves proposals. A chain of
    #: primitives calling each other is still a chain nothing enters, and
    #: counting the inner ones as governing is how a report flatters itself.
    GOVERNING_A_PROPOSAL = "governing a proposal"
    REACHABLE = "reachable"
    A_PROPOSAL = "a proposal"


@dataclass(frozen=True, slots=True)
class APrimitive:
    """One module, and who calls it."""

    module: str
    reaches: HowFarItReaches
    production_callers: tuple[str, ...]
    other_callers: tuple[str, ...]
    test_callers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "reaches": str(self.reaches),
            "production_callers": list(self.production_callers),
            "other_callers": list(self.other_callers),
            "tests": len(self.test_callers),
        }


def _dotted(path: pathlib.Path, root: pathlib.Path) -> str:
    return str(path.relative_to(root).with_suffix("")).replace("/", ".")


@functools.lru_cache(maxsize=4)
def _every_import(root: str = "") -> dict[str, tuple[str, ...]]:
    """module -> everything that imports it, anywhere in the tree."""
    base = pathlib.Path(root or ROOT)
    imports: dict[str, set[str]] = {}
    for where in (*_PRODUCTION, *_REACHABLE_FROM, "tests"):
        top = base / where
        if not top.exists():
            continue
        for path in top.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
            except (SyntaxError, ValueError, OSError):
                continue
            me = _dotted(path, base)
            for node in ast.walk(tree):
                named: list[str] = []
                if isinstance(node, ast.Import):
                    named = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    named = [node.module] + [
                        f"{node.module}.{a.name}" for a in node.names
                    ]
                for one in named:
                    imports.setdefault(one, set()).add(me)
    return {name: tuple(sorted(who)) for name, who in imports.items()}


def who_imports(module: str, root: str = "") -> tuple[str, ...]:
    """Everything that imports this module, by dotted name."""
    return _every_import(root).get(module, ())


def how_far_it_reaches(module: str, root: str = "") -> APrimitive:
    """Whether this module governs anything, is merely reachable, or is a proposal."""
    callers = who_imports(module, root)
    itself = module.rsplit(".", 1)[-1]
    production = tuple(
        one
        for one in callers
        if one.split(".")[0] in _PRODUCTION and one != module
    )
    other = tuple(one for one in callers if one.split(".")[0] in _REACHABLE_FROM)
    tests = tuple(one for one in callers if one.split(".")[0] == "tests")
    if production:
        reaches = HowFarItReaches.GOVERNING
    elif other:
        reaches = HowFarItReaches.REACHABLE
    else:
        reaches = HowFarItReaches.A_PROPOSAL
    logger.debug("%s: %s (%d production callers)", itself, reaches, len(production))
    return APrimitive(
        module=module,
        reaches=reaches,
        production_callers=production,
        other_callers=other,
        test_callers=tests,
    )


def what_governs_and_what_does_not(
    modules: tuple[str, ...], root: str = ""
) -> dict[str, Any]:
    """For the health report: which primitives decide anything, and which wait."""
    found = [how_far_it_reaches(one, root) for one in modules]
    # A module whose only production callers are themselves proposals is not
    # governing anything; it is the middle of a chain nothing enters.
    proposals = {
        one.module for one in found if one.reaches is HowFarItReaches.A_PROPOSAL
    }
    found = [
        one
        if not (
            one.reaches is HowFarItReaches.GOVERNING
            and set(one.production_callers) <= proposals
        )
        else APrimitive(
            module=one.module,
            reaches=HowFarItReaches.GOVERNING_A_PROPOSAL,
            production_callers=one.production_callers,
            other_callers=one.other_callers,
            test_callers=one.test_callers,
        )
        for one in found
    ]
    by_state: dict[str, list[str]] = {}
    for one in found:
        by_state.setdefault(str(one.reaches), []).append(one.module)
    return {
        "asked_about": len(found),
        "governing": len(by_state.get("governing", [])),
        "governing_a_proposal": len(by_state.get("governing a proposal", [])),
        "reachable": len(by_state.get("reachable", [])),
        "proposals": len(by_state.get("a proposal", [])),
        "which": {name: sorted(rows) for name, rows in sorted(by_state.items())},
        "each": [one.to_dict() for one in found],
    }
