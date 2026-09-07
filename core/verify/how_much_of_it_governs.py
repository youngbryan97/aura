"""How much of the tree decides anything, counted rather than assumed.

``does_this_govern_anything`` answers the question for one module at a time,
against a curated list. The list is the problem: it holds the primitives
somebody thought to name, so the report says what fraction of the *audited*
abstractions govern and reads as though it were the fraction of the system.
The review's line was to stop counting implemented abstractions and count the
ones that govern, and that has to mean all of them.

So this walks every module under the production roots and puts each in one of
five states. Two of them are the same reading as before — production code
imports it, or only a tool and a report do. The third is new and it fixes a
lie the simple version would tell: a module can govern without any import
naming it, because something loads it by string at runtime. A skill is the
plain case. There are 140 files under ``core/skills`` and a static scan finds
almost no importer for any of them, because the catalog builds itself from the
source tree and the engine imports each one by its module path when the skill
is called. Calling those proposals would be the exact error this file exists
to catch, told about the wrong things.

Each way in is asked for its own members rather than listed by hand: the
catalog parser names the skills it finds, and the tree names the modules some
production string names. What no way in reaches is a proposal, and that count
is the one worth watching.
"""
from __future__ import annotations

import ast
import functools
import logging
import pathlib
from collections import Counter
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("Aura.HowMuchOfItGoverns")

__all__ = [
    "ACensus",
    "THE_PRODUCTION_ROOTS",
    "how_much_of_it_governs",
    "what_no_way_in_reaches",
    "reached_by_name",
]

ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Where a module has to live to be part of the organism at all.
THE_PRODUCTION_ROOTS: tuple[str, ...] = (
    "core",
    "interface",
    "skills",
    "llm",
    "executors",
    "security",
)

GOVERNING = "governing"
GOVERNING_A_PROPOSAL = "governing a proposal"
REACHABLE = "reachable"
BY_NAME = "reached by name"
A_PROPOSAL = "a proposal"


@dataclass(frozen=True, slots=True)
class ACensus:
    """What the whole tree decides, by state."""

    modules: int
    by_state: dict[str, int]
    proposals: tuple[str, ...]
    by_package: dict[str, int]
    how_each_is_reached_by_name: dict[str, int]

    @property
    def governing(self) -> int:
        return self.by_state.get(GOVERNING, 0)

    @property
    def share_that_governs(self) -> float:
        return self.governing / self.modules if self.modules else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "modules": self.modules,
            "by_state": dict(self.by_state),
            "share_that_governs": round(self.share_that_governs, 4),
            "proposals": len(self.proposals),
            "proposals_by_package": dict(self.by_package),
            "reached_by_name": dict(self.how_each_is_reached_by_name),
        }


def _every_module(root: str = "") -> tuple[str, ...]:
    base = pathlib.Path(root or ROOT)
    found: list[str] = []
    for top in THE_PRODUCTION_ROOTS:
        where = base / top
        if not where.exists():
            continue
        for path in sorted(where.rglob("*.py")):
            if path.name == "__init__.py":
                continue
            found.append(str(path.relative_to(base).with_suffix("")).replace("/", "."))
    return tuple(found)


@functools.lru_cache(maxsize=4)
def _named_in_a_string(root: str = "") -> frozenset[str]:
    """Module paths that appear as a string constant in production code.

    ``importlib.import_module("core.self_improvement.program_dna")`` is an
    edge in the call graph that no import statement carries, and a census that
    cannot see it reports a live module as dead.
    """
    base = pathlib.Path(root or ROOT)
    found: set[str] = set()
    for top in THE_PRODUCTION_ROOTS:
        where = base / top
        if not where.exists():
            continue
        for path in where.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
            except (OSError, SyntaxError, ValueError):
                continue
            me = str(path.relative_to(base).with_suffix("")).replace("/", ".")
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant):
                    continue
                value = node.value
                if not isinstance(value, str) or " " in value or "." not in value:
                    continue
                if value.split(".")[0] in THE_PRODUCTION_ROOTS and value != me:
                    found.add(value)
    return frozenset(found)


@functools.lru_cache(maxsize=4)
def _in_the_skill_catalog(root: str = "") -> frozenset[str]:
    """The modules the skill catalog names, asked of the catalog itself.

    Only the real tree has one. A census run over a fixture answers "no skill
    catalog reaches anything here", which is true of the fixture.
    """
    if root and pathlib.Path(root) != ROOT:
        return frozenset()
    try:
        from core.skills.discovery import default_skill_roots, parse_skill_sources

        declarations, _issues, _files = parse_skill_sources(default_skill_roots())
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("the skill catalog could not be read: %s", exc)
        return frozenset()
    return frozenset(
        str(one.module_path) for one in declarations if getattr(one, "module_path", "")
    )


def reached_by_name(module: str, root: str = "") -> str:
    """Which runtime way in reaches this module, or "" if none does."""
    if module in _in_the_skill_catalog(root):
        return "the skill catalog"
    if module in _named_in_a_string(root):
        return "a string in production code"
    return ""


@functools.lru_cache(maxsize=4)
def how_much_of_it_governs(root: str = "") -> ACensus:
    """Every module under the production roots, by what it decides.

    Two full parses of the tree, so about a minute on this repo. It belongs in
    an inspector run, never in a health block something polls.
    """
    from core.verify.does_this_govern_anything import (
        HowFarItReaches,
        how_far_it_reaches,
    )

    states: Counter[str] = Counter()
    ways: Counter[str] = Counter()
    packages: Counter[str] = Counter()
    proposals: list[str] = []
    modules = _every_module(root)
    for module in modules:
        reach = how_far_it_reaches(module, root)
        if reach.reaches is not HowFarItReaches.A_PROPOSAL:
            states[str(reach.reaches)] += 1
            continue
        way = reached_by_name(module, root)
        if way:
            states[BY_NAME] += 1
            ways[way] += 1
            continue
        states[A_PROPOSAL] += 1
        proposals.append(module)
        parts = module.split(".")
        packages[".".join(parts[:-1]) or parts[0]] += 1

    return ACensus(
        modules=len(modules),
        by_state=dict(sorted(states.items())),
        proposals=tuple(proposals),
        by_package=dict(packages.most_common()),
        how_each_is_reached_by_name=dict(sorted(ways.items())),
    )


def what_no_way_in_reaches(root: str = "") -> tuple[str, ...]:
    """The modules nothing imports, nothing names, and no catalog carries."""
    return how_much_of_it_governs(root).proposals
