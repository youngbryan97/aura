"""Which version of a library to keep, and which of its tests to doubt, by what versions written apart agree on.

Her model writes a library's tests as well as the library, so a test can be wrong as easily as the code: an
expectation worked out in its head against what was read. CodeT (Chen et al., ICLR 2023) wrote several programs
and several tests and kept the program whose behaviour the most programs share, scored by how many tests that
behaviour passes. Here:

- versions that give the same value on every test form a group, which scores its size times the tests it passes
  that are not doubted, and the kept version is the first of the best group;
- a test is doubted where no version passes it, two or more versions give one same value that is not an error, and
  her model, asked for that one call's value from what was read and shown no code, works out that value and not
  the one expected. Versions of one model at a low temperature share their mistakes, so their agreement alone is
  not enough; the third witness is a separate line of reasoning.

A doubted test is reported. It is never counted as held; it is not shown back as a failure to write toward, and it
does not count against its function.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from core.rebuilding.her_libraries import LibraryPlan, LibraryTest, RunOfATest

__all__ = ["Version", "agreed_value", "doubted_tests", "the_best_of", "same_value"]

#: What a run gives where the call gave no value: a throw, a time-out, a library that would not load.
_NOT_A_VALUE = ("threw ", "ran past ", "the library did not load", "undefined")


class _Worked(BaseModel):
    value: str = Field(default="", max_length=2000, description="the value the call gives, as JSON")
    why: str = Field(default="", max_length=600)


@dataclass
class Version:
    code: str
    runs: list[RunOfATest] = field(default_factory=list)

    @property
    def loaded(self) -> bool:
        return bool(self.runs) and not all(r.got.startswith("the library did not load") for r in self.runs)

    def passes(self, doubted: Sequence[str] = ()) -> list[str]:
        return [r.test.call for r in self.runs if r.held and r.test.call not in doubted]

    @property
    def behaviour(self) -> tuple[str, ...]:
        return tuple(r.got for r in self.runs)


def same_value(a: Any, b: Any) -> bool:
    """JSON values equal, numbers to a part in a billion, as the page compares them."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, int | float) and isinstance(b, int | float):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same_value(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_value(a[k], b[k]) for k in a)
    return a == b


def _value(text: str) -> tuple[bool, Any]:
    try:
        return True, json.loads(text)
    except ValueError:
        return False, None


def agreed_value(versions: Sequence[Version], index: int) -> str | None:
    """The value two or more loaded versions give for test ``index`` where none passes it, if there is one."""
    runs = [v.runs[index] for v in versions if v.loaded and len(v.runs) > index]
    if not runs or any(r.held for r in runs):
        return None
    given = Counter(r.got for r in runs if r.got and not r.got.startswith(_NOT_A_VALUE) and _value(r.got)[0])
    if not given:
        return None
    value, count = given.most_common(1)[0]
    return value if count >= 2 else None


def _work_it_out_prompt(plan: LibraryPlan, test: LibraryTest, read: str) -> str:
    return (f"The JavaScript library {plan.name}: {plan.purpose}\nIts functions:\n{plan.said()}\n"
            + (f"What was read of what it rests on:\n{read}\n" if read else "")
            + f"Work out, step by step from what was read, the value of this call, as JSON:\n{test.call}")


async def doubted_tests(plan: LibraryPlan, versions: Sequence[Version], ask: Any, *, known: Any = None) -> list[str]:
    """The tests whose expectation is the likelier mistake: no version passes, versions agree on another value, and
    her model, working the call out apart from any code, reaches that value."""
    reads = known.of(" ".join([plan.purpose, *plan.rests_on])) if known is not None else []
    read = "".join(f"- {r.text} ({r.where})\n" for r in reads)
    doubted = []
    for index, test in enumerate(plan.tests):
        agreed = agreed_value(versions, index)
        if agreed is None:
            continue
        worked = await ask(_work_it_out_prompt(plan, test, read), _Worked, 1024)
        ok, value = _value(str(getattr(worked, "value", "") or ""))
        _, expected = _value(test.expect)
        if ok and same_value(value, _value(agreed)[1]) and not same_value(value, expected):
            doubted.append(test.call)
    return doubted


def the_best_of(versions: Sequence[Version], doubted: Sequence[str] = ()) -> Version | None:
    """The first version of the group that behaves alike and scores most: its size times what it passes."""
    groups: dict[tuple[str, ...], list[Version]] = {}
    for version in versions:
        if version.loaded:
            groups.setdefault(version.behaviour, []).append(version)
    if not groups:
        return versions[0] if versions else None
    return max(groups.values(), key=lambda group: len(group) * len(group[0].passes(doubted)))[0]
