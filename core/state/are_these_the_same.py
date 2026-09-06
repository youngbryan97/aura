"""Whether two snapshots mean the same thing, and where they do not.

OpenCog makes content comparison a core operation, and the closure asked for
``semantic_equivalence(A, B, tolerances)`` used in restore, compaction,
migration and shadow tests. Aura compared snapshots by digest, which answers a
different question: a digest says the bytes differ, and every one of those
four uses needs to know whether the MEANING differs.

The difference is not pedantry. A restore that rewrote a float's last bit, a
compaction that dropped a field nothing reads, a migration that renamed a key
— a digest calls all three a failure, and a reviewer then learns to ignore it.

So: a comparison that carries its tolerances, and that says WHERE two things
differ rather than only that they do. A verdict with no location cannot be
acted on, which is why the four uses had been reaching for the digest.

The tolerances are declared, not inferred. Guessing that two floats are close
enough is how a comparison quietly stops comparing.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.AreTheseTheSame")

__all__ = [
    "ADifference",
    "HowClose",
    "are_these_the_same",
]


@dataclass(frozen=True)
class HowClose:
    """What counts as the same. Every allowance is declared here, not guessed."""

    #: Absolute difference two floats may have and still be the same.
    floats_within: float = 0.0
    #: Keys whose value never matters. Timestamps, run ids, host names.
    ignore: frozenset[str] = frozenset()
    #: Keys whose lists mean a set rather than a sequence.
    order_does_not_matter: frozenset[str] = frozenset()
    #: Whether a key present on one side and absent on the other is a
    #: difference. False for a comparison across a migration that adds fields.
    missing_is_a_difference: bool = True


@dataclass(frozen=True)
class ADifference:
    """One place two snapshots disagree, addressed so it can be acted on."""

    where: str
    left: Any
    right: Any
    why: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "where": self.where,
            "left": self.left,
            "right": self.right,
            "why": self.why,
        }


@dataclass
class TheSameOrNot:
    """The verdict, and every place it rests on."""

    same: bool
    differences: list[ADifference] = field(default_factory=list)
    compared: int = 0

    def __bool__(self) -> bool:
        return self.same

    def to_dict(self) -> dict[str, Any]:
        return {
            "same": self.same,
            "compared": self.compared,
            "differences": [one.to_dict() for one in self.differences],
        }


def are_these_the_same(
    left: Any, right: Any, *, how_close: HowClose | None = None
) -> TheSameOrNot:
    """Whether these two mean the same thing, and where they do not.

    Walks both, so a key present only on the right is found too — a
    comparison that walks one side reports a truncated snapshot as identical.
    """
    allowance = how_close or HowClose()
    found: list[ADifference] = []
    counted = _walk("", left, right, allowance, found)
    return TheSameOrNot(same=not found, differences=found, compared=counted)


def _walk(
    where: str, left: Any, right: Any, how: HowClose, found: list[ADifference]
) -> int:
    """Compare one place. Returns how many leaves were looked at."""
    name = where.rsplit(".", 1)[-1].split("[", 1)[0]
    if name and name in how.ignore:
        return 0

    if isinstance(left, dict) and isinstance(right, dict):
        counted = 0
        for key in sorted(set(left) | set(right)):
            here = f"{where}.{key}" if where else key
            if key in how.ignore:
                continue
            if key not in left or key not in right:
                if how.missing_is_a_difference:
                    found.append(
                        ADifference(
                            where=here,
                            left=left.get(key),
                            right=right.get(key),
                            why="on one side only",
                        )
                    )
                continue
            counted += _walk(here, left[key], right[key], how, found)
        return counted

    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if name in how.order_does_not_matter:
            return _as_sets(where, left, right, how, found)
        if len(left) != len(right):
            found.append(
                ADifference(
                    where=where,
                    left=len(left),
                    right=len(right),
                    why="different lengths",
                )
            )
            return 1
        counted = 0
        for at, (one, other) in enumerate(zip(left, right, strict=True)):
            counted += _walk(f"{where}[{at}]", one, other, how, found)
        return counted

    return _a_leaf(where, left, right, how, found)


def _as_sets(
    where: str, left: Any, right: Any, how: HowClose, found: list[ADifference]
) -> int:
    """Compare two lists whose order was declared not to matter."""
    try:
        only_left = sorted(set(map(repr, left)) - set(map(repr, right)))
        only_right = sorted(set(map(repr, right)) - set(map(repr, left)))
    except TypeError:
        return _a_leaf(where, left, right, how, found)
    if only_left or only_right:
        found.append(
            ADifference(
                where=where,
                left=only_left[:8],
                right=only_right[:8],
                why="different members, order aside",
            )
        )
    return 1


def _a_leaf(
    where: str, left: Any, right: Any, how: HowClose, found: list[ADifference]
) -> int:
    if isinstance(left, bool) != isinstance(right, bool):
        found.append(ADifference(where, left, right, "one is a boolean"))
        return 1
    if (
        isinstance(left, (int, float))
        and isinstance(right, (int, float))
        and not isinstance(left, bool)
    ):
        if math.isnan(left) and math.isnan(right):
            return 1
        if abs(float(left) - float(right)) > how.floats_within:
            found.append(
                ADifference(
                    where=where,
                    left=left,
                    right=right,
                    why=(
                        f"differ by {abs(float(left) - float(right)):g}, "
                        f"which is more than {how.floats_within:g}"
                    ),
                )
            )
        return 1
    if left != right:
        found.append(ADifference(where, left, right, "different values"))
    return 1
