"""core/connectome/anatomy_law.py — a change may not leave the body worse built.

``anatomy_gate`` can measure what a change did to the shape of the system.
Until now the promotion layer only WROTE that measurement into the receipt, so
the strongest thing the record could say about a change that made the anatomy
worse was that somebody had noticed. A measurement nothing refuses on is a
diagnosis, not a law.

This is the law. It is fail-closed in the direction that matters: a change that
degrades the anatomy past what its blast radius allows does not get promoted,
and ``promote`` raises rather than returning a receipt saying it did. A receipt
for a promotion that must not happen is worse than no receipt.

Two things it deliberately does not do.

It does not refuse every regression. A change that adds a module necessarily
adds coupling, and a law that treated that as a fault would stop development
outright. What it refuses is a regression larger than the change's own claim
size — the same ladder ``how_far_it_reaches`` already derives from Hoeffding,
so the tolerance is read off the part rather than picked. A change to a word may
cost a little shape; a change to the gate may cost none.

And it does not refuse everything that could not be measured. Refusing every
unmeasurable change would stop development on any faculty without a probe, which
is the argument the developmental layer already makes about held-out families.
Unmeasured is recorded and passes — except at the two tiers where being wrong is
expensive enough that "nobody looked" is not an acceptable answer. There the
absence of a measurement is itself the refusal, and the message says so.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.Connectome.AnatomyLaw")

__all__ = [
    "AnatomicalVerdict",
    "MUST_BE_MEASURED",
    "anatomy_permits",
    "tolerated_regression",
]


# The refusal is raised by ``core.cognition.how_a_change_is_promoted``, which
# owns the promotion. Defining a second exception of the same name here would
# give ``except AnatomyRefusedError`` two meanings and let one of them slip
# past a caller that imported the other.

#: Parts where "the anatomy could not be measured" is not an acceptable answer.
#: Both are places every later decision runs through, so a change made there
#: without anybody looking at what it did to the shape is a change nobody can
#: account for afterwards. Named by the same keys ``HOW_FAR`` uses.
MUST_BE_MEASURED: frozenset[str] = frozenset({"the search", "the deciding"})

#: An axis whose regression is counted in whole units rather than as a share.
#: One more single point of failure is one more, whatever the total was, and
#: dividing by a count that changed with the same edit measures the denominator.
_COUNTED_AXES: frozenset[str] = frozenset(
    {"single_points_of_failure", "half_wired_channels", "dormant_machinery"}
)


def tolerated_regression(at: str) -> float:
    """How far the shape may move the wrong way for a change to this part.

    Derived, not chosen. ``how_far_it_reaches`` returns the claim size a change
    to this part has to support — smaller means further-reaching — and the
    tolerance is that number scaled to the axis. A change to a word supports a
    claim of 0.5 and may cost half a unit of shape; a change to the gate
    supports 0.1 and may cost a tenth.
    """
    from core.cognition.how_a_change_is_promoted import how_far_it_reaches

    return max(0.0, float(how_far_it_reaches(at)))


@dataclass(frozen=True)
class AnatomicalVerdict:
    """Whether the shape of the system permits this change."""

    allowed: bool
    at: str
    reason: str
    tolerance: float
    measured: bool = True
    blocked_by: tuple[str, ...] = ()
    regressions: dict[str, float] = field(default_factory=dict)
    improved: tuple[str, ...] = ()
    unmeasured: tuple[str, ...] = ()

    def sentence(self) -> str:
        if self.allowed:
            return self.reason
        return f"refused at {self.at}: {self.reason}"

    def as_json(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "at": self.at,
            "reason": self.reason,
            "tolerance": round(self.tolerance, 4),
            "measured": self.measured,
            "blocked_by": list(self.blocked_by),
            "regressions": {k: round(v, 5) for k, v in sorted(self.regressions.items())},
            "improved": list(self.improved),
            "unmeasured": list(self.unmeasured),
        }


def _regression_size(name: str, move: Mapping[str, Any]) -> float:
    """How far this axis moved the wrong way, in units the axis is read in."""
    before = float(move.get("before", 0.0) or 0.0)
    after = float(move.get("after", 0.0) or 0.0)
    if name in _COUNTED_AXES:
        return abs(after - before)
    scale = max(abs(before), 1e-9)
    return abs(after - before) / scale


def anatomy_permits(delta: Any, *, at: str = "word") -> AnatomicalVerdict:
    """Does the shape of the system permit this change to be promoted?

    ``delta`` is a :class:`core.connectome.anatomy_gate.QualityDelta`, or None
    where nothing measured it.
    """
    tolerance = tolerated_regression(at)
    part = str(at).split("/", 1)[0]

    if delta is None:
        if part in MUST_BE_MEASURED:
            return AnatomicalVerdict(
                allowed=False,
                at=at,
                reason=(
                    "nothing measured what this did to the shape of the system, and "
                    f"every later decision runs through {part}. An unmeasured change "
                    "here is one nobody can account for afterwards"
                ),
                tolerance=tolerance,
                measured=False,
            )
        return AnatomicalVerdict(
            allowed=True,
            at=at,
            reason="the anatomy was not measured, which is recorded and is not a refusal",
            tolerance=tolerance,
            measured=False,
        )

    moves = getattr(delta, "moves", None)
    if not isinstance(moves, dict):
        return AnatomicalVerdict(
            allowed=True,
            at=at,
            reason="the anatomy reading could not be read, and is recorded as absent",
            tolerance=tolerance,
            measured=False,
        )

    regressions = {
        name: _regression_size(name, move)
        for name, move in moves.items()
        if isinstance(move, Mapping) and move.get("worsened")
    }
    blocked = tuple(
        sorted(name for name, size in regressions.items() if size > tolerance)
    )
    improved = tuple(getattr(delta, "improved", ()) or ())
    unmeasured = tuple(getattr(delta, "unmeasured", ()) or ())

    if blocked:
        worst = max(blocked, key=lambda name: regressions[name])
        return AnatomicalVerdict(
            allowed=False,
            at=at,
            reason=(
                f"{_axis_question(worst)} — it went the wrong way by "
                f"{regressions[worst]:.4g} against a tolerance of {tolerance:.4g} for a "
                f"change to {part}"
                + (
                    f", and {len(blocked) - 1} more axis did the same"
                    if len(blocked) == 2
                    else f", and {len(blocked) - 1} more axes did the same"
                    if len(blocked) > 2
                    else ""
                )
            ),
            tolerance=tolerance,
            blocked_by=blocked,
            regressions=regressions,
            improved=improved,
            unmeasured=unmeasured,
        )

    if regressions:
        return AnatomicalVerdict(
            allowed=True,
            at=at,
            reason=(
                f"the anatomy is worse on {len(regressions)} axis by less than a change "
                f"to {part} is allowed to cost"
                if len(regressions) == 1
                else f"the anatomy is worse on {len(regressions)} axes, each by less "
                f"than a change to {part} is allowed to cost"
            ),
            tolerance=tolerance,
            regressions=regressions,
            improved=improved,
            unmeasured=unmeasured,
        )

    return AnatomicalVerdict(
        allowed=True,
        at=at,
        reason=(
            f"the anatomy is better on {len(improved)} axes and worse on none"
            if improved
            else "the anatomy did not move on any axis that was measured"
        ),
        tolerance=tolerance,
        improved=improved,
        unmeasured=unmeasured,
    )


def _axis_question(name: str) -> str:
    """The axis's own question, so a refusal says what it is about."""
    from core.connectome.anatomy_gate import AXES

    for axis in AXES:
        if axis.name == name:
            return axis.question
    return name
