"""A decision point that only ever answers one way is not deciding anything.

Two failures, and they are the same defect seen from opposite sides.

A gate nobody can pass takes a working system and gates it into a coma. Live on
2026-09-07: `/api/readyz` answered 503 for the whole of every turn while the
runtime answered perfectly, and anything watching readiness concluded it was
unhealthy whenever somebody was talking to it. The prompt cache reported a miss
on every single turn for the life of the process. The foreground
non-parametric memory refused every turn — "371 entries is too sparse to steer
a 5,120-wide space, need 50,000" — a threshold nothing on this host will ever
reach.

A gate nobody can fail is the other half. It costs nothing visible, which is
why it survives, and it means the check is decorative.

Neither is findable in the syntax tree: the code branches both ways and only
the traffic says which branch is real. So it is counted, and the count is
reported. It is never enforced — a runtime that refuses to start because a
counter looks lopsided is the coma, arriving by a different route.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from core.runtime.lockdep import checked_lock

#: Below this a run of one answer is ordinary. Ten decisions all going the same
#: way is a Tuesday; a hundred is a mechanism.
ENOUGH_TO_BE_A_PATTERN = 40

_lock = checked_lock("core.verify.one_way_decisions.lock")
_admitted: dict[str, int] = {}
_refused: dict[str, int] = {}
_reasons: dict[str, str] = {}


@dataclass(frozen=True)
class ADecisionThatOnlyGoesOneWay:
    name: str
    admitted: int
    refused: int
    last_reason: str

    @property
    def verdict(self) -> str:
        return "never admits" if self.admitted == 0 else "never refuses"

    @property
    def total(self) -> int:
        return self.admitted + self.refused


def record_decision(name: str, *, admitted: bool, reason: str = "") -> None:
    """Count one decision by name. Cheap enough for a hot path."""

    key = str(name or "").strip()
    if not key:
        return
    with _lock:
        table = _admitted if admitted else _refused
        table[key] = table.get(key, 0) + 1
        if not admitted and reason:
            _reasons[key] = str(reason)[:200]


def one_way_decisions(
    minimum: int = ENOUGH_TO_BE_A_PATTERN,
) -> tuple[ADecisionThatOnlyGoesOneWay, ...]:
    """Every counted decision that has never once gone the other way."""

    with _lock:
        names = set(_admitted) | set(_refused)
        found = []
        for name in sorted(names):
            admitted = _admitted.get(name, 0)
            refused = _refused.get(name, 0)
            if admitted + refused < minimum:
                continue
            if admitted and refused:
                continue
            found.append(
                ADecisionThatOnlyGoesOneWay(
                    name=name,
                    admitted=admitted,
                    refused=refused,
                    last_reason=_reasons.get(name, ""),
                )
            )
    return tuple(found)


def decision_census() -> dict[str, dict[str, object]]:
    """Every counted decision and how it has gone, for the health surface."""

    with _lock:
        names = sorted(set(_admitted) | set(_refused))
        return {
            name: {
                "admitted": _admitted.get(name, 0),
                "refused": _refused.get(name, 0),
                "last_refusal_reason": _reasons.get(name, ""),
            }
            for name in names
        }


def reset_decisions() -> None:
    """Only for tests. A running system never forgets what it decided."""

    with _lock:
        _admitted.clear()
        _refused.clear()
        _reasons.clear()
