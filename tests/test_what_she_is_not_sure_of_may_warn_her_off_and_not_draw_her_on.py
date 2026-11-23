"""What she is not sure of may warn her off a move and may not draw her to one.

A prediction her model holds on little, that an act leads somewhere better, raises the act's worth only as far as she
is sure of it; one that it leads somewhere worse counts in full. An option a doubtful prediction makes look good is the
option most likely to be a misreading (suggestions of 23 November 2026, item 22).
"""
from __future__ import annotations

from dataclasses import dataclass

from core.agency.looking_ahead import _as_far_as_sure, look_ahead


@dataclass(frozen=True)
class _APlace:
    at: int

    def as_text(self) -> str:
        return str(self.at)

    def numbers(self) -> tuple[float, ...]:
        return (2.0 ** self.at,)

    def newness(self) -> float:
        return 1.0

    def empty(self) -> int:
        return 2

    def places(self) -> int:
        return 4


class _AWorldHalfKnown:
    """"leap" is predicted to go three places on, on a reading seen once; "step" one place on, seen often."""

    def __init__(self, leap_sure: float) -> None:
        self.leap_sure = leap_sure

    def expect(self, state: _APlace, act: str) -> _APlace | None:
        return _APlace(state.at + {"leap": 3, "step": 1}[act]) if act in ("leap", "step") else None

    def confidence(self) -> float:
        return 1.0

    def how_sure(self, state: _APlace, act: str) -> float:
        return self.leap_sure if act == "leap" else 1.0


def test_a_move_a_doubtful_prediction_makes_look_best_is_not_taken_over_one_she_is_sure_of() -> None:
    here = _APlace(at=2)
    sure = look_ahead(_AWorldHalfKnown(leap_sure=1.0), here, ["leap", "step"], toward="256", budget_s=0.05, depth=1)
    assert sure["leap"][0] > sure["step"][0]
    doubtful = look_ahead(_AWorldHalfKnown(leap_sure=0.05), here, ["leap", "step"], toward="256", budget_s=0.05, depth=1)
    assert doubtful["step"][0] > doubtful["leap"][0]


def test_a_doubtful_prediction_of_worse_counts_in_full() -> None:
    assert _as_far_as_sure(-5.0, 1.0, 0.1) == -5.0
    assert _as_far_as_sure(11.0, 1.0, 0.1) == 2.0


def test_her_learned_rules_say_how_sure_each_prediction_is() -> None:
    from core.perception.how_it_moves import HowItMoves

    rules = HowItMoves()
    rules.confidence = lambda: 0.8  # type: ignore[method-assign]
    rules.pushes = {"press up": {"up": 9}, "press left": {"left": 1}}
    assert rules.how_sure(None, "press up") > rules.how_sure(None, "press left") > rules.how_sure(None, "press z")
    assert abs(rules.how_sure(None, "press z") - 0.4) < 1e-9
