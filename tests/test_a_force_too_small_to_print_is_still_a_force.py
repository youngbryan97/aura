"""Six decimal places recorded a live channel as exactly zero.

`mot.forces` is the five quantities that moved her drive budgets this turn, and
the deliberation domain reads every one of them: they were added because a
displacement two turns long cannot show in the budgets themselves. Four are order
one. The fifth, what being met returned to a need, is about six millionths of the
gap in a turn — its rate is declared per day and a turn is a second of her life —
so rounding the dict to six places for readability recorded it as 0.0 on all
79,200 frames of the seed-7 run of 25 September while the mechanism was firing.

A number the instrument reads is not a number for display.
"""

from __future__ import annotations

import pytest

from core.motivation.constants import MOTIVATION_BUDGET_DEFAULTS
from core.phases.motivation_update import MotivationUpdatePhase


class _Motivation:
    def __init__(self, **levels) -> None:
        self.budgets = {
            name: dict(declared) for name, declared in MOTIVATION_BUDGET_DEFAULTS.items()
        }
        for name, level in levels.items():
            self.budgets[name]["level"] = level


def test_a_need_at_its_rest_returns_nothing():
    mot = _Motivation()
    assert MotivationUpdatePhase._warmth_returns_a_drive_to_rest(mot, 1.0) == 0.0


def test_a_need_above_its_rest_is_not_pushed_higher():
    rest = float(MOTIVATION_BUDGET_DEFAULTS["integrity"]["level"])
    mot = _Motivation(integrity=rest + 2.0, social=rest + 2.0)
    assert MotivationUpdatePhase._warmth_returns_a_drive_to_rest(mot, 1.0) == 0.0
    assert mot.budgets["integrity"]["level"] == pytest.approx(rest + 2.0)


def test_a_need_just_under_its_rest_returns_a_little_and_not_nothing():
    """The size the campaign actually sees: a hundredth of a unit of deficit."""
    rest = float(MOTIVATION_BUDGET_DEFAULTS["integrity"]["level"])
    mot = _Motivation(integrity=rest - 0.01, social=float(MOTIVATION_BUDGET_DEFAULTS["social"]["level"]))
    returned = MotivationUpdatePhase._warmth_returns_a_drive_to_rest(mot, 1.0)
    assert returned > 0.0
    assert returned < 1e-6
    assert round(returned, 6) == 0.0, "which is why the forces must not be rounded"


def test_no_time_returns_nothing():
    rest = float(MOTIVATION_BUDGET_DEFAULTS["integrity"]["level"])
    mot = _Motivation(integrity=rest - 1.0)
    assert MotivationUpdatePhase._warmth_returns_a_drive_to_rest(mot, 0.0) == 0.0
    assert MotivationUpdatePhase._warmth_returns_a_drive_to_rest(mot, "soon") == 0.0


def test_the_forces_keep_their_digits():
    """The deliberation domain reads this dict, so nothing in it is rounded."""
    import inspect

    from core.phases import motivation_update

    source = inspect.getsource(motivation_update)
    block = source[source.index("mot.forces = {") : source.index("mot.forces = {") + 400]
    assert "round(" not in block
    for name in ("pressure", "social_hold", "warmth_return", "attended_credit", "resolve_hold"):
        assert f'"{name}"' in block
