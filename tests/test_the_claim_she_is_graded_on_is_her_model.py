"""A move's claim should be the one whose failure means her model is wrong.

LIVE 2026-09-01, playing a real 2048 board, every move came back:

  left: predicted 'the view to be different after left', held=True, closer=False

"The view will be different" is satisfied by almost any keystroke on almost any
screen, and the length of her plans, the part of the screen she believes answers
to her, and what her moves are worth are all read off that verdict.

She had a better claim available and was not making it. The rule she works out
by watching says what the arrangement becomes, exactly — a claim that can be
wrong, which is the only kind worth checking.
"""

from __future__ import annotations

import pytest

from core.agency.deliberate_action import Expectation

pytestmark = pytest.mark.unit


def test_a_bare_change_claim_says_nothing() -> None:
    assert Expectation(changed=True).says_something() is False


def test_a_foretold_arrangement_always_says_something() -> None:
    assert Expectation(becomes=object()).says_something() is True


def _Reading(*values: str):
    """A one-row arrangement, which is all this needs to be checked."""

    from core.perception.what_is_there import Arrangement, Cell

    return Arrangement(
        rows=1,
        columns=len(values),
        cells=tuple(
            Cell(row=0, column=index, says=value, at=(float(index), 0.0))
            for index, value in enumerate(values)
        ),
    )


def test_the_verdict_fails_when_what_turned_up_is_not_what_was_foretold() -> None:
    claim = Expectation(
        changed=True,
        becomes=_Reading("2", "4", "8"),
        becomes_holds=lambda after: after.as_text() == _Reading("2", "4", "8").as_text(),
    )
    verdict = claim.check_in(_Reading("2", "2", "4"), _Reading("4", "8", "16"))
    assert verdict.held is False
    assert verdict.observed_change is True
    assert any("foretold" in why for why in verdict.missing + verdict.lingering)


def test_the_verdict_holds_when_it_is() -> None:
    claim = Expectation(
        changed=True,
        becomes=_Reading("2", "4", "8"),
        becomes_holds=lambda after: after.as_text() == _Reading("2", "4", "8").as_text(),
    )
    assert claim.check_in(_Reading("2", "2", "4"), _Reading("2", "4", "8")).held is True


def test_a_checker_that_raises_does_not_call_her_wrong() -> None:
    """A broken comparison is not evidence against her model."""

    def explodes(_after: object) -> bool:
        raise TypeError("no")

    claim = Expectation(changed=True, becomes=_Reading("x"), becomes_holds=explodes)
    assert claim.check_in(_Reading("a"), _Reading("b")).held is True


def test_the_pursuit_loop_attaches_the_rule_to_the_claim() -> None:
    from pathlib import Path

    source = Path("core/skills/screen_pursuit.py").read_text(encoding="utf-8")
    assert "knows.rules.expect(laid_out, key)" in source
    assert "becomes_holds=" in source
    # Scored the way the rules themselves are, so a dealt tile is not her error.
    assert "prediction_held" in source


def test_the_comparison_is_the_one_the_rules_are_scored_by() -> None:
    """A rule discredited here and credited there is two models in one name."""

    from core.perception.how_it_moves import prediction_held

    assert prediction_held(None, None) is False
