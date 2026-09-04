"""She can write a developmental operator for any place a term can go.

``WHERE_A_TERM_CAN_GO`` names seven destinations. ``_WHERE_IT_GOES`` held
four installers, so an action SHE invented could reach four of the places a
hand-written one could reach, and asking for one of the other three raised
ValueError before anything ran. That is the gap between "she can write a
developmental operator" and "she can write one that goes where the operator
needs to go" — the arrow an external review named as missing from the loop.

The second thing held here is that a term handed in as surface syntax is
compiled. It was not: ``run`` refused it, the refusal was caught as "gave
nothing", and the action returned None forever. An operator that cannot fire
and says nothing about why is the worst of the three outcomes, because it
looks exactly like an operator that decided not to.
"""

from __future__ import annotations

import pytest

from core.cognition.the_floor_she_stands_on import L, QUOTE, V, build
from core.cognition.what_she_can_take_back import as_it_stands
from core.cognition.what_she_could_do_next import (
    WHAT_SHE_COULD_DO,
    WHERE_A_TERM_CAN_GO,
    _WHERE_IT_GOES,
    the_action_she_wrote,
    the_actions_she_has,
)


@pytest.fixture(autouse=True)
def _leave_it_as_found():
    was = as_it_stands()
    held = dict(WHAT_SHE_COULD_DO)
    yield
    was.restore()
    WHAT_SHE_COULD_DO.clear()
    WHAT_SHE_COULD_DO.update(held)


def _identity():
    """A term that gives back what it is given, quoted so it arrives as a term."""
    return QUOTE(build(L("x", V("x"))))


def test_every_destination_has_an_installer():
    """Three of seven had none, so three of seven raised before running."""
    missing = [one for one in WHERE_A_TERM_CAN_GO if one not in _WHERE_IT_GOES]
    assert not missing, f"she cannot invent an action for {missing}"


@pytest.mark.parametrize("where", WHERE_A_TERM_CAN_GO)
def test_she_can_write_an_action_for_every_place_a_term_can_go(where):
    made = the_action_she_wrote(
        f"one she wrote for {where}", over=where, look_for=_identity()
    )
    assert made.do_it(None) is not None, f"the action for {where} did nothing"
    assert any(one.name == made.name for one in the_actions_she_has()), (
        f"the action for {where} is not available to the next search"
    )


def test_surface_syntax_is_compiled_rather_than_silently_refused():
    """An operator that cannot fire looks like one that decided not to."""
    made = the_action_she_wrote(
        "one written in surface syntax",
        over="the words",
        look_for=QUOTE(build(L("x", V("x")))),
    )
    assert made.do_it(None) is not None


def test_a_word_she_installed_actually_runs():
    """A registry key is not a capability. The thing has to compute."""
    from core.cognition.the_floor_she_stands_on import PLUS
    from core.cognition.what_she_could_do_next import _install_a_word

    says = _install_a_word(L("size", L("where", PLUS(V("size"), V("where")))))
    assert says(4, 3) == 7


def test_a_rule_she_installed_claims_no_evidence_it_does_not_have():
    """Fitted at nothing and judged at nothing, because it has been neither."""
    from core.cognition.what_she_could_do_next import _install_a_shape_for_a_rule

    rule = _install_a_shape_for_a_rule(L("cells", V("cells")))
    assert rule.fitted_at == ()
    assert rule.judged_at == ()


def test_asking_for_a_place_that_does_not_exist_still_refuses():
    """Adding three installers must not turn a typo into a silent no-op."""
    with pytest.raises(ValueError, match="nothing installs at"):
        the_action_she_wrote(
            "one for nowhere", over="somewhere else", look_for=_identity()
        )


def test_the_two_tables_agree():
    """A destination in one table and not the other is a place nothing reaches."""
    assert set(_WHERE_IT_GOES) == set(WHERE_A_TERM_CAN_GO)
