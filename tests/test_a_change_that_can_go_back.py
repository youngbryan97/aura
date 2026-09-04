"""Every promotion carries what it replaced, so every promotion can be undone.

The module's own docstring said "anything can go back, because every
promotion pushes what it replaced onto a stack". No caller anywhere passed
``replaced``. The stack was always empty, ``what_it_replaced`` always
returned None, and ``put_it_back`` handed the caller a None it was expected
to restore from. Nothing could go back, and the mechanism had never once
been reachable.

The second thing held here is what ``shadow`` means. It does not mean
isolated: a shadow change is installed and in use the moment it is made, and
the only line in the module that reads a state at all is the one that
archives a retirement. What shadow buys is reversibility, not containment,
and the tests say so out loud so nobody reads the industry meaning into it.
"""

from __future__ import annotations

import pytest

from core.cognition import she_decides_to_develop as S
from core.cognition.an_invented_kind import WHERE_FROM
from core.cognition.how_a_change_is_promoted import (
    promote,
    put_it_back,
    the_chain_holds,
    the_receipts,
    what_it_replaced,
)
from core.cognition.what_she_can_take_back import as_it_stands
from core.cognition.what_she_could_do_next import (
    WHAT_SHE_COULD_DO,
    what_she_could_do,
)


@pytest.fixture
def an_action_that_writes_a_word():
    held = dict(WHAT_SHE_COULD_DO)
    was = as_it_stands()
    WHAT_SHE_COULD_DO.clear()

    def writes_a_word(situation=None):
        WHERE_FROM["a_word_she_promoted"] = lambda a, b: a
        return "wrote a word"

    yield what_she_could_do(
        "write a word",
        over="the words",
        kind="a test",
        do_it=writes_a_word,
        needs_a_case=False,
    )
    was.restore()
    WHAT_SHE_COULD_DO.clear()
    WHAT_SHE_COULD_DO.update(held)


def _promoted(action) -> str:
    decided = S.Decision(
        action=action, worth=None, because="chosen", grounds="a test"
    )
    S.she_develops_herself(decided)
    return f"{action.over}/{action.name}"


def test_a_promotion_records_what_it_replaced(an_action_that_writes_a_word):
    """The stack was empty for the life of the module."""
    at = _promoted(an_action_that_writes_a_word)
    assert what_it_replaced(at) is not None, "the promotion carried nothing to undo"


def test_a_change_can_actually_go_back(an_action_that_writes_a_word):
    """put_it_back handed back a value and left the state alone."""
    at = _promoted(an_action_that_writes_a_word)
    assert "a_word_she_promoted" in WHERE_FROM
    assert put_it_back(at) is not None
    assert "a_word_she_promoted" not in WHERE_FROM


def test_going_back_writes_a_receipt_that_says_whether_it_worked(
    an_action_that_writes_a_word,
):
    at = _promoted(an_action_that_writes_a_word)
    put_it_back(at)
    last = the_receipts()[-1]
    assert last.at == at
    assert last.became in {"rolled back", "would not go back"}
    assert the_chain_holds()


def test_an_undo_that_raises_is_recorded_as_one_that_would_not_go_back():
    """A failed undo is a result, and the record has to say so."""

    class _WillNotGoBack:
        def restore(self):
            raise RuntimeError("the registry moved under it")

    promote(
        "the words/one_that_will_not_go_back",
        became="shadow",
        started_by="she",
        evidence="a test",
        replaced=_WillNotGoBack(),
    )
    put_it_back("the words/one_that_will_not_go_back")
    last = the_receipts()[-1]
    assert last.became == "would not go back"
    assert "did not come back" in last.evidence


def test_shadow_does_not_mean_isolated(an_action_that_writes_a_word):
    """It means reversible. The change is live from the moment it is made."""
    at = _promoted(an_action_that_writes_a_word)
    assert "a_word_she_promoted" in WHERE_FROM, (
        "a shadow change is installed and in use; if this ever becomes false, "
        "the docstring saying so must change with it"
    )
    assert what_it_replaced(at) is not None
