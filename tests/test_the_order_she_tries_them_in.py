"""The rule that decides what to try first is a term, and behaves as it did.

Two things have to hold at once, and the second is the one that makes the
first worth having.

Nothing has improved. The term computes what the Python expression computed,
and the order over her real vocabulary is identical. A change of behaviour here
would be a change nobody asked for, hiding inside a change of representation.

And the kind of thing has changed. The rule is now the same kind of object as
the things she invents — installed, kept, removed and replaced by the code that
installs, keeps, removes and replaces a head — which is what would have to be
true before she could ever write a better one.

Whether she writes a better one is experiment H and it has not been run. These
tests claim no self-improvement.
"""

from __future__ import annotations

import pytest

from core.cognition.an_invented_kind import addressings
from core.cognition.how_she_learns_to_look import (
    how_often_it_worked,
    in_the_order_worth_trying,
)
from core.cognition.one_algebra import _tells_her_the_answer  # noqa: PLC2701
from core.cognition.the_floor_she_stands_on import (
    HOW_MANY_PARTS,
    L,
    N,
    V,
    build,
    how_long,
)
from core.cognition.the_order_she_tries_them_in import (
    AS_FINE_AS_INTEGERS_ALLOW,
    THE_ORDER,
    WHAT_THE_ORDER_IS_GIVEN,
    forget_the_order,
    how_it_scores,
    order_read_back,
    the_order_she_uses,
    the_order_she_wrote,
    written_order,
)
from core.cognition.what_it_costs_to_say import _symbols  # noqa: PLC2701

_WANTED = {
    4: (1, 2, 3, 0),
    5: (1, 2, 3, 4, 0),
    6: (1, 2, 3, 4, 5, 0),
}


@pytest.fixture(autouse=True)
def _restore():
    yield
    forget_the_order()


def _the_python_rule_order(every) -> list[str]:
    """What how_she_learns_to_look computed before the rule became a term."""
    most = sum(len(found) for found in _WANTED.values())

    def worth(name: str):
        try:
            agreed = max(0, int(_tells_her_the_answer(every[name], _WANTED)))
        except (ArithmeticError, TypeError, ValueError):
            agreed = 0
        return (
            -((agreed + 1) / (most + 2) * how_often_it_worked(name).rate),
            _symbols(name),
            name,
        )

    return sorted(every, key=worth)


def test_the_term_gives_the_order_the_expression_gave() -> None:
    every = addressings()
    got = in_the_order_worth_trying(
        every, _tells_her_the_answer, _WANTED, shortest=_symbols
    )
    assert got == _the_python_rule_order(every)


def test_integers_are_fine_enough_that_rounding_reorders_nothing() -> None:
    """The one way moving to a term could quietly change behaviour."""
    for agreed, places, won, of in (
        (0, 30, 0, 0), (1, 30, 0, 5), (5, 30, 3, 10), (29, 30, 9, 10),
        (2, 3, 1, 2), (0, 1, 0, 1),
    ):
        exact = (agreed + 1) / (places + 2) * ((won + 1) / (of + 2))
        scored = how_it_scores(
            agreed=agreed, places=places, won=won, of=of, symbols=1
        )
        assert abs(scored / AS_FINE_AS_INTEGERS_ALLOW - exact) < 1e-6


def test_the_rule_is_a_term_of_the_floor_and_nothing_else() -> None:
    def heads(code):
        yield code.head
        for part in code.parts:
            yield from heads(part)

    assert set(heads(THE_ORDER)) <= set(HOW_MANY_PARTS)
    assert how_long(THE_ORDER) == 22
    assert len(WHAT_THE_ORDER_IS_GIVEN) == 5


def test_it_can_be_written_down_and_read_back() -> None:
    assert order_read_back(written_order()) == THE_ORDER


def test_a_different_rule_changes_the_order_and_the_lesion_puts_it_back() -> None:
    """Replaced and restored by the same shape of call a head gets."""
    every = addressings()
    before = in_the_order_worth_trying(
        every, _tells_her_the_answer, _WANTED, shortest=_symbols
    )
    # A rule that scores everything alike, so length alone decides.
    flat = build(
        L("agreed", L("places", L("won", L("of", L("symbols", N(1))))))
    )
    the_order_she_wrote(flat)
    assert the_order_she_uses() == flat
    during = in_the_order_worth_trying(
        every, _tells_her_the_answer, _WANTED, shortest=_symbols
    )
    assert during == sorted(every, key=lambda name: (_symbols(name), name))
    assert during != before
    forget_the_order()
    assert in_the_order_worth_trying(
        every, _tells_her_the_answer, _WANTED, shortest=_symbols
    ) == before


def test_a_rule_that_will_not_answer_sends_the_word_to_the_back() -> None:
    """A learned rule that proposes rubbish loses time and keeps nothing.

    That is the whole safety argument for learning the order at all, so it has
    to be true of a rule that does not answer as well as of one that answers
    badly. What is kept is decided by a gate the rule cannot reach.
    """
    from core.cognition.the_floor_she_stands_on import FST, Y, A

    forever = build(
        L("agreed", L("places", L("won", L("of", L("symbols",
            A(Y("go", L("k", A(V("go"), V("k")))), N(1)))))))
    )
    the_order_she_wrote(forever)
    assert how_it_scores(agreed=9, places=9, won=9, of=9, symbols=1) == 0
    nonsense = build(
        L("agreed", L("places", L("won", L("of", L("symbols", FST(N(1)))))))
    )
    the_order_she_wrote(nonsense)
    assert how_it_scores(agreed=9, places=9, won=9, of=9, symbols=1) == 0
    # The search still runs; every word simply scores nothing and length decides.
    every = addressings()
    got = in_the_order_worth_trying(
        every, _tells_her_the_answer, _WANTED, shortest=_symbols
    )
    assert got == sorted(every, key=lambda name: (_symbols(name), name))


def test_no_self_improvement_is_claimed_here() -> None:
    """The rule in force at import is the one that was authored.

    If a later change makes her start from something she wrote, that is a
    claim about experiment H and it needs experiment H's evidence.
    """
    assert the_order_she_uses() == THE_ORDER
