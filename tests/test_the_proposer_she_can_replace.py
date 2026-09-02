"""The thing that says what to try next is a term, and it comes back after a restart.

The order module made one part of the machinery an object: the rule deciding
which word goes in a hole first. What stayed authored was the larger half — the
loop producing candidates at all. A proposer written as a Python generator is a
proposer she cannot replace.

A proposer here is a term of one shape: given a number and how many leaves
there are, it hands back the ENCODING of a candidate. So a stream is that term
asked for nought, one, two and so on, and it is homoiconic all the way down —
what comes back is a term written as numbers and pairs, which is what quotation
already makes a term into.

Scope, said rather than implied: the default proposer covers one arithmetic
head over two leaves, which is where nearly every short answer sits. Longer
candidates still come from the enumerator. The claim is that the proposal path
has a replaceable term in it, not that no Python remains anywhere.
"""

from __future__ import annotations

import json
import pathlib
import tempfile

import pytest

from core.cognition.the_floor_she_stands_on import (
    HOW_MANY_PARTS,
    L,
    LEFTOVER,
    N,
    NIL,
    PAIR,
    SIGNATURE,
    V,
    build,
    how_long,
)
from core.cognition.the_proposer_she_can_replace import (
    HOW_MANY_ARITHMETIC_HEADS,
    THE_PROPOSER,
    WHAT_A_PROPOSER_IS_GIVEN,
    forget_the_proposer,
    the_candidate_at,
    the_proposer_in_use,
    the_proposer_read_back,
    the_proposer_she_wrote,
    the_proposer_written_down,
)


@pytest.fixture(autouse=True)
def _restore():
    yield
    forget_the_proposer()


def _only_plus():
    """A proposer that offers one shape, so a replacement is unmistakable."""
    return build(
        L(
            "which",
            L(
                "leaves",
                PAIR(
                    N(SIGNATURE.index("plus")),
                    PAIR(
                        N(0),
                        PAIR(
                            PAIR(
                                N(SIGNATURE.index("the one it was given")),
                                PAIR(LEFTOVER(V("which"), V("leaves")), NIL),
                            ),
                            PAIR(
                                PAIR(
                                    N(SIGNATURE.index("a number")),
                                    PAIR(N(1), NIL),
                                ),
                                NIL,
                            ),
                        ),
                    ),
                ),
            ),
        )
    )


def test_a_term_hands_back_a_term() -> None:
    """Homoiconicity as behaviour rather than as a property of the design."""
    made = [the_candidate_at(k, leaves=10) for k in range(24)]
    assert all(one is not None for one in made)
    assert len({repr(one) for one in made}) == 24
    for one in made:
        assert one.head in HOW_MANY_PARTS
        assert len(one.parts) == HOW_MANY_PARTS[one.head]


def test_it_walks_every_arithmetic_head() -> None:
    heads = {the_candidate_at(k, leaves=10).head for k in range(HOW_MANY_ARITHMETIC_HEADS)}
    assert len(heads) == HOW_MANY_ARITHMETIC_HEADS
    assert heads == {
        SIGNATURE[at]
        for at in range(
            SIGNATURE.index("plus"), SIGNATURE.index("same as") + 1
        )
    }


def test_the_proposer_is_a_term_of_the_floor_and_nothing_else() -> None:
    def heads(code):
        yield code.head
        for part in code.parts:
            yield from heads(part)

    assert set(heads(THE_PROPOSER)) <= set(HOW_MANY_PARTS)
    assert how_long(THE_PROPOSER) < 200
    assert len(WHAT_A_PROPOSER_IS_GIVEN) == 2


def test_a_different_proposer_offers_different_things_and_the_lesion_puts_it_back() -> None:
    before = [repr(the_candidate_at(k, leaves=5)) for k in range(4)]
    mine = _only_plus()
    the_proposer_she_wrote(mine)
    assert the_proposer_in_use() == mine
    during = [repr(the_candidate_at(k, leaves=5)) for k in range(4)]
    assert during != before
    assert all(one.startswith("(plus") for one in during)
    forget_the_proposer()
    assert the_proposer_in_use() == THE_PROPOSER
    assert [repr(the_candidate_at(k, leaves=5)) for k in range(4)] == before


def test_a_proposer_that_will_not_answer_offers_nothing_rather_than_stopping() -> None:
    """A search asks thousands of times; one bad answer must not end it."""
    from core.cognition.the_floor_she_stands_on import A, FST, V, Y

    forever = build(L("which", L("leaves", A(Y("go", L("k", A(V("go"), V("k")))), N(1)))))
    the_proposer_she_wrote(forever)
    assert the_candidate_at(0, leaves=5) is None
    nonsense = build(L("which", L("leaves", FST(N(1)))))
    the_proposer_she_wrote(nonsense)
    assert the_candidate_at(0, leaves=5) is None
    not_a_term = build(L("which", L("leaves", N(7))))
    the_proposer_she_wrote(not_a_term)
    assert the_candidate_at(0, leaves=5) is None


def test_the_search_asks_the_proposer_before_the_enumerator() -> None:
    import inspect

    from core.cognition import a_way_of_computing_she_wrote as writing

    source = inspect.getsource(writing.a_way_of_computing_she_wrote)
    assert "the_candidate_at" in source
    assert source.index("the_candidate_at") < source.index("for body in stream")


def test_what_she_replaced_survives_a_restart() -> None:
    from core.cognition import what_she_gave_meaning as keeping
    from core.cognition.the_order_she_tries_them_in import (
        THE_ORDER,
        forget_the_order,
        the_order_she_uses,
        the_order_she_wrote,
    )

    mine = _only_plus()
    flat = build(L("a", L("b", L("c", L("d", L("e", N(1)))))))
    the_proposer_she_wrote(mine)
    the_order_she_wrote(flat)
    kept_at = keeping._KEPT_AT
    try:
        with tempfile.TemporaryDirectory(prefix="aura-machinery-") as somewhere:
            keeping._KEPT_AT = pathlib.Path(somewhere) / "meanings.json"
            assert keeping.keep()
            saved = json.loads(keeping._KEPT_AT.read_text(encoding="utf-8"))
            assert sorted(saved["language"]["machinery"]) == ["order", "proposer"]
            forget_the_proposer()
            forget_the_order()
            assert the_proposer_in_use() == THE_PROPOSER
            assert the_order_she_uses() == THE_ORDER
            keeping.recall()
            assert the_proposer_in_use() == mine
            assert the_order_she_uses() == flat
    finally:
        keeping._KEPT_AT = kept_at
        forget_the_proposer()
        forget_the_order()


def test_only_what_she_changed_is_kept() -> None:
    """A file recording the authored default records nothing she did."""
    from core.cognition import what_she_gave_meaning as keeping

    forget_the_proposer()
    written = keeping._words_she_derived()
    assert written["machinery"] == {}


def test_it_can_be_written_down_and_read_back() -> None:
    assert the_proposer_read_back(the_proposer_written_down()) == THE_PROPOSER
    mine = _only_plus()
    the_proposer_she_wrote(mine)
    assert the_proposer_read_back(the_proposer_written_down()) == mine


def test_no_replacement_is_claimed_at_import() -> None:
    """The proposer in force at import is the one that was authored.

    If a later change makes her start from something she wrote, that is a
    claim about experiment H and it needs experiment H's evidence.
    """
    assert the_proposer_in_use() == THE_PROPOSER
