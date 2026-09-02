"""The operator kernel, the search objective, and the library as one thing.

Three modules that each close a gap the previous work left open, and each of
them is the kind of thing that is easy to build and easy to leave unreachable.
A gate with no caller is a claim about what would happen if something asked; a
scoring function in Python bounds the space of rules a search can look for; a
library judged one entry at a time cannot see that two entries should be one.
"""

from __future__ import annotations

import pytest

from core.cognition.an_operator_she_invents import (
    how_it_has_gone,
    note_how_it_went,
    offer_inventing_an_operator,
    the_kernel,
)
from core.cognition.the_shape_of_her_library import (
    how_long_the_library_is,
    recompress,
    specialise,
    what_the_library_costs,
    where_the_budget_is,
)
from core.cognition.what_counts_as_better import (
    THE_OBJECTIVE,
    WHAT_THE_OBJECTIVE_IS_GIVEN,
    forget_the_objective,
    how_bad_that_is,
    the_objective_read_back,
    the_objective_she_uses,
    the_objective_she_wrote,
    written_objective,
)
from core.cognition.what_she_could_do_next import WHAT_SHE_COULD_DO


@pytest.fixture(autouse=True)
def _restore():
    held = dict(WHAT_SHE_COULD_DO)
    yield
    forget_the_objective()
    WHAT_SHE_COULD_DO.clear()
    WHAT_SHE_COULD_DO.update(held)


# ── the kernel ───────────────────────────────────────────────────────────


def test_a_family_that_fails_once_is_not_persistent():
    """The refusal that was firing for the right reason and the wrong cause."""
    kernel = the_kernel()
    kernel._residuals.clear()  # noqa: SLF001
    note_how_it_went("tried once", solved=False, probes=[1, 2])
    assert not kernel.residuals()
    for _ in range(5):
        note_how_it_went("keeps failing", solved=False, probes=[1, 2])
    assert [one.family for one in kernel.residuals()] == ["keeps failing"]


def test_the_diagnostic_shows_families_before_anything_is_wrong():
    """A reading that says nothing until something is broken cannot say nothing is."""
    kernel = the_kernel()
    kernel._residuals.clear()  # noqa: SLF001
    note_how_it_went("fine", solved=True, probes=[1])
    assert [one["family"] for one in how_it_has_gone()] == ["fine"]
    assert how_it_has_gone()[0]["persistent"] is False


def test_the_kernel_installs_through_its_own_gate():
    """It had a complete gate and nobody to open it. This is the something."""
    kernel = the_kernel()
    kernel._residuals.clear()  # noqa: SLF001
    kernel._operators.clear()  # noqa: SLF001
    offer_inventing_an_operator()
    for _ in range(5):
        note_how_it_went("doubling", solved=False, probes=[1, 2, 3, 5, 8])
    came = WHAT_SHE_COULD_DO["invent an operator for what keeps failing"].do_it(None)
    assert came is not None
    assert kernel._operators  # noqa: SLF001


def test_a_candidate_that_computes_nothing_never_reaches_the_dear_machinery():
    """The cheap probe first, which is what refusals should be spent on."""
    from core.cognition.an_operator_she_invents import (
        _computes_a_number,  # noqa: PLC2701
    )
    from core.cognition.the_floor_she_stands_on import L, N, PLUS, V, build

    # The kernel applies a term to one value, so a candidate that computes is
    # one that takes a value. A bare body with a free name is not a candidate;
    # it is a fragment.
    adds_one = build(L("it", PLUS(V("it"), N(1))))
    assert _computes_a_number(adds_one, [3]) is True
    assert _computes_a_number(build(N(1)), []) is False
    assert _computes_a_number(build(N(1)), [3]) is False


# ── the objective ────────────────────────────────────────────────────────


def test_the_objective_is_where_the_winner_sat_and_nothing_else():
    assert how_bad_that_is(sat=3, of=10, symbols=5) == 3.0
    assert how_bad_that_is(sat=0, of=10, symbols=5) == 0.0
    assert len(WHAT_THE_OBJECTIVE_IS_GIVEN) == 3


def test_a_different_objective_is_installed_and_lesioned_like_a_head():
    from core.cognition.the_floor_she_stands_on import L, N, build

    always_seven = build(L("sat", L("of", L("symbols", N(7)))))
    the_objective_she_wrote(always_seven)
    assert the_objective_she_uses() is always_seven
    assert how_bad_that_is(sat=0, of=10, symbols=1) == 7.0
    forget_the_objective()
    assert the_objective_she_uses() is THE_OBJECTIVE
    assert how_bad_that_is(sat=0, of=10, symbols=1) == 0.0


def test_an_objective_that_refuses_scores_as_badly_as_possible():
    """A broken objective must not be able to look like a good result."""
    from core.cognition.the_floor_she_stands_on import Code

    the_objective_she_wrote(Code("a number", parts=(), value=1))
    assert how_bad_that_is(sat=0, of=42, symbols=1) == 42.0


def test_the_objective_survives_being_written_down():
    row = written_objective()
    again = the_objective_read_back(row)
    assert again is not None
    assert written_objective() == row


# ── the library ──────────────────────────────────────────────────────────


def test_the_two_part_code_is_the_library_plus_everything_given_it():
    probe = [("a", ()), ("b", ())]
    whole = what_the_library_costs(probe, costs=lambda cases: 5)
    assert whole == how_long_the_library_is() + 10


def test_the_budget_refuses_to_guess_where_the_record_cannot_say():
    from core.cognition.what_she_could_do_next import WHAT_THEY_HAVE_DONE
    from core.cognition.what_she_is_made_of import what_she_is_made_of

    held = dict(WHAT_THEY_HAVE_DONE)
    WHAT_THEY_HAVE_DONE.clear()
    try:
        parts = [one for one in what_she_is_made_of() if one.term is not None]
        assert where_the_budget_is() == len(parts)
    finally:
        WHAT_THEY_HAVE_DONE.clear()
        WHAT_THEY_HAVE_DONE.update(held)


def test_a_shape_in_one_term_is_not_a_shape_that_recurs():
    """Local repetition inside a term is not evidence of shared structure."""
    from core.cognition.the_floor_she_stands_on import N, PLUS, TIMES, build
    from core.cognition.what_this_reminds_her_of import what_keeps_coming_up

    twice_inside_one = build(PLUS(TIMES(N(1), N(2)), TIMES(N(3), N(4))))
    found = dict(what_keeps_coming_up([twice_inside_one]))
    assert "times(.,.)" not in found


def test_specialising_reports_nothing_where_there_is_nothing_to_narrow():
    assert specialise("word/nothing here", [], costs=lambda cases: 1) is None


def test_recompressing_needs_something_to_compress():
    assert recompress([], costs=lambda cases: 1, at_least=99) == []
