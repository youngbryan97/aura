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
def _restore(monkeypatch):
    from core.cognition import an_operator_she_invents as proposer
    from core.cognition.a_rule_with_no_shape import RULES_WITH_NO_SHAPE
    from core.cognition.one_algebra import DERIVED_HEADS
    from core.cognition.operator_invention import OperatorKernel

    monkeypatch.setattr(proposer, "_KERNEL", OperatorKernel())
    # These mechanism tests start without a held-out history. The rejection
    # integration below separately exercises the outer transaction.
    monkeypatch.setattr(
        "core.cognition.what_she_could_do_next._how_things_stand", lambda: ({}, ())
    )
    libraries = [(registry, dict(registry)) for registry in (RULES_WITH_NO_SHAPE, DERIVED_HEADS)]
    for registry, _ in libraries:
        registry.clear()
    held = dict(WHAT_SHE_COULD_DO)
    WHAT_SHE_COULD_DO.pop("invent an operator for what keeps failing", None)
    yield
    forget_the_objective()
    WHAT_SHE_COULD_DO.clear()
    WHAT_SHE_COULD_DO.update(held)
    for registry, contents in libraries:
        registry.clear()
        registry.update(contents)


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
        note_how_it_went(
            "doubling", solved=False, cases=[(x, 2 * x) for x in (1, 2, 3, 5, 8)]
        )
    came = WHAT_SHE_COULD_DO["invent an operator for what keeps failing"].do_it(None)
    from core.cognition.an_operator_she_invents import how_far_the_last_search_reached

    assert came is not None, (kernel.report(), how_far_the_last_search_reached())
    assert kernel._operators  # noqa: SLF001
    operator = next(iter(kernel.operators().values()))
    assert [operator.fn(x) for x in (-7, 0, 13, 29)] == [-14, 0, 26, 58]


def test_unlabelled_probes_do_not_certify_returning_constants():
    offer_inventing_an_operator()
    for _ in range(3):
        note_how_it_went("unknown", solved=False, probes=[1, 2, 3, 5, 8])
    assert WHAT_SHE_COULD_DO["invent an operator for what keeps failing"].do_it() is None
    assert not the_kernel().operators()


def test_installed_terms_return_to_search_and_rollback_removes_them():
    from core.cognition.what_she_already_knows_how_to_say import (
        what_she_already_knows_how_to_say,
    )

    offer_inventing_an_operator()
    for _ in range(3):
        note_how_it_went("doubling", solved=False, cases=[(x, 2*x) for x in range(5)])
    assert WHAT_SHE_COULD_DO["invent an operator for what keeps failing"].do_it(), the_kernel().report()
    operator = next(iter(the_kernel().operators().values()))
    assert operator.term in what_she_already_knows_how_to_say()
    the_kernel().rollback(operator.name)
    assert operator.term not in what_she_already_knows_how_to_say()


def test_learned_leaf_changes_matched_depth_search_with_lesion_and_rescue(monkeypatch):
    from core.cognition import what_she_already_knows_how_to_say as library
    from core.cognition.an_operator_she_invents import _a_candidate_for

    offer_inventing_an_operator()
    for _ in range(3):
        note_how_it_went("doubling", solved=False, cases=[(x, 2*x) for x in range(5)])
    assert WHAT_SHE_COULD_DO["invent an operator for what keeps failing"].do_it(), the_kernel().report()
    term = next(iter(the_kernel().operators().values())).term
    held_out = (-13, -1, 0, 11, 101)

    def succeeds(leaves):
        monkeypatch.setattr(library, "what_she_already_knows_how_to_say", lambda: leaves)
        for candidate in _a_candidate_for(
            "transfer", held_out, deepest=3, how_many=400, max_offered=None
        ):
            run = candidate.how_it_computes()
            try:
                if all(run(x) == 2*x for x in held_out):
                    return True
            except (TypeError, ValueError, ArithmeticError, TimeoutError):
                pass
        return False

    assert succeeds((term,))
    assert not succeeds(())
    assert succeeds((term,))


def test_live_sequence_boundary_records_complete_examples_once(monkeypatch):
    from core.cognition import sequence_induction as sequence

    monkeypatch.setattr(sequence, "_the_sequence_answer", lambda _: "")
    asked = "[1,2,3] becomes [2,3]. [4,5,6,7] becomes [5,6,7]. What does [6,7,8,9] become?"
    assert sequence.answer_sequence_question(asked) == ""
    residual = next(iter(the_kernel()._residuals.values()))
    assert residual.attempts == 1
    assert residual.solved == 0
    assert residual.cases == (((1, 2, 3), (2, 3)), ((4, 5, 6, 7), (5, 6, 7)))
    assert residual.probes == (6, 7, 8, 9)


def test_outer_rejection_restores_operator_semantics_and_lineage(monkeypatch):
    monkeypatch.setattr(
        "core.cognition.what_she_could_do_next._held_out_says_it_paid", lambda *_: False
    )
    offer_inventing_an_operator()
    for _ in range(3):
        note_how_it_went("doubling", solved=False, cases=[(x, 2*x) for x in range(5)])
    before = the_kernel().snapshot()
    action = WHAT_SHE_COULD_DO["invent an operator for what keeps failing"].do_it
    assert action() is None
    assert action.last_outcome == "did not pay"
    assert the_kernel().snapshot() == before
    assert the_kernel().report()["considered"] > 0


def test_exception_rolls_back_invention_and_keeps_observations():
    from core.cognition.what_she_can_take_back import only_if_it_pays

    offer_inventing_an_operator()
    for _ in range(3):
        note_how_it_went("doubling", solved=False, cases=[(x, 2*x) for x in range(5)])
    before = the_kernel().snapshot()
    with pytest.raises(RuntimeError, match="after install"):
        with only_if_it_pays():
            assert WHAT_SHE_COULD_DO["invent an operator for what keeps failing"].do_it()
            raise RuntimeError("after install")
    assert the_kernel().snapshot() == before
    assert the_kernel().residuals()[0].attempts == 3


def test_whole_sequence_examples_are_not_zipped_into_scalar_examples():
    from core.cognition.operator_invention import _the_term_as_a_function
    from core.cognition.the_floor_she_stands_on import Code

    identity = Code("given a thing", parts=(Code("the one it was given", value=0),))
    run = _the_term_as_a_function(identity)
    assert run([1, [2, 3], []]) == (1, (2, 3), ())
    assert run(2**100 + 1) == 2**100 + 1
    with pytest.raises(TypeError, match="integers"):
        run(1.25)


def test_conflicting_examples_cannot_admit_an_operator():
    offer_inventing_an_operator()
    for _ in range(3):
        note_how_it_went("contradiction", solved=False, cases=[(1, 2), (1, 3)])
    assert WHAT_SHE_COULD_DO["invent an operator for what keeps failing"].do_it() is None
    assert not the_kernel().operators()


def test_a_candidate_that_computes_nothing_never_reaches_the_dear_machinery():
    """The cheap probe first, which is what refusals should be spent on."""
    from core.cognition.an_operator_she_invents import (
        _computes_a_number,  # noqa: PLC2701
    )
    from core.cognition.the_floor_she_stands_on import PLUS, L, N, V, build

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
    from core.cognition.the_floor_she_stands_on import PLUS, TIMES, N, build
    from core.cognition.what_this_reminds_her_of import what_keeps_coming_up

    twice_inside_one = build(PLUS(TIMES(N(1), N(2)), TIMES(N(3), N(4))))
    found = dict(what_keeps_coming_up([twice_inside_one]))
    assert "times(.,.)" not in found


def test_specialising_reports_nothing_where_there_is_nothing_to_narrow():
    assert specialise("word/nothing here", [], costs=lambda cases: 1) is None


def test_recompressing_needs_something_to_compress():
    assert recompress([], costs=lambda cases: 1, at_least=99) == []
