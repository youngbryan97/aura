"""The arithmetic of a language that grows itself, and where it stops paying.

One test per claim, because a claim about growth that nothing checks is a
claim she will keep making after it stops being true.
"""

from __future__ import annotations

import pytest

from core.cognition import an_invented_kind as kinds
from core.cognition import widening_the_language as widening
from core.cognition.keeping_the_language_small import (
    how_capable,
    how_long_growth_can_last,
    what_a_word_is_worth,
    what_to_merge,
    worth_per_cost,
)
from core.cognition.what_growth_cannot_do import (
    BOUNDED,
    UNIVERSAL,
    UNKNOWN,
    can_be_decided,
    how_expressive,
    no_updater_wins_everywhere,
    what_a_new_word_can_buy,
    what_verification_is_available,
    what_would_need_an_oracle,
)
from core.cognition.what_it_costs_to_say import (
    everything_sayable,
    how_long_it_is,
    how_many_expressions,
    how_much_shorter,
    in_order_of_length,
    more_probable_by,
    what_already_says_it,
    what_it_means,
)


@pytest.fixture(autouse=True)
def _language_left_as_found():
    """Each test gets the language she was given, and gives it back."""
    where, what, ways, known = (
        dict(kinds.WHERE_FROM),
        dict(kinds.WHAT_OF_IT),
        dict(kinds.WAYS_TO_BUILD),
        dict(kinds.KINDS),
    )
    kinds.WAYS_TO_BUILD.clear()
    try:
        yield
    finally:
        for holds, was in (
            (kinds.WHERE_FROM, where),
            (kinds.WHAT_OF_IT, what),
            (kinds.WAYS_TO_BUILD, ways),
            (kinds.KINDS, known),
        ):
            holds.clear()
            holds.update(was)


def _only_the_closure_says(closure: dict) -> tuple[int, ...]:
    """A correspondence some composition produces and no primitive does.

    Chosen rather than written down, because which compositions are genuinely
    new depends on the words she has, and that is the thing under test.
    """
    for name, word in closure.items():
        if ", then " not in name:
            continue
        where = tuple(word(index, 4) % 4 for index in range(4))
        if len(set(where)) != 4:
            continue
        if all(
            tuple(one(index, 4) % 4 for index in range(4)) != where
            for one in kinds.WHERE_FROM.values()
        ):
            return where
    raise AssertionError("the ways of building produced nothing new")


def test_the_three_sizes_are_not_the_same_number():
    """A(t), P(A(t)) and E(t) are different, and only the third decides growth."""
    expressions = len(list(kinds.every_meaning()))
    meanings = len(everything_sayable())
    assert len(kinds.WHERE_FROM) < meanings < expressions


def test_admitting_a_way_of_building_enlarges_the_meanings_not_only_the_spelling():
    before = len(everything_sayable())
    kinds.WAYS_TO_BUILD["one after another"] = widening.one_after_another
    assert len(everything_sayable()) > before


def test_a_word_the_closure_already_says_is_refused():
    """The macro check runs against everything constructible, not the primitives."""
    kinds.WAYS_TO_BUILD["one after another"] = widening.one_after_another
    closure = kinds.addressings()
    where = _only_the_closure_says(closure)
    states = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 1, 2, 6)]
    pairs = [(s, tuple(s[i] for i in where)) for s in states]

    assert widening.an_addressing_nobody_wrote(pairs, already=closure) is None
    # And the primitives alone would have called this growth.
    assert widening.an_addressing_nobody_wrote(pairs, already=kinds.WHERE_FROM) is not None


def test_a_macro_that_pays_for_itself_is_offered_as_shorthand():
    kinds.WAYS_TO_BUILD["one after another"] = widening.one_after_another
    closure = kinds.addressings()
    where = _only_the_closure_says(closure)
    states = [(1, 2, 3, 4), (5, 6, 7, 8)]
    pairs = [(s, tuple(s[i] for i in where)) for s in states]

    found = widening.a_shorthand_worth_having(pairs, already=closure)
    assert found is not None
    _, said_by = found
    assert ", then " in said_by


def test_description_length_counts_what_a_word_is_built_from():
    plain = kinds.Induced("the far end", "the far end", "as it is")
    built = kinds.Induced("the far end, then one back", "the far end", "as it is")
    assert how_long_it_is(built) > how_long_it_is(plain)


def test_the_search_is_walked_shortest_first():
    lengths = [how_long_it_is(one) for one in kinds.every_meaning()]
    assert lengths == sorted(lengths)


def test_widening_does_not_push_the_simple_answer_out_of_reach():
    """The point of ordering: a bigger language must not bury a short truth."""
    simple = "take the far end"
    narrow = [one.name for one in kinds.every_meaning()].index(simple)
    kinds.WAYS_TO_BUILD["one after another"] = widening.one_after_another
    wide = [one.name for one in kinds.every_meaning()].index(simple)
    assert wide == narrow


def test_the_search_space_formula_is_the_closed_form():
    for words in (2, 5, 20):
        for longest in (1, 3, 5):
            walked = sum(words**step for step in range(1, longest + 1))
            assert how_many_expressions(words, longest) == walked


def test_shortening_multiplies_the_prior_by_two_to_the_saving():
    assert more_probable_by(10) == 1024.0
    assert more_probable_by(0) == 1.0


def test_a_word_used_k_times_saves_k_times_over():
    assert how_much_shorter(before=7, after=1, used=4) == 24


def test_her_rule_language_is_bounded_so_a_word_can_still_mean_something_new():
    mine = how_expressive(repeats_without_bound=False, branches_on_its_own_values=False)
    assert mine.verdict == BOUNDED
    assert "did not have" in what_a_new_word_can_buy(mine)


def test_a_universal_language_cannot_be_made_more_expressive_from_inside():
    turing = how_expressive(repeats_without_bound=True, branches_on_its_own_values=True)
    assert turing.verdict == UNIVERSAL
    assert "nothing newly expressible" in what_a_new_word_can_buy(turing)


@pytest.mark.parametrize(
    "rule",
    [lambda choices: choices[0], lambda choices: choices[1], lambda choices: choices[-1]],
)
def test_no_update_rule_improves_on_every_environment(rule):
    refuted = no_updater_wins_everywhere(rule)
    assert refuted.holds
    assert refuted.scored < refuted.the_other_scored


def test_unsayable_is_asserted_only_after_the_whole_small_language_was_walked():
    mine = how_expressive(repeats_without_bound=False, branches_on_its_own_values=False)
    turing = how_expressive(repeats_without_bound=True, branches_on_its_own_values=True)
    assert can_be_decided(language=mine, exhaustive_search_finished=True) is True
    assert can_be_decided(language=mine, exhaustive_search_finished=False) == UNKNOWN
    assert can_be_decided(language=turing, exhaustive_search_finished=True) == UNKNOWN


def test_an_arbitrary_self_change_never_offers_a_proof():
    assert "proof over the restricted form" not in what_verification_is_available(
        change_is_arbitrary=True
    )
    assert "rollback" in what_verification_is_available(change_is_arbitrary=True)


def test_a_question_needing_an_oracle_is_refused_rather_than_approximated():
    said = what_would_need_an_oracle("whether this settles")
    assert "refused" in said and "oracle" in said


def test_a_word_nothing_uses_never_pays():
    assert not what_a_word_is_worth(
        "unused", vocabulary=25, longest=6, shorter_by=6, used=0
    ).pays


def test_a_word_that_shortens_nothing_never_pays():
    assert not what_a_word_is_worth(
        "flat", vocabulary=25, longest=6, shorter_by=0, used=9
    ).pays


def test_a_word_that_shortens_a_used_structure_pays():
    assert what_a_word_is_worth(
        "worth it", vocabulary=25, longest=6, shorter_by=4, used=3
    ).pays


def test_synonyms_collapse_so_the_language_can_end_smaller():
    kinds.WAYS_TO_BUILD["one after another"] = widening.one_after_another
    expressions = len(list(kinds.every_meaning()))
    kept = what_to_merge(everything_sayable())
    assert len(kept) < expressions


def test_growth_on_fixed_memory_has_a_last_step():
    assert how_long_growth_can_last(8) == 256
    assert how_long_growth_can_last(0) == 1


def test_capability_names_its_budget_and_rises_with_it():
    states = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 1, 2, 6)]
    tasks = [(3, 2, 1, 0), (1, 2, 3, 0)]

    def solvable(perm, within):
        want = [(s, tuple(s[i] for i in perm)) for s in states]
        for at, one in enumerate(kinds.every_meaning(), 1):
            if at > within:
                return False
            if all(one.read(before) == after for before, after in want):
                return True
        return False

    plenty = how_capable(tasks, solvable, within=500)
    # The tight budget is derived from where the answers actually sit, so the
    # test measures the effect of a budget rather than a number I picked.
    deepest = max(
        next(
            (at for at, one in enumerate(kinds.every_meaning(), 1)
             if all(one.read(s) == tuple(s[i] for i in perm) for s in states)),
            0,
        )
        for perm in tasks
    )
    mean = how_capable(tasks, solvable, within=max(1, deepest - 1))
    assert plenty.share > mean.share
    assert plenty.within == 500


def test_capability_per_cost_is_not_maximised_by_counting_words():
    assert worth_per_cost(0.5, 5) > worth_per_cost(0.5, 50)
    assert worth_per_cost(0.9, 10) > worth_per_cost(0.5, 10)


def test_two_ways_of_saying_one_thing_have_one_meaning():
    plain = kinds.Induced("the far end", "the far end", "as it is")
    same = kinds.Induced("the far end", "the far end", "as it is")
    assert what_it_means(plain) == what_it_means(same)
    assert what_already_says_it(what_it_means(plain)) is not None


def test_ordering_is_stable_for_equal_lengths():
    made = list(kinds.every_meaning())
    assert in_order_of_length(made) == in_order_of_length(list(reversed(made)))
