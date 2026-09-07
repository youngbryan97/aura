"""A universal language, and a search that walks 380 terms of it."""
from __future__ import annotations

import collections

import pytest

from core.cognition.how_far_the_search_reaches import (
    how_far_it_reaches,
    how_many_at,
    how_many_up_to,
    what_the_library_buys,
    what_the_search_reached,
)

#: What the operator search stands on: nothing, three constants, one variable.
FLOOR_LEAVES = 5


@pytest.mark.parametrize("size", [1, 2, 3])
def test_the_count_agrees_with_the_generator_it_counts(size: int) -> None:
    """Counted rather than generated, and the two must not disagree.

    A denominator computed by a different rule from the thing it divides is a
    ratio of two different questions.
    """
    from core.cognition.the_floor_she_stands_on import every_code, how_long

    seen = collections.Counter(
        how_long(one)
        for one in every_code(deepest=3, variables=1, constants=(0, 1, 2), also=())
    )
    assert how_many_at(size, leaves=FLOOR_LEAVES) == seen[size]


def test_the_whole_space_at_depth_three_is_the_number_the_code_already_knew() -> None:
    """`an_operator_she_invents` says it "walked the same 380 terms"."""
    assert how_many_up_to(3, leaves=FLOOR_LEAVES) == 380


def test_a_cap_above_the_space_is_a_search_that_exhausted_it() -> None:
    """Reporting the cap as the count reads as sampling when it was exhaustive."""
    reached = what_the_search_reached(
        deepest=3, leaves=FLOOR_LEAVES, would_examine=4000
    )
    assert reached.there_were == 380
    assert reached.examined == 380, "it cannot walk more terms than exist"
    assert reached.exhausted
    assert reached.share_examined == 1.0


def test_a_cap_below_the_space_is_a_sample_and_says_so() -> None:
    reached = what_the_search_reached(
        deepest=4, leaves=FLOOR_LEAVES, would_examine=4000
    )
    assert reached.there_were > reached.would_examine
    assert not reached.exhausted
    assert 0.0 < reached.share_examined < 1.0
    assert reached.to_dict()["one_in"] >= 1


def test_a_library_entry_is_eventually_worth_more_than_a_depth() -> None:
    """The only thing that moves a shortest-first horizon is better leaves."""
    few = what_the_library_buys(deepest=3, floor_leaves=FLOOR_LEAVES, from_her_library=2)
    assert few["a_library_entry_is_worth"] == "less than a depth"

    many = what_the_library_buys(
        deepest=3, floor_leaves=FLOOR_LEAVES, from_her_library=20
    )
    assert many["a_library_entry_is_worth"] == "more than a depth"
    assert many["times_larger"] > few["times_larger"]


def test_more_leaves_and_more_depth_both_grow_the_space() -> None:
    assert how_many_up_to(3, leaves=6) > how_many_up_to(3, leaves=5)
    assert how_many_up_to(4, leaves=5) > how_many_up_to(3, leaves=5)


def test_nothing_is_admitted_below_size_one() -> None:
    assert how_many_at(0, leaves=5) == 0
    assert how_many_at(-3, leaves=5) == 0
    assert how_many_up_to(1, leaves=5) == 5


def test_the_report_carries_both_the_reach_and_what_widens_it() -> None:
    said = how_far_it_reaches(deepest=3, leaves=FLOOR_LEAVES, would_examine=4000)
    assert said["reach"]["there_were"] == 380
    assert said["library"]["terms_at_one_more_depth"] > 380


def test_what_her_library_actually_offers_is_reported_rather_than_assumed() -> None:
    """The mechanism that moves the horizon, and how much is in it."""
    from core.cognition.what_she_already_knows_how_to_say import (
        what_she_already_knows_how_to_say,
    )

    hers = what_she_already_knows_how_to_say()
    said = how_far_it_reaches(
        deepest=3,
        leaves=FLOOR_LEAVES + len(hers),
        would_examine=4000,
        from_her_library=len(hers),
    )
    assert said["library"]["from_her_library"] == len(hers)
    # No assertion about the size: an empty library is a fact about this
    # process rather than a failure, and the point is that it is reported.
    assert said["reach"]["there_were"] >= 380


def test_the_proposer_reports_the_walk_it_actually_took() -> None:
    """Wired, not beside it: measured on the walk rather than recomputed.

    A reach recomputed from the parameters says what the walk should have
    covered. This says what it did, which is the only version that can
    disagree with the parameters and be right.
    """
    from core.cognition.an_operator_she_invents import (
        _a_candidate_for,  # noqa: PLC2701
        how_far_the_last_search_reached,
    )

    list(_a_candidate_for("a family", probes=(1, 2, 3), how_many=500))
    said = how_far_the_last_search_reached()
    assert said["searched"] is True
    assert said["reach"]["there_were"] == 380
    assert said["reach"]["walked"] > 0
    assert said["reach"]["walked"] <= said["reach"]["there_were"]
    assert said["reach"]["exhausted"] is True, (
        "a cap of 500 over 380 terms exhausts the space"
    )


def test_a_process_that_has_not_searched_says_so_rather_than_reporting_zero() -> None:
    from core.cognition import an_operator_she_invents as proposer

    was = dict(proposer._NOTE_THE_REACH)  # noqa: PLC2701, SLF001
    proposer._NOTE_THE_REACH.clear()  # noqa: PLC2701, SLF001
    try:
        said = proposer.how_far_the_last_search_reached()
        assert said["searched"] is False
        assert "no operator search has run" in said["why"]
    finally:
        proposer._NOTE_THE_REACH.update(was)  # noqa: PLC2701, SLF001


def test_circumplex_drift_cannot_manufacture_a_measurement() -> None:
    """A thousandth of a degree is drift, and it read as a faculty.

    The circumplex is read once per generation and moves between reads: one
    run had intact at 0.6812 and an arm that removes nothing about sampling
    at 0.6811. A mean-difference test called that arm MEASURED. A lesion of
    this channel does one exact thing — the arm samples at the neutral — so
    that is what the reading looks for.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "matched_protocol", "tools/matched/run_matched_substrate.py"
    )
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)

    protocol._TEMPERATURE_ASKED.update(
        {
            protocol.INTACT: [0.6812] * 3,
            protocol.NO_ENDOGENOUS: [protocol.NEUTRAL_TEMPERATURE] * 3,
            protocol.NO_DEVELOPMENTAL: [0.6811] * 3,
            protocol.NO_RECURRENT: [0.6812] * 3,
        }
    )
    measured = {"delta_mean": -0.01, "separated": False}

    lesioned = protocol._faculty_reading(measured, protocol.NO_ENDOGENOUS)
    assert lesioned["outcome"] == "MEASURED"
    assert lesioned["sampled_at"] == protocol.NEUTRAL_TEMPERATURE

    for arm in (protocol.NO_DEVELOPMENTAL, protocol.NO_RECURRENT):
        drifted = protocol._faculty_reading(measured, arm)
        assert drifted["outcome"] == "NOT_MEASURED", arm
        assert "what_would_measure_it" in drifted


def _protocol():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "matched_protocol", "tools/matched/run_matched_substrate.py"
    )
    made = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(made)
    return made


def test_a_faculty_in_the_path_with_nothing_in_it_is_not_a_faculty_out_of_it() -> None:
    """Two reasons a delta cannot be read, and only one is about the harness.

    An arm whose faculty is not in this path cannot be measured here however
    long it runs. An arm whose faculty is in the path and empty is measured
    perfectly and finds nothing, which is a reading of her library. Reporting
    both as "not in the path" hides the second behind the first.
    """
    protocol = _protocol()
    protocol._TEMPERATURE_ASKED.update(
        {protocol.INTACT: [0.681] * 3, protocol.NO_DEVELOPMENTAL: [0.681] * 3,
         protocol.NO_RECURRENT: [0.681] * 3}
    )
    protocol._PROMPT_LENGTH.update(
        {protocol.INTACT: [400] * 3, protocol.NO_DEVELOPMENTAL: [400] * 3,
         protocol.NO_RECURRENT: [400] * 3}
    )
    measured = {"delta_mean": -0.01, "separated": False}

    developmental = protocol._faculty_reading(measured, protocol.NO_DEVELOPMENTAL)
    recurrent = protocol._faculty_reading(measured, protocol.NO_RECURRENT)
    assert developmental["outcome"] == "NOT_MEASURED"
    assert recurrent["outcome"] == "NOT_MEASURED"
    assert developmental["why"] != recurrent["why"], (
        "an empty library and an absent path are the same sentence"
    )
    assert "nothing in it" in developmental["why"]
    assert "not in the path" in recurrent["why"]


def test_a_shorter_context_is_a_faculty_that_reached_the_generation() -> None:
    """String length has no drift: the block is there or it is not."""
    protocol = _protocol()
    protocol._TEMPERATURE_ASKED.update(
        {protocol.INTACT: [0.681] * 3, protocol.NO_DEVELOPMENTAL: [0.681] * 3}
    )
    protocol._PROMPT_LENGTH.update(
        {protocol.INTACT: [400] * 3, protocol.NO_DEVELOPMENTAL: [340] * 3}
    )
    got = protocol._faculty_reading(
        {"delta_mean": -0.01, "separated": False}, protocol.NO_DEVELOPMENTAL
    )
    assert got["outcome"] == "MEASURED"
    assert got["prompt_characters"] == 340
    assert got["intact_prompt_characters"] == 400


def test_what_she_has_learned_is_read_rather_than_described() -> None:
    """Empty today, and empty because the library is empty."""
    protocol = _protocol()
    said = protocol._what_she_has_learned_to_say()
    from core.cognition.what_she_already_knows_how_to_say import (
        what_she_already_knows_how_to_say,
    )

    assert bool(said) == bool(what_she_already_knows_how_to_say())
