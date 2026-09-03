"""Every faculty built from the recordings reaches something that runs.

A module nothing imports is a module that never fires, however well it is
tested. This is the check that the wiring exists — not that it is good, which
is what the other tests are for, but that it is there at all.
"""

from __future__ import annotations

import pathlib

#: Everything pulled out of the eleven recordings, and where each one has to
#: reach. A name here with nothing importing it is a faculty she has not got.
FROM_THE_RECORDINGS = (
    "something_she_keeps_true",
    "what_would_have_to_be_true",
    "what_is_still_open",
    "what_she_cannot_afford_to_lose",
    "a_shape_that_makes_it_safe",
    "which_way_to_win",
    "what_the_other_one_is_holding",
    "the_ones_she_reaches_for",
    "where_to_spend_the_next_one",
    "the_furthest_she_has_got",
    "enough_rather_than_most",
    "the_same_thing_turned_around",
    "how_far_to_go_before_looking",
    "what_happens_while_she_acts",
    "what_kind_of_thing_was_said",
    "what_nobody_could_show",
    "when_to_say_it_outright",
    "what_she_has_set_in_motion",
    "the_same_problem_one_size_down",
    "what_having_it_lets_her_do",
    "what_works_against_what",
    "what_it_is_worth_by_the_time_it_comes",
    "getting_ready_for_what_is_coming",
    "telling_one_kind_from_another",
    "two_ways_out",
    "does_this_world_repeat",
    "when_the_move_is_forbidden",
    "what_a_question_gives_away",
    "what_they_will_do_next",
    "what_an_act_costs_beyond_now",
    "marks_she_leaves_behind",
    "a_window_not_a_maximum",
    "when_the_situation_decides",
)


def _who_uses(name: str) -> list[str]:
    used = []
    for path in pathlib.Path("core").rglob("*.py"):
        if path.name == f"{name}.py":
            continue
        try:
            if f"cognition.{name}" in path.read_text(encoding="utf-8"):
                used.append(str(path))
        except (OSError, UnicodeDecodeError):
            continue
    return used


def test_every_one_of_them_is_reached_from_somewhere_that_runs() -> None:
    orphaned = [one for one in FROM_THE_RECORDINGS if not _who_uses(one)]
    assert not orphaned, f"built and never called: {orphaned}"


def test_each_one_exists_as_its_own_thing() -> None:
    missing = [
        one
        for one in FROM_THE_RECORDINGS
        if not pathlib.Path(f"core/cognition/{one}.py").exists()
    ]
    assert not missing, missing


def test_each_one_is_tested_on_its_own() -> None:
    """Wired and untested is as bad as tested and unwired."""
    untested = [
        one
        for one in FROM_THE_RECORDINGS
        if not pathlib.Path(f"tests/test_{one}.py").exists()
        and not any(
            f"cognition.{one}" in path.read_text(encoding="utf-8", errors="ignore")
            for path in pathlib.Path("tests").glob("test_*.py")
        )
    ]
    assert not untested, f"no test names these: {untested}"
