"""Revising the self-narrative when enough has happened, rather than every twenty turns.

The phase has said since it was written that it "revisits the self-narrative
when enough has happened to justify it". What it did was increment on
`state.version % 20`, so twenty turns of silence and twenty turns that changed
everything got the same answer.

The records say where to read what has happened. "Remember the Time" is fifteen
questions and no statements — the past being brought back is what the
relationship is made of. Oddisee's "you were there from the start" is the same
move pointed inward, where the recollection is not evidence about the
self-model, it is the self-model.
"""

from __future__ import annotations

import pytest

from core.self.revision import (
    MIN_HISTORY,
    Revision,
    RevisionLedger,
    unaccounted_share,
)

STORY = "She builds instruments and measures what she builds and keeps records of it."
ORDINARY = ["she measured the records", "building instruments again", "records of the build"]


def _settled() -> RevisionLedger:
    led = RevisionLedger()
    for _ in range(MIN_HISTORY + 3):
        led.read(ORDINARY, STORY)
    return led


def test_what_the_story_already_holds_does_not_count() -> None:
    assert unaccounted_share(["instruments and records"], STORY) == pytest.approx(0.0)


def test_what_it_does_not_hold_counts() -> None:
    share = unaccounted_share(["saxophone phrasing changed everything"], STORY)
    assert share is not None and share > 0.9


def test_an_ordinary_recollection_is_not_worth_revisiting() -> None:
    reading = _settled().read(ORDINARY, STORY)
    assert reading.measured
    assert not reading.worth_revisiting


def test_a_recollection_the_story_has_no_room_for_is() -> None:
    reading = _settled().read(
        ["a concert in Brooklyn changed how she hears saxophone phrasing entirely"], STORY
    )
    assert reading.measured
    assert reading.worth_revisiting
    assert "does not hold" in reading.why


def test_a_life_where_recall_always_brings_new_things_does_not_revise_every_turn() -> None:
    """The bar is her own variation, so a turbulent life is not a standing alarm."""
    led = RevisionLedger()
    varied = [
        ["saxophone phrasing"], ["harbour weather"], ["a broken compiler"],
        ["someone's birthday"], ["a train timetable"], ["the price of coffee"],
        ["a new proof technique"], ["a film about divers"],
    ]
    for items in varied:
        led.read(items, STORY)
    reading = led.read(["another unrelated thing entirely"], STORY)
    assert reading.measured
    assert not reading.worth_revisiting


def test_nothing_recalled_is_not_a_story_that_accounts_for_everything() -> None:
    assert unaccounted_share([], STORY) is None
    reading = _settled().read([], STORY)
    assert not reading.measured
    assert not reading.worth_revisiting
    assert "nothing came back" in reading.why


def test_no_story_yet_is_not_a_story_that_accounts_for_everything() -> None:
    assert unaccounted_share(["anything at all"], "") is None


def test_with_too_few_readings_it_says_so() -> None:
    led = RevisionLedger()
    led.read(ORDINARY, STORY)
    reading = led.read(["saxophone phrasing changed everything"], STORY)
    assert not reading.measured
    assert not reading.worth_revisiting
    assert "readings needed" in reading.why


def test_a_history_that_never_varied_still_gives_an_answer() -> None:
    """Recall has never brought back more than this, and now it has."""
    reading = _settled().read(["entirely unrelated saxophone concert Brooklyn"], STORY)
    assert reading.measured
    assert reading.worth_revisiting


def test_the_phase_reads_the_recollection_rather_than_a_counter() -> None:
    """Checked against the code rather than the prose.

    The comment that explains the old trigger names it, which is what a comment
    is for, and a scan over the text would read that as the behaviour.
    """
    import ast
    import inspect

    from core.phases import identity_reflection

    tree = ast.parse(inspect.getsource(identity_reflection))
    counters = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Mod)
        and isinstance(node.left, ast.Attribute)
        and node.left.attr == "version"
    ]
    assert not counters, "the narrative still turns on a counter"
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_worth_revisiting"
    ]
    assert calls, "the phase does not ask whether anything arrived"


def test_the_reading_serialises_what_a_reader_needs() -> None:
    row = _settled().read(ORDINARY, STORY).as_dict()
    for key in ("unaccounted", "usual", "spread", "z", "worth_revisiting",
                "readings", "measured", "why"):
        assert key in row, key


def test_an_empty_reading_claims_nothing() -> None:
    assert not Revision().measured
    assert not Revision().worth_revisiting
