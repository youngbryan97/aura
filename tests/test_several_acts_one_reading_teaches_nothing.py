"""Several acts and one reading cannot say which act did what.

The sitting that first went more than one act between looks carried in a rule
at eighty two per cent and finished with it at one right out of sixty four,
because every pair she learned from named the wrong act — the first of the
run, against a board produced by all of them.
"""

from __future__ import annotations


def test_the_loop_refuses_to_learn_from_a_run_of_several() -> None:
    from core.skills import screen_pursuit

    with open(screen_pursuit.__file__, encoding="utf-8") as handle:
        text = handle.read()
    at = text.index("knows.watched(pending[")
    near = text[at - 900 : at]
    assert 'expected["took"] > 1' in near
    assert "more than one act, one reading" in near


def test_it_is_counted_rather_than_dropped_in_silence() -> None:
    """A pair thrown away is a move she made and learned nothing from, and
    that has to show up somewhere."""
    from core.skills import screen_pursuit

    with open(screen_pursuit.__file__, encoding="utf-8") as handle:
        text = handle.read()
    assert 'dropped["more than one act, one reading"] += 1' in text


def test_a_single_act_still_teaches() -> None:
    """The refusal must not become a way of never learning."""
    from core.skills import screen_pursuit

    with open(screen_pursuit.__file__, encoding="utf-8") as handle:
        text = handle.read()
    at = text.index("knows.watched(pending[")
    near = text[at - 300 : at]
    assert "elif _in_the_same_grid(" in near, "one act still goes in"
