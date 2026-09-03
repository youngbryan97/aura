"""The wiring: what works against a KIND of situation, and whether the world
is the sort where a fact about a place is worth keeping.

She learned which acts work HERE, which dies with the place, and which acts
work generally, which averages over places with nothing in common. Neither
could say the thing that is true: this act works on that kind of thing.
"""

from __future__ import annotations


def test_the_loop_learns_against_the_kind_and_not_only_the_place() -> None:
    from core.skills import screen_pursuit

    with open(screen_pursuit.__file__, encoding="utf-8") as handle:
        text = handle.read()
    assert "beats.it_went(" in text
    at = text.index("beats.it_went(")
    near = text[at : at + 400]
    assert "against=kind" in near
    assert "well=bool(attempt.progressed)" in near


def test_it_is_only_leaned_on_where_the_world_repeats() -> None:
    """Where it is dealt fresh, a fact about one place is noise she would be
    storing at her own expense."""
    from core.skills import screen_pursuit

    with open(screen_pursuit.__file__, encoding="utf-8") as handle:
        text = handle.read()
    at = text.index("beats.in_order(")
    near = text[at - 500 : at]
    assert "repeats.worth_remembering_places()" in near


def test_both_are_kept_between_sittings() -> None:
    from core.skills import screen_pursuit

    with open(screen_pursuit.__file__, encoding="utf-8") as handle:
        text = handle.read()
    assert '"beats": beats.as_memory()' in text
    assert '"repeats": repeats.as_memory()' in text
    assert 'knew.get("beats")' in text
    assert 'knew.get("repeats")' in text


def test_an_act_never_tried_against_this_kind_does_not_get_reordered() -> None:
    """An untried act is an experiment, not a bad act — so it keeps its place
    rather than being sorted to the back on no evidence."""
    from core.skills import screen_pursuit

    with open(screen_pursuit.__file__, encoding="utf-8") as handle:
        text = handle.read()
    at = text.index("beats.in_order(")
    near = text[at : at + 400]
    assert "if tried" in near
