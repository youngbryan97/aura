"""A shim with no removal date is a second permanent API."""
from __future__ import annotations

import warnings

import pytest

from core.runtime.what_is_on_its_way_out import (
    THE_SHIMS,
    going_away,
    how_the_deprecations_stand,
    overdue,
    prose_markers_in_the_tree,
    shims_with_no_date,
)


def test_every_file_that_admits_to_a_shim_has_been_given_terms() -> None:
    """The ratchet. A new shim shows up here rather than as a second API."""
    undated = shims_with_no_date()
    assert undated == (), (
        f"{len(undated)} file(s) declare a compatibility surface with no "
        f"removal date: {list(undated)}. Add it to THE_SHIMS, or remove it."
    )


def test_the_tree_really_does_carry_shims() -> None:
    """A gate that matched nothing would report green forever."""
    assert len(prose_markers_in_the_tree()) >= 20


def test_every_declared_shim_names_a_replacement_and_a_reason() -> None:
    for shim in THE_SHIMS:
        assert shim.instead.strip(), f"{shim.where}::{shim.name} has no replacement"
        assert shim.why.strip(), f"{shim.where}::{shim.name} has no reason"
        assert shim.since < shim.remove_after, f"{shim.name} goes before it arrives"


def test_nothing_is_past_its_date() -> None:
    late = overdue()
    assert late == (), (
        "past their removal date: "
        + ", ".join(f"{s.where}::{s.name} (due {s.remove_after})" for s in late)
        + ". Remove it, or argue for a new date."
    )


def test_a_date_that_has_passed_is_reported_as_overdue() -> None:
    """The check has to be able to fire, or it says nothing every day."""
    assert overdue(today="2099-01-02"), "no date in the table is ever reachable"


def test_a_marked_callable_warns_once_per_call_site_not_once_per_call() -> None:
    @going_away(
        since="2026-01-01",
        remove_after="2027-01-01",
        instead="the_new_one",
        why="it was the first attempt",
    )
    def old_one(x: int) -> int:
        return x * 2

    with warnings.catch_warnings(record=True) as said:
        warnings.simplefilter("always")
        assert [old_one(n) for n in range(5)] == [0, 2, 4, 6, 8]

    assert len(said) == 1, "a shim in a loop must not fill the log"
    assert issubclass(said[0].category, DeprecationWarning)
    assert "the_new_one" in str(said[0].message)
    assert "2027-01-01" in str(said[0].message)


def test_the_marker_keeps_the_terms_on_the_function() -> None:
    @going_away(since="2026-01-01", remove_after="2027-01-01", instead="x")
    def old() -> None: ...

    terms = old.__aura_going_away__
    assert terms.instead == "x"
    assert terms.remove_after == "2027-01-01"


def test_a_marked_callable_still_does_what_it_did() -> None:
    @going_away(since="2026-01-01", remove_after="2027-01-01", instead="x")
    def add(a: int, b: int = 3) -> int:
        """Adds."""
        return a + b

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert add(1) == 4
        assert add(1, b=10) == 11
    assert add.__doc__ == "Adds."


def test_the_report_says_what_is_going_and_when() -> None:
    seen = how_the_deprecations_stand()
    assert seen["declared"] == len(THE_SHIMS)
    assert seen["with_no_date"] == []
    assert seen["overdue"] == []
    assert seen["earliest_removal"]
