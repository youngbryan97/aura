"""Three-way merge, and the case last-write-wins cannot see.

Letta's memory is a git-backed filesystem, so it gets an explicit merge and
conflict vocabulary for free. The review said Aura's memory conflicts vary by
backend and asked for three-way merge with conflict objects and evidence-aware
resolution — and that a divergent identity edit is never silently overwritten.

Three-way is the part that matters. Two versions cannot tell "you changed this
and I did not" from "we both changed it differently", so last-write-wins
discards one of them without knowing it did.
"""
from __future__ import annotations

import pytest

from core.memory.when_two_memories_disagree import (
    THE_IDENTITY_FIELDS,
    merge_three_ways,
)

BASE = {"tone": "warm", "city": "London", "turns": 4}


# ------------------------------------------------------ what merges cleanly


def test_a_change_only_one_side_made_is_taken():
    merged = merge_three_ways(BASE, {**BASE, "city": "Lisbon"}, dict(BASE))
    assert merged.clean
    assert merged.merged["city"] == "Lisbon"
    assert merged.took_mine == ["city"]


def test_a_change_only_the_other_side_made_is_taken():
    merged = merge_three_ways(BASE, dict(BASE), {**BASE, "tone": "dry"})
    assert merged.clean
    assert merged.merged["tone"] == "dry"
    assert merged.took_theirs == ["tone"]


def test_both_sides_changing_different_fields_is_not_a_conflict():
    """The case last-write-wins gets wrong by throwing one side away."""
    merged = merge_three_ways(
        BASE, {**BASE, "city": "Lisbon"}, {**BASE, "tone": "dry"}
    )
    assert merged.clean
    assert merged.merged["city"] == "Lisbon"
    assert merged.merged["tone"] == "dry"


def test_both_sides_making_the_same_change_is_agreement():
    merged = merge_three_ways(
        BASE, {**BASE, "city": "Lisbon"}, {**BASE, "city": "Lisbon"}
    )
    assert merged.clean
    assert merged.agreed == ["city"]
    assert merged.merged["city"] == "Lisbon"


def test_both_sides_adding_the_same_new_key_is_the_same_thing():
    merged = merge_three_ways(BASE, {**BASE, "pet": "cat"}, {**BASE, "pet": "cat"})
    assert merged.clean
    assert merged.merged["pet"] == "cat"


def test_a_field_neither_side_touched_comes_through():
    merged = merge_three_ways(BASE, dict(BASE), dict(BASE))
    assert merged.merged == BASE
    assert merged.clean


def test_one_side_deleting_a_field_removes_it():
    without = {name: value for name, value in BASE.items() if name != "city"}
    merged = merge_three_ways(BASE, without, dict(BASE))
    assert "city" not in merged.merged
    assert merged.clean


# ----------------------------------------------------------- what conflicts


def test_both_sides_changing_one_field_differently_is_a_conflict():
    merged = merge_three_ways(
        BASE, {**BASE, "city": "Lisbon"}, {**BASE, "city": "Berlin"}
    )
    assert not merged.clean
    conflict = merged.conflicts[0]
    assert conflict.field == "city"
    assert (conflict.ancestor, conflict.mine, conflict.theirs) == (
        "London", "Lisbon", "Berlin"
    )
    assert conflict.why == "both sides changed it differently"


def test_a_conflicted_field_is_absent_from_the_merged_result():
    """Writing a guess there is how a caller ends up applying the guess."""
    merged = merge_three_ways(
        BASE, {**BASE, "city": "Lisbon"}, {**BASE, "city": "Berlin"}
    )
    assert "city" not in merged.merged
    assert merged.merged["tone"] == "warm"


def test_one_side_deleting_what_the_other_changed_is_a_conflict():
    without = {name: value for name, value in BASE.items() if name != "city"}
    merged = merge_three_ways(BASE, without, {**BASE, "city": "Berlin"})
    assert not merged.clean
    assert merged.conflicts[0].mine is None


def test_both_sides_adding_the_same_key_with_different_values_conflicts():
    merged = merge_three_ways(BASE, {**BASE, "pet": "cat"}, {**BASE, "pet": "dog"})
    assert not merged.clean
    assert merged.conflicts[0].ancestor is None


# -------------------------------------------------------- identity is held


@pytest.mark.parametrize("guarded", sorted(THE_IDENTITY_FIELDS))
def test_a_one_sided_identity_change_is_offered_rather_than_applied(guarded):
    """A merge that is usually right is not good enough for who she is."""
    base = {guarded: "as it was"}
    merged = merge_three_ways(base, {guarded: "changed"}, dict(base))

    assert not merged.clean
    assert merged.conflicts[0].field == guarded
    assert "identity field" in merged.conflicts[0].why
    assert guarded not in merged.merged


def test_both_sides_agreeing_on_an_identity_change_is_still_agreement():
    """Guarded means never silently overwritten, not never changed."""
    base = {"name": "Aura"}
    merged = merge_three_ways(base, {"name": "Aura II"}, {"name": "Aura II"})
    assert merged.clean
    assert merged.merged["name"] == "Aura II"


def test_the_guarded_set_can_be_named_by_the_caller():
    merged = merge_three_ways(
        {"x": 1}, {"x": 2}, {"x": 1}, identity_fields=frozenset({"x"})
    )
    assert not merged.clean


def test_nothing_is_guarded_when_the_caller_says_so():
    merged = merge_three_ways(
        {"name": "Aura"}, {"name": "Aura II"}, {"name": "Aura"},
        identity_fields=frozenset(),
    )
    assert merged.clean


# ------------------------------------------------------------- evidence


def test_a_conflict_carries_the_reason_each_side_offers():
    """A conflict with no reasons can only be settled by whoever is asked last."""
    merged = merge_three_ways(
        BASE,
        {**BASE, "city": "Lisbon"},
        {**BASE, "city": "Berlin"},
        my_evidence={"city": "she said so on turn 40"},
        their_evidence={"city": "a calendar entry"},
    )
    conflict = merged.conflicts[0]
    assert conflict.my_evidence == "she said so on turn 40"
    assert conflict.their_evidence == "a calendar entry"


def test_a_conflict_with_no_evidence_says_none_rather_than_inventing_one():
    merged = merge_three_ways(
        BASE, {**BASE, "city": "Lisbon"}, {**BASE, "city": "Berlin"}
    )
    assert merged.conflicts[0].my_evidence is None


def test_the_merge_reads_back_as_data():
    merged = merge_three_ways(
        BASE, {**BASE, "city": "Lisbon"}, {**BASE, "city": "Berlin"}
    )
    import json

    back = json.loads(json.dumps(merged.to_dict()))
    assert back["clean"] is False
    assert back["conflicts"][0]["field"] == "city"
