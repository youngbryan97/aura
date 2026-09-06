"""Whether two snapshots mean the same thing, and where they do not.

Aura compared snapshots by digest, which answers a different question: a
digest says the bytes differ, and restore, compaction, migration and shadow
tests all need to know whether the MEANING differs.

A restore that rewrote a float's last bit, a compaction that dropped a field
nothing reads, a migration that renamed a key — a digest calls all three a
failure, and a reviewer then learns to ignore it.
"""
from __future__ import annotations

import math

import pytest

from core.state.are_these_the_same import HowClose, are_these_the_same


# ------------------------------------------------------------ the verdict


def test_two_identical_snapshots_are_the_same():
    said = are_these_the_same({"a": 1}, {"a": 1})
    assert said
    assert said.differences == []


def test_a_difference_says_where_it_is():
    """A verdict with no location cannot be acted on."""
    said = are_these_the_same(
        {"identity": {"name": "Aura"}}, {"identity": {"name": "Aura II"}}
    )
    assert not said
    assert said.differences[0].where == "identity.name"
    assert said.differences[0].left == "Aura"
    assert said.differences[0].right == "Aura II"


def test_it_walks_both_sides():
    """One that walks one side calls a truncated snapshot identical."""
    said = are_these_the_same({"a": 1}, {"a": 1, "b": 2})
    assert not said
    assert said.differences[0].where == "b"
    assert said.differences[0].why == "on one side only"


def test_the_verdict_reads_as_a_boolean():
    assert bool(are_these_the_same({"a": 1}, {"a": 1})) is True
    assert bool(are_these_the_same({"a": 1}, {"a": 2})) is False


# --------------------------------------------------------- the tolerances


def test_floats_within_a_declared_allowance_are_the_same():
    apart = HowClose(floats_within=1e-6)
    assert are_these_the_same({"phi": 0.5}, {"phi": 0.5000001}, how_close=apart)
    assert not are_these_the_same({"phi": 0.5}, {"phi": 0.51}, how_close=apart)


def test_no_allowance_is_the_default():
    """Guessing that two floats are close enough is how a comparison stops."""
    assert HowClose().floats_within == 0.0
    assert not are_these_the_same({"phi": 0.5}, {"phi": 0.5000001})


def test_the_difference_says_how_far_apart_they_were():
    said = are_these_the_same({"phi": 0.5}, {"phi": 0.7})
    assert "differ by 0.2" in said.differences[0].why


def test_an_ignored_key_is_not_compared():
    ignoring = HowClose(ignore=frozenset({"at", "run_id"}))
    assert are_these_the_same(
        {"at": 1.0, "run_id": "a", "kept": 1},
        {"at": 999.0, "run_id": "b", "kept": 1},
        how_close=ignoring,
    )


def test_a_list_whose_order_was_declared_not_to_matter():
    loose = HowClose(order_does_not_matter=frozenset({"tags"}))
    assert are_these_the_same(
        {"tags": ["b", "a"]}, {"tags": ["a", "b"]}, how_close=loose
    )
    said = are_these_the_same(
        {"tags": ["a"]}, {"tags": ["b"]}, how_close=loose
    )
    assert not said
    assert said.differences[0].why == "different members, order aside"


def test_order_matters_by_default():
    assert not are_these_the_same({"tags": ["b", "a"]}, {"tags": ["a", "b"]})


def test_a_migration_may_be_allowed_to_add_fields():
    adding = HowClose(missing_is_a_difference=False)
    assert are_these_the_same({"a": 1}, {"a": 1, "new": 2}, how_close=adding)
    assert not are_these_the_same(
        {"a": 1}, {"a": 2, "new": 2}, how_close=adding
    )


# ------------------------------------------------------------- the edges


def test_a_list_of_a_different_length_says_so():
    said = are_these_the_same({"a": [1, 2]}, {"a": [1, 2, 3]})
    assert said.differences[0].why == "different lengths"
    assert (said.differences[0].left, said.differences[0].right) == (2, 3)


def test_a_boolean_is_not_a_number():
    """True == 1 in Python, and a snapshot where a flag became a count is not
    the same snapshot."""
    said = are_these_the_same({"ok": True}, {"ok": 1})
    assert not said
    assert said.differences[0].why == "one is a boolean"


def test_two_nans_are_the_same_place():
    assert are_these_the_same({"x": math.nan}, {"x": math.nan})


def test_nested_lists_of_dicts_are_addressed_precisely():
    said = are_these_the_same(
        {"rows": [{"a": 1}, {"a": 2}]}, {"rows": [{"a": 1}, {"a": 3}]}
    )
    assert said.differences[0].where == "rows[1].a"


def test_it_counts_what_it_looked_at():
    said = are_these_the_same({"a": 1, "b": {"c": 2}}, {"a": 1, "b": {"c": 2}})
    assert said.compared == 2


def test_the_verdict_reads_back_as_data():
    import json

    said = are_these_the_same({"a": 1}, {"a": 2})
    back = json.loads(json.dumps(said.to_dict()))
    assert back["same"] is False
    assert back["differences"][0]["where"] == "a"


@pytest.mark.parametrize(
    "left,right",
    [({"a": 1}, {"a": "1"}), ({"a": None}, {"a": 0}), ({"a": []}, {"a": {}})],
)
def test_a_change_of_kind_is_a_difference(left, right):
    assert not are_these_the_same(left, right)
