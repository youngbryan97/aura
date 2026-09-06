"""Every durable field has one authority, or is counted as not having one.

98 leaf fields across eight organs. Generative Agents avoids this question by
having a persona state small enough that one object owns each fact; Aura's is
not, so the question has to be answered rather than avoided.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.state.who_owns_each_field import (
    THE_DECLARED_OWNERS,
    every_field,
    how_ownership_stands,
    the_owner_of,
    what_nobody_owns,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "config" / "field_ownership_baseline.json"


def _baseline() -> dict:
    return json.loads(BASELINE.read_text("utf-8"))


def test_the_fields_are_found_at_all():
    fields = every_field()
    assert len(fields) > 50
    assert "cognition.working_memory" in fields
    assert "identity.name" in fields


def test_nothing_is_counted_twice():
    stands = how_ownership_stands()
    buckets = (
        stands["owned_by_a_phase"]
        + stands["owned_by_one_module"]
        + stands["owned_by_declaration"]
        + stands["written_by_several_with_no_authority"]
        + stands["written_by_nothing_that_assigns"]
    )
    assert len(buckets) == len(set(buckets))
    assert len(buckets) == stands["nested"]


def test_the_unclassified_count_only_goes_down():
    """The ratchet. Raising it needs a reason, and there has not been one."""
    unowned = what_nobody_owns()
    allowed = _baseline()["count"]
    assert len(unowned) <= allowed, (
        f"{len(unowned)} fields have no authority, baseline is {allowed}. "
        f"New ones: {sorted(set(unowned) - set(_baseline()['unclassified']))}"
    )


def test_the_baseline_names_the_fields_and_not_only_the_number():
    """A count with no names cannot tell you what got worse."""
    baseline = _baseline()
    assert baseline["count"] == len(baseline["unclassified"])
    assert all("." in one for one in baseline["unclassified"])


def test_no_field_becomes_unowned_that_was_owned():
    was = set(_baseline()["unclassified"])
    now = set(what_nobody_owns())
    assert not (now - was), f"newly unowned: {sorted(now - was)}"


# ------------------------------------------------------- the three answers


def test_a_phase_declared_field_names_its_phase():
    owner = the_owner_of("cognition.working_memory")
    assert owner is not None
    assert owner["how"] == "declared in the phase contract"


def test_a_field_written_by_one_module_names_that_module():
    stands = how_ownership_stands()
    assert stands["owned_by_one_module"], "nothing is owned by exactly one module"
    one = stands["owned_by_one_module"][0]
    owner = the_owner_of(one)
    assert owner["how"] == "the only module that assigns it"
    assert owner["owner"].endswith(".py")


def test_a_declared_owner_says_what_the_others_are_doing():
    """One owner is only true if the rest are something other than writers."""
    for path, row in THE_DECLARED_OWNERS.items():
        assert row["owner"].startswith("core/"), path
        assert row["others"], f"{path} does not say what the other writers do"


def test_an_unowned_field_says_so_rather_than_naming_a_guess():
    for path in what_nobody_owns():
        assert the_owner_of(path) is None, f"{path} is in both answers"


def test_a_field_that_does_not_exist_has_no_owner():
    assert the_owner_of("cognition.a_field_nobody_declared") is None


# ------------------------------------------------------------ the limit


def test_the_top_level_scalars_are_named_out_of_reach_rather_than_guessed():
    """`x.version = 1` matches a dozen unrelated dataclasses.

    A count built on that would be confidently wrong, so the eighteen
    top-level scalars are excluded and said to be excluded.
    """
    stands = how_ownership_stands()
    assert stands["out_of_reach"] == stands["fields"] - stands["nested"]
    assert stands["out_of_reach"] > 0


@pytest.mark.parametrize(
    "path", ["motivation.latent_interests", "world.known_entities"]
)
def test_the_known_unowned_are_mutated_in_place_not_missing(path):
    """A field mutated through a container reference has no assignment site.

    Which is the finding, not a gap in the measurement: a field written only
    by `.update()` on a reference somebody else holds has no owner in any
    sense a reader could act on.
    """
    assert path in what_nobody_owns()
