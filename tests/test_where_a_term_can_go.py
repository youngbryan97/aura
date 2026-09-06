"""Seven destinations, and whether each one really is a destination.

`WHERE_A_TERM_CAN_GO` names seven places a term can be installed, and the
comment above it makes a specific claim about why that set is the set:

    Each has an installer, a lesion, and a persistence path, and those three
    are what makes something a destination rather than a wish.

That is the right criterion and nothing checked it. A destination with an
installer and no persistence path is a change that does not survive the
process; one with no lesion is a change nobody can measure the absence of.
Either way the set is smaller than it says.
"""
from __future__ import annotations

import pytest

from core.cognition.sequence_induction import _register_what_she_could_do
from core.cognition.what_she_could_do_next import WHERE_A_TERM_CAN_GO
from core.cognition.where_a_term_can_go import (
    THE_DESTINATIONS,
    _resolves,
    what_is_not_really_a_destination,
    where_a_term_can_go,
)


@pytest.fixture(scope="module")
def destinations():
    _register_what_she_could_do()
    return where_a_term_can_go()


def test_every_declared_destination_is_one_of_the_seven():
    assert set(THE_DESTINATIONS) == set(WHERE_A_TERM_CAN_GO)


def test_every_destination_has_all_three_parts(destinations):
    """The claim in the comment, checked."""
    assert what_is_not_really_a_destination(destinations["each"]) == []


@pytest.mark.parametrize("name", sorted(THE_DESTINATIONS))
def test_each_part_resolves_to_something_that_is_there(name):
    for key, where in THE_DESTINATIONS[name].items():
        assert _resolves(where), f"{name}: {key} -> {where} does not resolve"


def test_a_name_that_is_not_there_does_not_resolve():
    """The gate has to be able to fail.

    It did fail: the proposer's installer was first written as a function in
    primitive_invention that does not exist, and this said so — which is why
    a resolved name beats a comment.
    """
    assert not _resolves("core.cognition.primitive_invention:a_thing_nobody_wrote")
    assert not _resolves("core.nothing.at.all:anything")


def test_every_destination_has_at_least_one_action_installing_into_it(destinations):
    """A slot nothing fills is not a destination either."""
    for name, parts in destinations["each"].items():
        assert parts["actions"] > 0, name


def test_the_report_counts_the_actions_it_found(destinations):
    assert destinations["destinations"] == len(WHERE_A_TERM_CAN_GO)
    assert destinations["actions"] >= destinations["destinations"]


def test_a_missing_declaration_is_reported_rather_than_passed():
    said = {"somewhere new": {"declared": False, "actions": 0}}
    wrong = what_is_not_really_a_destination(said)
    assert wrong == ["somewhere new: nothing declares its three parts"]


def test_a_destination_with_no_persistence_is_reported():
    said = {
        "somewhere": {
            "declared": True, "actions": 1,
            "installs": True, "lesion": True, "keeps": False,
        }
    }
    assert what_is_not_really_a_destination(said) == ["somewhere: its keeps does not resolve"]
