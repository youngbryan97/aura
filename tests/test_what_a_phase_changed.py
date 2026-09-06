"""Working memory changed at the boundary, not in the middle of a phase.

Soar came out above Aura on engineering maturity after its reputation was
stripped away, and the reason given was authority: one agent state, one
decision cycle over it, and working-memory changes buffered and committed at
defined phase boundaries with explicit addition, removal and refcount
semantics.

The refcount is the part a list of dicts silently does not have. `remove()`
on a list takes the first match away from whoever put it there.
"""
from __future__ import annotations

import pytest

from core.state.aura_state import AuraState
from core.state.one_working_memory import the_working_memory
from core.state.what_a_phase_changed import (
    at_the_boundary,
    forget_everything,
    how_the_boundaries_have_gone,
    the_boundary_for,
)


@pytest.fixture(autouse=True)
def _clean():
    forget_everything()
    yield
    forget_everything()


@pytest.fixture
def state():
    return AuraState()


AN_ITEM = {"role": "user", "content": "hello"}
ANOTHER = {"role": "thought", "content": "she is thinking"}


# ------------------------------------------------------------ the boundary


def test_nothing_changes_until_the_boundary(state):
    """A phase must not see the one before it half-finished."""
    with at_the_boundary(state, "MemoryRetrievalPhase") as boundary:
        boundary.add(AN_ITEM)
        assert the_working_memory(state) == []
    assert the_working_memory(state) == [AN_ITEM]


def test_a_phase_that_raises_still_commits_what_it_decided(state):
    """Otherwise the state depends on where the exception happened."""
    with pytest.raises(ZeroDivisionError):
        with at_the_boundary(state, "AffectUpdatePhase") as boundary:
            boundary.add(AN_ITEM)
            raise ZeroDivisionError
    assert the_working_memory(state) == [AN_ITEM]


def test_two_phases_commit_in_the_order_they_ran(state):
    with at_the_boundary(state, "first") as boundary:
        boundary.add(AN_ITEM)
    with at_the_boundary(state, "second") as boundary:
        boundary.add(ANOTHER)
    assert the_working_memory(state) == [AN_ITEM, ANOTHER]


def test_a_boundary_that_changed_nothing_leaves_the_memory_alone(state):
    the_working_memory(state).append(AN_ITEM)
    with at_the_boundary(state, "quiet"):
        pass
    assert the_working_memory(state) == [AN_ITEM]


# --------------------------------------------------------------- refcounts


def test_one_item_added_by_two_phases_appears_once(state):
    for phase in ("MemoryRetrievalPhase", "AffectUpdatePhase"):
        with at_the_boundary(state, phase) as boundary:
            boundary.add(AN_ITEM)
    assert the_working_memory(state) == [AN_ITEM]


def test_one_remover_does_not_take_away_the_others_reason(state):
    """The semantics a list silently does not have."""
    for phase in ("MemoryRetrievalPhase", "AffectUpdatePhase"):
        with at_the_boundary(state, phase) as boundary:
            boundary.add(AN_ITEM)

    with at_the_boundary(state, "AffectUpdatePhase") as boundary:
        boundary.remove(AN_ITEM)
    assert the_working_memory(state) == [AN_ITEM]

    with at_the_boundary(state, "MemoryRetrievalPhase") as boundary:
        boundary.remove(AN_ITEM)
    assert the_working_memory(state) == []


def test_removing_something_nobody_added_is_counted_not_raised(state):
    """A phase that cleans up defensively is not a defect."""
    with at_the_boundary(state, "tidy") as boundary:
        boundary.remove(AN_ITEM)
    assert the_working_memory(state) == []


def test_two_equal_items_refcount_together(state):
    """Equal content is one item, whoever built the dict."""
    with at_the_boundary(state, "one") as boundary:
        boundary.add({"role": "user", "content": "hello"})
    with at_the_boundary(state, "two") as boundary:
        boundary.add({"content": "hello", "role": "user"})
    assert len(the_working_memory(state)) == 1


def test_an_item_that_will_not_serialise_still_refcounts(state):
    class Odd:
        def __repr__(self) -> str:
            return "<the same odd thing>"

    odd = Odd()
    with at_the_boundary(state, "one") as boundary:
        boundary.add(odd)
    with at_the_boundary(state, "two") as boundary:
        boundary.add(odd)
    assert len(the_working_memory(state)) == 1


# ----------------------------------------------------------------- reading


def test_the_report_says_what_each_phase_committed(state):
    with at_the_boundary(state, "MemoryRetrievalPhase") as boundary:
        boundary.add(AN_ITEM)
        boundary.add(ANOTHER)
    with at_the_boundary(state, "MemoryRetrievalPhase") as boundary:
        boundary.remove(ANOTHER)

    went = how_the_boundaries_have_gone()["by_phase"]["MemoryRetrievalPhase"]
    assert went["additions"] == 2
    assert went["removals"] == 1


def test_the_report_names_who_is_holding_a_shared_item(state):
    """A stuck refcount has to be answerable, not a mystery."""
    for phase in ("MemoryRetrievalPhase", "AffectUpdatePhase"):
        with at_the_boundary(state, phase) as boundary:
            boundary.add(AN_ITEM)

    shared = how_the_boundaries_have_gone()["held_by_more_than_one"]
    assert len(shared) == 1
    assert shared[0]["count"] == 2
    assert set(shared[0]["by"]) == {"MemoryRetrievalPhase", "AffectUpdatePhase"}


def test_a_buffer_counts_what_it_holds_before_it_commits():
    boundary = the_boundary_for("a phase")
    boundary.add(AN_ITEM)
    boundary.add(ANOTHER)
    boundary.remove(AN_ITEM)
    assert boundary.additions == 2
    assert boundary.removals == 1


def test_the_report_is_in_the_health_report():
    from core.runtime.health_contract import runtime_health_report

    block = runtime_health_report()["integrity"]["what_a_phase_changed"]
    assert set(block) >= {"by_phase", "items_held"}
