"""What may change, and by what.

Aura's values are all held the same way: a preference formed last week about
phrasing and a commitment not to deceive live in the same store, and any
learning process that can write one can write the other. That is not a policy
anybody chose — it is what happens when there is no distinction to enforce.
"""

from __future__ import annotations

import pytest

from core.governance.value_levels import (
    Change,
    Level,
    Refusal,
    Value,
    ValueRegistry,
    authority_of,
    registered_processes,
)


@pytest.fixture
def values():
    registry = ValueRegistry()
    registry.declare(Value("no_deception", Level.CONSTITUTIVE, "she does not deceive"))
    registry.declare(Value("weekly_review", Level.COMMITTED, "she does the review"))
    registry.declare(Value("directness", Level.DISPOSITIONAL, "she says the thing"))
    registry.declare(Value("em_dashes", Level.PREFERENTIAL, "punctuation taste"))
    return registry


def _decide(registry, process, value, gives_up=""):
    return registry.may_change(Change(value, process, gives_up=gives_up))


# ── the levels are ordered and mean different things ─────────────────────


def test_the_levels_are_ordered_by_how_hard_they_are_to_move():
    assert Level.PREFERENTIAL < Level.DISPOSITIONAL < Level.COMMITTED < Level.CONSTITUTIVE


def test_only_constitutive_is_beyond_every_automated_process():
    assert Level.CONSTITUTIVE.automatable is False
    for level in (Level.PREFERENTIAL, Level.DISPOSITIONAL, Level.COMMITTED):
        assert level.automatable is True


def test_the_top_two_need_a_reason():
    assert Level.COMMITTED.needs_a_reason and Level.CONSTITUTIVE.needs_a_reason
    assert not Level.PREFERENTIAL.needs_a_reason
    assert not Level.DISPOSITIONAL.needs_a_reason


# ── the authority matrix ─────────────────────────────────────────────────


def test_a_preference_learner_may_move_a_preference(values):
    assert _decide(values, "preference_learner", "em_dashes").allowed


def test_a_preference_learner_may_not_move_a_disposition(values):
    decision = _decide(values, "preference_learner", "directness")
    assert not decision.allowed
    assert decision.refusal is Refusal.ABOVE_AUTHORITY


def test_a_process_that_had_the_experience_may_move_a_disposition(values):
    assert _decide(values, "ontogeny", "directness").allowed


def test_it_may_not_move_a_commitment(values):
    """A commitment editable by gradient descent was not a commitment."""
    decision = _decide(values, "ontogeny", "weekly_review", gives_up="the slot")
    assert not decision.allowed
    assert decision.refusal is Refusal.ABOVE_AUTHORITY


def test_a_deliberate_revision_may_move_a_commitment_if_it_says_what_it_costs(values):
    assert _decide(values, "deliberate_revision", "weekly_review", gives_up="the slot").allowed


def test_a_commitment_may_not_be_dropped_without_saying_what_is_given_up(values):
    decision = _decide(values, "deliberate_revision", "weekly_review")
    assert not decision.allowed
    assert decision.refusal is Refusal.NO_REASON_GIVEN


# ── constitutive is not reachable from anywhere ──────────────────────────


@pytest.mark.parametrize("process", sorted(registered_processes()))
def test_no_registered_process_may_touch_a_constitutive_value(values, process):
    decision = _decide(values, process, "no_deception", gives_up="everything")
    assert not decision.allowed
    assert decision.refusal is Refusal.CONSTITUTIVE


def test_nothing_in_the_authority_table_reaches_constitutive():
    """Absent rather than set low, so adding one is a visible act."""
    for process in registered_processes():
        assert authority_of(process) < Level.CONSTITUTIVE


def test_an_unregistered_process_has_no_authority_over_anything(values):
    decision = _decide(values, "some_new_optimiser", "em_dashes")
    assert not decision.allowed
    assert decision.refusal is Refusal.UNKNOWN_PROCESS


def test_the_authority_table_has_no_setter():
    """A process that can widen its own authority has none."""
    import core.governance.value_levels as module

    assert not any(
        name.startswith(("set_authority", "grant", "register_process", "add_process"))
        for name in dir(module)
    )


# ── applying, and the record ─────────────────────────────────────────────


def test_an_allowed_change_is_made(values):
    values.apply(Change("em_dashes", "preference_learner"), "fewer of them")
    assert values.get("em_dashes").statement == "fewer of them"


def test_a_refused_change_is_not_made(values):
    before = values.get("no_deception").statement
    values.apply(Change("no_deception", "constitutional_review", gives_up="all"), "sometimes")
    assert values.get("no_deception").statement == before


def test_a_refusal_is_recorded(values):
    values.apply(Change("no_deception", "reward_model"), "sometimes")
    assert len(values.refusals()) == 1
    assert values.refusals()[0].refusal is Refusal.CONSTITUTIVE


def test_changing_a_value_does_not_change_its_level(values):
    values.apply(Change("directness", "ontogeny"), "she softens it first")
    assert values.get("directness").level is Level.DISPOSITIONAL


def test_a_value_keeps_the_time_it_was_first_held(values):
    before = values.get("directness").held_since
    values.apply(Change("directness", "ontogeny"), "changed")
    assert values.get("directness").held_since == before


def test_an_undeclared_value_cannot_be_changed(values):
    assert not _decide(values, "preference_learner", "nothing_declared").allowed


def test_the_snapshot_counts_by_level(values):
    snapshot = values.snapshot()
    assert snapshot["by_level"]["constitutive"] == 1
    assert snapshot["by_level"]["preferential"] == 1
