"""Having the structure, and living through it, are different things.

`long_horizon.py` computes two things a relationship of years has that a
conversation does not: trust that was tested, and beliefs about beliefs with
acts licensing them. Both were correct and neither was on a production path,
so the distinction between

    the ability to represent relationship history

and

    ordinary life continuously producing that history

was left standing. Existing live social cognition is not empty — there are
agent models, trust evidence, rupture risk, dossiers, theory-of-mind
projection, affect and bond state — but the richer rupture, repair and
commitment model was underfed by experience.

Ordinary life does write the record. Every chat turn reaches
`chat_turn_logger`, which asks the interpersonal store to observe it, and the
store writes typed observations with occurrences and resolutions. What was
missing was the reading, and the direction matters: this reads what living
wrote and never writes anything to make the reading nicer.
"""

from __future__ import annotations

import time

import pytest

from core.memory.interpersonal_model import (
    Facet,
    Occurrence,
    PersonModel,
    Subject,
    Valence,
)
from core.social.long_horizon import Discriminates, Episode
from core.social.what_the_years_add_up_to import (
    acts_towards,
    history_with,
    how_they_stand,
    what_he_takes_her_to_know,
)

NOW = time.time()
HOUR = 3600.0


@pytest.fixture
def a_life():
    """A record of the kind ordinary turns actually leave behind."""

    model = PersonModel("bryan")
    model.observe(
        "shipped what he asked for", episode_id="e1",
        facet=Facet.TRUST_BUILT, at=NOW - 40 * HOUR,
    )
    model.observe(
        "worked late on the release", episode_id="e2",
        facet=Facet.EXPERIENCE, valence=Valence.WARM, at=NOW - 30 * HOUR,
    )
    model.observe(
        "I claimed a result I had not run", episode_id="e3",
        facet=Facet.RUPTURE, subject=Subject.SELF, at=NOW - 20 * HOUR,
    )
    return model


def test_the_record_becomes_a_history(a_life):
    events = history_with("bryan", model=a_life)
    assert [one.episode for one in events] == [
        Episode.KEPT, Episode.CONTACT, Episode.BROKE
    ]
    assert events == tuple(sorted(events, key=lambda one: one.at))


def test_an_unrepaired_rupture_costs_and_keeps_costing(a_life):
    where = how_they_stand("bryan", model=a_life, now=NOW)
    assert where.tested
    assert where.unrepaired == 1
    assert not where.stronger_for_it
    assert where.trust < 0.5


def test_a_repair_is_dated_from_the_resolution_and_not_from_hope(a_life):
    """Guessing that time healed it is how a ledger of grievances becomes a
    story of growth without anything having happened."""

    before = how_they_stand("bryan", model=a_life, now=NOW)
    for observation in a_life:
        if observation.facet is Facet.RUPTURE:
            observation.resolved_by = Occurrence(
                episode_id="e4", at=NOW - 10 * HOUR, note="ran it and showed the log"
            )
    after = how_they_stand("bryan", model=a_life, now=NOW)
    assert before.unrepaired == 1 and after.unrepaired == 0
    assert after.repairs == 1
    assert after.trust > before.trust


def test_a_bond_that_held_after_a_repair_is_proved(a_life):
    """The finding a monotone score cannot represent."""

    for observation in a_life:
        if observation.facet is Facet.RUPTURE:
            observation.resolved_by = Occurrence(episode_id="e4", at=NOW - 10 * HOUR)
    for index in range(4):
        a_life.observe(
            f"kept a commitment {index}", episode_id=f"k{index}",
            facet=Facet.COMMITMENT, at=NOW - (8 - index) * HOUR,
        )
    where = how_they_stand("bryan", model=a_life, now=NOW)
    assert where.proved > 0.0
    assert where.stronger_for_it


def test_a_difficult_conversation_is_not_a_rupture(a_life):
    """Turning affect into a betrayal is the manufacture the store forbids."""

    a_life.observe(
        "that exchange was hard going", episode_id="e9",
        facet=Facet.AFFECT, valence=Valence.DIFFICULT, at=NOW - 5 * HOUR,
    )
    events = history_with("bryan", model=a_life)
    assert sum(1 for one in events if one.episode is Episode.BROKE) == 1


def test_a_second_order_belief_needs_an_act_that_discriminates(a_life):
    """Most acts say nothing, and saying so is the whole difference between
    this and inventing a nested mind from a trust score."""

    nothing = what_he_takes_her_to_know("bryan", "the write gateway", model=a_life)
    assert not nothing.held
    assert nothing.believes_she_knows is None

    a_life.observe(
        "explained the write gateway again", episode_id="e10",
        facet=Facet.MISUNDERSTANDING, at=NOW - 2 * HOUR,
    )
    held = what_he_takes_her_to_know("bryan", "the write gateway", model=a_life)
    assert held.held
    assert held.believes_she_knows is False
    assert held.evidence


def test_acts_are_only_the_ones_that_discriminate(a_life):
    a_life.observe(
        "we both see how the gateway works now", episode_id="e11",
        facet=Facet.UNDERSTANDING, at=NOW - HOUR,
    )
    acts = acts_towards("bryan", "gateway", model=a_life)
    assert [one.discriminates for one in acts] == [Discriminates.THINKS_SHE_KNOWS]


def test_the_reading_reaches_the_block_ordinary_life_assembles(a_life):
    """The seam. It must not be another module nothing reaches.

    `_trust` above it counts — so many kept, so many repaired, so many not —
    and a count has no order. A bond that broke, was repaired and then held
    through four more commitments produces the same counts as one repaired
    and never tested again, and they are not the same relationship. The
    ordered reading is a dynamic beside the counted one, in the block a turn
    already assembles.
    """

    for observation in a_life:
        if observation.facet is Facet.RUPTURE:
            observation.resolved_by = Occurrence(episode_id="e4", at=NOW - 10 * HOUR)
    for index in range(4):
        a_life.observe(
            f"kept a commitment {index}", episode_id=f"k{index}",
            facet=Facet.COMMITMENT, at=NOW - (8 - index) * HOUR,
        )
    said = {one.name: one for one in a_life.dynamics(now=NOW)}
    assert "what it has been through" in said
    assert "stronger for it" in said["what it has been through"].standing
    assert any("proved" in one for one in said["what it has been through"].basis)
    assert "what it has been through" in a_life.render(now=NOW)


def test_the_counted_reading_cannot_tell_these_apart(a_life):
    """Why the ordered one earns its place.

    Two histories with identical counts and opposite meanings: repaired then
    held, and repaired then nothing. `_trust` reports the same standing for
    both; the ordered reading does not.
    """

    for observation in a_life:
        if observation.facet is Facet.RUPTURE:
            observation.resolved_by = Occurrence(episode_id="e4", at=NOW - 10 * HOUR)
    quiet = {one.name: one for one in a_life.dynamics(now=NOW)}
    for index in range(4):
        a_life.observe(
            f"kept a commitment {index}", episode_id=f"k{index}",
            facet=Facet.COMMITMENT, at=NOW - (8 - index) * HOUR,
        )
    held = {one.name: one for one in a_life.dynamics(now=NOW)}
    assert (
        quiet["what it has been through"].standing
        != held["what it has been through"].standing
    )


def test_an_untested_bond_adds_no_reading():
    """Reporting the absence of a history as a finding fills a block with
    the absence of things."""

    model = PersonModel("someone")
    model.observe("likes short answers", episode_id="p1", facet=Facet.PREFERENCE)
    said = {one.name for one in model.dynamics()}
    assert "what it has been through" not in said
    assert "what it has been through" not in model.render()
    assert "trust" in said, "the counted reading still speaks"


def test_an_open_rupture_is_said_out_loud():
    model = PersonModel("someone")
    model.observe("I got this wrong", episode_id="r1", facet=Facet.RUPTURE)
    said = {one.name: one for one in model.dynamics()}
    assert "still open" in said["what it has been through"].standing
