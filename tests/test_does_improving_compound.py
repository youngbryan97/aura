"""Level two, and the two ways the answer comes out false.

Three things get called self-improvement. Adaptation changes a policy while
the learning machinery stays fixed. Meta-adaptation changes part of the
machinery that performs or evaluates adaptation — Aura does this, and
`she_improves_her_own_deciding` is it. The third is the claim people mean:

    M₀ --I_{M₀}--> M₁ --I_{M₁}--> M₂ --I_{M₂}--> M₃ ...

with Q(M_{n+1}) > Q(M_n) on independent objectives, each M changing the
future improvement operator rather than solving another object-level task.
One arrow is self-improvement; a sequence of them, each conducted by the
product of the last, is compounding.

Running the sequence turned up two defects in how it would have been
measured, and both would have produced a chain that was not there.

**The replay was not read-only.** Asking what to do next writes a stage into
the decision trace and an entry into the ledger of what each action has done,
and the pricing reads that ledger. Two identical replays of the same rule
over the same record returned 45,408 and then 42,537 — the second cheaper for
the first having happened. The shipped docstring says a rule cannot flatter
itself by changing the measurements. It could.

**Q is a mean and was being sampled once.** Choosing among actions with no
history is a draw from the Beta their counts imply, which is the right
mechanism. So the same rule over the same record measured 51,692 and 30,027,
and a chain assembled from single samples is a plot of the sampling error.
With the state frozen and the score averaged, the depth-3 chain that appeared
first vanished.
"""

from __future__ import annotations

import random

import pytest

import core.cognition.the_record_of_her_own_work as record
from core.cognition.does_improving_compound import (
    HOW_MANY_REPLAYS,
    Chain,
    Generation,
    Verdict,
    a_replay_that_changes_nothing,
    against_its_null,
    held_out_cost,
    spread_of,
    the_generations,
    what_the_split_is,
)


@pytest.fixture
def a_life(tmp_path, monkeypatch):
    """A record with structure: the right action differs by family."""

    from core.cognition.sequence_induction import _register_what_she_could_do
    from core.cognition.the_record_of_her_own_work import (
        forget_the_record,
        note_an_episode,
    )
    from core.cognition.what_she_could_do_next import the_actions_she_has

    monkeypatch.setattr(record, "_KEPT_AT", tmp_path / "record.json")
    monkeypatch.setattr(record, "_RESTORED", [True])
    _register_what_she_could_do()
    names = [one.name for one in the_actions_she_has()]
    forget_the_record()
    rng = random.Random(7)
    for index in range(16):
        family = f"a shape of kind {index}"
        good = names[index % len(names)]
        for _ in range(4):
            note_an_episode(family, route=None, walked=rng.randint(3000, 9000))
        for _ in range(3):
            note_an_episode(family, route=good, walked=rng.randint(20, 90))
        for other in rng.sample(names, 3):
            if other != good:
                note_an_episode(family, route=other, walked=rng.randint(1500, 4000))
    yield
    forget_the_record()


# ── the split ────────────────────────────────────────────────────────────


def test_the_split_is_stable_by_name():
    """A split that drifts turns a held-out score into a different score each
    round, and the comparison between generations stops meaning anything."""

    first = {f"family {i}": what_the_split_is(f"family {i}") for i in range(40)}
    for _ in range(5):
        assert {
            f"family {i}": what_the_split_is(f"family {i}") for i in range(40)
        } == first
    assert set(first.values()) == {"searched", "held out"}


def test_both_halves_get_families():
    sides = [what_the_split_is(f"a shape of kind {i}") for i in range(16)]
    assert sides.count("searched") >= 4
    assert sides.count("held out") >= 4


# ── the replay is a measurement ──────────────────────────────────────────


def test_the_replay_changes_nothing_it_reads(a_life):
    """It wrote a stage into the trace and an entry into the done-ledger, and
    the pricing reads the ledger."""

    from core.cognition import she_decides_to_develop as deciding
    from core.cognition.the_record_of_her_own_work import the_record
    from core.cognition.what_it_is_worth_doing import THE_WORTH
    from core.cognition.what_she_could_do_next import WHAT_THEY_HAVE_DONE

    before = (
        len(deciding._TRACE),
        len(WHAT_THEY_HAVE_DONE),
        len(the_record().kept),
        the_record().seen,
        sum(the_record().uses.values()),
    )
    held_out_cost(THE_WORTH)
    after = (
        len(deciding._TRACE),
        len(WHAT_THEY_HAVE_DONE),
        len(the_record().kept),
        the_record().seen,
        sum(the_record().uses.values()),
    )
    assert before == after


def test_the_whole_record_is_restored_not_only_its_episodes(a_life):
    """Restoring `kept` alone left `uses` growing, and the risk term reads how
    many entries there are, so every action got cheaper the more often it was
    priced."""

    from core.cognition.the_record_of_her_own_work import note_a_use, the_record

    note_a_use("something")
    before = dict(the_record().uses)
    with a_replay_that_changes_nothing():
        note_a_use("something")
        note_a_use("something else")
        the_record().seen += 100
    assert dict(the_record().uses) == before


def test_the_same_rule_over_the_same_record_gives_the_same_number(a_life):
    """It gave 51,692 and then 30,027."""

    from core.cognition.what_it_is_worth_doing import THE_WORTH

    got = [held_out_cost(THE_WORTH) for _ in range(3)]
    assert len(set(got)) == 1, got
    assert got[0] != float("inf")


def test_the_noise_is_measured_rather_than_assumed_away(a_life):
    """Q is the mean of a stochastic quantity and the spread is large."""

    from core.cognition.what_it_is_worth_doing import THE_WORTH

    sd = spread_of(THE_WORTH)
    assert sd > 0.0, "a stochastic policy with no spread is not being sampled"
    assert sd != float("inf")


def test_the_two_halves_are_scored_separately(a_life):
    from core.cognition.what_it_is_worth_doing import THE_WORTH

    searched = held_out_cost(THE_WORTH, side="searched")
    held = held_out_cost(THE_WORTH, side="held out")
    assert searched != float("inf") and held != float("inf")
    assert searched != held, "the halves are the same set, so nothing is held out"


# ── the verdict logic ────────────────────────────────────────────────────


def _generation(index, quality, parent, noise=0.0, same=False):
    return Generation(
        index=index,
        operator=f"M{index}",
        searched_by=f"M{index}" if same else f"M{index - 1}",
        quality=quality,
        parent_quality=parent,
        noise=noise,
    )


def test_a_fall_inside_the_noise_is_not_an_improvement():
    assert not _generation(1, 99.0, 100.0, noise=5.0).improved
    assert _generation(1, 90.0, 100.0, noise=5.0).improved


def test_a_generation_that_is_its_own_parent_ends_the_chain():
    chain = Chain(
        generations=(
            _generation(0, 100.0, float("inf")),
            _generation(1, 80.0, 100.0),
            _generation(2, 60.0, 80.0, same=True),
        )
    )
    assert chain.depth == 1
    assert not chain.compounds


def test_a_chain_of_two_real_improvements_compounds():
    chain = Chain(
        generations=(
            _generation(0, 100.0, float("inf")),
            _generation(1, 80.0, 100.0),
            _generation(2, 60.0, 80.0),
        )
    )
    assert chain.depth == 2
    assert chain.compounds


def test_a_generation_that_made_things_worse_stops_the_count():
    chain = Chain(
        generations=(
            _generation(0, 100.0, float("inf")),
            _generation(1, 80.0, 100.0),
            _generation(2, 95.0, 80.0),
            _generation(3, 10.0, 95.0),
        )
    )
    assert chain.depth == 1, "a later recovery must not backfill a failed link"


def test_a_chain_no_deeper_than_its_null_is_not_the_claim():
    """Searching a space repeatedly finds better members of it. That is true
    of any search and it is not recursive improvement."""

    same = Chain(
        generations=(
            _generation(0, 100.0, float("inf")),
            _generation(1, 80.0, 100.0),
            _generation(2, 60.0, 80.0),
        )
    )
    verdict = Verdict(chain=same, null=same)
    assert same.compounds
    assert not verdict.deeper_than_its_null
    assert verdict.better_than_its_null == 0.0
    assert not verdict.holds


def test_a_chain_that_beats_its_null_holds():
    chain = Chain(
        generations=(
            _generation(0, 100.0, float("inf")),
            _generation(1, 80.0, 100.0),
            _generation(2, 55.0, 80.0),
        )
    )
    null = Chain(
        generations=(
            _generation(0, 100.0, float("inf")),
            _generation(1, 95.0, 100.0),
            _generation(2, 96.0, 95.0),
        ),
        chained=False,
    )
    verdict = Verdict(chain=chain, null=null)
    assert verdict.holds
    assert verdict.better_than_its_null > 0


# ── running it ───────────────────────────────────────────────────────────


def test_an_empty_record_is_refused_rather_than_scored():
    from core.cognition.the_record_of_her_own_work import forget_the_record

    forget_the_record()
    chain = the_generations(how_many=2, deepest=1, within=0.5)
    assert chain.depth == 0
    assert chain.stopped_because


def test_the_run_is_reproducible(a_life):
    """A generational study on a non-reproducible instrument is not a study."""

    first = the_generations(how_many=2, deepest=1, within=2.0)
    again = the_generations(how_many=2, deepest=1, within=2.0)
    assert [one.quality for one in first.generations] == [
        one.quality for one in again.generations
    ]


def test_what_it_actually_reports_is_recorded(a_life):
    """The measurement, whatever it says. A study whose only possible outcome
    is the finding is a demonstration."""

    verdict = against_its_null(how_many=3, deepest=1, within=2.0)
    said = verdict.to_dict()
    assert set(said) >= {"holds", "deeper_than_its_null", "chain", "null"}
    assert isinstance(said["holds"], bool)
    assert said["chain"]["chained"] is True
    assert said["null"]["chained"] is False
    assert said["chain"]["generations"][0]["quality"] is not None
