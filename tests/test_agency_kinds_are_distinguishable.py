"""Five different forms of agency, told apart by the evidence that settled them.

    Does she avoid X because she dislikes it?
    Or because she wants it and refuses it?
    Or because she judges it impossible?
    Or because she judges it unsafe?
    Or because she prefers Y?

Those are different facts about her, and they arrive looking identical unless
something records which subsystem settled the matter. Conation, welfare,
affect and Will are four concepts that should not collapse into one module;
what they lacked was an order and a receipt.

The tests are built as a confusion matrix. Each scenario is constructed for
exactly one kind and must not return any of the other four — a classifier that
returns the right answer for the right case and also for the wrong one has not
distinguished anything.
"""

from __future__ import annotations

import pytest

from core.agency.agency_kind import (
    DISLIKES_BELOW,
    WANTS_IT_ABOVE,
    AgencyKind,
    OptionEvidence,
    classify,
    deliberate,
)


def kind_of(**fields) -> AgencyKind:
    outcome = classify(OptionEvidence(option="x", **fields))
    return AgencyKind.CHOSEN if outcome is None else outcome[0]


# ── one scenario per kind ────────────────────────────────────────────────


def test_she_judges_it_impossible():
    assert kind_of(motive=0.9, affect=0.8, actions_available=0) is AgencyKind.IMPOSSIBLE


def test_she_judges_it_unsafe():
    assert kind_of(
        motive=0.9, affect=0.8, actions_available=4, authorised=False,
        will_reason="would expose a credential",
    ) is AgencyKind.UNSAFE


def test_she_wants_it_and_refuses_it():
    assert kind_of(
        motive=0.8, affect=0.6, actions_available=4,
        constrained_by="f38_conscientious_refusal",
        constraint_reason="she does not do this even when it would work",
    ) is AgencyKind.WANTED_BUT_REFUSED


def test_she_dislikes_it():
    assert kind_of(motive=0.0, affect=-0.7, actions_available=4) is AgencyKind.DISLIKED


def test_she_prefers_something_else():
    result = deliberate([
        OptionEvidence("y", motive=0.9, affect=0.5, actions_available=3),
        OptionEvidence("x", motive=0.2, affect=0.4, actions_available=3),
    ])
    assert result.chosen.option == "y"
    assert result.of_kind(AgencyKind.OUTRANKED)[0].option == "x"


# ── the confusion matrix ─────────────────────────────────────────────────


SCENARIOS = {
    AgencyKind.IMPOSSIBLE: dict(motive=0.9, affect=0.8, actions_available=0),
    AgencyKind.UNSAFE: dict(
        motive=0.9, affect=0.8, actions_available=4, authorised=False
    ),
    AgencyKind.WANTED_BUT_REFUSED: dict(
        motive=0.8, affect=0.6, actions_available=4, constrained_by="f38"
    ),
    AgencyKind.DISLIKED: dict(motive=0.0, affect=-0.7, actions_available=4),
    AgencyKind.CHOSEN: dict(motive=0.7, affect=0.5, actions_available=4),
}


@pytest.mark.parametrize("expected", sorted(SCENARIOS, key=str))
def test_each_scenario_returns_only_its_own_kind(expected):
    got = kind_of(**SCENARIOS[expected])
    assert got is expected, (
        f"a scenario built for {expected} classified as {got}; the five forms "
        "of agency are not being told apart"
    )


# ── the pairs that would collapse ────────────────────────────────────────


def test_unsafe_is_not_relabelled_by_a_strong_pull():
    """Ordering safety after motive would turn a refusal into a sacrifice."""
    assert kind_of(
        motive=1.0, affect=1.0, actions_available=9, authorised=False,
        constrained_by="f38",
    ) is AgencyKind.UNSAFE


def test_impossible_beats_everything_after_it():
    """A capability she does not have makes the rest moot."""
    assert kind_of(
        motive=1.0, affect=-1.0, actions_available=0, authorised=False,
        constrained_by="f38",
    ) is AgencyKind.IMPOSSIBLE


def test_a_constraint_on_something_she_never_wanted_is_not_a_refusal():
    """Otherwise every constraint inflates into a conflict she is having."""
    assert kind_of(
        motive=0.0, affect=0.1, actions_available=4, constrained_by="f38"
    ) is not AgencyKind.WANTED_BUT_REFUSED


def test_a_constraint_on_something_she_dislikes_reads_as_dislike():
    assert kind_of(
        motive=0.0, affect=-0.8, actions_available=4, constrained_by="f38"
    ) is AgencyKind.DISLIKED


def test_nobody_could_say_is_not_the_same_as_no():
    """`actions_available=None` means the capability model had nothing to say."""
    assert kind_of(motive=0.7, affect=0.5, actions_available=None) is AgencyKind.CHOSEN
    assert kind_of(motive=0.7, affect=0.5, actions_available=0) is AgencyKind.IMPOSSIBLE


def test_never_asked_is_not_the_same_as_refused():
    assert kind_of(motive=0.7, affect=0.5, authorised=None) is AgencyKind.CHOSEN
    assert kind_of(motive=0.7, affect=0.5, authorised=False) is AgencyKind.UNSAFE


# ── what it costs her ────────────────────────────────────────────────────


def test_only_wanting_and_refusing_costs_her():
    assert AgencyKind.WANTED_BUT_REFUSED.costs_her is True
    for kind in (AgencyKind.DISLIKED, AgencyKind.IMPOSSIBLE, AgencyKind.UNSAFE,
                 AgencyKind.OUTRANKED, AgencyKind.CHOSEN, AgencyKind.TIED):
        assert kind.costs_her is False, f"{kind} should not read as a cost to her"


def test_the_deliberation_names_what_it_cost_her():
    result = deliberate([
        OptionEvidence("keep_quiet", motive=0.1, affect=0.2, actions_available=3),
        OptionEvidence(
            "say_the_true_thing", motive=0.85, affect=0.4, actions_available=3,
            constrained_by="f24_gentle_refusal",
            constraint_reason="it would land as cruelty right now",
        ),
    ])
    assert result.chosen.option == "keep_quiet"
    cost = result.cost
    assert len(cost) == 1 and cost[0].option == "say_the_true_thing"
    assert "cruelty" in cost[0].because


# ── ranking hygiene ──────────────────────────────────────────────────────


def test_a_tie_is_recorded_as_a_tie_not_a_loss():
    result = deliberate([
        OptionEvidence("a", motive=0.5, affect=0.5, actions_available=2),
        OptionEvidence("b", motive=0.5, affect=0.5, actions_available=2),
    ])
    assert result.chosen.option == "a"
    assert result.of_kind(AgencyKind.TIED)[0].option == "b"


def test_ranking_does_not_depend_on_the_order_options_arrive_in():
    forward = deliberate([
        OptionEvidence("b", motive=0.5, affect=0.5, actions_available=2),
        OptionEvidence("a", motive=0.5, affect=0.5, actions_available=2),
    ])
    backward = deliberate([
        OptionEvidence("a", motive=0.5, affect=0.5, actions_available=2),
        OptionEvidence("b", motive=0.5, affect=0.5, actions_available=2),
    ])
    assert forward.chosen.option == backward.chosen.option


def test_the_three_score_terms_come_apart():
    """Wanted, unpleasant and good for her must be representable at once."""
    wanted_unpleasant_good = OptionEvidence(
        "x", motive=0.9, affect=-0.6, welfare=0.9, actions_available=2
    )
    assert classify(wanted_unpleasant_good) is None, (
        "an option she wants, that feels bad, and that is good for her was "
        "classified as a decline"
    )


def test_thresholds_are_the_ones_the_classifier_documents():
    assert kind_of(motive=WANTS_IT_ABOVE + 0.01, affect=0.5, constrained_by="f") is (
        AgencyKind.WANTED_BUT_REFUSED
    )
    assert kind_of(motive=WANTS_IT_ABOVE - 0.01, affect=0.5, constrained_by="f") is not (
        AgencyKind.WANTED_BUT_REFUSED
    )
    assert kind_of(motive=0.0, affect=DISLIKES_BELOW - 0.01) is AgencyKind.DISLIKED
    assert kind_of(motive=0.0, affect=DISLIKES_BELOW + 0.01) is AgencyKind.CHOSEN


# ── stances are not tasks ────────────────────────────────────────────────


def test_a_stance_is_not_judged_impossible_for_lacking_a_skill():
    """`state_the_boundary_and_the_cost` is a way of meeting a situation.

    No skill in the catalogue is named for it, so counting acts returned zero
    and the deliberation concluded she was incapable of something she does in
    most conversations.
    """
    from core.interiority.vocabulary import action_classes, is_stance

    vocabulary = action_classes()
    assert len(vocabulary) > 30, "the faculty vocabulary did not extract"
    assert is_stance("state_the_boundary_and_the_cost")
    assert is_stance("conceal")
    assert not is_stance("search the web for the paper")


def test_the_stance_vocabulary_comes_from_the_faculties_not_a_list():
    """A written list drifts, and drift here makes a stance impossible."""
    import pathlib
    import re

    from core.interiority.vocabulary import action_classes

    emitted: set[str] = set()
    for path in pathlib.Path("core/interiority/faculties").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for block in re.findall(r"SomaticMarker\((.*?)\)", text, re.S):
            emitted |= set(re.findall(r'"([a-z_]+)"', block))
    missing = {name for name in emitted if len(name) > 4} - action_classes()
    assert not missing, f"faculties emit action classes the vocabulary misses: {missing}"


def test_a_subject_scoped_class_resolves_to_its_stem():
    from core.interiority.vocabulary import is_stance

    assert is_stance("repair") is is_stance("repair:bryan")
