"""Tests for the DecisionPreferenceLearner: her choices author her future preferences.

Contract: a choice that scored unusually high on dimension D, when it turns out WELL,
raises D's learned multiplier; when it turns out BADLY, lowers it. Multipliers stay
bounded, persist across sessions, and feed back into the arbiter's scoring.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.agency.decision_preference_learner import (
    DIMENSIONS,
    W_MAX,
    W_MIN,
    DecisionPreferenceLearner,
)


def _scores(**overrides):
    base = {d: 0.5 for d in DIMENSIONS}
    base.update(overrides)
    return base


def test_positive_outcome_reinforces_the_distinguishing_dimension(tmp_path):
    learner = DecisionPreferenceLearner(state_path=tmp_path / "p.json")
    before = learner.multipliers()["novelty"]
    # A choice that was unusually high on novelty vs the pool, that paid off.
    cid = learner.record_choice(
        chosen_scores=_scores(novelty=0.95),
        pool_scores=[_scores(novelty=0.95), _scores(novelty=0.2), _scores(novelty=0.1)],
        goal="try something new",
    )
    learner.resolve_choice(cid, reward=1.0)
    after = learner.multipliers()["novelty"]
    assert after > before  # novelty served her → weighted up


def test_negative_outcome_attenuates_the_distinguishing_dimension(tmp_path):
    learner = DecisionPreferenceLearner(state_path=tmp_path / "p.json")
    before = learner.multipliers()["novelty"]
    cid = learner.record_choice(
        chosen_scores=_scores(novelty=0.95),
        pool_scores=[_scores(novelty=0.95), _scores(novelty=0.2)],
        goal="risky novelty",
    )
    learner.resolve_choice(cid, reward=-1.0)
    after = learner.multipliers()["novelty"]
    assert after < before  # novelty disappointed → weighted down


def test_multipliers_stay_bounded_under_repeated_outcomes(tmp_path):
    learner = DecisionPreferenceLearner(state_path=tmp_path / "p.json")
    for _ in range(200):
        cid = learner.record_choice(
            chosen_scores=_scores(urgency=1.0),
            pool_scores=[_scores(urgency=1.0), _scores(urgency=0.0)],
            goal="always urgent",
        )
        learner.resolve_choice(cid, reward=1.0)
    assert learner.multipliers()["urgency"] <= W_MAX
    for _ in range(200):
        cid = learner.record_choice(
            chosen_scores=_scores(urgency=1.0),
            pool_scores=[_scores(urgency=1.0), _scores(urgency=0.0)],
            goal="urgent flops",
        )
        learner.resolve_choice(cid, reward=-1.0)
    assert learner.multipliers()["urgency"] >= W_MIN


def test_effective_weights_apply_multiplier(tmp_path):
    learner = DecisionPreferenceLearner(state_path=tmp_path / "p.json")
    cid = learner.record_choice(
        chosen_scores=_scores(expected_value=0.95),
        pool_scores=[_scores(expected_value=0.95), _scores(expected_value=0.1)],
        goal="high-EV choice",
    )
    learner.resolve_choice(cid, reward=1.0)
    base = {d: 1.0 for d in DIMENSIONS}
    eff = learner.effective_weights(base)
    assert eff["expected_value"] > 1.0  # learned-up dimension raises the effective weight


def test_persistence_round_trip(tmp_path):
    path = tmp_path / "p.json"
    learner = DecisionPreferenceLearner(state_path=path)
    cid = learner.record_choice(
        chosen_scores=_scores(continuity=0.95),
        pool_scores=[_scores(continuity=0.95), _scores(continuity=0.1)],
        goal="continuity choice",
    )
    learner.resolve_choice(cid, reward=1.0)
    learned = learner.multipliers()["continuity"]
    assert path.exists()
    # A fresh instance loads the learned weighting — the lesson survives a restart.
    reloaded = DecisionPreferenceLearner(state_path=path)
    assert abs(reloaded.multipliers()["continuity"] - learned) < 1e-6


def test_value_set_is_fixed_never_grows(tmp_path):
    learner = DecisionPreferenceLearner(state_path=tmp_path / "p.json")
    cid = learner.record_choice(
        chosen_scores={**_scores(), "invented_value": 0.99},  # try to smuggle a new value
        pool_scores=[_scores()],
        goal="smuggle",
    )
    learner.resolve_choice(cid, reward=1.0)
    # It learns weighting of the FIXED dimensions only — no new value enters.
    assert set(learner.multipliers().keys()) == set(DIMENSIONS)


def test_arbiter_applies_learned_weights(tmp_path, monkeypatch):
    import core.agency.decision_preference_learner as dpl
    from core.agency.initiative_arbiter import InitiativeArbiter

    learner = DecisionPreferenceLearner(state_path=tmp_path / "p.json")
    # Drive novelty's multiplier up via a paid-off novelty choice.
    cid = learner.record_choice(
        chosen_scores=_scores(novelty=0.95),
        pool_scores=[_scores(novelty=0.95), _scores(novelty=0.1)],
        goal="novel",
    )
    learner.resolve_choice(cid, reward=1.0)
    monkeypatch.setattr(dpl, "get_decision_preference_learner", lambda: learner)

    arb = InitiativeArbiter()
    flat = {d: 0.5 for d in DIMENSIONS}
    score_high_novelty = arb._compute_weighted_score({**flat, "novelty": 1.0})
    # With novelty up-weighted, a high-novelty option scores higher than under neutral weights.
    assert score_high_novelty > 0.5


def test_a_choice_awaiting_its_outcome_survives_a_restart_and_is_credited_once(tmp_path):
    """The habit ledger outlives a restart; the choice it resolves now does too."""
    from core.runtime.atomic_writer import flush_writes_behind

    path = tmp_path / "p.json"
    before = DecisionPreferenceLearner(state_path=path)
    cid = before.record_choice(
        chosen_scores=_scores(continuity=0.95),
        pool_scores=[_scores(continuity=0.95), _scores(continuity=0.1)],
        goal="continuity choice",
    )
    flush_writes_behind()
    after = DecisionPreferenceLearner(state_path=path)
    assert after.stats()["pending_choices"] == 1
    credited = after.resolve_choice(cid, reward=1.0)["continuity"]
    assert credited > 1.0
    # A second delivery of the same outcome changes nothing and is counted.
    assert after.resolve_choice(cid, reward=1.0)["continuity"] == credited
    assert after.stats()["outcomes_for_unknown_choices"] == 1
