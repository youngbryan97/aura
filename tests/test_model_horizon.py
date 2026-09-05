"""A simulator is trustworthy where it has been checked and was right.

Outside that region a prediction must not be what settles an irreversible
action — not because it is wrong, but because nothing has established that it
is, and an irreversible act is the one case where "probably" is not a good
enough reason.

The two verdicts that matter most are the ones a naive check collapses:
"nothing near this has been tested" is a different finding from "this region
is bad", and treating them alike is how a model with no track record ends up
trusted by default.
"""

from __future__ import annotations

import random

import pytest

from core.verify.model_horizon import (
    MIN_NEIGHBOURS,
    MIN_RECORD,
    ModelHorizon,
    Standing,
    horizon,
    reset_horizons,
)


@pytest.fixture
def model():
    reset_horizons()
    yield ModelHorizon("test")
    reset_horizons()


def _accurate(model, n=40, seed=5, centre=0.15, spread=0.15):
    rng = random.Random(seed)
    for _ in range(n):
        point = [rng.uniform(centre - spread, centre + spread) for _ in range(2)]
        model.observe(point, 0.5, 0.5 + rng.uniform(-0.05, 0.05))


def _wrong(model, n=40, seed=6, centre=0.85, spread=0.15):
    rng = random.Random(seed)
    for _ in range(n):
        point = [rng.uniform(centre - spread, centre + spread) for _ in range(2)]
        model.observe(point, 0.5, 0.5 + rng.uniform(-0.9, 0.9))


# ── the four standings ───────────────────────────────────────────────────


def test_an_untested_model_says_so_rather_than_passing(model):
    verdict = model.standing([0.5, 0.5])
    assert verdict.standing is Standing.UNMEASURED
    assert verdict.may_drive_irreversible is False


def test_a_checked_and_accurate_region_is_inside(model):
    _accurate(model)
    _wrong(model)
    verdict = model.standing([0.15, 0.15])
    assert verdict.standing is Standing.INSIDE
    assert verdict.may_drive_irreversible is True
    assert verdict.ceiling() == 1.0


def test_a_checked_and_wrong_region_is_unreliable(model):
    _accurate(model)
    _wrong(model)
    verdict = model.standing([0.85, 0.85])
    assert verdict.standing is Standing.UNRELIABLE
    assert verdict.may_drive_irreversible is False
    assert verdict.local_error > 0.25


def test_an_unchecked_region_is_unsupported_not_unreliable(model):
    """The distinction a support-only or accuracy-only check would lose."""
    _accurate(model)
    _wrong(model)
    verdict = model.standing([0.5, 0.5])
    assert verdict.standing is Standing.UNSUPPORTED
    assert verdict.local_error is None, (
        "an unsupported region reported an accuracy, which means it was "
        "judged on cases that were not near it"
    )


def test_not_knowing_ranks_above_knowing_it_is_bad(model):
    """Both are below INSIDE, and they are not the same position."""
    _accurate(model)
    _wrong(model)
    unsupported = model.standing([0.5, 0.5])
    unreliable = model.standing([0.85, 0.85])
    assert unsupported.ceiling() > unreliable.ceiling()
    assert not unsupported.may_drive_irreversible
    assert not unreliable.may_drive_irreversible


# ── the radius has to be relative ────────────────────────────────────────


def test_a_query_between_two_clusters_is_not_supported_by_both(model):
    """An absolute radius counted every point in both clusters as nearby."""
    _accurate(model)
    _wrong(model)
    assert model.standing([0.5, 0.5]).neighbours == 0


def test_a_far_query_is_unsupported_whatever_the_feature_scale(model):
    _accurate(model)
    _wrong(model)
    assert model.standing([50.0, 50.0]).standing is Standing.UNSUPPORTED


def test_the_neighbourhood_is_the_same_across_processes(model):
    """A sampled estimate that varied would flip a verdict between boots."""
    _accurate(model)
    _wrong(model)
    first = model.standing([0.15, 0.15])
    second = model.standing([0.15, 0.15])
    assert first.neighbours == second.neighbours
    assert first.standing is second.standing


# ── the record ───────────────────────────────────────────────────────────


def test_a_small_record_cannot_speak_about_any_region(model):
    for index in range(MIN_RECORD - 1):
        model.observe([0.1, 0.1], 0.5, 0.5)
    assert model.standing([0.1, 0.1]).standing is Standing.UNMEASURED


def test_an_unresolved_prediction_is_not_evidence(model):
    for _ in range(MIN_RECORD * 3):
        model.predicted([0.1, 0.1], 0.5)
    assert model.standing([0.1, 0.1]).standing is Standing.UNMEASURED


def test_a_prediction_can_be_resolved_later(model):
    index = model.predicted([0.2, 0.2], 0.5, label="rollout 1")
    assert model.resolved(index, 0.55) is True
    assert model.snapshot()["resolved"] == 1
    assert model.resolved(9999, 0.1) is False


def test_a_few_neighbours_are_not_a_neighbourhood(model):
    """Two nearby cases and their accuracy is anecdote."""
    rng = random.Random(3)
    for _ in range(MIN_RECORD + 4):
        model.observe([rng.uniform(0.8, 1.0), rng.uniform(0.8, 1.0)], 0.5, 0.5)
    for _ in range(MIN_NEIGHBOURS - 2):
        model.observe([0.1, 0.1], 0.5, 0.5)
    verdict = model.standing([0.1, 0.1])
    assert verdict.standing is Standing.UNSUPPORTED
    assert verdict.neighbours < MIN_NEIGHBOURS


# ── the consequence ──────────────────────────────────────────────────────


def test_only_an_earned_region_may_settle_an_irreversible_act():
    for standing in Standing:
        assert standing.may_drive_irreversible is (standing is Standing.INSIDE)


def test_being_outside_still_allows_a_cheap_undoable_step(model):
    """Outside the horizon is a reason to be careful, not a reason to stop."""
    _accurate(model)
    _wrong(model)
    assert model.standing([0.5, 0.5]).ceiling() > 0.0
    assert model.standing([0.85, 0.85]).ceiling() > 0.0


def test_the_error_bar_is_sealed_before_any_model_is_measured():
    from core.verify.epistemic_independence import registry

    registry().clear()
    model = ModelHorizon("sealed")
    _accurate(model)
    _wrong(model)
    assert model.standing([0.15, 0.15]).standing is Standing.INSIDE
    criterion = registry().get("model_horizon.local_error")
    assert criterion is not None and criterion.direction == "below"
    assert criterion.rationale
    registry().clear()


def test_horizons_are_named_and_separate():
    reset_horizons()
    horizon("a").observe([0.1], 0.5, 0.5)
    assert horizon("b").snapshot()["cases"] == 0
    assert horizon("a").snapshot()["cases"] == 1
    assert horizon("a") is horizon("a")
    reset_horizons()
