"""One identity per concept, one track per thing, one receipt per action.

Cards 040, 050, 063, 173 (concept identity), 196 (entity identity across
occlusion), 195 (action transition receipts), A13.3 and A2.12.
"""
from __future__ import annotations

import pytest

from core.cognition.action_receipt import (
    TransitionVerdict,
    UnqualifiedTransition,
    get_receipt_ledger,
    observe_twice,
    qualified,
    reset_receipt_ledger_for_test,
    verify_transition,
)
from core.cognition.concept_handle import (
    BindingMethod,
    ConceptHandle,
    Projection,
    Substrate,
    reset_concept_registry_for_test,
)
from core.cognition.entity_track import (
    Observation,
    TrackState,
    reset_track_store_for_test,
)


# ── concept identity ──────────────────────────────────────────────────────

def test_one_concept_reaches_four_substrates():
    registry = reset_concept_registry_for_test()
    registry.bind("kettle", Substrate.ATOMSPACE, "Concept:kettle", method=BindingMethod.DECLARED)
    registry.bind("kettle", Substrate.GROUNDED, "gc_88", method=BindingMethod.MEASURED, confidence=0.95)
    registry.bind("kettle", Substrate.EMBEDDING, "vec_12", method=BindingMethod.SIMILARITY, confidence=0.7)
    handle = registry.bind("kettle", Substrate.WORLD_LATENT, "slot_3", method=BindingMethod.MEASURED, confidence=0.9)
    assert handle.reach == 4
    assert registry.report()["reach_four_or_more"] == 1


def test_a_label_match_cannot_claim_high_confidence():
    projection = Projection(Substrate.EMBEDDING, "v1", method=BindingMethod.LABEL, confidence=0.99)
    assert projection.confidence <= 0.4, "a string comparison is not strong evidence of identity"


def test_a_weaker_binding_never_overwrites_a_stronger_one():
    registry = reset_concept_registry_for_test()
    registry.bind("kettle", Substrate.GROUNDED, "gc_1", method=BindingMethod.MEASURED, confidence=0.95)
    handle = registry.bind("kettle", Substrate.GROUNDED, "gc_1", method=BindingMethod.LABEL)
    assert handle.projection(Substrate.GROUNDED).method is BindingMethod.MEASURED


def test_binding_confidence_is_the_weakest_link():
    registry = reset_concept_registry_for_test()
    registry.bind("kettle", Substrate.GROUNDED, "gc_1", method=BindingMethod.MEASURED, confidence=0.95)
    handle = registry.bind("kettle", Substrate.EMBEDDING, "v1", method=BindingMethod.LABEL)
    assert handle.bound_confidence() <= 0.4


def test_substrates_that_disagree_are_reported_not_averaged():
    registry = reset_concept_registry_for_test()
    registry.bind("kettle", Substrate.ATOMSPACE, "a", method=BindingMethod.DECLARED)
    handle = registry.bind("kettle", Substrate.GROUNDED, "g", method=BindingMethod.MEASURED, confidence=0.9)
    report = handle.disagreement({Substrate.ATOMSPACE: 0.9, Substrate.GROUNDED: 0.2})
    assert report["comparable"] and report["spread"] == pytest.approx(0.7)
    assert report["lowest"] == "grounded"


def test_a_single_substrate_is_not_a_disagreement():
    handle = ConceptHandle(handle_id="c1", label="kettle")
    assert handle.disagreement({Substrate.ATOMSPACE: 0.9})["comparable"] is False


def test_any_substrate_reference_resolves_back_to_the_concept():
    registry = reset_concept_registry_for_test()
    registry.bind("kettle", Substrate.GROUNDED, "gc_88", method=BindingMethod.MEASURED, confidence=0.9)
    assert registry.resolve(Substrate.GROUNDED, "gc_88").label == "kettle"
    assert registry.resolve(Substrate.GROUNDED, "nope") is None


def test_merging_keeps_lineage_and_reindexes_references():
    registry = reset_concept_registry_for_test()
    registry.bind("kettle", Substrate.ATOMSPACE, "a1", method=BindingMethod.DECLARED)
    registry.bind("the kettle", Substrate.GROUNDED, "g1", method=BindingMethod.MEASURED, confidence=0.9)
    merged = registry.merge("kettle", "the kettle")
    assert merged.reach == 2
    assert merged.lineage
    assert registry.resolve(Substrate.GROUNDED, "g1").handle_id == merged.handle_id


def test_labels_normalise_so_case_and_spacing_do_not_fork_identity():
    registry = reset_concept_registry_for_test()
    a = registry.handle_for("The  Kettle")
    b = registry.handle_for("the kettle")
    assert a.handle_id == b.handle_id


# ── entity identity ───────────────────────────────────────────────────────

def _at(t, x):
    return Observation(at=float(t), geometry=(float(x), 0.0))


def test_a_thing_survives_occlusion_and_is_reacquired_as_itself():
    store = reset_track_store_for_test()
    for t in range(6):
        store.update([_at(t, t * 0.05)])
    track_id = store.tracks()[0].track_id
    for _ in range(4):
        store.update([])
    assert store.tracks()[0].state is TrackState.OCCLUDED
    store.update([_at(10, 0.35)])
    assert len(store.tracks()) == 1
    assert store.tracks()[0].track_id == track_id


def test_persistence_is_proportional_to_how_well_established_the_track_is():
    store = reset_track_store_for_test()
    store.update([_at(0, 0.0)])
    brief = store.tracks()[0]
    for t in range(1, 40):
        store.update([_at(t, 0.0)])
    established = store.tracks()[0]
    assert established.persistence_budget > brief.persistence_budget or established.support > 1


def test_a_track_that_runs_out_of_budget_is_lost_not_forgotten():
    store = reset_track_store_for_test()
    store.update([_at(0, 0.0)])
    for _ in range(10):
        store.update([])
    assert store.tracks()[0].state is TrackState.LOST


def test_an_ambiguous_association_is_refused_rather_than_guessed():
    store = reset_track_store_for_test()
    store.update([_at(0, 0.0), _at(0, 0.02)])
    before = len(store.tracks())
    store.update([_at(1, 0.01)])
    assert store.report()["ambiguous_associations_refused"] >= 1
    assert len(store.tracks()) > before, "a new track is honest; picking one writes a false history"


def test_incomparable_observations_never_look_like_a_close_match():
    a = Observation(at=0.0, geometry=(1.0, 2.0))
    b = Observation(at=1.0, features=(0.5, 0.5))
    assert a.distance(b) == float("inf")


def test_merge_keeps_support_and_a_reference_to_the_old_track_still_resolves():
    store = reset_track_store_for_test()
    store.update([_at(0, 0.0)])
    store.update([_at(1, 5.0)])
    ids = [t.track_id for t in store.tracks()]
    kept = store.merge(ids[0], ids[1])
    assert kept.support == 2
    assert store.resolve(ids[1]).track_id == kept.track_id


def test_hypotheses_stay_separate_from_support():
    store = reset_track_store_for_test()
    store.update([_at(0, 0.0)])
    track = store.tracks()[0]
    track.suggest("save button", 0.9)
    assert track.support == 1 and track.hypotheses["save button"] == 0.9


# ── action receipts ───────────────────────────────────────────────────────

def test_an_action_that_did_what_it_predicted_is_confirmed():
    receipt = verify_transition(
        action="click", target="save", authority="user",
        before={"saved": False}, after={"saved": True},
        predicted={"saved": True}, stable=True,
    )
    assert receipt.verdict is TransitionVerdict.CONFIRMED
    assert receipt.is_qualified


def test_an_action_that_moved_nothing_is_no_change_not_success():
    receipt = verify_transition(
        action="click", target="save", authority="user",
        before={"saved": False}, after={"saved": False},
        predicted={"saved": True}, stable=True,
    )
    assert receipt.verdict is TransitionVerdict.NO_CHANGE
    assert not receipt.is_qualified


def test_typing_into_the_wrong_window_is_wrong_target_not_success():
    """Thirty-five moves into a terminal while the game was one window back."""
    receipt = verify_transition(
        action="key", target="game", authority="user",
        before={"game": "x", "terminal": "$"}, after={"game": "x", "terminal": "$w"},
        predicted={"game": "y"}, stable=True, target_key="game",
    )
    assert receipt.verdict is TransitionVerdict.WRONG_TARGET
    assert receipt.unexpected_change == ("terminal",)


def test_no_observation_after_is_unverified_not_a_weak_positive():
    receipt = verify_transition(
        action="click", target="save", authority="user",
        before={"saved": False}, after=None, predicted={"saved": True},
    )
    assert receipt.verdict is TransitionVerdict.UNVERIFIED
    assert not receipt.is_qualified


def test_an_action_that_predicted_nothing_confirms_nothing():
    receipt = verify_transition(
        action="click", target="save", authority="user",
        before={"a": 1}, after={"a": 2}, stable=True,
    )
    assert receipt.verdict is TransitionVerdict.UNVERIFIED


def test_a_confirmed_but_unstable_reading_is_not_qualified():
    receipt = verify_transition(
        action="click", target="save", authority="user",
        before={"saved": False}, after={"saved": True},
        predicted={"saved": True}, stable=False,
    )
    assert receipt.verdict is TransitionVerdict.CONFIRMED
    assert not receipt.is_qualified, "one read of a settling interface is a guess"


def test_a_learner_cannot_learn_from_an_unverified_action():
    reset_receipt_ledger_for_test()
    receipt = verify_transition(
        action="click", target="save", authority="user",
        before={"saved": False}, after=None, predicted={"saved": True},
    )
    with pytest.raises(UnqualifiedTransition):
        qualified(receipt, learner="screen_pursuit")
    report = get_receipt_ledger().report()
    assert report["by_learner"]["screen_pursuit"]["unverified"] == 1


def test_the_ledger_reports_how_much_of_what_learners_saw_was_real():
    reset_receipt_ledger_for_test()
    good = verify_transition(
        action="a", target="t", authority="u",
        before={"x": 0}, after={"x": 1}, predicted={"x": 1}, stable=True,
    )
    bad = verify_transition(action="a", target="t", authority="u", before={"x": 0}, after=None)
    qualified(good, learner="L")
    with pytest.raises(UnqualifiedTransition):
        qualified(bad, learner="L")
    assert get_receipt_ledger().report()["confirmed_fraction"] == pytest.approx(0.5)


def test_reading_twice_is_what_stable_means():
    readings = iter([{"x": 1}, {"x": 1}, {"x": 1}, {"x": 2}])
    _, agreed = observe_twice(lambda: next(readings))
    assert agreed
    _, agreed = observe_twice(lambda: next(readings))
    assert not agreed


# ── a thing that kept moving comes back where it was going ────────────────
#
# Card 133. Running the tracking campaign found two defects, and both were
# invisible to a test that only moved things a little.


def test_a_track_carries_the_motion_it_measured():
    from core.cognition.entity_track import EntityTrack, Observation

    track = EntityTrack(track_id="t1")
    track.observe(Observation(at=0.0, geometry=(0.0, 0.0)))
    # One sighting is a position, not a trajectory.
    assert track.velocity == ()
    assert track.predicted_at(5.0).geometry == (0.0, 0.0)

    track.observe(Observation(at=1.0, geometry=(4.0, 2.0)))
    assert track.velocity == (4.0, 2.0)
    assert track.predicted_at(3.0).geometry == pytest.approx((12.0, 6.0))


def test_a_thing_that_crossed_an_occluder_is_still_one_thing():
    """Matching on where it was last seen breaks the track of anything that
    moves, which is every case this module is for."""
    from core.cognition.entity_track import Observation, TrackStore

    store = TrackStore(match_threshold=8.0)
    x, hidden = 5.0, (40.0, 55.0)
    seen: set[str] = set()
    for step in range(20):
        x += 4.0
        if hidden[0] <= x <= hidden[1]:
            for track in store.tracks():
                track.miss()
            continue
        for track in store.update(
            [Observation(at=float(step), geometry=(x, 50.0), label="one")]
        ):
            seen.add(track.track_id)
    assert len(seen) == 1, f"the track broke into {len(seen)} things"


def test_an_exact_match_is_the_least_ambiguous_case_not_the_most():
    """A best distance of zero used to refuse the association outright, so the
    tracker got worse the better its evidence became."""
    from core.cognition.entity_track import Observation, TrackStore

    store = TrackStore(match_threshold=5.0)
    store.update([Observation(at=0.0, geometry=(0.0, 0.0), label="a")])
    store.update([Observation(at=1.0, geometry=(1.0, 0.0), label="a")])
    # Predicted exactly: one track, two more sightings, still one track.
    store.update([Observation(at=2.0, geometry=(2.0, 0.0), label="a")])
    store.update([Observation(at=3.0, geometry=(3.0, 0.0), label="a")])
    assert len(store.tracks()) == 1
    assert store.tracks()[0].support == 4


def test_two_candidates_in_the_same_place_are_still_ambiguous():
    """The floor must not turn the ambiguity check off."""
    from core.cognition.entity_track import Observation, TrackStore

    store = TrackStore(match_threshold=5.0)
    store.update(
        [
            Observation(at=0.0, geometry=(0.0, 0.0), label="a"),
            Observation(at=0.0, geometry=(0.0, 1.0), label="b"),
        ]
    )
    before = len(store.tracks())
    # Something exactly between them: neither track owns it.
    store.update([Observation(at=1.0, geometry=(0.0, 0.5), label="?")])
    assert len(store.tracks()) > before
    assert store.report()["ambiguous_associations_refused"] >= 1
