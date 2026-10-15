"""Independent visual experiments retain actual pixels and delivery windows."""
from dataclasses import replace

import numpy as np
import pytest

from core.agency.pixels_inside_control_trials import (
    PixelPress,
    RandomizedPixelExperiment,
    freeze_frame,
)
from core.perception.frames_as_they_are_drawn import CanvasFrameTime


def frame(sequence, lower, *, surface="document/canvas", epoch=2, picture=None):
    picture = np.zeros((60, 80, 3), np.uint8) if picture is None else picture
    at = CanvasFrameTime(lower, lower=lower, upper=lower + .01, arrived=lower + .02,
                         source="canvas-bracket", source_at=None, epoch=epoch,
                         surface=surface, sequence=sequence)
    return freeze_frame(picture, at)


def press(experiment, index=0, *, neutral=None):
    at = 1. + index
    return PixelPress(experiment.schedule[index], at + .05, at + .06, at + .4, at + .41,
                      at if neutral is None else neutral,
                      frame(index * 2 + 1, at + .01), frame(index * 2 + 2, at + .25))


def test_schedule_precedes_pixels_and_has_distinct_balanced_cohorts():
    experiment = RandomizedPixelExperiment(("a", "b"), seed=8)
    assert experiment.schedule == RandomizedPixelExperiment(("a", "b"), seed=8).schedule
    assert [p.ordinal for p in experiment.schedule] == list(range(24))
    for cohort in ("discovery", "confirmation"):
        for key in ("a", "b"):
            assert sum(p.key == key and p.cohort == cohort for p in experiment.schedule) == 6
    with pytest.raises(AttributeError):
        experiment.schedule = ()
    with pytest.raises(AttributeError):
        experiment.trials = ()


def test_complete_experiment_uses_distinct_captures_even_for_identical_pixels():
    experiment = RandomizedPixelExperiment(("a", "b"), seed=8)
    for index in range(24):
        assert experiment.record(press(experiment, index))
    assert experiment.complete and experiment.next_press is None
    assert len(experiment.trials) == 24
    assert not experiment.record(press(experiment, 0))


def test_pixels_are_frozen_without_aliasing_a_reused_capture_buffer():
    image = np.full((480, 640, 3), 12, np.uint8)
    frozen = frame(1, 1., picture=image)
    assert frozen.valid() and (frozen.width, frozen.height) == (320, 240)
    image[:] = 220
    assert frozen.rgb().max() == 12
    with pytest.raises(ValueError):
        frozen.rgb()[:] = 0
    assert not replace(frozen, digest="fabricated").valid()


@pytest.mark.parametrize("mutation,reason", [
    (lambda p: replace(p, before=None), "capture_values"),
    (lambda p: replace(p, down_ack=float("nan")), "delivery_values"),
    (lambda p: replace(p, down_ack=p.up_ack), "delivery_order"),
    (lambda p: replace(p, neutral_began=p.before.upper), "before_outside_neutral"),
    (lambda p: replace(p, after=replace(p.after, lower=p.down_ack)), "after_outside_hold"),
    (lambda p: replace(p, after=replace(p.after, epoch=3)), "changed_capture"),
    (lambda p: replace(p, after=replace(p.after, surface="other/canvas")), "changed_capture"),
    (lambda p: replace(p, after=replace(p.after, sequence=p.before.sequence)), "capture_order"),
])
def test_malformed_or_unowned_capture_retires_whole_experiment(mutation, reason):
    experiment = RandomizedPixelExperiment(("a", "b"), seed=8)
    assert not experiment.record(mutation(press(experiment)))
    assert experiment.refused == reason and not experiment.complete
    assert not experiment.record(press(experiment))


def test_failed_trial_cannot_be_omitted_or_relabeled():
    experiment = RandomizedPixelExperiment(("a", "b"), seed=8)
    assert not experiment.record(press(experiment, 1))
    assert experiment.refused == "wrong_schedule" and not experiment.trials


def test_source_context_must_continue_between_distinct_trials():
    experiment = RandomizedPixelExperiment(("a", "b"), seed=8)
    assert experiment.record(press(experiment))
    second = press(experiment, 1)
    changed = replace(second, before=replace(second.before, epoch=3), after=replace(second.after, epoch=3))
    assert not experiment.record(changed)
    assert experiment.refused == "experiment_changed_capture"


def test_capture_sequence_cannot_be_replayed_with_new_press_labels():
    experiment = RandomizedPixelExperiment(("a", "b"), seed=8)
    assert experiment.record(press(experiment))
    second = press(experiment, 1)
    assert not experiment.record(replace(second, before=replace(second.before, sequence=1)))
    assert experiment.refused == "reused_capture"


def test_overlapping_actions_cannot_supply_independent_trials():
    experiment = RandomizedPixelExperiment(("a", "b"), seed=8)
    assert experiment.record(press(experiment))
    assert not experiment.record(press(experiment, 1, neutral=1.2))
    assert experiment.refused == "overlapping_trials"


def test_different_input_exposures_cannot_supply_a_label_contrast():
    experiment = RandomizedPixelExperiment(("a", "b"), seed=8)
    assert experiment.record(press(experiment))
    second = press(experiment, 1)
    assert not experiment.record(replace(second, after=frame(4, 2.36)))
    assert experiment.refused == "unmatched_exposure"


def test_unknown_times_and_unbounded_pixels_do_not_become_captures():
    assert freeze_frame(np.zeros((60, 80, 3), np.uint8), 1.) is None
    assert frame(1, 1., picture=np.zeros((12, 80, 3), np.uint8)) is None
    assert frame(1, 1., picture=np.zeros((60, 80, 4), np.uint8)) is None


@pytest.mark.parametrize("keys,per_key", [(("a", "a"), 12), (("a", "b"), 9), (("a", "b"), 22), (["a", "b"], 12)])
def test_experiment_budget_is_bounded(keys, per_key):
    with pytest.raises(ValueError):
        RandomizedPixelExperiment(keys, seed=8, per_key=per_key)
