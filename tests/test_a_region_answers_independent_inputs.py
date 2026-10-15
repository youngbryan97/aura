"""Discovery must survive separate trials without selecting on confirmation."""
import numpy as np
import pytest

from core.agency.pixels_inside_control_trials import RandomizedPixelExperiment
from core.perception.a_region_that_answers_independent_inputs import region_from_press_features


def features(seed=42, *, effect=True):
    plan = RandomizedPixelExperiment(("a", "b"), seed=31)
    rng = np.random.default_rng(seed)
    values = rng.normal(0, .05, (24, 8, 10, 3))
    for index, press in enumerate(plan.schedule):
        if effect:
            values[index, 3:5, 4:6, 2] += 1.8 if press.key == "a" else -1.8
    return dict(features=values, counts=np.full((24, 8, 10), 4), camera=np.zeros((24, 2)),
                keys=tuple(p.key for p in plan.schedule), cohorts=tuple(p.cohort for p in plan.schedule),
                first_key="a", second_key="b")


def test_independent_turning_response_supplies_only_a_search_hint():
    result = region_from_press_features(**features())
    assert result is not None
    assert result.bounds == (48, 36, 72, 60)
    assert result.discovery_p <= .01 and result.confirmation_p <= .01
    assert result.status == "search_hint_only" and not hasattr(result, "identity")


@pytest.mark.parametrize("seed", range(8))
def test_independent_nonresponsive_noise_does_not_supply_a_region(seed):
    assert region_from_press_features(**features(seed, effect=False)) is None


def test_confirmation_values_do_not_move_the_discovery_region():
    data = features()
    data["features"][12:, 3:5, 4:6] = 0.
    for index in range(12, 24):
        data["features"][index, 1:3, 1:3, 2] = 4. if data["keys"][index] == "a" else -4.
    assert region_from_press_features(**data) is None


def test_reversed_confirmation_refuses_the_selected_direction():
    data = features()
    data["features"][12:] *= -1
    assert region_from_press_features(**data) is None


def test_two_equally_responsive_regions_are_ambiguous():
    data = features()
    for index, key in enumerate(data["keys"]):
        data["features"][index, 0:2, 0:2, 2] += 1.8 if key == "a" else -1.8
    assert region_from_press_features(**data) is None


def test_action_dependent_scene_translation_cannot_name_local_material():
    data = features()
    data["camera"][:, 0] = [3 if key == "a" else -3 for key in data["keys"]]
    assert region_from_press_features(**data) is None


def test_missing_angular_fit_is_unavailable_instead_of_measured_zero():
    data = features()
    available = np.ones(data["features"].shape, dtype=bool)
    available[..., 2] = False
    assert region_from_press_features(**data, available=available) is None


def test_confirmation_requires_the_selected_channels_current_support():
    data = features()
    available = np.ones(data["features"].shape, dtype=bool)
    available[-1, 3:5, 4:6, 2] = False
    assert region_from_press_features(**data, available=available) is None


def test_translation_can_be_confirmed_without_claiming_unseen_angle():
    data = features()
    data["features"][..., 0] = data["features"][..., 2]
    available = np.ones(data["features"].shape, dtype=bool)
    available[..., 2] = False
    result = region_from_press_features(**data, available=available)
    assert result is not None and result.bounds == (48, 36, 72, 60)


@pytest.mark.parametrize("bad", ["cohort", "shape", "count", "budget", "finite"])
def test_invalid_or_insufficient_trials_cannot_be_scored(bad):
    data = features()
    if bad == "cohort":
        data["cohorts"] = ("discovery",) * 24
    elif bad == "shape":
        data["features"] = data["features"][:8]
    elif bad == "count":
        data["counts"] = data["counts"].astype(float)
    elif bad == "budget":
        data["permutations"] = 2000
    else:
        data["features"][0, 0, 0, 0] = np.nan
    assert region_from_press_features(**data) is None
