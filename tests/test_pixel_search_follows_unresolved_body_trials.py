"""The live actor's search phase follows ordinary delivered body trials."""
from types import SimpleNamespace

import numpy as np
import pytest

from core.agency.how_a_game_works import STEERED, Answer, HowItWorks
from core.agency.searching_before_a_body_is_known import (
    PixelSearchState,
    eligible_pixel_search,
    search_after_unresolved_body,
    search_inputs,
)


def state():
    how = HowItWorks(me=Answer(STEERED, 1.), body_keys_from_code=("left", "right"))
    run = SimpleNamespace(_pixel_search=PixelSearchState(), how=how, keys=["left", "right"],
                          trying=8, held="", body_trial=None)
    hers = SimpleNamespace(number=None, follows_pointer=False)
    moves = SimpleNamespace(seeing=True)
    return run, hers, moves


def test_search_follows_exhausted_press_and_neutral_observation_phases():
    run, hers, moves = state()
    assert eligible_pixel_search(run, hers, moves, 10., 30.) == ("left", "right")
    run.body_trial = SimpleNamespace(key="", until=10.2)
    assert eligible_pixel_search(run, hers, moves, 10., 30.) is None
    assert eligible_pixel_search(run, hers, moves, 10.3, 30.) == ("left", "right")
    run.body_trial = SimpleNamespace(key="left", until=9.)
    assert eligible_pixel_search(run, hers, moves, 10., 30.) is None


@pytest.mark.parametrize("attribute,value", [("trying", 7), ("held", "left"), ("pointer_first", True),
                                            ("pointer_trigger", True), ("_reflex_input", object()),
                                            ("_prepared_input_set", object())])
def test_another_action_phase_retains_priority(attribute, value):
    run, hers, moves = state()
    setattr(run, attribute, value)
    assert eligible_pixel_search(run, hers, moves, 10., 30.) is None


def test_search_is_once_per_actor_state_and_requires_time_to_measure():
    run, hers, moves = state()
    assert eligible_pixel_search(run, hers, moves, 10., 7.) is None
    run._pixel_search.attempted = True
    assert eligible_pixel_search(run, hers, moves, 10., 30.) is None


def test_identified_body_or_unknown_interaction_keeps_ordinary_play():
    run, hers, moves = state()
    hers.number = 12
    assert eligible_pixel_search(run, hers, moves, 10., 30.) is None
    hers.number = None
    run.how = HowItWorks()
    assert eligible_pixel_search(run, hers, moves, 10., 30.) is None


def test_wrapper_is_the_same_owned_receipt_lane_during_ordinary_actions():
    from core.agency.inputs_while_measuring_control import HandsWithControlReceipts

    run, hers, _moves = state()
    hands = object()
    wrapped = search_inputs(hands, run, hers)
    assert isinstance(wrapped, HandsWithControlReceipts) and wrapped.hands is hands
    assert not wrapped.owns()
    run._pixel_search.active = True
    run._pixel_search.may_collect = lambda: True
    assert wrapped.owns()


def test_current_source_body_controls_precede_old_arrow_guesses():
    run, hers, moves = state()
    run.keys = ['left', 'right', 'j', 'k']
    run.trying = 16
    run.how.body_keys_from_code = ('j', 'k')
    assert eligible_pixel_search(run, hers, moves, 10., 30.) == ('j', 'k')


def test_declared_translation_precedes_turning_without_using_key_names():
    run, hers, moves = state()
    run.keys = ['opaque-yaw-a', 'opaque-yaw-b', 'opaque-drive-a', 'opaque-drive-b']
    run.trying = 16
    run.how.body_keys_from_code = tuple(run.keys)
    run.how.translating_keys_from_code = ('opaque-drive-a', 'opaque-drive-b')
    assert eligible_pixel_search(run, hers, moves, 10., 30.) == ('opaque-drive-a', 'opaque-drive-b')


def test_arrow_names_do_not_override_source_acquisition_order():
    run, hers, moves = state()
    run.keys = ['up', 'down', 'left', 'right']
    run.trying = 16
    run.how.body_keys_from_code = tuple(run.keys)
    assert eligible_pixel_search(run, hers, moves, 10., 30.) == ('up', 'down')


def test_only_current_nonzero_source_channels_prioritize_translation():
    from core.agency.how_a_game_works import _translating_controls_from_code
    from core.perception.a_world_as_its_code_makes_it import HerBody

    body = HerBody(paces={'move': (0., 12.), 'obsolete': (3., 0.)},
                   pushes={'idle': (0., 0.)}, lifts={'bad': float('nan')},
                   drives={'drive': 20., 'reverse': -20.}, turns={'yaw': 3.})
    guide = SimpleNamespace(controls={key: object() for key in ('move', 'idle', 'bad', 'drive', 'reverse', 'yaw')},
                            world_its_code_makes=SimpleNamespace(hers=body))
    assert _translating_controls_from_code(guide) == ('move', 'drive', 'reverse')


@pytest.mark.asyncio
async def test_unmeasured_frame_cannot_start_a_physical_intervention():
    run, hers, moves = state()
    wrapped = search_inputs(object(), run, hers)
    async def never():
        raise AssertionError("unmeasured capture started collection")
    assert not await search_after_unresolved_body(wrapped, run, hers, moves, look=never,
        context=SimpleNamespace(check=lambda: None), picture=np.zeros((40, 60, 3), np.uint8),
        at=1., deadline=float("inf"))
    assert not run._pixel_search.attempted


@pytest.mark.asyncio
@pytest.mark.parametrize('intervals,expected', [([.1] * 6, .22), ([.18] * 6, .38), ([], None), ([.3] * 6, None)])
async def test_actor_selects_one_bounded_exposure_from_prior_measured_cadence(monkeypatch, intervals, expected):
    import time

    from core.agency import searching_before_a_body_is_known as search
    from core.agency.collecting_independent_pixel_trials import PixelCollectionResult
    from core.perception.frames_as_they_are_drawn import CanvasFrameTime

    run, hers, moves = state()
    run.intervals = intervals
    wrapped = search_inputs(object(), run, hers)
    requests = []

    async def collect(_experiment, **options):
        requests.append(options)
        # Delivery is not simulated here; this test checks actor configuration.
        return PixelCollectionResult('measurement unavailable in this wiring test', 0, True)

    monkeypatch.setattr(search, 'collect_independent_pixel_trials', collect)
    now = time.monotonic()
    at = CanvasFrameTime(now - .01, lower=now - .01, upper=now - .005, arrived=now - .003,
                         source='canvas-bracket', source_at=None, epoch=1, surface='owned', sequence=1)
    result = await search_after_unresolved_body(wrapped, run, hers, moves, look=None,
        context=SimpleNamespace(check=lambda: None), picture=np.zeros((40, 60, 3), np.uint8),
        at=at, deadline=now + 40.)
    if expected is None:
        assert not result and not requests and not run._pixel_search.attempted
    else:
        assert result and len(requests) == 1
        assert requests[0]['settle_s'] == pytest.approx(expected)
        assert 0 < requests[0]['total_s'] <= 30.
        assert run._pixel_search.settle_s == pytest.approx(expected)
