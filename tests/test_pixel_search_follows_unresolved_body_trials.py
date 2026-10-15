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
