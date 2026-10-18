"""Prepare the classical transport before the next action starts a moving phase."""
import time
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from core.cognition.a_guide_to_a_place import Guide, SCREEN, THE_GUIDE
from core.perception.observed_transfer import ScreenReading, make_snapshot
from core.skills import screen_pursuit_as_it_happens as play

pytestmark = pytest.mark.unit


def _menu():
    image = np.zeros((64, 64, 3), np.uint8)
    image[16:48, 16:48] = (((np.indices((32, 32)).sum(axis=0)//4) % 2)*240)[:, :, None]
    return ScreenReading({'text': 'PLAY', 'layout': [{'text': 'PLAY', 'x': .25, 'y': .25,
                         'width': .5, 'height': .5}]}, picture_snapshot=make_snapshot(image))


@pytest.mark.parametrize('mode', ['ready', 'pending', 'failed', 'cancelled'])
@pytest.mark.asyncio
async def test_the_actual_menu_route_prepares_once_and_retires_its_empty_observer(monkeypatch, mode):
    from core.perception import play_observation_without_holding_the_loop as observation
    from core.perception import what_a_scene_is
    from core.skills import current_words_before_play as words

    guide = Guide(place='a clocked environment')
    guide.take_in(SCREEN, 'Catch the falling fruit before it hits the ground.')
    guide_token = THE_GUIDE.set(guide)
    events = []
    monkeypatch.setenv('AURA_PLAY_POINT_BACKEND', 'classical')
    monkeypatch.setenv('AURA_PLAY_EYES_IN_THIS_PROCESS', '0')
    monkeypatch.setattr(what_a_scene_is, 'glance_beside', lambda *_args, **_kwargs: None)

    class EmptyObserver:
        status = mode
        def __init__(self, **kwargs):
            from core.perception.seeing_play_elsewhere import SeenElsewhere
            assert kwargs == {'factory': SeenElsewhere}
            events.append('construct')
        async def start(self):
            events.append('start')
            if mode == 'failed':
                raise RuntimeError('optional transport unavailable')
            if mode == 'cancelled':
                raise asyncio.CancelledError
            return mode == 'ready'
        async def close(self):
            events.append('close')
        async def see(self, *_args):
            pytest.fail('preparation consumed a game picture')

    monkeypatch.setattr(observation, 'PlayObservation', EmptyObserver)
    reflexes = play.PlayingAsItHappens(SimpleNamespace(url='canvas://one'), (0., 0., 1., 1.),
                                     'Catch', time.monotonic()+30.)
    async def nothing(*_args, **_kwargs):
        return None
    async def menu():
        return _menu()
    monkeypatch.setattr(words, 'read_current_words', nothing)
    monkeypatch.setattr(reflexes, 'while_it_moves', nothing)
    monkeypatch.setattr(reflexes, 'read_a_legend', nothing)
    monkeypatch.setattr(reflexes, 'goes_on_playing', lambda *_args: False)
    token = play.AS_IT_HAPPENS.set(reflexes)
    try:
        if mode == 'cancelled':
            with pytest.raises(asyncio.CancelledError):
                await play.looked_at_as_it_happens(menu)
        else:
            await play.looked_at_as_it_happens(menu)
        await play.looked_at_as_it_happens(menu)
        assert events == ['construct', 'start', 'close']
        assert reflexes.way_on_shown
    finally:
        play.AS_IT_HAPPENS.reset(token)
        THE_GUIDE.reset(guide_token)


@pytest.mark.parametrize('reason', ['uncaptured', 'active', 'no_layout', 'no_clock', 'learned', 'local_only'])
@pytest.mark.asyncio
async def test_preparation_cannot_launch_without_its_menu_and_classical_conditions(monkeypatch, reason):
    from core.perception import play_observation_without_holding_the_loop as observation
    from core.skills.current_words_before_play import prepare_classical_observation_on_a_menu

    def forbidden(**_kwargs):
        pytest.fail('empty preparation did not have its conditions')

    monkeypatch.setattr(observation, 'PlayObservation', forbidden)
    monkeypatch.setenv('AURA_PLAY_POINT_BACKEND', 'online_tapir' if reason == 'learned' else 'classical')
    monkeypatch.setenv('AURA_PLAY_EYES_IN_THIS_PROCESS', '1' if reason == 'local_only' else '0')
    reflex = SimpleNamespace(page=SimpleNamespace(url='canvas://one'), band=(0., 0., 1., 1.),
                            way_on_shown=True, words=['Catch the falling fruit before it hits the ground.'], keep={})
    seen = _menu()
    if reason == 'uncaptured':
        seen = dict(seen)
    elif reason == 'active':
        seen['layout'] = [{'text': 'FUEL 73'}]
    elif reason == 'no_layout':
        seen['layout'] = None
    elif reason == 'no_clock':
        reflex.words = ['YOUR TURN: move a piece.']
    token = THE_GUIDE.set(None)
    try:
        await prepare_classical_observation_on_a_menu(reflex, seen)
    finally:
        THE_GUIDE.reset(token)


def test_menu_preparation_is_a_registered_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite
    from core.skills.current_words_before_play import _menu_preparation_invariant

    assert _menu_preparation_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'classical_preparation_requires_a_captured_menu')
    assert check.predict(None) and any(c.test == check.name for c in suite.claims())
