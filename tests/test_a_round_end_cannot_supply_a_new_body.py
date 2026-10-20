"""A current terminal screen cannot become another control experiment."""
import asyncio
import time
from types import SimpleNamespace

import numpy as np
import pytest

from core.agency.navigation_before_body_experiments import (
    capture_before_reading, keep_navigation_reading,
    navigation_before_body_experiments, words_bound_to_capture,
)
from tests.test_current_navigation_precedes_body_experiments import frame, menu

pytestmark = pytest.mark.unit
ENDED = 'the screen says the round is over'


def test_current_terminal_pixels_have_a_registered_measurement():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'current_round_ends_prevent_body_experiments')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())


@pytest.mark.parametrize('text', ['TRY AGAIN', 'GAME OVER', 'You win!', 'Round complete', 'TRY ABAIN'])
@pytest.mark.parametrize('body', [None, 7])
def test_current_terminal_pixels_end_the_stretch_even_when_present_at_its_start(text, body):
    run, hers, pixels, _ = menu(text)
    hers.number = body
    keep = {}
    for sequence in range(1, 5):
        at = frame(10. + .05 * sequence, sequence=sequence)
        assert navigation_before_body_experiments(run, hers, pixels, at, keep, now=float(at) + .01) == ENDED
    assert hers.number == body and not keep.get('pixel_navigation_phase')


@pytest.mark.parametrize('text', ['If you miss, try again', 'Avoid a crash', 'You can win!', 'Your turn. Try again'])
def test_instructions_and_current_turns_cannot_supply_a_terminal_phase(text):
    run, hers, pixels, at = menu(text)
    assert navigation_before_body_experiments(run, hers, pixels, at, {}, now=10.1) != ENDED


@pytest.mark.parametrize('change', ['pixels', 'surface', 'epoch', 'sequence', 'time', 'unmeasured'])
def test_old_or_unavailable_terminal_pixels_do_not_end_current_play(change):
    run, hers, pixels, at = menu('GAME OVER')
    now = 10.1
    if change == 'pixels':
        pixels[:] = 60
    elif change == 'surface':
        at = frame(surface='other')
    elif change == 'epoch':
        at = frame(epoch=2)
    elif change == 'sequence':
        at = frame(sequence=0)
    elif change == 'time':
        now = 11.
    else:
        at = 10.
    assert navigation_before_body_experiments(run, hers, pixels, at, {}, now=now) != ENDED


def test_a_changed_result_patch_is_not_outvoted_by_unchanged_readouts():
    pixels = np.zeros((100, 100, 3), np.uint8)
    regions = []
    for index, text in enumerate(['GAME OVER', 'Score: 50', 'Time: 30', 'Level: 2', 'Music: on']):
        y = .02 + index * .19
        pixels[int(100*y):int(100*(y+.15)), 10:90:2] = 240
        regions.append(dict(text=text, x=.1, y=y, width=.8, height=.15))
    run = SimpleNamespace()
    keep_navigation_reading(run, words_bound_to_capture(regions, capture_before_reading(pixels, frame())))
    pixels[2:17] = 60
    hers = SimpleNamespace(number=None, follows_pointer=False)
    assert navigation_before_body_experiments(run, hers, pixels, frame(10.1, sequence=2), {}, now=10.2) != ENDED


def test_a_completed_empty_reading_retires_the_terminal_phase():
    run, hers, pixels, at = menu('GAME OVER')
    keep_navigation_reading(run, words_bound_to_capture([], capture_before_reading(pixels, at)))
    assert navigation_before_body_experiments(run, hers, pixels, at, {}, now=10.1) != ENDED


def test_an_end_fragment_inside_a_conditional_reading_cannot_end_play():
    run, hers, pixels, at = menu()
    regions = [dict(text=text, x=.25, y=.25, width=.5, height=.5)
               for text in ('If you lose it says', 'GAME OVER')]
    keep_navigation_reading(run, words_bound_to_capture(regions, capture_before_reading(pixels, at)))
    assert navigation_before_body_experiments(run, hers, pixels, at, {}, now=10.1) != ENDED


def test_a_delayed_visual_guess_cannot_be_adopted_from_a_current_loss_screen(monkeypatch):
    from core.agency.a_body_seen_then import ABodySeenThen
    from core.agency.naming_what_she_sees import SeeingInPlay, _who_i_am_to_my_eyes
    from core.agency.which_one_answers_to_her import WhichIsHers
    from core.cognition.a_guide_to_a_place import Guide, TOLD
    from core.perception.where_i_am_on_screen import Avatar
    from tests.test_a_body_guess_keeps_its_observed_track import _body, _moves

    run, _, pixels, at = menu('TRY AGAIN')
    run.keys, run.trying = ['left', 'right'], 20
    run.presses = SimpleNamespace(moves_anything=lambda _px: True)
    guide = Guide()
    guide.take_in(TOLD, 'Use the arrow keys to move.')
    body = _body(at=at)
    moves = _moves(body, at=at)
    observation = ABodySeenThen(moves, at)
    avatar = Avatar('body', (.1, .1, .2, .2))
    observation.bind(avatar)
    seeing = SeeingInPlay(began=0., avatar=avatar, avatar_observation=observation)
    hers = WhichIsHers()
    meeting = SimpleNamespace(stance=lambda _kind: '', known=lambda _kind: False)
    monkeypatch.setattr(time, 'monotonic', lambda: 10.1)
    _who_i_am_to_my_eyes(run, seeing, guide, moves, hers, meeting, pixels, at, lambda *_args: None)
    assert hers.number is None and seeing.avatar is None and seeing.avatar_observation is None


@pytest.mark.asyncio
async def test_an_answer_returned_after_terminal_cancellation_cannot_restore_the_guess(monkeypatch):
    from core.agency.naming_what_she_sees import SeeingInPlay, _who_i_am_to_my_eyes
    from core.agency.which_one_answers_to_her import WhichIsHers
    from core.cognition.a_guide_to_a_place import Guide, TOLD
    from core.cognition.guidance_lifetime import guidance_work
    from core.perception import where_i_am_on_screen
    from core.perception.where_i_am_on_screen import Avatar
    from tests.test_a_body_guess_keeps_its_observed_track import _body, _moves

    finish = asyncio.Event()
    async def eyes(*_args):
        try:
            await finish.wait()
        except asyncio.CancelledError:
            await finish.wait()
        return Avatar('body', (.1, .1, .2, .2))
    monkeypatch.setattr(where_i_am_on_screen, 'where_the_avatar_is', eyes)
    guide = Guide()
    guide.take_in(TOLD, 'Use the arrow keys to move.')
    began = time.monotonic() - 1.
    seeing = SeeingInPlay(began=began - 3.)
    run = SimpleNamespace(keys=['left', 'right'])
    hers = WhichIsHers()
    body = _body(at=began)
    async with guidance_work():
        _who_i_am_to_my_eyes(run, seeing, guide, _moves(body, at=began), hers, None,
                            np.zeros((100, 100, 3), np.uint8), began, lambda *_args: None)
        pending = seeing.avatar_asking
        await asyncio.sleep(0)
        _, _, pixels, _ = menu('TRY AGAIN')
        at = frame(time.monotonic() - .01, sequence=2)
        regions = [dict(text='TRY AGAIN', x=.25, y=.25, width=.5, height=.5)]
        keep_navigation_reading(run, words_bound_to_capture(regions, capture_before_reading(pixels, at)))
        _who_i_am_to_my_eyes(run, seeing, guide, _moves(body, at=at), hers, None,
                            pixels, at, lambda *_args: None)
        await asyncio.sleep(0)
        finish.set()
        await pending
        assert seeing.avatar is None and seeing.avatar_asking is None and seeing.avatar_observation is None
        assert hers.number is None


@pytest.mark.asyncio
async def test_the_actor_returns_an_initial_end_screen_before_its_keyboard_trials():
    from core.agency.playing_as_it_happens import play_as_it_happens

    _, _, pixels, _ = menu('GAME OVER')
    sequence, inputs = 0, []
    async def look():
        nonlocal sequence
        await asyncio.sleep(.01)
        sequence += 1
        return pixels.copy(), frame(time.monotonic() - .01, sequence=sequence)
    async def down(key):
        inputs.append(('down', key))
    async def up(key):
        inputs.append(('up', key))
    regions = [dict(text='GAME OVER', x=.25, y=.25, width=.5, height=.5)]
    result = await play_as_it_happens(look, SimpleNamespace(down=down, up=up), keys=['a', 'b'],
                                    seconds=3., read_words=lambda _bgr: regions, keep={})
    assert result['ended'] == ENDED
    assert sum(kind == 'down' for kind, _key in inputs) < 8
    assert sum(kind == 'down' for kind, _key in inputs) == sum(kind == 'up' for kind, _key in inputs)
    assert not result['pixel_search']['attempted']
