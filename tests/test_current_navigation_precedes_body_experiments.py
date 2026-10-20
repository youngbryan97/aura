"""Measured menu pixels interrupt experiments without lending control or success."""
import asyncio
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from core.agency.navigation_before_body_experiments import (
    _current_navigation_invariant,
    _navigation_occurrence_invariant,
    capture_before_reading,
    keep_navigation_reading,
    navigation_before_body_experiments,
    words_bound_to_capture,
)
from core.perception.frames_as_they_are_drawn import CanvasFrameTime

pytestmark = pytest.mark.unit


def frame(when=10., *, surface='menu', sequence=1, epoch=1):
    return CanvasFrameTime(when, lower=when, upper=when + .001, arrived=when + .002,
                           source='canvas-bracket', source_at=None, epoch=epoch,
                           surface=surface, sequence=sequence)


def menu(text='PLAY'):
    picture = np.zeros((64, 64, 3), np.uint8)
    pattern = ((np.indices((32, 32)).sum(axis=0) // 4) % 2).astype(np.uint8)
    picture[16:48, 16:48] = pattern[:, :, None] * np.array((240, 150, 30), np.uint8)
    at = frame()
    regions = [{'text': text, 'x': .25, 'y': .25, 'width': .5, 'height': .5}]
    run = SimpleNamespace()
    keep_navigation_reading(run, words_bound_to_capture(regions, capture_before_reading(picture, at)))
    return run, SimpleNamespace(number=None, follows_pointer=False), picture, at


def test_a_current_navigation_request_has_a_registered_invariant():
    assert _current_navigation_invariant() == ()


@pytest.mark.parametrize('text', ['PLAY', 'Start', 'Continue', 'Next', 'OK', 'Choose difficulty',
                                 'Next Level', 'Play Now!', 'Skip Intro', 'Try again', 'BTART', 'NEX7', 'PL4Y', 'TRY ABAIN'])
def test_current_navigation_can_precede_body_discovery(text):
    run, hers, pixels, at = menu(text)
    assert navigation_before_body_experiments(run, hers, pixels, at, {}, now=10.1)
    assert hers.number is None


@pytest.mark.parametrize('text', ['TIME 30', 'LEVEL 1', 'MUSIC ON', 'Press up to play a note', 'Display'])
def test_instructions_readouts_and_titles_do_not_become_controls(text):
    run, hers, pixels, at = menu(text)
    assert not navigation_before_body_experiments(run, hers, pixels, at, {}, now=10.1)


@pytest.mark.parametrize('at,now', [(frame(surface='other'), 10.1), (frame(epoch=2), 10.1),
                                   (frame(sequence=0), 10.1), (frame(9.9), 10.1),
                                   (frame(), 11.), (frame(), 9.), (10., 10.1)])
def test_old_future_reordered_and_other_surface_frames_cannot_interrupt_play(at, now):
    run, hers, pixels, _ = menu()
    assert not navigation_before_body_experiments(run, hers, pixels, at, {}, now=now)


def test_completed_ocr_keeps_immutable_pixels_from_before_its_worker_ran():
    run, hers, pixels, at = menu()
    assert run.navigation_reading.words.still_in(pixels) is True
    pixels[:] = 60
    assert not navigation_before_body_experiments(run, hers, pixels, at, {}, now=10.1)
    assert run.navigation_reading.words.still_in(pixels) is False


def test_damaged_navigation_text_still_requires_the_original_word_pixels():
    run, hers, pixels, at = menu('NEX7')
    assert navigation_before_body_experiments(run, hers, pixels, at, {}, now=10.1)
    pixels[:] = 60
    assert not navigation_before_body_experiments(run, hers, pixels, frame(10.1, sequence=2), {}, now=10.2)


def test_a_recent_frame_can_confirm_words_from_a_slow_ocr_pass():
    run, hers, pixels, _ = menu()
    assert navigation_before_body_experiments(run, hers, pixels, frame(12., sequence=4), {}, now=12.1)


def test_invalid_word_geometry_or_uniform_pixels_cannot_supply_menu_evidence():
    run, hers, pixels, at = menu()
    for region in [{'text': 'PLAY'}, {'text': 'PLAY', 'x': 5, 'y': .25, 'width': .5, 'height': .5}]:
        keep_navigation_reading(run, words_bound_to_capture([region], capture_before_reading(pixels, at)))
        assert not navigation_before_body_experiments(run, hers, pixels, at, {}, now=10.1)
    pixels[:] = 0
    keep_navigation_reading(run, words_bound_to_capture(
        [{'text': 'PLAY', 'x': .25, 'y': .25, 'width': .5, 'height': .5}], capture_before_reading(pixels, at)))
    assert not navigation_before_body_experiments(run, hers, pixels, at, {}, now=10.1)


def test_unchanged_menu_budget_survives_actor_reentry_and_colour_flashing():
    run, hers, pixels, at = menu()
    keep = {}
    assert navigation_before_body_experiments(run, hers, pixels, at, keep, now=10.1)
    assert not navigation_before_body_experiments(run, hers, pixels, at, keep, now=10.1)
    assert navigation_before_body_experiments(run, hers, pixels, frame(10.1, sequence=2), keep, now=10.2)
    other, _, _, _ = menu()
    other.navigation_reading = replace(run.navigation_reading)
    assert not navigation_before_body_experiments(other, hers, pixels, frame(10.2, sequence=3), keep, now=10.3)
    assert not navigation_before_body_experiments(other, hers, (pixels * .8).astype(np.uint8),
                                                 frame(10.3, sequence=4), keep, now=10.4)


def test_a_later_identical_button_has_a_new_budget_only_after_measured_disappearance():
    assert _navigation_occurrence_invariant() == ()


def test_an_empty_read_or_one_changed_frame_cannot_renew_an_unchanged_menu():
    run, hers, pixels, at = menu()
    keep = {}
    assert navigation_before_body_experiments(run, hers, pixels, at, keep, now=10.1)
    assert navigation_before_body_experiments(run, hers, pixels, frame(10.1, sequence=2), keep, now=10.2)
    reading = run.navigation_reading
    run.navigation_reading = None
    assert not navigation_before_body_experiments(run, hers, np.full_like(pixels, 60),
                                                 frame(10.2, sequence=3), keep, now=10.3)
    run.navigation_reading = reading
    assert not navigation_before_body_experiments(run, hers, pixels, frame(10.3, sequence=4), keep, now=10.4)


def test_out_of_order_changed_frames_do_not_retire_the_menu_occurrence():
    run, hers, pixels, at = menu()
    keep = {}
    assert navigation_before_body_experiments(run, hers, pixels, at, keep, now=10.1)
    assert navigation_before_body_experiments(run, hers, pixels, frame(10.1, sequence=2), keep, now=10.2)
    reading = run.navigation_reading
    run.navigation_reading = None
    for when in (10.2, 10.3):
        assert not navigation_before_body_experiments(run, hers, np.full_like(pixels, 60),
                                                     frame(when, sequence=2), keep, now=when + .1)
    run.navigation_reading = reading
    assert not navigation_before_body_experiments(run, hers, pixels, frame(10.4, sequence=3), keep, now=10.5)


@pytest.mark.parametrize('field,value', [('number', 3), ('follows_pointer', True)])
def test_an_established_body_keeps_playing(field, value):
    run, hers, pixels, at = menu()
    setattr(hers, field, value)
    assert not navigation_before_body_experiments(run, hers, pixels, at, {}, now=10.1)


def test_a_completed_empty_reading_retires_the_old_menu():
    run, hers, pixels, at = menu()
    keep_navigation_reading(run, words_bound_to_capture([], capture_before_reading(pixels, at)))
    assert not navigation_before_body_experiments(run, hers, pixels, at, {}, now=10.1)


@pytest.mark.asyncio
async def test_actor_hands_a_visible_menu_back_before_exhausting_its_key_trials():
    from core.agency.playing_as_it_happens import play_as_it_happens

    _, _, pixels, _ = menu()
    sequence = 0
    inputs = []

    async def look():
        nonlocal sequence
        await asyncio.sleep(.01)
        sequence += 1
        return pixels.copy(), frame(time.monotonic() - .01, sequence=sequence)

    async def down(key):
        inputs.append(('down', key))

    async def up(key):
        inputs.append(('up', key))

    regions = [{'text': 'PLAY', 'x': .25, 'y': .25, 'width': .5, 'height': .5}]
    result = await play_as_it_happens(look, SimpleNamespace(down=down, up=up), keys=['a', 'b'],
                                      seconds=3., read_words=lambda _bgr: regions, keep={})
    assert result['ended'] == 'a current navigation control is shown'
    assert sum(kind == 'down' for kind, _key in inputs) < 8
    assert sum(kind == 'down' for kind, _key in inputs) == sum(kind == 'up' for kind, _key in inputs)
    assert not result['pixel_search']['attempted']
