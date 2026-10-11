"""Coordinates belong to the observed surface and image, through delayed answers and fresh attempts."""
from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from core.perception import where_the_words_point as places_module
from core.perception.where_the_words_point import PlacesNamed, place_from

pytestmark = pytest.mark.unit

PLACE = "the highlighted destination"
LEFT = '{"bbox_2d": [100, 200, 300, 400]}'
RIGHT = '{"bbox_2d": [600, 500, 800, 700]}'


@pytest.fixture
def encode(monkeypatch):
    from core.perception import where_i_am_on_screen

    monkeypatch.setattr(where_i_am_on_screen, "_as_jpeg", lambda picture: str(int(picture[0, 0, 0])))


def picture(value=0):
    return np.full((8, 12, 3), value, dtype=np.uint8)


async def settle(places):
    await asyncio.gather(*tuple(places._tasks), return_exceptions=True)
    await asyncio.sleep(0)                 # task completion releases the bounded request slots


@pytest.mark.asyncio
async def test_an_answer_from_scene_a_cannot_replace_the_answer_from_scene_b_even_without_cancellation(encode):
    started, release = asyncio.Event(), asyncio.Event()

    async def see(_prompt, image):
        if image == "0":
            started.set()
            await release.wait()
            return LEFT
        return RIGHT

    places = PlacesNamed(see=see)
    places.remember(picture(), surface="editor", episode="first")
    # A request outside the tracked task table cannot be cancelled by a scene change. Its result must still be rejected.
    old = asyncio.create_task(places._look(PLACE, picture(), places.identity))
    await asyncio.wait_for(started.wait(), 2)
    places.remember(picture(1))
    assert places.where(PLACE) is None
    await settle(places)
    assert places.where(PLACE) == (0.7, 0.6)
    release.set()
    await asyncio.wait_for(old, 2)
    assert places.where(PLACE) == (0.7, 0.6)


@pytest.mark.asyncio
async def test_a_normal_request_is_cancelled_and_the_new_scene_has_its_own_answer(encode):
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def see(_prompt, image):
        if image == "0":
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        return RIGHT

    places = PlacesNamed(see=see)
    places.remember(picture())
    assert places.where(PLACE) is None
    await asyncio.wait_for(started.wait(), 2)
    places.remember(picture(1))
    assert places.where(PLACE) is None
    await settle(places)
    assert cancelled.is_set()
    assert places.where(PLACE) == (0.7, 0.6)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["surface", "episode", "viewport"])
async def test_the_same_phrase_on_a_different_context_requires_a_new_picture_and_answer(encode, change):
    calls = []

    async def see(prompt, _image):
        calls.append(prompt)
        return LEFT if len(calls) == 1 else RIGHT

    places = PlacesNamed(see=see)
    context = dict(surface="editor-one", episode="first", viewport=(0, 0, 12, 8))
    places.remember(picture(), **context)
    places.where(PLACE)
    await settle(places)
    assert places.where(PLACE) == (0.2, 0.3)
    context[change] = {"surface": "editor-two", "episode": "second", "viewport": (0, 0, 24, 16)}[change]
    places.set_context(**context)
    assert places.where(PLACE) is None and places.picture is None
    assert len(calls) == 1
    places.remember(picture())
    assert places.where(PLACE) is None
    await settle(places)
    assert places.where(PLACE) == (0.7, 0.6) and len(calls) == 2


@pytest.mark.asyncio
async def test_expired_coordinates_are_unknown_while_the_replacement_answer_is_pending(encode):
    release = asyncio.Event()
    calls = 0

    async def see(_prompt, _image):
        nonlocal calls
        calls += 1
        if calls > 1:
            await release.wait()
        return LEFT if calls == 1 else RIGHT

    places = PlacesNamed(see=see, picture=picture())
    places.where(PLACE)
    await settle(places)
    assert places.where(PLACE) == (0.2, 0.3)
    places.found[PLACE] = ((0.2, 0.3), time.monotonic() - places_module.KEPT_FOR_S)
    assert places.where(PLACE) is None
    release.set()
    await settle(places)
    assert places.where(PLACE) == (0.7, 0.6)


@pytest.mark.asyncio
async def test_equivalent_pixels_in_a_new_array_keep_the_same_revision_and_answer(encode):
    calls = 0

    async def see(_prompt, _image):
        nonlocal calls
        calls += 1
        return LEFT

    places = PlacesNamed(see=see)
    places.remember(picture(), surface="editor", episode="first", viewport=(0, 0, 12, 8))
    places.where(PLACE)
    await settle(places)
    identity = places.identity
    places.remember(picture(), surface="editor", episode="first", viewport=(0, 0, 12, 8))
    assert places.identity == identity
    assert places.where(PLACE) == (0.2, 0.3) and calls == 1


@pytest.mark.asyncio
async def test_a_local_image_changed_in_place_invalidates_its_prior_geometry(encode):
    async def see(_prompt, image):
        return LEFT if image == "0" else RIGHT

    image = picture()
    places = PlacesNamed(see=see)
    places.remember(image)
    places.where(PLACE)
    await settle(places)
    assert places.where(PLACE) == (0.2, 0.3)
    image[:] = 1                         # caller reused its image buffer without replacing the object
    assert places.where(PLACE) is None
    await settle(places)
    assert places.where(PLACE) == (0.7, 0.6)


@pytest.mark.asyncio
async def test_a_changed_local_buffer_rejects_the_delayed_answer_even_before_another_lookup(encode):
    started, release = asyncio.Event(), asyncio.Event()

    async def see(_prompt, _image):
        started.set()
        await release.wait()
        return LEFT

    image = picture()
    places = PlacesNamed(see=see)
    places.remember(image)
    old = asyncio.create_task(places._look(PLACE, image.copy(), places.identity))
    await asyncio.wait_for(started.wait(), 2)
    image[:] = 1
    release.set()
    await asyncio.wait_for(old, 2)
    assert PLACE not in places.found


@pytest.mark.asyncio
async def test_a_request_copies_the_image_before_the_caller_reuses_its_buffer(monkeypatch):
    from core.perception import where_i_am_on_screen

    encoded = []

    def encode(pixels):
        encoded.append(int(pixels[0, 0, 0]))
        return str(encoded[-1])

    async def see(_prompt, _image):
        raise AssertionError("changed pixels must be rejected before vision is called")

    monkeypatch.setattr(where_i_am_on_screen, "_as_jpeg", encode)
    image = picture()
    places = PlacesNamed(see=see, picture=image)
    assert places.where(PLACE) is None
    image[:] = 9
    await settle(places)
    assert encoded == [0] and not places.found


@pytest.mark.asyncio
async def test_movement_waits_for_a_fresh_capture_even_if_the_old_pixels_were_retained(encode):
    async def see(_prompt, _image):
        return LEFT

    places = PlacesNamed(see=see, picture=picture())
    places.where(PLACE)
    await settle(places)
    assert places.where(PLACE) == (0.2, 0.3)
    identity = places.identity
    places.moved()
    assert places.where(PLACE) is None and not places._tasks
    places.remember(picture())
    assert places.identity != identity and places.where(PLACE) is None
    await settle(places)
    assert places.where(PLACE) == (0.2, 0.3)


@pytest.mark.asyncio
async def test_rapid_context_changes_cannot_exceed_the_request_bound_even_when_cancellation_is_delayed(encode):
    release = asyncio.Event()
    started = asyncio.Queue()

    async def see(_prompt, image):
        started.put_nowait(image)
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()         # observer needs time to release its own resource
        return LEFT

    places = PlacesNamed(see=see)
    for index in range(places_module.MOST_LOOKS):
        places.remember(picture(index))
        assert places.where(PLACE) is None
        assert await asyncio.wait_for(started.get(), 2) == str(index)
    places.remember(picture(99))
    assert places.where(PLACE) is None
    assert len(places._tasks) == places_module.MOST_LOOKS and started.empty()
    release.set()
    await settle(places)
    assert not places.found and not places._tasks
    assert places.where(PLACE) is None
    await settle(places)
    assert places.where(PLACE) == (0.2, 0.3)


@pytest.mark.asyncio
async def test_retained_answers_are_bounded_and_unknown_answers_are_cached_without_inventing_coordinates(encode):
    calls = 0

    async def see(_prompt, _image):
        nonlocal calls
        calls += 1
        return '{"bbox_2d": null}'

    places = PlacesNamed(see=see, picture=picture())
    for index in range(places_module.MOST_PLACES + 5):
        assert places.where(f"destination {index}") is None
        await settle(places)
    assert len(places.found) == places_module.MOST_PLACES
    before = calls
    assert places.where(f"destination {places_module.MOST_PLACES + 4}") is None
    assert calls == before and not places._tasks


@pytest.mark.parametrize("answer", [
    '{}', '{"bbox_2d": null}', '{"bbox_2d": [0, 0, 10]}', '{"bbox_2d": [0, 0, 10, 10, 20]}',
    '{"bbox_2d": [0, 0, 0, 10]}', '{"bbox_2d": [0, 20, 10, 10]}', '{"bbox_2d": [-1, 0, 10, 10]}',
    '{"bbox_2d": [0, 0, 1001, 10]}', '{"bbox_2d": [0, 0, NaN, 10]}', '{"bbox_2d": [0, 0, Infinity, 10]}',
    '{"bbox_2d": [false, 0, 10, 10]}', '{"bbox_2d": ["0", 0, 10, 10]}',
    '{"bbox_2d": [0, 0, 10, 10]} {"bbox_2d": [20, 20, 30, 30]}',
])
def test_invalid_or_ambiguous_boxes_have_no_coordinates(answer):
    assert place_from(answer) is None


@pytest.mark.parametrize("image", [None, object(), np.zeros((3, 4)), np.zeros((3, 4, 2)),
                                   np.full((3, 4, 3), float("nan"))])
def test_a_missing_or_unreadable_image_has_no_geometry_or_request(image):
    places = PlacesNamed(picture=image)
    assert places.where(PLACE) is None and not places._tasks


def test_a_viewport_with_nonfinite_coordinates_is_rejected_without_rebinding():
    places = PlacesNamed()
    before = places.identity
    with pytest.raises(ValueError, match="finite"):
        places.set_context(surface="editor", episode="one", viewport=(0, 0, float("nan"), 10))
    assert places.identity == before


def test_registered_invariant_checks_geometry_custody_without_observing_a_screen():
    assert places_module._named_places_invariant() == ()
