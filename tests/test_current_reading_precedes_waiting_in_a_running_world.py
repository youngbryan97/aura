"""A previous menu cannot delay measuring a world whose clock keeps moving."""
import time
from types import SimpleNamespace

import pytest

from core.cognition.a_guide_to_a_place import Guide, SCREEN, THE_GUIDE
from core.skills import screen_pursuit_as_it_happens as play
from core.skills.screen_pursuit_on_a_page import OnAPage

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def no_optional_scene_worker(monkeypatch):
    from core.perception import what_a_scene_is

    monkeypatch.setattr(what_a_scene_is, 'glance_beside', lambda *_args, **_kwargs: None)


@pytest.mark.parametrize('words', [
    'Use the arrow keys to guide a robot safely onto its charging platform.',
    'Catch the falling fruit before it hits the ground.',
    'Hold out until help arrives.',
])
@pytest.mark.parametrize('current', ['FUEL 73', 'NEXT'])
@pytest.mark.asyncio
async def test_a_new_screen_is_read_before_waiting_even_when_the_previous_screen_was_a_menu(monkeypatch, words, current):
    from core.perception import what_her_page_shows

    guide = Guide(place='a moving environment')
    guide.take_in(SCREEN, words)
    guide_token = THE_GUIDE.set(guide)
    page = object()
    reflex = play.PlayingAsItHappens(page, (0., 0., 1., 1.), 'Proceed', time.monotonic()+30.)
    reflex.read({'text': 'PLAY', 'layout': [{'text': 'PLAY'}]})
    assert reflex.way_on_shown and not play.a_screen_of_play_in_time(reflex)
    events = []

    async def wait(**_kwargs):
        events.append('wait')

    async def photograph(*_args, **_kwargs):
        events.append('read')
        return {'text': current, 'layout': [{'text': current}]}

    async def cursor(*_args, **_kwargs):
        events.append('cursor')

    on = OnAPage(page, 'owned view')
    on.watching = SimpleNamespace(watch=wait, acts=[])
    on._watching_over = (0., 0., 1., 1.)
    monkeypatch.setattr(what_her_page_shows, 'look_at_a_page', photograph)
    monkeypatch.setattr(on, '_where_the_pointer_turns', cursor)
    token = play.AS_IT_HAPPENS.set(reflex)
    try:
        observed = await on.read(on._watching_over)
        assert events[0] == 'read' and 'wait' not in events
        reflex.read(observed)
        assert reflex.way_on_shown is (current == 'NEXT')
    finally:
        play.AS_IT_HAPPENS.reset(token)
        THE_GUIDE.reset(guide_token)


def test_current_reading_priority_is_a_registered_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite
    from core.skills.screen_pursuit_on_a_page import _read_before_scene_wait_invariant

    assert _read_before_scene_wait_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'current_reading_precedes_waiting_in_a_running_world')
    assert check.predict(None) and any(c.test == check.name for c in suite.claims())


@pytest.mark.parametrize('words', ['', 'YOUR TURN: move a piece.', 'Drag the pieces into place.'])
@pytest.mark.asyncio
async def test_an_unknown_or_waiting_world_keeps_its_scene_wait(words):
    guide = Guide(place='a waiting environment')
    if words:
        guide.take_in(SCREEN, words)
    guide_token = THE_GUIDE.set(guide)
    reflex = play.PlayingAsItHappens(object(), (0., 0., 1., 1.), 'Proceed', time.monotonic()+30.)
    reflex.read({'text': 'PLAY', 'layout': [{'text': 'PLAY'}]})
    waits = []

    async def wait(**kwargs):
        waits.append(kwargs['at_most_s'])

    on = OnAPage(reflex.page, 'owned view')
    on.watching = SimpleNamespace(watch=wait)
    on._watching_over = (0., 0., 1., 1.)
    token = play.AS_IT_HAPPENS.set(reflex)
    try:
        await on._watch(on._watching_over)
        assert waits == [on.WATCH_IN_A_LOOK_S]
    finally:
        play.AS_IT_HAPPENS.reset(token)
        THE_GUIDE.reset(guide_token)
