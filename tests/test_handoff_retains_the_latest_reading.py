"""An instruction page cannot remain the reported current text after room entry."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.perception.the_latest_reading_at_handoff import (
    _latest_reading_invariant,
    keep_latest_reading,
)


def test_historical_words_remain_available_without_claiming_they_are_current():
    held = {}
    keep_latest_reading(held, dict(ok=True, text='Collect the tools', capture_at=1.))
    keep_latest_reading(held, dict(ok=True, text='', capture_at=2.))
    assert held['last_seen'] == '' and held['last_observation']['capture_at'] == 2.
    assert held['last_read_words']['text'] == 'Collect the tools'
    assert _latest_reading_invariant() == ()


@pytest.mark.asyncio
async def test_a_retained_ready_reading_updates_the_real_observer_handoff(monkeypatch):
    from core.skills import screen_pursuit_observing as observing

    async def in_front(*args):
        return True

    monkeypatch.setattr(observing, '_ensure_frontmost', in_front)
    monkeypatch.setattr(observing, '_ensure_page', in_front)
    room = dict(ok=True, text='', layout=[], capture_at=2.)
    run = SimpleNamespace(anchor={'app': 'surface', 'page': 'example', 'settled': True},
        at_rest={'reading': room}, drawn={'where': None, 'last_seen': 'Instructions'}, expect_page='example',
        lost_page={'value': False}, open_page=None, reading_took=[], target_app='')
    assert await observing.observe_the_screen(run) is room
    assert run.drawn['last_seen'] == '' and run.drawn['last_observation']['status'] == 'no_words_read'


def test_the_pursuit_result_keeps_blank_current_text_and_separate_historical_words():
    from core.skills.screen_pursuit_steps import _pursue_on_screen_result

    drawn = {}
    keep_latest_reading(drawn, dict(ok=True, text='Instructions', capture_at=1.))
    keep_latest_reading(drawn, dict(ok=True, text='', capture_at=2.))
    receipt = SimpleNamespace(completed=False, to_dict=lambda: {'completed': False})
    result = _pursue_on_screen_result({'value': False}, {'count': 0}, [], [],
        {'choice': '', 'waits': 0}, receipt, {'count': 0, 'because': ''}, {'value': False}, '', drawn)
    assert result['last_seen'] == '' and result['last_observation']['status'] == 'no_words_read'
    assert result['last_read_words']['text'] == 'Instructions'
    result['last_read_words']['text'] = 'changed result'
    assert drawn['last_read_words']['text'] == 'Instructions'
