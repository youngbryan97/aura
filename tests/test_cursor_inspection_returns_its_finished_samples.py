"""A slow cursor query cannot discard the physical samples already measured."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.perception import where_the_pointer_turns as cursor


class _SlowPage:
    viewport_size = {'width': 1000, 'height': 700}
    url = 'https://example.invalid/surface'

    def __init__(self, uniform=None, change=None):
        self.mouse = SimpleNamespace(move=self.move)
        self.at, self.read, self.uniform, self.change = (0, 0), [], uniform, change
        self.stalled = asyncio.Event()

    async def move(self, x, y):
        self.at = x, y

    async def evaluate(self, script, point):
        assert len(point) == 2  # This fixture requires the physical mouse.
        if len(self.read) == 6:
            if self.change == 'viewport':
                self.viewport_size = {'width': 2000, 'height': 700}
            elif self.change == 'address':
                self.url += '/next'
            self.stalled.set()
            await asyncio.Event().wait()
        x, y = point
        assert (x, y) == self.at
        shown = self.uniform if self.uniform is not None else ('pointer' if x < 100 else 'auto')
        self.read.append((x, y, shown))
        return shown


@pytest.mark.asyncio
async def test_a_deadline_returns_finished_physical_hits_without_inventing_the_unread_rest(monkeypatch):
    page = _SlowPage()
    cursor._NEEDS_THE_MOUSE.add(page)
    monkeypatch.setattr(cursor, 'SWEEP_WITHIN_S', .04)
    places = await asyncio.wait_for(cursor.where_the_pointer_turns(page), .5)
    assert places and len(page.read) == 6 and page.stalled.is_set()
    actual = {(x/1000, y/700) for x, y, shown in page.read if shown == 'pointer'}
    assert set(places) <= actual
    assert cursor._CURSOR_SAMPLES.get() is None
    assert cursor._CURSOR_OBSERVER.get() is None
    assert cursor._partial_cursor_invariant() == ()


@pytest.mark.asyncio
@pytest.mark.parametrize('uniform', ['auto', 'pointer'])
async def test_partial_uniform_samples_cannot_localize_a_target(monkeypatch, uniform):
    page = _SlowPage(uniform)
    cursor._NEEDS_THE_MOUSE.add(page)
    monkeypatch.setattr(cursor, 'SWEEP_WITHIN_S', .04)
    assert await cursor.where_the_pointer_turns(page) == []


@pytest.mark.asyncio
async def test_an_observer_receives_each_finished_physical_sample_before_the_next_move(monkeypatch):
    page = _SlowPage()
    cursor._NEEDS_THE_MOUSE.add(page)
    monkeypatch.setattr(cursor, 'SWEEP_WITHIN_S', .04)
    observed = []

    async def at_the_sample(x, y, shown):
        assert (x, y) == page.at
        assert page.read[-1] == (x, y, shown)
        observed.append((x, y, shown))

    await cursor.where_the_pointer_turns(page, on_sample=at_the_sample)
    assert observed == page.read and len(observed) == 6
    assert cursor._CURSOR_OBSERVER.get() is None


@pytest.mark.asyncio
async def test_a_stalled_sample_capture_spends_the_existing_sweep_budget(monkeypatch):
    page = _SlowPage()
    cursor._NEEDS_THE_MOUSE.add(page)
    monkeypatch.setattr(cursor, 'SWEEP_WITHIN_S', .04)

    async def stalled(*args):
        await asyncio.Event().wait()

    assert await asyncio.wait_for(cursor.where_the_pointer_turns(page, on_sample=stalled), .5) == []
    assert len(page.read) == 1  # The unchecked rest is neither sampled nor authorized.
    assert cursor._CURSOR_OBSERVER.get() is None


@pytest.mark.asyncio
async def test_a_partial_physical_sweep_covers_the_view_and_keeps_results_in_requested_order():
    class Page:
        def __init__(self):
            self.moves = []
            self.mouse = SimpleNamespace(move=self.move)

        async def move(self, x, y):
            self.moves.append((x, y))

        async def evaluate(self, script, point):
            return str(tuple(point))

    page = Page()
    cursor._NEEDS_THE_MOUSE.add(page)
    points = [(x, y) for y in (0., 1., 2., 50., 99., 100.) for x in (0., 1., 2., 50., 99., 100.)]
    result = await cursor._shown_at_each(page, points)
    assert set(page.moves[:4]) == {(0., 0.), (100., 0.), (0., 100.), (100., 100.)}
    assert result == [str(point) for point in points]


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['viewport', 'address'])
async def test_partial_samples_from_a_changed_coordinate_frame_are_discarded(monkeypatch, change):
    page = _SlowPage(change=change)
    cursor._NEEDS_THE_MOUSE.add(page)
    monkeypatch.setattr(cursor, 'SWEEP_WITHIN_S', .04)
    assert await cursor.where_the_pointer_turns(page) == []


@pytest.mark.asyncio
async def test_external_cancellation_stops_the_look_instead_of_returning_partial_authority(monkeypatch):
    page = _SlowPage()
    cursor._NEEDS_THE_MOUSE.add(page)
    monkeypatch.setattr(cursor, 'SWEEP_WITHIN_S', 5.)
    task = asyncio.create_task(cursor.where_the_pointer_turns(page))
    await asyncio.wait_for(page.stalled.wait(), .5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cursor._CURSOR_SAMPLES.get() is None


@pytest.mark.asyncio
async def test_a_batch_claim_cannot_supply_unchecked_hits_when_validation_stalls(monkeypatch):
    class _Unverified(_SlowPage):
        async def evaluate(self, script, point):
            if len(point) == 1:
                return ['pointer']*len(point[0])
            self.stalled.set()
            await asyncio.Event().wait()

    page = _Unverified()
    monkeypatch.setattr(cursor, 'SWEEP_WITHIN_S', .04)
    assert await cursor.where_the_pointer_turns(page) == []
