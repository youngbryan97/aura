"""A silent browser returns unavailable while other control-loop tasks can run."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.perception import a_browser_picture_has_a_deadline as deadline
from core.perception import frames_as_they_are_drawn as frames

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture(autouse=True)
def short_budget(monkeypatch):
    monkeypatch.setattr(deadline, 'PICTURE_REQUEST_S', .015)


class Browser:
    def __init__(self, blocked):
        self.blocked = set(blocked)
        self.cancelled = []
        self.detached = False
        self.context = SimpleNamespace(new_cdp_session=self.new_cdp_session)

    async def respond(self, step, value=None):
        if step in self.blocked:
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled.append(step)
        return value

    async def new_cdp_session(self, _page):
        return await self.respond('session', self)

    async def send(self, method, params=None):
        return await self.respond(method, {})

    async def evaluate(self, script, *args):
        return await self.respond('evaluate', [320, 200])

    async def screenshot(self, **kwargs):
        return await self.respond('screenshot')

    async def detach(self):
        await self.respond('detach')
        self.detached = True

    def on(self, *_args):
        pass


@pytest.mark.parametrize('blocked', ['session', 'Page.enable', 'evaluate', 'Page.startScreencast'])
async def test_an_unanswered_startup_step_cannot_hold_the_next_picture(blocked):
    browser = Browser([blocked])
    reader = frames.PageFrames(browser)
    assert await asyncio.wait_for(reader.look({}), .5) is None
    assert reader.unavailable and reader._session is None
    assert blocked in browser.cancelled
    assert browser.detached or blocked == 'session'


async def test_stalled_resampling_closes_its_session_without_waiting_forever(monkeypatch):
    browser = Browser(['Page.stopScreencast'])
    reader = frames.PageFrames(browser)
    reader._session = browser
    monkeypatch.setattr(reader._cadence, 'next_stride', lambda: 2)
    assert not await asyncio.wait_for(reader._sample_at_the_drawings_pace(), .5)
    assert reader.unavailable and reader._session is None and browser.detached
    assert not reader._resampling


async def test_a_stalled_stop_still_attempts_detachment_and_releases_local_ownership():
    browser = Browser(['Page.stopScreencast', 'detach'])
    reader = frames.PageFrames(browser)
    reader._session = browser
    await asyncio.wait_for(reader.close(), .5)
    assert reader._session is None and not reader._acks
    assert browser.cancelled == ['Page.stopScreencast', 'detach']


async def test_unanswered_frame_acknowledgements_finish_their_local_tasks():
    browser = Browser(['Page.screencastFrameAck'])
    reader = frames.PageFrames(browser)
    reader._session = browser
    reader._frame({'metadata': {}, 'sessionId': 1})
    pending = tuple(reader._acks)
    assert len(pending) == 1
    await asyncio.wait_for(asyncio.gather(*pending), .5)
    assert not reader._acks and browser.cancelled == ['Page.screencastFrameAck']


async def test_a_stalled_canvas_read_never_publishes_pixels_or_renews_capture_time():
    browser = Browser(['evaluate'])
    reader = frames.CanvasFrames(browser)
    reader._observing = True
    assert await asyncio.wait_for(reader.look({}), .5) is None
    assert reader.unavailable and browser.cancelled == ['evaluate']


async def test_both_screenshot_paths_can_fail_without_blocking_other_loop_work(monkeypatch):
    from core.perception import a_picture_of_her_page as pictures

    browser = Browser(['Page.captureScreenshot', 'screenshot'])
    async def session(_page):
        return browser
    monkeypatch.setattr(pictures, '_session', session)
    pulse = asyncio.Event()
    async def control_loop():
        await asyncio.sleep(.005)
        pulse.set()
    task = asyncio.create_task(control_loop())
    try:
        assert await asyncio.wait_for(pictures.picture_of(browser), .5) is None
        assert pulse.is_set()
        assert browser.cancelled == ['Page.captureScreenshot', 'screenshot']
    finally:
        await task


async def test_caller_cancellation_keeps_its_cancellation_meaning():
    began = asyncio.Event()
    async def pending():
        began.set()
        await asyncio.Event().wait()
    task = asyncio.create_task(deadline.picture_request(pending()))
    await began.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_picture_request_budget_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert deadline._picture_request_budget_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(test for test in suite.tests() if test.name == 'browser_picture_requests_have_a_finite_budget')
    assert check.predict(None) is True and any(claim.test == check.name for claim in suite.claims())
