import pytest

from core import phantom_browser as phantom_module
from core.phantom_browser import PhantomBrowser
from core.runtime.errors import get_degradation_tracker


class _FakePage:
    pass


class _FakeContext:
    async def new_page(self):
        return _FakePage()


class _FakeBrowser:
    async def new_context(self, **_kwargs):
        return _FakeContext()


class _Launcher:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.launched = False

    async def launch(self, **_kwargs):
        if self.fail:
            raise RuntimeError("browser unavailable")
        self.launched = True
        return _FakeBrowser()


class _FakePlaywright:
    def __init__(self, *, firefox_fail=True, chromium_fail=False):
        self.firefox = _Launcher(fail=firefox_fail)
        self.webkit = _Launcher(fail=True)
        self.chromium = _Launcher(fail=chromium_fail)
        self.stopped = False

    async def stop(self):
        self.stopped = True


class _AsyncPlaywrightFactory:
    def __init__(self, playwright):
        self.playwright = playwright

    async def start(self):
        return self.playwright


@pytest.fixture(autouse=True)
def _reset_tracker():
    get_degradation_tracker().reset()
    yield
    get_degradation_tracker().reset()


@pytest.mark.asyncio
async def test_ensure_ready_fails_closed_when_playwright_missing(monkeypatch):
    monkeypatch.setattr(phantom_module, "PLAYWRIGHT_AVAILABLE", False)

    browser = PhantomBrowser()

    assert await browser.ensure_ready() is False
    assert browser.get_status()["active"] is False
    assert browser.get_status()["startup_failure_count"] == 1
    last = get_degradation_tracker().recent(subsystem="phantom_browser")[-1]
    assert last.action == "kept phantom browser inactive because Playwright is unavailable"


@pytest.mark.asyncio
async def test_browser_startup_falls_back_to_chromium(monkeypatch):
    fake_playwright = _FakePlaywright(firefox_fail=True, chromium_fail=False)
    monkeypatch.setattr(phantom_module, "PLAYWRIGHT_AVAILABLE", True)
    monkeypatch.setattr(phantom_module, "STEALTH_AVAILABLE", False)
    monkeypatch.setattr(
        phantom_module,
        "async_playwright",
        lambda: _AsyncPlaywrightFactory(fake_playwright),
    )

    browser = PhantomBrowser(browser_type="firefox")

    assert await browser.ensure_ready() is True
    assert browser.get_status()["active"] is True
    assert browser.get_status()["last_launch_attempts"] == ["firefox", "chromium"]
    assert fake_playwright.chromium.launched is True
    last = get_degradation_tracker().recent(subsystem="phantom_browser")[-1]
    assert last.action == "trying next browser fallback after launch attempt failed"


@pytest.mark.asyncio
async def test_browser_startup_cleans_up_after_all_launches_fail(monkeypatch):
    fake_playwright = _FakePlaywright(firefox_fail=True, chromium_fail=True)
    monkeypatch.setattr(phantom_module, "PLAYWRIGHT_AVAILABLE", True)
    monkeypatch.setattr(
        phantom_module,
        "async_playwright",
        lambda: _AsyncPlaywrightFactory(fake_playwright),
    )

    browser = PhantomBrowser(browser_type="firefox")

    assert await browser.ensure_ready() is False
    assert browser.get_status()["active"] is False
    assert browser.playwright is None
    assert fake_playwright.stopped is True
    assert browser.get_status()["last_launch_attempts"] == ["firefox", "chromium"]
    last = get_degradation_tracker().recent(subsystem="phantom_browser")[-1]
    assert (
        last.action
        == "marked phantom browser inactive and released startup resources after startup failed"
    )
