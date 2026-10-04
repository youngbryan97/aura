from types import SimpleNamespace

import pytest

from core.capabilities import phantom_browser as phantom_module
from core.capabilities.phantom_browser import _KEEPS_RUNNING, PhantomBrowser
from core.runtime.errors import get_degradation_tracker


class _FakePage:
    pass


class _FakeContext:
    def __init__(self):
        self.page_created = False

    async def new_page(self):
        self.page_created = True
        return _FakePage()


class _FakeBrowser:
    def __init__(self):
        self.context = None

    async def new_context(self, **_kwargs):
        self.context = _FakeContext()
        return self.context


class _Launcher:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.launched = False
        self.launch_kwargs = []

    async def launch(self, **kwargs):
        self.launch_kwargs.append(dict(kwargs))
        if self.fail:
            raise RuntimeError("browser unavailable")
        self.launched = True
        return _FakeBrowser()


class _FakePlaywright:
    def __init__(self, *, firefox_fail=True, chromium_fail=False):
        self.firefox = _Launcher(fail=firefox_fail)
        self.webkit = _Launcher(fail=True)
        self.chromium = _Launcher(fail=chromium_fail)
        self.driver = SimpleNamespace(pid=54321, returncode=None)
        self._impl_obj = SimpleNamespace(
            _connection=SimpleNamespace(
                _transport=SimpleNamespace(_proc=self.driver),
            )
        )
        self.stopped = False

    async def stop(self):
        self.stopped = True


class _AsyncPlaywrightFactory:
    def __init__(self, playwright):
        self.playwright = playwright

    async def start(self):
        return self.playwright


@pytest.fixture(autouse=True)
def _reset_tracker(monkeypatch):
    class Hygiene:
        def __init__(self):
            self.registrations = []

        def register_process_handle(self, process, **metadata):
            self.registrations.append((process, metadata))

    hygiene = Hygiene()
    monkeypatch.setattr(phantom_module, "get_runtime_hygiene", lambda: hygiene)
    get_degradation_tracker().reset()
    yield hygiene
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
async def test_browser_uses_installed_chrome_when_playwright_cache_is_stale(
    monkeypatch,
    tmp_path,
):
    fake_playwright = _FakePlaywright(firefox_fail=True, chromium_fail=False)
    missing_bundle = tmp_path / "missing-playwright-chromium"
    system_chrome = tmp_path / "Google Chrome"
    system_chrome.write_text("browser", encoding="utf-8")
    system_chrome.chmod(0o755)
    fake_playwright.chromium.executable_path = str(missing_bundle)
    monkeypatch.setattr(phantom_module, "PLAYWRIGHT_AVAILABLE", True)
    monkeypatch.setattr(
        phantom_module,
        "_SYSTEM_CHROMIUM_EXECUTABLES",
        (str(system_chrome),),
    )
    monkeypatch.setattr(
        phantom_module,
        "async_playwright",
        lambda: _AsyncPlaywrightFactory(fake_playwright),
    )

    browser = PhantomBrowser(browser_type="chromium")

    assert await browser.ensure_ready() is True
    assert fake_playwright.chromium.launch_kwargs == [
        {
            "headless": True,
            "args": ["--disable-blink-features=AutomationControlled", *_KEEPS_RUNNING],
            "timeout": PhantomBrowser.LAUNCH_TIMEOUT_S * 1000.0,
            "executable_path": str(system_chrome),
        }
    ]
    status = browser.get_status()
    assert status["executable_launched"] == str(system_chrome)
    assert status["last_executable_attempts"] == [str(system_chrome)]


@pytest.mark.asyncio
async def test_browser_applies_current_stealth_api_before_page_creation(
    monkeypatch,
    _reset_tracker,
):
    fake_playwright = _FakePlaywright(firefox_fail=False)
    observations = []

    class FakeStealth:
        async def apply_stealth_async(self, context):
            observations.append(("stealth", context.page_created))

    monkeypatch.setattr(phantom_module, "PLAYWRIGHT_AVAILABLE", True)
    monkeypatch.setattr(phantom_module, "STEALTH_AVAILABLE", True)
    monkeypatch.setattr(phantom_module, "_STEALTH", FakeStealth())
    monkeypatch.setattr(
        phantom_module,
        "async_playwright",
        lambda: _AsyncPlaywrightFactory(fake_playwright),
    )

    browser = PhantomBrowser(browser_type="firefox")

    assert await browser.ensure_ready() is True
    assert observations == [("stealth", False)]
    assert browser.get_status()["stealth_applied"] is True
    assert browser.get_status()["stealth_error"] == ""
    assert browser.get_status()["driver_registered"] is True
    assert browser.get_status()["driver_pid"] == 54321
    assert _reset_tracker.registrations == [
        (
            fake_playwright.driver,
            {
                "kind": "subprocess",
                "name": "playwright.driver",
                "source": "core.capabilities.phantom_browser",
                "command": "playwright driver",
            },
        )
    ]


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
