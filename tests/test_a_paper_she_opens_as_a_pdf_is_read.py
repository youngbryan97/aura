"""A page her browser goes to that turns out to be a PDF is read as the page she asked for.

LIVE 15-16 November 2026 every arXiv paper she opened as a PDF failed with "Page.goto: Download is starting",
five times: the browser answered the navigation with a download, goto raised, and the file her browser kept was
never read. A stand-in page here does what Chromium does, so the PDF is a real one, read by her attachment reader.
"""
from __future__ import annotations

import pytest

from core.capabilities import phantom_browser
from core.capabilities.browser_authority import BrowserVerdict
from core.capabilities.phantom_browser import PhantomBrowser

pytestmark = pytest.mark.unit


def _pdf(tmp_path, words: str):
    import fitz

    document = fitz.open()
    document.new_page().insert_text((72, 72), words)
    path = tmp_path / "2511.20067.pdf"
    path.write_bytes(document.tobytes())
    return path


class _Download:
    def __init__(self, url, path):
        self.url, self.suggested_filename, self._path = url, path.name, path

    async def path(self):
        return str(self._path)


class _PageThatDownloads:
    """goto that is answered with a file: the download event, then the error Chromium raises."""

    def __init__(self, download=None, *, event_after_the_error=False, error="Page.goto: Download is starting"):
        self.url, self._download, self._heard = "about:blank", download, []
        self._late, self._error = event_after_the_error, error

    def on(self, event, listener):
        if event == "download":
            self._heard.append(listener)

    def remove_listener(self, event, listener):
        self._heard.remove(listener)

    async def goto(self, url, **_options):
        import asyncio

        if self._download is not None:
            for listener in list(self._heard):
                if self._late:
                    asyncio.get_running_loop().call_soon(listener, self._download)
                else:
                    listener(self._download)
        raise phantom_browser.PlaywrightError(self._error)


def _browser(monkeypatch, page, allowed=lambda url: True):
    monkeypatch.setattr(phantom_browser, "authorize_browser_action",
                        lambda action, **kw: BrowserVerdict(allowed(kw.get("url", "")), "test", action, "owner"))
    browser = PhantomBrowser(principal="owner")
    browser.page, browser.is_active = page, True
    return browser


@pytest.mark.asyncio
async def test_a_navigation_answered_with_a_pdf_is_read_as_the_page(monkeypatch, tmp_path):
    url = "https://arxiv.org/pdf/2511.20067"
    page = _PageThatDownloads(_Download(url, _pdf(tmp_path, "Ternary weights at two bits")))
    browser = _browser(monkeypatch, page)
    assert await browser.browse(url)
    assert browser._last_navigation["reason"] == "arrived_at_a_document"
    text = await browser.read_content()
    assert "Ternary weights at two bits" in text and text.startswith("# 2511.20067.pdf")
    assert browser._last_extraction["extraction"] == "document_text"
    assert page._heard == [], "the listener goes when the navigation ends"


@pytest.mark.asyncio
async def test_a_download_heard_after_the_error_is_still_read(monkeypatch, tmp_path):
    """On headless Chromium the download event came after goto's error in 20 of 20 navigations."""
    url = "https://arxiv.org/pdf/2511.20067"
    page = _PageThatDownloads(_Download(url, _pdf(tmp_path, "Ternary weights at two bits")), event_after_the_error=True)
    browser = _browser(monkeypatch, page)
    assert await browser.browse(url)
    assert "Ternary weights at two bits" in await browser.read_content()


@pytest.mark.asyncio
async def test_a_navigation_that_fails_with_no_download_still_fails(monkeypatch):
    browser = _browser(monkeypatch, _PageThatDownloads(error="Page.goto: net::ERR_NAME_NOT_RESOLVED"))
    assert not await browser.browse("https://arxiv.org/pdf/2511.20067")
    assert browser._last_navigation["reason"] == "navigation_error:" + phantom_browser.PlaywrightError.__name__


@pytest.mark.asyncio
async def test_a_download_that_never_comes_is_waited_for_no_longer_than_the_navigation(monkeypatch):
    import time

    monkeypatch.setattr(phantom_browser, "_NAVIGATION_MS", 300)
    browser = _browser(monkeypatch, _PageThatDownloads())
    began = time.monotonic()
    assert not await browser.browse("https://arxiv.org/pdf/2511.20067")
    assert time.monotonic() - began < 2.0
    assert browser._last_navigation["reason"].startswith("navigation_error:")


@pytest.mark.asyncio
async def test_a_download_that_lands_somewhere_refused_is_not_read(monkeypatch, tmp_path):
    asked = "https://arxiv.org/pdf/2511.20067"
    page = _PageThatDownloads(_Download("https://192.168.1.2/private.pdf", _pdf(tmp_path, "private")))
    browser = _browser(monkeypatch, page, allowed=lambda url: "192.168" not in url)
    assert not await browser.browse(asked)
    assert browser._last_navigation["reason"].startswith("redirect refused")
    assert browser._document is None


@pytest.mark.asyncio
async def test_a_document_is_not_read_once_the_page_has_gone_elsewhere(monkeypatch, tmp_path):
    url = "https://arxiv.org/pdf/2511.20067"
    page = _PageThatDownloads(_Download(url, _pdf(tmp_path, "Ternary weights at two bits")))
    browser = _browser(monkeypatch, page)
    assert await browser.browse(url)
    page.url = "https://example.org/after-a-click"
    assert browser._document_on_the_page() is None
