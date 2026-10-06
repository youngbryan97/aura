"""A link that opens a new tab is followed in hers: what it opened is what the click was for.

LIVE 2026-10-06 a museum's game page linked the game itself on another site, in
a new tab. The click left her own page as it was, the link was retired as one
that went nowhere, and the game was never reached.

Served at https addresses through route interception, as in
test_a_search_box_is_not_the_page.py.
"""
from __future__ import annotations

import pytest

PAGES = {
    "museum": """<html><body><h1>A game, in 2003</h1>
        <p><a id="elsewhere" href="https://example.org/aura-the-game" target="_blank" rel="noopener">Play it at the archive</a></p></body></html>""",
    "the-game": "<html><body><h1>The game itself</h1><canvas width=200 height=100></canvas></body></html>",
}


def _serves(body: str):
    async def _fulfil(route) -> None:
        await route.fulfill(status=200, content_type="text/html", body=body)

    return _fulfil


@pytest.fixture
async def browser():
    from core.capabilities.phantom_browser import PhantomBrowser

    instance = PhantomBrowser(visible=False, browser_type="chromium", principal="owner")
    if not await instance.ensure_ready():
        await instance.close()
        pytest.skip("no browser engine available in this environment")
    try:
        await instance.context.route("**/aura-museum", _serves(PAGES["museum"]))
        await instance.context.route("**/aura-the-game", _serves(PAGES["the-game"]))
        yield instance
    finally:
        await instance.close()


@pytest.fixture
def click_lease():
    from core.capabilities.browser_authority import (
        BrowserAction,
        issue_browser_lease,
        revoke_browser_lease,
    )

    lease = issue_browser_lease(principal="owner", origin="https://example.com", actions={BrowserAction.CLICK})
    yield lease.lease_id
    revoke_browser_lease(lease.lease_id)


@pytest.mark.asyncio
async def test_a_link_that_opens_a_new_tab_is_followed_in_hers(browser, click_lease):
    await browser.page.goto("https://example.com/aura-museum", wait_until="load")
    assert await browser.click("#elsewhere", principal="owner", lease_id=click_lease)
    assert browser.page.url == "https://example.org/aura-the-game"
    assert "The game itself" in await browser.page.inner_text("body")
    assert len(browser.context.pages) == 1  # the tab it opened was closed: she goes on in one
