"""The page's words are the page's, not the first form's.

LIVE 2026-10-03: the observer took its text from the first of `main`,
`[role="main"]` or `form` in the document. The Cartoon Network games list has a
search box in its header, so the list of fifty-six games she was asked to count
through read as no text at all.

Served at https addresses through route interception, as in
test_browser_observation_sees_real_controls.py.
"""

from __future__ import annotations

import pytest

LIST = "".join(f"<li><a href='/g{n}'>Game number {n}</a></li>" for n in range(1, 13))
PAGES = {
    "search-header": f"""<html><body>
        <header><form><input type="search" aria-label="Search"></form></header>
        <h1>Flash games</h1><ul>{LIST}</ul></body></html>""",
    "questionnaire": """<html><body><p>Site header</p>
        <form><p>makes lists</p><p>relies on memory</p>
        <p>sceptical</p><p>wants to believe</p>
        <input type="radio" name="q1" aria-label="1"><button>Submit</button></form></body></html>""",
    "landmark": """<html><body><nav>Home About</nav>
        <main><h1>The article</h1><p>What the page is for.</p></main></body></html>""",
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
        for name, body in PAGES.items():
            await instance.page.route(f"**/aura-{name}", _serves(body))
        yield instance
    finally:
        await instance.close()


async def _text_of(browser, name: str) -> str:
    await browser.page.goto(f"https://example.com/aura-{name}", wait_until="load")
    return (await browser.observe(principal="owner"))["text"]


@pytest.mark.asyncio
async def test_a_search_box_in_the_header_is_not_the_page(browser):
    text = await _text_of(browser, "search-header")
    assert "Game number 1" in text and "Game number 12" in text
    assert text.index("Game number 2") < text.index("Game number 11"), "in the page's order"


@pytest.mark.asyncio
async def test_a_page_that_is_its_form_is_read_as_its_form(browser):
    text = await _text_of(browser, "questionnaire")
    assert "relies on memory" in text and "Site header" not in text


@pytest.mark.asyncio
async def test_a_main_landmark_is_the_page(browser):
    text = await _text_of(browser, "landmark")
    assert "What the page is for." in text and "Home About" not in text


PAGES["scrolled-past"] = """<html><body>
    <a href="/more-1" id="m1">more</a><a href="/more-2" id="m2">more</a>
    <div style="height: 2600px"></div><p>the end of the page</p></body></html>"""


@pytest.mark.asyncio
async def test_a_control_she_scrolled_past_is_still_offered(browser):
    """LIVE 2026-10-03 04:31: "the 'more' links aren't exposed as clickable controls"."""
    await browser.page.route("**/aura-scrolled-past", _serves(PAGES["scrolled-past"]))
    await browser.page.goto("https://example.com/aura-scrolled-past", wait_until="load")
    await browser.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    names = [e.get("selector") for e in (await browser.observe(principal="owner"))["elements"]]
    assert "#m1" in names and "#m2" in names


PAGES["held-link"] = """<html><body>
    <a id="go" href="https://example.com/aura-landing">the game</a>
    <script>document.getElementById('go').addEventListener('click', (e) => {
        e.preventDefault(); location.hash = 'held_by_an_advert'; });</script></body></html>"""
PAGES["landing"] = "<html><body><p>You arrived.</p></body></html>"


@pytest.fixture
def click_lease():
    from core.capabilities.browser_authority import (
        BrowserAction,
        issue_browser_lease,
        revoke_browser_lease,
    )

    lease = issue_browser_lease(
        principal="owner", origin="https://example.com", actions={BrowserAction.CLICK}
    )
    yield lease.lease_id
    revoke_browser_lease(lease.lease_id)


@pytest.mark.asyncio
async def test_a_link_whose_click_is_held_is_followed_to_its_address(browser, click_lease):
    """LIVE 2026-10-03 06:14: an advert held a game link's click (#google_vignette)."""
    for name in ("held-link", "landing"):
        await browser.page.route(f"**/aura-{name}", _serves(PAGES[name]))
    await browser.page.goto("https://example.com/aura-held-link", wait_until="load")
    assert await browser.click("#go", principal="owner", lease_id=click_lease)
    assert browser.page.url.startswith("https://example.com/aura-landing")


@pytest.mark.asyncio
async def test_a_link_within_the_page_is_not_followed_anywhere(browser, click_lease):
    page = """<html><body><a id="jump" href="#below">down</a><div style="height:2000px"></div>
        <p id="below">below</p></body></html>"""
    await browser.page.route("**/aura-jump", _serves(page))
    await browser.page.goto("https://example.com/aura-jump", wait_until="load")
    assert await browser.click("#jump", principal="owner", lease_id=click_lease)
    assert browser.page.url.split("#")[0] == "https://example.com/aura-jump"


PAGES["slow-link"] = """<html><body>
    <a id="go" href="https://example.com/aura-landing">the game</a>
    <script>document.getElementById('go').addEventListener('click', (e) => {
        e.preventDefault(); setTimeout(() => { location.href = e.target.href; }, 2500); });</script>
    </body></html>"""


@pytest.mark.asyncio
async def test_a_navigation_still_on_its_way_is_not_started_again(browser, click_lease, monkeypatch):
    """LIVE 2026-10-03 08:07: a second navigation on top of one in flight closed the browser."""
    started: list[str] = []

    async def _browse(url: str, *, principal: str = "") -> bool:
        started.append(url)
        return True

    monkeypatch.setattr(browser, "browse", _browse)
    for name in ("slow-link", "landing"):
        await browser.page.route(f"**/aura-{name}", _serves(PAGES[name]))
    await browser.page.goto("https://example.com/aura-slow-link", wait_until="load")
    assert await browser.click("#go", principal="owner", lease_id=click_lease)
    assert started == []


PAGES["plain-link"] = """<html><body><a id="go" href="https://example.com/aura-landing">the game</a></body></html>"""


@pytest.mark.asyncio
async def test_an_ordinary_link_click_arrives_promptly(browser, click_lease):
    """LIVE 2026-10-03 08:32: an ordinary link click ran past its ten seconds and closed the browser."""
    import asyncio
    import time

    for name in ("plain-link", "landing"):
        await browser.page.route(f"**/aura-{name}", _serves(PAGES[name]))
    await browser.page.goto("https://example.com/aura-plain-link", wait_until="load")
    began = time.monotonic()
    assert await asyncio.wait_for(
        browser.click("#go", principal="owner", lease_id=click_lease), timeout=10.0
    )
    assert time.monotonic() - began < 10.0
    assert browser.page.url.startswith("https://example.com/aura-landing")
