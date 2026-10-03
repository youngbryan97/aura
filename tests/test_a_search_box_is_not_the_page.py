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
