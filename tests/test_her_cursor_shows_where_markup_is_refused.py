"""Her neuron mark is drawn on a page that refuses markup.

LIVE 7 and 8 November 2026: on YouTube, which requires Trusted Types, every
mount of the mark failed with "This document requires 'TrustedHTML'
assignment", and nobody watching her browser saw where she clicked.
"""
from __future__ import annotations

import pytest

from core.perception.browser_presence import BOOTSTRAP, as_nodes

STRICT = "require-trusted-types-for 'script'; trusted-types 'none'; style-src 'none'"
PAGE = "<!doctype html><html><body style='margin:0;background:#fff'><p>A page</p></body></html>"
CLICK = {"x": 200, "y": 150, "width": 120, "height": 40, "clickX": 260, "clickY": 170}


def test_the_glyph_is_handed_over_as_nodes():
    tree = as_nodes('<svg xmlns="http://www.w3.org/2000/svg"><!-- note --><defs><radialGradient id="soma"/>'
                    '</defs><path fill="url(#soma)"/></svg>')
    assert tree == ["svg", {}, [["defs", {}, [["radialGradient", {"id": "soma"}, []]]],
                                ["path", {"fill": "url(#soma)"}, []]]]


async def _strict_page(playwright):
    browser = await playwright.chromium.launch(headless=True)
    context = await browser.new_context(viewport={"width": 640, "height": 400})
    await context.add_init_script(BOOTSTRAP)
    page = await context.new_page()

    async def serve(route):
        await route.fulfill(status=200, body=PAGE, content_type="text/html",
                            headers={"Content-Security-Policy": STRICT})

    await page.route("https://strict.test/", serve)
    await page.goto("https://strict.test/")
    return browser, page


def _marked(picture: bytes) -> int:
    from io import BytesIO

    from PIL import Image

    pixels = Image.open(BytesIO(picture)).convert("RGB").getdata()
    return sum(1 for r, g, b in pixels if b > 120 and b - g > 50 and r - g > 20)


@pytest.mark.asyncio
async def test_the_mark_is_drawn_where_the_page_refuses_markup():
    playwright_api = pytest.importorskip("playwright.async_api")
    async with playwright_api.async_playwright() as playwright:
        browser, page = await _strict_page(playwright)
        try:
            refused = await page.evaluate(
                "() => { try { document.body.innerHTML = '<b>x</b>'; return ''; } catch (e) { return e.name; } }")
            assert refused == "TypeError", "the fixture page does not enforce Trusted Types"
            # Her pointer is centred where she clicks (8 November 2026), so the picture is taken around that point.
            where = {"x": CLICK["clickX"] - 22, "y": CLICK["clickY"] - 22, "width": 44, "height": 44}
            before = _marked(await page.screenshot(clip=where))
            await page.evaluate("data => window.__auraBrowserPresence.show(data)",
                                {"action": "clicking", "stage": "started", "box": CLICK})
            shown = await page.evaluate(
                "() => getComputedStyle(document.querySelector('[data-aura-presence]')).display")
            assert shown == "block"
            assert _marked(await page.screenshot(clip=where)) - before > 40
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_her_eyes_still_see_the_page_without_the_mark():
    playwright_api = pytest.importorskip("playwright.async_api")
    async with playwright_api.async_playwright() as playwright:
        browser, page = await _strict_page(playwright)
        try:
            await page.evaluate("data => window.__auraBrowserPresence.show(data)",
                                {"action": "clicking", "stage": "started", "box": CLICK})
            await page.evaluate("() => window.__auraBrowserPresence.suspend()")
            where = {"x": CLICK["clickX"] - 22, "y": CLICK["clickY"] - 22, "width": 44, "height": 44}
            assert _marked(await page.screenshot(clip=where)) == 0
        finally:
            await browser.close()
