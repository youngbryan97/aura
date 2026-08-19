"""The browser could click a selector but could not say what was clickable.

`PhantomBrowser` had `click(selector)`, `type()`, `read_content()` and
`get_links()`, and nothing that enumerated the page's INTERACTIVE elements. So
every interaction had to be scripted from selectors known in advance — an open
loop, fine for a known page and useless for a flow whose next screen depends on
the answer given to the last one.

`observe()` is the missing perception primitive, and it is the one the 2026
web-agent literature converges on: a pruned, indexed list of interactive
elements as structured text rather than pixels.

The visibility rule here is the part that was MEASURED rather than assumed. On a
live questionnaire every answer control reported `opacity: 0` with
`visibility: visible`, `display: block`, a real 36-56px box, and its meaning in
`aria-label`. Sites hide the native control and paint a custom graphic over it
constantly. An opacity filter therefore drops exactly the elements a
form-filling agent needs and nothing else — Playwright clicks them happily; only
the observer could not see them.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio

FIXTURE = """
<html><body>
  <main>
    <h1>Question 1 of 3: You regularly make new friends.</h1>
    <form>
      <!-- The live shape: a real control, styled invisible, meaning in aria-label. -->
      <input type="radio" name="q1" aria-label="I strongly agree" value="3"
             style="opacity:0; width:40px; height:40px; display:block">
      <input type="radio" name="q1" aria-label="I disagree" value="-1"
             style="opacity:0; width:36px; height:36px; display:block">
      <button id="next">Go to the next set of questions</button>
      <!-- Scenery styled invisible is still scenery. -->
      <div role="button" aria-label="decorative ghost" style="opacity:0; width:50px; height:50px"></div>
      <!-- Genuinely hidden controls stay hidden. -->
      <input type="radio" name="q1" aria-label="never rendered" style="display:none">
    </form>
  </main>
</body></html>
"""


#: The fixture is served at an https URL through route interception rather than
#: `set_content`. BrowserAuthority refuses any scheme but https — "url refused:
#: scheme 'about' not allowed" — which is the governance working, so the test
#: satisfies it instead of bypassing it, and still touches no network.
FIXTURE_URL = "https://example.com/aura-observation-fixture"


async def _observed(browser):
    return await browser.observe(principal="owner")


@pytest.fixture
async def browser():
    from core.capabilities.phantom_browser import PhantomBrowser

    instance = PhantomBrowser(visible=False, browser_type="chromium", principal="owner")
    if not await instance.ensure_ready():
        await instance.close()
        pytest.skip("no browser engine available in this environment")
    try:
        await instance.page.route(
            "**/aura-observation-fixture",
            lambda route: route.fulfill(status=200, content_type="text/html", body=FIXTURE),
        )
        await instance.page.goto(FIXTURE_URL)
        yield instance
    finally:
        await instance.close()


async def test_a_transparent_form_control_is_still_offered(browser):
    observation = await _observed(browser)
    names = {element["name"] for element in observation["elements"]}
    assert "I strongly agree" in names
    assert "I disagree" in names


async def test_transparent_scenery_is_not_offered(browser):
    """The relaxation is for controls, not for everything invisible."""
    observation = await _observed(browser)
    names = {element["name"] for element in observation["elements"]}
    assert "decorative ghost" not in names


async def test_display_none_stays_hidden(browser):
    observation = await _observed(browser)
    names = {element["name"] for element in observation["elements"]}
    assert "never rendered" not in names


async def test_every_element_carries_a_resolvable_selector(browser):
    """Indices are positions a re-render reorders; a path still resolves."""
    observation = await _observed(browser)
    assert observation["elements"]
    for element in observation["elements"]:
        selector = element.get("selector")
        assert selector, element
        found = await browser.page.query_selector(selector)
        assert found is not None, f"selector did not resolve: {selector}"


async def test_the_question_travels_with_the_controls(browser):
    """A questionnaire is unanswerable from radio labels alone."""
    observation = await _observed(browser)
    assert "You regularly make new friends" in observation["text"]


async def test_control_state_is_reported(browser):
    observation = await _observed(browser)
    radios = [e for e in observation["elements"] if e.get("role") == "radio"]
    assert radios
    assert all("checked" in radio for radio in radios)
    assert {radio.get("value") for radio in radios} >= {"3", "-1"}


async def test_a_control_under_its_own_decoration_needs_the_forced_path(browser):
    """Five rounds of acting on a page that never changed.

    Playwright's actionability check includes hit-target: the point clicked
    must actually receive the event. Sites hide the native input (opacity 0)
    and paint a custom graphic over it, so the overlay receives the click and
    the ordinary path times out — the element is visible, in view, and
    unclickable.

    MEASURED live 2026-08-18: a questionnaire ran five observe-decide-act
    rounds and the page never moved, because every answer click was landing on
    nothing. The loop's stall detector stopped it and reported `no_progress`,
    which named the symptom rather than this.

    Asserted at the Playwright level: the governed `click()` needs a real
    lease, and a test must not forge one.
    """
    await browser.page.set_content(
        """
        <html><body><main>
          <div style="position:relative; width:60px; height:60px">
            <input id="real" type="radio" aria-label="I agree"
                   style="opacity:0; width:60px; height:60px; position:absolute; inset:0">
            <div style="position:absolute; inset:0; background:#333"></div>
          </div>
        </main></body></html>
        """
    )
    element = browser.page.locator("#real")

    with pytest.raises(Exception):
        await element.click(timeout=1500)
    assert await browser.page.eval_on_selector("#real", "el => el.checked") is False

    # `force=True` is not the answer: it skips the actionability checks and
    # still clicks at a POINT, so the decoration receives it just the same.
    await element.click(force=True, timeout=4000)
    assert await browser.page.eval_on_selector("#real", "el => el.checked") is False

    # Dispatching to the element itself bypasses hit-testing rather than
    # ignoring it, and is what actually reaches the control.
    await element.dispatch_event("click")
    assert await browser.page.eval_on_selector("#real", "el => el.checked") is True


def test_the_click_path_falls_back_rather_than_giving_up():
    """The fallback is ordered: ordinary first, forced only on timeout.

    An unconditional forced click would punch through consent banners and
    anything genuinely in the way, which is the opposite of what is wanted.
    """
    import inspect

    from core.capabilities.phantom_browser import PhantomBrowser

    source = inspect.getsource(PhantomBrowser.click)
    ordinary = source.index("await element.click(timeout=")
    forced = source.index('dispatch_event("click")')
    assert ordinary < forced
