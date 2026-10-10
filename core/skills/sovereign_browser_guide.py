"""A guide to a thing a page shows, made before she touches it: what the page says of it, and what its program says.

Before a person plays a game on a page they glance at what the page says about
it ("Use the arrow keys to land the lander"), and someone who can read code
could glance at the program too. Both go into the guide to the place
(core/cognition/a_guide_to_a_place.py) with where they came from: the page's
own words, kept to those that say how the thing is worked (a site's menus,
sign-in links and comments say nothing of it), and the program the page runs,
where it can be had (core/skills/sovereign_browser_the_program.py). What she
looks up and what the screen shows her are taken in as they come.

Nothing here knows a game or a site.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any

__all__ = ["guide_to_the_page", "how_this_place_works"]

logger = logging.getLogger("Skills.SovereignBrowser.Guide")

#: The most of a page's own words read for how its thing is worked, in characters.
MOST_PAGE_TEXT = 20_000

_ITS_WORDS = f"(() => (document.body && document.body.innerText || '').slice(0, {MOST_PAGE_TEXT}))()"


async def guide_to_the_page(page: Any, thing: str) -> Any:
    """The guide to what ``page`` shows, from its words and its program; empty of either where it cannot be had."""
    from core.cognition.a_guide_to_a_place import PAGE, Guide
    from core.perception.reading_a_program import says_how_it_is_worked
    from core.skills.sovereign_browser_the_program import read_what_the_page_runs

    guide = Guide(place=thing)
    began = time.monotonic()
    try:
        text = await page.evaluate(_ITS_WORDS)
    except (RuntimeError, OSError, ValueError, TypeError, AttributeError):
        text = ""
    text = text if isinstance(text, str) else ""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", text) if 3 <= len(s.split()) <= 60]
    guide.take_in(PAGE, [s for s in sentences if says_how_it_is_worked(s)])
    read = await read_what_the_page_runs(page)
    guide.take_in_program(read)
    # And what its code says of how it is played, read by her model beside her work (core/cognition/reading_the_code.py).
    from core.cognition.reading_the_code import read_the_code_beside
    from core.rebuilding.her_model import ask_her_model
    from core.skills.screen_pursuit import _tell

    read_the_code_beside(guide, read, ask_her_model, tell=_tell)
    logger.info("a guide to %r in %.1fs: %s", thing, time.monotonic() - began, guide.for_thinking()[:600])
    return guide


#: The most of the guide put beside a page for her decision, in characters.
MOST_SAID = 700


def how_this_place_works(observation: Any, elements: list[Any]) -> str:
    """The guide to the site a page is on, taken up to date with what the page says and offers, as lines for her to
    reason with; "" where it holds nothing yet."""
    from urllib.parse import urlparse

    from core.cognition.a_guide_to_a_place import SCREEN, guide_for

    url = str(observation.get("url") or "")
    guide = guide_for(urlparse(url).netloc or url)
    text = str(observation.get("text") or "")[:4000]
    names = [str(e.get("name") or "") for e in elements if isinstance(e, dict) and e.get("name")]
    guide.take_in(SCREEN, [*re.split(r"(?<=[.!?])\s+|\n", text), *names[:80]])
    said = guide.for_thinking()
    return said[:MOST_SAID] if said.count("\n") >= 1 else ""
