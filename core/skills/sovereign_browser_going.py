"""Leaving a page that leads nowhere, the way a person does: an address, or the words looked up.

LIVE 2026-10-05 a museum's game page answered her browser with a block page.
She said the way on was to find the game elsewhere, and every action she could
choose was a control on the page in front of her. A decision may now "go":
to an address it names, or to the web's answers to the words it gives.
"""
from __future__ import annotations

import logging
import re
import urllib.parse
from typing import Any

__all__ = ["Where", "the_same_thing_elsewhere", "where_the_page_points", "where_to_go"]


def where_to_go(value: str) -> str:
    """An address as given, or else the web's answers to the words, as a person types either into the address bar."""
    said = " ".join(str(value or "").split())
    if re.match(r"^(https?://|www\.)\S+$", said, re.IGNORECASE):
        return said if said.lower().startswith("http") else f"https://{said}"
    if re.match(r"^[a-z0-9-]+(\.[a-z0-9-]+)+(/\S*)?$", said, re.IGNORECASE):
        return f"https://{said}"
    return f"https://html.duckduckgo.com/html/?q={urllib.parse.quote_plus(said)}"


#: What a page says when it refuses a browser rather than serving it.
_REFUSALS = re.compile(
    r"you have been blocked|attention required|access (?:to this page has been )?denied|403 forbidden|"
    r"verify you are (?:a )?human|not a robot|unusual traffic|captcha|too many requests",
    re.IGNORECASE,
)


def refused(title: str, text: str) -> bool:
    """Whether a page is a refusal of her browser rather than the page asked for."""
    return bool(_REFUSALS.search(f"{title} {str(text)[:600]}"))


def its_archived_copy(url: str) -> str:
    """The same address as the Internet Archive keeps it: the copy a person turns to when a site will not serve them."""
    import time

    return f"https://web.archive.org/web/{time.strftime('%Y')}/{url}"


async def the_archived_copy(skill: object, browser: object, url: str) -> str:
    """The archived copy of ``url``, opened in ``browser`` and said out loud, or '' when there is none that loads.

    A site that refuses automated browsers has said so, and nothing here tries
    to get past it: the archive is a different place, which serves copies of
    public pages to anyone who asks.
    """
    copy = its_archived_copy(url)
    if url.startswith("https://web.archive.org/") or not await skill._safe_browse(browser, copy):  # type: ignore[attr-defined]
        return ""
    page = getattr(browser, "page", None)
    if page is None or refused(await page.title(), await page.evaluate("document.body ? document.body.innerText : ''")):
        return ""
    skill._say_out_loud(  # type: ignore[attr-defined]
        f"{url} refused my browser, so I am using the copy of it the Internet Archive keeps.",
        {"label": "Going to", "said": copy},
    )
    return str(page.url)


_PLAIN = frozenset("the a an of and in to for on at by with".split())


def _words(text: str) -> list[str]:
    said = str(text or "").lower().replace("\u2019", "").replace("'", "")
    # A word and its plural, or its possessive, are one word: "Network's" is "Network".
    return [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
            for w in re.findall(r"[a-z0-9]+", said) if w not in _PLAIN and len(w) > 1]


def _alike(name: str, title: str) -> float:
    wanted = set(_words(name))
    return len(wanted & set(_words(title))) / max(1, len(wanted))


def _names_it(name: str, title: str) -> bool:
    """Whether ``title`` names the thing: every word of its own part (after a series name and a colon), and most of the rest.

    The own part whole, with a word of the series beside it, names it: LIVE
    2026-10-07 "Scooby-Doo: Scooby Snapshot" is kept as "Scooby Snapshot", and
    two of three words shared was taken for another thing. Without any word of
    the series, the same own name may be another thing ("Tunnel Rush" is not
    "Toonami: Tunnel Rush").
    """
    own, there = set(_words(str(name).split(":")[-1])), set(_words(title))
    if not own <= there:
        return False
    series = set(_words(":".join(str(name).split(":")[:-1])))
    return _alike(name, title) >= SAME_THING or bool(series & there)


def _own_name_in(name: str, title: str) -> bool:
    """Whether ``title`` holds every word of the thing's own name (after a series name and a colon)."""
    own = set(_words(str(name).split(":")[-1]))
    return bool(own) and own <= set(_words(title))


def _named_on_its_page(name: str, heading: str, text: str) -> bool:
    """Whether a page is of the thing: its title names it, or its title holds the own name and its own words (topics, a
    description, who made it) name the series. LIVE 2026-10-09 "Scooby-Doo: Ask Swami Shaggy" is kept under the title
    "Ask Swami Shaggy", topic Scooby-Doo, and was passed over; "Tunnel Rush", with no word of Toonami on its page,
    is still another thing."""
    if _names_it(name, heading):
        return True
    series = set(_words(":".join(str(name).split(":")[:-1])))
    return bool(series) and _own_name_in(name, heading) and bool(series & set(_words(text)))


logger = logging.getLogger("Skills.SovereignBrowser.Going")

#: How much of a name another title must share to be the same thing.
SAME_THING = 0.75

#: Tasks done with the thing itself, running: for these only a page where it runs will do.
RUNS = re.compile(r"\b(play|run|launch|watch|listen|hear)\b", re.I)

#: Whether a page has something in it that runs: an embedded program, an emulator, a game's canvas, a player.
_SOMETHING_RUNS = r"""
() => {
  const big = (el) => { const r = el.getBoundingClientRect(); return r.width >= 240 && r.height >= 160; };
  const found = [];
  const look = (root) => {
    for (const el of root.querySelectorAll("canvas, embed, object, iframe, video, audio, ruffle-player, ruffle-object, ruffle-embed")) {
      const tag = el.tagName.toLowerCase();
      if (tag === "audio" || big(el)) found.push(tag);
    }
    for (const el of root.querySelectorAll("*")) if (el.shadowRoot) look(el.shadowRoot);
  };
  look(document);
  return found;
}
"""


class Where(str):
    """An address found for a thing, and what was seen of it there (core/skills/whether_a_page_serves.py)."""

    serves: Any = None


async def the_same_thing_elsewhere(skill: object, browser: object, name: str, *, task: str = "", not_at: str = "") -> str:
    """Where else ``name`` is to be had, found as a person finds it: looked for, and each likely page opened and judged; or ''.

    LIVE 2026-10-06 games picked from a museum's list could be had neither
    from the museum, which refuses automated browsers, nor from an archived
    copy of its pages, which was never made. They were elsewhere all along.
    The web is searched and the catalogues that keep its kind of thing are
    asked (core/skills/where_things_are_kept.py). The candidates that name it
    are opened in turn, and the first is kept that is not a refusal, that names
    it in its title or heading, and, where the task is to run it, has something
    in it that runs. No site is known here: a page is kept for what it is.
    """
    from core.skills.where_things_are_kept import where_it_might_be

    page = getattr(browser, "page", None)
    if page is None or not _words(name):
        return ""
    candidates = await where_it_might_be(skill, browser, name, task=task)
    return await _the_first_that_serves(skill, browser, name, task, candidates, not_at)


#: The links of a page, as their words and where they go.
_LINKS = "() => Array.from(document.querySelectorAll('a[href]')).map((a) => [a.innerText || a.title || '', a.href]).slice(0, 400)"


_ARCHIVED = re.compile(r"^https?://web\.archive\.org/web/(\d+)[a-z_]*/(https?://.+)$")


def _unwrapped(href: str, on: str = "") -> str:
    """Where a link goes, out of the archive's copy ``on`` of the page it is on: the archive rewrites each link to its own
    copy of the same moment. A link to another moment's copy is the page's author pointing there, and is kept: LIVE
    2026-10-08 a museum's "in Internet Archive" link was to the 2020 copy of the game's own page."""
    inner, page = _ARCHIVED.match(href), _ARCHIVED.match(on)
    return inner.group(2) if inner and page and inner.group(1)[:14] == page.group(1)[:14] else href


async def where_the_page_points(skill: object, browser: object, name: str, *, task: str = "", not_at: str = "") -> str:
    """Where the page in front says ``name`` is kept, opened and judged; or ''.

    A page about a thing that does not run it often says where it is: LIVE
    2026-10-08 a museum page whose player could not load its own files had a
    link under "Adventure Time: Jumping Finn in Internet Archive". The links of
    the page whose words name the thing are its authors' answer, and are taken
    as candidates like any other.
    """
    from core.skills.where_things_are_kept import Candidate

    page = getattr(browser, "page", None)
    if page is None:
        return ""
    try:
        links = await page.evaluate(_LINKS)
    except Exception:  # noqa: BLE001 - a page that cannot list its links points nowhere
        return ""
    here = str(page.url).rstrip("/")
    candidates = [Candidate(" ".join(str(words).split()), _unwrapped(str(href), here), "the page itself")
                  for words, href in links if _names_it(name, str(words)) and _unwrapped(str(href), here).rstrip("/") != here]
    return await _the_first_that_serves(skill, browser, name, task, candidates, not_at or here) if candidates else ""


async def _the_first_that_serves(skill: object, browser: object, name: str, task: str, candidates: list[Any], not_at: str) -> str:
    """Of ``candidates`` that name the thing, the first opened and seen to serve the task; else the first that cannot yet be
    told; else ''."""
    page = getattr(browser, "page", None)
    runs = bool(RUNS.search(task or ""))
    seen: set[str] = set()
    likely = []
    for found in sorted(candidates, key=lambda c: (-_alike(name, c.title), c.runs_there is not True)):
        # A place that says it does not run the thing is not where it is played; one that says nothing is opened and looked at.
        if found.url not in seen and _own_name_in(name, found.title) and not (runs and found.runs_there is False):
            seen.add(found.url)
            likely.append(found)
    from core.skills.whether_a_page_serves import watching, what_the_task_needs, whether_it_serves

    # Each candidate is judged as the page the person was sent to would be: by its developer tools, its code and
    # its words, and by seeing the thing do what the task needs. One seen working is taken; else the first that
    # cannot yet be told; one that is seen not to work never is.
    unsure: Where | None = None
    watching(page)
    for found in [f for f in likely if f.url.rstrip("/") != not_at.rstrip("/")][:6]:
        if not await skill._safe_browse(browser, found.url):  # type: ignore[attr-defined]
            continue
        title = await page.title()
        text = await page.evaluate("document.body ? document.body.innerText.slice(0, 4000) : ''")
        heading = await page.evaluate("(() => { const h = document.querySelector('h1'); return h ? h.innerText : ''; })()")
        if refused(title, text) or not _named_on_its_page(name, f"{title} {heading}", text):
            continue
        if runs and not await page.evaluate(_SOMETHING_RUNS):
            continue
        verdict = await whether_it_serves(page, task) if what_the_task_needs(task) else None
        where = Where(str(page.url))
        where.serves = verdict
        if verdict is not None and verdict.ok is False:
            logger.info("the same thing at %s does not work: %s", page.url, verdict.says())
            continue
        if verdict is None or verdict.ok:
            return where
        unsure = unsure or where
    if unsure is not None and await skill._safe_browse(browser, str(unsure)):  # type: ignore[attr-defined]
        return unsure
    return ""
