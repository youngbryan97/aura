"""Leaving a page that leads nowhere, the way a person does: an address, or the words looked up.

LIVE 2026-10-05 a museum's game page answered her browser with a block page.
She said the way on was to find the game elsewhere, and every action she could
choose was a control on the page in front of her. A decision may now "go":
to an address it names, or to the web's answers to the words it gives.
"""
from __future__ import annotations

import re
import urllib.parse

__all__ = ["the_same_thing_elsewhere", "where_to_go"]


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
    return [w for w in re.findall(r"[a-z0-9]+", str(text or "").lower().replace("\u2019", "").replace("'", "")) if w not in _PLAIN and len(w) > 1]


def _alike(name: str, title: str) -> float:
    wanted = set(_words(name))
    return len(wanted & set(_words(title))) / max(1, len(wanted))


#: How much of a name another title must share to be the same thing.
SAME_THING = 0.6


async def the_same_thing_elsewhere(browser: object, name: str, *, runnable: bool = False) -> str:
    """Where the Internet Archive keeps the thing ``name`` names, or '' where it keeps nothing by that name.

    LIVE 2026-10-06 a game picked from a museum's list could be had neither
    from the museum, which refuses automated browsers, nor from the archive's
    copy of the museum's page, which it never made. The game itself was in the
    archive's collection all along, under nearly the same title. It is looked up
    by the whole name, then by its last part (a series name before a colon is
    often left off), through her own browser; the title that shares most of the
    name's words is taken, one the archive can run in its page first.
    """
    page = getattr(browser, "page", None)
    if page is None or not _words(name):
        return ""
    found: list[dict] = []
    for words in (_words(name), _words(str(name).split(":")[-1])):
        if not words:
            continue
        query = "title:(" + " AND ".join(words) + ")"
        address = "https://archive.org/advancedsearch.php?" + urllib.parse.urlencode(
            {"q": query, "fl[]": ["identifier", "title", "emulator"], "rows": 12, "output": "json"}, doseq=True)
        try:
            answer = await page.request.get(address, timeout=20000)
            found = (await answer.json()).get("response", {}).get("docs", []) if answer.ok else []
        except Exception:  # noqa: BLE001 - an archive that does not answer has nothing to say here
            found = []
        if found:
            break
    if runnable:
        found = [d for d in found if d.get("emulator")]  # only a copy the archive runs in its page
    best = max(found, key=lambda d: (_alike(name, str(d.get("title") or "")), bool(d.get("emulator"))), default=None)
    if best is None or _alike(name, str(best.get("title") or "")) < SAME_THING:
        return ""
    return f"https://archive.org/details/{best['identifier']}"
