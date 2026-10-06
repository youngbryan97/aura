"""Where a thing might be had: the web's answers, and the public catalogues of what is kept, by the kind of thing it is.

A person who cannot get a thing where they were sent looks for it: they search
the web, and, knowing the kind of thing it is, they ask the places that keep
such things. An old program or game is asked of a software library, a book of a
library's catalogue, a picture or a recording of a media archive. Those places
answer searches openly, and none needs a browser to get past anything.

Here both are asked and every answer is a candidate: a title and an address.
Nothing is taken on a title: the caller opens each and judges the page
(core/skills/sovereign_browser_going.py). Search engines that refuse an
automated browser, as most do, are passed over, never got past.

The catalogues are knowledge of where things are kept, by kind, not of any
task: a game is one kind among those a software library holds.
"""
from __future__ import annotations

import logging
import re
import urllib.parse
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("Skills.WhereThingsAreKept")

__all__ = ["CATALOGUES", "Candidate", "kind_of_thing", "where_it_might_be"]


@dataclass(frozen=True)
class Candidate:
    title: str
    url: str
    found_by: str
    runs_there: bool | None = None  # what the place says: it runs the thing in its page, it does not, or it says nothing


@dataclass(frozen=True)
class Catalogue:
    name: str
    keeps: frozenset[str]
    search: Callable[[Any, str], Awaitable[list[Candidate]]]


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", str(text or "").lower().replace("’", "").replace("'", "")) if len(w) > 1]


async def _json(page: Any, address: str) -> Any:
    try:
        answer = await page.request.get(address, timeout=20000)
        return await answer.json() if answer.ok else None
    except Exception as why:  # noqa: BLE001 - a catalogue that does not answer has nothing to say
        logger.debug("no answer from %s: %s", address.split("?")[0], why)
        return None


async def _software_library(page: Any, name: str) -> list[Candidate]:
    """The Internet Archive's catalogue, which keeps old software, games, films, recordings and books, many runnable in its pages."""
    out: list[Candidate] = []
    for words in (_words(name), _words(str(name).split(":")[-1])):
        if not words:
            continue
        query = "title:(" + " AND ".join(words) + ")"
        found = await _json(page, "https://archive.org/advancedsearch.php?" + urllib.parse.urlencode(
            {"q": query, "fl[]": ["identifier", "title", "emulator"], "rows": 12, "output": "json"}, doseq=True))
        docs = (found or {}).get("response", {}).get("docs", []) if isinstance(found, dict) else []
        out += [Candidate(str(d.get("title") or ""), f"https://archive.org/details/{d['identifier']}", "the Internet Archive", bool(d.get("emulator")))
                for d in docs if d.get("identifier")]
        if docs:
            break
    return out


async def _books(page: Any, name: str) -> list[Candidate]:
    """Open Library's catalogue of books."""
    found = await _json(page, "https://openlibrary.org/search.json?" + urllib.parse.urlencode({"title": name, "limit": 8}))
    docs = (found or {}).get("docs", []) if isinstance(found, dict) else []
    return [Candidate(str(d.get("title") or ""), f"https://openlibrary.org{d['key']}", "Open Library") for d in docs if d.get("key")]


async def _media(page: Any, name: str) -> list[Candidate]:
    """Wikimedia Commons, which keeps pictures, recordings and films free to use."""
    found = await _json(page, "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(
        {"action": "query", "list": "search", "srsearch": name, "srnamespace": 6, "srlimit": 8, "format": "json"}))
    hits = (found or {}).get("query", {}).get("search", []) if isinstance(found, dict) else []
    return [Candidate(str(h.get("title") or "").removeprefix("File:"), "https://commons.wikimedia.org/wiki/" + urllib.parse.quote(str(h.get("title") or "").replace(" ", "_")),
                      "Wikimedia Commons") for h in hits]


#: Places that answer searches for what they keep, and the kinds of thing each keeps.
CATALOGUES: tuple[Catalogue, ...] = (
    Catalogue("the Internet Archive", frozenset("game games program software application app film movie video recording song album book magazine "
                                                "website cartoon show episode".split()), _software_library),
    Catalogue("Open Library", frozenset("book books novel".split()), _books),
    Catalogue("Wikimedia Commons", frozenset("picture photo image photograph painting sound recording video".split()), _media),
)

#: Search engines a person types words into, read the way any list on a page is read.
_ENGINES = ("https://www.bing.com/search?q=", "https://html.duckduckgo.com/html/?q=")


def kind_of_thing(task: str, fallback: str = "") -> str:
    """The kind of thing a task is about, by its own noun: "Play this game" is about a game."""
    said = re.search(r"\b(?:this|that|the|each|every|a|an)\s+([a-z]+)", str(task or ""), re.I)
    return said.group(1).lower() if said else fallback


async def _searched(skill: Any, browser: Any, name: str, kind: str) -> list[Candidate]:
    from core.skills.sovereign_browser_going import refused
    from core.skills.sovereign_browser_picking import the_list_on_the_page

    page = getattr(browser, "page", None)
    query = f"{name} {kind}".strip()
    for engine in _ENGINES:
        if not await skill._safe_browse(browser, engine + urllib.parse.quote_plus(query)):
            continue
        title = await page.title()
        text = await page.evaluate("document.body ? document.body.innerText.slice(0, 800) : ''")
        if refused(title, text):
            continue  # an engine that will not answer an automated browser is not argued with
        found = await the_list_on_the_page(browser)
        if found:
            return [Candidate(i["text"], i["href"], engine.split("/")[2]) for i in found]
    return []


async def where_it_might_be(skill: Any, browser: Any, name: str, *, task: str = "") -> list[Candidate]:
    """Every place the web and the catalogues for its kind say the thing ``name`` might be, as titles and addresses."""
    kind = kind_of_thing(task)
    out = await _searched(skill, browser, name, kind)
    page = getattr(browser, "page", None)
    if page is not None:
        for catalogue in CATALOGUES:
            if kind and kind not in catalogue.keeps:
                continue
            out += await catalogue.search(page, name)
    return out
