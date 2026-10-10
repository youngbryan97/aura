"""Looking something up on the web while she is in the middle of something else, and reading what it finds.

A person in the middle of something who needs to know how it is done searches,
reads a page or two, and goes back. She does it without leaving the page she is
working in: the search and the pages are fetched and read as text, beside it.

Search engines are asked in turn. One that refuses or answers with a challenge
("are you a robot?") is not argued with; the next is asked. LIVE 2026-10-09 one
engine answered every question by its first word alone ("how to win ..."
returned a page about archive software), and another asked to be shown a human;
a third answered as it answers anyone. LIVE 2026-10-10 the one that had answered
answered "Tom's Trap-O-Matic" with a talking-cat app: the page a search engine
makes for browsers without scripts is asked first, and found the game.
"""
from __future__ import annotations

import asyncio
import html
import logging
import re
import time
import urllib.parse
from collections.abc import Callable

__all__ = ["ENGINES", "search_and_read"]

logger = logging.getLogger("Skills.LookingItUp")

#: How a person's browser introduces itself, so pages answer as they answer people.
_AS_A_BROWSER = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
                               "Chrome/126.0 Safari/537.36", "Accept-Language": "en-US,en;q=0.9"}

#: The most pages read for one search, and the most words kept of each.
PAGES = 3
WORDS_KEPT = 2500

#: What a page says when it will not answer a script.
_CHALLENGED = re.compile(r"not a robot|are you a human|verify you are|unusual traffic|captcha|bots use|access denied|"
                         r"please enable (?:javascript|cookies)", re.I)


def _yahoo(text: str) -> list[tuple[str, str]]:
    """Each result as (its heading, where it goes): the heading is the anchor's own <h3>, not the address shown above it."""
    found: list[tuple[str, str]] = []
    for match in re.finditer(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', text, re.S):
        heading = re.search(r"<h3[^>]*>(.*?)</h3>", match.group(2), re.S)
        if heading is None:
            continue
        href = html.unescape(match.group(1))
        wrapped = re.search(r"/RU=([^/]+)/", href)
        url = urllib.parse.unquote(wrapped.group(1)) if wrapped else href
        title = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", heading.group(1))).split())
        if url.startswith("http") and "yahoo." not in url and title:
            found.append((title, url))
    return found


def _duckduckgo(text: str) -> list[tuple[str, str]]:
    """Each result of the page made for browsers without scripts: its heading, and where its link goes."""
    found: list[tuple[str, str]] = []
    for href, heading in re.findall(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', text, re.S):
        href = html.unescape(href)
        url = urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("uddg", [href])[0]
        title = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", heading)).split())
        if url.startswith("http") and "duckduckgo.com" not in url and title:
            found.append((title, url))
    return found


def _marginalia(text: str) -> list[tuple[str, str]]:
    """Each result of an open search interface made for programs: its title, and where it is."""
    import json

    try:
        results = json.loads(text).get("results") or []
    except (ValueError, AttributeError):
        return []                                    # not an answer: a limit reached, said as a page
    return [(str(r.get("title") or ""), str(r.get("url") or "")) for r in results
            if str(r.get("url") or "").startswith("http") and r.get("title")]


def _bing(text: str) -> list[tuple[str, str]]:
    from core.capabilities.browser_controller import _bing_results_in

    return [(hit["title"], hit["url"]) for hit in _bing_results_in(text, 10)]


#: Where to search, in turn: the address with the query in it, and how to read the results.
ENGINES: tuple[tuple[str, str, Callable[[str], list[tuple[str, str]]]], ...] = (
    # An index of the independent web with an interface open to programs; LIVE 2026-10-10 it found a game that two
    # engines would not search for from here and a third answered with a talking-cat app.
    ("marginalia", "https://api.marginalia.nu/public/search/{q}?count=10", _marginalia),
    ("duckduckgo", "https://html.duckduckgo.com/html/?q={q}", _duckduckgo),
    ("yahoo", "https://search.yahoo.com/search?p={q}", _yahoo),
    ("bing", "https://www.bing.com/search?q={q}&setlang=en-US", _bing),
)


def _text_of(page: str) -> str:
    from core.search.research_pipeline import _html_to_text

    return _html_to_text(page)


async def search_and_read(query: str, seconds: float, *, pages: int = PAGES) -> list[tuple[str, str, str]]:
    """``query`` searched, and the first pages found read: (title, address, words), all within ``seconds``."""
    import httpx

    began = time.monotonic()

    def left() -> float:
        return max(0.5, seconds - (time.monotonic() - began))

    async with httpx.AsyncClient(headers=_AS_A_BROWSER, follow_redirects=True) as client:
        hits: list[tuple[str, str]] = []
        for name, address, read in ENGINES:
            try:
                answer = await client.get(address.format(q=urllib.parse.quote_plus(query)), timeout=min(left(), 8.0))
                if answer.status_code >= 500:  # a busy engine is asked once more, as a person reloads
                    await asyncio.sleep(1.0)
                    answer = await client.get(address.format(q=urllib.parse.quote_plus(query)), timeout=min(left(), 8.0))
            except Exception as why:  # noqa: BLE001 - an engine that does not answer is asked no more
                logger.info("%s did not answer: %s", name, str(why)[:120])
                continue
            # Anything but a plain answer (a 202 asks to be asked more slowly) is taken as no, and the next is asked.
            if answer.status_code != 200 or _CHALLENGED.search(answer.text[:4000]):
                logger.info("%s would not answer a search from here (%s)", name, answer.status_code)
                continue
            hits = read(answer.text)
            if hits:
                break
        read_pages: list[tuple[str, str, str]] = []

        async def fetch(title: str, url: str) -> None:
            try:
                answer = await client.get(url, timeout=min(left(), 8.0))
            except Exception as why:  # noqa: BLE001 - a page that will not open is one page fewer
                logger.info("could not read %s: %s", url[:80], str(why)[:100])
                return
            words = " ".join(_text_of(answer.text).split()[:WORDS_KEPT])
            if answer.status_code < 400 and not _CHALLENGED.search(words[:1500]):
                read_pages.append((title, str(answer.url), words))

        await asyncio.gather(*(fetch(title, url) for title, url in hits[:pages]))
    logger.info("looked up %r: read %s", query, [(t[:60], u[:60]) for t, u, _w in read_pages])
    return read_pages
