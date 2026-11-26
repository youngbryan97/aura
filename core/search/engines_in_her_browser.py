"""What a query finds, asked of a search engine in her own browser.

The plain-HTML engines answer a handful of queries and then a challenge page, and Bing answers a fetch with decoys: pages
about the first word of the query and nothing else. LIVE 23 November 2026, four queries into a search for openings,
DuckDuckGo answered 202 with no results and Bing gave "Research - Wikipedia" for "research operations manager AI safety
remote job". The same engines answer a browser a person could be at. A challenge is never answered for anybody: an engine
that puts one up gives nothing and the next is asked.
"""
from __future__ import annotations

import logging
import re
import urllib.parse
from collections.abc import Sequence
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.page_gateway import get_page_gateway, page_script

logger = logging.getLogger("Aura.Search.Browser")

__all__ = ["ENGINES", "are_decoys", "hits_in_her_browser", "result_links", "search_in_her_browser"]

#: The engines asked, in order, by the address of a search.
ENGINES: tuple[str, ...] = (
    "https://search.yahoo.com/search?p={}",
    "https://www.startpage.com/do/search?q={}",
    "https://duckduckgo.com/?q={}",
)
#: What a page's links are read as: each with its words and the words around it.
_LINKS = page_script("""() => [...document.querySelectorAll('a[href^="http"]')].map(a => ({
  href: a.href, words: (a.innerText || '').trim().slice(0, 300),
  heading: ((a.closest('h3') || a.querySelector('h3') || {}).innerText || '').trim().slice(0, 200),
  around: ((a.closest('li, article, [class*=result], [class*=algo]') || {}).innerText || '').trim().slice(0, 600)}))""")


def _where_it_goes(href: str) -> str:
    """The page a result's link goes to, past the engine's own redirect."""
    parts = urllib.parse.urlsplit(href)
    found = re.search(r"/RU=([^/]+)/R[KS]=", href)
    if found:
        return urllib.parse.unquote(found.group(1))
    query = urllib.parse.parse_qs(parts.query)
    for key in ("uddg", "u", "url"):
        value = (query.get(key) or [""])[0]
        if value.startswith("http"):
            return value
    return href


def result_links(rows: Sequence[dict[str, Any]], *, engine: str, at_most: int) -> list[tuple[str, str, str]]:
    """The results on a page of them: (address, title, words around it), the engine's own links left out."""
    own = ".".join(urllib.parse.urlsplit(engine).netloc.split(".")[-2:])
    found: dict[str, tuple[str, str, str]] = {}
    for row in rows:
        address = _where_it_goes(str(row.get("href") or ""))
        host = urllib.parse.urlsplit(address).netloc
        title = " ".join(str(row.get("heading") or row.get("words") or "").split())
        around = " ".join(str(row.get("around") or "").split())
        # A result has words of its own around its link. A menu's link has none, and an advertisement's goes through
        # a counter of somebody else's and not to a page.
        if not host or own in host or len(title) < 8 or len(around) <= len(title) or len(address) > 600:
            continue
        key = address.split("#")[0].rstrip("/")
        if key not in found:
            found[key] = (address, title[:200], around[:400])
        if len(found) >= at_most:
            break
    return list(found.values())


def are_decoys(query: str, hits: Sequence[Any], words_of: Any) -> bool:
    """Whether results are about one word of a query of several and not about the query: what an engine gives a fetch it
    takes for a machine. ``words_of`` reads the content words of a text."""
    wanted = set(words_of(query))
    if len(wanted) < 3 or not hits:
        return False
    shared = sorted(len(wanted & set(words_of(f"{hit.title} {hit.snippet} {hit.url}"))) for hit in hits)
    return shared[len(shared) // 2] <= 1


async def search_in_her_browser(queries: Sequence[str], *, at_most: int = 10, principal: str = "research_pipeline",
                                browser: Any = None) -> dict[str, list[tuple[str, str, str]]]:
    """For each query, what the first engine that answers it finds: (address, title, words around it)."""
    from core.capabilities.phantom_browser import PhantomBrowser

    own = browser is None
    browser = browser or PhantomBrowser(visible=False, principal=principal)
    answers: dict[str, list[tuple[str, str, str]]] = {query: [] for query in queries}
    refused: set[str] = set()
    try:
        if not await browser.ensure_ready():
            return answers
        for query in queries:
            for engine in ENGINES:
                if engine in refused:
                    continue
                try:
                    if not await browser.browse(engine.format(urllib.parse.quote_plus(query))) or browser.page is None:
                        refused.add(engine)
                        continue
                    await browser.page.wait_for_timeout(1200)
                    rows = await get_page_gateway().read(browser.page, _LINKS, source="search.engines_in_her_browser")
                except Exception as exc:  # noqa: BLE001 - one engine is optional, whatever its page raises
                    logger.info("%s did not answer: %s: %s", urllib.parse.urlsplit(engine).netloc, type(exc).__name__,
                                str(exc)[:120])
                    refused.add(engine)
                    continue
                answers[query] = result_links(rows, engine=engine, at_most=at_most)
                if answers[query]:
                    break
                refused.add(engine)
    except (ImportError, OSError, RuntimeError, AttributeError) as exc:
        record_degradation("search.browser", exc, severity="info", action="searched without her browser")
    finally:
        if own:
            await browser.close()
    return answers


async def hits_in_her_browser(query: str, at_most: int, hit: Any) -> list[Any]:
    """What a query finds in her browser, as the search pipeline's hits."""
    rows = (await search_in_her_browser([query], at_most=at_most)).get(query) or []
    return [hit(title=title, url=url, snippet=around[:360], source_engine="her_browser", position=n)
            for n, (url, title, around) in enumerate(rows, 1)]
