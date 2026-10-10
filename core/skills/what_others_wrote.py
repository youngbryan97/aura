"""What others have written of a thing she is about to use: the record where it is kept and what its visitors wrote
there, the wikis its fans keep, and the questions people asked of it and the answers they were given.

A person who cannot find a thing's instructions asks around. What the people who kept it wrote of it, what its fans
wrote down, and what other players asked and were told, often say what its own screens never do. LIVE 2026-10-10 a
game's own screens never said that waking the dog loses it; its fans' wiki did ("Setting the traps at the wrong
location or waking Spike up from his sleep results in a loss"). Her searches had found pages about a talking-cat app.

Each source is asked through its own open interface, as a program it is meant to answer, introducing itself as what
it is. One that refuses (a challenge, a 403) is not argued with. What comes back is (who said it, title, where, words),
for taking stock to weigh with everything else (core/cognition/taking_stock.py); a page is kept only where it is of
the thing.

Where to look for its fans' wiki comes from what the thing is part of: the series its name gives ("Scooby-Doo:
Scooby Snapshot"), whose it is ("Tom's Trap-o-Matic"), what the record of it says it belongs to, and who made it.

Nothing here knows a thing.
"""
from __future__ import annotations

import asyncio
import html
import logging
import re
import urllib.parse
from collections.abc import Callable
from typing import Any

__all__ = ["FANS", "KEPT", "PLAYERS", "what_others_wrote", "worlds_it_belongs_to"]

logger = logging.getLogger("Skills.WhatOthersWrote")

#: Who said it, as she says where counsel came from.
KEPT, FANS, PLAYERS = "where it is kept", "what its fans wrote", "what players asked"
#: How she introduces herself to an interface made for programs.
_AS_HERSELF = {"User-Agent": "Aura/1.0 (a personal assistant reading how a thing is used)", "Accept-Language": "en"}
#: The most wikis tried, pages read from each, and words kept of a page.
MOST_WIKIS = 5
PAGES_A_WIKI = 2
WORDS_KEPT = 1500

Found = tuple[str, str, str, str]


def _plain(text: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", str(text or ""))).split())


def worlds_it_belongs_to(thing: str, *, said_of_it: list[str] = ()) -> list[str]:
    """What a thing is part of, as names a wiki may be kept under: its series, whose it is, and what its record says
    it belongs to and who made it."""
    names: list[str] = []
    if ":" in thing:
        names.append(thing.split(":")[0])
    owner = re.match(r"^\s*([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*){0,2})['’]s\s", thing)
    if owner:
        names.append(owner.group(1))
    names += [s for s in said_of_it if s]
    names.append(thing)
    out: list[str] = []
    for name in names:
        key = re.sub(r"[^a-z0-9]", "", name.lower().replace("&", " and "))
        if 3 <= len(key) <= 40 and key not in out and key not in ("flash", "games", "game", "software"):
            out.append(key)
    return out


async def _where_it_is_kept(client: Any, thing: str, url: str, left: Callable[[], float]) -> tuple[list[Found], list[str]]:
    """The record of the thing where it is kept (an archive's item): its description and its visitors' reviews, and
    what the record says it belongs to and who made it."""
    item = re.search(r"archive\.org/details/([^/?#]+)", url or "")
    ident = item.group(1) if item else ""
    if not ident:
        query = urllib.parse.quote(f'title:("{thing}") AND mediatype:software')
        try:
            found = (await client.get(f"https://archive.org/advancedsearch.php?q={query}&fl[]=identifier&rows=1"
                                      "&output=json", timeout=min(left(), 8.0))).json()
            docs = found.get("response", {}).get("docs") or []
            ident = str(docs[0].get("identifier") or "") if docs else ""
        except Exception as why:  # noqa: BLE001 - a record not found is one source fewer
            logger.info("where %r is kept could not be searched: %s", thing, str(why)[:100])
    if not ident:
        return [], []
    try:
        record = (await client.get(f"https://archive.org/metadata/{ident}", timeout=min(left(), 8.0))).json()
    except Exception as why:  # noqa: BLE001
        logger.info("the record of %r could not be read: %s", ident, str(why)[:100])
        return [], []
    meta = record.get("metadata") or {}
    title = str(meta.get("title") or ident)
    where = f"https://archive.org/details/{ident}"
    subjects = [s.strip() for s in re.split(r"[;,]", " ; ".join(meta["subject"]) if isinstance(meta.get("subject"), list)
                                              else str(meta.get("subject") or "")) if s.strip()]
    found: list[Found] = []
    described = _plain(meta.get("description") or "")
    if described:
        found.append((KEPT, title, where, described))
    for review in record.get("reviews") or []:
        said = _plain(f"{review.get('reviewtitle') or ''}. {review.get('reviewbody') or ''}")
        if said.strip(". "):
            found.append((KEPT, f"a review of {title}", where, said))
    makers = [str(meta.get(k) or "") for k in ("publisher", "creator") if meta.get(k)]
    return found, [*subjects, *makers]


async def _what_fans_wrote(client: Any, thing: str, worlds: list[str], on_its_page: Callable[[str, str, str], bool],
                           left: Callable[[], float]) -> list[Found]:
    """Pages of its fans' wikis that are of the thing, read through each wiki's own interface."""
    found: list[Found] = []
    for world in worlds[:MOST_WIKIS]:
        api = f"https://{world}.fandom.com/api.php"
        try:
            answer = await client.get(f"{api}?action=query&list=search&srsearch={urllib.parse.quote(thing)}&format=json"
                                      "&srlimit=5", timeout=min(left(), 6.0))
            hits = answer.json().get("query", {}).get("search") or [] if answer.status_code == 200 else []
        except Exception:  # noqa: BLE001 - no wiki by that name, or none that answers a program
            continue
        for hit in hits[:PAGES_A_WIKI]:
            title = str(hit.get("title") or "")
            try:
                page = (await client.get(f"{api}?action=parse&page={urllib.parse.quote(title)}&prop=text&format=json"
                                         "&formatversion=2", timeout=min(left(), 6.0))).json()
            except Exception:  # noqa: BLE001
                continue
            words = " ".join(_plain((page.get("parse") or {}).get("text") or "").split()[:WORDS_KEPT])
            if words and on_its_page(thing, title, words):
                found.append((FANS, title, f"https://{world}.fandom.com/wiki/{urllib.parse.quote(title)}", words))
        if found:
            break
    return found


async def _what_players_asked(client: Any, thing: str, kind: str, left: Callable[[], float]) -> list[Found]:
    """Questions people asked of the thing where players ask, and what they were told."""
    found: list[Found] = []
    site = "gaming" if kind == "game" else "superuser"
    try:
        items = (await client.get("https://api.stackexchange.com/2.3/search/excerpts?order=desc&sort=relevance"
                                  f"&q={urllib.parse.quote(thing)}&site={site}", timeout=min(left(), 6.0))).json()
        for item in (items.get("items") or [])[:4]:
            found.append((PLAYERS, _plain(item.get("title") or ""), f"https://{site}.stackexchange.com",
                          _plain(item.get("excerpt") or "")))
    except Exception as why:  # noqa: BLE001
        logger.info("questions on %s could not be searched: %s", site, str(why)[:100])
    try:
        hits = (await client.get(f"https://hn.algolia.com/api/v1/search?query={urllib.parse.quote(thing)}"
                                 "&tags=comment&hitsPerPage=5", timeout=min(left(), 6.0))).json()
        for hit in (hits.get("hits") or [])[:4]:
            found.append((PLAYERS, f"a comment on {_plain(hit.get('story_title') or '')}", "https://news.ycombinator.com",
                          _plain(hit.get("comment_text") or "")))
    except Exception as why:  # noqa: BLE001
        logger.info("comments could not be searched: %s", str(why)[:100])
    return found


async def what_others_wrote(thing: str, seconds: float, *, url: str = "", kind: str = "",
                            on_its_page: Callable[[str, str, str], bool]) -> list[Found]:
    """What others wrote of ``thing``: its record where it is kept, its fans' wiki, and players' questions; only what
    is of the thing, all within ``seconds``."""
    import time

    import httpx

    began = time.monotonic()

    def left() -> float:
        return max(0.5, seconds - (time.monotonic() - began))

    async with httpx.AsyncClient(headers=_AS_HERSELF, follow_redirects=False) as client:
        kept, belongs_to = await _where_it_is_kept(client, thing, url, left)
        fans, players = await asyncio.gather(
            _what_fans_wrote(client, thing, worlds_it_belongs_to(thing, said_of_it=belongs_to), on_its_page, left),
            _what_players_asked(client, thing, kind, left))
    players = [p for p in players if on_its_page(thing, p[1], p[3])]
    logger.info("what others wrote of %r: %s", thing, [(who, title[:50]) for who, title, _w, _t in [*kept, *fans, *players]])
    return [*kept, *fans, *players]
