"""The program a page runs, where it can be had: the file a player plays, the scripts that work it.

A page that draws a thing loads the program that draws it: a Flash file its
player fetched, the scripts its canvas is run by. The browser has already
fetched them; what it fetched is listed in the page's own record of what it
loaded. The ones that are programs are fetched again through the page (from
its cache, as a rule), never larger than ``MOST_BYTES``, and read for how the
thing is worked (core/perception/reading_a_program.py). Nothing fetched is
run. What was read is kept by address, so a place visited again is not
fetched again.

Where no program can be had (a page that loads none, a file refused), there is
nothing to read, and the guide is made from what is seen and told.
"""
from __future__ import annotations

import base64
import logging
import re
from typing import Any

__all__ = ["programs_of", "read_what_the_page_runs"]

logger = logging.getLogger("Skills.SovereignBrowser.TheProgram")

#: The largest program fetched, in bytes, and the most programs read for one page.
MOST_BYTES = 40 * 1024 * 1024
MOST_PROGRAMS = 6

#: What a page loaded, from its own record, with the size the browser was given of each.
_WHAT_IT_LOADED = """
(() => performance.getEntriesByType('resource').map(e => [e.name, e.initiatorType, e.transferSize || e.encodedBodySize || 0]))()
"""

#: A program the page runs, by its address: a Flash file, a script, a WebAssembly module.
_A_PROGRAM = re.compile(r"\.(?:swf|js|mjs|wasm)(?:[?#]|$)", re.I)
#: Scripts that are the page's furniture, not the thing it draws: analytics, advertising, frameworks of the site itself.
_FURNITURE = re.compile(r"google|doubleclick|analytics|gtag|tagmanager|facebook|twitter|hotjar|segment|sentry|"
                        r"cloudflare|recaptcha|hcaptcha|jquery|bootstrap|polyfill|wp-includes|wp-content/plugins|"
                        r"cookie|consent|adsby|amazon-adsystem|newrelic|optimizely", re.I)

_FETCH_IT = """
async ([url, most]) => {
  try {
    const r = await fetch(url, {cache: 'force-cache'});
    if (!r.ok) return null;
    const b = await r.arrayBuffer();
    if (b.byteLength > most) return null;
    const u = new Uint8Array(b);
    let s = '';
    for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode.apply(null, u.subarray(i, i + 0x8000));
    return btoa(s);
  } catch (e) { return null; }
}
"""


def programs_of(loaded: list[Any]) -> list[str]:
    """The addresses among what a page loaded that are programs of the thing it shows, the Flash files first."""
    found: list[tuple[int, str]] = []
    for entry in loaded or []:
        try:
            url, _kind, size = str(entry[0]), str(entry[1]), int(entry[2] or 0)
        except (IndexError, TypeError, ValueError):
            continue
        if not _A_PROGRAM.search(url) or _FURNITURE.search(url) or size > MOST_BYTES:
            continue
        found.append((0 if re.search(r"\.swf", url, re.I) else 1, url))
    return [url for _rank, url in sorted(dict.fromkeys(found), key=lambda pair: pair[0])][:MOST_PROGRAMS]


async def read_what_the_page_runs(page: Any) -> Any:
    """What the programs a page runs say of how it is worked (a ``ProgramRead``), or an empty read."""
    from core.perception.reading_a_program import ProgramRead, merged, read_program
    from core.runtime.what_she_learned import named, recall, remember

    try:
        loaded = await page.evaluate(_WHAT_IT_LOADED)
    except (RuntimeError, OSError, ValueError, TypeError, AttributeError):
        return ProgramRead()
    reads: list[ProgramRead] = []
    for url in programs_of(list(loaded) if isinstance(loaded, list) else []):
        kept = recall(named("a program read", url))
        if isinstance(kept, dict) and kept.get("read"):
            reads.append(_from_kept(kept["read"]))
            continue
        try:
            got = await page.evaluate(_FETCH_IT, [url, MOST_BYTES])
        except (RuntimeError, OSError, ValueError, TypeError, AttributeError):
            got = None
        if not got:
            continue
        try:
            read = read_program(base64.b64decode(got), url)
        except (ValueError, MemoryError):
            continue
        if read:
            logger.info("read the program at %s: %s, keys %s, pointer %s", url[:120], read.kind, read.keys, read.pointer)
            remember(named("a program read", url), {"read": _kept(read)})
            reads.append(read)
    return merged(reads) if reads else ProgramRead()


def _kept(read: Any) -> dict[str, Any]:
    """A read as it is kept: what it says of how the thing is worked, and the mechanics' signs, not the program."""
    from core.agency.mechanics_she_knows import mechanics_in

    return {"kind": read.kind, "words": read.words[:40], "keys": read.keys, "keyed": read.keyed, "pointer": read.pointer,
            "buttons": read.buttons, "mechanics": sorted(mechanics_in(program=read.text))}


def _from_kept(held: dict[str, Any]) -> Any:
    from core.perception.reading_a_program import ProgramRead

    # The mechanics found in it stand in for its text, as their names, so the guide finds them again.
    return ProgramRead(words=list(held.get("words") or []), keys=dict(held.get("keys") or {}),
                       keyed=list(held.get("keyed") or []), pointer=list(held.get("pointer") or []),
                       buttons=list(held.get("buttons") or []), kind=str(held.get("kind") or ""),
                       text="", mechanics_found=list(held.get("mechanics") or []))
