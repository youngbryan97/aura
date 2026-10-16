"""The program a page runs, where it can be had: the file a player plays, the scripts that work it.

A page that draws a thing loads the program that draws it: a Flash file its
player fetched, the scripts its canvas is run by. The browser has already
fetched them; what it fetched is listed in the page's own record of what it
loaded. The ones that are programs are fetched again through the page (from
its cache, as a rule), never larger than ``MOST_BYTES``, and read for how the
thing is worked (core/perception/reading_a_program.py). Nothing fetched is
run. Each visit acquires the bytes through the page again, normally from its
browser cache. A saved summary cannot stand in for those bytes or retain the
exact instruction sources and visual declarations available in a current read.

Where no program can be had (a page that loads none, a file refused), there is
nothing to read, and the guide is made from what is seen and told.
"""
from __future__ import annotations

import base64
import logging
import re
from typing import Any

from core.verify import invariant

__all__ = ["programs_of", "read_what_the_page_runs"]

logger = logging.getLogger("Skills.SovereignBrowser.TheProgram")

#: The largest program fetched, in bytes, and the most programs read for one page.
MOST_BYTES = 40 * 1024 * 1024
MOST_PROGRAMS = 6
#: Which reader produced a historical summary; a current visit still requires bytes.
READER = 8

#: The most of a page's own scripts read for its world, in characters, all of them together.
MOST_SCRIPT_TEXT = 2_000_000

#: The scripts written into the page itself (a game drawn on a canvas often keeps its whole world in one), and the
#: size of the largest canvas's own drawing, which is the stage its code draws on.
_ITS_OWN_SCRIPTS = """
(most) => {
  const canvases = [...document.querySelectorAll('canvas')].sort((a, b) => b.width * b.height - a.width * a.height);
  return {scripts: [...document.scripts].filter((s) => !s.src).map((s) => (s.text || '').slice(0, most)),
          stage: canvases.length ? [canvases[0].width, canvases[0].height] : null};
}
"""

#: What a page loaded, from its own record, with the size the browser was given of each.
_WHAT_IT_LOADED = """
(() => performance.getEntriesByType('resource').map(e => [e.name, e.initiatorType, e.transferSize || e.encodedBodySize || 0]))()
"""

#: A program the page runs, by its address: a Flash file, a script, a WebAssembly module.
_A_PROGRAM = re.compile(r"\.(?:swf|js|mjs|wasm)(?:[?#]|$)", re.I)
#: Scripts that are the page's furniture, not the thing it draws: analytics, advertising, frameworks and players of the
#: site itself.
_FURNITURE = re.compile(r"google|doubleclick|analytics|gtag|tagmanager|facebook|twitter|hotjar|segment|sentry|"
                        r"cloudflare|recaptcha|hcaptcha|jquery|bootstrap|polyfill|wp-includes|wp-content/plugins|"
                        r"cookie|consent|adsby|amazon-adsystem|newrelic|optimizely|/includes/|/components/|/npm/|"
                        r"node_modules|vendor|webcomponents|topnav|ruffle|emulat", re.I)
#: A file a player plays: the thing itself, where a page plays one.
_PLAYED = re.compile(r"\.(?:swf|wasm)(?:[?#]|$)", re.I)

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
    """The addresses among what a page loaded that are programs of the thing it shows: the file a player plays where
    there is one (the site's own scripts, its player and its menus included, are the page's, not the thing's), else
    the scripts that are not the site's furniture."""
    found: list[str] = []
    for entry in loaded or []:
        try:
            url, _kind, size = str(entry[0]), str(entry[1]), int(entry[2] or 0)
        except (IndexError, TypeError, ValueError):
            continue
        if _A_PROGRAM.search(url) and not _FURNITURE.search(url) and size <= MOST_BYTES and url not in found:
            found.append(url)
    played = [url for url in found if _PLAYED.search(url)]
    return (played or found)[:MOST_PROGRAMS]


async def read_what_the_page_runs(page: Any) -> Any:
    """What the programs a page runs say of how it is worked (a ``ProgramRead``), or an empty read."""
    from core.perception.reading_a_program import ProgramRead, merged, read_program
    from core.runtime.what_she_learned import named, remember

    try:
        loaded = await page.evaluate(_WHAT_IT_LOADED)
    except (RuntimeError, OSError, ValueError, TypeError, AttributeError):
        return ProgramRead()
    reads: list[ProgramRead] = []
    scripts: list[str] = []
    for url in programs_of(list(loaded) if isinstance(loaded, list) else []):
        try:
            got = await page.evaluate(_FETCH_IT, [url, MOST_BYTES])
        except (RuntimeError, OSError, ValueError, TypeError, AttributeError):
            got = None
        data = _program_bytes(got)
        if data is None:
            continue
        try:
            read = read_program(data, url)
        except (ValueError, MemoryError):
            continue
        if not read.kind.startswith("flash"):
            scripts.append(data[:MOST_SCRIPT_TEXT].decode("utf-8", "replace"))
        if read:
            logger.info("read the program at %s: %s, keys %s, pointer %s", url[:120], read.kind, read.keys, read.pointer)
            remember(named("a program read", url), {"read": _kept(read), "reader": READER})
            reads.append(read)
    out = merged(reads) if reads else ProgramRead()
    out.world = out.world or await _the_world_its_scripts_make(page, scripts)
    return out


def _program_bytes(encoded: Any) -> bytes | None:
    """A refused, malformed or oversized response supplies no current program."""
    if not isinstance(encoded, str) or not encoded or len(encoded) > 4*((MOST_BYTES+2)//3):
        return None
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, UnicodeError):
        return None
    return data if data and len(data) <= MOST_BYTES else None


@invariant('agency.current_program_requires_acquired_bytes', scope='agency',
           owner='core/skills/sovereign_browser_the_program.py', observational=False)
def _current_program_bytes_invariant() -> tuple:
    assert _program_bytes(base64.b64encode(b'current program').decode()) == b'current program'
    assert all(_program_bytes(value) is None for value in (None, '', {'old read': 'controls'}, True, 'not base64!'))
    return ()


async def _the_world_its_scripts_make(page: Any, scripts: list[str]) -> Any:
    """The world a page's scripts make, its own and those it loaded read together, as one program is
    (core/perception/reading_a_world_s_code.py); kept by the page's address. None where they make none that is read."""
    import asyncio

    from core.perception.a_world_as_its_code_makes_it import kept
    from core.perception.reading_a_world_s_code import read_a_world_s_code
    from core.runtime.what_she_learned import named, remember

    address = str(getattr(page, "url", "") or "")
    try:
        own = await page.evaluate(_ITS_OWN_SCRIPTS, MOST_SCRIPT_TEXT)
    except (RuntimeError, OSError, ValueError, TypeError, AttributeError):
        own = {}
    own = own if isinstance(own, dict) else {}
    texts = [*scripts, *(str(t) for t in (own.get("scripts") or []) if t)]
    while sum(len(t) for t in texts) > MOST_SCRIPT_TEXT and len(texts) > 1:
        texts.pop(0)
    if not texts:
        return None
    stage = own.get("stage")
    world = await asyncio.to_thread(read_a_world_s_code, texts, tuple(stage) if stage else None)
    if address:
        await asyncio.to_thread(remember, named("a world read", address), {"world": kept(world), "reader": READER})
    if world:
        logger.info("read the world its scripts make at %s: %s", address[:120], kept(world))
    return world or None


def _kept(read: Any) -> dict[str, Any]:
    """A read as it is kept: what it says of how the thing is worked, and the mechanics' signs, not the program."""
    from core.agency.mechanics_she_knows import mechanics_in
    from core.perception.a_world_as_its_code_makes_it import kept as world_kept
    from core.perception.how_a_program_plays import kept

    return {"kind": read.kind, "words": read.words[:40], "keys": read.keys, "keyed": read.keyed, "pointer": read.pointer,
            "buttons": read.buttons, "mechanics": sorted(mechanics_in(program=read.text)),
            "plays": kept(getattr(read, "plays", None)), "world": world_kept(getattr(read, "world", None))}


def _from_kept(held: dict[str, Any]) -> Any:
    from core.perception.a_world_as_its_code_makes_it import unkept as world_unkept
    from core.perception.how_a_program_plays import unkept
    from core.perception.reading_a_program import ProgramRead

    # The mechanics found in it stand in for its text, as their names, so the guide finds them again; and what its code
    # says of how it is played, as it was read.
    return ProgramRead(words=list(held.get("words") or []), keys=dict(held.get("keys") or {}),
                       keyed=list(held.get("keyed") or []), pointer=list(held.get("pointer") or []),
                       buttons=list(held.get("buttons") or []), kind=str(held.get("kind") or ""),
                       text="", mechanics_found=list(held.get("mechanics") or []), plays=unkept(held.get("plays")),
                       world=world_unkept(held.get("world")))
