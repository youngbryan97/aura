"""Searching the ways a program could be put right, by running its tests with each: thousands of them, in one page.

Two pages are opened. The first runs every test once with the program as it is, and says which blocks of it each ran, so
that what is changed is what the failing tests run and the passing ones do not. The second holds the program with every
way written into it (core/making/many_ways.py), and runs the failing tests with one way let through at a time; a way that
mends any is held to the passing ones. A loop that never ends is stopped by count, for the page cannot be asked to.

It gives what it found, the best first. Whether any of it is kept is for a whole trial to say.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.making.code_parts import each_test_in
from core.making.design import Design
from core.making.many_ways import (
    FORCED,
    Read,
    Way,
    conditions,
    evidence_of,
    forced,
    likeliest,
    ways,
    with_all,
)
from core.making.trying import ANOTHER_WAY, Result, page_of, rule_of
from core.runtime.page_gateway import get_page_gateway, page_script

logger = logging.getLogger("Making.Searching")

__all__ = ["CHECKS_SEARCHED", "Found", "Mend", "could_be_believed", "is_believed", "search"]

#: How many times round its loops a test may go before it is taken not to end; how many ways are tried at a time, for how
#: long, and how long that may take before it is taken not to end either.
ROUNDS, A_BATCH, A_BATCH_S, BATCH_S = 400_000, 150, 10.0, 60.0
#: The checks of any program that take little enough to be run with each way.
CHECKS_SEARCHED = ("what is put in a control is what it shows", "what is marked as taking something away does", "it begins", "it goes on from its state as it was designed", "it is shown", "its state is as it was designed", "what is in it is kept",
          "it begins with nothing in it that nobody put there", "it runs without breaking, whatever is used", "it runs without breaking, whatever is pressed")

_BEFORE = """
let __S = 0, __W = 0, __n = 0; const __H = new Uint8Array(%d);
const __N = () => { if (++__n > %d) { __n = 0; throw new RangeError("it does not end"); } };
// The kinds of thing that were seen at each place, while the program was run as it is.
const __seen = []; let __noting = false;
const __kind = (v) => v === null || v === undefined ? 1 : typeof v === "number" ? 2 : typeof v === "string" ? 4 : typeof v === "boolean" ? 8 : Array.isArray(v) ? 16 : typeof v === "function" ? 64 : 32;
const __R = (p, v) => { if (__noting) __seen[p] |= __kind(v); return v; };
const __K = (p, v) => { if (__seen[p] && !(__seen[p] & __kind(v))) throw new TypeError("it is not the kind of thing that is here"); return v; };
"""
_AFTER = """
window.__runs = (name) => { __n = 0; const t = window.TESTS.find(([n]) => n === name); const r = t ? Frame.tried(PROGRAM, [t], false).rules[0] : Frame.checked(PROGRAM, [name])[0]; return r || { name, ok: false, why: "not reached" }; };
window.__cover = (names) => names.map((name) => { __H.fill(0); __S = 0; __W = 0; const r = window.__runs(name); const hit = []; for (let i = 0; i < __H.length; i++) if (__H[i]) hit.push(i); return { name, ok: r.ok, broke: !!r.broke, hit, ms: r.ms || 0 }; });
// What each test that holds comes to with the program as it is, and the kinds of thing seen on the way.
window.__asItIs = (names) => { __S = 0; __W = 0; __noting = true; window.__was = {}; for (const name of names) { const r = window.__runs(name); if (r.ok) window.__was[name] = Frame.last(); } __noting = false; return true; };
window.__tryWays = ([ways, failing, passing, ms]) => { const ends = performance.now() + ms, out = []; let done = 0;
  for (const [id, place, block] of ways) { if (performance.now() > ends) break; done++; __S = place; __W = id;
    const mends = []; for (const [rule, tests] of failing) { if (block >= 0 && !tests.some(([, hit]) => hit.includes(block))) continue; if (tests.every(([name]) => window.__runs(name).ok)) mends.push(rule); }
    if (mends.length) { const breaks = []; let other = 0, held = 0; for (const [name, hit] of passing) { if (block >= 0 && !hit.includes(block)) continue; held++; if (!window.__runs(name).ok) { breaks.push(name); if (breaks.length >= mends.length + 2) break; } else if (name in window.__was && Frame.last() !== window.__was[name]) other++; } out.push({ id, mends, breaks, other, held }); }
    __S = 0; __W = 0; }
  return { out, done }; };
"""
_COVER = page_script("(names) => window.__cover(names)")
_TRY_WAYS, _AS_IT_IS = page_script("(given) => window.__tryWays(given)"), page_script("(names) => window.__asItIs(names)")


@dataclass(frozen=True)
class Mend:
    """A way that mends something: the rules it mends, the tests it breaks, and how many of the tests that hold it changes
    what comes of, of how many that it bears on."""

    way: Way
    mends: tuple[str, ...]
    breaks: tuple[str, ...] = ()
    other: int = 0
    held: int = 0


@dataclass
class Found:
    """What a search found, the likeliest first; and how far it got."""

    mends: list[Mend] = field(default_factory=list)
    tried: int = 0
    of: int = 0
    seconds: float = 0.0
    read: Read | None = None
    #: The rules that were held to by two tests.
    twice: frozenset[str] = frozenset()

    def says(self) -> str:
        return f"{self.tried} of {self.of} ways to change it in {self.seconds:.0f} s"


def _suspected(failing: Mapping[str, set[int]], passing: Mapping[str, set[int]]) -> dict[int, float]:
    """The blocks the failing tests run, by how suspect each is."""
    from core.self_modification.fix_templates import ochiai

    ranked = ochiai({n: {("it", b) for b in hit} for n, hit in failing.items()}, {n: {("it", b) for b in hit} for n, hit in passing.items()})
    return {block: score for (_it, block), score in ranked if score > 0}


def could_be_believed(mend: Mend) -> bool:
    """Whether a way that fits the tests that failed leaves most of what came of the tests that held as it was, or puts right as
    many things as it leaves otherwise: where what nothing could be put in can be, it comes out otherwise wherever it was used."""
    return not mend.breaks and mend.way.sort != FORCED and not (mend.held >= 4 and mend.other > max(mend.held / 2, len(mend.mends)))


def is_believed(mend: Mend, found: Found) -> bool:
    """Whether a way that fits its tests has more to say for it than that. A test of a program is a few lines, and much that is
    nonsense fits one: so a rule's second test, written apart, has to hold with it too, or it mends more than one thing, or
    it is a guard against what broke."""
    return len(mend.mends) > 1 or all(rule in found.twice for rule in mend.mends) or (mend.way.sort == "guard" and not mend.other)


async def search(design: Design, code: str, tests: str, failing: Sequence[Result], browser: Any, where: Path, *, seconds: float = 45.0,
                 functions: Sequence[str] = (), pays: Mapping[str, float] | None = None) -> Found:
    """Search the ways this program could be changed for those that make a failing rule's every test hold and break none that held."""
    began = time.monotonic()
    read = await asyncio.to_thread(Read.of, code, design.state, functions)
    if read is None:
        return Found()
    held = [(name, text) for name, text in each_test_in(tests)[1]]
    names = [name for name, _text in held]
    wrong = [r for r in failing if (r.name in names or r.name in CHECKS_SEARCHED) and not r.the_test]
    if not wrong:
        return Found(read=read)
    texts = dict(held)
    evidence = evidence_of([(rule_of(r.name), texts.get(r.name, ""), r.why) for r in wrong])
    # The numbers of what fails first, and then of every rule: a branch that is missing is of a rule that still holds.
    numbers = [*evidence.numbers, *evidence_of([(name, "", "") for name in [*names, *(rule.says for rule in design.rules)]]).numbers]
    evidence = dataclasses.replace(evidence, numbers=tuple(dict.fromkeys(numbers))[:20])
    covers = await _in_page(design, with_all(read, ()), tests, len(read.blocks), browser, where,
                            [(_COVER, [*names, *(r.name for r in wrong if r.name not in names)])])
    if not covers:
        return Found(read=read)
    ran = {row["name"]: row for row in covers[0]}
    failed = {name: set(row["hit"]) for name, row in ran.items() if not row["ok"] and name in {r.name for r in wrong}}
    passed = {name: set(row["hit"]) for name, row in ran.items() if row["ok"] and (not name.endswith(ANOTHER_WAY) or row["ms"] < 250)}
    suspected = _suspected(failed, passed)
    if not failed or not suspected:
        return Found(read=read)
    found = await asyncio.to_thread(ways, read, evidence, suspected)
    chosen = likeliest(read, found, suspected, pays)
    given = Found(of=len(chosen), read=read)
    # A rule is held to by each of its tests: the one that fails, and any other that does or did.
    rules: dict[str, list[list[Any]]] = {}
    for name, hit in sorted(failed.items(), key=lambda one: ran[one[0]]["ms"]):
        rules.setdefault(rule_of(name), []).append([name, sorted(hit)])
    for rule, its in rules.items():
        its += [[name, sorted(hit)] for name, hit in passed.items() if rule_of(name) == rule]
    given.twice = frozenset(rule for rule, its in rules.items() if len(its) > 1)
    fails = [[rule, its] for rule, its in rules.items()]
    # The quickest to run first: a test that takes a tenth of a second is run ten times a second.
    holds = sorted(([name, sorted(hit)] for name, hit in passed.items() if rule_of(name) not in rules), key=lambda one: ran[one[0]]["ms"])

    async def run(some: Sequence[Way], until: float, holding: list[Any]) -> list[Mend]:
        """Which of some ways mend anything, and what each does to what holds."""
        by_id, out = {way.id: way for way in some}, []

        async def batches(page: Any) -> None:
            await asyncio.wait_for(get_page_gateway().read(page, _AS_IT_IS, [*(name for name, _hit in holding), *failed], source="making.searching"), timeout=BATCH_S)
            at = 0
            while at < len(some):
                left = until - (time.monotonic() - began)
                if left <= 0.5:
                    break
                batch = some[at:at + A_BATCH]
                got = await asyncio.wait_for(get_page_gateway().read(page, _TRY_WAYS, [[[w.id, w.place, w.block] for w in batch], fails, holding, min(left, A_BATCH_S) * 1000],
                                                                     source="making.searching"), timeout=BATCH_S)
                at += max(1, int(got["done"]))
                given.tried += int(got["done"])
                out.extend(Mend(by_id[row["id"]], tuple(row["mends"]), tuple(row["breaks"]), int(row["other"]), int(row["held"])) for row in got["out"])

        await _in_page(design, with_all(read, some), tests, len(read.blocks), browser, where, (), batches)
        return out

    given.mends = await run(chosen, seconds * 0.7, holds)
    if not any(not m.breaks and len(m.mends) == len(fails) for m in given.mends):
        # What a branch would have to decide, and what it could decide that by.
        asked = await run(forced(read, suspected, 0), seconds * 0.8, holds)
        asked.sort(key=lambda m: (len(m.breaks), m.other, -suspected.get(m.way.block, 0.0), m.way.end - m.way.start))
        spots = list(dict.fromkeys((m.way.start, m.way.end) for m in asked if m.way.sort == FORCED))
        worked = await asyncio.to_thread(conditions, read, spots[:8], evidence)
        given.of += len(worked)
        first = len(chosen)
        given.mends += await run([Way(first + w.id, w.place, w.sort, w.start, w.end, w.text, w.block, w.says) for w in worked], seconds, holds)
        chosen += worked
    given.seconds = time.monotonic() - began
    order = {(way.start, way.end, way.text): n for n, way in reversed(list(enumerate(chosen)))}
    # What mends the most and breaks the least; of those, what changes least of what came of the tests that hold (Xiong and
    # others 2018: what a right mend leaves alone is how it is told from one that only fits its test); then the likeliest.
    given.mends.sort(key=lambda m: (len(m.breaks) - len(m.mends), m.other, order.get((m.way.start, m.way.end, m.way.text), 0)))
    logger.info("🔎 %s: %s, %d mend something.", design.name, given.says(), len(given.mends))
    return given


async def _in_page(design: Design, program: str, tests: str, blocks: int, browser: Any, where: Path, asks: Sequence[tuple[Any, Any]],
                   act: Any = None) -> list[Any]:
    """What is asked of a page that holds the program as it is written here."""
    from core.rebuilding.her_build_writes import governed

    if not program:
        return []
    html = page_of(design, "", tests=tests, program=program, before=_BEFORE % (max(1, blocks), ROUNDS), after=_AFTER)
    await asyncio.to_thread(governed, "write_text", where, html)
    context = await browser.new_context(viewport={"width": 1180, "height": 760})
    out: list[Any] = []
    try:
        page = await context.new_page()
        problems: list[str] = []
        page.on("pageerror", lambda error: problems.append(str(error)))
        await get_page_gateway().go_to(page, await asyncio.to_thread(lambda: where.resolve().as_uri()), source="making.searching", wait_until="load", timeout=30000)
        for script, given in asks:
            out.append(await asyncio.wait_for(get_page_gateway().read(page, script, given, source="making.searching"), timeout=BATCH_S))
        if act is not None:
            await act(page)
    except TimeoutError:
        logger.info("🔎 %s: a batch of ways did not end.", design.name)
    except Exception as exc:  # noqa: BLE001 - what a page raises with a way let through is no fault of the search's
        logger.info("🔎 %s: the search broke off: %s %s", design.name, str(exc)[:200], "; ".join(problems)[:200] if "problems" in locals() else "")
    finally:
        await context.close()
        await asyncio.to_thread(governed, "delete_file", where)
    return out
