"""Trying a program that was made: run without a window for whatever is pressed, tested by its rules, and measured.

A trial page holds the program, its tests and the frame that runs them, and nothing else: no network, no file but itself.
A program that never comes back is stopped and said to be that.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.making.code_parts import BEFORE, each_test_in, halves, parts_in
from core.making.design import Design
from core.making.pieces import bundle, with_needs
from core.runtime.page_gateway import get_page_gateway, page_script

logger = logging.getLogger("Making.Trying")

__all__ = ["Result", "Trial", "as_object", "page_of", "site_of", "try_page"]

_HERE = Path(__file__).resolve().parent
_TRY = page_script("() => window.__try()")
_PLAY, _RULES = page_script("() => window.__played()"), page_script("() => window.__ruled()")
#: Whether the program itself could be read: a test that cannot be leaves it as it is.
_THERE = page_script("() => { const there = () => { try { return typeof PROGRAM === 'object'; } catch (e) { return false; } };"
                     " return typeof window.__try === 'function' && (there() || (!document.getElementById('program') && document.body.innerText.length > 0)); }")
#: How long a trial may take before it is taken to be one that does not end.
TRIAL_S = 120.0


@dataclass
class Result:
    name: str
    ok: bool
    why: str = ""
    #: Whether it failed because the program broke, and not because something was not as expected.
    broke: bool = False
    #: Whether it was the test that asked for what there is not.
    the_test: bool = False


@dataclass
class Trial:
    generic: list[Result] = field(default_factory=list)
    rules: list[Result] = field(default_factory=list)
    #: Tests that hold on the program with every event ignored: they test nothing that happens in it.
    hollow: list[str] = field(default_factory=list)
    #: What of it never ran while it was played or used, as it is written, with the function it stands in.
    never_ran: list[str] = field(default_factory=list)

    def failing(self) -> list[Result]:
        return [r for r in [*self.generic, *self.rules] if not r.ok]

    def passing(self) -> list[Result]:
        return [r for r in [*self.generic, *self.rules] if r.ok]


def as_object(code: str) -> str:
    """Some code that gives a program's functions, as an expression that is the object of them: however it was written."""
    parts = parts_in(code)
    if any(not name.startswith(BEFORE) for name in parts):
        before, body = halves(parts)
        return f"(() => {{ {before}\nreturn {body}; }})()"
    code = re.sub(r"^export\s+default\s+", "", code.strip()).rstrip(";").strip()
    if parts:
        return f"(() => {{ {code}\nreturn {list(parts)[-1].removeprefix(BEFORE)}; }})()"
    return f"(() => {{ return ({code or '{}'}); }})()"


def _tests_apart(tests: str) -> list[str]:
    """Each test in a script of its own, so that one that cannot be read is one that cannot be read."""
    before, found = each_test_in(tests)
    return [*([before] if before.strip() else []), *(text + ";" for _name, text in found)]


def _script(body: str) -> str:
    return "<script>" + body.replace("</script", "<\\/script") + "</script>"


def page_of(design: Design, code: str, *, tests: str | None = None) -> str:
    """The program as one page; with ``tests``, the page that tries it."""
    head = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, "
            "initial-scale=1\"><meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; script-src 'unsafe-inline'; "
            f"style-src 'unsafe-inline'; img-src data: blob:; media-src data: blob:\"><title>{design.name}</title>")
    style = (_HERE / "frame.css").read_text(encoding="utf-8")
    frame = (_HERE / "frame.js").read_text(encoding="utf-8")
    facts = {"name": design.name, "kind": design.kind, "about": design.about, "look": {"accent": design.look.accent},
             "controls": [c.model_dump() for c in design.controls], "ticks": bool(design.moves_by_itself),
             **({"designed": design.state} if tests is not None else {})}
    parts = [_script(frame), _script(bundle(design.pieces))]
    # What stands beside the program can be called as its own, too: this.between(...) is between(...).
    beside = ", ".join(["rnd", "between", "int", "pick", "shuffle", "play", "ui", "W", "H", *with_needs(design.pieces)])
    parts.append(_script(_program(beside, facts, code)))
    if tests is None:
        parts.append(_script("Frame.open(PROGRAM);"))
    else:
        parts.append(_script("window.TESTS = []; function test(name, body) { window.TESTS.push([name, body]); }"))
        parts += [_script(text) for text in _tests_apart(tests)]
        parts.append(_script("window.__missing = () => ['initial', 'update', PROGRAM.kind === 'app' ? 'view' : 'draw'].filter((k) => typeof PROGRAM[k] !== 'function');"
                             " window.__played = () => { const missing = window.__missing(); return { missing, generic: missing.length ? [] : Frame.tried(PROGRAM, null, false).generic }; };"
                             " window.__ruled = () => ({ hollow: Frame.tried(PROGRAM, window.TESTS, true).rules, rules: Frame.tried(PROGRAM, window.TESTS, false).rules });"
                             " window.__try = () => { const played = window.__played(); return played.missing.length ? played : { ...played, ...window.__ruled() }; };"))
    return f"{head}<style>{style}</style></head><body><div id=\"program\"></div>{''.join(parts)}</body></html>"


def _program(beside: str, facts: dict[str, Any], code: str) -> str:
    return f"const PROGRAM = Object.assign({{ {beside} }}, {json.dumps(facts)}, {as_object(code)});"


def _result(row: dict[str, Any]) -> Result:
    return Result(str(row["name"]), bool(row["ok"]), str(row.get("why") or ""), bool(row.get("broke")), bool(row.get("theTest")))


async def _watched(context: Any, page: Any) -> Any:
    """A watch on which parts of the page's programs run, kept by the browser itself; None where it keeps none."""
    try:
        watch = await context.new_cdp_session(page)
        await watch.send("Profiler.enable")
        await watch.send("Profiler.startPreciseCoverage", {"callCount": True, "detailed": True})
        return watch
    except Exception as exc:  # noqa: BLE001 - a browser that keeps no such watch is one the program is tried in without it
        logger.info("What of a program runs is not watched here: %s", exc)
        return None


async def _gone_through(page: Any, watch: Any) -> dict[str, Any]:
    """The program played or used, what of it ran in that, and then its tests."""
    if watch is None:
        return dict(await get_page_gateway().read(page, _TRY, source="making.trying"))
    played = dict(await get_page_gateway().read(page, _PLAY, source="making.trying"))
    if played.get("missing"):
        return played
    ran = (await watch.send("Profiler.takePreciseCoverage")).get("result") or []
    return {**played, **dict(await get_page_gateway().read(page, _RULES, source="making.trying")), "ran": ran}


#: What something that bears on what happens does: it sets something, or makes a sound, or ends.
_BEARS = re.compile(r"[\w$\])]\s*(?:\+\+|--|[-+*/%|&]?=(?![=>]))|\.(?:push|splice|pop|shift|unshift)\(|\bplay\(")


def _never_ran(ran: list[dict[str, Any]], html: str, *, most: int = 8) -> list[str]:
    """What never ran of the program, from the browser's own count of each block of it."""
    found = re.search(r"<script>(const PROGRAM = [\s\S]*?)</script>", html)
    if found is None:
        return []
    # The browser counts in UTF-16 units.
    body = found.group(1).encode("utf-16-le")
    said = lambda a, b: body[2 * a: 2 * b].decode("utf-16-le", "replace")  # noqa: E731
    script = next((s for s in ran if s.get("functions") and s["functions"][0]["ranges"][0]["endOffset"] * 2 == len(body)), None)
    if script is None:
        return []
    rows: list[tuple[int, str]] = []
    for function in script["functions"][1:]:
        name, whole = function.get("functionName") or "", function["ranges"][0]
        if not name or name == "PROGRAM":
            continue
        if whole["count"] == 0:
            text = said(whole["startOffset"], whole["endOffset"])
            if _BEARS.search(text):
                rows.append((0, f"{name}(...) was never called"))
            continue
        for block in function["ranges"][1:]:
            text = said(block["startOffset"], block["endOffset"])
            if block["count"] == 0 and _BEARS.search(text):
                upto = said(whole["startOffset"], block["startOffset"])
                line = upto[upto.rfind("\n") + 1:].strip()
                rows.append((1, f"in {name}: " + " ".join(f"{line[-160:]} {text.strip()}".split())[:260]))
    return [text for _first, text in sorted(dict.fromkeys(rows), key=lambda row: row[0])][:most]


async def try_page(html: str, where: Path, browser: Any) -> Trial:
    """What trying the program in a page shows."""
    from core.rebuilding.her_build_writes import governed

    await asyncio.to_thread(governed, "write_text", where, html)
    problems: list[str] = []
    context = await browser.new_context(viewport={"width": 1180, "height": 760})
    try:
        page = await context.new_page()
        page.on("pageerror", lambda error: problems.append(str(error)))
        watch = await _watched(context, page) if "window.__played" in html else None
        await get_page_gateway().go_to(page, await asyncio.to_thread(lambda: where.resolve().as_uri()),
                                       source="making.trying", wait_until="load", timeout=20000)
        try:
            # What a page raises before it is tried is the program's: a name used before it is given a value raises
            # wherever it is read. LIVE 24 November 2026 it raised here, and a game at 24 of 25 was given up for it.
            there = not problems or await get_page_gateway().read(page, _THERE, source="making.trying")
        except Exception as exc:  # noqa: BLE001 - what a page raises as it is read is what reading it shows
            problems.append(str(exc))
            there = False
        if not there:
            return Trial(generic=[Result("it can be read", False, "; ".join(problems)[:400])])
        try:
            got = await asyncio.wait_for(_gone_through(page, watch), timeout=TRIAL_S)
        except TimeoutError:
            return Trial(generic=[Result("it comes to an end", False, f"trying it took longer than {TRIAL_S:.0f} seconds: something in it goes round without end, or a list in its state grows without end")])
        except Exception as exc:  # noqa: BLE001 - what a page raises while it is tried is what trying it shows
            return Trial(generic=[Result("it can be tried", False, str(exc)[:400])])
    finally:
        await context.close()
    generic = [_result(r) for r in got.get("generic") or []]
    if got.get("missing"):
        generic.insert(0, Result("it has its functions", False, f"these are missing: {', '.join(got['missing'])}"))
    return Trial(generic=generic, rules=[_result(r) for r in got.get("rules") or []],
                 hollow=[str(r["name"]) for r in got.get("hollow") or [] if r["ok"]],
                 never_ran=_never_ran(got.get("ran") or [], html))


_SITE_HELP = """
window.TESTS = []; function test(name, body) { window.TESTS.push([name, body]); }
const __said = () => document.body.innerText.replace(/\\s+/g, " ").toLowerCase();
const see = (text) => { if (!__said().includes(String(text).toLowerCase())) throw new Error(`“${text}” is not on the page`); };
const one = (selector) => { const el = document.querySelector(selector); if (!el) throw new Error(`there is no ${selector}`); return el; };
const all = (selector) => [...document.querySelectorAll(selector)];
const press = (text) => { const el = all("a, button, summary, [role=button], [role=tab], label").find((e) => (e.innerText || e.getAttribute("aria-label") || "").toLowerCase().includes(String(text).toLowerCase())); if (!el) throw new Error(`there is nothing to press called “${text}”`); el.click(); return el; };
const expect = (ok, why) => { if (!ok) throw new Error(why || "what was expected is not so"); };
window.__try = () => { const run = (name, body) => { try { body(); return { name, ok: true, why: "" }; } catch (e) { return { name, ok: false, why: String((e && e.message) || e).slice(0, 300) }; } };
  const generic = [run("every link goes somewhere", () => { const lost = all("a[href^='#']").filter((a) => a.getAttribute("href").length > 1 && !document.getElementById(decodeURIComponent(a.getAttribute("href").slice(1)))).map((a) => a.getAttribute("href")); expect(!lost.length, "these lead nowhere: " + lost.slice(0, 6).join(", ")); const out = all("a[href]").filter((a) => /^https?:/.test(a.getAttribute("href"))); expect(!out.length, "links out of the page: " + out.slice(0, 4).map((a) => a.href).join(", ")); }),
    run("it has one main heading", () => { const n = all("h1").length; expect(n === 1, `there are ${n} h1 headings`); }),
    run("it is not cut off or wider than its window", () => { expect(document.documentElement.scrollWidth <= window.innerWidth + 1, `it is ${document.documentElement.scrollWidth} wide in a window of ${window.innerWidth}`); }),
    run("there is something to read", () => { expect(__said().length > 400, `it says ${__said().length} characters`); }),
    run("pictures say what they are", () => { const bare = all("img:not([alt]), svg:not([aria-hidden]):not([role]):not([aria-label])").length; expect(!bare, `${bare} pictures say nothing to somebody who cannot see them: give each role="img" and aria-label, or aria-hidden="true"`); }),
    run("nothing is filler", () => { const bad = ["lorem ipsum", "your text here", "placeholder", "example.com"].filter((w) => __said().includes(w)); expect(!bad.length, "it says: " + bad.join(", ")); })];
  return { missing: [], generic, rules: window.TESTS.map(([n, b]) => run(n, b)), hollow: [] }; };
"""


def site_of(design: Design, body: str, *, tests: str | None = None) -> str:
    """A site as one document; with ``tests``, the page that tries it."""
    body = re.sub(r"(?is)</?(?:!doctype[^>]*|html[^>]*|body[^>]*)>", "", body)
    head = re.findall(r"(?is)<head[^>]*>(.*?)</head>", body)
    body = re.sub(r"(?is)<head[^>]*>.*?</head>", "", body)
    kept = "".join(re.findall(r"(?is)<style[^>]*>.*?</style>", "".join(head)))
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:\">"
            f"<meta name=\"description\" content=\"{design.about.replace(chr(34), chr(39))}\"><title>{design.name}</title>{kept}</head><body>{body}"
            + (_script(_SITE_HELP) + "".join(_script(text) for text in _tests_apart(tests)) if tests is not None else "") + "</body></html>")
