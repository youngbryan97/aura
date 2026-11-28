"""Making a program from what somebody said: designed, its tests written before it, written, tried, mended, finished.

  1. A design: what kind of program, what has to be so of it in their own words, what it remembers, how it is worked, what
     pieces it is made of. A design that leaves out something they asked for is asked for again.
  2. Tests of its rules, before any of it exists. A test that holds on a program that does nothing tests nothing, and goes.
  3. Its rules, then its looks, in code, in parts where it is large, over a frame and pieces that are there already.
  4. It is tried: run for minutes of whatever can be pressed, searched for whether it can be won, scored in and lost, held
     to its tests, measured for speed and for what it looks like. What fails is handed back with the code, and a mend is
     kept only where less fails. A test nothing can be made to pass is read again for whether it is the test that is wrong.
  5. What is so of it, and what is not, is what is told.

Every step is a step of her work (core/understanding/workbench.py): watched as it goes, and changed by a word said to it.
"""
from __future__ import annotations

import asyncio
import hashlib
import dataclasses
import json
import logging
import re
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from itertools import chain, islice
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.making import design as how
from core.making import her_own_mends as own
from core.making.code_parts import BEFORE, assembled, each_test_in, patched, python_patched
from core.making.design import (
    Asked,
    Design,
    Rule,
    in_proportion,
    left_out,
    with_an_icon,
    with_every_ask,
)
from core.making.many_ways import SORTS
from core.making.many_ways import made as made_with
from core.making.pieces import bundle, calls, catalogue, near, pieces, used_in, with_needs
from core.making.searching import CHECKS_SEARCHED, could_be_believed, is_believed, search
from core.making.searching_python import made_python, search_python
from core.making.tools import Case, Cases, Language, languages, lines_each_ran, try_tool
from core.making.trying import (
    AGAIN,
    ANOTHER_WAY,
    CANNOT_DO_WITHOUT,
    Result,
    Trial,
    page_of,
    rule_of,
    site_of,
    try_page,
)
from core.making.what_has_worked import functions_like, get_kept
from core.rebuilding.her_model import HerModelIsAwayError
from core.understanding.carriers.finding import quoted_in
from core.understanding.workbench import PutDown, Work

logger = logging.getLogger("Making")

__all__ = ["Made", "Parts", "change", "make", "parts_of"]

#: Ask for typed data: (prompt, schema, most tokens). Write: (prompt, most tokens, warmth).
Asks = Callable[[str, Any, int], Awaitable[Any]]
Writes = Callable[..., Awaitable[str]]

#: The most tokens of a design, of tests, and of what is written of a program at once.
DESIGN_TOKENS, TEST_TOKENS, CODE_TOKENS, MEND_TOKENS, A_TEST, EDITS_TOKENS = 3200, 3600, 4600, 3200, 700, 1200
#: How many rules are built at once: a program is built a layer at a time, each tested before the next.
A_LAYER = 8
#: How many of what fails are mended at once.
AT_ONCE = 5
#: How long her own mends of one state of a program are tried for, in seconds.
OWN_S = 60.0
#: How many mends in a row may gain nothing before the last of what fails is left.
GAINED_NOTHING = 2
#: How many times a mending writes the program whole again where mending it a part at a time has stopped gaining. LIVE 28
#: November 2026 a dinner planner's form showed only once Save was pressed, and no mend of a part of it reached that.
WHOLE_AGAIN = 2
#: How many rules are tested again at once, and how many of the ways a search finds are tried whole.
A_FEW_RULES, BELIEVED = 6, 6


@dataclass
class Parts:
    """A program as it stands: its functions by name, or for a site or a tool its one text; and its tests."""

    functions: dict[str, str] = field(default_factory=dict)
    text: str = ""
    tests: str = ""
    cases: list[dict[str, Any]] = field(default_factory=list)

    def code(self) -> str:
        return self.text or assembled(self.functions)

    def copy(self) -> Parts:
        return Parts(dict(self.functions), self.text, self.tests, list(self.cases))


@dataclass
class Made:
    design: Design
    parts: Parts
    folder: Path
    path: Path
    trial: Trial
    seconds: float = 0.0
    why_not: str = ""
    #: The tests that were read again and corrected, which is told of them.
    corrected: frozenset[str] = frozenset()

    def working(self) -> bool:
        """Whether it runs, and something it was asked for is so."""
        return not self.trial.breaks() and (not self.trial.rules or any(r.ok for r in self.trial.rules))


def _first_up(text: str) -> str:
    return text[:1].upper() + text[1:]


def _prose(text: str) -> str:
    """What was said round the code."""
    said = re.sub(r"```[\s\S]*?(?:```|$)", " ", text)
    return " ".join(said.split())[:1200]


def _code(text: str) -> str:
    from core.rebuilding.her_model import _the_code_in

    return _the_code_in(text) or re.sub(r"^```[\w+-]*\n?", "", text.strip())


def _blocks(text: str) -> list[str]:
    """Each block of code in what was written, the last where it was cut off; all of it where it has no block."""
    found = [block.strip() for block in re.findall(r"```[\w+-]*\n(.*?)(?:```|$)", text, flags=re.S) if block.strip()]
    return found or [_code(text)]


class Another(BaseModel):
    function: str = Field(description="the name of one of its functions that works an answer out from what it is given")
    claims: str = Field(max_length=160, description="what has to be so of every answer it gives")
    inputs: str = Field(description="JavaScript, r => [arguments]: small arguments of the shape it takes, made by chance with r(), a number from 0 to 1")
    surely: str = Field(description="JavaScript, (...arguments) => the answer, or what the claim is about, worked out another way: by trying everything, slow and sure")
    holds: str = Field(description="JavaScript, (answer, sure, arguments) => true where the claim is so of the function's own answer")


class Anothers(BaseModel):
    checks: list[Another] = Field(default_factory=list, max_length=3)


def another_way(check: Another) -> str:
    """A check as a test: the function's answers, on many small inputs made by chance, held to what is surely so of them."""
    name = json.dumps(f"{check.claims.rstrip('.')}{ANOTHER_WAY}")
    return (f"test({name}, t => {{ const inputs = {check.inputs}, surely = {check.surely}, holds = {check.holds}; let z = 11, says_nothing = 0, probed = 0; "
            "const r = () => (z = (z * 48271) % 2147483647) / 2147483647, copy = (x) => JSON.parse(JSON.stringify(x)); "
            "Object.assign(r, { between: (a, b) => a + r() * (b - a), int: (a, b) => a + Math.floor(r() * (b - a + 1)), pick: (all) => all[Math.floor(r() * all.length)], shuffle: (all) => all.map((x) => [r(), x]).sort((a, b) => a[0] - b[0]).map((x) => x[1]) }); "
            # What it is made of going wrong is its own fault, and no fault of the function's.
            "const own = (f) => { try { return f(); } catch (e) { e.theTest = true; throw e; } }; "
            "for (let i = 0; i < 150; i++) { const args = own(() => inputs(r)), got = t.call(" + json.dumps(check.function) + ", ...copy(args)), sure = own(() => surely(...copy(args))); "
            # Where it breaks on the function's answer and breaks as well on its own sure one, it breaks on whatever it is given, and the
            # fault is the check's. LIVE 28 November 2026, "sure is not a function" was laid on a reading app's search.
            "let held; try { held = holds(got, sure, args); } catch (e) { try { holds(copy(sure), sure, args); } catch (x) { e.theTest = true; } throw e; } "
            # Held with no answer at all where the sure one is something: where the sure answer is itself nothing, holding of nothing is right.
            "if (sure !== null && sure !== undefined) { probed++; let nothing = false; try { nothing = holds(null, sure, args) === true; } catch (e) { nothing = false; } if (nothing) says_nothing++; } "
            "t.expect(held === true, `for ${JSON.stringify(args).slice(0, 220)} it gives ${JSON.stringify(got).slice(0, 220)}, and ${JSON.stringify(sure).slice(0, 120)} is what is surely so`); } "
            "if (probed >= 10 && says_nothing > (2 * probed) / 3) { const e = new Error('this holds of no answer at all, so it says nothing of one'); e.theTest = true; throw e; } });")


class _Maker:
    def __init__(self, asked: str, *, ask: Asks, write: Writes, work: Work, folder: Path, browser: Any) -> None:
        self.asked, self.ask, self.write, self.work, self.folder, self.browser = asked, ask, write, work, folder, browser
        self.design: Design | None = None
        self.parts = Parts()
        self.trial = Trial()
        self.told = ""
        #: The rules built so far, a layer at a time, and what is told where the program stands each time it is kept.
        self.has: list[str] = []
        self._looked_at: set[str] = set()
        self._searched: set[str] = set()
        #: The rules that have had a second test written of them.
        self._twice: set[str] = set()
        self._guessed_at: set[str] = set()
        self._known: set[str] = set()
        self._said_of_it = ""
        #: Which versions of the program each test held for and which it failed for; and the tests that were corrected.
        self._held_by: dict[str, set[str]] = {}
        self._failed_by: dict[str, set[str]] = {}
        self.corrected: set[str] = set()
        self.when_kept: Callable[[Path], None] | None = None
        #: What paints its pictures, and the painting of them, beside everything else.
        self.paints: Any = None
        #: What her sources said of it as it was designed.
        self.consulted: Any = None
        self._consulted_on: dict[str, str] = {}
        self.painting: asyncio.Task[list[str]] | None = None
        #: Her eyes, where she is to look at what she made herself (core/making/her_view.py); and whether she has, this time.
        self.eyes: Any = None
        self.viewed = False
        #: What she thought of it when she last looked, and of what code: kept with it, so that she does not look again at what has
        #: not changed, and is told what she thought where it has.
        self.last_view: dict[str, Any] = {}
        #: The rules of it that were her own changes, not asked for.
        self._hers: set[str] = set()
        #: Each test read as right against a program as it was, by the two of them.
        self._read_as_right: set[str] = set()
        #: How her model is given room to think where she is asked what she thinks, not what to write.
        self.thinking: dict[str, Any] = {}
        self.began = time.monotonic()

    # ── 1. the design ────────────────────────────────────────────────

    async def designed(self, before: Design | None = None) -> Design | None:
        self.work.step("Deciding what it is and what has to be so of it" if before is None else "Working what you said into the design")
        got_asks = await self.ask(f"Somebody said:\n{self.asked}\n\nWhat do they ask the program to be and do, and where do they want it put?", Asked, 600)
        asks = [a for a in (got_asks.asks if isinstance(got_asks, Asked) else []) if quoted_in(a, self.asked)]
        near_by = await asyncio.to_thread(near, self.asked)
        told = (f"Somebody asked for a program:\n{self.asked}\n\nWhat they ask of it:\n" + "\n".join(f"- {a}" for a in asks)
                + f"\n\nEngineered pieces there are to make it from:\n{catalogue()}\n"
                + f"\nFor a tool in Python, these:\n{catalogue('python')}\n"
                + (f"Nearest what they asked: {', '.join(near_by)}\n" if near_by else "")
                + f"\nTools can be written in: {', '.join(languages())}.\n")
        if self.told:
            told += f"\nYou have told them of it, and what you told them is kept to where it goes with what they asked:\n{self.told}\n"
        if before is None:
            # Her sources are asked about what was asked for and what she told them of it, and not about the pieces and
            # languages she makes things from. LIVE 28 November 2026, asked for a little app to keep track of the books
            # somebody reads, given those she asked how to structure a Python command line, twelve pages about argparse came
            # back, and it was designed as a command-line tool twice.
            told += await self._what_her_sources_say(f"Somebody asked for a program:\n{self.asked}"
                                                     + (f"\n\nYou have told them of it:\n{self.told}" if self.told else ""))
        if before is not None:
            told += f"\nIt was designed as:\n{before.model_dump_json()}\n\nWhat they have said since is the last of what they asked. "
        wrong: list[str] = []
        best: Design | None = None
        for _try in range(3):
            got = await self.ask(told + (f"\nA design of it was wrong: {'; '.join(wrong)}.\n" if wrong else "")
                                 + "\nDesign it, so that everything they asked for is a rule, and so is what anybody would expect of "
                                   "a finished program of its kind.", Design, DESIGN_TOKENS)
            if not isinstance(got, Design):
                wrong = ["it was not a whole design"]
                continue
            was = wrong
            wrong = left_out(got)
            got = with_every_ask(got, asks)
            got = with_an_icon(got)
            got.rules = in_proportion(got, asks)
            got.rules = await asyncio.to_thread(self._not_the_windows, got)
            held = pieces("python") if got.kind == "tool" else pieces()
            got.pieces = [name for name in dict.fromkeys(got.pieces) if name in held]
            if best is None or not wrong or len(wrong) < len(was):
                best = got
            if not wrong:
                break
            self.work.trouble("The design is not whole: " + "; ".join(wrong))
        if best is not None:
            self.design = best
            self.work.happened("design", design=best.model_dump())
            self.work.says(f"{best.name}: {best.about} {len(best.rules)} rules"
                           + (f", made from {', '.join(best.pieces)}" if best.pieces else "") + ".")
        return best

    async def _what_her_sources_say(self, about: str) -> str:
        """Her library, the web and her own model, asked what is worth knowing to make it well: a reading of what they say,
        each with where it came from (core/making/consulting.py), or nothing where none of them could be asked."""
        from core.making.consulting import consult

        self.work.step("Asking her library, the web and her own model what is worth knowing")
        try:
            consulted = await consult(about, ask=self.ask, write=self.write, advice_on=f"Somebody asked for a program:\n{self.asked}")
        except (OSError, RuntimeError, ValueError, TypeError) as why:
            self.work.trouble(f"Her sources could not be asked ({type(why).__name__}).")
            return ""
        self.consulted = consulted
        if consulted.said:
            self.work.happened("consulted", asked=consulted.asked, said=[{"question": s.question, "source": s.source, "where": s.where, "text": s.text} for s in consulted.said])
        return f"\n{consulted.reading()}\n" if consulted.said else ""

    async def _what_her_sources_say_of(self, failing: Result) -> str:
        """Where a mend has gained nothing, what her sources say of what still fails, asked once for each thing: how it is
        worked out, what it is called, how others did it."""
        from core.making.consulting import consult

        assert self.design is not None
        rule = rule_of(failing.name)
        if rule in self._consulted_on:
            return self._consulted_on[rule]
        self.work.step(f"Asking her library, the web and her own model about: {rule[:90]}")
        about = f"Making {self.design.name}: {self.design.about}\nThis of it is not so yet, and mends have not made it so: {rule}\nWhy, as trying it shows: {failing.why[:400]}"
        try:
            consulted = await consult(about, ask=self.ask, write=self.write, advice_on=about)
        except (OSError, RuntimeError, ValueError, TypeError) as why:
            self.work.trouble(f"Her sources could not be asked ({type(why).__name__}).")
            consulted = None
        if consulted is not None and consulted.said:
            self.work.happened("consulted", asked=consulted.asked, said=[{"question": s.question, "source": s.source, "where": s.where, "text": s.text} for s in consulted.said])
        self._consulted_on[rule] = f"\n{consulted.reading()}\n" if consulted is not None and consulted.said else ""
        return self._consulted_on[rule]

    def _not_the_windows(self, design: Design) -> list[Rule]:
        """Its rules but those nobody asked for that the window it runs in keeps by itself: a start screen built into a
        program that has one is a second one, and the program stands waiting behind it."""
        asked = [rule.says for rule in design.rules if rule.of]
        keys = design.controls
        if design.kind == "game" and asked:
            its_own = _the_windows_own([key.does for key in keys], asked)
            design.controls = [key for key, window_s in zip(keys, its_own, strict=True) if not window_s] or keys
        mine, window = [rule for rule in design.rules if not rule.of], _THE_WINDOW_HAS.get(design.kind, ())
        own = _the_windows_own([rule.says for rule in mine], [*asked, *(key.does for key in design.controls)], window) if window else [False] * len(mine)
        gone = [rule for rule, its in zip(mine, own, strict=True) if its]
        dropped = [*(rule.says.rstrip(".") for rule in gone), *(key.does for key in keys if key not in design.controls)]
        if dropped:
            self.work.says("The window it runs in has this already: " + "; ".join(dropped) + ".")
        return [rule for rule in design.rules if rule not in gone]

    # ── 2. its tests ─────────────────────────────────────────────────

    async def tested(self, rules: list[str]) -> None:
        design = self.design
        assert design is not None
        self.work.step("Writing its tests, before the code they test")
        listed = "\n".join(f"- {rule}" for rule in rules)
        if design.kind == "tool":
            got = await self.ask(f"Write cases that test these rules, two for each where it matters, with what each prints worked "
                                 f"out exactly:\n{listed}", Cases, TEST_TOKENS, given=self._given())
            cases = list(got.cases) if isinstance(got, Cases) else []
            self.parts.cases += [c.model_dump() for c in cases]
            self.work.tried("Cases to run", [{"name": f"{c.rule}: {' '.join(c.args) or c.typed[:40] or 'as it is'}", "ok": True} for c in cases])
            return
        said = await self.write(f"{_HOW_TESTS[design.kind]}\n\nWrite a test for each of these rules, as JavaScript, the tests alone and "
                                f"nothing of the program:\n{listed}", TEST_TOKENS, given=self._given())
        self.work.says(_prose(said))
        before, found = each_test_in(_code(said))
        code = "\n\n".join([*([before.strip()] if before.strip() else []), *(f"{text};" for _name, text in found)])
        if not found:
            self.work.trouble("No test of it could be read.")
        self.parts.tests = (self.parts.tests + "\n\n" + code).strip()
        self.work.wrote("tests", code)

    # ── 3. its code ──────────────────────────────────────────────────

    def _given(self) -> str:
        """What every ask about the program begins with: how such a program is written, and the design of this one. It
        goes to her model as a message of its own, so the next ask reads on from the end of it."""
        assert self.design is not None
        return f"{self._interface()}\n\n{self.design.said()}"

    def _interface(self) -> str:
        design = self.design
        assert design is not None
        if design.kind == "tool":
            language = self.language()
            run = " ".join(part.format(file=language.file, out="./program") for part in language.run) + " [arguments]"
            told = how.TOOL.format(language=language.name, run=run.replace(sys.executable, "python"), has=language.has)
            if language.name == "python" and design.pieces:
                told += f"\nThese are there to be called, above what is written, and are not written again:\n{calls(design.pieces, 'python')}"
            return told
        told = how.how_it_is_written(design)
        drawn_with: list[str] = []
        if design.kind == "game":
            told += f"\nThe pen: {how.PEN}"
            drawn_with = list(how.DRAWN_WITH)
        if any(p.kind != "icon" for p in design.pictures):
            told += f"\n{how.PICTURES[design.kind]}" if design.kind in how.PICTURES else ""
        if design.pieces or drawn_with:
            told += f"\nPieces that are there to be called, by these names:\n{calls(with_needs([*drawn_with, *design.pieces]))}"
        return told

    def language(self) -> Language:
        held = languages()
        design = self.design
        return held.get((design.language if design else "").lower()) or held["python"]

    def layers(self) -> list[list[str]]:
        design = self.design
        assert design is not None
        rules = [rule.says for rule in design.rules]
        if design.kind in {"site", "tool"} or len(rules) <= A_LAYER + 3:
            return [rules]
        return [rules[at:at + A_LAYER] for at in range(0, len(rules), A_LAYER)]

    def begin_painting(self) -> None:
        """Paint its pictures beside the writing of it: each takes a second or two, and the first some more."""
        from core.making.pictures import paint

        if self.paints is None or self.design is None or not self.design.pictures or (self.painting is not None and not self.painting.done()):
            return
        others = [p for p in self.folder.parent.glob("*/icon.png") if p.parent != self.folder]
        self.painting = asyncio.create_task(paint(self.design, self.folder, self.work, paints=self.paints, others=others, known_by=self._known_by))

    async def _known_by(self, tried: list[str]) -> str:
        """The one thing it would be known by, other than what was tried."""
        from core.making.design import KnownBy

        told = f"{self.design.name}: {self.design.about}" if self.design is not None else ""
        said = await self.ask(f"{told}\n\n" + (f"Painted as its icon, and it could not be told what it was: {'; '.join(tried)}\n\n" if tried else "") + "What one thing would it be known by?", KnownBy, 200)
        return said.thing if isinstance(said, KnownBy) else ""

    async def painted(self) -> None:
        """Until its pictures are painted, or cannot be."""
        if self.painting is None:
            return
        if not self.painting.done():
            self.work.step("Finishing its pictures")
        try:
            await self.painting
        except (OSError, RuntimeError, ValueError, ImportError, TypeError) as why:
            logger.info("Its pictures were not all painted: %s", why)
            self.work.trouble(f"Its pictures could not all be painted ({type(why).__name__}).")

    async def built(self) -> None:
        """The program, a layer of its rules at a time: tests of the layer, its code, and what fails mended before the next."""
        self.begin_painting()
        layers = self.layers()
        for n, rules in enumerate(layers):
            rules = [rule for rule in rules if rule not in self.has]
            if not rules:
                continue
            if len(layers) > 1:
                self.work.happened("layer", n=n + 1, of=len(layers), rules=rules)
            await self.tested(rules)
            if not (self.parts.functions or self.parts.text.strip()):
                await self.coded(rules)
                self.has += rules
                await self.mended()
            else:
                await self.added_where_it_is_not(rules)
            await self.kept()
        await self.worked_out_another_way()
        await self.finished()
        await self.her_own_that_do_not_hold_let_go()
        await self.looked_at_by_her()

    async def worked_out_another_way(self) -> None:
        """What its functions work out, held to the same worked out another way: by a slow sure method over many small inputs
        made by chance. LIVE 24 November 2026 an app at 20 of 20 was to settle up in the fewest payments, and its tests held
        it to settling up. A check that cannot be kept to by what she then writes is told, and not let through."""
        assert self.design is not None
        if self.design.kind not in {"app", "game"} or ANOTHER_WAY in self.parts.tests or not self.design.functions:
            return
        self.work.step("Working out another way what its functions work out")
        got = await self.ask(f"Its functions:\n```js\n{self.parts.code()}\n```\n\nWhich of these work an answer out that has to be right: a sum, a best, a "
                             "fewest, an order, a path, whether a rule of play allows something? For up to three of them, say what has to be so of every "
                             "answer, and how that is worked out another way, by trying everything, so that the two can be held to each other. None, "
                             "where none does.", Anothers, EDITS_TOKENS + 900, given=self._given())
        checks = [c for c in (got.checks if isinstance(got, Anothers) else []) if c.function in self.parts.functions]
        if not checks:
            self.work.says("None of them works out an answer that can be.")
            return
        was = self.parts.tests
        self.parts.tests = "\n\n".join([was, *(another_way(c) for c in checks)]).strip()
        self.work.wrote("tests", "\n\n".join(another_way(c) for c in checks))
        await self.mended()
        await self.where_two_workings_part(checks)
        await self.kept()

    async def where_two_workings_part(self, checks: list[Another]) -> None:
        """Where a function and the other way of working its answer out still differ: a third, written without sight of
        either, and whichever two agree. Where no two do, what was asked does not settle it, and they are asked, with the
        case they differ on (as ClarifyGPT, Mu and others 2023, finds what a request leaves open)."""
        assert self.design is not None
        for check in checks:
            name = f"{check.claims.rstrip('.')}{ANOTHER_WAY}"
            parted = next((r for r in self.trial.rules if r.name == name and not r.ok and not r.the_test), None)
            said = next((line for line in self.design.functions if line.split("(", 1)[0].strip() == check.function), "")
            if parted is None or not said or check.function not in self.parts.functions:
                continue
            self.work.step(f"Working out {check.function} a third time, without sight of the other two")
            third = await self.write(f"Write this one of its functions, whole, as one JavaScript object in one block: `({{ {check.function}(...) {{ ... }} }})`.\n"
                                     f"{said}\nWhat is so of every answer it gives: {check.claims}", A_TEST + 500, warmth=0.6, given=self._given())
            found = patched({}, _code(third))[0].get(check.function)
            if not found:
                continue
            parts = self.parts.copy()
            parts.functions[check.function] = found
            trial = await self.tried(parts, quiet=True)
            agrees = next((r.ok for r in trial.rules if r.name == name), False)
            if agrees and self._wrong(trial) < self._wrong(self.trial):
                self.work.says(f"Two of three workings of {check.function} agree, and the program has the one they agree on.")
                self.parts, self.trial = parts, trial
                continue
            before, tests = each_test_in(self.parts.tests)
            self.parts.tests = "\n\n".join([*([before.strip()] if before.strip() else []), *(f"{text};" for n, text in tests if n != name)])
            self.trial = await self.tried(quiet=True)
            self._known.discard(name)
            self.work.asks(f"{check.claims.rstrip('.')}: is that what you want of it? {_first_up(parted.why)}", "what it does now")


    async def added_where_it_is_not(self, rules: list[str], words: str = "") -> None:
        """More of its rules: nothing written where the program already holds to them, and nothing kept that leaves it worse
        than it was. LIVE 24 November 2026 a program at 18 of 18 had a rule built in that it already held to, and was at 14."""
        was, self._known = self.parts.copy(), set()
        before = await self.tried(what="Trying it with the tests of what is next")
        wrong = self._wrong(before)
        self.trial = before
        self.has += [rule for rule in rules if rule not in self.has]
        # A test that holds whatever happens says nothing of whether it does, and neither does one that is not there.
        shown = {r.name for r in before.rules if r.ok and r.name not in before.hollow}
        if rules and not words and not before.failing() and all(rule in shown for rule in rules):
            self.work.says("It does that already.")
            return
        await self.added(rules, words)
        known = set(self._known)
        await self.mended()
        self._known |= known
        if self._wrong(self.trial) > wrong:
            self.work.says(f"What was built in left it worse ({len(self.trial.failing())} things fail, and {len(before.failing())} did), so it is as it was before.")
            tests, cases = self.parts.tests, self.parts.cases
            self.parts = was
            self.parts.tests, self.parts.cases = tests, cases
            self.trial = await self.tried(what="Trying it as it was")

    async def gone_on_with(self) -> None:
        """A program as it was kept: what fails of it mended, and what of it was not built yet built."""
        await self.mended()
        await self.built()

    async def coded(self, rules: list[str]) -> None:
        design = self.design
        assert design is not None
        if design.kind in {"site", "tool"}:
            what = "its page" if design.kind == "site" else f"it, in {self.language().name}"
            self.work.step(f"Writing {what}")
            said = await self.write("Say in a sentence or two how you will do it, then write "
                                    f"{'the body of the page, whole' if design.kind == 'site' else 'the whole file'}, in one block.",
                                    CODE_TOKENS, given=self._given())
            # What she says of it is said where it is taken in, once.
            self._written(said, what)
            return
        first = "" if len(rules) == len(design.rules) else "Only these of its rules are built now:\n" + "\n".join(f"- {r}" for r in rules) + "\n\n"
        # What it is written to pass is in front of whoever writes it: the names its tests use for its controls and its state
        # are the names it is to have. Measured over three makings on 28 November 2026, nine tenths of their time went to
        # mending, much of it what was written with names its tests did not use, for the tests had been written and never shown.
        first += self._its_tests(rules)
        functions = "initial, update, score, over and hud" if design.kind == "game" else "initial and update"
        earlier = await asyncio.to_thread(functions_like, design.functions, self.folder.parent, kind=design.kind, but=self.folder)
        if earlier:
            self.work.says(f"Starting from {len(earlier)} functions of earlier programs whose checks held.")
            first += "Functions that worked in earlier programs of hers, to start from where they fit:\n```js\n" + ",\n\n".join(earlier) + "\n```\n\n"
        self.work.step("Writing its rules: how it begins and what each event does")
        said = await self.write(f"{first}Say in a sentence or two how you will do it, then write {functions}, as one JavaScript object in "
                                "one block: `({ ... })`. How it is shown is written after.", CODE_TOKENS, given=self._given())
        self._written(said, "rules")
        which = "draw" if design.kind == "game" else "view"
        self.work.step("Writing how it looks")
        said = await self.write(f"{first}Its rules are written:\n```js\n{self.parts.code()}\n```\n\n"
                                f"Say in a sentence how it will look, then write {which}, as one JavaScript object in one block: "
                                f"`({{ {which}(s, {'g' if which == 'draw' else 'ui'}) {{ ... }} }})`. It is to look finished.", CODE_TOKENS, given=self._given())
        self._written(said, "looks")

    async def added(self, rules: list[str], words: str = "") -> None:
        """More of its rules, built into the program as it stands."""
        design = self.design
        assert design is not None
        self.work.step("Building in: " + "; ".join(rules)[:160] if rules else "Changing it to what you said")
        told = (("\nTo be built in now:\n" + "\n".join(f"- {rule}" for rule in rules)) if rules else "") + (f"\nThey said: {words}" if words else "")
        told += f"\n{self._its_tests(rules)}" if rules else ""
        said = await self.write(f"It is written as:\n```\n{self.parts.code()}\n```\n{told}\n\n"
                                f"Say in a sentence or two what you change, then {self._how_to_change()}.", CODE_TOKENS, given=self._given())
        self._written(said, "what changed" if self.parts.functions else "it")

    def _its_tests(self, rules: list[str]) -> str:
        """The tests of these rules, as what is written is to pass them; "" where there are none."""
        tests = _tests_named(self.parts.tests, [*rules, *(f"{rule}{AGAIN}" for rule in rules)]) if self.parts.tests else ""
        return f"Its tests, which it is to pass:\n```js\n{tests[-14000:]}\n```\n\n" if tests.strip() else ""

    def _how_to_change(self) -> str:
        design = self.design
        assert design is not None
        if design.kind == "tool" and self.language().name == "python" and self.parts.text:
            return "write the functions you change or add, each whole, in one block; what you leave out stays as it is"
        if design.kind in {"site", "tool"}:
            return f"write {'the whole body of the page' if design.kind == 'site' else 'the whole file'} again, changed, in one block"
        return ("write the functions you change or add, each whole, as one JavaScript object in one block: `({ ... })`. What you leave out "
                "stays as it is")

    def _written(self, said: str, name: str, parts: Parts | None = None) -> list[str]:
        """What was written, taken into the program; the names of what changed."""
        design, parts = self.design, parts or self.parts
        assert design is not None
        self.work.says(_prose(said))
        code = _code(said)
        if design.kind in {"site", "tool"}:
            if design.kind == "tool" and self.language().name == "python" and parts.text:
                parts.text, changed = python_patched(parts.text, "\n\n".join(_blocks(said)))
            else:
                changed = ["it"] if code and code != parts.text else []
                parts.text = code or parts.text
            if design.kind == "tool" and self.language().name == "python":
                design.pieces += [name for name in used_in(parts.text, "python") if name not in design.pieces]
            self.work.wrote(name, parts.text, "html" if design.kind == "site" else self.language().name)
            return changed
        changed = []
        for block in _blocks(said):
            parts.functions, new = patched(parts.functions, block)
            changed += [name for name in new if name not in changed]
        for piece in used_in(assembled(parts.functions)):
            if piece not in design.pieces and BEFORE + piece not in parts.functions:
                design.pieces.append(piece)
        if changed:
            self.work.wrote(", ".join(c.removeprefix(BEFORE) for c in changed[:8]),
                            ",\n\n".join(parts.functions[c] for c in changed))
        return changed

    # ── 4. trying and mending ────────────────────────────────────────

    async def tried(self, parts: Parts | None = None, what: str = "Trying it", *, quiet: bool = False) -> Trial:
        design, parts = self.design, parts or self.parts
        assert design is not None
        if design.kind == "tool":
            trial = await try_tool(self.whole(parts), self.language(), [Case.model_validate(c) for c in parts.cases])
        elif design.kind == "site":
            trial = await try_page(site_of(design, parts.text, tests=parts.tests), self.folder / "trial.html", self.browser)
        else:
            trial = await try_page(page_of(design, parts.code(), tests=parts.tests), self.folder / "trial.html", self.browser)
        gone = {f"({does})" for does in self._dead_controls_go(trial.failing())}
        trial.generic = [r for r in trial.generic if not any(r.name.endswith(does) for does in gone)]
        if trial.hollow:
            # A test that holds where nothing happens says nothing of what happens, and one that fails there may.
            keep = set(trial.hollow)
            trial.rules = [r for r in trial.rules if r.name not in keep or not r.ok]
        for r in trial.rules:
            (self._held_by if r.ok else self._failed_by).setdefault(r.name, set()).add(parts.code())
        if not quiet:
            self.work.tried(what, [{"name": r.name, "ok": r.ok, "why": r.why} for r in [*trial.generic, *trial.rules]])
        return trial

    async def mended(self) -> None:
        """What fails, handed back with the code, as long as that gains; then a doubt of the tests that still fail."""
        assert self.design is not None
        self.trial = await self.tried()
        self._known = set()
        nothing_gained = doubted = rewritten = 0
        put_right = False
        while self.trial.failing():
            failing = self.trial.failing()
            # A test that asks for what there is not is a test to be put right, and no fault of the program's.
            amiss = [r for r in failing if r.the_test]
            if amiss and not put_right:
                put_right = True
                if await self._tests_read_again(amiss):
                    self._known = set()
                    self.trial = await self.tried(what="Trying it, with the tests put right")
                    continue
            if nothing_gained >= (GAINED_NOTHING if any(not r.ok for r in self.trial.generic) else 1):
                if doubted or self.design.kind == "site" or not [r for r in self.trial.rules if not r.ok]:
                    # What mending a part at a time does not reach may be how the whole of it is made: written whole again, kept where
                    # less fails, before what still fails is left.
                    if rewritten < WHOLE_AGAIN:
                        rewritten += 1
                        if await self._written_whole_again():
                            nothing_gained, self._known = 0, set()
                            continue
                    break
                doubted = 1
                if not await self._tests_read_again([r for r in self.trial.rules if not r.ok]):
                    break
                nothing_gained = 0
                self._known = set()
                self.trial = await self.tried(what="Trying it, with the tests read again")
                continue
            # What is so of any program is no test of hers that may be the one that is wrong: a mend that puts that right, and
            # something else, is found by searching before a second test of anything is written.
            if await self._a_searched_mend(failing, of_any_program=True):
                nothing_gained = 0
                continue
            if await self._tested_again(failing):
                continue
            if await self._a_searched_mend(failing) or await self._her_own_mend(failing) or await self._small_edits(failing):
                nothing_gained = 0
                continue
            parts = await self._a_mend(failing, warm=0.2 + 0.3 * nothing_gained)
            if parts is None:
                nothing_gained += 1
                continue
            trial = await self.tried(parts, "Trying the mend")
            before, after = self._wrong(self.trial), self._wrong(trial)
            if after >= before and (better := await self._the_best_of(parts, before)) is not None:
                parts, trial = better
                after = self._wrong(trial)
            if after < before:
                self.work.says(f"Kept: {len(failing)} things failed before it and {len(trial.failing())} do now"
                               + (", and what was underneath them is put right." if len(trial.failing()) >= len(failing) else "."))
                self.parts, self.trial, nothing_gained = parts, trial, 0
                await self._remembered(failing, self._said_of_it, "written again")
                await self.kept()
            else:
                nothing_gained += 1
                self.work.says(f"Not kept: {len(failing)} things failed before it and {len(trial.failing())} after.")

    async def _written_whole_again(self) -> bool:
        """The program written whole again, its rules and then how it looks, from its design, its tests and what still fails of
        it; kept where less fails. Whether it was."""
        design = self.design
        assert design is not None
        if design.kind not in {"app", "game"} or not self.parts.functions:
            return False
        failing = self.trial.failing()
        self.work.step("Writing it whole again, from its design, its tests and what still fails")
        told = "\n".join(f"- {r.name}: {r.why[:300]}" for r in failing[:12])
        tests = f"\nIts tests:\n```js\n{self.parts.tests[-14000:]}\n```\n" if self.parts.tests else ""
        functions = "initial, update, score, over and hud" if design.kind == "game" else "initial and update"
        which = "draw" if design.kind == "game" else "view"
        parts = self.parts.copy()
        said = await self.write(f"It is written as:\n```\n{self.parts.code()}\n```\n{tests}\nTrying it, these fail, and mending it a part at a time has "
                                f"not put them right:\n{told}\n\nSay in a few sentences what is wrong in how it is made, then write {functions} again, "
                                "with every function they call, as one JavaScript object in one block: `({ ... })`.", CODE_TOKENS, warmth=0.4, given=self._given())
        if not self._written(said, "its rules, written again", parts):
            return False
        said = await self.write(f"Its rules are written again:\n```js\n{parts.code()}\n```\n{tests}\nWhat failed:\n{told}\n\nSay in a sentence how it "
                                f"will look, then write {which} again, with every function it calls, as one JavaScript object in one block: "
                                f"`({{ {which}(s, {'g' if which == 'draw' else 'ui'}) {{ ... }} }})`. It is to look finished.", CODE_TOKENS, warmth=0.4, given=self._given())
        self._written(said, "how it looks, written again", parts)
        trial = await self.tried(parts, "Trying it written again")
        if self._wrong(trial) < self._wrong(self.trial):
            self.work.says(f"Kept: written again, {len(trial.failing())} things fail, and {len(failing)} did.")
            self.parts, self.trial = parts, trial
            await self.kept()
            return True
        self.work.says(f"Not kept: written again, {len(trial.failing())} things fail, and {len(failing)} did.")
        return False

    def _wrong(self, trial: Trial) -> tuple[int, float]:
        """How wrong a trial shows it, beside every check that any trial of it has got as far as."""
        self._known |= {r.name for r in [*trial.generic, *trial.rules]}
        return _how_wrong(trial, self._known)

    async def _tested_again(self, failing: list[Result]) -> bool:
        """A second test of each rule whose test fails, written apart from the first; whether there are any new. Where it
        holds, one of the two is wrong, and the first is read again before anything is mended to it. Where it fails too, the
        program is what is wrong, and a mend has to make both hold: much that is no mend at all will fit one test."""
        assert self.design is not None
        rules = [r.name for r in failing if r in self.trial.rules and not r.the_test and not r.name.endswith((AGAIN, ANOTHER_WAY)) and r.name not in self._twice][:A_FEW_RULES]
        if self.design.kind not in {"app", "game"} or not rules:
            return False
        self._twice |= set(rules)
        self.work.step("Testing again, another way, what fails")
        said = await self.write(f"{_HOW_TESTS[self.design.kind]}\n\nWrite a test for each of these rules, as JavaScript, the tests alone and nothing of the "
                                "program:\n" + "\n".join(f"- {rule}" for rule in rules), min(TEST_TOKENS, 200 + 320 * len(rules)), warmth=0.7, given=self._given())
        first = dict(each_test_in(self.parts.tests)[1])
        new = []
        for (name, text), rule in zip(each_test_in(_code(said))[1], rules, strict=False):
            rule = name if name in rules else rule
            if " ".join(text.split()) != " ".join(first.get(rule, "").split()):
                new.append(_named(text, f"{rule}{AGAIN}"))
        if not new:
            return False
        was = self.parts.tests
        self.parts.tests = "\n\n".join([was, *(f"{text};" for text in new)])
        self.work.wrote("tests", "\n\n".join(new))
        trial = await self.tried(what="Trying it, with those tests too")
        # One that asks for what there is not says nothing of the program.
        odd = {r.name for r in trial.rules if r.name.endswith(AGAIN) and not r.ok and r.the_test}
        if odd:
            before, tests = each_test_in(self.parts.tests)
            self.parts.tests = "\n\n".join([*([before.strip()] if before.strip() else []), *(f"{text};" for name, text in tests if name not in odd)])
            trial.rules = [r for r in trial.rules if r.name not in odd]
        self._known = set()
        self.trial = trial
        holds = {rule_of(r.name) for r in trial.rules if r.ok and r.name.endswith(AGAIN)}
        parted = [r for r in trial.rules if not r.ok and r.name in holds]
        if parted:
            self.work.says("Two tests of the same rule part: one holds and one fails. The one that fails is read again before anything is mended to it: " + "; ".join(r.name for r in parted)[:300])
            if await self._tests_read_again(parted):
                self._known = set()
                self.trial = await self.tried(what="Trying it, with the tests put right")
        return True

    async def _a_searched_mend(self, failing: list[Result], *, of_any_program: bool = False) -> bool:
        """A mend found by searching, thousands of ways tried against its tests without her model; whether one was kept.
        ``of_any_program`` is for one that puts right something that is so of any program."""
        assert self.design is not None
        anywhere = {r.name for r in self.trial.generic if not r.ok and r.name in CHECKS_SEARCHED}
        if of_any_program and not anywhere:
            return False
        code = self.parts.code()
        python = self.design.kind == "tool" and self.language().name == "python"
        with_tests = f"{of_any_program}\n{code}\n{self.parts.tests}\n{json.dumps(self.parts.cases)}"
        if not (python or self.design.kind in {"app", "game"}) or with_tests in self._searched:
            return False
        self._searched.add(with_tests)
        self.work.step("Searching the ways it could be changed, each tried against its tests")
        pays = {sort: get_kept().pays(sort) for sort in SORTS}
        if python:
            found = await self._searched_as_python(code, pays)
        else:
            found = await search(self.design, code, self.parts.tests, failing, self.browser, self.folder / "searched.html",
                                 functions=list(self.parts.functions), pays=pays)
        likely = [m for m in found.mends if could_be_believed(m)]
        if found.tried:
            self.work.says(f"Tried {found.says()}, without her model: {len(found.mends)} make something hold that failed"
                           + (f", and {len(likely)} of them leave the rest as it was." if found.mends else "."))
        before, had = self._wrong(self.trial), {r.name for r in failing}
        # What fails of a check of any program, more of it than did: a mend of one thing is not to be another of the same.
        amiss = {r.name: r.got for r in self.trial.generic if not r.ok}
        best: tuple[tuple[int, float], Parts, Trial, Any] | None = None
        for mend in sorted(likely, key=lambda m: not is_believed(m, found))[:BELIEVED]:
            parts = self._with_code(made_python(code, mend.way) if python else made_with(code, mend.way))
            if parts is None:
                continue
            trial = await self.tried(parts, quiet=True)
            wrong, now = self._wrong(trial), {r.name for r in trial.failing()}
            # What two things bear out: two tests of the rule, or two things that failed and now hold.
            borne_out = is_believed(mend, found) or len({rule_of(name) for name in had - now}) > 1
            if of_any_program and not anywhere & (had - now):
                continue
            if any(not r.ok and r.got < amiss.get(r.name, r.got) for r in trial.generic):
                continue
            if wrong < (best[0] if best else before) and not now - had and borne_out:
                best = (wrong, parts, trial, mend)
            else:
                get_kept().tried(mend.way.sort, kept=False)
            if best is not None and not best[2].failing():
                break
        if best is None:
            return False
        _how, self.parts, trial, mend = best
        get_kept().tried(mend.way.sort, kept=True)
        await self._remembered(failing, mend.way.says, mend.way.sort)
        self.work.says(f"Kept a mend she found by searching: {mend.way.says}. {len(failing)} things failed before it and {len(trial.failing())} do now.")
        self.trial = trial
        self.work.tried("Trying it with that", [{"name": r.name, "ok": r.ok, "why": r.why} for r in [*trial.generic, *trial.rules]])
        await self.kept()
        return True

    async def _searched_as_python(self, code: str, pays: dict[str, float]) -> Any:
        """A search of a tool in Python: of what was written of it, with the pieces it calls on above it and not changed."""
        from core.making.searching import Found

        cases = [Case.model_validate(c) for c in self.parts.cases]
        whole = self.whole(self.parts)
        if not whole.endswith(code):
            return Found()  # its pieces stand in the middle of it, and what is above them cannot be told from what is below
        lines = await lines_each_ran(whole, self.language(), cases)
        above = whole[: len(whole) - len(code)]
        found = await search_python(whole, cases, self.trial.rules, lines, above=above.count("\n"), pays=pays)
        found.mends = [dataclasses.replace(m, way=dataclasses.replace(m.way, start=m.way.start - len(above), end=m.way.end - len(above)))
                       for m in found.mends if m.way.start >= len(above)]
        return found

    async def _her_own_mend(self, failing: list[Result]) -> bool:
        """A mend she writes without her model, where one leaves less that fails; whether one was kept. Once for each state
        the program is in: what her own Python has to say of it does not change until it does."""
        assert self.design is not None
        code = self.parts.code()
        if self.design.kind == "site" or code in self._looked_at:
            return False
        self._looked_at.add(code)
        before, began, tried = self._wrong(self.trial), time.monotonic(), 0
        best: tuple[tuple[int, float], Parts, Trial, str, str] | None = None
        async for sort, says, parts in self._mends_of_her_own(failing):
            if tried == 0:
                self.work.step("Trying the mends her own Python writes, before her model is asked")
            tried += 1
            trial = await self.tried(parts, quiet=True)
            wrong = self._wrong(trial)
            # What was made to fit one case at a suspicious line may fit nothing else: it counts where it mends what broke, or more than one.
            fits = not sort.startswith(("template:", "ingredient:")) or len(failing) - len(trial.failing()) > 1 or any("Traceback" in r.why for r in failing)
            if wrong < (best[0] if best else before) and fits and not {r.name for r in trial.failing()} - {r.name for r in failing}:
                best = (wrong, parts, trial, says, sort)
            else:
                get_kept().tried(sort, kept=False)
            if time.monotonic() - began > OWN_S or (best is not None and not best[2].failing()):
                break
        if best is None:
            if tried:
                self.work.says(f"None of {tried} mends of her own left less wrong.")
            return False
        _wrong, self.parts, trial, says, sort = best
        get_kept().tried(sort, kept=True)
        await self._remembered(failing, says, sort)
        self.work.says(f"Kept a mend of her own, written without her model: {says}. {len(failing)} things failed before it and {len(trial.failing())} do now.")
        self.trial = trial
        self.work.tried("Trying it with that", [{"name": r.name, "ok": r.ok, "why": r.why} for r in [*trial.generic, *trial.rules]])
        await self.kept()
        return True

    async def _small_edits(self, failing: list[Result]) -> bool:
        """Small edits her model thinks could be it, each a guess that is tried by itself and in twos; whether one was kept.
        A guess is a few words of hers and a second of the machine's, where a function written again is a minute or two."""
        from core.self_modification.code_that_looks_wrong import applied
        from core.self_modification.edits_her_model_proposes import (
            ProposedEdits,
            edits_from_proposals,
        )
        from core.self_modification.repairing_by_behaviour import _pairs

        assert self.design is not None
        code = self.parts.code()
        if self.design.kind == "site" or code in self._guessed_at:
            return False
        self._guessed_at.add(code)
        first = sorted(failing, key=lambda r: -_how_much_is_wrong([r]))[:AT_ONCE]
        told = "\n".join(f"- {r.name}: {r.why}" for r in first)
        never = "\nWhile it was played, none of this of it ever ran:\n" + "\n".join(f"- {t}" for t in self.trial.never_ran) if self.trial.never_ran else ""
        numbered = "\n".join(f"{n:04d}: {line}" for n, line in enumerate(code.splitlines(), start=1))
        self.work.step("Asking what small edits could be it, and trying each")
        got = await self.ask(f"The program, its lines numbered:\n{numbered}\n\nTrying it, these fail:\n{told}{never}\n\nGive up to eight small edits, "
                             "each of which could by itself make one of these come right: the number of a line, the exact text there (old), and what "
                             "is to stand in its place (new). An empty old puts new before that line. They are guesses and each will be tried.",
                             ProposedEdits, EDITS_TOKENS, given=self._given())
        if not isinstance(got, ProposedEdits):
            return False
        guesses = [(s, s.edits) for s in edits_from_proposals(code, got)]
        before, best = self._wrong(self.trial), None
        began = time.monotonic()
        for suspicion, edits in [*guesses, *_pairs(guesses)]:
            if time.monotonic() - began > OWN_S or (best is not None and not best[2].failing()):
                break
            parts = self._with_code(applied(code, edits))
            if parts is None:
                continue
            trial = await self.tried(parts, quiet=True)
            wrong = self._wrong(trial)
            if wrong < (best[0] if best else before):
                best = (wrong, parts, trial, suspicion.why)
        if best is None:
            self.work.says(f"None of {len(guesses)} small edits left less wrong.")
            return False
        _wrong, self.parts, trial, why = best
        await self._remembered(failing, why, "small edit")
        self.work.says(f"Kept a small edit: {why} {len(failing)} things failed before it and {len(trial.failing())} do now.")
        self.trial = trial
        self.work.tried("Trying it with that", [{"name": r.name, "ok": r.ok, "why": r.why} for r in [*trial.generic, *trial.rules]])
        await self.kept()
        return True

    @staticmethod
    def _what_pays_first(mends: Iterable[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
        """Her own mends, the sorts that have been kept oftenest first, and otherwise in the order they come."""
        return sorted(mends, key=lambda mend: -get_kept().pays(mend[0]))

    async def _remembered(self, failing: list[Result], says: str, sort: str) -> None:
        """A mend that was kept, for the next program a failure like it is met in."""
        assert self.design is not None
        worst = max(failing, key=lambda r: _how_much_is_wrong([r]))
        get_kept().mended(kind=self.design.kind, failing=[r.name for r in failing], why=f"{worst.name}: {worst.why}", says=says, sort=sort)
        await asyncio.to_thread(get_kept().keep)

    def _with_code(self, code: str) -> Parts | None:
        """The program with its code as it is written here; None where that is not the same functions."""
        assert self.design is not None
        parts = self.parts.copy()
        if self.design.kind in {"tool", "site"}:
            parts.text = code
            return parts
        parts.functions = patched({}, code)[0]
        return parts if set(self.parts.functions) <= set(parts.functions) else None

    async def _mends_of_her_own(self, failing: list[Result]) -> AsyncIterator[tuple[str, str, Parts]]:
        assert self.design is not None
        if self.design.kind != "tool":
            code = assembled(self.parts.functions)
            for sort, says, changed in await asyncio.to_thread(lambda: self._what_pays_first(islice(own.of_its_shapes(code, ".js"), own.MOST))):
                parts = self._with_code(changed)
                if parts is not None:
                    yield sort, says, parts
            return
        language = self.language()
        if language.name != "python":
            return
        text = self.parts.text
        told = "\n".join(r.why for r in failing)
        shapes = await asyncio.to_thread(lambda: self._what_pays_first(islice(chain(own.missing_imports(text, told), own.of_its_shapes(text, ".py")), own.MOST)))
        for sort, says, changed in shapes:
            parts = self.parts.copy()
            parts.text = changed
            yield sort, says, parts
        cases = [Case.model_validate(c) for c in self.parts.cases]
        ok = [r.ok for r in self.trial.rules]
        if len(ok) != len(cases) or all(ok):
            return
        whole = self.whole(self.parts)
        lines = await lines_each_ran(whole, language, cases)
        # What stands above what was written of it, the pieces it calls on, is not what is wrong with it.
        above = whole[: whole.rfind(text[-200:])].count("\n") - text[: max(0, len(text) - 200)].count("\n") if whole != text else 0
        ranked = own.ran({str(i): ran for i, ran in enumerate(lines) if not ok[i]}, {str(i): ran for i, ran in enumerate(lines) if ok[i]}, after=above)
        error = next((m.group(1) for r in failing if (m := re.search(r"\b([A-Z][A-Za-z]+(?:Error|Exception))\b", r.why))), "")
        bundled = whole[: len(whole) - len(text)] if whole.endswith(text) else ""
        for sort, says, changed in await asyncio.to_thread(lambda: list(islice(own.of_its_lines(whole, ranked[:12], error), own.MOST))):
            if bundled and not changed.startswith(bundled):
                continue
            parts = self.parts.copy()
            parts.text = changed[len(bundled):] if bundled else changed
            if whole != text and not bundled:
                continue  # its pieces stand in the middle of it, and what is above them cannot be told from what is below
            yield sort, says, parts

    async def _the_best_of(self, mend: Parts, before: tuple[int, float]) -> tuple[Parts, Trial] | None:
        """A mend that gained nothing whole, taken apart: each function of it on its own over what gains, and the whole of it
        without each. A trial takes seconds where a mend takes a minute or two of her model's."""
        changed = [name for name, text in mend.functions.items() if self.parts.functions.get(name) != text]
        if not 2 <= len(changed) <= 7:
            return None
        self.work.says("Trying that mend a part at a time.")
        best: tuple[Parts, Trial] | None = None
        over = self.parts
        for name in changed:
            parts = over.copy()
            parts.functions[name] = mend.functions[name]
            trial = await self.tried(parts, quiet=True)
            if (wrong := self._wrong(trial)) < before:
                best, before, over = (parts, trial), wrong, parts
        for name in changed if len(changed) > 2 else ():
            parts = mend.copy()
            if name in self.parts.functions:
                parts.functions[name] = self.parts.functions[name]
            else:
                del parts.functions[name]
            trial = await self.tried(parts, quiet=True)
            if (wrong := self._wrong(trial)) < before:
                best, before = (parts, trial), wrong
        if best is not None:
            took = [name.removeprefix(BEFORE) for name in changed if best[0].functions.get(name) == mend.functions[name]]
            self.work.tried("Trying the part of it that gains: " + ", ".join(took)[:120],
                            [{"name": r.name, "ok": r.ok, "why": r.why} for r in [*best[1].generic, *best[1].rules]])
        return best

    async def finished(self) -> None:
        """What a person would see is not finished (one thing over another, words cut off or hard to read, a window that
        scrolls sideways), measured in the page and mended once, where that leaves less of it and breaks nothing."""
        assert self.design is not None
        if self.design.kind not in {"app", "site"} or not self.parts.code().strip():
            return
        before = await self._unfinished(self.parts)
        self.work.tried("How finished it looks", [{"name": found, "ok": False} for found in before] or [{"name": "nothing overlaps, is cut off or is hard to read", "ok": True}])
        if before:
            parts = await self._a_mend([Result(found, False, "seen in its window") for found in before], warm=0.2)
            if parts is not None:
                trial, after = await self.tried(parts, "Trying it with that mended"), await self._unfinished(parts)
                if len(after) < len(before) and self._wrong(trial) <= self._wrong(self.trial):
                    self.work.says(f"Kept: {len(before)} things looked unfinished and {len(after)} do now.")
                    self.parts, self.trial, before = parts, trial, after
        self.trial.generic.append(Result("nothing in it overlaps, is cut off or is hard to read", not before, "; ".join(before)[:400]))

    async def _screens(self) -> list[Any]:
        """What it looks like, as her eyes see it."""
        from core.making.her_view import as_she_saw_it, looked_at
        from core.making.pictures import as_data

        assert self.design is not None
        pictures = await asyncio.to_thread(as_data, self.folder, self.design)
        return await as_she_saw_it(await looked_at(self.design, self.parts.code(), self.parts.tests, self.browser, pictures=pictures), self.eyes)

    def her_own(self) -> set[str]:
        """The rules of it that were hers, from what she would change of it as she looked at it, and not anybody's asking."""
        return {str(c).strip() for c in self.last_view.get("would_change") or []} | set(self._hers)

    async def her_own_that_do_not_hold_let_go(self) -> None:
        """What she would have changed of it and does not hold once it has been mended, let go of, with its tests: what was asked
        and what any finished program is holds, and what she chose to add to it is had only where it holds too. LIVE 28 November
        2026 a vocabulary trainer was held to "Fix the rendering bug where 'Add a New Word' and 'Start Quiz' buttons appear
        twice", which her eyes had seen and was not so, and it could never be done."""
        design = self.design
        if design is None:
            return
        failing = {rule_of(r.name) for r in self.trial.failing()}
        gone = [rule for rule in design.rules if rule.says in self.her_own() and rule.says in failing]
        if not gone:
            return
        says = {rule.says for rule in gone}
        before, found = each_test_in(self.parts.tests)
        self.parts.tests = "\n\n".join([*([before.strip()] if before.strip() else []), *(f"{text};" for name, text in found if rule_of(name) not in says)]).strip()
        design.rules = [rule for rule in design.rules if rule.says not in says]
        self.has = [rule for rule in self.has if rule not in says]
        self._hers -= says
        self.work.says("What I would have changed of it and could not make hold, I let go: " + "; ".join(sorted(says))[:600] + ".")
        self.trial = await self.tried(what="Trying it without what I let go")
        await self.kept()

    async def looked_at_by_her(self) -> None:
        """Her own view of what she made: what she thinks of it as it looks, and what she would change of it, built in and tried
        like anything else; then she looks again, and what she does not think better made is taken back (core/making/her_view.py)."""
        from core.making.her_view import CHANGES, compared, her_view_of

        design = self.design
        if self.eyes is None or self.viewed or design is None or design.kind not in {"app", "site", "game"} or not self.parts.code().strip() or self.trial.breaks():
            return
        self.viewed = True
        code = hashlib.sha256(self.parts.code().encode("utf-8")).hexdigest()[:16]
        if self.last_view.get("code") == code:
            # LIVE 28 November 2026 she looked again, minutes after, at a dinner planner she had looked at, for its request was tried again.
            return
        self.work.step("Looking at it as somebody using it would")
        before = await self._screens()
        if not before:
            self.work.says("My eyes could not see it this time, so it is as it is.")
            return
        # She looks at it as somebody using it, not by its tests: what still fails is the mending's. LIVE 28 November 2026, given
        # them, each change she would make was a failing test said again, and she did not see that no book could be added.
        finish = next((r for r in self.trial.generic if r.name == "nothing in it overlaps, is cut off or is hard to read"), None)
        measured = "" if finish is None else "nothing overlaps, is cut off or is hard to read" if finish.ok else finish.why
        view = await her_view_of(before, asked=self.asked, design=design, ask=self.ask, write=self.write, how=self.thinking, before=self.last_view,
                                 measured=measured)
        if view is None:
            return
        changes = [c for c in view.would_change if c.change.strip() and c.change.strip() not in {r.says for r in design.rules}][:CHANGES]
        self.last_view = {"code": code, "thinks": view.thinks, "would_change": [c.change for c in changes], "came": "nothing to change" if not changes else ""}
        self.work.happened("her_view", thinks=view.thinks, works=view.works, would_change=[c.model_dump() for c in changes], looked_at=[s.caption for s in before],
                           saw=[{"caption": s.caption, "seen": s.seen[:900]} for s in before])
        if not changes:
            return
        was = (self.parts.copy(), self.trial, design.model_copy(deep=True), list(self.has))
        rules = [c.change.strip() for c in changes]
        design.rules += [Rule(says=rule) for rule in rules]
        self._hers |= set(rules)
        await self.tested(rules)
        await self.added_where_it_is_not(rules)
        # What she would change is had only where it is built: where anything fails that did not, her rules or what they
        # broke, it is taken back before she is asked whether it looks better, for a change not built looks as it did.
        if {r.name for r in self.trial.failing()} - {r.name for r in was[1].failing()}:
            self.parts, self.trial, self.design, self.has = was
            self.last_view["came"] = "what she would change could not be built without something failing, and was taken back"
            self.work.tried("As it was before", [{"name": r.name, "ok": r.ok, "why": r.why} for r in [*self.trial.generic, *self.trial.rules]])
            self.work.happened("her_view", thinks="", verdict="as it was")
            self.work.says("What I would change of it could not be built without something of it failing, so it is as it was.")
            await self.kept()
            return
        after = await self._screens()
        verdict = await compared(before, after, view, ask=self.ask, write=self.write, how=self.thinking) if after else None
        if verdict is not None and not verdict.better:
            self.parts, self.trial, self.design, self.has = was
            self.last_view["came"] = f"built, looked at again, and taken back: {verdict.thinks}"[:1200]
            self.work.tried("As it was before", [{"name": r.name, "ok": r.ok, "why": r.why} for r in [*self.trial.generic, *self.trial.rules]])
            self.work.happened("her_view", thinks=verdict.thinks, verdict="as it was")
            self.work.says("I thought it better as it was, so it is as it was.")
        else:
            self.last_view["came"] = f"built and kept: {verdict.thinks}"[:1200] if verdict is not None else "built and kept"
            self.work.happened("her_view", thinks=verdict.thinks if verdict is not None else "", verdict="kept")
        # What she looked at is the code as it stands now, kept or put back.
        self.last_view["code"] = hashlib.sha256(self.parts.code().encode("utf-8")).hexdigest()[:16]
        await self.kept()

    async def _unfinished(self, parts: Parts) -> list[str]:
        from core.rebuilding.her_build_writes import governed
        from core.rebuilding.how_finished_it_is import what_is_unfinished

        assert self.design is not None
        path = self.folder / "looked_at.html"
        from core.making.pictures import as_data

        html = site_of(self.design, parts.text) if self.design.kind == "site" else page_of(
            self.design, parts.code(), pictures=await asyncio.to_thread(as_data, self.folder, self.design))
        await asyncio.to_thread(governed, "write_text", path, html)
        try:
            seen = await what_is_unfinished(path, browser=self.browser)
            return [*seen.found, *(f"the page says: {error}" for error in seen.errors)][:12]
        finally:
            await asyncio.to_thread(governed, "delete_file", path)

    async def _a_mend(self, failing: list[Result], *, warm: float) -> Parts | None:
        design = self.design
        assert design is not None
        # What most stands on, first, and a few of them: a mend is small where what it mends is.
        first = sorted(failing, key=lambda r: -_how_much_is_wrong([r]))[:AT_ONCE]
        told = "\n".join(f"- {r.name}: {r.why}" for r in first)
        self.work.step(f"Mending it: {first[0].name}" + (f", and {len(failing) - 1} more" if len(failing) > 1 else ""))
        named = _tests_named(self.parts.tests, [r.name for r in first])
        tests = f"\nThe tests of them:\n```js\n{named}\n```\n" if named else ""
        if self.trial.never_ran and any(r in self.trial.generic for r in first):
            told += "\nWhile it was played, none of this of it ever ran:\n" + "\n".join(f"- {text}" for text in self.trial.never_ran)
        worked = await asyncio.to_thread(get_kept().like, design.kind, f"{first[0].name}: {first[0].why}")
        told += "\nWhat mended failures like these before:\n" + "\n".join(f"- {m}" for m in worked) if worked else ""
        if warm > 0.2 and rule_of(first[0].name) not in self._consulted_on:
            told += await self._what_her_sources_say_of(first[0])
            # What she does next is write the mend, and the work says so: the step of asking was over when they answered.
            self.work.step(f"Mending it, with what her sources said: {first[0].name}" + (f", and {len(failing) - 1} more" if len(failing) > 1 else ""))
        elif warm > 0.2:
            told += self._consulted_on[rule_of(first[0].name)]
        said = await self.write(f"It is written as:\n```\n{self.parts.code()}\n```\n{tests}\n"
                                f"Trying it, these fail:\n{told}\n\nSay in a sentence or two what is wrong and what you change, then "
                                f"{self._how_to_change()}.", MEND_TOKENS, warmth=min(0.8, warm), given=self._given())
        parts = self.parts.copy()
        self._said_of_it = _prose(said)
        if not self._written(said, "the mend", parts):
            self.work.trouble("The mend changed nothing.")
            for r in first:  # asked to mend it and finding nothing to change is a second witness for the program
                self._failed_by.setdefault(r.name, set()).add("nothing to change")
            return None
        return parts

    def _dead_controls_go(self, failing: list[Result]) -> list[str]:
        """A key that does nothing while it is played, and is for what the window does (to begin, go on, leave), is not a
        control of the program's own."""
        assert self.design is not None
        dead = [c for c in self.design.controls if any(not r.ok and r.why.startswith("pressing it changes nothing")
                                                       and r.name.endswith(f"({c.does})") for r in failing)]
        its_own = _the_windows_own([c.does for c in dead], [rule.says for rule in self.design.rules])
        gone = [c for c, own in zip(dead, its_own, strict=True) if own or c.input in {"Enter", "Escape", "p", "m", "r"}]
        if gone and len(gone) < len(self.design.controls):
            self._known = {name for name in self._known if not any(name.endswith(f"({c.does})") for c in gone)}
            self.design.controls = [c for c in self.design.controls if c not in gone]
            self.work.says("The window does this already, so it is no control of the program's: " + "; ".join(c.does for c in gone) + ".")
            return [c.does for c in gone]
        return []

    def _to_be_doubted(self, failing: list[Result]) -> list[Result]:
        """Of the tests that fail, those that may be what is wrong: what asks for what there is not, and what no version of
        the program has held to, of two or more tried. One that some version held to can be held to, and is the program's
        (CodeT, Chen and others 2023; core/rebuilding/what_versions_agree_on.py)."""
        twice = {rule_of(r.name) for r in self.trial.rules if r.ok and r.name.endswith(AGAIN)}
        return [r for r in failing if r.the_test or r.name in twice or (not self._held_by.get(r.name) and len(self._failed_by.get(r.name, ())) >= 2)]

    def tests_of_another_shape(self, before: Design | None) -> list[str]:
        """The rules whose tests set or read what the state has no longer, or has as another kind of thing, those tests taken
        out: they are of the design as it was, and are written again to the design as it is. LIVE 24 November 2026 an expense
        came to be for several people, every test of the app put one in that was for one, and the change was taken back for
        the tests it broke."""
        assert self.design is not None
        if before is None or self.design.kind not in {"app", "game"} or before.kind != self.design.kind:
            return []
        was, now = shape_of(before.state), shape_of(self.design.state)
        other = {name for name, kinds in was.items() if now.get(name) != kinds}
        if not other:
            return []
        first, tests = each_test_in(self.parts.tests)
        gone = [name for name, text in tests if any(re.search(rf"(?:\.|[{{,]\s*[\"']?){re.escape(n)}(?![\w$])", text) for n in other)]
        if gone:
            self.parts.tests = "\n\n".join([*([first.strip()] if first.strip() else []), *(f"{text};" for name, text in tests if name not in gone)])
            self.work.says(f"The state has another shape now ({', '.join(sorted(other)[:6])}), so {len(gone)} tests of it as it was are written again.")
            for name in gone:
                self._held_by.pop(name, None)
                self._failed_by.pop(name, None)
            self._twice -= {rule_of(name) for name in gone}
        rules = {rule.says for rule in self.design.rules}
        return [rule for rule in dict.fromkeys(rule_of(name) for name in gone) if rule in rules]

    async def _tests_read_again(self, failing: list[Result]) -> bool:
        """The tests that still fail, read again against the rule each is of: kept, corrected, or taken out with its reason."""
        assert self.design is not None
        self.work.step("Reading the tests that still fail again, for whether the test is what is wrong")
        told = "\n".join(f"- {r.name}: {r.why}" for r in failing[:8])
        if self.design.kind == "tool":
            cases = [c for c in self.parts.cases if any(r.name.startswith(str(c["rule"])) for r in failing)]
            got = await self.ask(f"These cases fail:\n{json.dumps(cases)}\nas:\n{told}\n\nThe program:\n```\n"
                                 f"{self.parts.text}\n```\n\nWork out by the rules alone what each should print, and give the cases that were "
                                 "wrong, corrected.", Cases, TEST_TOKENS, given=self._given())
            if not isinstance(got, Cases) or not got.cases:
                return False
            rules = {c.rule for c in got.cases}
            self.parts.cases = [c for c in self.parts.cases if c["rule"] not in rules] + [c.model_dump() for c in got.cases]
            return True
        before, tests = each_test_in(self.parts.tests)
        held = dict(tests)
        corrected = 0
        # Whether the test or the program is wrong is read with the program in front of her: what a name in its state means is in
        # how it is used. LIVE 28 November 2026 a test took the field an answer is typed in for the answer, and the program was blamed.
        # Given once, before every test read, so her model reads it once.
        given = f"{self._given()}\n\nThe program as it stands:\n```js\n{self.parts.code()}\n```"
        code = hashlib.sha256(self.parts.code().encode("utf-8")).hexdigest()
        for result in self._to_be_doubted(failing)[:AT_ONCE]:
            if result.name not in held:
                continue
            # A test read as right against the program as it is says the same read again: 33 readings in three makings of 28
            # November 2026, a minute each, kept nothing.
            read_as = hashlib.sha256(f"{held[result.name]}\n{code}".encode("utf-8")).hexdigest()
            if read_as in self._read_as_right:
                continue
            other = held.get(f"{result.name}{AGAIN}", "")
            holds = any(r.ok and r.name == f"{result.name}{AGAIN}" for r in self.trial.rules)
            beside = f"Another test of the same rule {'holds' if holds else 'fails too'}:\n```js\n{other}\n```\n" if other else ""
            said = await self.write(f"{_HOW_TESTS[self.design.kind]}\n\nA test of one rule:\n```js\n{held[result.name]}\n```\n"
                                    f"It fails: {result.why}\n{beside}\nBy the rule and what the state means, is the test wrong, or is it the program? In a sentence "
                                    "say which and why. Where it is the test, then write it corrected, in one block; where it is the program, write no "
                                    "code.", A_TEST, given=given)
            self.work.says(_prose(said))
            new = each_test_in(_code(said))[1] if "```" in said else []
            if not (new and new[0][1] != held[result.name]):
                self._read_as_right.add(read_as)
            if new and new[0][1] != held[result.name]:
                held[result.name] = _named(new[0][1], result.name)
                self.corrected.add(result.name)
                self._held_by.pop(result.name, None)
                self._failed_by.pop(result.name, None)
                corrected += 1
        if corrected:
            self.parts.tests = "\n\n".join([*([before.strip()] if before.strip() else []), *(f"{text};" for text in held.values())])
            self.work.wrote("tests", self.parts.tests)
        return bool(corrected)

    # ── 5. finished ──────────────────────────────────────────────────

    def whole(self, parts: Parts) -> str:
        """A tool as one file: the pieces it calls on, above what was written of it."""
        assert self.design is not None
        if self.language().name != "python":
            return parts.text
        used = [name for name in self.design.pieces if name in used_in(parts.text, "python")]
        if not used:
            return parts.text
        first = re.match(r"(?:#![^\n]*\n)?(?:from __future__[^\n]*\n)?", parts.text)
        at = first.end() if first else 0
        return f"{parts.text[:at]}{bundle(used, 'python')}\n\n\n{parts.text[at:]}"

    def page(self) -> str:
        assert self.design is not None
        if self.design.kind == "site":
            return site_of(self.design, self.parts.text)
        from core.making.pictures import as_data

        return page_of(self.design, self.parts.code(), pictures=as_data(self.folder, self.design))

    async def kept(self) -> Path:
        """The program, whole, in its folder, with what it was made from beside it."""
        from core.rebuilding.her_build_writes import governed

        assert self.design is not None
        design = self.design
        if design.kind == "tool":
            path = self.folder / self.language().file
            await asyncio.to_thread(governed, "write_text", path, self.whole(self.parts))
        else:
            path = self.folder / "index.html"
            # What runs is not put out by what does not: a change on its way that breaks it is gone on with, and is not what is opened.
            if self.trial.breaks() and await asyncio.to_thread(path.is_file):
                logger.info("%s does not run as it stands (%s); what ran before it is what is opened.", design.name, ", ".join(r.name for r in self.trial.breaks()))
            else:
                await asyncio.to_thread(lambda: governed("write_text", path, self.page()))
        record = {"title": design.name, "accent": design.look.accent, "asked": self.asked, "design": design.model_dump(),
                  "parts": {"functions": self.parts.functions, "text": self.parts.text, "tests": self.parts.tests,
                            "cases": self.parts.cases}, "built": self.has, **({"her_view": self.last_view} if self.last_view else {}),
                  "tried": [{"name": r.name, "ok": r.ok, "why": r.why} for r in [*self.trial.generic, *self.trial.rules]]}
        await asyncio.to_thread(governed, "write_text", self.folder / "program.json", json.dumps(record, indent=1, ensure_ascii=False))
        tried = [*self.trial.generic, *self.trial.rules]
        self.work.kept(sum(1 for r in tried if r.ok), len(tried))
        if self.when_kept is not None:
            self.when_kept(path)
        return path

    def made(self, path: Path, why_not: str = "") -> Made:
        assert self.design is not None
        return Made(self.design, self.parts, self.folder, path, self.trial, time.monotonic() - self.began, why_not, frozenset(self.corrected))


def shape_of(state: Any) -> dict[str, tuple[str, ...]]:
    """Every name a state has, at any depth, with the kinds of thing it holds."""
    found: dict[str, set[str]] = {}

    def kind(value: Any) -> str:
        if isinstance(value, list):
            held = sorted({kind(item) for item in value})
            return f"list of {held[0]}" if len(held) == 1 else "list"
        return "truth" if isinstance(value, bool) else "number" if isinstance(value, (int, float)) else "words" if isinstance(value, str) else "thing" if isinstance(value, dict) else "nothing"

    def of(value: Any) -> None:
        for name, held in value.items() if isinstance(value, dict) else ():
            found.setdefault(str(name), set()).add(kind(held))
            of(held)
        for item in value if isinstance(value, list) else ():
            of(item)

    of(state)
    return {name: tuple(sorted(kinds)) for name, kinds in found.items()}


def _named(test: str, name: str) -> str:
    """A test under this name."""
    return re.sub(r"test\s*\(\s*([\"'`])(?:\\.|(?!\1).)*\1", lambda _m: f"test({json.dumps(name)}", test, count=1, flags=re.S)


def _how_much_is_wrong(failing: list[Result]) -> float:
    """What fails, by how much stands on it: a program that breaks hides whatever is behind that, so putting it right counts
    for more than the rules it uncovers count against it."""
    def weight(result: Result) -> float:
        if result.name in CANNOT_DO_WITHOUT or result.name.startswith(("it runs without breaking", "it has its functions")):
            return 12.0
        if result.name.startswith("it can be ") or result.name in {"time passes in it", "it is not over before it can be played"}:
            return 4.0
        if result.why.startswith("pressing it changes nothing"):
            return 1.0
        # A button that does nothing is what a person finds first, and says least of what is wrong in its own words.
        if result.name == "every control does something":
            return 6.0
        # A test that got as far as what it expects is nearer than one the program broke in, and one that did more of what it
        # does before it failed is nearer than one that did less.
        return (3.0 if result.broke else 2.0) - 0.5 * result.got / (abs(result.got) + 4.0)

    # A rule that is tested twice is one rule, and is not had until both hold.
    most: dict[str, float] = {}
    for result in failing:
        most[rule_of(result.name)] = max(most.get(rule_of(result.name), 0.0), weight(result))
    return sum(most.values())


#: How far a program got, by what of it fails: one that cannot be read is tried for nothing else, so it fails at one thing.
_GOT_AS_FAR = {"it can be read": 3, "it has its functions": 3, "it can be tried": 3, "it comes to an end": 3, "it builds": 3, "it begins": 2,
               "it is shown": 1, "it is drawn": 1}


def _how_wrong(trial: Trial, known: set[str]) -> tuple[int, float]:
    """How wrong a program is, to be compared: how far it got first, and then what fails of it, with what it did not get as
    far as counted as failing. LIVE 24 November 2026 an edit that left a program unreadable was kept as the better of the
    two, for it failed at one thing where the other failed at eleven."""
    failing = trial.failing()
    stuck = max((_GOT_AS_FAR.get(r.name, 1 if r.name.startswith("it runs without breaking") else 0) for r in failing), default=0)
    # A test that holds whatever happens is left out of a trial once it holds, which is not a check it no longer gets as far as.
    here = {r.name for r in [*trial.generic, *trial.rules]} | set(trial.hollow)
    return stuck, _how_much_is_wrong(failing) + 3.0 * len(known - here)


#: What the window a program runs in does by itself.
_THE_WINDOW_DOES = ("pause the game and resume it", "start the game or restart it after it ends", "quit the game", "turn the sound on or off")
#: And what it has, as a rule of a program would say it.
_THE_WINDOW_HAS = {
    "game": ("There is a start screen with a title and how to play", "The game can be paused and gone on with",
             "When it ends a screen says how it went and offers to play again", "The score and the best score are shown on the screen",
             "The controls are listed for the player", "The sound can be turned off"),
    "app": ("Everything is saved and is still there when the app is opened again", "What was done can be undone",
            "The layout works on a narrow and a wide window"),
}


def _the_windows_own(does: list[str], rules: list[str], window_does: tuple[str, ...] = _THE_WINDOW_DOES) -> list[bool]:
    """Whether each of these is nearer what the window does by itself than any rule of the program is, by her encoder."""
    from core.cognition.evidence_relevance import vectors_if_ready

    vectors = vectors_if_ready([*does, *window_does, *rules]) if does else []
    if not vectors or len(vectors) != len(does) + len(window_does) + len(rules):
        return [False] * len(does)
    import numpy as np

    v = np.asarray(vectors, dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    mine, window, ruled = v[: len(does)], v[len(does): len(does) + len(window_does)], v[len(does) + len(window_does):]
    return [bool((window @ m).max() > ((ruled @ m).max() if len(rules) else 0.0)) for m in mine]


def _tests_named(tests: str, names: list[str]) -> str:
    """The tests of these names, as they are written."""
    return "\n".join(text for name, text in each_test_in(tests)[1] if name in names)


_HOW_TESTS = {
    "game": """A test is test("the rule, as it is written", t => { ... }). t begins at initial(). t.s is the state, to be read and set. t.tick(seconds) lets time pass, t.press(key), t.hold(key, seconds), t.point(x, y, seconds), t.click(x, y), t.send(event). t.score, t.over, t.won. t.draw() gives what is drawn, as [{what: "circle", args: [...]}] in the pen's own names; t.sees(text) is whether those words are drawn or counted above it. t.expect(so, "why not"). Set the state to where the rule bears, let the least happen, and expect what the rule says, and only that: where it gives no amount, any sensible amount holds.""",
    "app": """A test is test("the rule, as it is written", t => { ... }). t begins at initial(). t.s is the state, to be read and set. t.click(label), t.type(label, text), t.choose(label, value), t.enter(label) work the controls shown, by their labels. t.tick(seconds) lets time pass, t.press(key), t.send(event). t.sees(text) is whether it is shown, t.comesBefore(a, b) whether one thing is shown before another. t.expect(so, "why not"). What a rule says is seen is judged by what is shown, and not by how the state keeps it. Expect what the rule says, and only that: where it gives no words, form or amount, any sensible ones hold.""",
    "site": """A test is test("the rule, as it is written", () => { ... }) and runs in the page. see(text) fails where the text is not shown, one(selector), all(selector), press(text) presses the link or button that says it, expect(so, "why not").""",
}


async def _run(maker: _Maker, first: Callable[[], Awaitable[None]]) -> Made | None:
    """The steps in order, gone back to the design whenever somebody speaks to the work."""
    work = maker.work
    stage: Callable[[], Awaitable[None]] = first
    while True:
        try:
            await stage()
            if work.somebody_spoke():
                raise PutDown("somebody spoke to it")
            break
        except PutDown:
            if work.stopped:
                raise
            words = work.heard()
            maker.asked = "\n".join([maker.asked, *words])
            stage = maker_again(maker)
        except HerModelIsAwayError:
            # Not a step that broke off: she is to go on with it when her model is there.
            if maker.design is not None and maker.parts.code().strip():
                await maker.kept()
            raise
        except Exception as why:  # noqa: BLE001 - a step that broke off leaves the program as it stood before it
            if maker.design is None or not (maker.trial.passing() and maker.parts.code().strip()):
                raise
            logger.warning("A step of making %s broke off: %s", maker.design.name, why, exc_info=True)
            work.trouble(f"That step broke off ({type(why).__name__}: {str(why)[:200]}); the program is kept as it stood before it.")
            break
    if maker.design is None:
        return None
    await maker.painted()
    return maker.made(await maker.kept())


def maker_again(maker: _Maker) -> Callable[[], Awaitable[None]]:
    """From the design again, with what was said; what was written is written over only where the rules changed."""
    async def again() -> None:
        before = maker.design
        rules = [rule.says for rule in before.rules] if before is not None else []
        await maker.designed(before)
        if maker.design is None:
            return
        new = [rule.says for rule in maker.design.rules if rule.says not in rules]
        new += [rule for rule in maker.tests_of_another_shape(before) if rule not in new]
        same_kind = before is not None and before.kind == maker.design.kind
        if before is None or not same_kind or not (maker.parts.functions or maker.parts.text):
            maker.parts, maker.has = Parts(), []
            await maker.built()
            return
        for at in range(0, max(1, len(new)), A_LAYER):
            layer = new[at:at + A_LAYER]
            if layer:
                await maker.tested(layer)
            await maker.added_where_it_is_not(layer, maker.asked.splitlines()[-1] if at == 0 else "")
            await maker.kept()
        await maker.built()

    return again


async def make(asked: str, *, ask: Asks, write: Writes, work: Work, where: Path, browser: Any, told: str = "",
               when_kept: Callable[[Path], None] | None = None, paints: Any = None, eyes: Any = None, thinking: dict[str, Any] | None = None) -> Made | None:
    """Make what was asked for, in a folder of its own under ``where``; None where no design of it could be had. ``told`` is
    what she said to whoever asked as she began, which the design holds to; ``when_kept`` is told where it stands each time
    it is kept, so that what cuts it short leaves something to go on with. ``eyes`` are hers, to look at what she made with
    (core/making/her_view.py); without them she does not."""
    maker = _Maker(asked, ask=ask, write=write, work=work, folder=where, browser=browser)
    maker.told, maker.when_kept, maker.paints, maker.eyes, maker.thinking = told, when_kept, paints, eyes, thinking or {}

    async def first() -> None:
        if await maker.designed() is None:
            return
        maker.folder = await asyncio.to_thread(_a_folder, where, maker.design.name if maker.design else "program")
        await maker.built()

    return await _run(maker, first)


async def change(folder: Path, asked: str, *, ask: Asks, write: Writes, work: Work, browser: Any, paints: Any = None, eyes: Any = None,
                 thinking: dict[str, Any] | None = None) -> Made | None:
    """Change a program made before to what is asked of it now, its tests kept and held to."""
    design, parts, was_asked = await asyncio.to_thread(parts_of, folder)
    maker = _Maker(f"{was_asked}\n{asked}", ask=ask, write=write, work=work, folder=folder, browser=browser)
    maker.design, maker.parts, maker.paints, maker.eyes, maker.thinking = design, parts, paints, eyes, thinking or {}
    maker.has = await asyncio.to_thread(_built_of, folder, design)
    maker.last_view = await asyncio.to_thread(_her_view_of, folder)
    return await _run(maker, maker_again(maker))


async def mend(folder: Path, *, ask: Asks, write: Writes, work: Work, browser: Any, paints: Any = None, eyes: Any = None,
               thinking: dict[str, Any] | None = None) -> Made | None:
    """Go on with a program made before where something of it still fails."""
    design, parts, was_asked = await asyncio.to_thread(parts_of, folder)
    maker = _Maker(was_asked, ask=ask, write=write, work=work, folder=folder, browser=browser)
    maker.design, maker.parts, maker.paints, maker.eyes, maker.thinking = with_an_icon(design), parts, paints, eyes, thinking or {}
    maker.has = await asyncio.to_thread(_built_of, folder, design)
    maker.last_view = await asyncio.to_thread(_her_view_of, folder)
    return await _run(maker, maker.gone_on_with)


def parts_of(folder: Path) -> tuple[Design, Parts, str]:
    """A program made before: its design, its parts, and what was asked for."""
    kept = json.loads((Path(folder) / "program.json").read_text("utf-8"))
    return Design.model_validate(kept["design"]), Parts(**kept["parts"]), str(kept.get("asked") or "")


def _her_view_of(folder: Path) -> dict[str, Any]:
    """What she thought of a program when she last looked at it, as it is kept with it; {} where she has not looked."""
    try:
        view = json.loads((Path(folder) / "program.json").read_text("utf-8")).get("her_view")
    # not a failure: a program kept without her view is one she has not looked at.
    except (OSError, ValueError):
        return {}
    return dict(view) if isinstance(view, dict) else {}


def _built_of(folder: Path, design: Design) -> list[str]:
    """Which of its rules have been built: all of them, where that was not kept."""
    built = json.loads((Path(folder) / "program.json").read_text("utf-8")).get("built")
    return [str(rule) for rule in built] if isinstance(built, list) else [rule.says for rule in design.rules]


def _a_folder(where: Path, name: str) -> Path:
    from core.rebuilding.her_build_writes import governed

    base = Path(where) / (re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "program")
    folder, n = base, 1
    while folder.exists() and any(folder.iterdir()):
        n += 1
        folder = base.with_name(f"{base.name}-{n}")
    governed("ensure_directory", folder)
    return folder
