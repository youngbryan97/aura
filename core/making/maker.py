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
import json
import logging
import re
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.making import design as how
from core.making.code_parts import BEFORE, assembled, each_test_in, patched, python_patched
from core.making.design import Asked, Design, Rule, in_proportion, left_out, with_every_ask
from core.making.pieces import bundle, calls, catalogue, near, pieces, used_in, with_needs
from core.making.tools import Case, Cases, Language, languages, try_tool
from core.making.trying import Result, Trial, page_of, site_of, try_page
from core.understanding.carriers.finding import quoted_in
from core.understanding.workbench import PutDown, Work

logger = logging.getLogger("Making")

__all__ = ["Made", "Parts", "change", "make", "parts_of"]

#: Ask for typed data: (prompt, schema, most tokens). Write: (prompt, most tokens, warmth).
Asks = Callable[[str, Any, int], Awaitable[Any]]
Writes = Callable[..., Awaitable[str]]

#: The most tokens of a design, of tests, and of what is written of a program at once.
DESIGN_TOKENS, TEST_TOKENS, CODE_TOKENS, MEND_TOKENS, A_TEST = 3200, 3600, 4600, 3200, 700
#: How many rules are built at once: a program is built a layer at a time, each tested before the next.
A_LAYER = 8
#: How many of what fails are mended at once.
AT_ONCE = 5
#: How many mends in a row may gain nothing before the last of what fails is left.
GAINED_NOTHING = 2


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

    def working(self) -> bool:
        """Whether it runs, and something it was asked for is so."""
        broken = [r for r in self.trial.generic if not r.ok and r.name in _CANNOT_DO_WITHOUT]
        return not broken and (not self.trial.rules or any(r.ok for r in self.trial.rules))


_CANNOT_DO_WITHOUT = frozenset({"it can be read", "it has its functions", "it can be tried", "it comes to an end", "it begins", "it is drawn",
                                "it is shown", "it builds"})


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


class _Maker:
    def __init__(self, asked: str, *, ask: Asks, write: Writes, work: Work, folder: Path, browser: Any) -> None:
        self.asked, self.ask, self.write, self.work, self.folder, self.browser = asked, ask, write, work, folder, browser
        self.design: Design | None = None
        self.parts = Parts()
        self.trial = Trial()
        self.told = ""
        #: The rules built so far, a layer at a time, and what is told where the program stands each time it is kept.
        self.has: list[str] = []
        self.when_kept: Callable[[Path], None] | None = None
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

    async def built(self) -> None:
        """The program, a layer of its rules at a time: tests of the layer, its code, and what fails mended before the next."""
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
            else:
                await self.added(rules)
            self.has += rules
            await self.mended()
            await self.kept()
        await self.finished()

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
            self.work.says(_prose(said))
            self._written(said, what)
            return
        first = "" if len(rules) == len(design.rules) else "Only these of its rules are built now:\n" + "\n".join(f"- {r}" for r in rules) + "\n\n"
        functions = "initial, update, score, over and hud" if design.kind == "game" else "initial and update"
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
        said = await self.write(f"It is written as:\n```\n{self.parts.code()}\n```\n{told}\n\n"
                                f"Say in a sentence or two what you change, then {self._how_to_change()}.", CODE_TOKENS, given=self._given())
        self._written(said, "what changed" if self.parts.functions else "it")

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
        if not quiet:
            self.work.tried(what, [{"name": r.name, "ok": r.ok, "why": r.why} for r in [*trial.generic, *trial.rules]])
        return trial

    async def mended(self) -> None:
        """What fails, handed back with the code, as long as that gains; then a doubt of the tests that still fail."""
        assert self.design is not None
        self.trial = await self.tried()
        nothing_gained = doubted = 0
        put_right = False
        while self.trial.failing():
            failing = self.trial.failing()
            # A test that asks for what there is not is a test to be put right, and no fault of the program's.
            amiss = [r for r in failing if r.the_test]
            if amiss and not put_right:
                put_right = True
                if await self._tests_read_again(amiss):
                    self.trial = await self.tried(what="Trying it, with the tests put right")
                    continue
            if nothing_gained >= (GAINED_NOTHING if any(not r.ok for r in self.trial.generic) else 1):
                if doubted or self.design.kind == "site" or not [r for r in self.trial.rules if not r.ok]:
                    break
                doubted = 1
                if not await self._tests_read_again([r for r in self.trial.rules if not r.ok]):
                    break
                nothing_gained = 0
                self.trial = await self.tried(what="Trying it, with the tests read again")
                continue
            parts = await self._a_mend(failing, warm=0.2 + 0.3 * nothing_gained)
            if parts is None:
                nothing_gained += 1
                continue
            trial = await self.tried(parts, "Trying the mend")
            before, after = _how_much_is_wrong(failing), _how_much_is_wrong(trial.failing())
            if after >= before and (better := await self._the_best_of(parts, before)) is not None:
                parts, trial = better
                after = _how_much_is_wrong(trial.failing())
            if after < before:
                self.work.says(f"Kept: {len(failing)} things failed before it and {len(trial.failing())} do now"
                               + (", and what was underneath them is put right." if len(trial.failing()) >= len(failing) else "."))
                self.parts, self.trial, nothing_gained = parts, trial, 0
                await self.kept()
            else:
                nothing_gained += 1
                self.work.says(f"Not kept: {len(failing)} things failed before it and {len(trial.failing())} after.")

    async def _the_best_of(self, mend: Parts, before: float) -> tuple[Parts, Trial] | None:
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
            if (wrong := _how_much_is_wrong(trial.failing())) < before:
                best, before, over = (parts, trial), wrong, parts
        for name in changed if len(changed) > 2 else ():
            parts = mend.copy()
            if name in self.parts.functions:
                parts.functions[name] = self.parts.functions[name]
            else:
                del parts.functions[name]
            trial = await self.tried(parts, quiet=True)
            if (wrong := _how_much_is_wrong(trial.failing())) < before:
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
                if len(after) < len(before) and _how_much_is_wrong(trial.failing()) <= _how_much_is_wrong(self.trial.failing()):
                    self.work.says(f"Kept: {len(before)} things looked unfinished and {len(after)} do now.")
                    self.parts, self.trial, before = parts, trial, after
        self.trial.generic.append(Result("nothing in it overlaps, is cut off or is hard to read", not before, "; ".join(before)[:400]))

    async def _unfinished(self, parts: Parts) -> list[str]:
        from core.rebuilding.her_build_writes import governed
        from core.rebuilding.how_finished_it_is import what_is_unfinished

        assert self.design is not None
        path = self.folder / "looked_at.html"
        html = site_of(self.design, parts.text) if self.design.kind == "site" else page_of(self.design, parts.code())
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
        said = await self.write(f"It is written as:\n```\n{self.parts.code()}\n```\n{tests}\n"
                                f"Trying it, these fail:\n{told}\n\nSay in a sentence or two what is wrong and what you change, then "
                                f"{self._how_to_change()}.", MEND_TOKENS, warmth=min(0.8, warm), given=self._given())
        parts = self.parts.copy()
        if not self._written(said, "the mend", parts):
            self.work.trouble("The mend changed nothing.")
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
            self.design.controls = [c for c in self.design.controls if c not in gone]
            self.work.says("The window does this already, so it is no control of the program's: " + "; ".join(c.does for c in gone) + ".")
            return [c.does for c in gone]
        return []

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
        for result in failing[:AT_ONCE]:
            if result.name not in held:
                continue
            said = await self.write(f"{_HOW_TESTS[self.design.kind]}\n\nA test of one rule:\n```js\n{held[result.name]}\n```\n"
                                    f"It fails: {result.why}\n\nBy the rule and what the state means, is the test wrong, or is it the program? In a sentence "
                                    "say which and why. Where it is the test, then write it corrected, in one block; where it is the program, write no "
                                    "code.", A_TEST, given=self._given())
            self.work.says(_prose(said))
            new = each_test_in(_code(said))[1] if "```" in said else []
            if new and new[0][1] != held[result.name]:
                held[result.name] = re.sub(r"test\s*\(\s*([\"'`])(?:\\.|(?!\1).)*\1", lambda _m, n=result.name: f"test({json.dumps(n)}", new[0][1], count=1, flags=re.S)
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
        return page_of(self.design, self.parts.code())

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
            await asyncio.to_thread(governed, "write_text", path, self.page())
        record = {"title": design.name, "accent": design.look.accent, "asked": self.asked, "design": design.model_dump(),
                  "parts": {"functions": self.parts.functions, "text": self.parts.text, "tests": self.parts.tests,
                            "cases": self.parts.cases}, "built": self.has,
                  "tried": [{"name": r.name, "ok": r.ok, "why": r.why} for r in [*self.trial.generic, *self.trial.rules]]}
        await asyncio.to_thread(governed, "write_text", self.folder / "program.json", json.dumps(record, indent=1, ensure_ascii=False))
        if self.when_kept is not None:
            self.when_kept(path)
        return path

    def made(self, path: Path, why_not: str = "") -> Made:
        assert self.design is not None
        return Made(self.design, self.parts, self.folder, path, self.trial, time.monotonic() - self.began, why_not)


def _how_much_is_wrong(failing: list[Result]) -> float:
    """What fails, by how much stands on it: a program that breaks hides whatever is behind that, so putting it right counts
    for more than the rules it uncovers count against it."""
    def weight(result: Result) -> float:
        if result.name in _CANNOT_DO_WITHOUT or result.name.startswith(("it runs without breaking", "it has its functions")):
            return 12.0
        if result.name.startswith("it can be ") or result.name in {"time passes in it", "it is not over before it can be played"}:
            return 4.0
        if result.why.startswith("pressing it changes nothing"):
            return 1.0
        # A test that got as far as what it expects is nearer than one the program broke in.
        return 3.0 if result.broke else 2.0

    return sum(weight(result) for result in failing)


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
        except Exception as why:  # noqa: BLE001 - a step that broke off leaves the program as it stood before it
            if maker.design is None or not (maker.trial.passing() and maker.parts.code().strip()):
                raise
            logger.warning("A step of making %s broke off: %s", maker.design.name, why, exc_info=True)
            work.trouble(f"That step broke off ({type(why).__name__}: {str(why)[:200]}); the program is kept as it stood before it.")
            break
    if maker.design is None:
        return None
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
        same_kind = before is not None and before.kind == maker.design.kind
        if before is None or not same_kind or not (maker.parts.functions or maker.parts.text):
            maker.parts, maker.has = Parts(), []
            await maker.built()
            return
        for at in range(0, max(1, len(new)), A_LAYER):
            layer = new[at:at + A_LAYER]
            if layer:
                await maker.tested(layer)
            await maker.added(layer, maker.asked.splitlines()[-1] if at == 0 else "")
            maker.has += layer
            await maker.mended()
            await maker.kept()
        await maker.built()

    return again


async def make(asked: str, *, ask: Asks, write: Writes, work: Work, where: Path, browser: Any, told: str = "",
               when_kept: Callable[[Path], None] | None = None) -> Made | None:
    """Make what was asked for, in a folder of its own under ``where``; None where no design of it could be had. ``told`` is
    what she said to whoever asked as she began, which the design holds to; ``when_kept`` is told where it stands each time
    it is kept, so that what cuts it short leaves something to go on with."""
    maker = _Maker(asked, ask=ask, write=write, work=work, folder=where, browser=browser)
    maker.told, maker.when_kept = told, when_kept

    async def first() -> None:
        if await maker.designed() is None:
            return
        maker.folder = await asyncio.to_thread(_a_folder, where, maker.design.name if maker.design else "program")
        await maker.built()

    return await _run(maker, first)


async def change(folder: Path, asked: str, *, ask: Asks, write: Writes, work: Work, browser: Any) -> Made | None:
    """Change a program made before to what is asked of it now, its tests kept and held to."""
    design, parts, was_asked = await asyncio.to_thread(parts_of, folder)
    maker = _Maker(f"{was_asked}\n{asked}", ask=ask, write=write, work=work, folder=folder, browser=browser)
    maker.design, maker.parts = design, parts
    maker.has = await asyncio.to_thread(_built_of, folder, design)
    return await _run(maker, maker_again(maker))


async def mend(folder: Path, *, ask: Asks, write: Writes, work: Work, browser: Any) -> Made | None:
    """Go on with a program made before where something of it still fails."""
    design, parts, was_asked = await asyncio.to_thread(parts_of, folder)
    maker = _Maker(was_asked, ask=ask, write=write, work=work, folder=folder, browser=browser)
    maker.design, maker.parts = design, parts
    maker.has = await asyncio.to_thread(_built_of, folder, design)
    return await _run(maker, maker.gone_on_with)


def parts_of(folder: Path) -> tuple[Design, Parts, str]:
    """A program made before: its design, its parts, and what was asked for."""
    kept = json.loads((Path(folder) / "program.json").read_text("utf-8"))
    return Design.model_validate(kept["design"]), Parts(**kept["parts"]), str(kept.get("asked") or "")


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
