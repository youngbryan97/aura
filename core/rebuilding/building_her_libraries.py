"""The libraries a program is made of: planned from what it does and what that rests on, each built from its tests.

Before any feature is written, her model is shown the program's features, what was read of what they rest on
(core/rebuilding/what_is_known_of_it.py) and the libraries she already has (core/rebuilding/her_libraries.py), and
says which libraries the program needs: each one's functions and the calls that test them, written from what was
read (a flood fill's article says which cells it reaches). A test that holds where every function gives nothing is
left out. Then each library, in turn: one of hers that passes its tests is used as it is; else her model writes it,
and is shown the tests that failed and writes it again, up to ``TRIES`` times; the version that passed most is kept,
with the functions whose tests did not all hold named. Commit0 (ICLR 2025) found the tests' feedback was what
helped agents most at writing a library to a specification; CodeT (2023) wrote the tests first.

The libraries go into the program ahead of its features, and each feature's part is told what they offer and that
they are tested. A library kept is kept for every program after (her store), whatever program it was made for.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from core.rebuilding.her_libraries import (
    KeptLibrary,
    LibraryPlan,
    found_for,
    kept,
    run_tests,
    the_tests_that_test,
)

logger = logging.getLogger("Rebuilding.Libraries")

__all__ = ["BuiltLibrary", "build_libraries", "libraries_it_needs", "offered"]

#: Tries at one library before the best version written is what is kept.
TRIES = 3
#: The most libraries one program is planned with.
MOST_LIBRARIES = 6
#: The longest library her model is asked for, in tokens.
LIBRARY_TOKENS = 3072


class _Plan(BaseModel):
    libraries: list[LibraryPlan] = Field(default_factory=list)


class _Code(BaseModel):
    code: str = Field(default="", max_length=24_000, description="plain JavaScript that sets the object on globalThis")


@dataclass
class BuiltLibrary:
    plan: LibraryPlan
    code: str = ""
    held: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    tries: int = 0
    from_her_store: str = ""

    @property
    def usable(self) -> list[str]:
        """Its functions whose every test held: what a feature may count on."""
        broken = {call.split("(", 1)[0].split(".")[-1] for call in self.failed}
        return [f.name for f in self.plan.functions if f.name not in broken]


def _her_store_said(libraries: Sequence[KeptLibrary]) -> str:
    return "".join(f"- {lib.plan.name} (made for {lib.made_for}): {lib.plan.purpose}\n" for lib in libraries[:20])


async def libraries_it_needs(genome: Any, known: Any, ask: Any, *, browser: Any = None,
                             store: Sequence[KeptLibrary] | None = None) -> list[LibraryPlan]:
    """The libraries her model says the program is made of, each with the tests that test something."""
    store = kept() if store is None else list(store)
    features = "".join(f"- {f.name}: {f.how} -> {f.shows}\n" for f in genome.features)
    read = "".join(f"[{title}]\n" + " ".join(r.text for r in reads[:6]) + "\n"
                   for title, reads in (getattr(known, "rests_on", None) or {}).items())
    prompt = (f"{genome.name}: {genome.what_it_is}\nIts features:\n{features}\n"
              + (f"What was read of what they rest on:\n{read}\n" if read else "")
              + (f"Libraries already written and tested, which a library here may be one of:\n{_her_store_said(store)}\n"
                 if store else "")
              + f"Say which libraries of JavaScript functions it is made of, at most {MOST_LIBRARIES}: what the features "
              "share and rest on (its data, its algorithms, its file formats), not the features themselves. For each: "
              "the object it sets on globalThis, its functions, and calls of them with the JSON value each must give, "
              "worked out from what was read, using plain arrays, objects, numbers and strings.")
    said = await ask(prompt, _Plan, 3072)
    plans: list[LibraryPlan] = []
    for plan in getattr(said, "libraries", []) or []:
        if not plan.sound() or plan.name in {p.name for p in plans}:
            continue
        plan.tests = await the_tests_that_test(plan, browser=browser)
        if plan.tests:
            plans.append(plan)
    return plans[:MOST_LIBRARIES]


def _write_it_prompt(plan: LibraryPlan, known: Any, failed: str, before: str) -> str:
    read = known.of(" ".join([plan.purpose, *plan.rests_on, *(f.name for f in plan.functions)])) if known is not None else []
    tests = "".join(f"- {t.call} must give {t.expect}\n" for t in plan.tests)
    text = (f"Write the JavaScript library {plan.name}: {plan.purpose}\n"
            f"It sets globalThis.{plan.name} to an object with these functions:\n{plan.said()}\n"
            f"Its tests:\n{tests}"
            + ("What was read of what it rests on:\n" + "".join(f"- {r.text} ({r.where})\n" for r in read) if read else "")
            + "Plain JavaScript that runs in a browser page, no imports, no network, no page elements.")
    if failed:
        text += f"\n\nYour last version:\n```js\n{before[:8000]}\n```\nIts tests showed:\n{failed}\nWrite it again so they hold."
    return text


def _failures(runs: Sequence[Any]) -> str:
    return "".join(f"- {r.test.call}: wanted {r.test.expect}, got {r.got}\n" for r in runs if not r.held)


async def build_library(plan: LibraryPlan, ask: Any, *, known: Any = None, browser: Any = None,
                        store: Sequence[KeptLibrary] = ()) -> BuiltLibrary:
    """One library: one of hers that passes its tests, else written, shown its failures and written again."""
    for library in found_for(plan, list(store)):
        runs = await run_tests(library.code, plan.tests, browser=browser)
        if runs and all(r.held for r in runs):
            return BuiltLibrary(plan, library.code, [r.test.call for r in runs], [], 0, library.made_for)
    best = BuiltLibrary(plan)
    failed, before = "", ""
    for best.tries in range(1, TRIES + 1):
        written = await ask(_write_it_prompt(plan, known, failed, before), _Code, LIBRARY_TOKENS)
        code = str(getattr(written, "code", "") or "")
        if not code.strip():
            failed = "- nothing was written\n"
            continue
        runs = await run_tests(code, plan.tests, browser=browser)
        held = [r.test.call for r in runs if r.held]
        if len(held) > len(best.held) or not best.code:
            best.code, best.held, best.failed = code, held, [r.test.call for r in runs if not r.held]
        if len(held) == len(runs):
            break
        failed, before = _failures(runs), code
    logger.info("rebuilding: library %s, %d of %d tests hold after %d tries", plan.name, len(best.held), len(plan.tests), best.tries)
    return best


async def build_libraries(genome: Any, known: Any, ask: Any, *, browser: Any = None, tell: Any = None,
                          keep_them: bool = True) -> list[BuiltLibrary]:
    """Every library the program is made of, planned and built in turn; those with a test that held kept in her store."""
    from core.rebuilding.her_libraries import keep

    store = kept()
    plans = await libraries_it_needs(genome, known, ask, browser=browser, store=store)
    built: list[BuiltLibrary] = []
    for plan in plans:
        library = await build_library(plan, ask, known=known, browser=browser, store=store)
        built.append(library)
        if keep_them and library.held and not library.from_her_store:
            await keep(KeptLibrary(plan, library.code, genome.name, held=library.held))
    if tell is not None and built:
        said = "; ".join(f"{b.plan.name} ({len(b.held)} of {len(b.plan.tests)} tests"
                         + (f", mine from {b.from_her_store}" if b.from_her_store else "") + ")" for b in built)
        await tell(f"Before its features, the libraries it is made of: {said}.")
    return built


def offered(built: Sequence[BuiltLibrary]) -> str:
    """What the libraries in the program offer a feature's part, tested functions only."""
    lines = []
    for library in built:
        for function in library.plan.functions:
            if function.name in library.usable and library.code:
                lines.append(f"{library.plan.name}.{function.name}({', '.join(function.params)}): {function.does}")
    return ("Libraries already in the program, each function tested:\n" + "\n".join(f"- {line}" for line in lines) + "\n"
            if lines else "")
