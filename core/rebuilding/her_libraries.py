"""Her libraries: code she wrote, each with the calls that test it, kept for every program after the one it was made for.

A person who has built a paint program has a flood fill and a colour model, tested, and the next program that
needs them starts from them. The rebuilding engine's parts are written for one program and one feature each; a
library is written for what a feature rests on (an algorithm, a format, a structure; core/rebuilding/what_is_known_of_it.py)
and is used by any feature that needs it. Each is plain JavaScript that sets one object on ``globalThis`` (``Raster``,
``Csv``), its functions named in its plan, and each is known by its tests: calls of its functions and the values they
must give. A test that already holds where every function gives nothing tests nothing and is left out.

Code is run only in a browser page with the network refused, as the engine runs every part it builds
(core/rebuilding/checks_a_person_makes.py); it is model-written and never run on this machine otherwise. Kept
libraries are found again for a new plan by the words of their purpose and their functions, and taken only where the
new plan's own tests hold on them (Voyager's skill library; LILO's documented library, 2024).
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

__all__ = ["FunctionPlan", "KeptLibrary", "LibraryPlan", "LibraryTest", "RunOfATest", "found_for", "keep", "kept",
           "run_tests", "stub_of", "the_tests_that_test"]

#: How long one test call may run, in seconds.
TEST_S = 5.0
_NAME = re.compile(r"^[A-Z][A-Za-z0-9]{1,40}$")


class FunctionPlan(BaseModel):
    name: str = Field(max_length=60)
    params: list[str] = Field(default_factory=list, description="parameter names, in order")
    does: str = Field(default="", max_length=400, description="what it gives back, and from what")


class LibraryTest(BaseModel):
    call: str = Field(max_length=600, description='a call of one of its functions, e.g. Raster.floodFill([[0,1],[1,1]], 0, 0, 2)')
    expect: str = Field(max_length=2000, description="the value it must give, as JSON")
    why: str = Field(default="", max_length=300, description="what that follows from, in what was read")


class LibraryPlan(BaseModel):
    name: str = Field(max_length=41, description="the object it sets on globalThis, in CapitalCase")
    purpose: str = Field(default="", max_length=400)
    rests_on: list[str] = Field(default_factory=list, description="titles of what was read that it implements")
    functions: list[FunctionPlan] = Field(default_factory=list)
    tests: list[LibraryTest] = Field(default_factory=list)
    uses: list[str] = Field(default_factory=list, description="the other libraries of this program whose functions it calls")

    def sound(self) -> bool:
        """A name an object can have, functions with names, and tests that call them."""
        names = {f.name for f in self.functions}
        return bool(_NAME.match(self.name) and names and all(re.match(r"^[a-zA-Z_]\w*$", n) for n in names))

    def said(self) -> str:
        """The library's interface, as a part that uses it is told it."""
        return "\n".join(f"{self.name}.{f.name}({', '.join(f.params)}): {f.does}" for f in self.functions)


@dataclass
class RunOfATest:
    test: LibraryTest
    held: bool
    got: str = ""


@dataclass
class KeptLibrary:
    plan: LibraryPlan
    code: str
    made_for: str
    folder: Path | None = None
    held: list[str] = field(default_factory=list)


#: Runs one test in the page: its call, the value compared to the one expected, numbers to a part in a billion.
_ONE_TEST = r"""([call, expect]) => {
  const same = (a, b) => {
    if (typeof a === "number" && typeof b === "number") return Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a), Math.abs(b));
    if (ArrayBuffer.isView(a)) a = Array.from(a);
    if (Array.isArray(a) && Array.isArray(b)) return a.length === b.length && a.every((x, i) => same(x, b[i]));
    if (a && b && typeof a === "object" && typeof b === "object") {
      const keys = Object.keys(b); return keys.length === Object.keys(a).length && keys.every(k => same(a[k], b[k])); }
    return a === b; };
  try { const got = (0, eval)(call); const want = JSON.parse(expect);
        let shown; try { shown = JSON.stringify(ArrayBuffer.isView(got) ? Array.from(got) : got); } catch (e) { shown = String(got); }
        return {held: same(got, want), got: String(shown).slice(0, 300)}; }
  catch (e) { return {held: false, got: "threw " + String(e).slice(0, 300)}; }
}"""


def stub_of(plan: LibraryPlan) -> str:
    """The library with every function giving nothing: what a test that tests something fails on."""
    return f"globalThis.{plan.name} = {{{', '.join(f'{f.name}: () => undefined' for f in plan.functions)}}};"


async def run_tests(code: str, tests: list[LibraryTest], *, browser: Any = None) -> list[RunOfATest]:
    """Each test's call done in a page holding ``code``, the network refused; a call that runs past ``TEST_S`` fails."""
    if not tests:
        return []
    owned = None
    if browser is None:
        from playwright.async_api import async_playwright

        owned = await async_playwright().start()
        browser = await owned.chromium.launch(headless=True)
    try:
        return await _in_a_page(browser, code, tests)
    finally:
        if owned is not None:
            await browser.close()
            await owned.stop()


async def _in_a_page(browser: Any, code: str, tests: list[LibraryTest]) -> list[RunOfATest]:
    runs: list[RunOfATest] = []
    context = await browser.new_context()
    try:
        await context.route("**/*", lambda route: route.abort())
        page = await context.new_page()
        await page.set_content("<!doctype html><html><body></body></html>")
        try:
            await asyncio.wait_for(page.add_script_tag(content=code), timeout=TEST_S)
        except Exception as why:  # noqa: BLE001 - code that will not load fails every test, with why
            return [RunOfATest(t, False, f"the library did not load: {str(why)[:200]}") for t in tests]
        for test in tests:
            try:
                result = await asyncio.wait_for(page.evaluate(_ONE_TEST, [test.call, test.expect]), timeout=TEST_S)
                runs.append(RunOfATest(test, bool(result.get("held")), str(result.get("got") or "")))
            except TimeoutError:
                runs.append(RunOfATest(test, False, f"ran past {TEST_S:.0f}s"))
                await page.close()  # a call that does not end leaves the page busy for every test after it
                page = await context.new_page()
                await page.set_content("<!doctype html><html><body></body></html>")
                await page.add_script_tag(content=code)
            except Exception as why:  # noqa: BLE001 - the page's own error is the test's failure
                runs.append(RunOfATest(test, False, str(why)[:300]))
    finally:
        await context.close()
    return runs


async def the_tests_that_test(plan: LibraryPlan, *, browser: Any = None) -> list[LibraryTest]:
    """The plan's tests that call one of its own functions, expect a JSON value, and fail where its functions give
    nothing."""
    names = {f"{plan.name}.{f.name}(" for f in plan.functions}
    wellformed = []
    for test in plan.tests:
        try:
            json.loads(test.expect)
        except ValueError:
            continue
        if any(name in test.call for name in names):
            wellformed.append(test)
    empty = await run_tests(stub_of(plan), wellformed, browser=browser)
    return [run.test for run in empty if not run.held]


def _root() -> Path:
    from core.config import config

    return Path(config.paths.data_dir) / "her_libraries"


def kept(root: Path | None = None) -> list[KeptLibrary]:
    """Every library she has kept."""
    found = []
    for folder in sorted((root or _root()).glob("*/")):
        try:
            record = json.loads((folder / "library.json").read_text("utf-8"))
            code = (folder / "library.js").read_text("utf-8")
        except (OSError, ValueError):
            continue
        found.append(KeptLibrary(LibraryPlan.model_validate(record["plan"]), code, str(record.get("made_for") or ""),
                                 folder, list(record.get("held") or [])))
    return found


def _words(text: str) -> set[str]:
    return {w.lower().rstrip("s") for w in re.findall(r"[A-Za-z]{3,}", text)}


def found_for(plan: LibraryPlan, libraries: list[KeptLibrary], *, at_most: int = 3) -> list[KeptLibrary]:
    """Her libraries most like a planned one, by its name, purpose, what it rests on and its functions' names."""
    def about(p: LibraryPlan) -> set[str]:
        return _words(" ".join([p.name, p.purpose, *p.rests_on, *(f.name for f in p.functions)]))

    wanted = about(plan)
    scored = [(len(wanted & about(lib.plan)), lib) for lib in libraries]
    return [lib for score, lib in sorted(scored, key=lambda pair: -pair[0]) if score >= 2][:at_most]


async def keep(library: KeptLibrary, root: Path | None = None) -> Path:
    """A library kept in her store, through the write gateway: its code, its plan, the tests that held, what for."""
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    folder = (root or _root()) / f"{library.plan.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    record = {"plan": library.plan.model_dump(), "made_for": library.made_for, "held": library.held,
              "kept_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with local_internal_governed_scope("her_libraries.keep", domain="file_write"):
        gateway = get_file_write_gateway()
        await gateway.ensure_directory_async(folder, source="her_libraries.keep")
        await gateway.write_text_async(folder / "library.js", library.code, source="her_libraries.keep")
        await gateway.write_text_async(folder / "library.json", json.dumps(record, indent=1), source="her_libraries.keep")
    library.folder = folder
    return folder
