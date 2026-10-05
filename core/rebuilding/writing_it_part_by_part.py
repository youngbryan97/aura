"""Writing a program part by part, keeping each part only for what doing its checks shows.

The order is a developer's: first the place the work is done (the work area),
then each feature in turn, the ones a person would miss first ahead. For each,
her model writes a part, the part is put in the program, and the program is
used: the new feature's checks, and every check that held before. A part is
kept when some of its own checks now hold and none that held before has
stopped holding; otherwise what went wrong is shown to her model and it tries
again, a few times, and the feature is left out rather than kept broken.

Her model writes code here and nothing it says is taken for true: a part is
kept for what using the program shows, and the report says which features
were seen working and which were not.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.rebuilding.checks_a_person_makes import Check, CheckRun, run_checks
from core.rebuilding.the_program_as_built import Part, ProgramAsBuilt
from core.rebuilding.what_a_program_does import Asker, Feature, Genome

logger = logging.getLogger("Rebuilding.PartByPart")

__all__ = ["Built", "FRAME_API", "WrittenPart", "write_it"]

#: Tries at one part before the feature is left out.
TRIES = 3

#: Tries at the work area, which everything else is built on.
WORK_AREA_TRIES = 5

#: The longest part her model is asked for, in tokens.
PART_TOKENS = 2048

#: What a part plugs into. Shown to her model with every part it writes.
FRAME_API = """\
The program runs in a browser page inside a frame (title bar, menus, toolbar, work area, status bar).
Each part is the body of a function given `app`, run once when the page loads. Plain JavaScript, no imports,
no libraries; the page has no network. `app` has:
  app.work                      the main work area element (empty until the work-area part fills it)
  app.command({label, run, menu, group, icon, keys, active})
                                adds a command: a menu item under `menu` (e.g. "File", "Edit", "Insert", "Format")
                                and/or a toolbar button in toolbar `group`; `icon` is one short glyph or emoji;
                                `keys` like "Mod+B" (Mod = Ctrl or Cmd); `active()` returns true to show it pressed;
                                `run(arg)` may be async. The visible name is `label`.
  app.control({label, kind, options, group, set, value})
                                a toolbar field: kind "select" (options: [value] or [[value, text]]), "color" or
                                "number"; `set(value)` applies it to the selection; `value()` reads the current one.
  app.ask({title, fields, ok, cancel, text}) -> Promise of {name: value} or null
                                a dialog; each field {label, name, kind: "text"|"number"|"select"|"checkbox"|"textarea"|"color",
                                value, options}; result keys are field names (or labels).
  app.tell(title, text)         a message dialog.        app.notify(text)   a passing notice.
  app.status({label, value, side})   a status-bar item; value() is shown, refreshed after every change.
  app.side(element|null)        show or hide a side panel holding element.
  app.download(name, content, type)  save a file.        app.pickFile(accept) -> Promise of {name, text, dataUrl} or null
  app.keep(key, value) / app.kept(key, fallback)          remember across reopening (JSON values).
  app.zip({path: text or bytes}, type)  a Blob of a zip of those files: what .docx, .xlsx, .odt and .epub files are.
  app.on("change"|"selection"|"ready", fn)   app.changed()   call after changing the document.
  app.name                      the document's name (get/set).
  app.make(tag, props, children) make an element: props are attributes or DOM properties (class or className, text,
                                html, a style object, onClick ...); children are elements or text.
  app.selection() / app.restore(range)      save and put back the text selection.
  app.run(label)                run another command by its label.
Commands keep the text selection when their buttons are clicked. A part may also put things on `app` for later
parts (e.g. app.doc = {...}). document.execCommand may be used for rich text editing.
"""


class WrittenPart(BaseModel):
    code: str = Field(default="", max_length=16_000, description="the body of function(app), plain JavaScript")
    style: str = Field(default="", max_length=4_000, description="CSS this part needs, if any")


@dataclass
class FeatureOutcome:
    feature: Feature
    held: int = 0
    of: int = 0
    tries: int = 0
    kept: bool = False
    why_not: str = ""


@dataclass
class Built:
    program: ProgramAsBuilt
    path: Path
    outcomes: list[FeatureOutcome] = field(default_factory=list)
    holding: list[Check] = field(default_factory=list)
    seconds: float = 0.0
    unfinished: list[str] = field(default_factory=list)

    def working(self) -> list[FeatureOutcome]:
        return [o for o in self.outcomes if o.kept]


Teller = Callable[[str], Awaitable[None] | None]


async def _say(tell: Teller | None, line: str) -> None:
    logger.info("rebuilding: %s", line)
    if tell is not None:
        said = tell(line)
        if asyncio.iscoroutine(said):
            await said


def _the_work_area_prompt(genome: Genome, checks: list[Check]) -> str:
    shown = "\n".join(f"- {c.feature}: {c.said()}" for c in checks[:6])
    return (
        f"Write the work-area part of {genome.name}: {genome.what_it_is}\n"
        f"The work area: {genome.work}\n"
        "Fill app.work with it, styled so it looks like a polished desktop program, and put on `app` what later "
        "parts will need to work on it. No toolbar or menu commands yet; those are later parts.\n"
        f"Later checks will do things like:\n{shown}\n\n{FRAME_API}"
    )


def _a_feature_prompt(
    genome: Genome, built: ProgramAsBuilt, feature: Feature, checks: list[Check], wrong: str, before: str, *, as_it_is: bool = False,
) -> str:
    work = next((p for p in built.parts if p.name == "work area"), None)
    others = ", ".join(c for p in built.parts for c in p.serves if p.name != "work area") or "none yet"
    shown = "\n".join(f"- {c.said()}" for c in checks)
    text = (
        f"{genome.name}: {genome.what_it_is}\n"
        f"Write the part for the feature \"{feature.name}\": {feature.how} -> {feature.shows} (menu: {feature.place or 'any'}).\n"
        f"It must pass these checks, done by a person on a fresh, empty program:\n{shown}\n"
        f"Features already in the program: {others}.\n\n"
        f"The work-area part already written:\n```js\n{work.code if work else ''}\n```\n\n{FRAME_API}"
    )
    if as_it_is:
        text += f"\nThis part as it is now, to be changed so it does the above:\n```js\n{before[:6000]}\n```\n"
    elif wrong:
        text += f"\nYour last version of this part:\n```js\n{before[:5000]}\n```\nDoing the checks with it showed: {wrong}\nWrite it again so the checks pass."
    return text


async def _write(ask: Asker, prompt: str) -> WrittenPart | None:
    written = await ask(prompt, WrittenPart, PART_TOKENS)
    if not isinstance(written, WrittenPart) or not written.code.strip():
        return None
    code = written.code.strip()
    if code.startswith("```"):
        code = code.split("\n", 1)[-1].rsplit("```", 1)[0]
    return written.model_copy(update={"code": code})


async def _does_not_parse(code: str) -> str:
    """What the browser would say of code that does not parse, checked before it is tried."""
    try:
        process = await asyncio.create_subprocess_exec(
            "node", "--check", "-", stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
    except OSError:
        return ""
    _out, err = await process.communicate(f"(function (app) {{\n{code}\n}});".encode())
    if process.returncode == 0:
        return ""
    lines = [line for line in err.decode("utf-8", "ignore").splitlines() if "Error" in line]
    return (lines[0] if lines else "it does not parse")[:300]


def _what_went_wrong(runs: list[CheckRun], broke: list[CheckRun], errors: list[str]) -> str:
    said = [f"check \"{r.check.said()[:220]}\" did not hold: {r.why[:240]}" for r in runs if not r.held][:4]
    said += [f"it broke what worked before: \"{r.check.said()[:160]}\" ({r.why[:160]})" for r in broke][:3]
    said += [f"the page said: {e[:200]}" for e in errors[:3]]
    return "; ".join(said) or "nothing held"


class _Trying:
    """Putting a candidate program on disk and doing the checks on it: its own, and every one that holds so far."""

    def __init__(self, trial: Path, holding: list[Check], browser: Any) -> None:
        self.trial, self.holding, self.browser = trial, holding, browser

    async def __call__(self, candidate: ProgramAsBuilt, own: list[Check]) -> tuple[list[CheckRun], list[CheckRun], list[str]]:
        await asyncio.to_thread(candidate.write, self.trial)
        runs = await run_checks(self.trial, [*own, *self.holding], browser=self.browser)
        mine, before = runs[: len(own)], runs[len(own) :]
        errors = sorted({e for r in runs for e in r.errors})
        return mine, [r for r in before if not r.held], errors


async def write_a_feature(
    genome: Genome, program: ProgramAsBuilt, feature: Feature, own: list[Check], ask: Asker, tried: _Trying,
) -> tuple[ProgramAsBuilt, FeatureOutcome, list[Check]]:
    """One feature's part, written and tried until its checks hold without breaking any that held; the program with it, or as it was."""
    outcome = FeatureOutcome(feature, of=len(own))
    if not own:
        outcome.why_not = "no check could be written for it"
        return program, outcome, []
    existing = next((p for p in program.parts if p.name == feature.name), None)
    wrong, before, best = "", existing.code if existing else "", None
    for outcome.tries in range(1, TRIES + 1):
        written = await _write(ask, _a_feature_prompt(genome, program, feature, own, wrong, before, as_it_is=existing is not None and not wrong))
        if written is None:
            wrong = "nothing was written"
            continue
        before = written.code
        unparsed = await _does_not_parse(written.code)
        if unparsed:
            wrong = f"the code does not parse: {unparsed}"
            continue
        candidate = program.with_part(Part(feature.name, written.code, [feature.name]))
        candidate.style = "\n".join(s for s in (program.style, written.style) if s)
        mine, broke, errors = await tried(candidate, own)
        held = [r for r in mine if r.held]
        if held and not broke and (best is None or len(held) > len(best[1])):
            best = (candidate, held)
        if len(held) == len(own) and not broke:
            break
        wrong = _what_went_wrong(mine, broke, errors)
    if best is None:
        outcome.why_not = wrong
        logger.info("rebuilding: %s left out: %s", feature.name, wrong[:300])
        return program, outcome, []
    program, held = best
    outcome.kept, outcome.held = True, len(held)
    logger.info("rebuilding: %s works (%d of %d checks hold)", feature.name, len(held), len(own))
    return program, outcome, [r.check for r in held]


async def write_it(
    genome: Genome,
    checks: list[Check],
    ask: Asker,
    out: Path,
    *,
    tell: Teller | None = None,
    browser: Any = None,
    deadline_s: float = 3 * 3600.0,
) -> Built:
    """The program, written part by part: the work area, then each feature that can be made to hold."""
    began = time.monotonic()
    program = ProgramAsBuilt(genome.name, accent=genome.accent or "#2b579a")
    holding: list[Check] = []
    outcomes: list[FeatureOutcome] = []
    out = Path(out)
    tried = _Trying(out.with_name(out.stem + ".trying.html"), holding, browser)
    await _say(tell, f"Writing the work area of {genome.name}: {genome.work}")
    # Everything else is built on the work area, so it is kept only when it
    # loads without an error. LIVE 2026-10-05 one that threw as it loaded was
    # kept on its last try, and every feature after it failed reaching into it.
    wrong, before, fallback = "", "", None
    for _attempt in range(WORK_AREA_TRIES):
        again = f"\nYour last version:\n```js\n{before[:5000]}\n```\nLoading it, the page said: {wrong}\nWrite it again so it loads without errors." if wrong else ""
        written = await _write(ask, _the_work_area_prompt(genome, checks) + again)
        if written is None:
            continue
        before = written.code
        unparsed = await _does_not_parse(written.code)
        if unparsed:
            wrong = f"it does not parse: {unparsed}"
            continue
        candidate = program.with_part(Part("work area", written.code, ["the work area"]))
        candidate.style = written.style
        _mine, _broke, errors = await tried(candidate, [])
        if not errors:
            program, fallback = candidate, None
            break
        wrong, fallback = "; ".join(errors[:3]), candidate
    if fallback is not None:
        program = fallback
        logger.info("rebuilding: the work area still says %s", wrong[:300])
    for feature in genome.features:
        if time.monotonic() - began > deadline_s:
            outcomes.append(FeatureOutcome(feature, why_not="out of time"))
            continue
        program, outcome, held = await write_a_feature(genome, program, feature, [c for c in checks if c.feature == feature.name], ask, tried)
        outcomes.append(outcome)
        holding.extend(held)
        done = len(outcomes)
        if done % 5 == 0 and done < len(genome.features):
            await _say(tell, f"{done} of {len(genome.features)} features written; {sum(o.kept for o in outcomes)} of them work.")
    from core.rebuilding.how_finished_it_is import finishing

    await _say(tell, "Every feature written; now measuring how finished it looks and handles, and mending what is not.")
    program, unfinished = await finishing(program, holding, ask, tried, out.with_name(out.stem + ".finishing.html"), browser=browser)
    await asyncio.to_thread(program.write, out)
    await asyncio.to_thread(tried.trial.unlink, missing_ok=True)
    left = [*unfinished.found, *(f'using "{d}" changes nothing' for d in unfinished.dead), *unfinished.errors]
    return Built(program, out, outcomes, holding, time.monotonic() - began, left)
