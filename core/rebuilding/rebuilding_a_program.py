"""Rebuilding a program clean-room: from what is written about it to a working program, checked by use.

The whole of it, in the order a person would do it:

  1. read what is written about the program and its kind (her corpus, Wikipedia)
  2. say what it does, as features a person uses (her model, reading those)
  3. write how a person would check each feature, before any code exists,
     and keep only the checks that fail on the empty frame
  4. write the program part by part, keeping each part for what doing the
     checks shows (writing_it_part_by_part.py)
  5. say which features were seen working, and which were not

No code of the original is read at any point; only what is written about what
it does. Nothing here knows what any program is.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.rebuilding.checks_a_person_makes import Check, run_checks
from core.rebuilding.her_model import HerModelIsAwayError, patiently
from core.rebuilding.parts_a_maker_knows import what_she_knows_how_to_make
from core.rebuilding.the_program_as_built import ProgramAsBuilt
from core.rebuilding.what_a_program_does import (
    FEATURES_AT_ONCE,
    Asker,
    Genome,
    checks_for,
    genome_of,
    what_is_written_about,
    what_was_read,
)
from core.rebuilding.what_it_is_discerned_to_be import _and, discerned
from core.rebuilding.what_the_frame_gives import Given, what_the_frame_gives
from core.rebuilding.writing_it_part_by_part import Built, Teller, _say, so_far_in, write_it

logger = logging.getLogger("Rebuilding")

__all__ = ["Rebuilt", "checks_that_mean_something", "rebuild"]


@dataclass
class Rebuilt:
    program: str
    genome: Genome | None
    built: Built | None
    sources: list[str]
    seconds: float
    why_not: str = ""

    def summary(self) -> str:
        if self.built is None or self.genome is None:
            return f"I could not rebuild {self.program}: {self.why_not}"
        working = self.built.working()
        missing = [o for o in self.built.outcomes if not o.kept]
        line = (
            f"I rebuilt {self.program} clean-room as {self.genome.name}, from {_and(self.sources) or 'what I know'}: "
            f"{len(working)} of {len(self.built.outcomes)} features work, each one seen working by doing it "
            f"({sum(o.held for o in working)} checks hold). It is at {self.built.path}."
        )
        if working:
            line += " Working: " + _and([o.feature.name for o in working]) + "."
        if missing:
            line += " Not working: " + _and([o.feature.name for o in missing]) + "."
        if self.built.unfinished:
            line += f" Still unfinished, measured: {'; '.join(self.built.unfinished[:6])}."
        return line


async def checks_that_mean_something(checks: list[Check], scratch: Path, *, browser: Any = None) -> list[Check]:
    """Only the checks that fail on the frame with nothing in it: one that holds on nothing checks nothing."""
    empty = await asyncio.to_thread(ProgramAsBuilt("Empty").write, scratch)
    runs = await run_checks(empty, checks, browser=browser)
    await asyncio.to_thread(scratch.unlink, missing_ok=True)
    return [r.check for r in runs if not r.held]


def _recorded(ask: Asker, where: list[Path]) -> Asker:
    """``ask``, with each question and answer kept beside the program, so what her model wrote can be read afterwards."""

    async def asking(prompt: str, schema: type, max_tokens: int) -> Any:
        began = time.monotonic()
        answer = await ask(prompt, schema, max_tokens)
        if where:
            line = json.dumps({
                "asked_for": schema.__name__, "seconds": round(time.monotonic() - began, 1), "answered": answer is not None,
                "prompt": prompt[:6000], "answer": answer.model_dump() if answer is not None else None,
            })
            await asyncio.to_thread(_append, where[0], line)
        return answer

    return asking


def _append(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as out:
        out.write(line + "\n")


def _folder_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "rebuilt"


async def rebuild(
    program: str,
    ask: Asker,
    where: Path,
    *,
    tell: Teller | None = None,
    corpus: Any = None,
    online: bool = True,
    browser: Any = None,
    deadline_s: float = 3 * 3600.0,
    asked: str = "",
) -> Rebuilt:
    """Rebuild ``program`` from what is written about it into ``where``; see the module's account.

    With no program named, what is built is what ``asked`` specifies: the
    person's words are the only source, and everything after is the same.
    """
    began = time.monotonic()
    record: list[Path] = []
    ask = patiently(_recorded(ask, record), tell)
    sources = await what_is_written_about(program, corpus=corpus, online=online) if program else []
    named = [what_was_read(sources)] if sources else []
    taken_up = await asyncio.to_thread(an_unfinished_build_of, program, asked, Path(where))
    if taken_up is not None and await asyncio.to_thread(_on_a_page_of_its_own, taken_up):
        # Built on a page her model wrote, where she now knows how to make the page: its parts were made for
        # that page, and given parts do not fit it (LIVE 2026-10-06, four minutes rewriting Save as for it).
        await _say(tell, f"An earlier build of {program or 'it'} I had not finished was made on a page of its own; "
                         f"starting afresh on the page I know how to make, and leaving that one as it was, in {taken_up}.")
        taken_up = None
    try:
        if taken_up is not None:
            await _say(tell, f"Taking up the build of {program or 'it'} I had not finished, from what I kept in {taken_up}.")
            return await _building(program, ask, taken_up, record, sources, named, began, tell=tell, browser=browser, deadline_s=deadline_s, asked=asked)
        if program:
            await _say(tell, f"Reading about {program}: {named[0]}." if named else f"I found nothing written about {program}, so I work from what I know of it.")
        # Her model's reading is one witness to what the program is; what is written, the person's words and what follows
        # from what it does are the others, and code weighs them (core/rebuilding/what_it_is_discerned_to_be.py).
        # Where she can discern what it is, her model is asked what it knows of it from its own knowledge, without what
        # she read: a witness of its own, not an echo of the articles. Elsewhere it reads them and its reading stands.
        on_its_own = discerned(program, sources, asked) is not None
        if on_its_own:
            await _say(tell, f"Asking my model what it knows of {program or 'it'}, from its own knowledge: what it is and what a person does "
                             "with it. I weigh that against what I read, what you asked for, what I built before and what follows from what "
                             "the program does.")
        try:
            heard = await genome_of(program, [] if on_its_own else sources, ask, asked=asked)
        except HerModelIsAwayError as why:
            logger.info("rebuilding: her model is away (%s); discerning without what it knows", why)
            heard = None
        seen = discerned(program, sources, asked, heard, remembered=await asyncio.to_thread(_built_before, Path(where), program))
        if seen is not None and heard is not None and any("my model" in by for by in seen.witnesses.values()):
            named = [*named, "what my model knows of it"]
        genome = seen.genome if seen is not None else heard
        if genome is None:
            return Rebuilt(program or "it", None, None, named, time.monotonic() - began, "I could not say what it does")
        if seen is not None:
            for line in seen.said(program or "it"):
                await _say(tell, line)
        else:
            await _say(tell, f"{program or genome.name} does {len(genome.features)} things a person uses: " + ", ".join(f.name for f in genome.features) + ".")
        folder = _a_new_folder(Path(where), genome.name)
        (folder / "what_it_does.json").write_text(json.dumps(genome.model_dump(), indent=1), "utf-8")
        _the_build_is(folder, program, asked, finished=False)
        return await _building(program, ask, folder, record, sources, named, began, tell=tell, browser=browser, deadline_s=deadline_s, asked=asked)
    except HerModelIsAwayError as why:
        return await asyncio.to_thread(_as_far_as_it_got, program, asked, Path(where), named, time.monotonic() - began, str(why))


async def _building(program: str, ask: Asker, folder: Path, record: list[Path], sources: list[Any], named: list[str], began: float, *,
                    tell: Teller | None, browser: Any, deadline_s: float, asked: str) -> Rebuilt:
    """The build in ``folder``, from what is kept there: its features, its checks once written, its parts once kept."""
    record.append(folder / "what_she_asked.jsonl")
    genome = Genome.model_validate_json((folder / "what_it_does.json").read_text("utf-8"))
    if genome.kind.strip().lower() == "code":
        built = await _as_code(genome, ask, folder, tell)
        await asyncio.to_thread(keep_the_record, folder, built)
        _the_build_is(folder, program, asked, finished=True)
        return Rebuilt(program or genome.name, genome, built, named, time.monotonic() - began)
    # Parts she already knows how to make, and what the frame already does, are given by code with their checks
    # (core/rebuilding/parts_a_maker_knows.py, core/rebuilding/what_the_frame_gives.py); her model writes the rest.
    knows = what_she_knows_how_to_make(genome, sources)
    genome.features = [*genome.features, *knows.added]
    genome.features, given = what_the_frame_gives(genome.features, sources, asked, already=set(knows.given),
                                                  saving_as=any(part.name == "save as" for part, _checks in knows.given.values()),
                                                  usual="docx" if knows.page is not None else "")
    for feature in genome.features:
        if feature.name in knows.given:
            part, checks = knows.given[feature.name]
            given[feature.name] = Given(feature, part, checks)
    if knows.given or given:
        await _say(tell, knows.said(frame=sum(1 for name in given if name not in knows.given),
                                    left=[f.name for f in genome.features if f.name not in given]))
    if (folder / "checks.json").exists():
        # A feature given a part is checked by that part's checks, in place of any written for it before.
        checks = [Check.model_validate(c) for c in json.loads((folder / "checks.json").read_text("utf-8"))]
        checks = [c for c in checks if c.feature not in given] + [c for gift in given.values() for c in gift.checks]
    else:
        written: list[Check] = [check for gift in given.values() for check in gift.checks]
        asked_of_her = [f for f in genome.features if f.name not in given]
        for at in range(0, len(asked_of_her), FEATURES_AT_ONCE):
            written.extend(await checks_for(genome, asked_of_her[at : at + FEATURES_AT_ONCE], ask, sources=sources, asked=asked))
        checks = await checks_that_mean_something(written, folder / "empty.html", browser=browser)
        (folder / "checks.json").write_text(json.dumps([c.model_dump() for c in checks], indent=1), "utf-8")
        await _say(tell, f"Before any code, I wrote {len(written)} checks a person would make of it; "
                         + ("all of them" if len(checks) == len(written) else f"{len(checks)} of them")
                         + " fail on an empty program, so each tests something real.")
    left = max(60.0, deadline_s - (time.monotonic() - began))
    built = await write_it(genome, checks, ask, folder / "index.html", tell=tell, browser=browser, deadline_s=left,
                           given={name: gift.part for name, gift in given.items()}, so_far=await asyncio.to_thread(so_far_in, folder, genome),
                           page=knows.page)
    await asyncio.to_thread(keep_the_record, folder, built)
    _the_build_is(folder, program, asked, finished=True)
    return Rebuilt(program or genome.name, genome, built, named, time.monotonic() - began)


def _a_new_folder(where: Path, name: str) -> Path:
    """A folder of its own for a new build: never one an earlier build left things in, whose checks and parts are another program's."""
    base = where / _folder_name(name)
    folder, n = base, 1
    while folder.exists() and any(folder.iterdir()):
        n += 1
        folder = base.with_name(f"{base.name}-{n}")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _the_build_is(folder: Path, program: str, asked: str, *, finished: bool) -> None:
    """What the build in ``folder`` is of, and whether it was finished: how a build cut short is found again."""
    (folder / "build.json").write_text(json.dumps({"program": program, "asked": asked, "finished": finished}, indent=1), "utf-8")


def _on_a_page_of_its_own(folder: Path) -> bool:
    """Whether an unfinished build's work area is one her model wrote, for a program whose work is a page she knows how to make."""
    from core.rebuilding.parts_a_maker_knows import the_page_for

    try:
        genome = Genome.model_validate_json((folder / "what_it_does.json").read_text("utf-8"))
        parts = json.loads((folder / "program.json").read_text("utf-8")).get("parts", []) if (folder / "program.json").exists() else []
    except (OSError, ValueError):
        return False
    area = next((part for part in parts if part.get("name") == "work area"), None)
    return area is not None and "document page" not in (area.get("serves") or []) and the_page_for(genome) is not None


def an_unfinished_build_of(program: str, asked: str, where: Path) -> Path | None:
    """The newest build in ``where`` of the same program (or, with none named, of the same request) that was not finished."""
    for manifest in sorted(Path(where).glob("*/build.json"), key=lambda p: -p.stat().st_mtime):
        try:
            said = json.loads(manifest.read_text("utf-8"))
        except (OSError, ValueError):
            continue
        same = (str(said.get("program") or "").casefold() == program.casefold()) if program else str(said.get("asked") or "") == asked
        if same and not said.get("finished") and (manifest.parent / "what_it_does.json").exists():
            return manifest.parent
    return None


def _as_far_as_it_got(program: str, asked: str, where: Path, named: list[str], seconds: float, why: str) -> Rebuilt:
    """A build her model went away from, as far as it got: kept on disk, to be taken up when she is asked again."""
    folder = an_unfinished_build_of(program, asked, where)
    said = f"my model stopped answering ({why}) and did not come back while I waited"
    if folder is None:
        return Rebuilt(program or "it", None, None, named, seconds, said)
    genome = Genome.model_validate_json((folder / "what_it_does.json").read_text("utf-8"))
    so_far = so_far_in(folder, genome)
    if so_far is None:
        return Rebuilt(program or genome.name, None, None, named, seconds, f"{said}; nothing was kept yet, in {folder}")
    so_far.program.write(folder / "index.html")
    built = Built(so_far.program, folder / "index.html", so_far.outcomes, so_far.holding, seconds,
                  [f"{said}; what is kept is in {folder}, and asking again takes the build up from there"])
    return Rebuilt(program or genome.name, genome, built, named, seconds)


async def _as_code(genome: Genome, ask: Asker, folder: Path, tell: Teller | None) -> Built:
    """A program that is code: its calls written first, then its functions, each kept for what calling it showed."""
    from core.rebuilding.programs_in_code import calls_for, write_code

    calls = []
    for at in range(0, len(genome.features), FEATURES_AT_ONCE):
        calls.extend(await calls_for(genome, genome.features[at : at + FEATURES_AT_ONCE], ask))
    await _say(tell, f"Wrote {len(calls)} calls a person would make of it, before any code; each is made in a sandbox with no network.")
    began = time.monotonic()
    program, outcomes, holding = await write_code(genome, calls, ask, folder / "program.py")
    return Built(program, folder / "program.py", outcomes, holding, time.monotonic() - began)


def _built_before(where: Path, program: str) -> list[Genome]:
    """What she built before, as she said then what each did: her memory of programs, a witness to what one does."""
    known: list[Genome] = []
    for kept in sorted(where.glob("*/what_it_does.json")) if where.is_dir() else []:
        try:
            build = json.loads((kept.parent / "build.json").read_text("utf-8")) if (kept.parent / "build.json").exists() else {}
            if build.get("finished") and str(build.get("program") or "").lower() == str(program or "").lower():
                known.append(Genome.model_validate_json(kept.read_text("utf-8")))
        except (OSError, ValueError):
            continue
    return known


def keep_the_record(folder: Path, built: Built) -> None:
    """What works and what does not, and the program's parts, beside it."""
    from core.rebuilding.writing_it_part_by_part import what_works

    built.program.keep(folder)
    (folder / "what_works.json").write_text(json.dumps(what_works(built.outcomes), indent=1), "utf-8")
    (folder / "holding.json").write_text(json.dumps([c.model_dump() for c in built.holding], indent=1), "utf-8")


def the_build_meant(asked: str, roots: list[Path]) -> Path | None:
    """The folder of the build a request means: the one it names by path or by name, else the newest.

    "Add a dark theme to Inkwell" names it; "make the word processor you built
    save as PDF" means the one she built last, which is the newest kept.
    """
    from core.language.named_paths import first_existing_path

    named = first_existing_path(asked or "")
    if named is not None:
        folder = Path(named) if Path(named).is_dir() else Path(named).parent
        if (folder / "program.json").exists():
            return folder
    builds = [p.parent for root in roots if root.is_dir() for p in root.glob("*/program.json")]
    words = set(re.findall(r"[a-z0-9]+", str(asked or "").lower()))
    for folder in builds:
        if set(folder.name.split("-")) <= words:
            return folder
    return max(builds, key=lambda p: (p / "program.json").stat().st_mtime, default=None)
