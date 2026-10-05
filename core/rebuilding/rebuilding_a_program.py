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
from core.rebuilding.the_program_as_built import ProgramAsBuilt
from core.rebuilding.what_a_program_does import (
    FEATURES_AT_ONCE,
    Asker,
    Genome,
    checks_for,
    genome_of,
    what_is_written_about,
)
from core.rebuilding.writing_it_part_by_part import Built, Teller, _say, write_it

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
            f"I rebuilt {self.program} clean-room as {self.genome.name}, from {', '.join(self.sources) or 'what I know'}: "
            f"{len(working)} of {len(self.built.outcomes)} features work, each one seen working by doing it "
            f"({sum(o.held for o in working)} checks hold). It is at {self.built.path}."
        )
        if working:
            line += " Working: " + ", ".join(o.feature.name for o in working) + "."
        if missing:
            line += " Not working: " + ", ".join(o.feature.name for o in missing) + "."
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
    ask = _recorded(ask, record)
    sources = await what_is_written_about(program, corpus=corpus, online=online) if program else []
    named = [f"{s.title} ({s.where})" for s in sources]
    if program:
        await _say(tell, f"Reading about {program}: " + ("; ".join(named) or "nothing written found, so from what I know") + ".")
    genome = await genome_of(program, sources, ask, asked=asked)
    if genome is None:
        return Rebuilt(program or "it", None, None, named, time.monotonic() - began, "I could not say what it does")
    await _say(tell, f"{program or genome.name} does {len(genome.features)} things a person uses: " + ", ".join(f.name for f in genome.features) + ".")
    folder = Path(where) / _folder_name(genome.name)
    folder.mkdir(parents=True, exist_ok=True)
    record.append(folder / "what_she_asked.jsonl")
    (folder / "what_it_does.json").write_text(json.dumps(genome.model_dump(), indent=1), "utf-8")
    if genome.kind.strip().lower() == "code":
        built = await _as_code(genome, ask, folder, tell)
        await asyncio.to_thread(keep_the_record, folder, built)
        return Rebuilt(program or genome.name, genome, built, named, time.monotonic() - began)
    written: list[Check] = []
    for at in range(0, len(genome.features), FEATURES_AT_ONCE):
        written.extend(await checks_for(genome, genome.features[at : at + FEATURES_AT_ONCE], ask, sources=sources, asked=asked))
    checks = await checks_that_mean_something(written, folder / "empty.html", browser=browser)
    (folder / "checks.json").write_text(json.dumps([c.model_dump() for c in checks], indent=1), "utf-8")
    await _say(tell, f"Wrote {len(written)} checks a person would make, before any code; {len(checks)} of them fail on an empty program, so they test something.")
    left = max(60.0, deadline_s - (time.monotonic() - began))
    built = await write_it(genome, checks, ask, folder / "index.html", tell=tell, browser=browser, deadline_s=left)
    await asyncio.to_thread(keep_the_record, folder, built)
    return Rebuilt(program or genome.name, genome, built, named, time.monotonic() - began)


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


def keep_the_record(folder: Path, built: Built) -> None:
    """What works and what does not, and the program's parts, beside it."""
    built.program.keep(folder)
    (folder / "what_works.json").write_text(json.dumps([
        {"feature": o.feature.name, "works": o.kept, "checks_held": o.held, "checks": o.of, "tries": o.tries, "why_not": o.why_not}
        for o in built.outcomes
    ], indent=1), "utf-8")
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
