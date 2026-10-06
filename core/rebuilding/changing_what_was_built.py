"""Changing a program that was built, to what a person asks of it, without breaking what works.

"Add a dark theme", "make Save ask for a name", "a word count in the status
bar": a change is features, new ones or ones that now work differently, and is
built the way the program was. Her model reads the program's features and the
request and says which features the request adds or changes. The checks a
person would make of those are written first, and kept only if they fail on
the program as it is (a check that already holds is not a change). Each
feature's part is then written, or rewritten from the part as it is, and kept
only when its checks hold and every check that held before still holds,
except the checks of the features the request changes on purpose.

The program before the change is kept beside it, so a change can be undone.
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.rebuilding.checks_a_person_makes import Check, run_checks
from core.rebuilding.her_model import HerModelIsAwayError, patiently
from core.rebuilding.the_program_as_built import ProgramAsBuilt
from core.rebuilding.what_a_program_does import Asker, Feature, Genome, checks_for
from core.rebuilding.what_the_frame_gives import what_the_frame_gives
from core.rebuilding.writing_it_part_by_part import (
    FeatureOutcome,
    Teller,
    _reused,
    _say,
    _Trying,
    write_a_feature,
)

logger = logging.getLogger("Rebuilding.Changing")

__all__ = ["Changed", "change_it"]


class _TheChange(BaseModel):
    features: list[Feature] = Field(default_factory=list, description="features the request adds, or existing ones as they should now work, named as they are named in the program")
    changes: list[str] = Field(default_factory=list, description="names of existing features that the request makes work differently")


@dataclass
class Changed:
    folder: Path
    asked: str
    outcomes: list[FeatureOutcome] = field(default_factory=list)
    seconds: float = 0.0
    why_not: str = ""

    def summary(self) -> str:
        done = [o for o in self.outcomes if o.kept]
        if self.why_not and not done:
            return f"I did not change it: {self.why_not}"
        line = f"I changed it as asked: {len(done)} of {len(self.outcomes)} changes work, each seen working by doing it, and everything that worked before still works."
        made = [o for o in done if not o.why_not]
        if made:
            line += " Done: " + ", ".join(o.feature.name for o in made) + "."
        had = [o for o in done if o.why_not]
        if had:
            line += " It already did this, as checking it shows: " + ", ".join(o.feature.name for o in had) + "."
        missing = [o for o in self.outcomes if not o.kept]
        if missing:
            line += " Not done: " + ", ".join(f"{o.feature.name} ({o.why_not[:120]})" for o in missing) + "."
        if self.why_not:
            line += f" It stopped there: {self.why_not}."
        return line + f" The program before the change is beside it as index.before-change.html, in {self.folder}."


def _read(folder: Path) -> tuple[ProgramAsBuilt, Genome, list[Check]]:
    program = ProgramAsBuilt.kept_in(folder)
    genome = Genome.model_validate_json((folder / "what_it_does.json").read_text("utf-8"))
    holding_file = folder / "holding.json"
    holding = [Check.model_validate(c) for c in json.loads(holding_file.read_text("utf-8"))] if holding_file.exists() else []
    return program, genome, holding


async def change_it(folder: Path, asked: str, ask: Asker, *, tell: Teller | None = None, browser: Any = None) -> Changed:
    """Change the program kept in ``folder`` as ``asked`` says; see the module's account."""
    began = time.monotonic()
    folder = Path(folder)
    ask = patiently(ask, tell)
    try:
        program, genome, holding = await asyncio.to_thread(_read, folder)
    except (OSError, ValueError) as why:
        return Changed(folder, asked, why_not=f"there is no program I built in {folder} ({why})")
    listed = "\n".join(f"- {f.name}: {f.how} -> {f.shows}" for f in genome.features)
    try:
        change = await ask(
            f"{genome.name}: {genome.what_it_is}\nIts features:\n{listed}\n\n"
            f"The person asks for this change: {asked}\n"
            "Say which features the change adds, and which existing features it makes work differently, each as a person "
            "would use it and see it. An existing feature keeps its name.",
            _TheChange, 1500,
        )
    except HerModelIsAwayError:
        change = None
    if not isinstance(change, _TheChange):
        change = _TheChange()
    # What the frame does (saving as a format named) is read from the words and given by code, whatever her model said.
    change.features, given = what_the_frame_gives(change.features, [], asked)
    if not change.features:
        return Changed(folder, asked, why_not="I could not say what the change is")
    await _say(tell, "The change, as features: " + ", ".join(f.name for f in change.features) + ".")
    changed_names = {n.lower() for n in change.changes} | {f.name.lower() for f in change.features}
    still = [c for c in holding if c.feature.lower() not in changed_names]
    outcomes: list[FeatureOutcome] = []
    tried = _Trying(folder / "index.trying.html", still, browser)
    try:
        written = [c for gift in given.values() for c in gift.checks]
        written += await checks_for(genome, [f for f in change.features if f.name not in given], ask)
        now = folder / "before-change.trying.html"
        await asyncio.to_thread(program.write, now)
        runs = await run_checks(now, written, browser=browser)
        checks = [r.check for r in runs if not r.held]
        # A change every check of which holds on the program as it is was made already: it does that now.
        already = {r.check.feature for r in runs} - {c.feature for c in checks}
        await asyncio.to_thread(now.unlink, missing_ok=True)
        await _say(tell, f"Wrote {len(written)} checks for it before changing anything; {len(checks)} fail on the program as it is.")
        await asyncio.to_thread(shutil.copyfile, folder / "index.html", folder / "index.before-change.html")
        for feature in change.features:
            if feature.name in already:
                held_now = [r.check for r in runs if r.check.feature == feature.name]
                outcomes.append(FeatureOutcome(feature, held=len(held_now), of=len(held_now), kept=True, why_not="it already does this"))
                continue
            own = [c for c in checks if c.feature == feature.name]
            gift = given.get(feature.name)
            reused = await _reused(program, feature, own, gift.part, tried) if gift is not None and own else None
            if reused is not None:
                program, kept = reused
                outcome, held = FeatureOutcome(feature, held=len(kept), of=len(own), kept=True), [r.check for r in kept]
            else:
                program, outcome, held = await write_a_feature(genome, program, feature, own, ask, tried)
            outcomes.append(outcome)
            still.extend(held)
            if outcome.kept and feature.name.lower() not in {f.name.lower() for f in genome.features}:
                genome.features.append(feature)
    except HerModelIsAwayError as why:
        await asyncio.to_thread(_keep, folder, program, genome, still, tried)
        return Changed(folder, asked, outcomes, time.monotonic() - began, why_not=f"my model stopped answering ({why}); what changed so far is kept")
    await asyncio.to_thread(_keep, folder, program, genome, still, tried)
    return Changed(folder, asked, outcomes, time.monotonic() - began)


def _keep(folder: Path, program: ProgramAsBuilt, genome: Genome, holding: list[Check], tried: _Trying) -> None:
    program.write(folder / "index.html")
    program.keep(folder)
    (folder / "what_it_does.json").write_text(json.dumps(genome.model_dump(), indent=1), "utf-8")
    (folder / "holding.json").write_text(json.dumps([c.model_dump() for c in holding], indent=1), "utf-8")
    tried.trial.unlink(missing_ok=True)

