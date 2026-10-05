"""Unfinished is measured, not judged: overlaps, cut-off text, contrast, names, and controls that do nothing."""
from __future__ import annotations

import pytest

from core.rebuilding.how_finished_it_is import what_is_unfinished
from core.rebuilding.the_program_as_built import Part, ProgramAsBuilt

pytestmark = pytest.mark.asyncio

_WORK = Part("work area", 'const p = app.make("div",{class:"page",contenteditable:"true","aria-label":"Document"}); app.work.append(p);')


async def test_a_finished_program_measures_nothing_unfinished(tmp_path):
    program = ProgramAsBuilt("Writer", style=".page{background:#fff;min-height:300px;margin:20px;padding:20px}", parts=[
        _WORK, Part("bold", 'app.command({label:"Bold", icon:"B", group:"Font", run:()=>document.execCommand("bold")});'),
    ])
    assert (await what_is_unfinished(program.write(tmp_path / "p.html"))).count() == 0


async def test_what_a_person_would_notice_is_measured(tmp_path):
    program = ProgramAsBuilt("Writer", style=".page{background:#fff;color:#ddd;min-height:200px}", parts=[
        Part("work area", 'const p = app.make("div",{class:"page",contenteditable:"true"}); p.textContent="faint words"; app.work.append(p);'),
        Part("ghost", 'document.body.append(app.make("button",{style:"position:fixed;top:44px;left:8px;width:80px;height:30px"}));'),
        Part("sparkle", 'app.command({label:"Sparkle", icon:"*", group:"Font", run:()=>{}});'),
        Part("dark", 'app.command({label:"Dark", icon:"D", group:"View", run:()=>document.body.classList.toggle("dark")});'),
        Part("table", 'app.command({label:"Table", icon:"T", group:"Insert", run:()=>app.ask({title:"Table", fields:[{label:"Rows"}]})});'),
        Part("save", 'app.command({label:"Save", menu:"File", run:()=>app.download("a.txt","x")});'),
    ])
    seen = await what_is_unfinished(program.write(tmp_path / "p.html"))
    said = seen.said()
    assert seen.dead == ["Sparkle"]
    assert "hard to read" in said and "no name" in said and "drawn over each other" in said
