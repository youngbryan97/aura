"""A program rebuilt part by part keeps a part only for what doing its checks shows.

The model is stood in for by a script here; what is tested is the rest: that
a check which holds on an empty program is dropped, that a part whose checks
hold is kept, that a part which breaks what worked is not, and that a feature
nothing can be made to hold for is left out and said to be.
"""
from __future__ import annotations

import pytest

from core.rebuilding.checks_a_person_makes import Check, run_checks
from core.rebuilding.rebuilding_a_program import checks_that_mean_something, rebuild
from core.rebuilding.the_program_as_built import Part, ProgramAsBuilt
from core.rebuilding.what_a_program_does import Genome, _its_kind
from core.rebuilding.writing_it_part_by_part import WrittenPart

_WORK = """
const page = app.make("div", {class: "page", contenteditable: "true", "aria-label": "Document"});
app.work.append(page);
page.addEventListener("input", () => app.changed());
app.doc = page;
"""
_BOLD = 'app.command({label: "Bold", icon: "B", group: "Font", keys: "Mod+B", run: () => document.execCommand("bold")});'
_BREAKS = 'app.work.replaceChildren(); app.command({label: "Italic", group: "Font", run: () => {}});'
_COUNT = 'app.status({label: "Words", value: () => "Words: " + (app.doc.innerText.match(/\\S+/g) || []).length});'


def _check(feature: str, steps: list[dict], expect: list[dict]) -> Check:
    return Check.model_validate({"feature": feature, "steps": steps, "expect": expect})


_BOLD_CHECK = _check("Bold", [{"do": "type", "value": "hello world"}, {"do": "select_text", "target": "world"}, {"do": "click", "target": "Bold"}],
                     [{"see": "style", "target": "world", "property": "font-weight", "value": "bold"}])
_ITALIC_CHECK = _check("Italic", [{"do": "type", "value": "hello"}, {"do": "select_all"}, {"do": "click", "target": "Italic"}],
                       [{"see": "style", "target": "hello", "property": "font-style", "value": "italic"}])
_COUNT_CHECK = _check("Word count", [{"do": "type", "value": "one two three"}], [{"see": "text", "target": "Words: 3"}])
_EMPTY_CHECK = _check("Title", [], [{"see": "text", "target": "Untitled"}])


class _Script:
    """Answers each ask the way a scripted model would."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    async def __call__(self, prompt, schema, max_tokens):
        self.asked.append(prompt)
        if schema is Genome:
            return Genome.model_validate({"name": "Writer", "what_it_is": "a word processor", "work": "a page of text", "features": [
                {"name": "Bold", "how": "select text, press Bold", "shows": "bold text", "weight": 3},
                {"name": "Italic", "how": "select text, press Italic", "shows": "italic text", "weight": 2},
                {"name": "Word count", "how": "type", "shows": "the count of words", "weight": 1},
            ]})
        if schema.__name__ == "_Checks":
            return schema.model_validate({"checks": [c.model_dump() for c in (_BOLD_CHECK, _ITALIC_CHECK, _COUNT_CHECK, _EMPTY_CHECK)]})
        if schema is WrittenPart:
            if "work-area part of" in prompt:
                return WrittenPart(code=_WORK, style=".page{min-height:300px;padding:20px;background:#fff}")
            if '"Bold"' in prompt:
                return WrittenPart(code=_BOLD)
            if '"Italic"' in prompt:
                return WrittenPart(code=_BREAKS)
            if '"Word count"' in prompt:
                return WrittenPart(code=_COUNT)
        return None


class _NoCorpus:
    def by_title(self, title):
        return None

    def search(self, text, limit=5):
        return []


@pytest.mark.asyncio
async def test_a_check_that_holds_on_nothing_is_dropped(tmp_path):
    kept = await checks_that_mean_something([_BOLD_CHECK, _EMPTY_CHECK], tmp_path / "empty.html")
    assert kept == [_BOLD_CHECK]


@pytest.mark.asyncio
async def test_checks_are_done_to_the_page(tmp_path):
    program = ProgramAsBuilt("Writer", parts=[Part("work area", _WORK), Part("Bold", _BOLD)])
    runs = await run_checks(program.write(tmp_path / "p.html"), [_BOLD_CHECK, _ITALIC_CHECK])
    assert [r.held for r in runs] == [True, False]
    assert "no control named 'Italic'" in runs[1].why


@pytest.mark.asyncio
async def test_parts_are_kept_for_what_they_do_and_not_for_breaking_things(tmp_path):
    script = _Script()
    done = await rebuild("Writer", script, tmp_path, corpus=_NoCorpus(), online=False)
    outcome = {o.feature.name: o for o in done.built.outcomes}
    assert outcome["Bold"].kept and outcome["Word count"].kept
    # Italic's part emptied the work area: Bold stopped holding, so it was tried
    # again and then left out.
    assert not outcome["Italic"].kept and outcome["Italic"].tries == 3
    assert "broke what worked before" in outcome["Italic"].why_not
    page = done.built.path.read_text()
    assert 'execCommand("bold")' in page and 'label: "Italic"' not in page
    assert "2 of 3 features work" in done.summary()


def test_the_kind_of_program_is_read_from_how_its_article_opens():
    assert _its_kind("Microsoft Word, or simply Word, is a word processing program developed by Microsoft.", "Microsoft Word") == "word processor"
    assert _its_kind("Google Docs is an online word processor and part of the free suite.", "Google Docs") == "word processor"
    assert _its_kind("Microsoft Excel is a spreadsheet editor developed by Microsoft.", "Microsoft Excel") == "spreadsheet editor"


_ITALIC = 'app.command({label: "Italic", icon: "I", group: "Font", run: () => document.execCommand("italic")});'


class _ChangeScript(_Script):
    """Asked for a change, says it adds Italic; and writes Italic as it should be, or as `breaking` says."""

    def __init__(self, breaking: bool = False) -> None:
        super().__init__()
        self.breaking = breaking

    async def __call__(self, prompt, schema, max_tokens):
        if schema.__name__ == "_TheChange":
            return schema.model_validate({"features": [{"name": "Italic", "how": "select text, press Italic", "shows": "italic text"}]})
        if schema.__name__ == "_Checks" and "The person asks" not in prompt and '"Italic"' not in prompt and "- Italic:" in prompt and "- Bold" not in prompt:
            return schema.model_validate({"checks": [_ITALIC_CHECK.model_dump()]})
        if schema is WrittenPart and '"Italic"' in prompt and "Bold" not in prompt.split("Features already in the program:")[0]:
            return WrittenPart(code=_BREAKS if self.breaking else _ITALIC)
        return await super().__call__(prompt, schema, max_tokens)


@pytest.mark.asyncio
async def test_a_change_is_kept_when_it_works_and_breaks_nothing(tmp_path):
    from core.rebuilding.changing_what_was_built import change_it

    done = await rebuild("Writer", _Script(), tmp_path, corpus=_NoCorpus(), online=False)
    folder = done.built.path.parent
    changed = await change_it(folder, "add italic", _ChangeScript())
    assert [o.kept for o in changed.outcomes] == [True], changed.summary()
    page = (folder / "index.html").read_text()
    assert 'execCommand("italic")' in page and 'execCommand("bold")' in page
    assert (folder / "index.before-change.html").exists()


@pytest.mark.asyncio
async def test_a_change_that_breaks_what_worked_is_not_kept(tmp_path):
    from core.rebuilding.changing_what_was_built import change_it

    done = await rebuild("Writer", _Script(), tmp_path, corpus=_NoCorpus(), online=False)
    folder = done.built.path.parent
    changed = await change_it(folder, "add italic", _ChangeScript(breaking=True))
    assert [o.kept for o in changed.outcomes] == [False]
    assert "replaceChildren" not in (folder / "program.json").read_text()


@pytest.mark.asyncio
async def test_the_frame_writes_a_zip_that_opens(tmp_path):
    """A .docx, .xlsx or .odt is a zip; the frame writes one a zip reader opens."""
    import base64
    import io
    import zipfile

    from playwright.async_api import async_playwright

    page_file = ProgramAsBuilt("Z").write(tmp_path / "z.html")
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page()
        await page.goto(page_file.resolve().as_uri())
        made = await page.evaluate("""(async () => { const bytes = new Uint8Array(await app.zip({"a.txt": "café", "d/b.xml": "<x/>"}).arrayBuffer());
            let s = ""; for (const b of bytes) s += String.fromCharCode(b); return btoa(s); })()""")
        await browser.close()
    archive = zipfile.ZipFile(io.BytesIO(base64.b64decode(made)))
    assert archive.testzip() is None
    assert archive.read("a.txt").decode() == "café" and archive.namelist() == ["a.txt", "d/b.xml"]


def test_a_number_where_words_are_meant_is_read_as_words():
    """LIVE 2026-10-05 a batch of checks was lost to one 'wait 500' written as a number."""
    check = Check.model_validate({"feature": "f", "steps": [{"do": "wait", "value": 500}], "expect": [{"see": "count", "target": "tr", "value": 3}]})
    assert check.steps[0].value == "500" and check.expect[0].value == "3"


def test_code_given_as_code_is_taken():
    """LIVE 2026-10-05 a part came back as the code itself, not inside the shape asked for, and was dropped."""
    import asyncio

    from core.rebuilding.her_model import ask_her_model

    class _Router:
        async def generate_with_metadata(self, *args, **kwargs):
            return {"text": "Here it is:\n```javascript\napp.command({label: 'New', run: () => {}});\n```"}

    got = asyncio.run(ask_her_model("write it", WrittenPart, 100, router=_Router()))
    assert isinstance(got, WrittenPart) and got.code.startswith("app.command")


@pytest.mark.asyncio
async def test_elements_are_made_as_either_convention_writes_them(tmp_path):
    """LIVE 2026-10-05 the work area was written with className and contentEditable, which the frame
    set as attributes nothing reads, and everything built on it failed."""
    part = Part("work area", '''const page = app.make("div", {className: "page"}, [app.make("div", {className: "body", contentEditable: "true", style: {minHeight: "50px"}})]);
app.work.append(page); app.doc = page.querySelector(".body"); app.doc.focus();''')
    check = Check.model_validate({"feature": "type", "steps": [{"do": "click", "target": "page"}, {"do": "type", "value": "hello"}], "expect": [{"see": "text", "target": "hello"}]})
    runs = await run_checks(ProgramAsBuilt("W", parts=[part]).write(tmp_path / "p.html"), [check])
    assert runs[0].held and not runs[0].errors


@pytest.mark.asyncio
async def test_the_file_a_check_gives_is_ready_before_the_click_that_asks_for_it(tmp_path):
    opener = Part("open", 'app.command({label: "Open", menu: "File", run: async () => { const f = await app.pickFile(".txt"); if (f) app.doc.innerText = f.text; }});')
    check = Check.model_validate({"feature": "open", "steps": [{"do": "click", "target": "Open"}, {"do": "give_file", "target": "r.txt", "value": "Quarterly Report"}],
                                  "expect": [{"see": "text", "target": "Quarterly Report"}]})
    runs = await run_checks(ProgramAsBuilt("W", parts=[Part("work area", _WORK), opener]).write(tmp_path / "p.html"), [check])
    assert runs[0].held, runs[0].why


_SAVE_AS = 'app.command({label: "Save As", menu: "File", run: async () => { const got = await app.ask({title: "Save As", fields: [{label: "Name", name: "name", value: app.name}], ok: "Save"}); if (got) app.name = got.name; }});'
_SAVE_AS_CHECK = Check.model_validate({"feature": "Save As", "steps": [{"do": "type", "value": "My Document"}, {"do": "click", "target": "Save As"},
                                       {"do": "fill", "target": "Name", "value": "Renamed"}, {"do": "click", "target": "Save"}],
                                       "expect": [{"see": "dialog", "target": "Save As"}, {"see": "name", "value": "Renamed"}]})


@pytest.mark.asyncio
async def test_a_dialog_opened_and_closed_by_the_steps_was_seen(tmp_path):
    """LIVE 2026-10-05 a right Save As was left out: its check expected the dialog still open after Save was pressed in it."""
    runs = await run_checks(ProgramAsBuilt("W", parts=[Part("work area", _WORK), Part("Save As", _SAVE_AS)]).write(tmp_path / "p.html"), [_SAVE_AS_CHECK])
    assert runs[0].held, runs[0].why
    runs = await run_checks(ProgramAsBuilt("W", parts=[Part("work area", _WORK)]).write(tmp_path / "e.html"), [_SAVE_AS_CHECK])
    assert not runs[0].held


def test_the_option_a_person_means_is_chosen():
    from core.rebuilding.checks_a_person_makes import the_option_meant

    formats = [["txt", "Plain Text (.txt)"], ["docx", "Word Document (.docx)"], ["pdf", "PDF"]]
    assert the_option_meant(formats, "Word (.docx)") == "docx"
    assert the_option_meant(formats, "PDF") == "pdf" and the_option_meant(formats, "txt") == "txt"
    assert the_option_meant(formats, "Rich Text") is None
    assert the_option_meant([["a", "Document one"], ["b", "Document two"]], "Document") is None  # no one of them


@pytest.mark.asyncio
async def test_a_control_is_found_as_its_check_names_it_wherever_it_is(tmp_path):
    """LIVE 2026-10-05 Bold was left out: its check named the button as the driver reads it, and only the page was looked at."""
    check = Check.model_validate({"feature": "Bold", "steps": [{"do": "type", "value": "hi"}],
                                  "expect": [{"see": "element", "target": "button[title='Bold'], button:has-text('B')"}]})
    runs = await run_checks(ProgramAsBuilt("W", parts=[Part("work area", _WORK), Part("Bold", _BOLD)]).write(tmp_path / "p.html"), [check])
    assert runs[0].held, runs[0].why
    runs = await run_checks(ProgramAsBuilt("W", parts=[Part("work area", _WORK)]).write(tmp_path / "e.html"), [check])
    assert not runs[0].held


def test_checks_are_written_from_what_is_written_about_each_feature():
    """One check for each rule, each feature shown what is written about it (specification grounding)."""
    import asyncio

    from core.rebuilding.what_a_program_does import Feature, Source, checks_for, what_is_written_of

    article = Source("Microsoft Word", (
        "Microsoft Word is a word processor. Its native file format is DOCX, introduced in 2007. "
        "Word can export documents to PDF and to plain text. The company was founded in 1975 in Albuquerque."
    ), "her corpus")
    export = Feature(name="Export to DOCX", how="File, Export, choose DOCX", shows="a .docx file is saved")
    assert "native file format is DOCX" in what_is_written_of(export, [article])
    assert what_is_written_of(Feature(name="Bold", how="press B", shows="bold text"), [article]) == ""

    asked: list[str] = []

    async def ask(prompt, schema, max_tokens):
        asked.append(prompt)
        return schema.model_validate({"checks": [{"feature": "Export to DOCX", "rule": "saves its native format, DOCX",
                                                  "steps": [{"do": "click", "target": "Export"}], "expect": [{"see": "download", "value": ".docx"}]}]})

    genome = Genome.model_validate({"name": "Writer", "features": [export.model_dump()]})
    got = asyncio.run(checks_for(genome, [export], ask, sources=[article], asked="and export it to my Desktop"))
    assert "native file format is DOCX" in asked[0] and "export it to my Desktop" in asked[0]
    assert got[0].rule and got[0].said().startswith("[saves its native format")


class _SecondProgram(_Script):
    """Another word processor, whose Bold part is asked for after Writer was built."""

    async def __call__(self, prompt, schema, max_tokens):
        if schema is Genome:
            return Genome.model_validate({"name": "Scribe", "what_it_is": "a word processor", "work": "a page of text", "features": [
                {"name": "Bold", "how": "select text, press Bold", "shows": "bold text", "weight": 3}]})
        if schema.__name__ == "_Checks":
            return schema.model_validate({"checks": [_BOLD_CHECK.model_dump()]})
        return await super().__call__(prompt, schema, max_tokens)


@pytest.mark.asyncio
async def test_a_part_that_worked_before_is_used_for_a_feature_like_it(tmp_path):
    await rebuild("Writer", _Script(), tmp_path, corpus=_NoCorpus(), online=False)
    second = _SecondProgram()
    done = await rebuild("Scribe", second, tmp_path, corpus=_NoCorpus(), online=False)
    # Writer's Bold part holds Scribe's checks as it is: kept with nothing written for it.
    assert not [p for p in second.asked if 'feature "Bold"' in p]
    asked_for_area = [p for p in second.asked if "work-area part of" in p]
    assert "earlier program of hers, Writer" in asked_for_area[0]
    assert {o.feature.name for o in done.built.outcomes if o.kept} == {"Bold"}
    assert 'execCommand("bold")' in done.built.path.read_text()


class _CutShort(_Script):
    async def __call__(self, prompt, schema, max_tokens):
        if schema is WrittenPart and '"Word count"' in prompt:
            raise RuntimeError("the machine was restarted")
        return await super().__call__(prompt, schema, max_tokens)


@pytest.mark.asyncio
async def test_a_build_cut_short_keeps_what_it_had_made(tmp_path):
    with pytest.raises(RuntimeError):
        await rebuild("Writer", _CutShort(), tmp_path, corpus=_NoCorpus(), online=False)
    kept = ProgramAsBuilt.kept_in(next(tmp_path.glob("*/program.json")).parent)
    assert {p.name for p in kept.parts} >= {"work area", "Bold"}
