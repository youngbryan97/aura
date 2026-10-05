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
