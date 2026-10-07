"""Parts she already knows how to make go to whatever program asks for them by its features, and to no program they do not fit.

Offline 2026-10-06 a word processor rebuilt with them had all 18 of its
features working without one part written by her model, where one rebuilt part
by part had 2 of 18. The same parts are asked for by a notes app, an email
composer and a Markdown editor in their own words; a spreadsheet, a drawing
program and a to-do list get none of them.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from core.rebuilding.parts_a_maker_knows import (
    PARTS,
    parts_for,
    the_page_for,
    what_she_knows_how_to_make,
)
from core.rebuilding.what_a_program_does import Feature, Genome


def _given(name: str) -> list[str]:
    return [p.name for p in parts_for(Feature(name=name))]


@pytest.mark.unit
@pytest.mark.parametrize(("feature", "part"), [
    # A word processor's features, as her model named them.
    ("Bold, Italic, Underline", "character styles"), ("Font family and size", "fonts"), ("Text colour and highlight", "text colours"),
    ("Paragraph alignment", "alignment"), ("Bulleted and numbered lists", "lists"), ("Undo and Redo", "history"),
    ("Find and Replace", "find and replace"), ("Spell-check and grammar", "proofing"), ("Insert a table", "tables"),
    ("Print preview and print", "printing"), ("Line and paragraph spacing", "spacing"), ("Increase / decrease indent", "indent"),
    ("Insert an image", "pictures"), ("Save As / Export to Desktop", "save as"), ("New blank document", "new document"), ("Close Document", "new document"),
    # A notes app's, an email composer's and a Markdown editor's, in their own words.
    ("Make text bold or italic", "character styles"), ("Search notes", "find and replace"), ("Bulleted notes", "lists"),
    ("Export note as PDF", "save as"), ("Undo typing", "history"), ("Insert link", "links"), ("Headings", "paragraph styles"),
    ("Code blocks", "paragraph styles"), ("Word count", "proofing"), ("Margins and paper size", "page setup"), ("Centre a line", "alignment"),
])
def test_a_feature_is_given_the_part_its_name_asks_for(feature, part):
    assert _given(feature) == [part]


@pytest.mark.unit
@pytest.mark.parametrize("feature", [
    "Page numbers",            # numbered, but not a list
    "Table of contents",       # a table, but not one of rows and columns
    "Attach a file",           # nothing here attaches
    "Send email",              # nor sends
    "Save a copy to the cloud",
    "Checklists with tick boxes",
    "Track changes",
    "Mail merge",
])
def test_a_feature_no_part_does_all_of_is_left_to_be_written(feature):
    assert _given(feature) == []


@pytest.mark.unit
@pytest.mark.parametrize(("work", "a_page"), [
    ("A white A4 page centred in the window; the cursor blinks at the top-left.", True),
    ("A list of notes on the left and the open note on the right, where you write.", True),
    ("A compose window: To, Subject, and the message body you write in.", True),
    ("A grid of cells with lettered columns and numbered rows.", False),
    ("A canvas to draw on with brushes and shapes.", False),
    ("A board of tasks in columns: to do, doing, done.", False),
])
def test_the_page_is_given_only_where_the_work_is_a_document(work, a_page):
    genome = Genome(name="It", what_it_is="", work=work, features=[Feature(name="Undo and Redo"), Feature(name="Bold")])
    assert (the_page_for(genome) is not None) == a_page
    known = what_she_knows_how_to_make(genome)
    assert bool(known.given) == a_page  # a drawing's Undo is not a page's


@pytest.mark.unit
def test_the_paper_is_the_one_the_program_is_said_to_use():
    a4 = the_page_for(Genome(name="Q", work="A white A4 page with a ribbon of tabs (Home, Insert, View).", features=[]))
    letter = the_page_for(Genome(name="Q", work="A Letter-size page to write on.", features=[]))
    assert 'const said = "A4"' in a4.code and 'const said = "Letter"' in letter.code


@pytest.mark.unit
def test_one_feature_is_checked_on_the_commands_it_names():
    styles = next(p for p in PARTS if p.name == "character styles")
    bold_only = styles.checks_for(Feature(name="Bold"))
    assert bold_only and all("bold" in c.rule.lower() for c in bold_only) and all(c.feature == "Bold" for c in bold_only)
    assert len(styles.checks_for(Feature(name="Character styles"))) == len(styles.checks)


def _build(genome: Genome, folder: Path) -> dict:
    """The build flow with her model away: only what she knows and what the frame gives can work."""
    from core.rebuilding import rebuilding_a_program as rb

    async def away(prompt, schema, max_tokens):
        return None

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "what_it_does.json").write_text(genome.model_dump_json(), "utf-8")
    rb._the_build_is(folder, "", genome.name, finished=False)
    built = asyncio.run(rb._building("", away, folder, [], [], [], time.monotonic(), tell=None, browser=None, deadline_s=900, asked=""))
    return {o.feature.name: o.kept for o in built.built.outcomes}


@pytest.mark.slow
@pytest.mark.parametrize("genome", [
    Genome(name="Jotter", what_it_is="A notes app", work="A list of notes on the left and the open note on the right, where you write.", features=[
        Feature(name="Make text bold or italic"), Feature(name="Bulleted notes"), Feature(name="Search notes"), Feature(name="Headings"),
        Feature(name="Export note as PDF"), Feature(name="Undo typing")]),
    Genome(name="Courier", what_it_is="An email composer", work="A compose window with the message body you write in.", features=[
        Feature(name="Bold and italic text"), Feature(name="Insert link"), Feature(name="Numbered list"), Feature(name="Text colour"),
        Feature(name="Check spelling"), Feature(name="Align text")]),
], ids=["notes app", "email composer"])
def test_programs_of_other_kinds_work_with_the_same_parts(genome, tmp_path):
    works = _build(genome, tmp_path / genome.name.lower())
    assert all(works.get(f.name) for f in genome.features), works


@pytest.mark.slow
def test_a_ribbon_is_seen_working_in_a_program_without_the_tools_of_others(tmp_path):
    """LIVE 2026-10-06 the ribbon's check clicked "Insert table" in a program with no tables, and her model was asked to write it again."""
    genome = Genome(name="Slate", what_it_is="A word processor", work="A white A4 page under a ribbon of tabs (Home, Insert, View).", features=[
        Feature(name="Bold"), Feature(name="Bulleted list"), Feature(name="Save as"), Feature(name="Undo")])
    works = _build(genome, tmp_path / "slate")
    assert works.get("Tabbed toolbar") and all(works.get(f.name) for f in genome.features), works


@pytest.mark.slow
def test_each_kind_of_file_is_seen_written_as_that_kind(tmp_path):
    """LIVE 2026-10-06 "Export to Plain Text" was not seen as Save as, and her model spent minutes writing it again."""
    genome = Genome(name="Ledger", what_it_is="A letter writer", work="A white Letter page you write on.", features=[
        Feature(name="Export to Plain Text"), Feature(name="Export to RTF"), Feature(name="Export as Web Page"),
        Feature(name="Save as OpenDocument"), Feature(name="Save as Markdown"), Feature(name="Export to PDF")])
    known = what_she_knows_how_to_make(genome)
    assert set(known.given) == {f.name for f in genome.features}
    assert [c.rule for c in known.given["Export to Plain Text"][1]] == ["saving as plain text writes a .txt file"]
    works = _build(genome, tmp_path / "ledger")
    assert all(works.get(f.name) for f in genome.features), works


@pytest.mark.slow
def test_a_document_exported_as_pdf_says_what_was_written(tmp_path):
    from playwright.async_api import async_playwright

    from core.rebuilding.checks_a_person_makes import _what_a_file_says
    from core.rebuilding.the_program_as_built import Part, ProgramAsBuilt

    genome = Genome(name="Q", work="A white Letter page.", features=[])
    page = tmp_path / "q.html"
    ProgramAsBuilt("Q").with_part(the_page_for(genome)).with_part(Part("save as", what_she_knows_how_to_make(
        Genome(name="Q", work="A page", features=[Feature(name="Export as PDF")])).given["Export as PDF"][0].code)).write(page)

    async def export() -> bytes:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                tab = await (await browser.new_context(accept_downloads=True)).new_page()
                await tab.goto(page.as_uri())
                await tab.evaluate("""() => { app.doc.innerHTML = '<h1>Café “Plan”</h1><p>Dear Ms. Rivera, <b>thank you</b> for €12.</p>'
                    + '<ol><li>First</li><li>Second</li></ol><table><tr><td>Task</td><td>Owner</td></tr></table>'; }""")
                async with tab.expect_download() as saved:
                    await tab.evaluate("app.run('Export as PDF')")
                return await asyncio.to_thread(Path(await (await saved.value).path()).read_bytes)
            finally:
                await browser.close()

    data = asyncio.run(export())
    said = _what_a_file_says(data)
    assert data.startswith(b"%PDF-1.4") and b"%%EOF" in data[-10:]
    for words in ("Café “Plan”", "Dear Ms. Rivera,", "thank you", "€12.", "1.", "Second", "Task", "Owner"):
        assert words in said, (words, said)


@pytest.mark.unit
def test_a_build_on_a_page_her_model_wrote_is_not_taken_up_where_she_knows_the_page(tmp_path):
    """LIVE 2026-10-06 taking one up, the given parts did not fit its page and her model spent minutes rewriting them."""
    import json

    from core.rebuilding.rebuilding_a_program import _on_a_page_of_its_own

    genome = Genome(name="Quill", what_it_is="a word processor", work="A white page to write letters on.", features=[])
    for name, serves, own in (("old", ["the work area"], True), ("known", ["the work area", "document page"], False)):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "what_it_does.json").write_text(genome.model_dump_json(), "utf-8")
        (folder / "program.json").write_text(json.dumps({"title": "Quill", "parts": [{"name": "work area", "code": "", "serves": serves}]}), "utf-8")
        assert _on_a_page_of_its_own(folder) is own
