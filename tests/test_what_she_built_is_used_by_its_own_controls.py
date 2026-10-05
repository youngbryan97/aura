"""Writing a letter in a program she built and saving it where the person said, with the program's own controls.

Her model writes the letter; the program's page is typed into and its own save
command pressed by code, and the file kept is read back.
"""
from __future__ import annotations

import pytest

from core.rebuilding.the_program_as_built import Part, ProgramAsBuilt
from core.rebuilding.using_it_by_its_controls import (
    _Writing,
    a_writing_task,
    the_command_that_saves,
    write_it_in,
)

_WORK = """
const page = app.make("div", {class: "page", contenteditable: "true", "aria-label": "Document", style: {minHeight: "200px"}});
app.work.append(page); app.doc = page;
"""
_SAVE_DOCX = 'app.command({label: "Save as Word Document (.docx)", menu: "File", run: () => app.formats.save("docx")});'
_SAVE_TXT = 'app.command({label: "Save as Plain Text (.txt)", menu: "File", run: () => app.formats.save("txt")});'


def test_the_command_pressed_is_the_kind_asked_for_else_the_programs_own():
    labels = ["New", "Save As…", "Save as Plain Text (.txt)", "Save as Word Document (.docx)", "Bold"]
    assert the_command_that_saves(labels, "write a letter and export it to my Desktop") == "Save as Word Document (.docx)"
    assert the_command_that_saves(labels, "write a note and save it as plain text") == "Save as Plain Text (.txt)"
    assert the_command_that_saves(["New", "Bold"], "write a letter and save it") is None
    assert a_writing_task("writing a one-page letter in it and exporting it to my Desktop")
    assert not a_writing_task("drawing a cat")


@pytest.mark.asyncio
async def test_a_letter_is_written_and_kept_where_the_person_said(tmp_path):
    import docx

    page = ProgramAsBuilt("Writer", parts=[Part("work area", _WORK), Part("txt", _SAVE_TXT), Part("docx", _SAVE_DOCX)]).write(tmp_path / "index.html")

    async def ask(prompt, schema, max_tokens):
        assert schema is _Writing and "one-page letter" in prompt
        return _Writing(title="Letter to Ms. Reyes", paragraphs=["Dear Ms. Reyes,", "Thank you for the quarterly figures.", "Yours sincerely, Aura"])

    desktop = tmp_path / "Desktop"
    used = await write_it_in(page, "writing a one-page letter in it and exporting it to my Desktop", desktop, ask)
    assert used is not None and used.ok, used and used.why_not
    assert used.files == [desktop / "Letter to Ms. Reyes.docx"]
    said = [p.text for p in docx.Document(str(used.files[0])).paragraphs]
    assert said[:2] == ["Dear Ms. Reyes,", "Thank you for the quarterly figures."]
