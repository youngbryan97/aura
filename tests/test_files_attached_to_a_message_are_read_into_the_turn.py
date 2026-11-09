"""A file the person attaches is kept, read for what it says, and handed to the turn of the message it came with.

8 November 2026: Bryan wanted her thoughts on his paper, as a PDF or a Word file, "also, general file upload. not
limited to those two". The window had no way to give her a file.
"""
from __future__ import annotations

import asyncio
import io
import zipfile

import pytest

from core.conversation import attachments


@pytest.fixture
def kept_in(tmp_path, monkeypatch):
    monkeypatch.setattr(attachments, "attachments_dir", lambda: tmp_path / "attachments")
    return tmp_path / "attachments"


def _pdf(pages: list[str]) -> bytes:
    import fitz

    document = fitz.open()
    for text in pages:
        document.new_page().insert_text((72, 72), text)
    data = document.tobytes()
    document.close()
    return data


def _docx(paragraphs: list[str]) -> bytes:
    import docx

    document = docx.Document()
    for text in paragraphs:
        document.add_paragraph(text)
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Seed"
    table.rows[0].cells[1].text = "97"
    out = io.BytesIO()
    document.save(out)
    return out.getvalue()


def _xlsx(strings: list[str]) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as package:
        cells = "".join(f"<si><t>{text}</t></si>" for text in strings)
        package.writestr("xl/sharedStrings.xml", f"<sst>{cells}</sst>")
        package.writestr("xl/worksheets/sheet1.xml", "<worksheet><sheetData/></worksheet>")
    return out.getvalue()


def test_a_pdf_is_read_page_by_page():
    kind, text = attachments.read_text("paper.pdf", _pdf(["Integrated information", "The bridge at parity"]))
    assert kind == "pdf"
    assert "[page 1]\nIntegrated information" in text and "[page 2]\nThe bridge at parity" in text


def test_a_word_file_is_read_with_its_tables():
    kind, text = attachments.read_text("paper.docx", _docx(["Abstract", "We measure what she reports."]))
    assert kind == "document"
    assert "We measure what she reports." in text and "Seed | 97" in text


def test_a_spreadsheet_is_read_from_its_parts():
    kind, text = attachments.read_text("runs.xlsx", _xlsx(["seed", "verdict"]))
    assert kind == "spreadsheet" and "seed verdict" in text


@pytest.mark.parametrize(("name", "data", "kind"), [
    ("notes.md", "# Notes\nsomething".encode(), "text"),
    ("run.py", b"print('hi')\n", "text"),
    ("data.bin", b"\x00\x01\x02" * 10, "binary"),
])
def test_any_other_file_is_read_as_text_or_said_to_have_none(name, data, kind):
    assert attachments.read_text(name, data)[0] == kind


def test_a_long_file_is_given_by_its_opening_and_the_passages_nearest_the_message_with_the_rest_marked():
    paragraphs = [f"Paragraph {n} is about weather and gardens." for n in range(200)]
    paragraphs[150] = "Paragraph 150 says the bridge crossed at parity on seed 97."
    text = "\n\n".join(paragraphs)
    given = attachments.fitted(text, 1500, "Did the bridge cross at parity?")
    assert given.startswith("Paragraph 0 ")
    assert "the bridge crossed at parity on seed 97" in given
    assert "characters of the file not read here" in given
    assert len(given) < 1800


def test_a_file_that_fits_is_given_whole():
    assert attachments.fitted("short", 100, "anything") == "short"


def test_a_kept_file_is_handed_to_the_turn_and_an_unreadable_one_is_described(kept_in):
    paper = asyncio.run(attachments.keep("paper.pdf", _pdf(["Our claim is narrow."]), "application/pdf"))
    blob = asyncio.run(attachments.keep("weights.bin", b"\x00" * 64, "application/octet-stream"))
    assert paper["readable"] and not blob["readable"]
    assert (kept_in / paper["id"] / "original.pdf").read_bytes().startswith(b"%PDF-")

    block, given = attachments.context_block([paper["id"], blob["id"], "not-an-id"], query="claim", room=4000)
    assert [name for name, _ in given] == ["paper.pdf", "weights.bin"] or [name for name, _ in given] == [
        "weights.bin", "paper.pdf"]
    assert "=== ATTACHED FILE: paper.pdf ===" in block and "Our claim is narrow." in block
    assert "Nothing could be read from this file. It is a binary file of 64 bytes" in block


def test_an_empty_or_oversized_file_is_refused(kept_in, monkeypatch):
    with pytest.raises(attachments.AttachmentRefused):
        asyncio.run(attachments.keep("empty.txt", b""))
    monkeypatch.setattr(attachments, "MOST_BYTES", 10)
    with pytest.raises(attachments.AttachmentRefused):
        asyncio.run(attachments.keep("big.txt", b"x" * 11))


def test_a_name_cannot_leave_its_folder():
    assert attachments.safe_name("../../etc/passwd") == "passwd"
    assert attachments.safe_name("") == "file"


def test_the_window_sends_the_ids_with_the_message():
    from pathlib import Path

    script = Path("interface/static/aura.js").read_text(encoding="utf-8")
    page = Path("interface/static/index.html").read_text(encoding="utf-8")
    assert "{ message, attachments: item.attachments }" in script
    assert "fetch('/api/attachments'" in script
    assert 'id="attach-btn"' in page and 'id="attach-input" multiple' in page


def test_the_turn_reads_what_was_attached():
    import inspect

    from interface.routes import chat_preflight

    source = inspect.getsource(chat_preflight)
    assert "context_block, _attached, query=_original_user_message" in source


def test_her_window_shows_the_file_picker_the_attach_button_asks_for():
    """LIVE 8 November 2026: the paperclip did nothing. A file input asks the app's WKUIDelegate to run the open
    panel, and the launcher had no handler for it, so WebKit chose no files and showed nothing."""
    from pathlib import Path

    launcher = Path("scripts/AuraLauncher.swift").read_text(encoding="utf-8")
    handler = launcher.split("runOpenPanelWith parameters: WKOpenPanelParameters", 1)[1].split("\n    }\n", 1)[0]
    assert "isLocalRuntimeOrigin(frame.securityOrigin)" in handler
    assert "NSOpenPanel()" in handler and "parameters.allowsMultipleSelection" in handler
    assert "completionHandler(response == .OK ? panel.urls : nil)" in handler
