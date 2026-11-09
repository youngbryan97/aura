"""Files the person hands her in the conversation: kept with her own data, and read for what they say.

A file is kept whole where her data is kept (``data/attachments/<id>/``), beside the text read from it and a note of
what it is. What is read depends on what the file is: text and code as they are; a PDF page by page, and by the
words on its pages where it is a scan; Word, RTF, OpenDocument and web pages through the system's own converter
(``textutil``); spreadsheets and slides from the text in their parts; a picture by the words on it and what her eyes
see in it. Anything else is kept and said to be unreadable, with its name, size and type, so she can say what it is
rather than guess at it.

When a message carries attachments, what was read is handed to the turn in full where it fits the turn's room, and
otherwise as its opening and the passages nearest the message, with every range left out marked, so she can say
what she did not read instead of supplying it.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
import secrets
import subprocess
import tempfile
import time
import zipfile
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Conversation.Attachments")

#: The largest file kept, in bytes. A long illustrated PDF is tens of megabytes; a file past this is a dataset or a
#: disk image, which the conversation's room could not hold a useful part of.
MOST_BYTES = 100 * 1024 * 1024

#: The most pages of a scanned PDF whose words are read off the page pictures.
MOST_PAGES_READ_BY_SIGHT = 200

_ID = re.compile(r"^[0-9a-f]{16}$")
_BY_TEXTUTIL = {".doc", ".rtf", ".rtfd", ".odt", ".html", ".htm", ".webarchive", ".wordml"}
_PICTURES = {".png", ".jpg", ".jpeg", ".gif", ".heic", ".webp", ".tif", ".tiff", ".bmp"}
#: Office files that are zips of XML, by the parts that hold their text, and what kind of thing each is.
_ZIPPED_XML = {
    ".xlsx": ("spreadsheet", ("xl/sharedStrings.xml", "xl/worksheets/")),
    ".pptx": ("slides", ("ppt/slides/slide",)),
    ".ods": ("spreadsheet", ("content.xml",)),
    ".odp": ("slides", ("content.xml",)),
}
_RECOVERABLE = (OSError, RuntimeError, ValueError, TypeError, KeyError, zipfile.BadZipFile, subprocess.SubprocessError)


class AttachmentRefused(ValueError):
    """A file that cannot be kept: too large, or empty."""


def attachments_dir() -> Path:
    from core.config import config

    return Path(config.paths.data_dir) / "attachments"


def safe_name(name: str) -> str:
    """The file's own name, without a directory or control characters, as it can be written."""
    base = Path(str(name or "").replace("\\", "/")).name
    base = re.sub(r"[\x00-\x1f\x7f/:]", "_", base).strip(" .") or "file"
    return base[:120]


# ── reading what a file says ──────────────────────────────────────────────────


def read_text(name: str, data: bytes) -> tuple[str, str]:
    """What kind of file ``data`` is, and the text read from it ('' where it has none it can give)."""
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf" or data[:5] == b"%PDF-":
        return "pdf", _pdf_text(data)
    if suffix == ".docx":
        return "document", _docx_text(data)
    if suffix in _BY_TEXTUTIL:
        return "document", _converted_by_textutil(data, suffix)
    if suffix in _ZIPPED_XML:
        kind, parts = _ZIPPED_XML[suffix]
        return kind, _zipped_xml_text(data, parts)
    if suffix in _PICTURES:
        return "picture", ""
    if b"\x00" not in data[:8192]:
        return "text", data.decode("utf-8", errors="replace")
    return "binary", ""


def _pdf_text(data: bytes) -> str:
    import fitz

    pages: list[tuple[int, str]] = []
    with fitz.open(stream=data, filetype="pdf") as document:
        for number, page in enumerate(document, start=1):
            pages.append((number, page.get_text().strip()))
        if not any(text for _number, text in pages):
            # A scan: the words are only in the pictures of its pages.
            from core.perception.words_on_screen import words_in

            pages = []
            for number, page in enumerate(document, start=1):
                if number > MOST_PAGES_READ_BY_SIGHT:
                    break
                picture = page.get_pixmap(dpi=150).tobytes("png")
                pages.append((number, "\n".join(words_in(picture))))
    return "\n\n".join(f"[page {number}]\n{text}" for number, text in pages if text)


def _docx_text(data: bytes) -> str:
    try:
        import docx

        document = docx.Document(io.BytesIO(data))
        lines = [paragraph.text for paragraph in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                lines.append(" | ".join(cell.text.strip() for cell in row.cells))
        return "\n".join(line for line in lines if line.strip())
    except _RECOVERABLE + (ImportError,) as why:
        logger.info("a Word file read by python-docx failed (%s); trying the system converter", why)
        return _converted_by_textutil(data, ".docx")


def _converted_by_textutil(data: bytes, suffix: str) -> str:
    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / f"attached{suffix}"
        source.write_bytes(data)
        done = subprocess.run(["/usr/bin/textutil", "-convert", "txt", "-stdout", str(source)],
                              capture_output=True, timeout=120, check=False)
    return done.stdout.decode("utf-8", errors="replace") if done.returncode == 0 else ""


def _zipped_xml_text(data: bytes, parts: tuple[str, ...]) -> str:
    def natural(name: str) -> list[Any]:
        return [int(piece) if piece.isdigit() else piece for piece in re.split(r"(\d+)", name)]

    with zipfile.ZipFile(io.BytesIO(data)) as package:
        names = sorted((name for name in package.namelist()
                        if name.endswith(".xml") and any(name.startswith(part) for part in parts)), key=natural)
        texts = []
        for name in names:
            xml = package.read(name).decode("utf-8", errors="replace")
            text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", xml)).strip()
            if text:
                texts.append(f"[{name}]\n{text}")
    return "\n\n".join(texts)


async def _picture_text(data: bytes) -> str:
    """The words on a picture, and what her eyes see in it, as far as each could be had."""
    said: list[str] = []
    try:
        from core.perception.words_on_screen import words_in

        words = await asyncio.to_thread(words_in, data)
        if words:
            said.append("Words on it:\n" + "\n".join(words))
    except _RECOVERABLE + (ImportError,) as why:
        record_degradation("conversation.attachments", why, severity="info", action="read a picture without its words")
    try:
        from core.perception.her_eyes import look_with_her_eyes

        seen = await look_with_her_eyes("What is in this picture? Describe it plainly.",
                                        base64.b64encode(data).decode("ascii"), max_tokens=300, timeout_s=90.0)
        if str(seen or "").strip():
            said.append("What I see in it:\n" + str(seen).strip())
    except _RECOVERABLE + (ImportError, asyncio.TimeoutError) as why:
        record_degradation("conversation.attachments", why, severity="info", action="read a picture without her eyes")
    return "\n\n".join(said)


# ── keeping a file ────────────────────────────────────────────────────────────


async def keep(name: str, data: bytes, content_type: str = "") -> dict[str, Any]:
    """Keep ``data`` as an attachment and read it; what was kept, as the person's window shows it."""
    if not data:
        raise AttachmentRefused("the file is empty")
    if len(data) > MOST_BYTES:
        raise AttachmentRefused(f"the file is larger than {MOST_BYTES // (1024 * 1024)} MB")
    name = safe_name(name)
    try:
        kind, text = await asyncio.to_thread(read_text, name, data)
    except _RECOVERABLE + (ImportError,) as why:
        record_degradation("conversation.attachments", why, severity="info", action=f"kept {name} unread")
        kind, text = "unreadable", ""
    if kind == "picture":
        text = await _picture_text(data)
    attachment_id = secrets.token_hex(8)
    note = {"id": attachment_id, "name": name, "size": len(data), "content_type": str(content_type or "")[:120],
            "kind": kind, "chars": len(text), "readable": bool(text.strip()), "kept_at": time.time()}
    folder = attachments_dir() / attachment_id
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    gateway = get_file_write_gateway()
    with local_internal_governed_scope(
        "conversation.attachments",
        domain="state_mutation",
        receipt_prefix="conversation-attachment",
        constraints={"artifact": "aura.conversation_attachment.v1", "operation": "create"},
    ):
        await gateway.ensure_directory_async(folder, source="conversation.attachments")
        await gateway.write_bytes_async(folder / f"original{Path(name).suffix.lower()}", data,
                                        source="conversation.attachments")
        await gateway.write_text_async(folder / "text.txt", text, source="conversation.attachments")
        await gateway.write_text_async(folder / "note.json", json.dumps(note, indent=2), source="conversation.attachments")
    logger.info("kept attachment %s: %s, %s, %d bytes, %d characters read", attachment_id, name, kind, len(data), len(text))
    return note


def kept(attachment_id: str) -> tuple[dict[str, Any], str] | None:
    """The note and the text of a kept attachment; None for an id that names none."""
    if not _ID.match(str(attachment_id or "")):
        return None
    folder = attachments_dir() / str(attachment_id)
    try:
        note = json.loads((folder / "note.json").read_text(encoding="utf-8"))
        text = (folder / "text.txt").read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    return note, text


# ── handing what was read to a turn ───────────────────────────────────────────


def _words(text: str) -> set[str]:
    return {word.lower() for word in re.findall(r"[A-Za-z]{4,}", text or "")}


def fitted(text: str, room: int, query: str = "") -> str:
    """``text`` whole where it fits ``room`` characters; else its opening and the passages nearest ``query``, in
    order, each range left out marked with how much of the file it was."""
    if len(text) <= room:
        return text
    passages: list[str] = []
    for part in (part for part in re.split(r"\n\s*\n", text) if part.strip()):
        # A passage longer than a quarter of the room is taken in runs of its lines, so a file without blank lines
        # is not one block that never fits.
        while len(part) > room // 4:
            cut = part.rfind("\n", 0, room // 4)
            cut = cut if cut > 0 else room // 4
            passages.append(part[:cut])
            part = part[cut:].lstrip("\n")
        if part.strip():
            passages.append(part)
    asked = _words(query)
    opening_room = room // 3
    chosen: set[int] = set()
    used = 0
    for index, passage in enumerate(passages):
        if used + len(passage) > opening_room:
            break
        chosen.add(index)
        used += len(passage) + 2
    ranked = sorted((index for index in range(len(passages)) if index not in chosen),
                    key=lambda index: -len(_words(passages[index]) & asked))
    for index in ranked:
        if used + len(passages[index]) > room:
            continue
        chosen.add(index)
        used += len(passages[index]) + 2
    out: list[str] = []
    left_out = 0
    for index, passage in enumerate(passages):
        if index in chosen:
            if left_out:
                out.append(f"[... {left_out} characters of the file not read here ...]")
                left_out = 0
            out.append(passage)
        else:
            left_out += len(passage) + 2
    if left_out:
        out.append(f"[... {left_out} characters of the file not read here ...]")
    return "\n\n".join(out)


def context_block(attachment_ids: list[str], *, query: str = "", room: int) -> tuple[str, list[tuple[str, str]]]:
    """What the attached files say, for the turn, within ``room`` characters; and each file as (name, what was given).

    The room is shared by size: a short file is given whole, and what it does not use goes to the others."""
    found = [item for item in (kept(attachment_id) for attachment_id in dict.fromkeys(attachment_ids)) if item]
    if not found or room <= 0:
        return "", []
    given: list[tuple[str, str]] = []
    left = room
    for position, (note, text) in enumerate(sorted(found, key=lambda item: len(item[1]))):
        share = left // (len(found) - position)
        if not note.get("readable"):
            body = (f"[Nothing could be read from this file. It is a {note.get('kind')} file of "
                    f"{note.get('size')} bytes{', ' + note['content_type'] if note.get('content_type') else ''}.]")
        else:
            body = fitted(text, share, query)
        left -= len(body)
        given.append((str(note.get("name") or "file"), body))
    parts = ["[The person attached files to this message. What was read from each is below. Where a range is marked "
             "as not read here, it was not read: if the answer would be in it, say so instead of supplying it.]"]
    for name, body in given:
        parts.append(f"=== ATTACHED FILE: {name} ===\n{body}\n=== END {name} ===")
    return "\n\n".join(parts), given


#: The attachments the turn being answered carries. Set by the chat preflight when it reads them, so every path that
#: answers the turn can hand the same files to the model it asks, and none answers as though nothing was attached.
_THIS_TURN: ContextVar[tuple[str, ...]] = ContextVar("aura_attachments_this_turn", default=())

#: How much of the attached files a smaller model is given, in characters. Its room is smaller than her cortex's, and a
#: turn that reaches it is already late.
FALLBACK_ROOM = 8_000


def carry_this_turn(attachment_ids: list[str]) -> None:
    """Record which files the turn being answered carries."""
    _THIS_TURN.set(tuple(str(item) for item in attachment_ids))


def this_turn() -> tuple[str, ...]:
    """The files the turn being answered carries, or none."""
    return _THIS_TURN.get()


#: What every block of a turn's files holds; a model call that already holds it is given nothing twice.
ATTACHED_MARK = "=== ATTACHED FILE: "


def with_this_turns_files(message: str, *, room: int = FALLBACK_ROOM) -> str:
    """The message, preceded by what was read from this turn's files, as the main path gives it.

    LIVE 8 November 2026: Bryan attached the paper about her. The preflight read it into the turn, her cortex could not
    serve the turn, and the smaller model that answered instead was handed his message alone. It said, "I don't see the
    paper attached in this message", about a file the turn held.
    """
    attached = this_turn()
    if not attached:
        return message
    block, _given = context_block(list(attached), query=message, room=room)
    if not block:
        return message
    return f"{block}\nUser message: {message}"


def carried_into_call(prompt: str, messages: list[Any] | None = None, *, room: int = FALLBACK_ROOM
                      ) -> tuple[str, list[Any] | None]:
    """A model call made on this turn, with this turn's files when it does not already hold them.

    LIVE 9 November 2026: the final paper was attached and her cortex was handed it, but the reply failed a check and
    was rewritten by the stabilizer, whose prompt held the person's words alone. The rewrite began "I want to be
    straight with you: I can't see it right", the gate refused it as contradicting what the turn held, and she answered
    "I couldn't get my full attention onto that one". The fix of 8 November gave the files to one path, the fallback
    ladder; four more paths ask a model on the same turn. The gate calls this for every foreground request, so every
    path is given the files, and a path that already holds them is given nothing twice.
    """
    if not this_turn():
        return prompt, messages
    held = [str(prompt or "")]
    held += [str(item.get("content") or "") for item in messages or () if isinstance(item, dict)]
    if any(ATTACHED_MARK in text for text in held):
        return prompt, messages
    if messages:
        users = [index for index, item in enumerate(messages) if isinstance(item, dict) and item.get("role") == "user"]
        if users:
            given = list(messages)
            last = dict(given[users[-1]])
            last["content"] = with_this_turns_files(str(last.get("content") or ""), room=room)
            given[users[-1]] = last
            return prompt, given
    return with_this_turns_files(str(prompt or ""), room=room), messages

