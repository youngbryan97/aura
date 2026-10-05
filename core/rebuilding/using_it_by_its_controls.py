"""Writing something in a program she built and saving it where the person said, with the program's own controls.

"Prove it works by writing a one-page letter in it and exporting it to my
Desktop" is three things: the letter, which is writing and is her model's to
write; using the program to put it on the page and save it, which is a person
at its controls; and the file on the Desktop, which is read back to show it is
there and says what was written. The middle one was handed to her browser
pursuit, a model deciding each click; it is done here the way checks are done
(core/rebuilding/checks_a_person_makes.py): the text typed into the program's
page, and its own save or export command pressed, chosen by code: the kind of
file asked for, else the kind the program itself saves.

Nothing here knows which program it is in. A program with no page to type in,
or no command that saves a file, is not used here, and the caller uses it
another way.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("Rebuilding.UsingIt")

__all__ = ["Used", "a_writing_task", "the_command_that_saves", "write_it_in"]

#: What a task must ask for to be writing something and keeping it.
_WRITES = re.compile(r"\b(writ\w*|typ\w*|compos\w*|draft\w*|put)\b", re.I)
_KEEPS = re.compile(r"\b(export\w*|sav\w*|keep|download\w*|put it|store)\b", re.I)

#: The kinds of file a person names, as the frame's formats name them, and the order a program's own are preferred in.
_KINDS = {"docx": r"\b(docx|word document|word file)\b", "odt": r"\b(odt|opendocument)\b", "rtf": r"\b(rtf|rich text)\b",
          "pdf": r"\bpdf\b", "html": r"\b(html?|web page)\b", "md": r"\b(markdown|\.md)\b", "txt": r"\b(txt|plain text|text file)\b"}
_OWN_FIRST = ("docx", "odt", "rtf", "html", "md", "txt", "pdf")


class _Writing(BaseModel):
    title: str = Field(default="", max_length=120, description="a short name for the document")
    paragraphs: list[str] = Field(default_factory=list, max_length=40, description="the text, one paragraph each, in order")


@dataclass
class Used:
    ok: bool
    files: list[Path] = field(default_factory=list)
    why_not: str = ""
    said: str = ""

    def summary(self) -> str:
        if not self.ok:
            return f"I could not use it as asked: {self.why_not}"
        return f"I wrote it in the program and saved it with the program's own command: {', '.join(str(f) for f in self.files)}, which reads back: \"{self.said[:160]}\"."


def a_writing_task(task: str) -> bool:
    """Whether ``task`` asks for something to be written and kept: what this does."""
    return bool(_WRITES.search(task or "")) and bool(_KEEPS.search(task or ""))


def the_command_that_saves(labels: list[str], task: str) -> str | None:
    """The command a person would press to keep the file ``task`` asks for: the kind it names, else the program's own."""
    keeps = [label for label in labels if re.search(r"\b(sav\w*|export\w*|download\w*)\b", label, re.I) and not re.search(r"\bas\s*\.\.\.$|…$", label)]
    if not keeps:
        return None
    named = [kind for kind, said in _KINDS.items() if re.search(said, task or "", re.I)]
    for kind in named + list(_OWN_FIRST):
        for label in keeps:
            if re.search(_KINDS[kind], label, re.I):
                return label
    return keeps[0]


async def write_it_in(page_file: Path, task: str, folder: Path, ask: Any, *, browser: Any = None) -> Used | None:
    """Do ``task`` in the program at ``page_file`` with its own controls and keep what it saves in ``folder``; None when this program is not used this way."""
    from playwright.async_api import async_playwright

    from core.rebuilding.checks_a_person_makes import (
        _HELPERS,
        Check,
        Step,
        _Doing,
        _what_a_file_says,
    )

    if not a_writing_task(task):
        return None
    await asyncio.to_thread(folder.mkdir, parents=True, exist_ok=True)
    kept: list[Path] = []
    command: str | None = None

    async def keep(download: Any) -> None:
        target = folder / re.sub(r"[/\\:\x00]", "_", download.suggested_filename or "document")
        target = await asyncio.to_thread(_free, target)
        await download.save_as(str(target))
        kept.append(target)

    async with async_playwright() as pw:
        owned = browser is None
        chromium = await pw.chromium.launch() if owned else browser
        try:
            context = await chromium.new_context(accept_downloads=True)
            page = await context.new_page()
            page.on("download", lambda d: asyncio.ensure_future(keep(d)))
            await page.goto(page_file.resolve().as_uri())
            # A program of hers, with a command that keeps a file, before anything is written for it.
            try:
                labels = [str(x) for x in await page.evaluate("[...app.commands.values()].map((c) => c.label)")]
            except Exception:  # noqa: BLE001 - a page that is not one of her programs is used another way
                return None
            command = the_command_that_saves(labels, task)
            if command is None:
                return None
            writing = await ask(f"Write this, as the person asked for it: {task}\nThe text only, in paragraphs; no notes about it.", _Writing, 2048)
            if not isinstance(writing, _Writing) or not any(p.strip() for p in writing.paragraphs):
                return Used(False, why_not="the text to write could not be written")
            text = "\n".join(p.strip() for p in writing.paragraphs if p.strip())
            await page.evaluate(_HELPERS)
            if writing.title:
                await page.evaluate("(name) => { app.name = name; }", writing.title)
            doing = _Doing(page, Check.model_validate({"feature": "use", "steps": [], "expect": [{"see": "text", "target": "x"}]}))
            for step in (Step(do="type", value=text), Step(do="click", target=command)):
                wrong = await doing.step(step)
                if wrong:
                    return Used(False, why_not=f"{step.do} {step.target or 'the text'}: {wrong}")
            for _ in range(40):
                if kept:
                    break
                await asyncio.sleep(0.25)
            await context.close()
        finally:
            if owned:
                await chromium.close()
    if not kept:
        return Used(False, why_not=f'pressing "{command}" saved no file')
    said = await asyncio.to_thread(lambda: _what_a_file_says(kept[0].read_bytes()))
    first = " ".join(text.split()[:6])
    if first.lower() not in " ".join(said.split()).lower():
        return Used(False, kept, why_not=f"{kept[0]} does not hold what was written")
    logger.info("used %s: wrote %d characters and saved %s", page_file, len(text), kept)
    return Used(True, kept, said=" ".join(said.split()))


def _free(wanted: Path) -> Path:
    """``wanted``, or the same name numbered, so nothing already there is written over."""
    target, n = wanted, 1
    while target.exists():
        n += 1
        target = wanted.with_name(f"{wanted.stem} ({n}){wanted.suffix}")
    return target
