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

from core.language.the_form_of_a_kind import fitted, the_form_for

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


async def write_it_in(page_file: Path, task: str, folder: Path, ask: Any, *, browser: Any = None, visible: bool = False,
                      tell: Any = None) -> Used | None:
    """Do ``task`` in the program at ``page_file`` with its own controls and keep what it saves in ``folder``; None when this program is not used this way.

    ``visible``: in a window of its own, typed at a person's pace, for a
    person watching it done, and said as it goes (``tell``).
    """
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
        chromium = await pw.chromium.launch(headless=not visible) if owned else browser
        try:
            context = await chromium.new_context(accept_downloads=True, **({"viewport": {"width": 1280, "height": 860}} if visible else {}))
            page = await context.new_page()
            page.on("download", lambda d: asyncio.ensure_future(keep(d)))
            await page.goto(await asyncio.to_thread(lambda: page_file.resolve().as_uri()))
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
            if visible:
                await page.bring_to_front()
                await _said(tell, f"Writing it in {await page.title() or 'the program'}, in a window you can watch; then \"{command}\" saves it.")
            # The caret where writing goes on, in the page's own surface, and the page settled, before the first key: LIVE
            # 2026-10-06 the first keys typed in a window just brought to the front landed out of place ("Der Future Self").
            await page.evaluate(_AT_THE_END)
            await page.wait_for_timeout(600 if visible else 50)
            doing = _Doing(page, Check.model_validate({"feature": "use", "steps": [], "expect": [{"see": "text", "target": "x"}]}),
                           typing_ms=14 if visible else 4)
            # Set as its kind is set (core/language/the_form_of_a_kind.py): a letter's greeting and sign-off on lines of their own.
            form = the_form_for(task)
            fit = fitted(writing.paragraphs, form)
            if form is not None and fit.missing:
                await _said(tell, f"A {form.kind} usually has its {_and(fit.missing)}; this one does not, and I have not made one up.")
            for step in (*_typed(fit.paragraphs), Step(do="click", target=command)):
                wrong = await doing.step(step)
                if wrong:
                    return Used(False, why_not=f"{step.do} {step.target or 'the text'}: {wrong}")
            for _ in range(40):
                if kept:
                    break
                await asyncio.sleep(0.25)
            if visible:
                await asyncio.sleep(2.0)  # what was written, left in view a moment after it is saved
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


#: Focus a program's writing surface and put the caret at the end of what it holds.
_AT_THE_END = """() => {
  const s = (window.app && app.editing && app.editing.surface && app.editing.surface())
    || document.querySelector("[contenteditable=true], [contenteditable=''], textarea");
  if (!s) return false;
  s.focus();
  if (s.isContentEditable) {
    const r = document.createRange(); r.selectNodeContents(s); r.collapse(false);
    const sel = getSelection(); sel.removeAllRanges(); sel.addRange(r);
  } else if (typeof s.setSelectionRange === "function") { s.setSelectionRange(s.value.length, s.value.length); }
  return true;
}"""


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1] if items else ""


def _typed(paragraphs: list[str]) -> list[Any]:
    """Typing the paragraphs as a person does: a paragraph at a time, Enter between, a long one in pieces at its spaces."""
    from core.rebuilding.checks_a_person_makes import Step

    steps: list[Any] = []
    for n, paragraph in enumerate(p.strip() for p in paragraphs if p.strip()):
        if n:
            steps.append(Step(do="press", value="Enter"))
        piece = ""
        for word in paragraph.split(" "):
            if piece and len(piece) + 1 + len(word) > 500:
                steps.append(Step(do="type", value=piece + " "))
                piece = word
            else:
                piece = f"{piece} {word}" if piece else word
        if piece:
            steps.append(Step(do="type", value=piece))
    return steps


async def _said(tell: Any, line: str) -> None:
    if tell is None:
        return
    said = tell(line)
    if asyncio.iscoroutine(said):
        await said


def _free(wanted: Path) -> Path:
    """``wanted``, or the same name numbered, so nothing already there is written over."""
    target, n = wanted, 1
    while target.exists():
        n += 1
        target = wanted.with_name(f"{wanted.stem} ({n}){wanted.suffix}")
    return target
