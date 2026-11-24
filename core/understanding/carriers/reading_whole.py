"""A document read whole, and an answer from all of it.

A document longer than one reading can hold is read in sections, in order, to its end. Each section's reading says what
it states, with a quotation for each that has to be in the section, and what a careful reader would question there. What
has been read is kept as ranges of the document, and a section that could not be read is tried again in halves, so what
is told of it can say how much was read, in pages, and never more than was.

The readings are kept by the document's own hash: asked again about the same document, she has read it. The answer is
written from all of them with the parts it turns on open beside it, and what it quotes is checked against the document.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.understanding.carriers.finding import quoted_in
from core.understanding.carrying_out import CameTo, Tell
from core.understanding.requests_of_her import Part, Request

logger = logging.getLogger("Aura.Understanding.Reading")

__all__ = ["Section", "SectionRead", "read_whole", "reading_of", "sections_of", "unread"]

#: How much of a document one reading holds, in characters, and the least a section is halved to.
A_SECTION, LEAST_SECTION = 9000, 1500
#: How much of the readings and of the document's own text the answer is written beside.
NOTES_ROOM, OPEN_ROOM = 26000, 12000


class Point(BaseModel):
    says: str = Field(description="what the document states here")
    quote: str = Field("", description="its words for it, copied exactly")


class SectionRead(BaseModel):
    about: str = Field(description="what this section does, in a sentence")
    points: list[Point] = Field(default_factory=list, description="what it states that the whole turns on")
    questions: list[Point] = Field(default_factory=list, description="what a careful reader would question, or could not follow, each with the words it is about")
    strong: list[Point] = Field(default_factory=list, description="what is done well here, each with the words that show it")


class Answer(BaseModel):
    answer: str = Field(description="the answer, whole, as it is to be told")
    quotes: list[str] = Field(default_factory=list, description="every passage it quotes, copied exactly")


@dataclass
class Section:
    start: int
    end: int
    heading: str = ""
    read: dict[str, Any] = field(default_factory=dict)


def _breaks(text: str) -> list[int]:
    """Where a document can be cut: between its paragraphs."""
    return sorted({match.end() for match in re.finditer(r"\n\s*\n", text)} | {len(text)})


def sections_of(text: str, *, size: int = A_SECTION) -> list[Section]:
    """The document in order, cut between paragraphs, none longer than ``size`` unless one paragraph is; all of it."""
    sections: list[Section] = []
    breaks, start = _breaks(text), 0
    while start < len(text):
        within = [at for at in breaks if start < at <= start + size]
        end = within[-1] if within else min(len(text), start + size)
        if text[start:end].strip():
            first = next((line.strip() for line in text[start:end].splitlines() if line.strip()), "")
            sections.append(Section(start, end, first[:90]))
        start = end
    return sections


def unread(text: str, sections: list[Section]) -> int:
    """How many characters of the document no reading covers."""
    covered = sum(section.end - section.start for section in sections if section.read)
    blank = sum(len(text[a:b]) for a, b in _gaps(text, sections) if not text[a:b].strip())
    return max(0, len(text) - covered - blank)


def _gaps(text: str, sections: list[Section]) -> list[tuple[int, int]]:
    gaps, at = [], 0
    for section in sorted(sections, key=lambda s: s.start):
        if section.start > at:
            gaps.append((at, section.start))
        at = max(at, section.end)
    if at < len(text):
        gaps.append((at, len(text)))
    return gaps


async def _ask(prompt: str, schema: Any, max_tokens: int) -> Any:
    from core.rebuilding.her_model import ask_her_model

    return await ask_her_model(prompt, schema, max_tokens, origin="work_asked_for")


def _where(digest: str) -> Path:
    from core.runtime.state_ownership import state_root

    return state_root() / "data" / "understanding" / "read" / f"{digest}.json"


def _kept_reading(digest: str) -> list[Section]:
    path = _where(digest)
    if not path.is_file():
        return []
    try:
        return [Section(**row) for row in json.loads(path.read_text(encoding="utf-8"))]
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("A reading kept at %s could not be read: %s", path, exc)
        return []


async def _keep(digest: str, sections: list[Section]) -> None:
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    with local_internal_governed_scope("understanding.reading", domain="file_write"):
        await get_file_write_gateway().write_text_async(
            _where(digest), json.dumps([asdict(s) for s in sections], ensure_ascii=False), source="understanding.reading")


def _held_to_the_text(read: SectionRead, text: str) -> dict[str, Any]:
    """A section's reading with every quotation that is not in the section left out."""
    held = read.model_dump()
    for key in ("points", "questions", "strong"):
        for point in held[key]:
            if not quoted_in(point["quote"], text):
                point["quote"] = ""
        held[key] = [p for p in held[key] if p["says"].strip() and (p["quote"] or key == "points")]
    return held


async def _read_a_section(name: str, text: str, section: Section, n: int, of: int, before: str, ask: Any) -> list[Section]:
    """A section read; in halves where it could not be read whole, down to the least a half is."""
    part = text[section.start:section.end]
    read = await ask(f"{name}, part {n} of {of}. What came before it: {before or 'nothing; it begins here'}\n\n{part}\n\n"
                     "What does this part state, what is done well, and what would a careful reader question?",
                     SectionRead, 1400)
    if isinstance(read, SectionRead):
        section.read = _held_to_the_text(read, part)
        return [section]
    if len(part) <= LEAST_SECTION * 2:
        return [section]
    middle = section.start + len(part) // 2
    cut = min((at for at in _breaks(text) if section.start < at < section.end), key=lambda at: abs(at - middle), default=middle)
    halves = [Section(section.start, cut, section.heading), Section(cut, section.end, section.heading)]
    out: list[Section] = []
    for half in halves:
        out += await _read_a_section(name, text, half, n, of, before, ask)
    return out


async def reading_of(name: str, text: str, tell: Tell, *, ask: Any) -> list[Section]:
    """The document read, section by section, what was read before taken up."""
    digest = hashlib.sha256(text.encode()).hexdigest()[:32]
    sections = await asyncio.to_thread(_kept_reading, digest) or sections_of(text)
    done: list[Section] = []
    before = ""
    for n, section in enumerate(sections, 1):
        if not section.read:
            tell(f"Reading {name}, part {n} of {len(sections)}.")
            parts = await _read_a_section(name, text, section, n, len(sections), before, ask)
        else:
            parts = [section]
        done += parts
        before = " ".join(filter(None, (p.read.get("about", "") for p in done[-4:])))[-600:]
        await _keep(digest, done + sections[n:])
    return done


def _notes(sections: list[Section], *, room: int = NOTES_ROOM) -> str:
    """The readings in order, evenly cut where together they are longer than the room."""
    whole = []
    for n, section in enumerate(sections, 1):
        read = section.read
        if not read:
            whole.append(f"[{n}] (not read) {section.heading}")
            continue
        lines = [f"[{n}] {read['about']}"]
        for key, mark in (("points", "states"), ("strong", "strong"), ("questions", "question")):
            lines += [f"  {mark}: {p['says']}" + (f" “{p['quote']}”" if p["quote"] else "") for p in read[key]]
        whole.append("\n".join(lines))
    each = max(200, room // max(1, len(whole)))
    return "\n".join(note if sum(map(len, whole)) <= room else note[:each] for note in whole)


def _open_beside(asked: str, text: str, sections: list[Section], *, room: int = OPEN_ROOM) -> str:
    """The parts of the document nearest what was asked, by her encoder where it is ready; its ends where it is not."""
    from core.cognition.evidence_relevance import vectors_if_ready

    if len(text) <= room:
        return text
    passages = [p for s in sections for p in re.split(r"\n\s*\n", text[s.start:s.end]) if len(p.split()) > 12]
    vectors = vectors_if_ready([asked, *(p[:1200] for p in passages)]) if passages else None
    if vectors is None or len(vectors) != len(passages) + 1:
        return f"{text[: room // 2]}\n…\n{text[-room // 2:]}"
    import numpy as np

    v = np.asarray(vectors, dtype=np.float32)
    near = v[1:] @ v[0] / (np.linalg.norm(v[1:], axis=1) * np.linalg.norm(v[0]) + 1e-9)
    picked, left = [], room
    for i in np.argsort(-near):
        if len(passages[i]) > left:
            continue
        picked.append(int(i))
        left -= len(passages[i])
        if left < 400:
            break
    return "\n…\n".join(passages[i] for i in sorted(picked))


def _pages(text: str, sections: list[Section]) -> str:
    pages = len(re.findall(r"^\[page \d+\]$", text, flags=re.M))  # as a PDF's text is kept (core/conversation/attachments.py)
    missing = unread(text, sections)
    amount = f"all {pages} pages" if pages > 1 else f"all {len(text.split()):,} words"
    if missing <= 0:
        return f"{amount} of it, in {len(sections)} parts"
    return f"{100 - round(100 * missing / max(1, len(text)))}% of it; {len([s for s in sections if not s.read])} parts could not be read"


def _quotes_not_there(answer: str, quotes: list[str], text: str) -> list[str]:
    inside = re.findall(r"[“\"]([^”\"]{20,400})[”\"]", answer)
    return [q for q in dict.fromkeys([*quotes, *inside]) if q.strip() and not quoted_in(q.strip(" .…"), text)]


async def read_whole(request: Request, tell: Tell, *, ask: Any = None) -> CameTo:
    """Read the documents a request is about, to their ends, and answer it from all of them."""
    from core.rebuilding.her_model import patiently
    from core.understanding.carriers.finding import _materials

    ask = patiently(ask or _ask, tell)
    documents = await asyncio.to_thread(_materials, request)
    if not documents:
        return CameTo(how="looked for the document", again=False, came_to="no document came with the request",
                      parts=[Part("the document is here", "unmet", "nothing readable came with the request")])
    parts: list[Part] = []
    notes, beside = [], []
    for name, text in documents:
        sections = await reading_of(name, text, tell, ask=ask)
        missing = unread(text, sections)
        parts.append(Part(f"{name} was read whole", "met" if missing <= 0 else "unmet", _pages(text, sections)))
        notes.append(f"{name}:\n{_notes(sections, room=NOTES_ROOM // len(documents))}")
        opened = await asyncio.to_thread(_open_beside, request.whole, text, sections, room=OPEN_ROOM // len(documents))
        beside.append(f"{name}:\n{opened}")
    everything = "\n".join(text for _name, text in documents)
    answered = await _answer(request, notes, beside, everything, ask)
    parts.append(Part("what was asked of it is answered from it", "met" if answered is not None else "unmet",
                      "every passage it quotes is in the document" if answered is not None else "no answer could be written"))
    return CameTo(how="read in order to its end, each part's reading held to its words", parts=parts,
                  made={"told": answered or ""}, came_to="; ".join(part.shown_by for part in parts))


async def _answer(request: Request, notes: list[str], beside: list[str], everything: str, ask: Any) -> str | None:
    """The answer, written from the readings; written again where it quotes what the document does not say."""
    asked = request.whole
    base = ("Your readings of it, part by part:\n\n" + "\n\n".join(notes) + "\n\nPassages of it:\n\n" + "\n\n".join(beside)
            + f"\n\nThey asked: {asked}")
    strange: list[str] = []
    for _try in range(2):
        told = f"\n\nThese are not in the document and cannot be quoted from it: {strange}" if strange else ""
        read = await ask(base + told, Answer, 2600)
        if not isinstance(read, Answer) or not read.answer.strip():
            continue
        strange = _quotes_not_there(read.answer, read.quotes, everything)
        if not strange:
            return read.answer.strip()
    if strange and isinstance(read, Answer):
        # Its quotations are the document's or they are not quotations: the marks come off what is not there.
        answer = read.answer
        for quote in strange:
            answer = re.sub(r"[“\"]" + re.escape(quote) + r"[”\"]", quote, answer)
        return answer.strip()
    return None
