"""Build a structured document from a plan, and render it however it is wanted.

LIVE, 2026-08-22. Asked for a six-slide deck for a funding panel, she wrote
two slides and stopped, and the log shows why she stopped writing prose: she
tried to call a tool.

    Tools offered (code_repl,diagnose_repo,quantum_lab) and none called;
    model produced: <tool_call> {"name": "create_slides", "arguments":
    {"slide_contents": [{"title": "What I Am", ...

There is no create_slides. She reached for the capability the task needed,
named it correctly, and nothing was there.

A deck is not its own kind of thing, though. It is a document — a title and a
sequence of sections, each with a heading and some lines — rendered as one
section per screen. A report is the same document rendered as one flowing
page, and a memo is the same again. So the format is a parameter here rather
than a module: the model decides what the sections say, and the runtime lays
them out, checks the result and writes it.

Nothing here knows what any particular document is about, and adding a third
rendering means adding a renderer, not a capability.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

__all__ = [
    "Section",
    "Document",
    "document_from_plan",
    "plan_from_json",
    "render_document",
    "check_document",
    "document_schema",
    "RENDERERS",
]

#: A document longer than this is a book, not a deliverable.
_MAX_SECTIONS = 40

#: Lines past this stop being read.
_MAX_LINES = 8


@dataclass(frozen=True, slots=True)
class Section:
    title: str
    lines: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True, slots=True)
class Document:
    title: str
    sections: tuple[Section, ...] = ()
    subtitle: str = ""
    repairs: tuple[str, ...] = field(default_factory=tuple)

    def problems(self) -> tuple[str, ...]:
        found: list[str] = []
        if not self.title.strip():
            found.append("the document has no title")
        if not self.sections:
            found.append("the document has no sections")
        for index, section in enumerate(self.sections, start=1):
            if not section.title.strip() and not section.lines:
                found.append(f"section {index} is empty")
        return tuple(found)


def document_schema() -> dict:
    """The shape of a document, as a schema for a typed call."""
    return {
        "type": "object",
        "required": ["title", "sections"],
        "properties": {
            "title": {"type": "string"},
            "subtitle": {"type": "string"},
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["title"],
                    "properties": {
                        "title": {"type": "string"},
                        "lines": {"type": "array", "items": {"type": "string"}},
                        "notes": {"type": "string", "description": "Notes not shown in the body."},
                    },
                },
            },
        },
    }


def _text(value: object, limit: int = 200) -> str:
    return " ".join(str(value or "").split())[:limit]


def _lines(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        parts = [part for part in re.split(r"\n|(?<=[.;])\s+", value) if part.strip()]
    elif isinstance(value, (list, tuple)):
        parts = [str(item) for item in value]
    else:
        return ()
    cleaned = [_text(part, 240).lstrip("-•* ") for part in parts]
    return tuple(part for part in cleaned if part)[:_MAX_LINES]


def document_from_plan(plan: dict, request: str = "") -> Document:
    """A document from whatever the plan contains, repaired where it is thin."""
    raw = dict(plan or {})
    repairs: list[str] = []
    sections: list[Section] = []
    # `sections` is the declared field. `slides` and `slide_contents` are read
    # too: the second is the name the model used when it reached for a tool
    # that did not exist, and a plan should not fail for calling a section a
    # slide.
    entries = raw.get("sections") or raw.get("slides") or raw.get("slide_contents") or raw.get("content") or []
    for index, item in enumerate(list(entries)[:_MAX_SECTIONS], start=1):
        if isinstance(item, str):
            sections.append(Section(title=_text(item, 120)))
            continue
        item = dict(item or {})
        title = _text(item.get("title") or item.get("heading") or "", 120)
        # `lines` is the declared field; the rest are what a plan may call it.
        lines = _lines(
            item.get("lines")
            or item.get("bullets")
            or item.get("content")
            or item.get("points")
            or item.get("body")
        )
        if not title and lines:
            title, lines = lines[0], lines[1:]
            repairs.append(f"section {index} had no title, so its first line became one")
        if not title and not lines:
            repairs.append(f"dropped section {index}, which was empty")
            continue
        sections.append(Section(title=title, lines=lines, notes=_text(item.get("notes"), 600)))

    title = _text(raw.get("title") or "", 120)
    if not title:
        words = [word for word in re.findall(r"[A-Za-z][A-Za-z'-]*", request) if len(word) > 2]
        title = " ".join(words[:6]).capitalize() or "Document"
        repairs.append("the plan had no title, so one was taken from the request")
    return Document(
        title=title,
        subtitle=_text(raw.get("subtitle") or "", 160),
        sections=tuple(sections),
        repairs=tuple(dict.fromkeys(repairs)),
    )


def plan_from_json(text: str, request: str = "") -> Document | None:
    """Read a document out of whatever came back, or None."""
    raw = str(text or "").strip()
    if not raw:
        return None
    for candidate in sorted(_objects(raw), key=len, reverse=True):
        try:
            loaded = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if not isinstance(loaded, dict):
            continue
        # A tool call carries the document inside `arguments`.
        payload = loaded.get("arguments") if isinstance(loaded.get("arguments"), dict) else loaded
        if payload.get("sections") or payload.get("slides") or payload.get("slide_contents"):
            document = document_from_plan(payload, request)
            if not document.problems():
                return document
    return None


def _objects(text: str) -> list[str]:
    found: list[str] = []
    for start, character in enumerate(text):
        if character != "{":
            continue
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    found.append(text[start : end + 1])
                    break
    return found


def _escape(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


#: Shared look. Both renderings read the same tokens, so a document does not
#: change character when it changes shape.
_BASE_STYLE = """
:root { color-scheme: light dark;
  --ink:#14171a; --paper:#fbfbfa; --line:#d8d6d1; --quiet:#6b6f76; --accent:#2f5d50; }
@media (prefers-color-scheme: dark) {
  :root { --ink:#ecebe8; --paper:#16181a; --line:#33373b; --quiet:#9aa0a6; --accent:#7fbfa8; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--paper); color:var(--ink);
  font:16px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif; }
h1 { font-weight:600; letter-spacing:-0.02em; }
h2 { font-weight:600; letter-spacing:-0.01em; }
.sub { color:var(--quiet); }
ul { margin:0; padding:0; list-style:none; }
li { position:relative; padding-left:1.4rem; }
li::before { content:""; position:absolute; left:0; top:0.62em; width:0.5rem; height:1px;
  background:var(--accent); }
.notes { display:none; }
"""

_DECK_STYLE = _BASE_STYLE + """
.section { min-height:100vh; padding:9vh 8vw; display:none; flex-direction:column;
  justify-content:center; }
.section.on { display:flex; }
h1 { font-size:clamp(1.8rem,4vw,3rem); margin:0 0 2rem; }
h2 { font-size:clamp(1.4rem,3vw,2.2rem); margin:0 0 1.6rem; }
.sub { font-size:1.1rem; margin-top:-1.2rem; margin-bottom:2rem; }
ul { max-width:52rem; }
li { margin-bottom:1rem; font-size:clamp(1rem,1.6vw,1.25rem); }
.bar { position:fixed; left:0; bottom:0; height:2px; background:var(--accent); transition:width .2s; }
.count { position:fixed; right:1.5rem; bottom:1.2rem; color:var(--quiet); font-size:0.8rem;
  font-variant-numeric:tabular-nums; }
@media print { .section { display:flex; page-break-after:always; min-height:auto; }
  .bar,.count { display:none; } }
"""

_PAGE_STYLE = _BASE_STYLE + """
body { display:flex; justify-content:center; padding:3.5rem 1.25rem; }
main { width:100%; max-width:44rem; }
h1 { font-size:1.9rem; margin:0 0 0.5rem; }
h2 { font-size:1.15rem; margin:2.4rem 0 0.8rem; }
.sub { margin:0 0 2.5rem; }
.section { border-top:1px solid var(--line); padding-top:0.4rem; }
.section:first-of-type { border-top:0; }
li { margin-bottom:0.6rem; }
"""

#: How the deck moves. Nothing here knows what the sections say.
_DECK_SCRIPT = """
const sections = Array.from(document.querySelectorAll('.section'));
const bar = document.querySelector('.bar');
const count = document.querySelector('.count');
let at = 0;
function show(next) {
  at = Math.max(0, Math.min(sections.length - 1, next));
  sections.forEach((section, index) => section.classList.toggle('on', index === at));
  bar.style.width = ((at + 1) / sections.length * 100).toFixed(2) + '%';
  count.textContent = (at + 1) + ' / ' + sections.length;
}
document.addEventListener('keydown', (event) => {
  if (['ArrowRight', 'PageDown', ' '].includes(event.key)) { event.preventDefault(); show(at + 1); }
  if (['ArrowLeft', 'PageUp'].includes(event.key)) { event.preventDefault(); show(at - 1); }
  if (event.key === 'Home') show(0);
  if (event.key === 'End') show(sections.length - 1);
});
document.addEventListener('click', (event) =>
  show(at + (event.clientX < window.innerWidth / 3 ? -1 : 1)));
show(0);
"""


def _section_html(section: Section, *, level: str) -> str:
    body = [f'<section class="section"><{level}>{_escape(section.title)}</{level}>']
    if section.lines:
        body.append("<ul>")
        body.extend(f"<li>{_escape(line)}</li>" for line in section.lines)
        body.append("</ul>")
    if section.notes:
        body.append(f'<div class="notes">{_escape(section.notes)}</div>')
    body.append("</section>")
    return "".join(body)


def _shell(title: str, style: str, body: str, script: str = "") -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape(title)}</title>
<style>{style}</style>
</head>
<body>
{body}
{f'<script>{script}</script>' if script else ''}
</body>
</html>
"""


def _render_deck(document: Document) -> str:
    """One section per screen, moved through with the keyboard."""
    cover = [f'<section class="section on"><h1>{_escape(document.title)}</h1>']
    if document.subtitle:
        cover.append(f'<div class="sub">{_escape(document.subtitle)}</div>')
    cover.append("</section>")
    parts = ["".join(cover)]
    parts.extend(_section_html(section, level="h2") for section in document.sections)
    total = len(parts)
    body = (
        "".join(parts)
        + f'<div class="bar" style="width:{100.0 / total:.2f}%"></div>'
        + f'<div class="count">1 / {total}</div>'
    )
    return _shell(document.title, _DECK_STYLE, body, _DECK_SCRIPT)


def _render_page(document: Document) -> str:
    """One flowing page, for a report or a memo."""
    head = [f"<main><h1>{_escape(document.title)}</h1>"]
    if document.subtitle:
        head.append(f'<div class="sub">{_escape(document.subtitle)}</div>')
    head.extend(_section_html(section, level="h2") for section in document.sections)
    head.append("</main>")
    return _shell(document.title, _PAGE_STYLE, "".join(head))


#: The renderings. A third is a function here, not a capability.
RENDERERS = {"deck": _render_deck, "page": _render_page}


def render_document(document: Document, *, form: str = "deck") -> str:
    """The whole document as one file. Raises ValueError if it cannot render."""
    problems = document.problems()
    if problems:
        raise ValueError("; ".join(problems))
    renderer = RENDERERS.get(str(form or "deck").strip().lower())
    if renderer is None:
        raise ValueError(f"no renderer named {form!r}; have {', '.join(sorted(RENDERERS))}")
    return renderer(document)


@dataclass(frozen=True, slots=True)
class DocumentCheck:
    ok: bool
    sections: int = 0
    checks: tuple[str, ...] = ()
    problems: tuple[str, ...] = ()


def check_document(
    document: Document, html: str, *, wanted: int = 0, form: str = "deck"
) -> DocumentCheck:
    """Everything checkable about a built document, whatever it was rendered as."""
    from html.parser import HTMLParser

    class Counter(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.sections = 0
            self.title_seen = False
            self._in_title = False
            self.title = ""

        def handle_starttag(self, tag, attrs):
            found = {key: (value or "") for key, value in attrs}
            if tag == "section" and "section" in found.get("class", ""):
                self.sections += 1
            if tag == "title":
                self._in_title = True

        def handle_endtag(self, tag):
            if tag == "title":
                self._in_title = False

        def handle_data(self, data):
            if self._in_title:
                self.title += data
                self.title_seen = True

    checks: list[str] = []
    problems: list[str] = list(document.problems())
    counter = Counter()
    try:
        counter.feed(html)
    except (ValueError, AssertionError) as exc:
        return DocumentCheck(False, 0, (), (f"the document did not parse: {exc}",))

    if not counter.title_seen or not counter.title.strip():
        problems.append("the document has no title")
    else:
        checks.append("the document parses and is titled")

    # A deck carries a cover section; a page puts the title in the heading.
    expected = len(document.sections) + (1 if str(form).lower() == "deck" else 0)
    if counter.sections != expected:
        problems.append(f"{counter.sections} sections in the file, {expected} in the plan")
    else:
        checks.append(f"{len(document.sections)} section(s) laid out")

    for outside in ("http://", "https://", "<link", "cdn."):
        if outside in html.lower():
            problems.append(f"the document reaches for {outside}")
    if not problems:
        checks.append("nothing is loaded from the network")

    if wanted and len(document.sections) != wanted:
        problems.append(
            f"{wanted} were asked for and {len(document.sections)} were written"
        )
    elif wanted:
        checks.append(f"{wanted} asked for, {wanted} written")

    return DocumentCheck(
        not problems, len(document.sections), tuple(checks), tuple(dict.fromkeys(problems))
    )
