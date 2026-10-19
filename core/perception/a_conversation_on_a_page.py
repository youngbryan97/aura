"""A conversation on a page: who said what, what is new from the other side, where to write, and when it ends.

A page that holds a live exchange with someone (a chat with a stranger, a support window, a game played by talking)
is read from what any such page has: a stream of lines that grows as the two sides write, and a place to write in.
Nothing here knows a site. The stream is the part of the page that grows: the element her own line landed in once she
has written one, and before that the one that gained text between two looks. Lines in it are hers when they are
what she sent, and the other side's when they are not. What is outside it is the page itself: its title, its rules,
its clock. The conversation is over when the place to write is gone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["LOOK", "Conversation", "Line", "read_conversation"]

#: What a page is asked for: each element with text of its own, where it is and under which parent; every element's
#: place, so that one added can be told from one whose words changed; and the place to write in, if one is open.
LOOK = r"""() => {
  const pathOf = (el) => { const parts = []; for (let e = el; e && e !== document.body; e = e.parentElement) {
      let i = 1; for (let s = e.previousElementSibling; s; s = s.previousElementSibling) if (s.tagName === e.tagName) i++;
      parts.unshift(e.tagName.toLowerCase() + ":" + i); } return parts.join("/"); };
  const blocks = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT);
  for (let el = walker.currentNode; el; el = walker.nextNode()) {
    if (["SCRIPT", "STYLE", "NOSCRIPT"].includes(el.tagName)) continue;
    const own = Array.from(el.childNodes).filter(n => n.nodeType === 3).map(n => n.textContent).join(" ").replace(/\s+/g, " ").trim();
    if (!own) continue;
    const style = getComputedStyle(el); const r = el.getBoundingClientRect();
    if (style.display === "none" || style.visibility === "hidden" || r.width <= 0 || r.height <= 0) continue;
    blocks.push({text: own, path: pathOf(el), x: r.left, y: r.top, w: r.width, h: r.height,
                 control: !!el.closest("button,a,input,textarea,select,label,[role=button]")});
  }
  // A place to write a line in: not a search box, and alone in its form (a form of several fields is filled, not talked in).
  const open = Array.from(document.querySelectorAll("textarea, input, [contenteditable=true]")).filter(e => {
    const t = (e.getAttribute("type") || "text").toLowerCase(); const r = e.getBoundingClientRect();
    const named = [e.name, e.placeholder, e.getAttribute("aria-label")].join(" ");
    return !e.disabled && !e.readOnly && ["text", ""].includes(t) && r.width > 0 && r.height > 0
      && getComputedStyle(e).visibility !== "hidden" && !e.closest("[role=search]") && !/search/i.test(named)
      && !(e.form && e.form.querySelectorAll("input:not([type=hidden]):not([type=submit]):not([type=button]), textarea, select").length > 1); });
  const box = open.find(e => e.tagName !== "INPUT") || open[0];
  const at = box ? box.getBoundingClientRect() : null;
  const elements = Array.from(document.body.querySelectorAll("*")).map(pathOf);
  return {blocks, elements, writing: at ? {x: at.left, y: at.top, w: at.width, h: at.height, path: pathOf(box),
          says: box.placeholder || box.getAttribute("aria-label") || ""} : null, width: innerWidth, height: innerHeight};
}"""


@dataclass(frozen=True)
class Line:
    """One line of the conversation: who wrote it (``me`` or ``them``) and what it says."""

    who: str
    text: str


@dataclass
class Conversation:
    """The conversation as one look found it."""

    lines: list[Line] = field(default_factory=list)
    stream: str = ""
    writing: dict[str, Any] | None = None
    page_text: list[str] = field(default_factory=list)

    @property
    def over(self) -> bool:
        """No place to write is left: the exchange has ended, whatever the page says next."""
        return self.writing is None

    def theirs(self) -> list[str]:
        return [line.text for line in self.lines if line.who == "them"]


def _normal(text: str) -> str:
    return " ".join(str(text).lower().split())


def _under(path: str, ancestor: str) -> bool:
    return bool(ancestor) and (path == ancestor or path.startswith(ancestor + "/"))


def _parent(path: str) -> str:
    return path.rsplit("/", 1)[0] if "/" in path else ""


def _stream_from_her_line(blocks: list[dict[str, Any]], sent: set[str]) -> str:
    """The nearest element holding her line and at least one other line: the stream they are both in."""
    for block in blocks:
        if block["control"] or _normal(block["text"]) not in sent:
            continue
        ancestor = _parent(block["path"])
        while ancestor:
            inside = [b for b in blocks if not b["control"] and _under(b["path"], ancestor)]
            if len(inside) >= 2:
                return ancestor
            ancestor = _parent(ancestor)
    return ""


def _stream_that_grew(blocks: list[dict[str, Any]], before: dict[str, Any]) -> str:
    """The element a new element with words was added to between two looks, before she has written anything. A line
    is added; a clock or a counter changes its words in place, in an element that was there before (empty, it may
    not have been a block then)."""
    there = set(before.get("elements") or ())
    new = [b for b in blocks if not b["control"] and b["path"] not in there]
    return _parent(new[0]["path"]) if new and there else ""


def read_conversation(look: dict[str, Any], sent: list[str], *, before: dict[str, Any] | None = None,
                      stream: str = "") -> Conversation:
    """The conversation in one look at a page (``LOOK``'s answer), given what she has sent there and the look before.

    ``stream``, once found, is kept by the caller and given back: a stream does not move between looks."""
    blocks = list(look.get("blocks") or [])
    mine = {_normal(text) for text in sent if text.strip()}
    stream = stream or _stream_from_her_line(blocks, mine) or (_stream_that_grew(blocks, before) if before else "")
    lines, page = [], []
    for block in blocks:
        if stream and not block["control"] and _under(block["path"], stream):
            lines.append(Line("me" if _normal(block["text"]) in mine else "them", block["text"]))
        elif not block["control"]:
            page.append(block["text"])
    return Conversation(lines=lines, stream=stream, writing=look.get("writing"), page_text=page)
