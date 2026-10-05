"""The program a request names, read by code: its capitalised name, else a run of its words that is the title of an article about a program.

"Do a clean-room reconstruction of Microsoft Word" names it in capitals. "can
you rebuild microsoft paint for me" and "make me my own version of the
spreadsheet program excel" do not, and a person still knows which program is
meant because they know the programs. So does her corpus: a run of the
request's words that is the title of an article whose opening says it is a
program (software, an application, an editor, a game...) is the program. The
longest such run wins ("Microsoft Word" over "Word"). Nothing is asked of her
model, and it works offline.
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("Rebuilding.WhichProgram")

__all__ = ["the_program_named_in"]

#: "a clean-room reconstruction of Microsoft Word", "rebuild Paint", "clone of Trello"
_NAMED_AFTER = re.compile(
    r"(?:reconstruct(?:ion)?|rebuild|recreate|re-create|clone|reimplement(?:ation)?|copy|version)\s+(?:of\s+)?(?:the\s+)?"
    r"((?:[A-Z][\w+.#'-]*)(?:\s+(?:[A-Z0-9][\w+.#'-]*))*)"
)

#: What an article's opening says a program is.
_A_PROGRAM = re.compile(
    r"\b(software|program|programme|application|app|editor|processor|spreadsheet|game|video game|browser|client|"
    r"suite|player|database|utility|tool|service|platform|engine|environment)\b", re.I)

#: Words a program's name does not begin or end with.
_NOT_A_NAME_EDGE = frozenset(
    "a an the of for to in on at by with and or my me you your our it its this that from as is be can could would "
    "please do make build rebuild recreate clone copy version reconstruction reconstruct clean room clean-room own "
    "program programs app application software like same one new working polished complete full".split())

#: The words after which what follows is the thing to be remade.
_REMADE = frozenset("of like rebuild recreate redo remake clone copy reconstruct reconstruction version reimplement reimplementation".split())

#: The longest run of words tried as a name.
LONGEST_NAME = 4


def the_program_named_in(asked: str, corpus: Any = None) -> str:
    """The program ``asked`` names, by its capitalised name or as her corpus knows it; empty when it names none."""
    found = _NAMED_AFTER.search(asked or "")
    if found:
        return found.group(1).strip(" .,;:")
    words = re.findall(r"[A-Za-z0-9][\w+.#'-]*", asked or "")
    if not words:
        return ""
    if corpus is None:
        try:
            from core.knowledge.local_corpus import get_local_corpus_store

            corpus = get_local_corpus_store()
        except Exception as why:  # noqa: BLE001 - no corpus here is no knowing names by it
            logger.info("no corpus to know program names by: %s", why)
            corpus = None
    if corpus is None:
        return ""
    # Longest runs first. A run that is the name of a kind of program ("word
    # processor", "spreadsheet") is not one program, and its words are not
    # tried again alone: "build me a word processor" names no program.
    used: set[int] = set()
    for size in range(min(LONGEST_NAME, len(words)), 0, -1):
        for at in range(len(words) - size + 1):
            run = words[at : at + size]
            if used & set(range(at, at + size)):
                continue
            if run[0].lower() in _NOT_A_NAME_EDGE or run[-1].lower() in _NOT_A_NAME_EDGE or (size == 1 and len(run[0]) < 3):
                continue
            # A word's other meanings are looked through only for the thing asked to be remade ("a copy of
            # notepad", "like word"), never for the request's own words ("write me a letter").
            title, kind = _known_program(corpus, " ".join(run), meanings=any(w.lower() in _REMADE for w in words[:at]))
            if title and not kind:
                return title
            if kind:
                used.update(range(at, at + size))
    return ""


def _defining_sentence(corpus: Any, hit: Any) -> str:
    """The sentence an article says what its subject is in ("X is a ..."), past captions and markup before it."""
    try:
        body = str(corpus.body(hit.doc_id, max_chars=3000) or "")
    except Exception:  # noqa: BLE001 - an article that cannot be read says nothing of what it is
        return ""
    named = {w for w in re.findall(r"[a-z0-9+#]+", str(hit.title).lower()) if len(w) > 1}
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", body):
        said = re.sub(r"^.*\]\]\s*", "", sentence).strip()  # what an image's caption left before it
        if re.search(r"\b(?:is|was|are)\s+(?:an?|the)\b", said) and named & set(re.findall(r"[a-z0-9+#]+", said.lower())):
            return said
    return ""


def _by_title(corpus: Any, title: str) -> tuple[str, bool]:
    """(the title of the article ``title`` names, when it is about one program; whether it is about a kind of program instead)."""
    for spelling in dict.fromkeys((title, title[:1].upper() + title[1:], " ".join(w[:1].upper() + w[1:] for w in title.split()), title.title())):
        try:
            hit = corpus.by_title(spelling)
        except Exception:  # noqa: BLE001 - a lookup that fails is a name not known
            hit = None
        if hit is None:
            continue
        said = _defining_sentence(corpus, hit)
        if not _A_PROGRAM.search(said):
            return "", False
        # One program's article begins with its name; a kind's with an article: "A spreadsheet is ...".
        return ("", True) if re.match(r"(?:an?|the)\s", said, re.I) else (str(hit.title), False)
    return "", False


def _known_program(corpus: Any, words: str, *, meanings: bool = True) -> tuple[str, bool]:
    """The title of the article about the program ``words`` names, and whether ``words`` names a kind of program instead.

    By its own title first; else as the encyclopedia's page of the things a
    word may refer to lists it ("Notepad may also refer to: Windows Notepad, a
    plain text editor"), or says it opens ("Excel is a spreadsheet program by
    Microsoft": Microsoft Excel).
    """
    title, kind = _by_title(corpus, words)
    if title or kind or not meanings:
        return title, kind
    try:
        page = corpus.by_title(f"{words[:1].upper() + words[1:]} (disambiguation)") or corpus.by_title(f"{words.title()} (disambiguation)")
        listed = str(corpus.body(page.doc_id, max_chars=8000) or "") if page is not None else ""
    except Exception:  # noqa: BLE001 - no page of meanings is no meaning found there
        listed = ""
    for line in listed.splitlines():
        item = re.match(r"\s*\*+\s*\"?([^,\"]+?)\"?\s*,\s*(.*)", line)
        if item and words.lower() in item.group(1).lower() and _A_PROGRAM.search(item.group(2)):
            title, _kind = _by_title(corpus, item.group(1).strip())
            if title:
                return title, False
    opening = re.search(r"\bis an? [^.]*?\b(?:program|application|software|editor|processor|game)\b[^.]*?\bby ([A-Z][\w&.-]*)", listed.split("\n", 1)[0])
    if opening:
        title, _kind = _by_title(corpus, f"{opening.group(1)} {words.title()}")
        if title:
            return title, False
    return "", False
