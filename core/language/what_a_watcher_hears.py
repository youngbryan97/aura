"""What a person watching her work hears: each line once, in words she could read.

Someone talking while they work says what they are trying and what came of it.
They say "trying the arrows" once, not once for every arrow, and they do not
read out writing they could not make out. LIVE 2026-10-07 a game's chat filled
with "Going up — to see what it does", "Going down — to see what it does" and
the like every few seconds, and quoted misread writing ("Clicking "toct"",
"It says: ....... 5°") as if it were words.

So every line said while she works passes through here, whatever the task:

- **Its shape.** A line is reduced to its form: what it quotes, its numbers, the
  keys and ways it names are taken out, and of a move line ("Going up — ...")
  only its verb is kept. A line of the same shape heard a moment ago tells the
  watcher nothing new.
- **Its words.** Writing she quotes off a screen is quoted only where most of it
  is words of the language or names she was given. A near miss is mended
  ("Leuel" is level); a word she cannot make out is left out of the quote; a
  quote with too little left in it is not said at all.

Nothing here knows what she is doing. The same rules hold for a game, a form,
a document or a page.
"""
from __future__ import annotations

import contextvars
import logging
import re
import time
from dataclasses import dataclass, field

from core.language.words_of_the_language import the_word

__all__ = ["WhatAWatcherHears", "a_fresh_watcher", "done_watching", "forget_what_was_heard", "heard", "in_her_own_words",
           "legible_share", "shape_of", "the_watcher"]

logger = logging.getLogger("Aura.WhatAWatcherHears")

#: How long a line of one shape is not said again, in seconds.
SHAPE_AGAIN_S = 45.0

#: How long the same line, word for word, is not said again, in seconds.
SAME_AGAIN_S = 300.0

#: The share of a quote's words that must be words for it to be quoted.
LEGIBLE = 0.6

#: Writing quoted in a line: in straight or curly quotes, or after "It says:". A single quote opens only after
#: something not a letter, so the one in "I've" opens nothing.
_QUOTED = re.compile(r"\"([^\"]+)\"|“([^”]+)”|(?<![A-Za-z])'([^']+)'(?![A-Za-z])")
_IT_SAYS = re.compile(r"^(?P<lead>(?:it|the screen|the page)\s+says:?\s*)(?P<said>.+)$", re.I)

#: Keys and ways a move line names; in its shape they are one: "Going up" and "Going left" are the same line.
_WAYS = re.compile(r"\b(?:up|down|left|right|space|enter|return|escape|tab|shift|ctrl|control|alt|option|command|[a-z])\b(?=\s|$|[—,.;:])")

#: A move line: a verb of moving, pressing or clicking, then what, then why after a dash.
_A_MOVE = re.compile(r"^(?P<verb>going|pressing|clicking|holding|typing|moving|pointing|scrolling|dragging)\b[^—]*(?:—\s*(?P<why>.*))?$", re.I)

#: Words of two letters that are words; a word list's own two-letter entries are mostly abbreviations
#: ("ur", "te"), and misread writing is full of pairs of letters.
_SHORT = frozenset("a i am an as at be by do go he hi if in is it me my no of oh ok on or so to up us we".split())

#: A place on the screen given as shares of it ("the shape at 90% across, 40% down").
_A_PLACE = re.compile(r"(the shape|the one that stands out) at (\d+)% across, (\d+)% down")


def _plain(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "", word.lower())


def shape_of(line: str) -> str:
    """The form of ``line`` with what it is about taken out: a move line is its verb; any line its words, without quotes, numbers or keys."""
    said = " ".join(str(line or "").split()).lower()
    if _IT_SAYS.match(said):
        return said  # a reading is as new as what it reads
    move = _A_MOVE.match(said)
    if move:
        why = _WAYS.sub("~", _QUOTED.sub("_", move.group("why") or ""))
        return f"{move.group('verb')} — {re.sub(r'_.*?(?= has| is|$)', '_', why)}".strip(" —")
    said = _QUOTED.sub("_", said)
    said = re.sub(r"\d+(?:[.,:]\d+)*\s*%?", "#", said)
    said = _WAYS.sub("~", said)
    return said


def in_her_own_words(line: str) -> str:
    """``line`` without the writing it quotes, so what is judged as her prose is only what she said herself.

    LIVE 2026-10-07 "It says: NUMBUH: 361,188 Special Talents: Super Hero
    Strength" was withheld from the chat as collapsed prose. The words were the
    screen's, read out, and a list of labels has few small words in it.
    """
    lines = []
    for one in str(line or "").splitlines():
        reading = _IT_SAYS.match(one.strip())
        lines.append(_QUOTED.sub(" ", reading.group("lead") if reading else one))
    return "\n".join(lines)


def _the_move_once(said: str) -> str:
    """A move line whose reason begins by naming the move again says "it": 'Going up — up is the only thing
    available' is 'Going up — it is the only thing available'."""
    move = re.match(r"^(?:going|pressing|clicking|holding)\s+(?P<what>.+?)\s+—\s+(?P<why>.+)$", said, re.I)
    if not move:
        return said
    what, why = move.group("what"), move.group("why")
    for named in (f"click {what}", what):
        if why.lower().startswith(named.lower() + " "):
            return said[: move.start("why")] + "it" + why[len(named):]
    return said


def _seen_before(plain: str, seen: frozenset[str]) -> str | None:
    """A word this place showed before that ``plain`` is one letter from, of the same length: what it was misread
    from. LIVE 2026-10-10 "YOU SAVED 1 ITEMS" was read, and then "YOU SAYED 3 ITEMS", which the language also spells."""
    if len(plain) < 4 or plain in seen:
        return None
    near = [word for word in seen if len(word) == len(plain) and sum(a != b for a, b in zip(word, plain, strict=True)) == 1]
    return near[0] if len(near) == 1 else None


def _a_word(token: str, names: frozenset[str]) -> str | None:
    """The word ``token`` is: a name she was given, a short word of the language, or a word one misread letter away.
    A word joined by hyphens ("late-night") is a word where each of its parts is."""
    parts = [part for part in token.split("-") if part]
    if len(parts) > 1:
        read = [_a_word(part, names) for part in parts]
        return "-".join(read) if all(read) else None  # type: ignore[arg-type]
    plain = _plain(token.replace("’", "'").split("'")[0])
    if plain in names:
        return plain
    if len(plain) <= 2:
        return plain if plain in _SHORT else None
    return the_word(plain)


def where_it_is(across: float, down: float) -> str:
    """A place given as shares of the screen, in the words a person points with: "the top right", "the middle"."""
    row = "top" if down < 0.34 else "bottom" if down > 0.66 else ""
    column = "left" if across < 0.34 else "right" if across > 0.66 else ""
    return f"the {row} {column}".replace("  ", " ").strip() if row or column else "the middle"


def legible_share(text: str, names: frozenset[str] = frozenset()) -> float:
    """The share of ``text``'s words that are words of the language or names she knows; one-letter words do not count either way."""
    words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9'’-]*", str(text or "")) if len(_plain(w)) > 1 or w.isdigit()]
    if not any(re.search(r"[A-Za-z]", w) for w in words):
        return 0.0  # numbers alone are a readout, not writing to quote
    read = sum(1 for w in words if w.isdigit() or _a_word(w, names) is not None)
    return read / len(words)


def _mended(text: str, names: frozenset[str], seen: frozenset[str] = frozenset()) -> str:
    """``text`` with each near miss written as the word it is, and each word not made out left out as "…".

    In a reading that is plainly prose, a capitalised word the language does not have is a name ("Bloo"), and
    is kept as written.
    """
    out: list[str] = []
    prose = len(str(text or "").split()) >= 6 and legible_share(text, names) >= 0.8
    latin = sum(ch.isascii() for ch in str(text or "") if ch.isalpha()) >= 0.8 * max(1, sum(ch.isalpha() for ch in str(text or "")))
    for token in str(text or "").split():
        # Letters of another script in writing of this one are a misreading of it ("(ог" for "or").
        if latin and any(ch.isalpha() and not ch.isascii() for ch in token):
            if not (out and out[-1] == "…"):
                out.append("…")
            continue
        core = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", token)
        if not core or len(_plain(core)) <= 1 or core.isdigit() or _plain(core) in names:
            out.append(token)
            continue
        word = _seen_before(_plain(core), seen) or _a_word(core, names)
        if word is None and prose and re.fullmatch(r"[A-Z][a-z]{2,}", core) and out and out[-1][-1:] not in ".!?":
            out.append(token)
            continue
        if word is None:
            if out and out[-1] == "…":
                continue
            out.append("…")
            continue
        if word != _plain(core.replace("’", "'").split("'")[0]):
            fixed = word.upper() if core.isupper() else word.capitalize() if core[:1].isupper() else word
            token = token.replace(core, fixed)
        out.append(token)
    while out and out[-1] == "…":
        out.pop()
    while out and out[0] == "…":
        out.pop(0)
    # Capitals read off a drawn screen come back in mixed case ("no InvenToRY"); a quote with a word like that was
    # written in capitals.
    if len(out) <= 4 and any(re.search(r"[a-z][A-Z]|[A-Z]{2}[a-z]", token) for token in out):
        out = [token.upper() for token in out]
    return " ".join(out)


@dataclass
class WhatAWatcherHears:
    """The lines a watcher has heard lately, and the names she was given."""

    names: set[str] = field(default_factory=set)
    #: Words the place has shown her, read whole: what a later misreading of one of them is mended to.
    seen: set[str] = field(default_factory=set)
    shapes: dict[str, float] = field(default_factory=dict)
    lines: dict[str, float] = field(default_factory=dict)

    def knows(self, text: str) -> None:
        """Take the names in ``text`` (a task, a title) as words: "Scooby" and "Bloo" are not in the language, and are not misread."""
        for word in re.findall(r"[A-Za-z][A-Za-z'’-]+", str(text or "")):
            self.names.add(_plain(word.replace("’", "'").split("'")[0]))

    def heard(self, line: str, now: float | None = None) -> str | None:
        """``line`` as a watcher should hear it, or None where it would tell them nothing or quote what she could not read."""
        now = time.monotonic() if now is None else now
        said = " ".join(str(line or "").split())
        if not said:
            return None
        names = frozenset(self.names)
        said = self._quoted_legibly(said, names)
        if said is None:
            return None
        said = _the_move_once(said)
        shape = shape_of(said)
        if now - self.lines.get(said, -SAME_AGAIN_S) < SAME_AGAIN_S:
            logger.info("not said again: %r", said[:120])
            return None
        if now - self.shapes.get(shape, -SHAPE_AGAIN_S) < SHAPE_AGAIN_S:
            logger.info("not said, a line like it was a moment ago: %r", said[:120])
            return None
        self.lines[said], self.shapes[shape] = now, now
        return said

    def _quoted_legibly(self, said: str, names: frozenset[str]) -> str | None:
        reading = _IT_SAYS.match(said)
        if reading:
            text = reading.group("said")
            if legible_share(text, names) < LEGIBLE:
                logger.info("not said, too little of it could be read: %r", said[:120])
                return None
            mended = _mended(text, names, frozenset(self.seen))
            self.seen.update(word for word in (_plain(w) for w in mended.split()) if len(word) >= 4 and the_word(word) == word)
            return reading.group("lead") + mended
        said = _A_PLACE.sub(lambda m: f"{m.group(1)} at " + where_it_is(int(m.group(2)) / 100, int(m.group(3)) / 100), said)
        said = re.sub(r"\"((?:the shape|the one that stands out) at [^\"]+|the middle of the picture)\"", r"\1", said)
        for match in list(_QUOTED.finditer(said)):
            text = match.group(1) or match.group(2) or match.group(3) or ""
            if text.startswith(("the shape at ", "the one that stands out at ", "the middle of the picture")):
                continue
            # A label's stray marks read off its edges ('START."') are not part of it; its own "!" or "?" is.
            trimmed = re.sub(r"[\s.,:;'\"“”‘’]+$|^[\s.,:;'\"“”‘’]+", "", text)
            if trimmed and trimmed != text and len(trimmed.split()) <= 4:
                said = said.replace(text, trimmed, 1)
                text = trimmed
            if legible_share(text, names) < LEGIBLE:
                logger.info("not said, what it quotes could not be read: %r", said[:120])
                return None
            mended = _mended(text, names)
            if mended != text:
                said = said.replace(text, mended, 1)
        return re.sub(r'"{2,}', '"', said)


_THE_WATCHER = WhatAWatcherHears()

#: The watcher of the piece of work in hand (one game, one form, one document), where one was begun for it.
_WATCHING: contextvars.ContextVar[WhatAWatcherHears | None] = contextvars.ContextVar("aura_what_a_watcher_hears", default=None)


def the_watcher() -> WhatAWatcherHears:
    """Whoever is watching the work in hand: the one begun for it, else the one person watching her at all."""
    return _WATCHING.get() or _THE_WATCHER


def a_fresh_watcher(task: str = "") -> contextvars.Token[WhatAWatcherHears | None]:
    """Begin a piece of work: what was said of the last one is not what this one has said. Reset with the token."""
    watcher = WhatAWatcherHears()
    watcher.knows(task)
    return _WATCHING.set(watcher)


def done_watching(token: contextvars.Token[WhatAWatcherHears | None]) -> None:
    """The piece of work begun with ``token`` is over; whoever watched before it watches on."""
    _WATCHING.reset(token)


def heard(line: str) -> str | None:
    """``line`` as the watcher hears it, or None."""
    return the_watcher().heard(line)


def forget_what_was_heard() -> None:
    """Every line as if never said: for a new sitting, and for checks that must not hear each other."""
    _THE_WATCHER.shapes.clear()
    _THE_WATCHER.lines.clear()
    _THE_WATCHER.names.clear()
