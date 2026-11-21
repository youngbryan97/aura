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
           "in_words_not_code",
           "legible_share", "shape_of", "the_watcher"]

logger = logging.getLogger("Aura.WhatAWatcherHears")

#: How long a line of one shape is not said again, in seconds.
SHAPE_AGAIN_S = 45.0

#: How long the same line, word for word, is not said again, in seconds.
SAME_AGAIN_S = 300.0

#: How soon after one line a remark made while playing may be said: a person playing and talking says something every
#: few seconds, not five things in three (offline and live 2026-10-12: "That's me…", "I'm the black bar…", "I think I've
#: noticed…", "The bar at the middle goes down…", "The black tree trunk is worth getting…" inside three seconds).
PACE_S = 4.0
#: Lines that are never held back for pace: what she does, what the place says or asks of her, how a go ended and what
#: she made of it, how she understands the place, and the picking of what to play.
_ALWAYS_SAID = re.compile(
    r"^(?:That one ended|Round \d|That's \d+ run|There's nothing to win|I've given this one|My time with|It says|It asks|"
    r"Its rules|Run \d|As I understand it|Now I understand|There are \d+ games|By your rule|Clicking|Pressing|Going |"
    r"Carrying|Typing|Holding|The game has a button|My guide|I'm checking|Nothing I looked up|From |I looked up|"
    r"This is like|Something is playing|I keep losing|Nothing I do|I won|I lost|My move)", re.I)
#: Lines that are news each time, however like the last they are: how each go came out, each move made. Live 2026-10-12
#: "Run 2 of 3: scored 10" was held back as a line like "Run 1 of 3: scored 190", and the second run went untold.
_NEWS_EACH_TIME = re.compile(r"^(?:There's nothing to win in this one, only a score\. )?(?:Run \d+ of \d+|Round \d+|My move)")

#: The share of a quote's words that must be words for it to be quoted.
LEGIBLE = 0.6

#: How many words a line must have besides a quote she could not read for it to be said with that quote left out.
SAID_WITHOUT_THE_QUOTE = 10

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


#: A name out of a program, not a word: joined by underscores (ST_KILL, SCR_LEVEL01), a letter or two before a capital
#: (nLevel, mState), capitals run into a number (CH1), a call, or a comparison.
_CODE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b|\b[a-z]{1,2}[A-Z][A-Za-z0-9]+\b|\b[A-Z]{2,}[0-9]+\b"
                   r"|\b\w+\(\)|==|!=|&&|\|\||<=|>=")


def in_words_not_code(said: str) -> str:
    """``said`` as words a person reads: what a program names things by, in brackets, left out; a sentence that still
    rests on such names not said. "" where nothing is left. A reading of the screen is left as it was read. LIVE
    2026-10-11 "What I'm after: level 3 boss (Boss 03) reaches ST_KILL ... character-select screen (CH1, CH2, or CH3),
    then Level 1 (SCR_LEVEL01)" and "it's won when The code checks if nLevel == 3" were said to the person watching."""
    if _IT_SAYS.match(said) or not _CODE.search(_QUOTED.sub(" ", said)):
        return said
    kept: list[str] = []
    lead = ""
    for sentence in re.split(r"(?<=[.;!?])\s+", said):
        sentence = re.sub(r"\s*\(([^()]*)\)", lambda m: "" if _CODE.search(m.group(1)) else m.group(0), sentence)
        if _CODE.search(_QUOTED.sub(" ", sentence)):
            # What the line is ("From its code:") stays with what is left of it.
            heading = re.match(r"^([^:;.]{1,30}:)\s", sentence)
            lead = lead or (heading.group(1) if heading and not kept else "")
            continue
        if lead and not kept:
            sentence = f"{lead} {sentence}"
        kept.append(sentence)
    return " ".join(" ".join(kept).split()).strip()


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


#: The names of keys, as instructions write them: words of the place's controls, not of the language. LIVE-like
#: 2026-10-11 "USE WASD TO MOVE" was said as "USE … TO MOVE".
_KEY_NAMES = frozenset("wasd ijkl esdf zqsd arrows ctrl shift esc escape enter tab alt spacebar backspace pgup pgdn".split())


def _a_word(token: str, names: frozenset[str]) -> str | None:
    """The word ``token`` is: a name she was given, a short word of the language, or a word one misread letter away.
    A word joined by hyphens ("late-night") is a word where each of its parts is."""
    parts = [part for part in token.split("-") if part]
    if len(parts) > 1:
        read = [_a_word(part, names) for part in parts]
        return "-".join(read) if all(read) else None  # type: ignore[arg-type]
    plain = _plain(token.replace("’", "'").split("'")[0])
    if plain in names or plain in _KEY_NAMES or re.fullmatch(r"f(?:[1-9]|1[0-2])", plain):
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
    #: When the last line was said.
    said_at: float = -1e9
    shapes: dict[str, float] = field(default_factory=dict)
    lines: dict[str, float] = field(default_factory=dict)

    def knows(self, text: str) -> None:
        """Take the names in ``text`` (a task, a title) as words: "Scooby" and "Bloo" are not in the language, and are not misread."""
        for word in re.findall(r"[A-Za-z][A-Za-z'’-]+", str(text or "")):
            self.names.add(_plain(word.replace("’", "'").split("'")[0]))

    def heard(self, line: str, now: float | None = None, *, news: bool = False) -> str | None:
        """``line`` as a watcher should hear it, or None where it would tell them nothing or quote what she could not read.

        ``news``: what happened, said as it happens (a move made in a game, and its answer). It is never held back for
        pace or for being shaped like the last, only for being the same line word for word. LIVE 21 November 2026 she
        was asked to tell each move against Stockfish, and every move of hers came within four seconds of the move it
        answered and was held back as "a remark too soon after the last line"."""
        now = time.monotonic() if now is None else now
        said = " ".join(str(line or "").split())
        if not said:
            return None
        names = frozenset(self.names)
        said = self._quoted_legibly(said, names)
        if said is None:
            return None
        said = in_words_not_code(said)
        if not said:
            logger.info("not said, it was the program's names, not words: %r", str(line)[:120])
            return None
        said = _the_move_once(said)
        # One mark to end a sentence: a quote that ends on its own "!" or "..." is not ended again ("pizza!.").
        said = re.sub(r"([!?…]|\.\.\.)\.(?=[\s\"'”’)]|$)", r"\1", said)
        shape = shape_of(said)
        if now - self.lines.get(said, -SAME_AGAIN_S) < SAME_AGAIN_S:
            logger.info("not said again: %r", said[:120])
            return None
        if (not news and now - self.shapes.get(shape, -SHAPE_AGAIN_S) < SHAPE_AGAIN_S
                and not _NEWS_EACH_TIME.match(said)):
            logger.info("not said, a line like it was a moment ago: %r", said[:120])
            return None
        if not news and now - self.said_at < PACE_S and not _ALWAYS_SAID.match(said):
            logger.info("not said, a remark too soon after the last line: %r", said[:120])
            return None
        self.lines[said], self.shapes[shape], self.said_at = now, now, now
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
        unread = []
        for match in list(_QUOTED.finditer(said)):
            text = match.group(1) or match.group(2) or match.group(3) or ""
            if text.startswith(("the shape at ", "the one that stands out at ", "the middle of the picture")):
                continue
            quoted = match.group(0)
            # A label's stray marks read off its edges ('START."') are not part of it; its own "!" or "?" is.
            trimmed = re.sub(r"[\s.,:;'\"“”‘’]+$|^[\s.,:;'\"“”‘’]+", "", text)
            if trimmed and trimmed != text and len(trimmed.split()) <= 4:
                said = said.replace(text, trimmed, 1)
                quoted = quoted.replace(text, trimmed, 1)
                text = trimmed
            if legible_share(text, names) < LEGIBLE:
                unread.append(quoted)
                continue
            mended = _mended(text, names)
            if mended != text:
                said = said.replace(text, mended, 1)
        if unread:
            # What she could not read is left out, and the rest said, as a person reading aloud passes over a word they
            # cannot make out: offline 2026-10-12 what she made in a thing for making was never told, for a caption
            # quoted in it. A line that is little more than the quote says nothing without it.
            # Counted outside every quote: a line made of quotes and a few words between says little without one.
            rest = _QUOTED.sub(" ", said)
            if len(re.findall(r"[A-Za-z]{2,}", rest)) < SAID_WITHOUT_THE_QUOTE:
                logger.info("not said, what it quotes could not be read: %r", said[:120])
                return None
            for quoted in unread:
                said = said.replace(quoted, "words I could not make out", 1)
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


def heard(line: str, *, news: bool = False) -> str | None:
    """``line`` as the watcher hears it, or None."""
    return the_watcher().heard(line, news=news)


def forget_what_was_heard() -> None:
    """Every line as if never said: for a new sitting, and for checks that must not hear each other."""
    _THE_WATCHER.shapes.clear()
    _THE_WATCHER.lines.clear()
    _THE_WATCHER.names.clear()
