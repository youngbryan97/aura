"""Whether a piece of writing read off a screen is a word of the language, and which word a near miss was.

Text read off a game's screen comes back with its letters half right: "Leuel"
for Level, "Ini" for nothing at all. Said aloud as read, a readout of the
score was "Ini 0, iin 0, x 5" (LIVE 2026-10-07). A person reading the same
screen either knows the word or does not say it. The words are the system's
own word list where there is one, and the system's spelling checker, which
knows the words and names of now that a 1934 dictionary does not ("box",
"website", "Scooby"); a near miss of a longer word (one letter out, or two in
a long word) is that word; anything else is not a word.
"""
from __future__ import annotations

import re
import threading
from functools import cache
from pathlib import Path

__all__ = ["the_word", "words_of"]

#: Where a system keeps its word list.
_LISTS = (Path("/usr/share/dict/words"), Path("/usr/share/dict/web2"))

#: Small words that join a name rather than name anything.
_JOINING = frozenset("a an the of to in on at by for and or with from".split())


@cache
def _known() -> frozenset[str]:
    for listed in _LISTS:
        try:
            return frozenset(w.strip().lower() for w in listed.read_text("utf-8", errors="ignore").splitlines() if w.strip())
        except OSError:
            continue
    return frozenset()


#: Letters reading text mistakes for one another, both ways: what is read, and what it may have been.
_MISREAD = (("u", "v"), ("v", "u"), ("i", "l"), ("l", "i"), ("1", "l"), ("0", "o"), ("rn", "m"), ("vv", "w"), ("cl", "d"),
            ("5", "s"), ("8", "b"), ("c", "e"), ("e", "c"), ("n", "h"), ("h", "n"))


def _is_known(word: str, known: frozenset[str]) -> bool:
    """Whether ``word`` is in the list as it is, or as a plural or other form of a word that is."""
    if word in known:
        return True
    for ending, base in (("ies", "y"), ("es", ""), ("s", ""), ("ed", ""), ("ing", ""), ("ing", "e"), ("ed", "e")):
        if word.endswith(ending) and len(word) - len(ending) >= 3 and word[: -len(ending)] + base in known:
            return True
    return False


#: One question at a time to the spelling checker, which is not made to be asked from many threads at once.
_ASKING = threading.Lock()


@cache
def _checker() -> object | None:
    """The system's spelling checker, where the system has one she can ask."""
    try:
        from AppKit import NSSpellChecker  # noqa: PLC0415 - macOS only, and only when first asked

        return NSSpellChecker.sharedSpellChecker()
    except Exception:  # noqa: BLE001 - no checker is a narrower vocabulary, not a fault
        return None


#: Letters that look alike enough for reading text to take one for another. A guess that differs from what was read
#: only by such letters is what was written; one that differs otherwise is another word ("trelalts" is not "trellis").
_LOOK_ALIKE = tuple(frozenset(group) for group in ("aoecu", "ilj1tf", "uvy", "nmhr", "cge", "bh86", "s5", "z2", "qg9", "dcl"))


def _misread_as(read: str, word: str) -> bool:
    """Whether ``word`` could have been read as ``read``: the same length, apart only by letters that look alike,
    by one letter (two in a word of seven or more)."""
    if len(read) != len(word):
        return False
    apart = [(a, b) for a, b in zip(read, word, strict=True) if a != b]
    return 0 < len(apart) <= (1 if len(read) < 7 else 2) and all(any(a in g and b in g for g in _LOOK_ALIKE) for a, b in apart)


#: The language the spelling checker is asked in.
_SPELLED_IN = "en_US"


@cache
def _spelled(read: str) -> str | None:
    """The word the spelling checker takes ``read`` (four letters or more, as written) for: itself where it is spelled
    right, its first guess where that could have been misread as it, else None."""
    checker = _checker()
    plain = read.lower()
    if checker is None or not read.isalpha() or len(read) < 4:
        return None
    form = read.capitalize() if read.isupper() else read
    try:
        with _ASKING:
            missed = checker.checkSpellingOfString_startingAt_language_wrap_inSpellDocumentWithTag_wordCount_(  # type: ignore[attr-defined]
                form, 0, _SPELLED_IN, False, 0, None)[0]
            if missed.length == 0:
                return plain
            guesses = checker.guessesForWordRange_inString_language_inSpellDocumentWithTag_(  # type: ignore[attr-defined]
                (0, len(form)), form, _SPELLED_IN, 0) or []
    except Exception:  # noqa: BLE001 - a checker that cannot answer has said nothing
        return None
    # A guess only for a word of five letters or more: a short one has too many neighbours, and "toct" taken for
    # "tact" is a wrong word where "toct" was only an unread one.
    for guess in list(guesses)[:1] if len(plain) >= 5 else []:
        word = str(guess).lower()
        if word.isalpha() and _misread_as(plain, word):
            return word
    return None


def the_word(read: str) -> str | None:
    """The word ``read`` is, as the language spells it: itself, or the word one misread letter from it, or None."""
    raw = str(read or "").strip().lower()
    plain = re.sub(r"[^a-z0-9]", "", raw)
    if len(re.sub(r"[^a-z]", "", plain)) < 2:
        return None
    known = _known()
    if not known:
        letters = re.sub(r"[^a-z]", "", plain)
        return letters if len(letters) >= 3 and letters == plain else None  # no list here: a word is letters enough to be one
    if plain.isalpha() and _is_known(plain, known):
        return plain
    # One misreading of the kind reading text makes, anywhere in it, and the word it then is; only in a word long
    # enough that a near neighbour is likely the one meant ("Leuel" is level; "Ini" is not anything).
    if len(plain) < 4:
        return None
    for wrong, right in _MISREAD:
        start = plain.find(wrong)
        while start >= 0:
            fixed = plain[:start] + right + plain[start + len(wrong):]
            if fixed.isalpha() and len(fixed) >= 3 and _is_known(fixed, known):
                return fixed
            start = plain.find(wrong, start + 1)
    return _spelled(re.sub(r"[^A-Za-z]", "", str(read or "").strip()) if plain.isalpha() else plain)


def words_of(name: str, *, most: int = 2) -> str | None:
    """A name read off a screen as words of the language ("Leuel" is "level"), or None where it is not a name of words.

    A name is at most ``most`` words that name something, every one of them a
    word; the small words that join them do not count, and a name of nothing
    else ("of the") is not a name.
    """
    parts = [p for p in re.findall(r"[A-Za-z0-9]+", str(name or "")) if re.search(r"[A-Za-z]", p)]
    naming = [p for p in parts if p.lower() not in _JOINING]
    if not naming or len(naming) > most or parts[0].lower() in _JOINING or parts[-1].lower() in _JOINING:
        return None
    words = [the_word(p) if p.lower() not in _JOINING else p.lower() for p in parts]
    return " ".join(words) if all(words) else None
