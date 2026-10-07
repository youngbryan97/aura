"""Whether a piece of writing read off a screen is a word of the language, and which word a near miss was.

Text read off a game's screen comes back with its letters half right: "Leuel"
for Level, "Ini" for nothing at all. Said aloud as read, a readout of the
score was "Ini 0, iin 0, x 5" (LIVE 2026-10-07). A person reading the same
screen either knows the word or does not say it. The words are the system's
own word list where there is one; a near miss of a longer word (one letter
out) is that word; anything else is not a word.
"""
from __future__ import annotations

import re
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
    return None


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
