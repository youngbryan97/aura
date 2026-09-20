"""Word-form equivalence for lexical evidence, not semantic correctness."""

from functools import lru_cache
from threading import local

import snowballstemmer

_state = local()


@lru_cache(maxsize=4096)
def word_form(word: str) -> str:
    """Normalize English inflection with an isolated Snowball instance."""
    stemmer = getattr(_state, "stemmer", None)
    if stemmer is None:
        stemmer = snowballstemmer.stemmer("english")
        _state.stemmer = stemmer
    return str(stemmer.stemWord(word.lower()))


def matching_word_forms(wanted: set[str], observed: set[str]) -> set[str]:
    """Return requested anchors present in another grammatical word form."""
    observed_forms = {word_form(word) for word in observed}
    return {word for word in wanted if word_form(word) in observed_forms}


_INFLECTIONS = ("s", "es", "ed", "d", "ing", "ies", "ied")


def same_lexeme_inflected(a: str, b: str) -> bool:
    """Whether two words are one word in two inflections, not two words.

    "drain"/"drains", "minute"/"minutes", "manage"/"managed" are one lexeme.
    "part"/"partly", "amaze"/"amazing part" are not: a derivation changes
    what the word is, and a stemmer folds both (LIVE 2026-09-20: "the
    amazing part" and "partly managed by their skin" read as a shared
    topic, and a reply about octopus skin passed as staying with a thread
    about people being amazed by her).
    """
    a, b = a.lower(), b.lower()
    if a == b:
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    # y -> ies / ied: body/bodies, carry/carried
    if short.endswith("y") and long.startswith(short[:-1]):
        return long[len(short) - 1:] in ("ies", "ied")
    # a final e drops before -ing / -ed: amaze/amazing, manage/managed
    if short.endswith("e") and long.startswith(short[:-1]):
        return long[len(short) - 1:] in ("ing", "ed", "es", "ed")
    if not long.startswith(short):
        return False
    tail = long[len(short):]
    # a doubled consonant before -ing / -ed: run/running, stop/stopped
    if len(tail) >= 3 and tail[0] == short[-1] and tail[1:] in ("ing", "ed"):
        return True
    return tail in _INFLECTIONS


def matching_inflections(wanted: set[str], observed: set[str]) -> set[str]:
    """Requested anchors present in another inflection of the same lexeme."""
    return {
        word
        for word in wanted
        if any(same_lexeme_inflected(word, other) for other in observed)
    }
