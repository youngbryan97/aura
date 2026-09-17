"""The part of a sentence a denial is about.

Two detectors read her denials against what the runtime measures: the
capability ledger (``core/self/capability_ledger.py``) and the registry check
(``core/conversation/capability_denial.py``). Both used to bind a denial to
any capability word anywhere in the sentence.

LIVE, 2026-09-16: "I can't measure the sun from here, so I'll work it out from
the geometry" was read as a denial of reaching the world, because "world"
appeared in the clause that says what she does instead. In the same session
"I'd rather flag that than repeat a number I can't verify", in a sentence that
also said "largest file", was read as a denial of the filesystem. Each was
replaced with a status line about a capability nobody had denied.

A denial is about the clause it sits in. What follows the comma, the "so", the
"because" is a different claim.
"""

from __future__ import annotations

import re

__all__ = ["denied_bounds", "denied_span", "complement_names_something_else"]

#: Where a clause ends: punctuation, a dash, or a word that opens the next
#: clause. The lookahead keeps "I can't do so." whole — that "so" is the
#: object, not a conjunction.
_CLAUSE_WORDS = (
    r"\s(?:so|but|because|since|which|though|although|unless|instead|rather"
    r"|while|whereas|yet|therefore|hence|until|if|when|where)(?=\s+\w)"
)
_CLAUSE_BOUNDARY = re.compile(r"[,;:]|—|–|\s-\s|\s\(|\.\.\." + "|" + _CLAUSE_WORDS, re.IGNORECASE)
#: Looking back from the denial, a colon is a label, not a boundary: "Current
#: energy and focus numbers: Not readable." names its subject before it.
_BACK_BOUNDARY = re.compile(r"[,;]|—|–|\s-\s|\s\(|\.\.\." + "|" + _CLAUSE_WORDS, re.IGNORECASE)

#: "a sensor FOR daylight", "no reading OF the sky": the thing the instrument
#: is being asked to point at.
_COMPLEMENT = re.compile(
    r"^\s*(?:(?:in|on|inside|within)\s+(?:me|myself|here)\s+|of\s+(?:mine|my\s+own)\s+)?"
    r"(?:for|of|on|at|about|over|pointed\s+at|aimed\s+at|that\s+reads?"
    r"|that\s+measures?)\s+(?P<what>(?:[\w'-]+\s*){1,4})",
    re.IGNORECASE,
)

_FUNCTION_WORDS = frozenset(
    "the a an my your her his its their our this that these those it them any "
    "some own here there now right kind sort type whatsoever".split()
)


def denied_bounds(sentence: str, frame: re.Match[str]) -> tuple[int, int]:
    """The clause of ``sentence`` that holds the denial ``frame``, as bounds."""
    start = 0
    for match in _BACK_BOUNDARY.finditer(sentence):
        if match.end() > frame.start():
            break
        start = match.end()
    cut = _CLAUSE_BOUNDARY.search(sentence, frame.end())
    return start, cut.start() if cut else len(sentence)


def denied_span(sentence: str, frame: re.Match[str]) -> str:
    """The clause of ``sentence`` that holds the denial ``frame``."""
    start, end = denied_bounds(sentence, frame)
    return sentence[start:end]


def _stem(word: str) -> str:
    word = word.lower()
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def complement_names_something_else(after: str, own: tuple[str, ...] | set[str]) -> bool:
    """True when the words after a term point the instrument at something not in ``own``.

    "I have no sensor for daylight" denies a daylight sensor, which no
    interoception the runtime has would contradict. "I have no sensors" and
    "no sensor for my energy" are denials of what it does read.
    """
    match = _COMPLEMENT.match(after)
    if not match:
        return False
    content = {
        _stem(word)
        for word in re.findall(r"[a-z][a-z'-]*", match.group("what").lower())
        if word not in _FUNCTION_WORDS
    }
    if not content:
        return False
    return content.isdisjoint({_stem(word) for phrase in own for word in phrase.split()})
