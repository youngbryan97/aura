"""A reply's dated claims, read against what the turn's own sources say about the same thing.

LIVE 2026-10-04. She answered "When did the Kaseya Center open?" from the
Wikipedia page and added that FTX bought the naming rights "in 2022" in a deal
that "lasted barely two months". The page she had read gives FTX Arena as 2021
to 2023. Asked a turn later where that came from, she went back to the page,
kept what it said and named the rest as half-remembered. Bryan: those sources
should be used the first time too.

The passage she is shown from each source is chosen by the question's words,
so an answer that goes past the question goes past the passage. This reads the
whole of each source for the claims the reply makes:

* a claim is a clause of the reply that carries a year;
* the source sentences that could bear on it are the ones that name something
  the claim names and carry a year themselves;
* which of them is about the same thing is decided by meaning, by the evidence
  judge (core/cognition/evidence_relevance.py) and its measured boundary;
* the claim disagrees when the sentence most about the same thing gives years
  and the claim's year is not among them.

A claim no source speaks to is not a disagreement, and nothing is reported
when the judge has not measured.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from core.conversation.thread_continuity import content_terms
from core.utils.readable_text import SENTENCE_END

logger = logging.getLogger(__name__)

__all__ = ["Disagreement", "claims_with_years", "disagreements"]

_YEAR = re.compile(r"(?<![\d$£€.,])\b(1[5-9]\d\d|20\d\d)\b(?![\d,]\d)")

#: Where one claim in a sentence ends and the next begins: clause punctuation,
#: a spaced dash, or a comma before a lower-case word ("…(1999–2021), then FTX
#: Arena …"). A comma before a digit is a date ("December 31, 1999").
_BETWEEN_CLAUSES = re.compile(r"\s*[;:—]\s*|\s+–\s+|,\s+(?=[a-z])")

#: Source sentences weighed per claim; the nearest by the order they are read.
_CANDIDATES_PER_CLAIM = 48

#: The calendar's words date a thing; they do not name one.
_CALENDAR = frozenset(
    "january february march april may june july august september october november december "
    "monday tuesday wednesday thursday friday saturday sunday".split()
)


@dataclass(frozen=True)
class Disagreement:
    """A claim in the reply, and the sentence of a source about the same thing that dates it otherwise."""

    claim: str
    says: str
    title: str
    location: str
    origin: str
    score: float


def _years(text: str) -> set[str]:
    return set(_YEAR.findall(text))


def _names(text: str) -> set[str]:
    """What a span names: its capitalised words that carry content."""
    terms = content_terms(text)
    return {
        word.lower().strip("'’")
        for word in re.findall(r"[A-Z][\w'’-]+", text)
        if word.lower().strip("'’") in terms and word.lower() not in _CALENDAR
    }


def _sentences(text: str) -> list[str]:
    units: list[str] = []
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        start = 0
        for end in SENTENCE_END.finditer(line):
            piece = line[start:end.end()].strip()
            if piece:
                units.append(piece)
            start = end.end()
        if line[start:].strip():
            units.append(line[start:].strip())
    return units


def claims_with_years(reply: str, request: str = "") -> list[tuple[str, frozenset[str]]]:
    """The clauses of a reply that carry a year, each with what it names.

    A clause that names nothing ("December 31, 1999") is about what was asked.
    """
    asked = frozenset(_names(request))
    claims = []
    for sentence in _sentences(reply):
        for clause in _BETWEEN_CLAUSES.split(sentence):
            clause = clause.strip(" .!?")
            names = frozenset(_names(clause)) or asked
            if _years(clause) and names:
                claims.append((clause, names))
    return claims


def disagreements(
    reply: str, sources: Sequence[Mapping[str, str]], request: str = ""
) -> list[Disagreement]:
    """The reply's dated claims that the sentence of a source most about the same thing dates otherwise."""
    claims = claims_with_years(reply, request)
    if not claims or not sources:
        return []
    from core.cognition.evidence_relevance import assess_evidence_alignments

    found: list[Disagreement] = []
    for claim, names in claims:
        candidates: list[tuple[str, Mapping[str, str]]] = []
        for source in sources:
            for sentence in _sentences(source.get("text", "")):
                if _years(sentence) and names & _names(sentence):
                    candidates.append((sentence, source))
        candidates = candidates[:_CANDIDATES_PER_CLAIM]
        if not candidates:
            continue
        try:
            verdicts = assess_evidence_alignments(claim, [sentence for sentence, _ in candidates])
        except (RuntimeError, TypeError, ValueError, OSError) as exc:
            logger.debug("a claim was not read against its sources: %s", exc)
            return []
        measured = [
            (verdict.score, sentence, source)
            for verdict, (sentence, source) in zip(verdicts, candidates, strict=True)
            if verdict.measured and verdict.relevant and verdict.score is not None
        ]
        if not measured:
            continue
        score, sentence, source = max(measured, key=lambda item: item[0])
        if not _years(claim) & _years(sentence):
            found.append(Disagreement(
                claim=claim,
                says=sentence,
                title=str(source.get("title", "")),
                location=str(source.get("location", "")),
                origin=str(source.get("origin", "")),
                score=float(score),
            ))
    return found
