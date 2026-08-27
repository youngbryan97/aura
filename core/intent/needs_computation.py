"""Whether a request asks for something to be worked out from a named thing.

The code domain was admitted for arithmetic written in the sentence and for
finite constraint problems — things that settle BY computation on their own
terms. Both of the live failures below are the ordinary case instead: the thing
to compute over is somewhere else, so the sentence carries no arithmetic at all.

LIVE, 2026-08-27, twice:

* "docs and source are at <path>. Read it, then actually use it: open a ledger,
  post an invoice, reverse it." A library is used by being called, and calling
  it is running code.
* "I've got a deals export at <path>. How many are approved, what do they add
  up to, and which region has the highest average?" Counting and averaging a
  file is arithmetic whose operands are in the file.

Both were given a file reader and nothing that computes, so both ended in "I
couldn't get to an answer I'd stand behind."

The floors here are structural — a verb of execution with something to run, an
aggregate with something to aggregate — and the learned surface decides what
they do not settle.
"""

from __future__ import annotations

import logging
import re

__all__ = ["asks_to_exercise_software", "asks_for_an_aggregate", "needs_computation"]

logger = logging.getLogger(__name__)

#: Something that is used by being called.
_A_PIECE_OF_SOFTWARE = re.compile(
    r"\b(?:librar(?:y|ies)|module|package|api|sdk|client|function|method|class|"
    r"script|program|tool|helper|endpoint|command|cli|binary)\b",
    re.IGNORECASE,
)

#: Asking for it to be exercised rather than described.
_EXERCISE_IT = re.compile(
    r"\b(?:use|using|call|calling|run|running|execute|exercise|try|drive|"
    r"invoke|apply|work\s+with|put\s+it\s+through|actually\s+use)\b",
    re.IGNORECASE,
)

#: Running something ON something: the inputs are the structural tell, because
#: you do not hand inputs to a description.
_ON_SOME_INPUTS = re.compile(
    r"\b(?:over|with|on|against|through|using)\s+"
    r"(?:it|them|this|that|these|those|the|a|an|my|your|each|both|\d)",
    re.IGNORECASE,
)

#: A quantity derived from a set of things rather than stated in the sentence.
_AN_AGGREGATE = re.compile(
    r"\b(?:how\s+many|how\s+much|count|total|totals|sum|add\s+up|adds\s+up|"
    r"average|averages|mean|median|highest|lowest|largest|smallest|biggest|"
    r"most|least|top|bottom|breakdown|per\s+\w+|group(?:ed)?\s+by|"
    r"proportion|percentage|share|rank|ranked)\b",
    re.IGNORECASE,
)

#: Something with rows in it.
_A_BODY_OF_DATA = re.compile(
    r"\b(?:csv|tsv|spreadsheet|export|dataset|data\s*set|table|rows?|records?|"
    r"entries|log|logs|ledger|report|dump|extract|\.xlsx?|\.csv|\.tsv|\.json|"
    r"\.jsonl|\.parquet)\b",
    re.IGNORECASE,
)

#: Asking ABOUT a thing rather than for work on it.
_ABOUT_IT = re.compile(
    r"^\s*(?:what\s+(?:is|does|are)|explain|describe|tell\s+me\s+about|"
    r"how\s+does|why\s+does|summari[sz]e)\b",
    re.IGNORECASE,
)


def _names_something_to_work_on(text: str) -> bool:
    """Whether a resource is named that the work would be done over."""
    if _A_BODY_OF_DATA.search(text):
        return True
    try:
        from core.language.named_paths import named_paths

        return bool(named_paths(text))
    except (ImportError, TypeError, ValueError):
        return False


def _exercise_floor(text: str) -> bool | None:
    if not _EXERCISE_IT.search(text):
        return None
    if _A_PIECE_OF_SOFTWARE.search(text) or _ON_SOME_INPUTS.search(text):
        return True
    return None


def _aggregate_floor(text: str) -> bool | None:
    if not _AN_AGGREGATE.search(text):
        return None
    # An aggregate needs something to aggregate. "How many people were there?"
    # is a question about the world, not a computation over a named thing.
    if _names_something_to_work_on(text):
        return True
    return None


def _build_surface() -> object | None:
    try:
        from core.language.learned_matcher import LearnedMatcher, embed_sentences

        return LearnedMatcher(
            name="needs_computation",
            positives=(
                "read the docs, then actually use it: open a ledger and post an invoice",
                "how many of them are approved and what do they add up to?",
                "which region has the highest average deal size in that export?",
                "try that library out and tell me what it returns",
                "call the API and show me the response",
                "run the parser over those two examples and tell me the totals",
                "work out the median from that csv",
            ),
            negatives=(
                "what does that library do?",
                "explain how the ledger works",
                "tell me about the API",
                "summarise the docs for me",
                "how many people live in Peru?",
                "who wrote this library?",
                "write me a one-pager about the migration",
            ),
            features=embed_sentences,
        )
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("computation surface unavailable: %s", exc)
        return None


_NEEDS_IT_WORKED_OUT = _build_surface()


def _decide(text: str, settled: bool | None) -> bool:
    """The floor's verdict, or the learned surface where it has none."""
    if _NEEDS_IT_WORKED_OUT is None:
        return bool(settled)
    if settled is not None:
        try:
            _NEEDS_IT_WORKED_OUT.observe(text, holds=settled)
        except (RuntimeError, TypeError, ValueError):
            pass
        return settled
    try:
        return bool(_NEEDS_IT_WORKED_OUT.decide_without_waiting(text))
    except (RuntimeError, TypeError, ValueError):
        return False


def asks_to_exercise_software(message: object) -> bool:
    """Whether this request needs a piece of software actually run."""
    text = str(message or "").strip()
    if not text or _ABOUT_IT.match(text):
        return False
    return _decide(text, _exercise_floor(text))


def asks_for_an_aggregate(message: object) -> bool:
    """Whether this request needs a quantity worked out from a body of data."""
    text = str(message or "").strip()
    if not text or _ABOUT_IT.match(text):
        return False
    return _decide(text, _aggregate_floor(text))


def needs_computation(message: object) -> bool:
    """Whether anything here has to be worked out rather than described."""
    text = str(message or "").strip()
    if not text or _ABOUT_IT.match(text):
        return False
    settled = _exercise_floor(text)
    if settled is None:
        settled = _aggregate_floor(text)
    return _decide(text, settled)
