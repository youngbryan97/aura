"""Whether a request asks for a piece of software to be RUN.

`requested_foundational_domains` admits the code domain for arithmetic and for
finite constraint problems — things that settle by computation. That covers a
sum and a puzzle, and misses the ordinary case of being handed a library and
asked to use it.

LIVE, 2026-08-27: "docs and source are at <path>. Read it, then actually use
it: open a ledger, post an invoice, reverse it, tell me the trial balance." Only
the file domain was recognised, so the only tool offered could read files. The
model tried three times to WRITE a script with it, was vetoed three times, and
the turn ended in "I couldn't get to an answer I'd stand behind" — for a task
the sandbox completes in forty milliseconds.

A library is used by calling it, and calling it is running code. Which phrasings
mean that is a judgement about meaning, so the words below are the floor and the
learned surface is the mechanism.
"""

from __future__ import annotations

import logging
import re

__all__ = ["asks_to_exercise_software"]

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

#: Asking ABOUT it rather than for it to be run.
_ABOUT_IT = re.compile(
    r"^\s*(?:what\s+(?:is|does|are)|explain|describe|tell\s+me\s+about|"
    r"how\s+does|why\s+does|summari[sz]e)\b",
    re.IGNORECASE,
)


#: Running something ON something: "run the parser over those two examples",
#: "call it with these inputs", "try it against the sample". The inputs are the
#: structural tell — you do not hand inputs to a description.
_ON_SOME_INPUTS = re.compile(
    r"\b(?:over|with|on|against|through|using)\s+"
    r"(?:it|them|this|that|these|those|the|a|an|my|your|each|both|\d)",
    re.IGNORECASE,
)


def _floor_says(text: str) -> bool | None:
    """What the words settle, or None when they settle nothing."""
    if _ABOUT_IT.match(text):
        return False
    if not _EXERCISE_IT.search(text):
        return None
    # Either it names a thing that is used by being called, or it hands that
    # thing inputs. Requiring a noun from a list meant "run the parser over
    # those two examples" settled nothing, because nobody had written down
    # "parser".
    if _A_PIECE_OF_SOFTWARE.search(text) or _ON_SOME_INPUTS.search(text):
        return True
    return None


def _build_surface() -> object | None:
    try:
        from core.language.learned_matcher import LearnedMatcher, embed_sentences

        return LearnedMatcher(
            name="exercising_software",
            positives=(
                "read the docs, then actually use it: open a ledger and post an invoice",
                "try that library out and tell me what it returns",
                "work out what this function does when you pass it a negative number",
                "call the API and show me the response",
                "run it with those inputs and tell me the total",
                "put the parser through a couple of examples",
            ),
            negatives=(
                "what does that library do?",
                "explain how the ledger works",
                "tell me about the API",
                "summarise the docs for me",
                "who wrote this library?",
                "write me a one-pager about the migration",
            ),
            features=embed_sentences,
        )
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("exercise surface unavailable: %s", exc)
        return None


_WANTS_IT_RUN = _build_surface()


def asks_to_exercise_software(message: object) -> bool:
    """Whether this request needs a piece of software actually run."""
    text = str(message or "").strip()
    if not text:
        return False
    settled = _floor_says(text)
    if _WANTS_IT_RUN is None:
        return bool(settled)
    if settled is not None:
        try:
            _WANTS_IT_RUN.observe(text, holds=settled)
        except (RuntimeError, TypeError, ValueError):
            pass
        return settled
    try:
        return bool(_WANTS_IT_RUN.decide_without_waiting(text))
    except (RuntimeError, TypeError, ValueError):
        return False
