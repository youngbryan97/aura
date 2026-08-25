"""Decide whether a turn needs evidence from outside this machine.

LIVE, 2026-08-22. "what can you tell me about the company Hugging Face?
founders, what they sell, rough size. link your sources." ran no search. The
grounding taken was her own source code, the cortex answered from memory in
twenty-six seconds, and the reply read "It was founded by <NAME> and <NAME>"
with no citations — to a question that had asked for them in as many words.

Asked instead as "can you LOOK UP Hugging Face", the same question searched.
The decision rested on pattern lists, and three ordinary ways of asking about
a company matched none of them.

Two things are decided here, and neither is a list of topics or verbs.

The first is an instruction: somebody who asks for sources has asked for
evidence, and an answer without any is a broken promise rather than a style
choice.

The second is structural: a question about a named thing that is not her, not
on this disk, and not a matter of opinion cannot be answered from what she
happens to remember. Names are recognised by their shape, so this works for a
company nobody has heard of as well as for one that ships in a pattern list.
"""

from __future__ import annotations

import re

from core.language.learned_matcher import LearnedMatcher as _LearnedMatcher
from core.language.model_features import model_hidden_features as _model_hidden_features

__all__ = ["asks_for_sources", "asks_about_a_named_thing", "wants_outside_evidence"]

#: Asking where something came from. An instruction, not a topic.
_WANTS_SOURCES = re.compile(
    r"\b(?:cite|citation|citations|sources?|referenced?|references|"
    r"link (?:me )?(?:the|your|to)|show me where|where did you (?:get|find|read)|"
    r"back (?:it|that) up with|with links?)\b",
    re.IGNORECASE,
)

#: Asking for facts rather than an opinion or a feeling.
_ASKS_FOR_FACTS = re.compile(
    r"\b(?:who|what|when|where|which|how many|how much|how big|how old|"
    r"tell me about|what do you know about|what can you tell me about|"
    r"give me (?:a )?(?:rundown|summary|overview|background)|"
    r"look (?:up|into)|research|find out|background on|"
    r"founded|founders?|headquarters|revenue|employees|valuation|"
    r"ceo|founder|owner|acquired|launched|released)\b",
    re.IGNORECASE,
)

#: A name, by its shape rather than by being on a list: several capitalised
#: words, an acronym, or a single capitalised word that is not merely the
#: first word of its sentence.
_NAME_SHAPE = re.compile(
    r"\b(?:[A-Z][a-z0-9&.'-]+(?:\s+[A-Z][a-z0-9&.'-]+)+"
    r"|[A-Z]{2,}[A-Za-z0-9]*"
    r"|[A-Z][a-z0-9&.'-]+)"
)

#: "live" is missing from the bare recency alternation on purpose.
#:
#: It is the one word there with a second, commoner meaning: running, as in
#: "the live runtime", "the live instance", "the live response path". Reading
#: it as recency sent a turn about her own machinery to a web search —
#: "plan how you would debug the live response path" came back as "I don't
#: have grounded results for that yet, and I shouldn't guess", because the
#: contract had already decided the answer was outside. The recency sense is
#: still caught where the word is the object of a lookup, in the
#: "find live ..." branch above it.
_EXPLICIT_EXTERNAL_LOOKUP = re.compile(
    r"\b(?:search(?: the)? (?:web|internet|online)|browse(?: the)? (?:web|internet)|"
    r"look (?:it |this |that )?up|find (?:recent|current|latest|live) )",
    re.IGNORECASE,
)

#: A recency adjective. Weaker than the instructions above: "search the web"
#: says where to go, "the latest X" only says when.
#:
#: It sat in the explicit-lookup pattern and settled the turn before anything
#: else was read. "It would help if you compared the latest runtime incidents"
#: was therefore routed to a web search for her own crash records, exactly as
#: `live` had been before it — same pattern, one word over.
_RECENCY_ADJECTIVE = re.compile(
    r"\b(?:current|latest|recent|today'?s|this week'?s)\b",
    re.IGNORECASE,
)

#: What this runtime keeps its own records of. A recency adjective on one of
#: these asks about her own history, which the web cannot answer and her own
#: logs can. Deliberately the runtime's own vocabulary — the directories under
#: data/error_logs and the subsystems degradations are recorded against —
#: rather than an attempt to enumerate topics the world might contain.
_HER_OWN_RECORDS = re.compile(
    r"\b(?:runtime|uptime|incidents?|crash(?:es)?|stalls?|degradations?|"
    r"logs?|traces?|tracebacks?|errors?|failures?|regressions?|"
    r"tests?|suites?|builds?|deploys?|commits?|branch(?:es)?|"
    r"sessions?|turns?|episodes?|checkpoints?|benchmarks?)\b",
    re.IGNORECASE,
)

_ENTITY_RELATION = re.compile(
    r"\b(?:company|organization|institution|person|found(?:ed|er|ers)?|"
    r"headquarters|revenue|employees|valuation|ceo|owner|acquired|"
    r"background|rundown|overview|"
    r"announce[ds]?|launch(?:ed)?|release[ds]?)\b",
    re.IGNORECASE,
)

#: Capitalised because of where they sit, not because they name anything.
_CAPITALISED_BUT_ORDINARY = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december", "today",
    "tomorrow", "yesterday", "morning", "afternoon", "evening", "tonight",
    "ok", "okay", "hi", "hey", "hello", "thanks", "please", "yes", "no",
}

#: Words that look like names but are the subject of the conversation itself.
_NOT_A_SUBJECT = {
    "i", "you", "we", "us", "me", "my", "your", "our", "yourself", "myself",
    "aura", "luna", "aura luna",
}

#: Openings that are about her rather than about the world.
_ABOUT_HER = re.compile(
    r"\b(?:your(?:self)?|you'?re|you are|you have|you feel|you think|"
    r"how are you|who are you|what are you)\b",
    re.IGNORECASE,
)


def asks_for_sources(message: object) -> bool:
    """Whether the person asked where the answer comes from."""
    return bool(_WANTS_SOURCES.search(str(message or "")))


def _names(message: str) -> list[str]:
    """Candidate names, with the sentence-initial word discounted."""
    found: list[str] = []
    for sentence in re.split(r"(?<=[.?!])\s+", message):
        stripped = sentence.strip()
        if not stripped:
            continue
        first = stripped.split(" ", 1)[0]
        for match in _NAME_SHAPE.finditer(stripped):
            text = match.group(0).strip()
            if text.lower() in _NOT_A_SUBJECT or text.lower() in _CAPITALISED_BUT_ORDINARY:
                continue
            # A single capitalised word that only leads the sentence is
            # capitalisation, not a name.
            if match.start() == 0 and text == first.rstrip(",;:") and " " not in text:
                continue
            # An acronym before a common noun is usually a category modifier,
            # not a referential name: "AI system", "CPU architecture", "RLC
            # design".  The old shape reader treated every such phrase as an
            # external entity and forced a web search for timeless questions.
            # This is a grammatical relation, so it applies to unseen acronyms.
            if text.isupper() and " " not in text:
                remainder = stripped[match.end() :]
                next_word = re.match(r"^[\s,:-]+([a-z][a-z0-9'-]*)", remainder)
                if next_word:
                    continue
            found.append(text)
    return found


def asks_about_a_named_thing(message: object) -> bool:
    """A factual question about something named, that is not her."""
    text = str(message or "")
    if not text.strip() or _ABOUT_HER.search(text):
        return False
    if not _ASKS_FOR_FACTS.search(text):
        return False
    return bool(_names(text))


#: Whether the answer has to come from outside her.
#:
#: The readers above are the floor. This is the mechanism: whether a question
#: can be answered from memory is a judgement about meaning, and a list of
#: fact-words will always be the list somebody thought of.
_NEEDS_OUTSIDE = _LearnedMatcher(
    name="wants_outside_evidence",
    positives=(
        "who founded Hugging Face?",
        "tell me about Anthropic the company",
        "what's the background on Cerebras",
        "how big is Mistral these days",
        "is that startup still going?",
        "what did they announce last week",
    ),
    negatives=(
        "how are you feeling today?",
        "tell me about yourself",
        "what is 7919 * 6367?",
        "what do you think about consciousness?",
        "what have you been working on lately?",
        "read CONTRIBUTING.md and tell me the first rule",
    ),
    features=_model_hidden_features,
)


def wants_outside_evidence(message: object) -> bool:
    """Whether this turn should not be answered from memory alone.

    The readers settle what they can and teach the surface as they go, so
    "is that startup still going?" can reach the same answer as "who founded
    X" without anyone adding a word to a list.
    """
    text = str(message or "")
    if not text.strip():
        return False
    if asks_for_sources(text):
        _teach(text, True)
        return True
    if _EXPLICIT_EXTERNAL_LOOKUP.search(text):
        _teach(text, True)
        return True
    # A turn plainly about her, or about this machine, is settled the other
    # way and is worth teaching too.
    if _ABOUT_HER.search(text):
        _teach(text, False)
        return False
    # A recency adjective says when, not where. It reaches outside unless what
    # it modifies is something this runtime keeps its own record of.
    if _RECENCY_ADJECTIVE.search(text):
        outside = not _HER_OWN_RECORDS.search(text)
        _teach(text, outside)
        return outside
    # The prewarmed local evidence router distinguishes a question about a
    # named/current external fact from a timeless explanation about a concept.
    # It runs before the structural floor because capitalization and acronyms
    # are candidate evidence, not authority over meaning.
    try:
        from core.cognition.evidence_relevance import (
            EXTERNAL_WORLD,
            semantic_routing_ready,
            wants_evidence,
        )

        if semantic_routing_ready():
            routed = wants_evidence(text, EXTERNAL_WORLD)
            if routed:
                _teach(text, True)
                return True
    except (ImportError, RuntimeError, TypeError, ValueError):
        pass

    # No semantic measurement is available.  Keep a conservative structural
    # floor for entity-specific relations, but do not make capitalization or
    # an acronym sufficient by itself.
    if asks_about_a_named_thing(text) and _ENTITY_RELATION.search(text):
        _teach(text, True)
        return True
    try:
        learned = _NEEDS_OUTSIDE.decide_without_waiting(text)
    except (RuntimeError, TypeError, ValueError):
        learned = None
    return bool(learned)


def _teach(text: str, holds: bool) -> None:
    try:
        _NEEDS_OUTSIDE.observe(text, holds=holds)
    except (RuntimeError, TypeError, ValueError):
        pass
