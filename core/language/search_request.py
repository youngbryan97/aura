"""A request to find something out, and what it asks to find.

Part of the language substrate, in the manner of action_semantics and
proposition_semantics: one bounded question, answered from the structure of the
sentence, abstaining when the structure is not there.

The question is whether the person asked her to go and find something out, and
if so, about what. LIVE 2026-10-03: "Google it. Bam had an 83 point game" was
read as a desktop action because "google" sat in a list of desktop verbs; the
lane opened a results page for the film "It" and reported "Done". "Look it up"
in the same place would have been routed as research. Which verb was used
decided whether she learned anything.

Two readers, the same arrangement as core/conversation/asks_about_the_world.py:

* the instruction's grammar is the floor. An instruction to search opens its
  sentence ("google", "search for", "look up", "find out") and its object is
  what to search for; an object that only points back ("it", "that") takes its
  content from the rest of the request or, failing that, from what was asked
  before. That parse is exact where it applies and silent where it does not;
* a learned surface over her own model's representation decides the cases the
  grammar does not reach ("can you check what time the game starts?", "what
  does the internet say about X"), with a boundary measured by leave-one-out
  and an abstention between the classes. Receipts teach it: a search that ran
  and was read is a labelled example.

What it does not decide: where to search. A request to search her own files,
mail, notes or memory, or something on the screen, names its place, and that
place is read from the substrate's shared object classes
(core/language/concepts.py), not from a list kept here.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from core.language.concepts import mentions_object_class
from core.language.learned_matcher import LearnedMatcher
from core.language.model_features import model_hidden_features

logger = logging.getLogger(__name__)

__all__ = [
    "SearchRequest",
    "asks_about_grounds",
    "asks_to_find_out",
    "asks_what_was_found",
    "read_search_request",
    "teach_from_the_floor",
]

#: The opening of an instruction to search. The object follows it.
_INSTRUCTION = re.compile(
    r"^\s*(?:(?:please|can you|could you|would you|will you|go|and)\s+)*"
    r"(?:google|bing|search(?:\s+(?:the\s+)?(?:web|internet|online))?(?:\s+for)?|"
    r"look\s+(?:it|this|that|them|him|her)\s+up|look\s+up|look\s+into|"
    r"find\s+out(?:\s+about)?)"
    r"\b[\s:,]*",
    re.IGNORECASE,
)

#: An object that only points back at something already said.
_POINTS_BACK = re.compile(
    r"^(?:it|this|that|them|him|her|those|these)(?:\s+up)?[\s.!?]*$|^[\s.!?]*$",
    re.IGNORECASE,
)

#: Places a search can be about that are hers or on her screen, read from the
#: substrate's shared classes.
_LOCAL_PLACES = ("screen", "file", "memory", "email")


@dataclass(frozen=True)
class SearchRequest:
    """What the request asks her to find out, and how that was decided."""

    asked: bool
    #: The text to search for; empty when only what came before can supply it.
    query: str = ""
    #: Where the request said to look, when that place is hers or on screen.
    local_place: str = ""
    decided_by: str = ""

    @property
    def about_the_world(self) -> bool:
        return self.asked and not self.local_place


_FINDS_OUT = LearnedMatcher(
    name="asks_to_find_out",
    positives=(
        "google it",
        "look that up for me",
        "can you find out who won the game last night",
        "search for the latest release notes",
        "check online what time the match starts",
        "what does the internet say about this",
        "go read about the eruption and tell me what happened",
        "research the company before my interview",
    ),
    negatives=(
        "how are you feeling today?",
        "open safari",
        "what do you think about consciousness?",
        "click the search bar",
        "tell me a story about a dragon",
        "what is 7919 times 6421?",
        "remember that my sister's birthday is in May",
        "write a poem about autumn",
    ),
    features=model_hidden_features,
)


#: A question about what a lookup found, as opposed to a request to make one or
#: a question about her training. "What did you learn" right after a search
#: means what the search read; LIVE 2026-10-03 it fetched her weight-training
#: ledger instead, because a phrase list in core/learning/learning_selfreport.py
#: read "what did you learn" as asking about her learning stack.
_ASKS_WHAT_WAS_FOUND = LearnedMatcher(
    name="asks_what_was_found",
    positives=(
        "what did you learn",
        "so what did you find?",
        "what did the article say",
        "tell me what you read",
        "anything interesting in there?",
        "summarize what you found",
        "what does the page say about him",
        "what did it say?",
    ),
    negatives=(
        "how are you feeling today?",
        "have you been training lately?",
        "are your weights improving?",
        "google it",
        "what do you think about consciousness?",
        "what is 7919 times 6421?",
        "tell me a story about a dragon",
        "open safari",
    ),
    features=model_hidden_features,
)


#: The grammar floor: a question about what was found, read or said, with
#: nothing new as its object ("what did you learn", "what did the article say
#: about it?"). Exact where it applies; the learned surface takes the rest.
_FOUND_QUESTION = re.compile(
    r"^\s*(?:(?:so|and|ok|okay|well)[\s,]+)?what\s+(?:did|have|has)\s+"
    # She learns, finds or reads; a source says or shows. "What did you say?"
    # asks her to repeat herself, not what a lookup found.
    r"(?:you\s+(?:learn(?:ed|t)?|find|found|read|discover(?:ed)?)"
    r"|(?:it|they|the\s+\w+(?:\s+\w+)?)\s+(?:say|said|show|showed|find|found))"
    r"(?:\s+(?:from|about|in|on)\s+(?:it|that|this|there|them|him|her|the\s+\w+(?:\s+\w+)?))?"
    r"[\s?.!]*$",
    re.IGNORECASE,
)


#: A question about how she knows what she just said: how sure she is, where
#: it came from, what her source was. LIVE 2026-10-04 "How sure are you about
#: the FTX part, and where did that come from?" went to a web search as
#: written and came back with stories about the exchange's collapse; the page
#: her answer had been read from was not looked at again.
_ASKS_ABOUT_GROUNDS = LearnedMatcher(
    name="asks_about_her_grounds",
    positives=(
        "how sure are you about that?",
        "where did that come from?",
        "what's your source for that",
        "how do you know that?",
        "are you sure about the date?",
        "can you back that up?",
        "did you make that up?",
        "says who?",
    ),
    negatives=(
        "how are you feeling today?",
        "where did you grow up?",
        "what is the source of the Nile?",
        "google it",
        "what did you learn",
        "how reliable is the weather forecast for tomorrow?",
        "tell me a story about a dragon",
        "what is 7919 times 6421?",
    ),
    features=model_hidden_features,
)

#: The grammar floor for it: her certainty, or where something she said came
#: from, with her or what she said as the subject.
_GROUNDS_QUESTION = re.compile(
    r"\b(?:how\s+(?:sure|certain|confident)\s+are\s+you"
    r"|are\s+you\s+(?:sure|certain|positive)(?=\s*[?.!,]|\s*$|\s+(?:about|of|that)\b)"
    r"|where\s+did\s+(?:that|this|it|the\s+\w+(?:\s+\w+)?)\s+come\s+from"
    r"|where\s+did\s+you\s+(?:get|read|find|hear)\s+(?:that|this|it)"
    r"|what(?:'s|’s|\s+is|\s+was|\s+are|\s+were)\s+your\s+sources?"
    r"|how\s+do\s+you\s+know(?:\s+(?:that|this|it))?\s*[?.!]*$)",
    re.IGNORECASE,
)


def asks_about_grounds(text: str) -> bool | None:
    """Whether ``text`` asks how she knows what she said; None when neither reader can tell."""
    message = str(text or "").strip()
    if not message:
        return False
    if _GROUNDS_QUESTION.search(message):
        return True
    return _ASKS_ABOUT_GROUNDS.decide_without_waiting(message)


def asks_what_was_found(text: str) -> bool | None:
    """Whether ``text`` asks what a lookup found; None when neither reader can tell."""
    message = str(text or "").strip()
    if not message:
        return False
    if _FOUND_QUESTION.match(message):
        return True
    return _ASKS_WHAT_WAS_FOUND.decide_without_waiting(message)


def _local_place(text: str) -> str:
    for place in _LOCAL_PLACES:
        if mentions_object_class(text, place):
            return place
    return ""


def read_search_request(text: str, previous_request: str = "") -> SearchRequest:
    """Whether ``text`` asks her to find something out, and what."""
    message = re.sub(r"https?://\S+", " ", str(text or "")).strip()
    if not message:
        return SearchRequest(False)
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", message) if part.strip()]
    rest: list[str] = []
    instructed = False
    target = ""
    for sentence in sentences:
        match = _INSTRUCTION.match(sentence)
        if match is None:
            rest.append(sentence)
            continue
        instructed = True
        found = sentence[match.end():].strip(" .!?")
        if found and not _POINTS_BACK.match(found) and not target:
            target = re.sub(r"\s+(?:up|for me|please)$", "", found, flags=re.IGNORECASE)
    place = _local_place(message)
    if instructed:
        query = target or " ".join(rest).strip(" .!?") or str(previous_request or "").strip()
        return SearchRequest(True, query, place, "instruction")
    learned = _FINDS_OUT.decide_without_waiting(message)
    if learned:
        return SearchRequest(True, message, place, "learned")
    return SearchRequest(False)


def asks_to_find_out(text: str) -> bool:
    """Whether ``text`` asks her to go and find something out in the world."""
    return read_search_request(text).about_the_world


def teach(text: str, *, holds: bool) -> None:
    """Record a request whose nature a receipt settled."""
    try:
        _FINDS_OUT.observe(text, holds=holds)
    except (RuntimeError, TypeError, ValueError) as exc:
        logger.debug("the search-request surface was not taught this example: %s", exc)


def teach_from_the_floor(text: str) -> None:
    """What the grammar read exactly becomes an example for the learned surfaces.

    Called where the reading acted: a search that ran, a question answered from
    earlier reading. The learned surfaces then decide the phrasings the grammar
    does not reach with these among their examples.
    """
    message = re.sub(r"https?://\S+", " ", str(text or "")).strip()
    try:
        if _FOUND_QUESTION.match(message):
            _ASKS_WHAT_WAS_FOUND.observe(message, holds=True)
        elif _GROUNDS_QUESTION.search(message):
            _ASKS_ABOUT_GROUNDS.observe(message, holds=True)
        elif read_search_request(message).decided_by == "instruction":
            _FINDS_OUT.observe(message, holds=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        logger.debug("the floor's reading was not kept as an example: %s", exc)
