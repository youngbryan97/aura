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
* a learned surface over her encoder decides the cases the grammar does not
  reach ("can you check what time the game starts?", "you should be able to
  google them"), from each side described rather than listed, with a boundary
  measured by leave-one-out and an abstention between the classes. Receipts
  teach it: a search that ran and was read is a labelled example.

What it does not decide: where to search. A request to search her own files,
mail, notes or memory, or something on the screen, names its place, and that
place is read from the substrate's shared object classes
(core/language/concepts.py), not from a list kept here.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from core.language.concepts import mentions_object_class
from core.language.learned_matcher import LearnedMatcher
from core.language.model_features import model_hidden_features

logger = logging.getLogger(__name__)

__all__ = [
    "CheckRequest",
    "SearchRequest",
    "asks_about_grounds",
    "asks_to_check",
    "asks_to_find_out",
    "asks_what_was_found",
    "only_points_back",
    "prewarm_search_requests",
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


#: Whether a request asks her to go and find something out, judged by meaning: each side described in the voice a
#: request comes in, not listed, on the encoder her other meaning readers use, with the boundary the matcher measures
#: by leaving each description out in turn (core/language/learned_matcher.py). Where the grammar above does not reach,
#: this decides.
#:
#: LIVE 17 November 2026, "You should be able to google them. All flights were grounded on Friday. Major banks were
#: having outages yesterday. Apple outages today. NSA outage today": the instruction does not open its sentence, so
#: the grammar did not read it; the learned surface before this one read her model's hidden states, which no live turn
#: waits for, so it abstained on every phrasing it had not met; the request went to the desktop lane, which opened a
#: search for "them" and said "Done". Measured before this went in, on 40 requests to find something out and 60 that
#: were not (opening applications and windows, clicks and typing, documents, arithmetic, her own life, small talk,
#: what someone thinks of a team, her view of which was better), none of them used to write these descriptions: it
#: read 25 of the 40 and none of the 60, and abstained on the rest, where the grammar's reading stands. The ones it
#: leaves are mostly questions about the world as it is now ("is ChatGPT down right now?"), which the world's own
#: reader takes (core/conversation/asks_about_the_world.py).
def _her_encoder(texts: Sequence[str]) -> list[list[float]]:
    """Vectors from the encoder her evidence judge holds, only once it is loaded; none otherwise. The shared runtime's
    handle embeds by word counts when a model lane refuses it, and a boundary measured on those decides nothing."""
    from core.cognition.evidence_relevance import vectors_if_ready

    return vectors_if_ready(texts)


_FINDS_OUT = LearnedMatcher(
    name="asks_to_find_out_by_meaning",
    positives=(
        "look something up on the web and find out what is true",
        "search the internet about a news event, a person, a company or a fact",
        "you can check online yourself instead of asking me",
        "go and read what the news and the web say about something that happened",
        "find out who won, what happened, when something starts or what it costs",
        "check a claim against sources online",
        "research a topic and tell me what you found",
        "how something in the world stands right now: news, scores, prices, schedules, opening hours, closures, outages",
        "what people online are saying about something new",
    ),
    negatives=(
        "open an application, a website or a window on my computer and do something in it",
        "click, type, scroll, drag or press keys on the screen",
        "create, write or edit a document, note, file or spreadsheet",
        "what you yourself think, feel, want or remember",
        "answer from what you already know: explain or define something",
        "do arithmetic, convert units or solve a math problem",
        "write a story, a poem or a joke, or brainstorm ideas",
        "greetings, thanks and small talk",
        "my own day, my plans, my health and my feelings",
        "remember something I tell you",
        "describe what is on my screen right now",
        "change a setting, play music, set a timer or take a screenshot",
        "telling you what I or someone else thinks about a person, a team or a thing",
        "asking your own opinion, or which of two things you think was better",
    ),
    features=_her_encoder,
)


def _finds_out_by_meaning(text: str) -> bool | None:
    """True or False when her encoder can say whether ``text`` asks her to find something out; None when it is not
    loaded yet. It never loads a model on a turn: before the encoder is up, the grammar alone stands."""
    try:
        from core.cognition.evidence_relevance import semantic_routing_ready

        if not semantic_routing_ready():
            return None
        return _FINDS_OUT.decide(text)
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("whether this asks to find something out was not measured (%s: %s)", type(exc).__name__, exc)
        return None


def prewarm_search_requests() -> bool:
    """Measure the find-out boundary at boot, after her encoder is up, so no turn pays for it.

    Never raises: a boundary not measured here is measured by the first turn that asks, and chat stays ready.
    """
    try:
        return _FINDS_OUT.decide("Can you look up what happened at the election?") is not None
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.info("the find-out boundary was not measured at boot (%s: %s); a turn measures it", type(exc).__name__, exc)
        return False


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


def asks_about_grounds(text: str, *, grammar_only: bool = False) -> bool | None:
    """Whether ``text`` asks how she knows what she said; None when neither reader can tell.

    ``grammar_only`` asks the floor alone, for a caller that uses it as one.
    """
    message = str(text or "").strip()
    if not message:
        return False
    if _GROUNDS_QUESTION.search(message):
        return True
    if grammar_only:
        return False
    return _ASKS_ABOUT_GROUNDS.decide_without_waiting(message)


def asks_what_was_found(text: str) -> bool | None:
    """Whether ``text`` asks what a lookup found; None when neither reader can tell."""
    message = str(text or "").strip()
    if not message:
        return False
    if _FOUND_QUESTION.match(message):
        return True
    return _ASKS_WHAT_WAS_FOUND.decide_without_waiting(message)


#: A request to check whether something is true, as opposed to a request to look something up or a question about how
#: she knows. LIVE 2026-10-20 "Is it true that the speed of light in a vacuum is exactly 299,792,458 metres per second?
#: Can you check that?" was answered from what she remembered: nothing read it as asking her to check.
_ASKS_TO_CHECK = LearnedMatcher(
    name="asks_to_check",
    positives=(
        "can you check that?",
        "is that true?",
        "is it true that the Great Wall is visible from space?",
        "verify that for me",
        "fact-check this please",
        "check whether Canberra is the capital of Australia",
        "is that actually right?",
        "double-check that claim for me",
    ),
    negatives=(
        "how are you feeling today?",
        "check my email",
        "can you check the oven timer on my screen?",
        "google it",
        "how sure are you about that?",
        "what is 7919 times 6421?",
        "tell me a story about a dragon",
        "what do you think about consciousness?",
    ),
    features=model_hidden_features,
)

#: The grammar floor for it, each a sentence of the request: an instruction to check whose object is what was said
#: ("check that", "verify it for me") or a whether-clause; or a question whether something is true.
_CHECK_INSTRUCTION = re.compile(
    r"^\s*(?:(?:please|so|ok(?:ay)?|and|can you|could you|would you|will you)[\s,]+)*"
    r"(?:check|verify|fact[- ]?check|double[- ]?check|confirm)\b[\s,]*(?P<object>.*?)[\s.!?]*$",
    re.IGNORECASE,
)
_POINTS_AT_WHAT_WAS_SAID = re.compile(
    r"^(?:(?:that|this|it)(?:\s+(?:claim|fact|statement|one|part))?)?(?:\s+(?:for me|online|please|out|again))*$",
    re.IGNORECASE,
)
_WHETHER = re.compile(r"^(?:for me\s+)?(?:whether|if)\s+(?P<claim>.+)$", re.IGNORECASE)
_TRUE_THAT = re.compile(
    r"^\s*(?:so\s+|but\s+)?(?:is|isn'?t)\s+it\s+(?:really\s+|actually\s+)?(?:true|right|correct|accurate|the case|a fact)"
    r"\s+that\s+(?P<claim>.+?)[\s?.!]*$",
    re.IGNORECASE,
)
#: Who said it, before what was said: "I read that X" asks about X, not about the reading.
_REPORTED = re.compile(
    r"^(?:i|we|someone|somebody|people|they|my\s+\w+)\s+(?:read|heard|saw|was told|were told|told me|said|say|says|"
    r"claim(?:s|ed)?)\s+(?:somewhere\s+|online\s+)?that\s+",
    re.IGNORECASE,
)
_IS_THAT_TRUE = re.compile(
    r"^\s*(?:so\s+|but\s+)?(?:is|isn'?t)\s+(?:that|this|it)\s+(?:really\s+|actually\s+)?(?:true|right|correct|accurate|so)"
    r"[\s?.!]*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CheckRequest:
    """Whether a request asks her to check something, the claim when it states one, and whether it points at what
    was said before instead (her last answer, or the request's own statement)."""

    asked: bool
    claim: str = ""
    points_back: bool = False


def asks_to_check(text: str) -> CheckRequest:
    """Whether ``text`` asks her to check whether something is true, and what."""
    message = re.sub(r"https?://\S+", " ", str(text or "")).strip()
    # Each sentence with its own end, so a question is still a question; a figure's comma or point does not end one.
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", message) if part.strip()]
    asked, claim, points_back = False, "", False
    statements: list[str] = []
    for sentence in sentences:
        instruction, said = _CHECK_INSTRUCTION.match(sentence), _TRUE_THAT.match(sentence)
        if said:
            asked, claim = True, claim or said["claim"].strip()
        elif instruction and _WHETHER.match(instruction["object"]):
            asked, claim = True, claim or _WHETHER.match(instruction["object"])["claim"].strip()
        elif (instruction and _POINTS_AT_WHAT_WAS_SAID.match(instruction["object"])) or _IS_THAT_TRUE.match(sentence):
            asked, points_back = True, True
        elif not sentence.endswith("?"):
            statements.append(sentence.rstrip(" .!"))
    if not asked and message and _ASKS_TO_CHECK.decide_without_waiting(message):
        asked, points_back = True, True
    if asked and not claim and statements:
        # "The speed of light is exactly c. Can you check that?": what it points at is said in the request itself.
        claim, points_back = " ".join(statements), False
    return CheckRequest(asked, _REPORTED.sub("", claim), points_back and not claim)


def only_points_back(text: str) -> bool:
    """Whether the object of an instruction to search only points back ("them", "it", "that") and names nothing."""
    return bool(_POINTS_BACK.match(str(text or "").strip()))


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
    learned = _finds_out_by_meaning(message)
    if learned:
        return SearchRequest(True, message, place, "learned")
    return SearchRequest(False)


def asks_to_find_out(text: str) -> bool:
    """Whether ``text`` asks her to go and find something out in the world."""
    return read_search_request(text).about_the_world


def teach_from_the_floor(text: str) -> None:
    """What the grammar read exactly becomes an example for the learned surfaces.

    Called where the reading acted: a search that ran, a question answered from
    earlier reading. The learned surfaces then decide the phrasings the grammar
    does not reach with these among their examples.

    Not the find-out surface: it reads what a request asks for from descriptions
    of each kind, and a request taught to it carries its topic too. Under test on
    17 November 2026, once "Google Bam's 83 point game" was taught, "Which was
    better, Bam's 83 point game or Kobe's 81 point game?" read as a request to
    search, by what it was about.
    """
    message = re.sub(r"https?://\S+", " ", str(text or "")).strip()
    try:
        if _FOUND_QUESTION.match(message):
            _ASKS_WHAT_WAS_FOUND.observe(message, holds=True)
        elif _GROUNDS_QUESTION.search(message):
            _ASKS_ABOUT_GROUNDS.observe(message, holds=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        logger.debug("the floor's reading was not kept as an example: %s", exc)
