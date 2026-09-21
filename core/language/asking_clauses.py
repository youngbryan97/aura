"""Split a message into the parts that ask, and the parts that describe.

Two defects came from reading a whole message as one lump.

A message was tested for being a question against all of it, and for what it
was about against all of it, so a question at the end licensed a topic match in
any sentence — the rules of an invented game were answered with the
maintenance queue, because "you cannot move ... directly next" put "you"
within sixty characters of "next".

And a message that asks two things was answered with a computed fact covering
one of them, which then replaced the whole reply. "How long have you been up,
and what have you got going on today?" came back as an uptime figure and
nothing else.

Both need the same thing: the clauses, separately.

LIVE, 2026-08-22: a deck came out titled "I've got a slot at a funders'". The
clause that asked was "Put together five slides for me — who you are…", and
nothing here recognised it, because the openings below are questions and a
handful of imperatives that somebody thought of. Which clause asks is a
judgement about meaning, so it belongs to the learned surface, with the
openings as the floor that settles the obvious cases and teaches it.
"""

from __future__ import annotations

import logging
import re

__all__ = [
    "DELIVERABLE_NOUNS",
    "asking_clauses",
    "asking_part",
    "asks_for_a_quantity",
    "asks_more_than_one_thing",
    "asks_what_the_person_said",
    "recalls_what_the_person_said",
    "states_a_quantity",
]

logger = logging.getLogger(__name__)

#: Openings that make a clause a request even without a question mark.
_ASKING = re.compile(
    r"^\s*(?:what|when|where|which|who|whom|whose|how|why|whats|what's|"
    r"anything|any\s+more|is\s+there|are\s+there|do\s+you|did\s+you|does\s+it|"
    r"have\s+you|has\s+it|will\s+you|would\s+you|can\s+you|could\s+you|"
    r"got\s+anything|tell\s+me|show\s+me|list|give\s+me|explain|describe)\b",
    re.IGNORECASE,
)

#: Things a person asks to be given. Named here once, because the title
#: extractor strips the same nouns and two lists that must agree will not.
DELIVERABLE_NOUNS = (
    r"deck|slides?|presentation|report|memo|summary|document|docs?|"
    r"one[\s-]?pager|onepager|one[\s-]?page|write[\s-]?up|page|checklist|"
    r"outline|plan|draft|app|dashboard|spreadsheet|chart|diagram|script|"
    r"table|letter|email|post|readme|guide"
)

#: A count against a named thing is asking for that many of it: "Six slides",
#: "a one-pager", "4 charts". This is structure, not a judgement about meaning.
_COUNTED_DELIVERABLE = re.compile(
    r"\b(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"\d{1,3})\s+(?:\w+\s+){0,2}?(?:" + DELIVERABLE_NOUNS + r")\b",
    re.IGNORECASE,
)

def names_a_deliverable(text: object) -> bool:
    """Whether the words name a thing to be MADE, rather than mention one.

    An indefinite article or a count is what separates the two. "Create a
    concise report" asks for a report to exist; "what's the plan?" and "then
    report the answer" mention one and ask for words. A reader that matched the
    bare noun could not tell them apart, and read the verb in "check your work,
    then report the answer" as a demand for a document — so a train problem was
    answered with a ticket receipt instead of a number.
    """
    return bool(_COUNTED_DELIVERABLE.search(str(text or "")))


#: Openings that ask for something to be made. The learned surface below is
#: the mechanism; these settle the plain cases and teach it.
_PRODUCING = re.compile(
    r"^\s*(?:please\s+)?(?:put\s+together|write|make|build|draft|prepare|"
    r"produce|create|assemble|knock\s+up|throw\s+together|sketch|outline|"
    r"summari[sz]e|sum|count|compare|work\s+out|figure\s+out|look\s+up|"
    # "Present the numbers" asks for something to be made as much as "write"
    # does. Its absence is what let the live defect through: a deck came back
    # titled "Present you funding panel minutes six" — the request's own words
    # in the order the model found them — and the title extractor asked this
    # floor whether that read as an ask, and it said no.
    r"present|find|check|fix|diagnose|run|open|send)"
    # A space after the verb, not a word boundary. `\b` matches before a
    # hyphen, so "Make-or-Buy Analysis" and "Write-Down Policy" opened with an
    # imperative as far as this was concerned. They are compound nouns and the
    # hyphen is what says so. End-of-string still counts: "Compare." is an ask.
    r"(?=\s|$)",
    re.IGNORECASE,
)

#: Inside one sentence, these join two separate requests: "how long have you
#: been up, AND what have you got going on today?"
_JOINED = re.compile(r"(?:,|;)?\s+(?:and|but|also|then)\s+(?=\w)", re.IGNORECASE)


#: Whether a clause asks for something or reports something.
#:
#: The openings above are the floor. This is the mechanism: a request can open
#: any way a person opens one, and a list of openings is always the list one
#: person thought of.
def _build_surface() -> object | None:
    try:
        from core.language.learned_matcher import LearnedMatcher, embed_sentences

        return LearnedMatcher(
            name="asking_clause",
            positives=(
                "Put together five slides for me",
                "write me a one-pager about the migration",
                "make a deck for Thursday",
                "build something I can show the team",
                "what have you got going on today?",
                "how long have you been up?",
                "look up who founded that company",
                "sum the approved rows for me",
                "I need a checklist by five",
                "see if you can work out why it fails",
            ),
            negatives=(
                "I've got a slot at a funders' meeting Thursday.",
                "I have to present you to a panel in 10 minutes.",
                "we shipped it last week and nobody noticed.",
                "the build has been red since Tuesday.",
                "plain language, no marketing.",
                "it is a small project, only three files.",
                "thanks, that was useful.",
                "my flight lands at six.",
            ),
            features=embed_sentences,
        )
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("asking-clause surface unavailable: %s", exc)
        return None


#: Registered at import, so the background warming pass finds it. Constructing
#: one registers it; it does not load a model.
_ASKS_FOR_SOMETHING = _build_surface()


def _surface() -> object | None:
    return _ASKS_FOR_SOMETHING


def _floor_says(piece: str) -> bool:
    """What the shape of the words settles on its own."""
    return bool(
        _ASKING.match(piece) or _PRODUCING.match(piece) or _COUNTED_DELIVERABLE.search(piece)
    )


def _clause_asks(piece: str, *, settled: bool) -> bool:
    """Whether this clause asks, learned where the openings say nothing."""
    surface = _surface()
    if surface is None:
        return settled
    if settled:
        try:
            surface.observe(piece, holds=True)
        except (RuntimeError, TypeError, ValueError):
            # Teaching the surface is a side effect of answering, never the
            # answer. The floor already decided this clause asks, and it is
            # right whether or not the learned surface could record it.
            pass
        return True
    try:
        learned = surface.decide_without_waiting(piece)
    except (RuntimeError, TypeError, ValueError):
        learned = None
    return bool(learned)


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.?!])\s+", str(text or "")) if part.strip()]


def asking_clauses(text: str) -> tuple[str, ...]:
    """Every clause in this message that asks for something.

    A sentence carries its question mark to each of its halves, so both halves
    of "how long have you been up, and what have you got going on today?" are
    returned as requests rather than one being read as a trailing remark.
    """
    found: list[str] = []
    for sentence in _sentences(text):
        interrogative = sentence.endswith("?")
        pieces = [part.strip() for part in _JOINED.split(sentence) if part.strip()]
        opening_asks = bool(interrogative) or _clause_asks(
            pieces[0] if pieces else sentence, settled=_floor_says(pieces[0] if pieces else sentence)
        )
        for index, piece in enumerate(pieces):
            if index == 0:
                asks = opening_asks
            else:
                # LIVE, 2026-08-22: "Put together five slides — who you are …
                # and how we'd know it worked" came back as the last item
                # alone, and the deck was titled from it. A clause after "and"
                # continues the request before it; it only stands on its own
                # when the piece before it was a request too.
                asks = opening_asks and _clause_asks(piece, settled=_floor_says(piece))
            if asks:
                found.append(piece if piece.endswith("?") or not interrogative else piece + "?")
    return tuple(dict.fromkeys(found))


def asking_part(text: str) -> str:
    """The asking clauses as one string, or the whole message if none stand out."""
    clauses = asking_clauses(text)
    return " ".join(clauses) if clauses else str(text or "")


def asks_more_than_one_thing(text: str) -> bool:
    return len(asking_clauses(text)) > 1


#: A clause whose answer is a measurement: the interrogatives of amount,
#: length, distance, speed, age and time, and the nouns that name a quantity.
_ASKS_A_QUANTITY = re.compile(
    r"\bhow\s+(?:long|many|much|far|fast|often|old|big|tall|heavy|wide|deep|soon|late|high|large|small|quickly|slowly)\b"
    r"|\bwhat\s+(?:time|percentage|percent|fraction|proportion|share|ratio|rate|speed|distance|"
    r"temperature|cost|price|total|sum|difference|probability|odds|average|mean|median)\b"
    r"|\b(?:at\s+what|in\s+what)\s+(?:rate|speed|time|year|month|hour)\b"
    r"|\bwhen\s+(?:will|would|does|do|did|is|are|was|were|should)\b",
    re.IGNORECASE,
)

#: A number as people write one, with or without a unit after it. "240
#: minutes", "4 h", "3,600", "12.5%", "twice" is not one and "1st" is not.
_STATES_A_QUANTITY = re.compile(
    r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!(?:st|nd|rd|th)\b)(?:\s*(?:%|°[CF]?|[A-Za-z]{1,12}\b))?"
    r"|\b(?:zero|none|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|"
    r"fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|dozen|half|"
    r"quarter|third|twice|once)\b",
    re.IGNORECASE,
)


def asks_for_a_quantity(clause: str) -> bool:
    """Whether this clause is answered by a number.

    "How long until it is empty?" has no non-numeric right answer. A reply
    that says "240 minutes" has engaged it whether or not it repeats "long",
    "empty" or "full" — LIVE 2026-09-19, a correct four-word answer to a
    word problem was rejected as an unanswered part and destroyed, because
    coverage was measured as shared words.
    """
    return bool(_ASKS_A_QUANTITY.search(str(clause or "")))


def states_a_quantity(text: str) -> bool:
    """Whether the text carries a number, with or without its unit."""
    return bool(_STATES_A_QUANTITY.search(str(text or "")))


#: "what preference did I state?", "what did I just tell you?", "which of
#: those did I ask for?" — a question whose answer is the CONTENT of
#: something the person said, not the word they used for it.
_ASKS_WHAT_THE_PERSON_SAID = re.compile(
    r"\b(?:what|which)\b[^?]{0,80}?\b(?:did\s+)?i\b[^?]{0,40}?"
    r"\b(?:say|said|state|stated|tell|told|mention|mentioned|ask(?:ed)?(?:\s+for)?|"
    r"want(?:ed)?|prefer(?:red)?|request(?:ed)?)\b",
    re.IGNORECASE,
)

#: "you want answers short and concrete", "you asked for X", "you told me Y".
#: The attribution and at least two words of what was attributed: "And you
#: want" at the end of a truncated draft has recalled nothing.
_RECALLS_WHAT_THE_PERSON_SAID = re.compile(
    r"\byou(?:'ve|\s+have)?\s+"
    r"(?:said|stated|told\s+me|mentioned|asked(?:\s+for)?|want(?:ed)?|prefer(?:red)?|"
    r"requested|like|liked)\b\s+(?:\w+\W+){2,}",
    re.IGNORECASE,
)


def asks_what_the_person_said(clause: str) -> bool:
    """Whether this clause asks for the content of something the person said.

    The answer to "what preference did I state?" is the preference — "you
    want answers short and concrete" — and it contains neither "preference"
    nor "state". LIVE 2026-09-20: a correct two-part recall was rejected as
    an unanswered part four times for that, and the turn ended with nothing
    served.
    """
    return bool(_ASKS_WHAT_THE_PERSON_SAID.search(str(clause or "")))


def recalls_what_the_person_said(body: str) -> bool:
    """Whether this text attributes something, with content, to the person."""
    return bool(_RECALLS_WHAT_THE_PERSON_SAID.search(str(body or "")))
