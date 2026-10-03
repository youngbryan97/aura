"""A page can be READ, or it can be WORKED. They are different requests.

MEASURED live 2026-08-18. "go take it for real:
https://www.16personalities.com/free-personality-test — work through the whole
thing, answer every question as yourself" was classified as a search, because
`has_url` alone forces `requires_search`. The page was fetched, synthesis
produced nothing usable for "take the test", and the person got "I couldn't get
to an answer I'd stand behind on that one."

Search was never going to serve that request. A questionnaire's second screen
does not exist until you answer the first, so there is nothing to fetch and
nothing to summarise — the answer has to be produced by acting.

The distinction is one this runtime already draws one layer down, in
BrowserAuthority: a read needs no lease, while a click "changes state on the far
side and needs a lease". This is that same line, applied where the request is
classified rather than where it is executed.

Retrieval phrasings stay with search, which serves them better. What this
recovers is only the case search cannot serve at all.
"""

from __future__ import annotations

import functools
import re
from typing import Any

__all__ = ["page_interaction_target", "asks_to_act_on_a_page"]

#: A page named outright. Callers need where it starts, not merely that it is
#: present, because that is the page to open.
_EXPLICIT_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+", re.IGNORECASE)

#: A site named as a place, without a link: "on openpsychometrics.org", "at
#: www.example.co.uk", "go to example.com". Named that way it is where the
#: request is to be done, and she can find the rest from its front page. The
#: preposition is what makes it a place: "setup.py" alone is a file, and
#: "run setup.py" names nothing to open.
_NAMED_SITE_RE = re.compile(
    r"\b(?:on|at|from|via|visit|to|open)\s+((?:www\.)?(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"([a-z]{2,24}))\b(?![./-]\w)",
    re.IGNORECASE,
)

#: Endings that name a file far more often than a country: a place given as
#: one of these is left alone.
_FILE_ENDINGS = frozenset(
    "py js ts md txt json csv pdf png jpg jpeg gif html htm yaml yml toml sh rb "
    "rs go java kt swift cpp log zip doc docx xls xlsx ppt pptx".split()
)

#: Verbs that change something on the far side of a page rather than read it.
#:
#: Deliberately about the ACT, not about any site or task. "Take a personality
#: test" is not a category here; "take", "complete", "submit" and "answer" are
#: things one does to a page, and a checkout, a survey, a signup wizard and a
#: booking flow are all the same request wearing different nouns.
_INTERACTION_VERB_RE = re.compile(
    r"\b(?:take|complete|finish|fill(?:\s+(?:in|out))?|answer|submit|apply|"
    r"sign\s*(?:up|in)|log\s*in|register|book|order|buy|checkout|check\s+out|"
    r"vote|rate|review|post|comment|subscribe|unsubscribe|click|select|choose|"
    r"toggle|enable|disable|configure|set\s+up|walk\s+through|work\s+through|"
    r"go\s+through|play|solve|do\s+it|do\s+the)\b",
    re.IGNORECASE,
)

#: What comes before an interaction word when it is not an instruction to her:
#: an infinitive inside a question about whether to act ("whether to sign up"),
#: somebody else's action ("before I buy it"), or a determiner, which makes the
#: word a noun and not a verb at all ("the checkout page", "your order").
#: "you" is not here, because "you should take it" is an instruction.
_NOT_TOLD_TO_HER_RE = re.compile(
    r"(?:\bto|\bi|\bwe|\bthey|\bhe|\bshe|\bsomeone|\bpeople"
    r"|\bthe|\ba|\ban|\bthis|\bthat|\bmy|\byour|\btheir|\bits|\bno|\bany)"
    r"\s+$",
    re.IGNORECASE,
)

#: Phrasings that are unambiguously about getting the page's CONTENT back. When
#: one of these is present the request is retrieval even if an action verb also
#: appears — "read it and tell me whether to sign up" is a reading.
_RETRIEVAL_RE = re.compile(
    r"\b(?:summari[sz]e|summary of|what does .{0,20}say|tell me about|"
    r"read (?:me |out )?(?:it|the|this|that)|look up|find out|research|"
    r"what'?s on|quote)\b",
    re.IGNORECASE,
)


def page_interaction_target(text: Any) -> str:
    """The page this request wants ACTED ON, or "" when it wants reading.

    Both halves are required. A URL with no action verb is a page to read; an
    action verb with no URL names no page to open and belongs to whatever other
    routing the turn would have had. A site named as a place counts as its
    front page.
    """

    body = str(text or "")
    match = _EXPLICIT_URL_RE.search(body)
    if match:
        page = match.group(0).rstrip(".,;:!?")
    else:
        # A site named without its link. Where on it the thing is, she finds
        # from its front page: "take the X test on a-site.org" named the site
        # and the test and no address, and was answered in conversation
        # instead of being done (live, 26 Sep).
        page = ""
        for named in _NAMED_SITE_RE.finditer(body):
            if named.group(2).lower() not in _FILE_ENDINGS:
                page = "https://" + named.group(1).rstrip(".").lower()
                break
        if not page:
            return ""
    # An interaction word she was not told to act on is not a request to act.
    # A bare match let "what does the sign up flow on example.com look like" —
    # a question about a page, with "sign up" as a noun in it — name a page to
    # be worked.
    if _told_to_do_it(body) or _asked_to_use_it(body):
        return page
    return ""


#: What a request that wants a page USED says, as a sentence her reader can test
#: a request against. One sentence for every kind of page work, so nothing here
#: names a verb, a site or a task.
_USE_IT = "The speaker wants the listener to do something on the website."

#: Asking only for what a page already says. Kept narrow on purpose: "find out"
#: is not here, because "find out your type on a-site.org" is a test to take.
_READS_ONLY_RE = re.compile(
    r"\b(?:summari[sz]e|summary of|what does .{0,20}say|tell me about|"
    r"read (?:me |out )?(?:it|the|this|that|what)|look up|research|what'?s on|quote)\b",
    re.IGNORECASE,
)

#: A question that opens like one: about a page rather than an instruction,
#: unless its mood says it asks her to act ("can you do the test on ...?").
_OPENS_AS_A_QUESTION_RE = re.compile(
    r"^\s*(?:what|who|whom|whose|which|where|when|why|how|is|are|was|were|does|do|"
    r"did|should|would|could|can|will|has|have)\b",
    re.IGNORECASE,
)


@functools.lru_cache(maxsize=256)
def _asked_to_use_it(body: str) -> bool:
    """Whether the request, read for what it means, asks her to use the page.

    The verbs above are one way of saying it and people have a hundred. Measured
    2 October 2026 on requests that name a site: the verb list sent 22 of 35
    held-out requests the right way, and missed "go to openpsychometrics.org and
    find out your type", "head over and work your way through the test",
    "try out the quiz", "see what type you get" — each answered in conversation
    instead of done. Read by meaning, with the narrow reading veto below, 33 of
    35 went the right way.

    Sending a reading to the lane that acts costs time; sending a task to the
    lane that reads loses the task. So the reader decides only what the words
    leave open, and a request that only asks what a page says, or a question
    that is not asking her to act, stays a reading.
    """
    if _READS_ONLY_RE.search(body):
        return False
    try:
        from core.conversation.request_mood import RequestMood, assess_request_mood

        directive = assess_request_mood(body).mood is RequestMood.DIRECTIVE
    except (ImportError, AttributeError, TypeError, ValueError):
        directive = False
    if not directive and (body.strip().endswith("?") or _OPENS_AS_A_QUESTION_RE.match(body)):
        return False
    try:
        from core.self.how_her_words_stand import chance_it_follows

        chance = chance_it_follows(body, _USE_IT)
    except (ImportError, AttributeError, TypeError, ValueError):
        return False
    # More likely than not, which is the only line that needs no choosing.
    return chance is not None and chance > 0.5


def _told_to_do_it(body: str) -> bool:
    """Whether an interaction verb here is an instruction given to HER.

    The retrieval veto above exists so that asking about a page is not sent to
    the lane that acts on one, and "read it and tell me whether to sign up" is
    the case it was written for: the reading is the request and the signing up
    is what she is being asked about.

    It vetoed the opposite case just as hard. LIVE 2026-09-29: "Take the Open
    Extended Jungian Type Scales personality test on openpsychometrics.org …
    When you get your result, read it and tell me whether you think it is
    accurate about you." The same request without its last sentence routed to
    the lane that can work a page; with it, two words — "read it" — sent the
    whole thing to retrieval, and she answered in prose that she could not take
    a test she had not walked through. The more completely the request was
    described, the less of it was recognised.

    A result does not exist until the work is done, so reading one cannot be
    served by fetching. What separates the two is not which clause comes first
    but whether she is being TOLD to do the thing: an imperative, or "you" and a
    verb. An infinitive in a question about whether to act is not, and neither is
    something the person says they will do themselves.
    """
    for found in _INTERACTION_VERB_RE.finditer(body):
        if not _NOT_TOLD_TO_HER_RE.search(body[: found.start()]):
            return True
    return False


def asks_to_act_on_a_page(text: Any) -> bool:
    """Whether this turn is an interaction with a page rather than a lookup."""
    return bool(page_interaction_target(text))
