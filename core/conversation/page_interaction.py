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

import re
from typing import Any

__all__ = ["page_interaction_target", "asks_to_act_on_a_page"]

#: A page named outright. Callers need where it starts, not merely that it is
#: present, because that is the page to open.
_EXPLICIT_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+", re.IGNORECASE)

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
    routing the turn would have had.
    """

    body = str(text or "")
    match = _EXPLICIT_URL_RE.search(body)
    if not match:
        return ""
    if _RETRIEVAL_RE.search(body):
        return ""
    if not _INTERACTION_VERB_RE.search(body):
        return ""
    return match.group(0).rstrip(".,;:!?")


def asks_to_act_on_a_page(text: Any) -> bool:
    """Whether this turn is an interaction with a page rather than a lookup."""
    return bool(page_interaction_target(text))
