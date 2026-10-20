"""Whether the turn asks her to go somewhere online: a site, a page, a search on the web.

The effect ceiling (core/phases/response_contract.py ``requested_effect_ceiling``) says what a turn may do without
the person having asked for that effect, and lifts it where they asked: "asking for a page to exist is asking for
that effect". Asking her to go to a site is asking for her browser in the same way.

LIVE, 20 October 2026, the watching demo's dress rehearsal: "Let's watch something together. Go to YouTube and find
a random video from Allen Pan" ran under the self-service ceiling; her own reading of it named the browser, and the
browser was withheld as above that ceiling. The turn answered in words, two minutes later, with no video found.
A request that named an address went to her browser by another route, so the films on her own machine were watched
and the demo's requests, which name a site, were not.
"""

from __future__ import annotations

import re

__all__ = ["asks_to_go_online"]

#: A site or the web, named: an address, a domain, a name of a site as people say it, or the web itself.
_A_PLACE_ONLINE = (
    r"(?:https?://\S+|www\.\S+|[a-z0-9-]+\.(?:com|org|net|io|tv|dev|co|edu|gov|ai|app)\b|"
    r"youtube|reddit|wikipedia|google|twitter|x\.com|instagram|tiktok|github|amazon|netflix|twitch|"
    r"the\s+(?:web|internet|net)|online)"
)

#: Going there, or doing something there: the verbs people use to send somebody to a place online.
_GO_THERE = re.compile(
    rf"\b(?:go|head|hop|jump)\s+(?:over\s+)?(?:to|on|onto)\s+{_A_PLACE_ONLINE}|"
    rf"\b(?:open|visit|browse|pull\s+up|load|check|search|look\s+(?:up|through|at))\s+(?:up\s+)?(?:the\s+)?{_A_PLACE_ONLINE}|"
    rf"\b(?:find|search\s+for|look\s+(?:for|up)|watch|read|pick|choose|play)\b[^.?!\n]{{0,80}}\b(?:on|from|in)\s+{_A_PLACE_ONLINE}|"
    rf"\b(?:look\s+it\s+up|search)\s+online\b|"
    # An address handed to her is a page to go to.
    rf"https?://\S+",
    re.IGNORECASE,
)


def asks_to_go_online(message: object) -> bool:
    """Whether ``message`` asks her to go to a site, a page or the web, or to do something there."""
    return bool(_GO_THERE.search(str(message or "")))
