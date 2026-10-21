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

__all__ = ["FRONT_PAGES", "asks_to_go_online", "site_named"]

#: Sites people send her to by name rather than by address, at their front pages. LIVE 20 October 2026, the dress
#: rehearsal: "Go to YouTube and find a random video from Allen Pan" named no address, so it reached neither the route an
#: addressed request takes to her browser nor a page for her browser to start from.
FRONT_PAGES = {
    "youtube": "https://www.youtube.com", "reddit": "https://www.reddit.com", "wikipedia": "https://en.wikipedia.org",
    "google": "https://www.google.com", "github": "https://github.com", "amazon": "https://www.amazon.com",
    "netflix": "https://www.netflix.com", "twitch": "https://www.twitch.tv", "instagram": "https://www.instagram.com",
    "tiktok": "https://www.tiktok.com", "twitter": "https://x.com",
}

#: A site or the web, named: an address, a domain, a name of a site as people say it, or the web itself.
_A_PLACE_ONLINE = (
    r"(?:https?://\S+|www\.\S+|[a-z0-9-]+\.(?:com|org|net|io|tv|dev|co|edu|gov|ai|app)\b|"
    r"youtube|reddit|wikipedia|google|twitter|x\.com|instagram|tiktok|github|amazon|netflix|twitch|"
    r"the\s+(?:web|internet|net)|online)"
)

#: Going there, or doing something there: the verbs people use to send somebody to a place online.
_GO_THERE = re.compile(
    # Whose place it is may come between: "Go to Allen Pan's YouTube channel" (Bryan, 20 October 2026, the demo).
    rf"\b(?:go|head|hop|jump)\s+(?:over\s+)?(?:to|on|onto)\s+(?:[\w'’.-]+\s+){{0,5}}?{_A_PLACE_ONLINE}|"
    rf"\b(?:open|visit|browse|pull\s+up|load|check|search|look\s+(?:up|through|at))\s+(?:up\s+)?(?:the\s+)?{_A_PLACE_ONLINE}|"
    rf"\b(?:find|search\s+for|look\s+(?:for|up)|watch|read|pick|choose|play)\b[^.?!\n]{{0,80}}\b(?:on|from|in)\s+{_A_PLACE_ONLINE}|"
    rf"\b(?:look\s+it\s+up|search)\s+online\b|"
    # An address handed to her is a page to go to.
    rf"https?://\S+",
    re.IGNORECASE,
)


def site_named(message: object) -> str:
    """The front page of the first site ``message`` names by name ("Go to YouTube"); "" when it names none."""
    for word in re.findall(r"[a-z]+", str(message or "").lower()):
        if word in FRONT_PAGES:
            return FRONT_PAGES[word]
    return ""


def asks_to_go_online(message: object) -> bool:
    """Whether ``message`` asks her to go to a site, a page or the web, or to do something there."""
    return bool(_GO_THERE.search(str(message or "")))
