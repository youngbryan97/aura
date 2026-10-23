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

__all__ = ["FRONT_PAGES", "acts_asked", "asks_to_go_online", "place_named", "site_named", "the_place_it_continues"]

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


_AN_ADDRESS = re.compile(r"\bhttps?://([^/\s<>\"')\]]+)|\b(www\.[a-z0-9.-]+\.[a-z]{2,})\b|"
                         r"\b([a-z0-9-]+\.(?:com|org|net|io|tv|dev|co|edu|gov|ai|app))\b", re.IGNORECASE)


def place_named(message: object) -> str:
    """The front page of the place ``message`` names itself: an address's site, a domain, or a site named by name;
    ""."""
    text = str(message or "")
    found = _AN_ADDRESS.search(text)
    if found:
        return f"https://{next(group for group in found.groups() if group).lower().rstrip('.,;:!?')}"
    return site_named(text)


#: What a request asks to be done somewhere, by kind: the verb, or the thing it is done to ("a Salad Fingers episode"
#: is something to watch). A verb aimed at the person or at what is in front of them ("watch me", "listen to this")
#: is not about the place, and "read" alone is as often a label or a file as a page.
_NOT_THERE = r"(?!\s+(?:to\s+)?(?:me|us|this|that|him|her)\b)"
_ACTS = {
    "watch": re.compile(rf"\b(?:watch|view)\w*\b{_NOT_THERE}|\b(?:videos?|episodes?|films?|movies?|clips?|channels?|"
                        r"trailers?|cartoons?|documentar\w+|vlogs?|shorts)\b", re.IGNORECASE),
    "listen": re.compile(rf"\blisten\w*\b{_NOT_THERE}|\b(?:songs?|music|albums?|podcasts?)\b", re.IGNORECASE),
    "play": re.compile(r"\b(?:games?|play\s+(?:a|an|the|another|some|one)\b)", re.IGNORECASE),
    "read": re.compile(r"\b(?:articles?|posts?|threads?|stories|story|news)\b", re.IGNORECASE),
}


def acts_asked(message: object) -> set[str]:
    """The kinds of thing ``message`` asks to be done: watching, listening, playing, reading. Where it hands over an
    address, "watch this" is the address."""
    text = str(message or "")
    if _AN_ADDRESS.search(text):
        text = re.sub(r"\b(watch|view|listen)(\w*)\s+(?:to\s+)?(?:this|that)\b", r"\1\2", text, flags=re.IGNORECASE)
    return {act for act, said in _ACTS.items() if said.search(text)}


def the_place_it_continues(message: object, earlier: list[str]) -> tuple[str, str]:
    """Where ``message`` means, when it names no place and goes on with what the conversation was doing somewhere: the
    front page of the latest request in ``earlier`` (oldest first) that named a place, where it asked for the same
    kind of thing, and that request; ("", "") otherwise.

    Bryan, 22 October 2026: after two YouTube videos, "Now Salad Fingers. Pick whichever episode you want" means a third
    on YouTube, as it would to anybody watching with him; read alone, it named no place and never reached her browser.
    """
    if place_named(message):
        return "", ""
    acts = acts_asked(message)
    if not acts:
        return "", ""
    for said in reversed(earlier):
        place = place_named(said)
        if place:
            return (place, said) if acts & acts_asked(said) else ("", "")
    return "", ""
