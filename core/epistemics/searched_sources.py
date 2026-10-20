"""The open web as her check reads it (L10): one search for the claim's words, and of what it finds, the pages on the
operator's allowlist whose sources count, each publisher once, read in the order found.

Her check reads only hosts on the operator's read allowlist (core/skills/reach_gateway.py). Until 19 October 2026 the
one source it knew was Wikipedia's API, then each allowlisted site was searched with ``site:`` one after another: with
the operator's reputable list (config/reputable_sources.json, a few hundred hosts and every .gov and .edu under its
classes) that is hundreds of searches for one claim. A person checking a fact searches once and reads the reputable
results. Here the claim's words go to the first allowlisted engine that answers (the engines her lookups use,
core/skills/looking_it_up.py); a result is read when its host is allowlisted, its tier is one that counts
(core/epistemics/what_she_has_verified.py ``tier_of``), it is not Wikimedia's (Wikipedia is read through its own
interface), and its publisher has not been read already. The check stops reading once what was read settles the claim.
"""
from __future__ import annotations

import logging
import urllib.parse
from collections.abc import AsyncIterator
from typing import Any

logger = logging.getLogger("Aura.Epistemics.Searched")

_EVIDENCE_CHARS = 2000


def _host(url: str) -> str:
    return (urllib.parse.urlparse(str(url or "")).hostname or "").lower()


def _reads(policy: Any, host: str) -> bool:
    if callable(getattr(policy, "reads", None)):
        return bool(policy.reads(host))
    try:
        return host in {str(h).lower() for h in (getattr(policy, "read_hosts", ()) or ())}
    except TypeError:
        return False


def _engines(policy: Any) -> list[tuple[str, str, Any]]:
    """The search engines the allowlist admits, other than an encyclopedia's own search."""
    from core.skills.looking_it_up import ENGINES

    return [(name, address, read) for name, address, read in ENGINES
            if name != "wikipedia" and _reads(policy, _host(address))]


def keywords(claim: str) -> str:
    """A claim as a search engine is best asked it: its words that say something, and its figures, in its order."""
    import re

    from core.epistemics.epistemic_reach import _content_words

    said = _content_words(claim)
    kept = [t for t in re.findall(r"\d[\d,.]*\d|\d|[A-Za-z][A-Za-z'-]*", claim or "")
            if t[0].isdigit() or t.lower() in said or (t[0].isupper() and len(t) > 1 and t.lower() not in _SMALL)]
    return " ".join(kept)


#: Short words a capital letter does not make a name of, at a sentence's start.
_SMALL = frozenset("a an the is are was were it its in on at of to by for and or but this that these those".split())


def where_it_speaks_of(text: str, claim: str) -> str:
    """The passage of a page that speaks of the claim: the run of three sentences sharing most of its words and
    figures. LIVE 19 October 2026: a page's first 2,000 characters were its menus, and the judge found nothing."""
    import re

    from core.epistemics.epistemic_reach import _numbers, _stems

    if not claim:
        return text[:_EVIDENCE_CHARS]
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    wanted, figures = _stems(claim), _numbers(claim)
    best, at = -1, 0
    for i in range(len(sentences)):
        window = " ".join(sentences[i:i + 3])
        score = len(wanted & _stems(window)) + 2 * len(figures & _numbers(window))
        if score > best:
            best, at = score, i
    return " ".join(sentences[at:at + 3])[:_EVIDENCE_CHARS]


class OpenSearch:
    """The claim searched once on the open web, and the reputable pages it finds read."""

    NAME = "searched:open"

    def is_permitted(self, policy: Any) -> bool:
        return bool(_engines(policy))

    async def found(self, gateway: Any, terms: list[str], claim: str = "") -> list[str]:
        """The addresses the first engine that answers gives, in its order: for the claim's own words in its order,
        then for its rarest words and its figures. LIVE 19 October 2026: asked "Mount Everest is 8,849 metres tall",
        an engine answered for "mount" alone (dictionaries); asked "Everest height 8,849 metres", with Wikipedia,
        Britannica and National Geographic."""
        from core.epistemics.epistemic_reach import _numbers
        from core.epistemics.what_she_has_verified import counts, publisher, tier_of
        from core.skills.looking_it_up import _AS_A_BROWSER

        figures = sorted(_numbers(claim), key=len, reverse=True)[:2]
        queries = list(dict.fromkeys(q for q in (keywords(claim), " ".join([*terms[:4], *figures])) if q.strip()))
        policy = getattr(gateway, "policy", None)
        every: list[str] = []
        for name, address, read in _engines(policy):
            found: list[str] = []
            for query in queries or [" ".join(terms[:8])]:
                answer = await gateway.get(address.format(q=urllib.parse.quote_plus(query)), headers=_AS_A_BROWSER)
                if not getattr(answer, "ok", False):
                    break
                try:
                    found += [url for _title, url in read(answer.body_preview or "")]
                except (ValueError, TypeError, AttributeError, KeyError) as exc:
                    logger.info("%s's results could not be read: %s", name, exc)
                    break
            every = list(dict.fromkeys([*every, *found]))
            # Two publishers of standing besides Wikipedia, which is read on its own, or the next engine is asked as
            # well: LIVE 20 October 2026 one engine's answer for "the speed of light in a vacuum is exactly 299,792,458
            # metres per second" held no reputable page, and the same search a minute later had NIST's last of eleven.
            standing = {publisher(url) for url in every if _reads(policy, _host(url)) and counts(tier_of(url))}
            if len(standing - {"wikimedia"}) >= 2:
                break
        return every

    async def readings(self, gateway: Any, terms: list[str], claim: str = "") -> AsyncIterator[tuple[str, str]]:
        """(evidence, address) for each result worth reading, each publisher once, lazily: the caller stops when settled."""
        from core.epistemics.what_she_has_verified import counts, publisher, tier_of
        from core.skills.looking_it_up import _AS_A_BROWSER, _text_of

        policy = getattr(gateway, "policy", None)
        read_from: set[str] = set()
        for url in await self.found(gateway, terms, claim):
            who = publisher(url)
            if not _reads(policy, _host(url)) or who == "wikimedia" or who in read_from or not counts(tier_of(url)):
                continue
            page = await gateway.get(url, headers=_AS_A_BROWSER)
            text = " ".join(_text_of(page.body_preview or "").split()) if getattr(page, "ok", False) else ""
            if text:
                read_from.add(who)
                yield where_it_speaks_of(text, claim), url


def searched_sources(policy: Any) -> list[OpenSearch]:
    """The open search, when an engine is allowlisted."""
    return [OpenSearch()] if _engines(policy) else []
