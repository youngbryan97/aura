"""The sources the encyclopedia cites for a claim, read where they are published (L10).

A claim is verified only when two independent publishers support it and one is reference, official or primary
(core/epistemics/what_she_has_verified.py). Wikipedia is one publisher, and her open search was the way to a second.
LIVE 10 November 2026 the search engines stopped giving a program anything to read: DuckDuckGo answered her requests
with a page asking whether a person was searching, and Bing answered them with results for other searches entirely
(internet speed tests for "the speed of light in a vacuum is exactly 299,792,458 metres per second", Google Flights
for its figure in quotes). Nothing here tries to look like a person to them.

The article that supports a claim says where it got it. Its wikitext carries each citation beside the sentence it
backs, a reused one by name, with its address. So this reads the paragraphs of the article that speak of the claim,
its figures first and then most of its words, follows their citations to the addresses they give, and reads those
whose publishers her list counts (the operator's reputable sources), each publisher once. NIST's page for the speed
of light is NIST's, not Wikipedia's, however it was found. An archived copy, a scanned book or a person's own page is
passed over, as her list says, and so is a PDF, which the gateway does not turn into text.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.parse
from collections.abc import AsyncIterator
from typing import Any

logger = logging.getLogger("Aura.Epistemics.Cited")

__all__ = ["WhatTheArticleCites", "cited_for", "cited_sources"]

#: A paragraph of the article's own prose, not a template, a table row or an image.
_NOT_PROSE = ("{{", "|", "{|", "[[File", "[[Image", "==")
#: A citation in place, its whole body; a citation named and reused; a citation defined by name.
_REF_BODY = re.compile(r"<ref(?:\s[^>/]*)?>(.*?)</ref>", re.DOTALL)
_REF_NAMED = re.compile(r'<ref\s+name\s*=\s*"?([^"/>]+?)"?\s*/?>')
#: An address a citation gives: a template's url= field, or a bracketed external link.
_ADDRESS = re.compile(r"url\s*=\s*(https?://[^\s|}<]+)|\[(https?://[^\s\]<]+)")


def _figures(text: str) -> list[str]:
    """The claim's numbers as digits alone, so "299,792,458" finds {{val|299792458}}."""
    return [re.sub(r"[,\s]", "", f) for f in re.findall(r"\d[\d,\s]*\d|\d", text) if len(re.sub(r"\D", "", f)) >= 2]


def cited_for(wikitext: str, claim: str) -> list[str]:
    """The addresses an article's citations give for the paragraphs that speak of ``claim``, the closest first."""
    from core.epistemics.searched_sources import keywords

    figures = _figures(claim)
    words = {w.lower() for w in re.findall(r"[A-Za-z]{3,}", keywords(claim))}
    ranked: list[tuple[int, int, str]] = []
    for paragraph in re.split(r"\n\s*\n", str(wikitext or "")):
        if paragraph.lstrip().startswith(_NOT_PROSE):
            continue
        digits = re.sub(r"[,\s]", "", paragraph)
        found = sum(1 for f in figures if f in digits)
        if figures and not found:
            continue
        said = {w.lower() for w in re.findall(r"[A-Za-z]{3,}", _REF_BODY.sub("", paragraph))}
        shared = len(words & said)
        if shared or found:
            ranked.append((found, shared, paragraph))
    ranked.sort(key=lambda row: (-row[0], -row[1]))
    addresses: list[str] = []
    for _found, _shared, paragraph in ranked:
        bodies = _REF_BODY.findall(paragraph)
        for name in dict.fromkeys(_REF_NAMED.findall(paragraph)):
            defined = re.search(rf'<ref\s+name\s*=\s*"?{re.escape(name.strip())}"?\s*>(.*?)</ref>', wikitext, re.DOTALL)
            if defined:
                bodies.append(defined.group(1))
        for body in bodies:
            for field, bracketed in _ADDRESS.findall(body):
                addresses.append((field or bracketed).rstrip(".,;)"))
    return list(dict.fromkeys(addresses))


class WhatTheArticleCites:
    """A second publisher for a claim the encyclopedia supports: what its own citations say, read at their source."""

    HOST = "en.wikipedia.org"
    NAME = "cited"

    def is_permitted(self, policy: Any) -> bool:
        from core.epistemics.searched_sources import _reads

        return _reads(policy, self.HOST)

    async def readings(self, gateway: Any, terms: list[str], claim: str = "") -> AsyncIterator[tuple[str, str]]:
        """(evidence, address) for each cited page her list counts, each publisher once, as it is read."""
        from core.epistemics.epistemic_reach import _AS_HER_CHECK, WikipediaSource
        from core.epistemics.searched_sources import (
            _a_words_entry,
            _host,
            _reads,
            where_it_speaks_of,
        )
        from core.epistemics.what_she_has_verified import counts, publisher, tier_of
        from core.skills.looking_it_up import _AS_A_BROWSER, _text_of

        if not claim:
            return
        policy = getattr(gateway, "policy", None)
        titles = (await WikipediaSource()._titles(gateway, terms, claim))[:1]
        if not titles:
            return
        source = await gateway.get(
            f"https://{self.HOST}/w/api.php?action=query&format=json&formatversion=2&prop=revisions&rvprop=content"
            f"&rvslots=main&redirects=1&titles={urllib.parse.quote(titles[0])}", headers=_AS_HER_CHECK)
        try:
            wikitext = json.loads(source.body_preview or "{}")["query"]["pages"][0]["revisions"][0]["slots"]["main"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            return
        read_from: set[str] = set()
        for url in cited_for(wikitext, claim):
            who = publisher(url)
            if (who in read_from or who == "wikimedia" or not counts(tier_of(url)) or not _reads(policy, _host(url))
                    or urllib.parse.urlparse(url).path.lower().endswith(".pdf") or _a_words_entry(url, claim)):
                continue
            page = await gateway.get(url, headers=_AS_A_BROWSER)
            text = " ".join(_text_of(page.body_preview or "").split()) if getattr(page, "ok", False) else ""
            if text:
                read_from.add(who)
                logger.info("📚 %s cites %s for this; read there.", titles[0], url[:120])
                yield where_it_speaks_of(text, claim), url


def cited_sources(policy: Any) -> list[WhatTheArticleCites]:
    """The cited sources, when the encyclopedia is on the read list."""
    cited = WhatTheArticleCites()
    return [cited] if cited.is_permitted(policy) else []
