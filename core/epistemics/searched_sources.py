"""Reputable sites the operator allowlisted, as sources her check can read (L10): each searched through an
allowlisted search engine for the claim's words, its first page on the site read.

Her check reads only hosts on the operator's read allowlist (core/skills/reach_gateway.py), and until 19 October 2026
the one source it knew was Wikipedia's API, so no claim could be read from two publishers. Here every allowlisted
host that is a reference, official or primary source (core/epistemics/what_she_has_verified.py ``tier_of``) and not
Wikimedia's is a source of its own, found by searching ``site:<host>`` on whichever allowlisted engine answers, as
her lookups search (core/skills/looking_it_up.py). Nothing here knows a site; the operator's allowlist says which.
"""
from __future__ import annotations

import logging
import urllib.parse
from typing import Any

logger = logging.getLogger("Aura.Epistemics.Searched")

_EVIDENCE_CHARS = 2000


def _host(url: str) -> str:
    return (urllib.parse.urlparse(str(url or "")).hostname or "").lower()


def _on(site: str, url: str) -> bool:
    host = _host(url)
    return host == site or host.endswith("." + site)


def _read_hosts(policy: Any) -> set[str]:
    try:
        return {str(h).lower() for h in (getattr(policy, "read_hosts", ()) or ())}
    except TypeError:
        return set()


def _engines(hosts: set[str]) -> list[tuple[str, str, Any]]:
    """The search engines on the allowlist, other than an encyclopedia's own search."""
    from core.skills.looking_it_up import ENGINES

    return [(name, address, read) for name, address, read in ENGINES
            if name != "wikipedia" and _host(address) in hosts]


class SearchedSource:
    """One reputable site, searched and read."""

    def __init__(self, site: str) -> None:
        self.site = site.lower()
        self.NAME = f"searched:{self.site}"

    def is_permitted(self, policy: Any) -> bool:
        hosts = _read_hosts(policy)
        return self.site in hosts and bool(_engines(hosts))

    async def lookup(self, gateway: Any, terms: list[str]) -> tuple[str, str]:
        """(evidence, address) from the site's first page an engine finds for the terms, or ("", "")."""
        from core.skills.looking_it_up import _AS_A_BROWSER, _text_of

        query = urllib.parse.quote_plus(f"site:{self.site} " + " ".join(terms[:6]))
        for name, address, read in _engines(_read_hosts(getattr(gateway, "policy", None))):
            found = await gateway.get(address.format(q=query), headers=_AS_A_BROWSER)
            if not getattr(found, "ok", False):
                continue
            try:
                hits = read(found.body_preview or "")
            except (ValueError, TypeError, AttributeError, KeyError) as exc:
                logger.info("%s's results could not be read: %s", name, exc)
                continue
            for _title, url in hits:
                if not _on(self.site, url):
                    continue
                page = await gateway.get(url, headers=_AS_A_BROWSER)
                text = " ".join(_text_of(page.body_preview or "").split()) if getattr(page, "ok", False) else ""
                if text:
                    return text[:_EVIDENCE_CHARS], url
        return "", ""


def searched_sources(policy: Any) -> list[SearchedSource]:
    """A source for each allowlisted host that is reference, official or primary and not Wikimedia's."""
    from core.epistemics.what_she_has_verified import publisher, tier_of

    sites = []
    for host in sorted(_read_hosts(policy)):
        url = f"https://{host}/"
        if tier_of(url) in ("reference", "official", "primary") and publisher(url) != "wikimedia":
            sites.append(SearchedSource(host))
    return sites
