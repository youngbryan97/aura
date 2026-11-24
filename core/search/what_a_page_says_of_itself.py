"""What a page says it is: the structured data it carries, the links on it, and its text.

A page about one thing usually says so to machines: schema.org data in JSON, with the thing's name, place, price, hours,
pay or last day. That is read first, because it is the page's own statement and takes no reading. Its links are kept with
their words and numbered, so a page that lists things can be followed to the things.
"""
from __future__ import annotations

import json
import re
import urllib.parse
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

__all__ = ["PageSays", "has_lapsed", "page_says"]

#: Kinds of structured data that are about the site or the page and not about a thing on it.
_NOT_A_THING = frozenset({"WebSite", "WebPage", "BreadcrumbList", "SearchAction", "ImageObject", "Organization", "Corporation",
                          "SiteNavigationElement", "CollectionPage", "Person", "ReadAction", "WPHeader", "WPFooter"})
#: The most links kept of a page, and the longest text.
MOST_LINKS, MOST_TEXT = 160, 60000


@dataclass
class PageSays:
    url: str
    title: str = ""
    text: str = ""
    #: Each thing the page states of itself, as flat facts.
    things: list[dict[str, str]] = field(default_factory=list)
    #: (words, address) of each link, the site's own chrome left out.
    links: list[tuple[str, str]] = field(default_factory=list)

    def stated(self) -> str:
        """The things it states, as lines."""
        lines = []
        for thing in self.things[:8]:
            lines.append("; ".join(f"{key}: {value}" for key, value in thing.items()))
        return "\n".join(lines)


def _flat(value: Any, *, deep: int = 0) -> str:
    if isinstance(value, dict):
        if deep >= 3:
            return ""
        parts = [(key, _flat(inner, deep=deep + 1)) for key, inner in value.items() if not str(key).startswith("@")]
        return ", ".join(f"{key} {inner}" if not re.fullmatch(r"name|value|text", str(key)) else inner for key, inner in parts if inner)
    if isinstance(value, (list, tuple)):
        return "; ".join(filter(None, (_flat(inner, deep=deep + 1) for inner in value[:6])))
    text = re.sub(r"<[^>]+>", " ", str(value if value is not None else ""))
    return " ".join(text.split())[:600]


def _things(data: Any) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    if isinstance(data, list):
        for item in data:
            found += _things(item)
    elif isinstance(data, dict):
        if isinstance(data.get("@graph"), list):
            found += _things(data["@graph"])
        kinds = data.get("@type")
        kinds = [kinds] if isinstance(kinds, str) else [str(k) for k in (kinds or [])]
        if "ItemList" in kinds:
            for item in data.get("itemListElement") or []:
                found += _things(item.get("item") if isinstance(item, dict) and isinstance(item.get("item"), dict) else item)
        elif kinds and not set(kinds) <= _NOT_A_THING and "ListItem" not in kinds:
            facts = {"is": ", ".join(kinds)}
            for key, value in data.items():
                if str(key).startswith("@"):
                    continue
                said = _flat(value)
                if said:
                    facts[str(key)] = said
            if len(facts) > 2:
                found.append(facts)
    return found


def page_says(html: str, url: str) -> PageSays:
    """What a page says, from its HTML."""
    from bs4 import BeautifulSoup

    from core.search.page_text import _html_to_text

    soup = BeautifulSoup(html, "html.parser")
    says = PageSays(url=url, title=" ".join((soup.title.get_text() if soup.title else "").split())[:220])
    for script in soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)}):
        try:
            says.things += _things(json.loads(script.string or script.get_text() or ""))
        except ValueError:
            continue  # not a failure: a page's broken data is the page's, and its text is read instead
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        words = " ".join(anchor.get_text(" ").split())
        address = urllib.parse.urljoin(url, str(anchor["href"])).split("#")[0]
        if len(words) < 6 or not address.startswith("http") or address.rstrip("/") == url.rstrip("/") or address in seen:
            continue
        seen.add(address)
        says.links.append((words[:160], address))
        if len(says.links) >= MOST_LINKS:
            break
    says.text = _html_to_text(html)[:MOST_TEXT]
    return says


def has_lapsed(thing: dict[str, str], *, today: date | None = None) -> bool:
    """Whether a thing states a last day that has passed."""
    today = today or date.today()
    for key in ("validThrough", "endDate", "expires", "availabilityEnds"):
        said = str(thing.get(key) or "").strip()
        found = re.match(r"(\d{4}-\d{2}-\d{2})", said)
        if found:
            try:
                if datetime.strptime(found.group(1), "%Y-%m-%d").date() < today:
                    return True
            except ValueError:
                continue
    return False
