"""Getting to where a task happens, as a decision rather than a jump.

A goal that names a place is not doing anything until she is there. Somebody
had to open the game before she could play it, and that is the part of the
task she should be doing herself.

Naming a URL makes this simple: open it, look, and confirm the page is the
one asked for. Naming only the thing — a game, a site, a tool — makes it a
search, and search markup is written by whoever wants to be found. The
browser controller refuses to open a scraped link for exactly that reason:
choosing a destination out of search results is a decision, and it does not
make it on anyone's behalf.

So it is made here, the same way every other decision is. The vetted results
are the options, choosing between them goes through the ordinary
deliberation, and arriving is checked by reading the page back rather than
assumed from the click. A destination she cannot justify is not opened.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Sequence
from urllib.parse import urlparse

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.ReachPlace")

URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
#: How many vetted results are worth choosing between. More than a handful is
#: a reading exercise, not a decision.
CANDIDATES = 5


@dataclass
class Reached:
    """Where she ended up, and how she decided to go there."""

    wanted: str
    url: str = ""
    title: str = ""
    arrived: bool = False
    because: str = ""
    considered: tuple[str, ...] = ()
    reason: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def narrate(self) -> str:
        if self.arrived:
            place = self.title or self.url
            return f"Opened {place}" + (f" — {self.because}" if self.because else "")
        return f"I could not get to {self.wanted}: {self.reason}"


def named_url(text: str) -> str:
    """A URL written into the request, if there is one."""
    found = URL_RE.search(str(text or ""))
    return found.group(0).rstrip(".,);") if found else ""


def host_of(url: str) -> str:
    try:
        return (urlparse(str(url or "")).hostname or "").lower().removeprefix("www.")
    except (ValueError, AttributeError):
        return ""


def _results_from(receipt: Any) -> list[dict[str, str]]:
    """The vetted candidates a search handed back."""
    raw = getattr(receipt, "result", "") or ""
    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return []
    rows = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    seen: set[str] = set()
    candidates: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        url = str(row.get("url") or "").strip()
        host = host_of(url)
        if not url or not host or host in seen:
            continue
        seen.add(host)
        candidates.append({"url": url, "title": str(row.get("title") or host).strip()})
    return candidates[:CANDIDATES]


async def _choose_destination(
    wanted: str,
    candidates: Sequence[dict[str, str]],
    *,
    think: Any,
    lived: bool,
) -> tuple[dict[str, str] | None, str, tuple[str, ...]]:
    """Pick which result is the place, through the ordinary deliberation."""
    from core.agency.deliberate_action import ActionOption, Expectation, deliberate  # noqa: PLC0415

    options = [
        ActionOption(
            name=host_of(row["url"]),
            params={"url": row["url"]},
            detail=row["title"][:120],
            expectation=Expectation(
                changed=True,
                contains=(host_of(row["url"]),),
                describes=f"the browser to be on {host_of(row['url'])}",
            ),
        )
        for row in candidates
    ]
    if not options:
        return None, "", ()
    decision = await deliberate(
        f"open the page for: {wanted}",
        "search results, none of them opened yet",
        options,
        think=think,
        control_point="agency.reach_place",
        lived=lived,
    )
    if not decision.reached or decision.chosen is None:
        return None, decision.reason, tuple(option.name for option in options)
    picked = next(
        (row for row in candidates if row["url"] == decision.chosen.params.get("url")), None
    )
    return picked, decision.rationale, tuple(option.name for option in options)


async def reach(
    wanted: str,
    *,
    think: Any = None,
    browser: Any = None,
    lived: bool = True,
) -> Reached:
    """Get to where ``wanted`` happens, and confirm arrival by looking.

    ``wanted`` may be a URL or the name of the thing. A URL is opened; a name
    is searched for, and which result is the place is decided rather than
    assumed.
    """
    wanted = " ".join(str(wanted or "").split())
    outcome = Reached(wanted=wanted)
    if not wanted:
        outcome.reason = "nowhere was named"
        return outcome

    if browser is None:
        try:
            from core.capabilities.browser_controller import get_browser_controller  # noqa: PLC0415

            browser = get_browser_controller()
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("reach_place", exc, action="reach the browser")
            outcome.reason = "the browser is not available"
            return outcome

    url = named_url(wanted)
    if not url:
        try:
            receipt = await browser.search_and_open(wanted, count=CANDIDATES)
        except (RuntimeError, OSError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("reach_place", exc, action="search for where a task happens")
            outcome.reason = f"the search did not run ({type(exc).__name__})"
            return outcome
        candidates = _results_from(receipt)
        if not candidates:
            outcome.reason = "the search returned nothing that could be opened"
            return outcome
        if think is None:
            from core.agency.her_reasoning import her_reasoning  # noqa: PLC0415

            think = her_reasoning()
        picked, because, considered = await _choose_destination(
            wanted, candidates, think=think, lived=lived
        )
        outcome.considered = considered
        if picked is None:
            outcome.reason = because or "she could not say which result was the place"
            return outcome
        url = picked["url"]
        outcome.because = because
        outcome.title = picked["title"]

    try:
        await browser.open_url(url, new_tab=False)
    except (RuntimeError, OSError, AttributeError, TypeError, ValueError) as exc:
        record_degradation("reach_place", exc, action="open the page a task happens on")
        outcome.reason = f"the page would not open ({type(exc).__name__})"
        return outcome

    # Arrival is read back, never assumed. A navigation that reported success
    # and left the browser somewhere else is the failure this checks for.
    try:
        page = await browser.current_page()
    except (RuntimeError, OSError, AttributeError, TypeError, ValueError):
        page = {}
    landed = str(page.get("url") or "")
    outcome.url = landed or url
    outcome.title = str(page.get("title") or outcome.title)
    wanted_host = host_of(url)
    outcome.arrived = bool(wanted_host) and host_of(landed) == wanted_host
    if not outcome.arrived:
        outcome.reason = f"the browser is on {host_of(landed) or 'nothing'}, not {wanted_host}"
    outcome.detail = {"asked_for": url, "landed_on": landed}
    return outcome
