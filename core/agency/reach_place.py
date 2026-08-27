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

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from core.runtime.errors import record_degradation
from core.runtime.watched_goal import named_url

logger = logging.getLogger("Aura.ReachPlace")

#: How many vetted results are worth choosing between. More than a handful is
#: a reading exercise, not a decision.
CANDIDATES = 5


@dataclass
class Reached:
    """Where she ended up, and how she decided to go there."""

    wanted: str
    url: str = ""
    title: str = ""
    #: The application the page is in, so whatever acts on it can bring it to
    #: the front. A page she opened and never fronted is a page she cannot
    #: read: the screen belongs to whatever was already there.
    app: str = ""
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


def host_of(url: str) -> str:
    try:
        return (urlparse(str(url or "")).hostname or "").lower().removeprefix("www.")
    except (ValueError, AttributeError):
        return ""


def _usable(rows: Any) -> list[dict[str, str]]:
    """One candidate per host, with the words that describe it."""
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



#: How an arrival is filed, so it can be found again by the same want.
REMEMBERED_AS = "reach"


def _remember_key(wanted: str) -> str:
    return f"{REMEMBERED_AS}:{' '.join(str(wanted or '').lower().split())[:80]}"


def where_she_went_before(wanted: str, *, graph: Any = None) -> str:
    """Somewhere she has already reached for this, if there is one.

    A place she has been is a place she knows how to get to. Searching for it
    again is forgetting: it costs a network round trip, it re-opens a decision
    she already made, and it can land somewhere else this time because search
    results move.
    """
    try:
        if graph is None:
            from core.world_model.acg import acg as graph  # noqa: PLC0415
        rows = graph.query_consequences(_remember_key(wanted)) or []
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("reach_place", exc, severity="info", action="recall where she went before")
        return ""
    for row in rows:
        if not isinstance(row, dict) or not row.get("success"):
            continue
        remembered = str(row.get("outcome") or "").strip()
        if remembered.startswith(("http://", "https://")):
            return remembered
    return ""


def _remember_arrival(wanted: str, url: str, *, graph: Any = None) -> None:
    """File where this want led, so the next one is a visit and not a search."""
    if not url:
        return
    try:
        if graph is None:
            from core.world_model.acg import acg as graph  # noqa: PLC0415
        graph.record_outcome(_remember_key(wanted), "getting to where a task happens", url, True)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("reach_place", exc, severity="info", action="remember where she went")


#: How long to let a page settle before reading back where she is. Long
#: enough for a browser to replace the tab it was showing, short enough that
#: a slow site costs a moment rather than the run.
SETTLING_SECONDS = 4.0

#: How often to look while it settles.
LOOK_EVERY_S = 0.2


async def _once_it_has_settled(browser: Any, going_to: str) -> dict[str, str]:
    """Where the browser is, once it has stopped being where it was.

    Waits for the tab to show the URL that was asked for — or anything the
    server redirected it to — with a title on it. Gives up and reports what it
    can see, because a page that never settles is a fact worth reading too.
    """
    asked_host = host_of(going_to)
    page: dict[str, str] = {}
    until = time.monotonic() + SETTLING_SECONDS
    while True:
        try:
            page = await browser.current_page()
        except (RuntimeError, OSError, AttributeError, TypeError, ValueError):
            page = {}
        landed = str(page.get("url") or "")
        arrived_here = landed == going_to or (bool(asked_host) and host_of(landed) == asked_host)
        if arrived_here and str(page.get("title") or "").strip():
            return page
        if time.monotonic() >= until:
            return page
        await asyncio.sleep(LOOK_EVERY_S)


def _forget_arrival(wanted: str, url: str, *, graph: Any = None) -> None:
    """Unfile a place that turned out not to be there."""
    if not url:
        return
    try:
        if graph is None:
            from core.world_model.acg import acg as graph  # noqa: PLC0415
        graph.record_outcome(
            _remember_key(wanted), "getting to where a task happens", url, False
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "reach_place", exc, severity="info", action="forget a place that is not there"
        )


async def _choose_destination(
    wanted: str,
    candidates: Sequence[dict[str, str]],
    *,
    think: Any,
    lived: bool,
    purpose: str = "",
) -> tuple[dict[str, str] | None, str, tuple[str, ...]]:
    """Pick which result is the place, through the ordinary deliberation.

    ``purpose`` is what she means to do when she gets there, and it is the
    difference between the right page and a page about the right thing.
    LIVE 2026-08-19: asked to find a game and play it, she searched, chose
    the encyclopedia article about the game, and landed somewhere with
    nothing to play. Both results were about 2048; only one of them was
    somewhere you could do the task.
    """
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
    doing = " ".join(str(purpose or "").split())[:160]
    decision = await deliberate(
        f"get to where this can be done: {doing or wanted}",
        f"search results for {wanted!r}, none of them opened yet",
        options,
        think=think,
        knowledge=(
            [f"What has to be possible there — {doing}"] if doing else []
        ),
        control_point="agency.reach_place",
        lived=lived,
    )
    if not decision.reached or decision.chosen is None:
        return None, decision.reason, tuple(option.name for option in options)
    picked = next(
        (row for row in candidates if row["url"] == decision.chosen.params.get("url")), None
    )
    return picked, decision.rationale, tuple(option.name for option in options)


#: How many of the results she ranked are worth opening before giving up.
#: Enough that one dead link does not end the task, few enough that a search
#: which returned nothing usable is not walked end to end.
MOST_TRIES = 3

#: How a page says it is not there.
#:
#: The web has one vocabulary for this and every site uses it, because it is
#: what a person reading the page has to understand. Nothing here is about any
#: particular site: these are the words a server puts in a title when the
#: thing asked for does not exist.
_NOT_THERE = (
    "not found",
    "could not be found",
    "couldn't be found",
    "cannot be found",
    "page doesn't exist",
    "page does not exist",
    "no longer available",
    "page unavailable",
    "error 404",
)


def _says_it_is_not_there(title: str) -> bool:
    """Whether the page in front announces that it is not the thing.

    Read off the title, which is where a server puts it, and matched on whole
    phrases so a page legitimately *about* missing things is not mistaken for
    a missing page.
    """
    said = " ".join(str(title or "").lower().split())
    if not said:
        return False
    # A bare status number only when it is the whole title. A cafe called 404
    # is a real place, and a page whose title merely contains the digits is
    # not announcing anything.
    if said.strip(" -–—|:") == "404":
        return True
    return any(phrase in said for phrase in _NOT_THERE)


async def reach(
    wanted: str,
    *,
    think: Any = None,
    browser: Any = None,
    lived: bool = True,
    purpose: str = "",
    graph: Any = None,
) -> Reached:
    """Get to where ``wanted`` happens, and confirm arrival by looking.

    ``wanted`` may be a URL or the name of the thing. A URL is opened; a name
    is searched for, and which result is the place is decided rather than
    assumed. ``purpose`` says what has to be possible once she is there,
    which is what separates the page for doing a thing from a page about it.
    """
    wanted = " ".join(str(wanted or "").split())
    outcome = Reached(wanted=wanted)
    if not wanted:
        outcome.reason = "nowhere was named"
        return outcome

    #: Everything the search turned up, when there was a search. A URL given
    #: outright or remembered leaves this empty, and then there is one thing
    #: to try rather than several.
    candidates: list[dict[str, str]] = []
    if browser is None:
        try:
            from core.capabilities.browser_controller import get_browser_controller  # noqa: PLC0415

            browser = get_browser_controller()
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("reach_place", exc, action="reach the browser")
            outcome.reason = "the browser is not available"
            return outcome

    # Already there is already arrived.
    #
    # A pursuit that is retried re-runs this, and searching again from a page
    # that is already the right one is how a run in progress gets navigated
    # away from itself. Measured live: a retried pursuit searched a second
    # time, chose differently, and left the game.
    try:
        already = await browser.current_page()
    except (RuntimeError, OSError, AttributeError, TypeError, ValueError):
        already = {}
    here = str(already.get("url") or "")
    asked_host = host_of(named_url(wanted) or wanted)
    if here and asked_host and host_of(here) == asked_host:
        outcome.url = here
        outcome.title = str(already.get("title") or "")
        outcome.arrived = True
        outcome.because = "already there"
        return outcome

    url = named_url(wanted)
    if not url:
        # Somewhere she has already been, before asking anyone.
        known = where_she_went_before(wanted, graph=graph)
        if known:
            url = known
            outcome.because = "I have been here before"
    if not url:
        # Searched without navigating.
        #
        # search_and_open puts the search page itself in a tab, which then IS
        # the page in front — so opening the destination replaced the wrong
        # thing and the arrival check, reading the front tab, found a search
        # engine. Measured live: she searched, and the run ended
        # could_not_get_there with the browser sitting on DuckDuckGo.
        try:
            candidates = await browser.search_results(wanted, count=CANDIDATES)
        except (RuntimeError, OSError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("reach_place", exc, action="search for where a task happens")
            outcome.reason = f"the search did not run ({type(exc).__name__})"
            return outcome
        candidates = _usable(candidates)
        if not candidates:
            outcome.reason = "the search returned nothing that could be opened"
            return outcome
        if think is None:
            from core.agency.her_reasoning import her_reasoning  # noqa: PLC0415

            think = her_reasoning()
        picked, because, considered = await _choose_destination(
            wanted, candidates, think=think, lived=lived, purpose=purpose
        )
        outcome.considered = considered
        if picked is None:
            outcome.reason = because or "she could not say which result was the place"
            return outcome
        url = picked["url"]
        outcome.because = because
        outcome.title = picked["title"]

    # The one she picked, then the others she considered, in the order she
    # ranked them. A result that turns out not to be the place is a reason to
    # try the next one, not a reason to stop: she already did the work of
    # deciding which results could be it.
    rest = [row for row in (candidates or ()) if str(row.get("url") or "") != url]
    tries = [{"url": url, "title": outcome.title}] + rest[: MOST_TRIES - 1]
    outcome.app = str(getattr(browser, "_preferred_browser", "") or "")
    for attempt, row in enumerate(tries):
        going_to = str(row.get("url") or "")
        if not going_to:
            continue
        try:
            await browser.open_url(going_to, new_tab=False)
        except (RuntimeError, OSError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("reach_place", exc, action="open the page a task happens on")
            outcome.reason = f"the page would not open ({type(exc).__name__})"
            continue

        # Arrival is read back, never assumed — and a read-back that happens
        # before the page arrives is not a read-back. Opening returns as soon
        # as the browser accepts the request, so reading straight away gets
        # whatever tab was showing a moment ago: the previous page's title
        # passes every check, and she anchors a whole run to a URL that is
        # about to render an error. LIVE 2026-08-27, twice, on the same
        # remembered dead link.
        page = await _once_it_has_settled(browser, going_to)
        landed = str(page.get("url") or "")
        outcome.url = landed or going_to
        outcome.title = str(page.get("title") or row.get("title") or "")
        outcome.detail = {"asked_for": going_to, "landed_on": landed}
        wanted_host = host_of(going_to)
        if not wanted_host or host_of(landed) != wanted_host:
            outcome.reason = f"the browser is on {host_of(landed) or 'nothing'}, not {wanted_host}"
            continue
        if _says_it_is_not_there(outcome.title):
            # And a place she remembered getting to, which is not there any
            # more, has to stop being the place she goes first. Left in, it
            # sends her to the same dead link every run, and the search she
            # would have done instead never happens.
            _forget_arrival(wanted, going_to, graph=graph)
            # Getting to the right server is not getting to the thing.
            #
            # Arrival was judged on the host alone, so a page that answered
            # "The page could not be found" counted as arrived, and a whole
            # run anchored itself to it: she read a not-found page, found no
            # part of it that answered to her, and reported truthfully that
            # nothing on screen offered a move. LIVE 2026-08-27, on a sliding
            # puzzle she had searched for and chosen correctly.
            outcome.reason = f"{wanted_host} answered with {outcome.title!r}"
            logger.info(
                "%s is not there (%r) — trying the next result", going_to, outcome.title
            )
            continue
        outcome.arrived = True
        if attempt:
            outcome.because = (
                f"{outcome.because + '; ' if outcome.because else ''}"
                f"the first {attempt} did not exist"
            )
        _remember_arrival(wanted, outcome.url, graph=graph)
        return outcome
    return outcome
