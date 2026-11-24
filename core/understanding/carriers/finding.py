"""Things of a kind, found and each checked against what was asked.

What is wanted is read into needs, and a need stands only where its words are the person's own, found in what they said
or gave. Queries find pages; a page that lists things is followed to them; each thing's own page is read, and every need
is judged against it with a quotation that has to be on the page. A figure a need turns on is compared here, in numbers,
whatever was said of it. A page that does not answer, has lapsed, or is for something else is dropped, and why is kept. What is told is
what was checked: the link, what the page says, what it is judged by, and what it does not say.

Rounds go on while they find anything new that fits. Where a round finds nothing, the queries are written again from what
failed, once; after that the search has come to what it comes to.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from core.search.what_a_page_says_of_itself import PageSays, has_lapsed, page_says
from core.understanding.carrying_out import CameTo, Tell
from core.understanding.her_model_between_turns import ask as _ask
from core.understanding.requests_of_her import Part, Request

logger = logging.getLogger("Aura.Understanding.Finding")

__all__ = ["Found", "Judged", "Wanted", "both_ends", "figure_disagrees", "find", "quoted_in", "told", "wanted_by"]

#: How much of a material and of a page one reading holds.
MATERIAL_ROOM, PAGE_ROOM = 7000, 9000
#: How long a search may go on, and how many fit where no number was asked for.
WITHIN_S, ENOUGH = 25 * 60.0, 8
#: How many results of a query are looked at, and how many links of a list.
PER_QUERY, PER_LIST = 8, 6


class Need(BaseModel):
    what: str = Field(description="one thing that has to be so of what is found")
    quote: str = Field(description="the words that ask for it, copied exactly from what the person said or gave")
    must: bool = Field(True, description="false where it is only preferred")
    figure: float | None = Field(None, description="the number it turns on, where it turns on one, in plain units")
    bound: Literal["", "at least", "at most"] = Field("", description="which side of the number is wanted")


class Wanted(BaseModel):
    one_is: str = Field(description="what one of the things is, in a few words")
    how_many: int = Field(0, description="how many were asked for; 0 where no number was given")
    needs: list[Need] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list, description="searches that would find single things of this kind, each a different way in")


class Judged(BaseModel):
    need: int
    stands: Literal["yes", "no", "unclear"]
    quote: str = Field("", description="the words on the page that show it, copied exactly; empty where the page does not say")


class PageRead(BaseModel):
    page_is: Literal["one", "list", "neither"] = Field(description="the page of one such thing, a list of them, or neither")
    name: str = ""
    by: str = Field("", description="who offers it, or where it is")
    facts: list[str] = Field(default_factory=list, description="what a person choosing would want to know, each copied from the page")
    judged: list[Judged] = Field(default_factory=list)
    open_now: Literal["yes", "no", "unclear"] = "unclear"
    why: str = Field("", description="one sentence on how it fits, from the page and what they gave")
    links: list[int] = Field(default_factory=list, description="for a list: the numbers of the links to things that could fit")


@dataclass
class Found:
    url: str
    name: str
    by: str
    facts: list[str]
    judged: dict[int, tuple[str, str]] = field(default_factory=dict)
    why: str = ""


@dataclass
class _Search:
    wanted: Wanted
    fits: list[Found] = field(default_factory=list)
    dropped: list[tuple[str, str]] = field(default_factory=list)
    seen: set[str] = field(default_factory=set)
    read: int = 0


def _plain(text: str) -> str:
    kept = re.sub(r"[^\w$€£%.]+", " ", str(text or "").casefold())
    return " ".join(re.sub(r"(?<!\d)\.|\.(?!\d)", " ", kept).split())


def quoted_in(quote: str, source: str) -> bool:
    """Whether a quotation is in a source, however its spaces and marks were set."""
    quote = _plain(quote)
    return bool(quote) and quote in _plain(source)


def _figures(text: str) -> list[float]:
    """The numbers in a text as amounts: ``100k`` is a hundred thousand, and ``1,400`` is one number."""
    found = []
    for number, scale in re.findall(r"(\d[\d,]*(?:\.\d+)?)\s*([kKmM])?\b", text):
        try:
            value = float(number.replace(",", ""))
        except ValueError:
            continue
        found.append(value * {"k": 1e3, "m": 1e6}.get(scale.lower(), 1.0))
    return found


def figure_disagrees(need: Need, quote: str) -> bool | None:
    """Whether the numbers quoted for a need are on the wrong side of its own; None where they cannot say."""
    if need.figure is None or not need.bound:
        return None
    # A number far below or above the need's is in another unit (an hour's pay, a year in a date), and says nothing here.
    near = [value for value in _figures(quote) if need.figure / 20 <= value <= need.figure * 20]
    if not near:
        return None
    return max(near) < need.figure if need.bound == "at least" else min(near) > need.figure


def _materials(request: Request) -> list[tuple[str, str]]:
    from core.conversation.attachments import kept

    held = []
    for attachment_id in request.materials:
        got = kept(attachment_id)
        if got is not None and got[1].strip():
            held.append((str(got[0].get("name") or "a file"), got[1]))
    return held


def both_ends(text: str, room: int) -> str:
    """A text whole where it fits, and its two ends where it does not: a person's own words stand at one or the other of
    what they paste."""
    return text if len(text) <= room else f"{text[: room // 2]}\n…\n{text[-room // 2 :]}"


def _what_they_gave(request: Request, room: int = MATERIAL_ROOM) -> str:
    parts = [f"They said:\n{both_ends(request.whole, room)}"]
    left = room
    for name, text in _materials(request):
        if left <= 0:
            break
        parts.append(f"They gave {name}:\n{text[:left]}")
        left -= len(text)
    return "\n\n".join(parts)


async def wanted_by(request: Request, ask: Any) -> Wanted | None:
    """What is wanted, as needs whose words are the person's own, and searches to begin with."""
    gave = _what_they_gave(request)
    read = await ask(f"{gave}\n\nThey want things found. What is one of them, what has to be so of it, and what searches "
                     "would find single ones?", Wanted, 1400)
    if not isinstance(read, Wanted):
        return None
    source = request.whole + "\n" + "\n".join(text for _name, text in _materials(request))
    own = [need for need in read.needs if quoted_in(need.quote, source)]
    if len(own) < len(read.needs):
        logger.info("🔎 %d of %d needs were not in the person's words and are not asked of anything.",
                    len(read.needs) - len(own), len(read.needs))
    read.needs = own
    read.queries = list(dict.fromkeys(q.strip() for q in read.queries if q.strip()))
    return read


async def _hits(queries: Sequence[str]) -> list[tuple[str, str, str]]:
    """What the queries find: in her browser, and by the pipeline where her browser finds nothing."""
    from core.search.engines_in_her_browser import search_in_her_browser

    answers = await search_in_her_browser(list(queries), at_most=PER_QUERY, principal="research_pipeline")
    found: dict[str, tuple[str, str, str]] = {}
    for rank in range(PER_QUERY):  # the first of each query before the second of any
        for query in queries:
            rows = answers.get(query) or []
            if rank < len(rows):
                found.setdefault(rows[rank][0].split("#")[0].rstrip("/"), rows[rank])
    if not found:
        from core.search.research_pipeline import ResearchSearchPipeline

        for hit in await ResearchSearchPipeline()._search_candidates(list(queries), num_results=PER_QUERY * 2):
            found.setdefault(hit.url.rstrip("/"), (hit.url, hit.title, hit.snippet))
    return list(found.values())


async def _page(url: str) -> tuple[PageSays | None, str]:
    """The page at an address, and why not where it cannot be read. Rendered where a fetch holds none of it."""
    from core.runtime.reading_public_pages import ThroughHerGateway
    from core.search.research_pipeline import _HEADERS

    why = ""
    try:
        answer = await ThroughHerGateway(headers=_HEADERS, source="finding.page", seconds=15.0).get(url)
        status = answer.status_code
        if 200 <= status < 400:
            says = page_says(answer.text, url)
            if says.things or len(says.text) >= 600:
                return says, ""
            why = "its page is filled in by scripts"
        elif status in (404, 410):
            return None, "its page is gone"
        else:
            why = f"its site answered {status or 'nothing'}"
    except Exception as exc:  # noqa: BLE001 - one page is optional, whatever its server does
        why = f"it could not be fetched ({type(exc).__name__})"
    rendered = await _rendered(url)
    return (rendered, "") if rendered is not None else (None, why)


async def _rendered(url: str) -> PageSays | None:
    from core.capabilities.phantom_browser import PhantomBrowser

    browser = PhantomBrowser(visible=False, principal="research_pipeline")
    try:
        if not await browser.ensure_ready() or not await browser.browse(url) or browser.page is None:
            return None
        await browser.page.wait_for_timeout(1500)
        says = page_says(await browser.page.content(), url)
        return says if says.things or len(says.text) >= 600 else None
    except Exception as exc:  # noqa: BLE001 - one page is optional, whatever its scripts raise
        logger.info("🔎 %s could not be read in her browser: %s", url[:120], type(exc).__name__)
        return None
    finally:
        await browser.close()


def _near_the_kind(page: PageSays, wanted: Wanted, *, at_most: int = 40) -> list[tuple[str, str]]:
    """The links of a page nearest what is wanted, by her encoder where it is ready and in the page's order where not."""
    from core.cognition.evidence_relevance import vectors_if_ready

    if len(page.links) <= at_most:
        return page.links
    text = f"{wanted.one_is}: {'; '.join(need.what for need in wanted.needs)}"
    vectors = vectors_if_ready([text, *(words for words, _address in page.links)])
    if vectors is None or len(vectors) != len(page.links) + 1:
        return page.links[:at_most]
    import numpy as np

    v = np.asarray(vectors, dtype=np.float32)
    near = v[1:] @ v[0] / (np.linalg.norm(v[1:], axis=1) * np.linalg.norm(v[0]) + 1e-9)
    return [page.links[i] for i in sorted(np.argsort(-near)[:at_most])]


async def _read(page: PageSays, wanted: Wanted, gave: str, ask: Any) -> tuple[PageRead | None, list[tuple[str, str]]]:
    needs = "\n".join(f"{i}. {need.what}" for i, need in enumerate(wanted.needs)) or "(none were given)"
    links = await asyncio.to_thread(_near_the_kind, page, wanted)
    numbered = "\n".join(f"[{i}] {words}" for i, (words, _address) in enumerate(links))
    stated = page.stated()
    prompt = (f"{gave}\n\nWanted: {wanted.one_is}. Needs:\n{needs}\n\nA page, {page.url}\nTitle: {page.title}\n"
              + (f"It states of itself:\n{stated[:3000]}\n" if stated else "")
              + f"Text:\n{page.text[:PAGE_ROOM]}\n\nIts links:\n{numbered}\n\nWhat is this page, and how does it stand "
              "against each need?")
    read = await ask(prompt, PageRead, 1100)
    return (read if isinstance(read, PageRead) else None), links


def _checked(read: PageRead, page: PageSays, wanted: Wanted, url: str) -> tuple[Found | None, str]:
    """The thing on a page, with each judgement held to its quotation; or why it is dropped."""
    source = f"{page.title}\n{page.stated()}\n{page.text}"
    if read.open_now == "no":
        return None, "it is no longer open"
    if any(has_lapsed(thing) for thing in page.things):
        return None, "its last day has passed"
    judged: dict[int, tuple[str, str]] = {}
    for verdict in read.judged:
        if not 0 <= verdict.need < len(wanted.needs):
            continue
        need, stands, quote = wanted.needs[verdict.need], verdict.stands, verdict.quote.strip()
        shown = quoted_in(quote, source)
        if stands != "unclear" and not shown:
            stands, quote = "unclear", ""
        if shown:
            wrong = figure_disagrees(need, quote)
            if wrong is not None:
                stands = "no" if wrong else "yes"
        judged[verdict.need] = (stands, quote)
    for i, need in enumerate(wanted.needs):
        stands, quote = judged.setdefault(i, ("unclear", ""))
        if need.must and stands == "no":
            return None, f"it does not meet “{need.what}” (“{quote[:120]}”)"
    facts = [fact for fact in read.facts if quoted_in(fact, source)][:5]
    return Found(url=url, name=(read.name or page.title)[:160], by=read.by[:120] if quoted_in(read.by, source) else "",
                 facts=facts, judged=judged, why=read.why.strip()[:400]), ""


async def _look_through(hits: Sequence[tuple[str, str, str]], search: _Search, gave: str, ask: Any, tell: Tell, *,
                        until: float, enough: int) -> int:
    """Read what was found, following lists to what they list; how many fit."""
    waiting = [(url, False) for url, _title, _around in hits]
    fitted = 0
    while waiting and time.monotonic() < until and len(search.fits) < enough:
        url, listed = waiting.pop(0)
        key = url.split("#")[0].split("?utm")[0].rstrip("/")
        if key in search.seen:
            continue
        search.seen.add(key)
        page, why = await _page(url)
        if page is None:
            search.dropped.append((url, why))
            continue
        read, links = await _read(page, search.wanted, gave, ask)
        search.read += 1
        if read is None:
            search.dropped.append((url, "it could not be read"))
            continue
        if read.page_is == "list" and not listed:
            follow = [links[i][1] for i in read.links if 0 <= i < len(links)][:PER_LIST]
            # What a list names is nearer than another search's results: it is looked at next.
            waiting = [(address, True) for address in follow] + waiting
            continue
        if read.page_is != "one":
            search.dropped.append((url, "it is not one of them"))
            continue
        found, why = _checked(read, page, search.wanted, url)
        if found is None:
            search.dropped.append((url, why))
            continue
        search.fits.append(found)
        fitted += 1
        tell(f"Found one that fits so far: {found.name}.")
    return fitted


def _order(found: Found, wanted: Wanted) -> tuple[int, int]:
    unclear = sum(1 for i, need in enumerate(wanted.needs) if need.must and found.judged[i][0] != "yes")
    preferred = sum(1 for i, need in enumerate(wanted.needs) if not need.must and found.judged[i][0] == "yes")
    return unclear, -preferred


def told(fits: Sequence[Found], wanted: Wanted) -> str:
    """What was found, as it is told: each with its link, what its page says, and what it does not."""
    lines = []
    for n, found in enumerate(fits, 1):
        lines.append(f"{n}. [{found.name}]({found.url})" + (f" — {found.by}" if found.by else ""))
        lines += [f"   - {fact}" for fact in found.facts[:4]]
        if found.why:
            lines.append(f"   - {found.why}")
        unsaid = [wanted.needs[i].what for i, (stands, _quote) in sorted(found.judged.items())
                  if stands == "unclear" and wanted.needs[i].must]
        if unsaid:
            lines.append(f"   - Its page does not say: {'; '.join(unsaid)}.")
    return "\n".join(lines)


async def find(request: Request, tell: Tell, *, ask: Any = None) -> CameTo:
    """Find what the request asks for, each checked; what was dropped and why is kept."""
    from core.rebuilding.her_model import patiently

    ask = patiently(ask or _ask, tell)
    wanted = await wanted_by(request, ask)
    if wanted is None or not wanted.queries:
        return CameTo(how="read what is wanted", came_to="what to look for could not be read",
                      parts=[Part("what is wanted was read", "unmet", "no reading of it came")])
    gave = _what_they_gave(request)
    search, enough = _Search(wanted), max(wanted.how_many, 0) or ENOUGH
    until = time.monotonic() + WITHIN_S
    tell(f"Looking for {wanted.one_is}: " + "; ".join(need.what for need in wanted.needs[:6]) + ".")
    queries, written_again = wanted.queries, False
    while queries and time.monotonic() < until and len(search.fits) < enough:
        gained = await _look_through(await _hits(queries), search, gave, ask, tell, until=until, enough=enough)
        if len(search.fits) >= enough or time.monotonic() >= until or (not gained and written_again):
            break
        written_again = not gained
        queries = await _other_queries(search, queries, gave, ask)
    search.fits.sort(key=lambda found: _order(found, wanted))
    parts = [Part(f"{wanted.one_is}: {found.name}", "met", found.url) for found in search.fits]
    if len(search.fits) < (wanted.how_many or 1):
        short = (wanted.how_many or 1) - len(search.fits)
        parts.append(Part(f"{short} more {wanted.one_is}" if search.fits else wanted.one_is, "unmet", _why_not(search)))
    return CameTo(how=f"searched, read {search.read} pages and checked each against {len(wanted.needs)} needs",
                  came_to=f"{len(search.fits)} fit; {len(search.dropped)} did not", parts=parts, again=False,
                  made={"told": told(search.fits, wanted), "links": [found.url for found in search.fits]})


def _why_not(search: _Search) -> str:
    counts: dict[str, int] = {}
    for _url, why in search.dropped:
        counts[why] = counts.get(why, 0) + 1
    most = sorted(counts.items(), key=lambda item: -item[1])[:4]
    return "; ".join(f"{n} pages: {why}" for why, n in most) or "no page was found for any search"


async def _other_queries(search: _Search, tried: Sequence[str], gave: str, ask: Any) -> list[str]:
    """Other ways in, from what the ones tried came to."""
    came_to = "\n".join(f"- {why}: {url}" for url, why in search.dropped[-14:])
    fits = "\n".join(f"- {found.name}: {found.url}" for found in search.fits)
    again = await ask(f"{gave}\n\nWanted: {search.wanted.one_is}. Searches made:\n" + "\n".join(f"- {q}" for q in tried)
                      + f"\nThat fit:\n{fits or '(none)'}\nThat did not, and why:\n{came_to}\n\nWhat other searches would find "
                      "single ones that fit?", Wanted, 700)
    made = set(tried)
    return [q.strip() for q in (again.queries if isinstance(again, Wanted) else []) if q.strip() and q.strip() not in made]
