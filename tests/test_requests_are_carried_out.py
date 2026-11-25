"""Searching and reading for a request: what is told is what was checked, and what was not is named."""
from __future__ import annotations

import asyncio
import time
from datetime import date
from types import SimpleNamespace

import pytest

from core.search.engines_in_her_browser import are_decoys, result_links, search_in_her_browser
from core.search.what_a_page_says_of_itself import has_lapsed, page_says
from core.understanding.carriers import finding, reading_whole
from core.understanding.carriers.finding import (
    Judged,
    Need,
    PageRead,
    Wanted,
    both_ends,
    figure_disagrees,
    find,
    quoted_in,
    wanted_by,
)
from core.understanding.carriers.reading_whole import (
    Answer,
    Point,
    SectionRead,
    read_whole,
    sections_of,
    unread,
)
from core.understanding.requests_of_her import Request, RequestsOfHer


def _request(said: str, **over) -> Request:
    return Request(id="r1", conversation="c", said=said, effect="found", opened_at=time.time(), **over)


# ── search ───────────────────────────────────────────────────────────

ROWS = [
    {"href": "https://search.example/about", "words": "About the engine we run here", "around": "About the engine we run here and more"},
    {"href": "https://menu.example/", "words": "A menu link", "around": "A menu link"},
    {"href": "https://r.search.yahoo.com/x/RU=https%3a%2f%2fboats.example%2fhire/RK=2/RS=z", "heading": "Boats for hire in Leith",
     "words": "", "around": "Boats for hire in Leith, by the hour, at the shore"},
    {"href": "https://boats.example/hire", "heading": "Boats for hire in Leith", "around": "Boats for hire in Leith again, twice"},
]


def test_results_are_the_pages_and_not_the_engines_own_links():
    links = result_links(ROWS, engine="https://search.example/search?q={}", at_most=10)
    assert [address for address, _title, _around in links] == ["https://boats.example/hire"]


def test_results_about_one_word_of_a_query_are_decoys():
    def words(text):
        return [w for w in text.lower().replace("-", " ").split() if len(w) > 2]

    hit = lambda title, url="https://x.example/": SimpleNamespace(title=title, snippet="", url=url)  # noqa: E731
    query = "research operations manager remote opening"
    assert are_decoys(query, [hit("Research wiki"), hit("Research dictionary"), hit("Find research")], words)
    assert not are_decoys(query, [hit("Research operations manager, remote"), hit("Operations manager opening")], words)
    assert not are_decoys("tides", [hit("Tides")], words)


def test_an_engine_that_refuses_is_not_asked_again():
    asked: list[str] = []

    class Page:
        async def wait_for_timeout(self, ms):
            return None

        async def evaluate(self, js):
            return [] if "yahoo" in asked[-1] else ROWS

    class Browser:
        page = Page()

        async def ensure_ready(self):
            return True

        async def browse(self, url):
            asked.append(url)
            return True

    found = asyncio.run(search_in_her_browser(["one", "two"], browser=Browser()))
    assert all(found[q] for q in ("one", "two"))
    assert sum("yahoo" in url for url in asked) == 1


# ── a page ───────────────────────────────────────────────────────────

PAGE = """<html><head><title>Harbour pilot - Port of Ness</title>
<script type="application/ld+json">{"@context":"https://schema.org","@graph":[{"@type":"BreadcrumbList"},
{"@type":"JobPosting","title":"Harbour pilot","validThrough":"%s","hiringOrganization":{"@type":"Organization","name":"Port of Ness"},
"jobLocationType":"TELECOMMUTE","baseSalary":{"@type":"MonetaryAmount","currency":"GBP","value":{"@type":"QuantitativeValue","minValue":61000,"maxValue":70000}}}]}</script></head>
<body><main><p>Guide ships into harbour. You will need a licence and ten years at sea. Pay is £61,000 to £70,000 a year.</p>
<p>Work from the tower or from home, two days a week at the port, in all weathers and at all hours of the tide.</p>
<a href="/jobs/2">Deckhand, Port of Ness</a></main></body></html>"""


def test_a_page_states_itself_and_says_when_it_has_lapsed():
    says = page_says(PAGE % "2020-01-01", "https://port.example/jobs/1")
    assert says.things and says.things[0]["is"] == "JobPosting" and "Port of Ness" in says.things[0]["hiringOrganization"]
    assert ("Deckhand, Port of Ness", "https://port.example/jobs/2") in says.links
    assert has_lapsed(says.things[0], today=date(2026, 11, 23))
    assert not has_lapsed(page_says(PAGE % "2030-01-01", "https://x.example").things[0], today=date(2026, 11, 23))


@pytest.mark.parametrize(("figure", "bound", "quote", "wrong"), [
    (60000, "at least", "£61,000 to £70,000 a year", False),
    (80000, "at least", "£61,000 to £70,000 a year", True),
    (100000, "at least", "$90k – $130k", False),
    (1400, "at most", "£1,550 pcm", True),
    (1400, "at most", "£1,250 per month", False),
    (100000, "at least", "$45 an hour", None),
    (None, "", "£61,000", None),
])
def test_a_figure_is_compared_in_numbers(figure, bound, quote, wrong):
    assert figure_disagrees(Need(what="pay", quote="", figure=figure, bound=bound), quote) is wrong


def test_a_quotation_has_to_be_there():
    assert quoted_in("ten years  at sea.", "a licence and ten years at sea")
    assert not quoted_in("twelve years at sea", "a licence and ten years at sea")
    assert not quoted_in("", "anything")


def test_both_ends_of_a_long_text_are_kept():
    text = "Find me one. " + "x " * 4000 + " thank you"
    assert both_ends(text, 400).startswith("Find me one.") and both_ends(text, 400).endswith("thank you")
    assert both_ends("short", 400) == "short"


# ── finding ──────────────────────────────────────────────────────────

def test_needs_that_are_not_the_persons_words_are_not_asked_of_anything():
    async def ask(prompt, schema, n):
        return Wanted(one_is="a post", queries=["a"], needs=[Need(what="pay of 60k", quote="at least 60k"),
                                                             Need(what="a car", quote="company car included")])

    wanted = asyncio.run(wanted_by(_request("Find me a post paying at least 60k"), ask))
    assert [need.what for need in wanted.needs] == ["pay of 60k"]


def _finding(monkeypatch, pages: dict[str, str | None], reads: dict[str, PageRead], hits: list[str]):
    async def page(url):
        html = pages.get(url)
        return (page_says(html, url), "") if html else (None, "its page is gone")

    async def found(queries):
        return [(url, url, "around") for url in hits]

    monkeypatch.setattr(finding, "page_at", page)
    monkeypatch.setattr(finding, "_hits", found)

    async def ask(prompt, schema, n):
        if schema is Wanted:
            if "Searches made" in prompt:
                return Wanted(one_is="a post", queries=[])
            return Wanted(one_is="a pilot's post", queries=["pilot post"],
                          needs=[Need(what="pays at least 60k", quote="at least 60k", figure=60000, bound="at least"),
                                 Need(what="can be done from home", quote="from home")])
        return next(read for url, read in reads.items() if f"A page, {url}\n" in prompt)

    return ask


SAID = "Find me a pilot's post paying at least 60k that I can do from home"


def test_only_what_was_checked_is_told_and_what_failed_is_named(monkeypatch):
    good, low, gone = "https://a.example/1", "https://b.example/2", "https://c.example/3"
    pages = {good: PAGE % "2030-01-01", low: (PAGE % "2030-01-01").replace("£61,000 to £70,000", "£31,000 to £40,000"), gone: None}
    quote_pay, quote_home = "Pay is £61,000 to £70,000 a year", "from the tower or from home"
    reads = {
        good: PageRead(page_is="one", name="Harbour pilot", by="Port of Ness", facts=["ten years at sea", "free lunch"],
                       judged=[Judged(need=0, stands="yes", quote=quote_pay), Judged(need=1, stands="yes", quote=quote_home)]),
        low: PageRead(page_is="one", name="Low pilot", judged=[Judged(need=0, stands="yes", quote="Pay is £31,000 to £40,000 a year")]),
    }
    ask = _finding(monkeypatch, pages, reads, [good, low, gone])
    came = asyncio.run(find(_request(SAID), lambda line: None, ask=ask))
    assert came.made["links"] == [good] and "Harbour pilot" in came.made["told"]
    assert "ten years at sea" in came.made["told"] and "free lunch" not in came.made["told"]
    assert [part.stands for part in came.parts] == ["met"]
    assert "2 did not" in came.came_to


def test_a_judgement_with_no_words_on_the_page_is_not_one(monkeypatch):
    url = "https://a.example/1"
    reads = {url: PageRead(page_is="one", name="Harbour pilot",
                           judged=[Judged(need=0, stands="yes", quote="Pay is £61,000 to £70,000 a year"),
                                   Judged(need=1, stands="yes", quote="fully remote, anywhere in the world")])}
    ask = _finding(monkeypatch, {url: PAGE % "2030-01-01"}, reads, [url])
    came = asyncio.run(find(_request(SAID), lambda line: None, ask=ask))
    assert "does not say: can be done from home" in came.made["told"]


def test_a_list_is_followed_to_what_it_lists(monkeypatch):
    top, one = "https://list.example/", "https://list.example/jobs/2"
    listing = f"<html><title>Posts</title><body><p>{'All our posts. ' * 60}</p><a href='/jobs/2'>Harbour pilot, tower</a></body></html>"
    reads = {top: PageRead(page_is="list", links=[0]),
             one: PageRead(page_is="one", name="Harbour pilot", judged=[
                 Judged(need=0, stands="yes", quote="Pay is £61,000 to £70,000 a year"),
                 Judged(need=1, stands="yes", quote="or from home")])}
    ask = _finding(monkeypatch, {top: listing, one: PAGE % "2030-01-01"}, reads, [top])
    assert asyncio.run(find(_request(SAID), lambda line: None, ask=ask)).made["links"] == [one]


def test_nothing_found_says_what_was_looked_at_and_why(monkeypatch):
    ask = _finding(monkeypatch, {"https://c.example/3": None}, {}, ["https://c.example/3"])
    came = asyncio.run(find(_request(SAID), lambda line: None, ask=ask))
    assert [part.stands for part in came.parts] == ["unmet"] and "its page is gone" in came.parts[0].shown_by


# ── reading whole ────────────────────────────────────────────────────

def _document(pages: int = 12) -> str:
    return "\n\n".join(f"[page {n}]\n" + "\n\n".join(f"On page {n} the {word} argument is made, at length, with care. " * 6
                                                     for word in ("first", "second", "third")) for n in range(1, pages + 1))


def test_every_character_of_a_document_is_in_a_section():
    text = _document()
    sections = sections_of(text, size=2500)
    assert sections[0].start == 0 and sections[-1].end == len(text)
    assert all(a.end == b.start for a, b in zip(sections, sections[1:], strict=False))
    assert unread(text, sections) > 0
    for section in sections:
        section.read = {"about": "x"}
    assert unread(text, sections) == 0


def _reading(monkeypatch, tmp_path, text: str):
    monkeypatch.setattr(reading_whole, "_where", lambda digest: tmp_path / f"{digest}.json")
    monkeypatch.setattr(finding, "_materials", lambda request: [("paper.pdf", text)])


def test_a_document_is_read_to_its_end_and_answered_from(monkeypatch, tmp_path):
    text = _document()
    _reading(monkeypatch, tmp_path, text)
    seen: list[str] = []

    async def ask(prompt, schema, n):
        if schema is SectionRead:
            seen.append(prompt)
            return SectionRead(about="an argument is made", points=[Point(says="made", quote="the first argument is made")],
                               questions=[Point(says="unclear", quote="words that are not there")])
        assert "an argument is made" in prompt and "How is it?" in prompt
        return Answer(answer="It holds: “the third argument is made, at length, with care”.", quotes=["the first argument is made"])

    came = asyncio.run(read_whole(_request("How is it?"), lambda line: None, ask=ask))
    assert [p.stands for p in came.parts] == ["met", "met"] and "all 12 pages" in came.parts[0].shown_by
    assert all(f"On page {n} " in "".join(seen) for n in range(1, 13)), "every page was before her model"
    assert came.made["told"].startswith("It holds")
    again = len(seen)
    asyncio.run(read_whole(_request("How is it?"), lambda line: None, ask=ask))
    assert len(seen) == again, "what was read once is not read again"


def test_a_part_that_could_not_be_read_is_halved_and_then_said(monkeypatch, tmp_path):
    text = _document()
    _reading(monkeypatch, tmp_path, text)

    async def ask(prompt, schema, n):
        if schema is SectionRead:
            return None if "On page 5 " in prompt else SectionRead(about="read")
        return Answer(answer="Of what I read, it holds.")

    came = asyncio.run(read_whole(_request("How is it?"), lambda line: None, ask=ask))
    assert came.parts[0].stands == "unmet" and "could not be read" in came.parts[0].shown_by
    assert int(came.parts[0].shown_by.split("%")[0]) > 60


def test_an_answer_quotes_the_document_or_does_not_quote(monkeypatch, tmp_path):
    text = _document(2)
    _reading(monkeypatch, tmp_path, text)

    async def ask(prompt, schema, n):
        if schema is SectionRead:
            return SectionRead(about="read")
        return Answer(answer="It says “the authors abandon the whole programme here”.")

    came = asyncio.run(read_whole(_request("How is it?"), lambda line: None, ask=ask))
    assert "“" not in came.made["told"]


def test_files_given_before_are_held_for_the_turns_after(tmp_path):
    ledger = RequestsOfHer(tmp_path / "r.json")
    ledger.gave("c", "Here's my paper.", ["0123456789abcdef"])
    asyncio.run(ledger.keep())
    again = RequestsOfHer(tmp_path / "r.json")
    assert again.given("c") == ["0123456789abcdef"]
    assert again.given("c", with_words=["Here's my  paper."]) == ["0123456789abcdef"]
    assert again.given("c", with_words=["Something else"]) == [] and again.given("d") == []
