"""What she finds out on her own is a list of questions, each answered by the words of a page or let go (core/agency/her_inquiries.py)."""
from __future__ import annotations

import asyncio

from core.agency.her_inquiries import (
    LET_GO_AFTER,
    Inquiry,
    Page,
    Question,
    Says,
    ToAsk,
    ToFindOut,
    look_into_one,
)
from core.agency.her_own_time import HerOwnTime
from core.agency.what_she_does_next import WhatSheDoesNext

HOMING = Page("https://a.example/limpets", "Limpets", "Each limpet returns to its own scar on the rock before low tide.")
PRICES = Page("https://b.example/prices", "Seafood prices", "Limpets sell for nine euros a kilo.")


def _of(*given):
    """Pages found for any words; a page is opened when it is come to."""
    opened = []

    async def pages(words):
        async def read(page):
            opened.append((words, page.url if page else ""))
            return page

        return [lambda page=page: read(page) for page in given]

    return pages, opened


def _model(*questions, says):
    asked = []

    async def ask(prompt, shape, _room):
        asked.append(prompt)
        if shape is ToFindOut:
            return ToFindOut(questions=list(questions))
        return next(answer for url, answer in says.items() if url in prompt)

    return ask, asked


def _pass(inquiry, ask, pages):
    seen = []
    return asyncio.run(look_into_one(inquiry, ask=ask, pages=pages, seen=seen.extend)), seen


ASKED = (ToAsk(question="Where does a limpet go at low tide?", search="limpet homing low tide"), ToAsk(question="How long do limpets live?", search="limpet lifespan"))


def test_a_thing_wanted_becomes_questions_and_the_first_is_answered_in_a_pages_own_words():
    """LIVE 25 November 2026 what she wanted to find out more of was searched by its name, and came to nothing."""
    inquiry = Inquiry("limpets")
    ask, _asked = _model(*ASKED, says={PRICES.url: Says(answers="no"), HOMING.url: Says(answers="yes", answer="Back to its own scar.", quote="returns to its own scar on the rock")})
    pages, opened = _of(PRICES, HOMING, Page("https://c.example", "More", "x"))
    one, seen = _pass(inquiry, ask, pages)
    assert one.got == 1.0 and one.question.stands == "answered" and one.question.where == HOMING.url
    assert [words for words, _url in opened] == ["limpet homing low tide"] * 2, "what was searched is the question's words, and what answered it ended the pass"
    assert seen == [PRICES.url, HOMING.url] and inquiry.met == 0.5 and not inquiry.is_done
    assert inquiry.next().question == "How long do limpets live?"


def test_an_answer_whose_words_are_not_on_the_page_is_no_answer():
    inquiry = Inquiry("limpets", [Question("Where does a limpet go?", "limpet homing")])
    ask, _ = _model(says={HOMING.url: Says(answers="yes", answer="To the sea.", quote="limpets swim out to sea at night")})
    one, _seen = _pass(inquiry, ask, _of(HOMING)[0])
    assert one.got == 0.0 and one.question.answer == "" and one.question.stands == "open"


def test_a_question_nothing_answers_is_let_go_and_what_is_asked_next_knows_it():
    inquiry = Inquiry("limpets", [Question("What do limpets dream of?", "limpet dreams")])
    ask, asked = _model(*ASKED, says={PRICES.url: Says(answers="no")})
    for _ in range(LET_GO_AFTER):
        _pass(inquiry, ask, _of(PRICES)[0])
    assert inquiry.questions[0].stands == "let go" and not inquiry.is_done
    _pass(inquiry, ask, _of(PRICES)[0])
    wrote = next(prompt for prompt in asked if "found out next" in prompt)
    assert "What do limpets dream of?" in wrote and [q.question for q in inquiry.questions[1:]] == [q.question for q in ASKED]


def test_what_is_asked_next_is_written_from_what_was_found_out_and_nothing_is_asked_twice():
    inquiry = Inquiry("limpets", [Question(ASKED[0].question.upper(), "x", "answered", "Back to its own scar.", "own scar", HOMING.url)])
    assert inquiry.is_done and inquiry.met == 1.0
    ask, asked = _model(*ASKED, says={PRICES.url: Says(answers="no")})
    _pass(inquiry, ask, _of(PRICES)[0])
    assert "Back to its own scar." in asked[0]
    assert [q.question for q in inquiry.questions[1:]] == [ASKED[1].question] and not inquiry.is_done


def test_part_of_an_answer_is_gone_on_with_once_and_stands():
    inquiry = Inquiry("limpets", [Question("How long do limpets live?", "limpet lifespan")])
    ask, _ = _model(says={PRICES.url: Says(answers="partly", answer="Some are old.", quote="nine euros")})
    first, _ = _pass(inquiry, ask, _of(PRICES)[0])
    assert first.got == 0.5 and inquiry.next() is inquiry.questions[0]
    _pass(inquiry, ask, _of(PRICES)[0])
    assert inquiry.next() is None and inquiry.is_done


def test_a_page_that_cannot_be_read_is_passed_over_and_one_with_nothing_to_ask_comes_to_nothing():
    inquiry = Inquiry("limpets", [Question("How long?", "limpet lifespan")])
    ask, _ = _model(says={HOMING.url: Says(answers="yes", answer="Years.", quote="limpet returns")})
    one, seen = _pass(inquiry, ask, _of(None, HOMING)[0])
    assert one.got == 1.0 and seen == [HOMING.url]
    empty, _asked = _model(says={})
    assert _pass(Inquiry("nothing"), empty, _of()[0])[0] is None


def test_what_she_is_finding_out_is_kept_when_she_is_closed(tmp_path, monkeypatch):
    from pathlib import Path

    from core.rebuilding import her_build_writes

    monkeypatch.setattr(her_build_writes, "governed", lambda what, path, *args, **_how: Path(path).write_text(args[0], encoding="utf-8") if what == "write_text" else None)
    hers = HerOwnTime(WhatSheDoesNext(tmp_path / "chooser.json"), tmp_path / "hers.json")
    monkeypatch.setattr(hers.chooser, "keep", lambda: None)
    inquiry = hers.inquiry("want:limpets", "limpets")
    inquiry.questions.append(Question("Where do they go?", "x", "answered", "Home.", "home", HOMING.url, 1))
    asyncio.run(hers.keep())
    again = HerOwnTime(WhatSheDoesNext(tmp_path / "chooser.json"), tmp_path / "hers.json")
    again._load()
    assert again.inquiry("want:limpets", "limpets").questions == inquiry.questions
    assert again.inquiry("want:limpets", "limpets in winter").questions == [], "another thing to find out is begun from nothing"


def test_the_inquiries_kept_are_the_ones_last_worked_on(tmp_path, monkeypatch):
    from core.agency import her_own_time

    monkeypatch.setattr(her_own_time, "KEPT", 3)
    hers = HerOwnTime(WhatSheDoesNext(tmp_path / "c.json"), tmp_path / "h.json")
    for n in range(5):
        hers.inquiry(f"want:{n}", str(n)).changed_at = float(n + 1)
    assert set(hers.inquiries) == {"want:2", "want:3", "want:4"}


def test_what_her_search_answered_from_her_library_is_read_from_her_library(monkeypatch):
    """LIVE 27 November 2026 "What are the standard times for quiet hours in residential areas?" was answered by five
    documents of her library, which have no address on the web, and her inquiry read none of them."""
    import asyncio
    from types import SimpleNamespace

    import core.knowledge.local_corpus as corpus
    from interface.routes import her_own_time as route

    async def searched(words):
        return {"ok": True, "provenance": "local_corpus", "results": [
            {"title": "Noise regulation", "snippet": "quiet hours", "source": "wikipedia", "provenance": "local_corpus", "doc_id": 7},
            {"title": "Elsewhere", "snippet": "", "source": "web", "url": "https://example.org/quiet"}]}

    bodies = {7: "Many towns set quiet hours from 10 pm to 7 am."}
    monkeypatch.setattr(route, "_search", searched)
    monkeypatch.setattr(corpus, "get_local_corpus_store", lambda: SimpleNamespace(body=lambda doc_id, max_chars: bodies.get(doc_id, "")[:max_chars]))

    async def read_them():
        pages = await route._pages("standard quiet hours")
        return len(pages), await pages[-1]()

    count, page = asyncio.run(read_them())
    assert count == 2, "the page on the web and the document of her library"
    assert page.url == "her library, document 7" and page.title == "Noise regulation" and "10 pm to 7 am" in page.text
