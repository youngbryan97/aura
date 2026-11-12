"""L10: a claim the encyclopedia supports is read again where the encyclopedia says it got it.

LIVE 2026-11-10 her search engines gave a program nothing to read (a challenge from DuckDuckGo, other searches'
results from Bing), so no second publisher was found and the claim stayed single-source. The article's own citations
give addresses, and the pages there are their publishers' own.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from core.epistemics.cited_sources import WhatTheArticleCites, cited_for

pytestmark = pytest.mark.unit

CLAIM = "the speed of light in a vacuum is exactly 299,792,458 metres per second"

WIKITEXT = """{{Infobox
| value = 299792458 <ref>{{cite web |url=https://example.com/infobox}}</ref>
}}

The '''speed of light in vacuum''' is a universal constant, exactly equal to 299792458 metres per second.<ref name="codata">{{cite web |url=http://physics.nist.gov/cgi-bin/cuu/Value?c |title=CODATA}}</ref> It is exact by definition.<ref>[https://www.bipm.org/en/committees/cg/cgpm/17-1983/resolution-1 Resolution 1]</ref>

Light was first timed by Rømer.<ref>{{cite book |url=https://books.google.com/books?id=x}}</ref>

Since 1983 the metre is the length light travels in 1/299792458 of a second.<ref name="codata"/><ref>{{cite web |url=https://www.nist.gov/report.pdf}}</ref><ref>{{cite web |url=https://web.archive.org/web/2015/http://www.bipm.org/x}}</ref>
"""


def test_the_paragraphs_with_the_claims_figure_give_their_citations_and_a_reused_one_is_found_by_name():
    found = cited_for(WIKITEXT, CLAIM)
    assert found[:2] == ["http://physics.nist.gov/cgi-bin/cuu/Value?c",
                         "https://www.bipm.org/en/committees/cg/cgpm/17-1983/resolution-1"]
    assert "https://example.com/infobox" not in found  # a template's table is not the article's prose
    assert "https://books.google.com/books?id=x" not in found  # a paragraph without the claim's figure


class _Gateway:
    def __init__(self, pages):
        self.pages = pages
        self.asked: list[str] = []
        self.policy = SimpleNamespace(reads=lambda host: True)

    async def get(self, url, headers=None):
        self.asked.append(url)
        if "list=search" in url:
            body = json.dumps({"query": {"search": [{"title": "Speed of light"}]}})
        elif "prop=revisions" in url:
            body = json.dumps({"query": {"pages": [{"revisions": [{"slots": {"main": {"content": WIKITEXT}}}]}]}})
        else:
            body = self.pages.get(url, "")
        return SimpleNamespace(ok=bool(body), body_preview=body)


def _read(gateway):
    async def go():
        return [pair async for pair in WhatTheArticleCites().readings(gateway, ["speed", "light"], claim=CLAIM)]

    return asyncio.run(go())


def test_counted_publishers_are_read_once_each_and_the_rest_passed_over():
    gateway = _Gateway({
        "http://physics.nist.gov/cgi-bin/cuu/Value?c": "<p>speed of light in vacuum Numerical value 299 792 458 m s-1 (exact)</p>",
        "https://www.bipm.org/en/committees/cg/cgpm/17-1983/resolution-1": "<p>The metre is the length of the path "
        "travelled by light in vacuum during a time interval of 1/299 792 458 of a second.</p>",
    })
    read = _read(gateway)
    assert [url for _evidence, url in read] == ["http://physics.nist.gov/cgi-bin/cuu/Value?c",
                                                 "https://www.bipm.org/en/committees/cg/cgpm/17-1983/resolution-1"]
    assert "299 792 458" in read[0][0]
    fetched = [u for u in gateway.asked if "wikipedia" not in u]
    assert "https://www.nist.gov/report.pdf" not in fetched  # a PDF is not turned into text
    assert not any("archive.org" in u or "books.google" in u for u in fetched)  # not counted on her list


def test_a_publisher_read_once_is_not_read_again():
    gateway = _Gateway({"http://physics.nist.gov/cgi-bin/cuu/Value?c": "<p>299 792 458 m s-1</p>"})
    _read(gateway)
    assert sum(1 for u in gateway.asked if "nist.gov" in u) == 1


@pytest.mark.parametrize("text, figures", [
    ("Numerical value 299 792 458 m s -1", {"299792458", "1"}),
    ("1/299 792 458 of a second", {"1", "299792458"}),
    ("299,792,458 metres per second", {"299792458"}),
    ("between 1983 2019 and 25,300", {"1983", "2019", "25300"}),
])
def test_a_figure_written_the_si_way_is_one_figure(text, figures):
    """LIVE 2026-11-12: NIST's page reads "299 792 458", her reader took the passage as entailing the claim (0.92),
    and the figure check failed because it saw three numbers."""
    from core.epistemics.epistemic_reach import _numbers

    assert _numbers(text) == figures


def test_nists_passage_bears_out_the_claims_figure():
    from core.epistemics.epistemic_reach import _figures_borne_out

    assert _figures_borne_out(CLAIM, "speed of light in vacuum Numerical value 299 792 458 m s -1 (exact)")


_NIST_PAGE = ("CODATA Value: speed of light in vacuum Constants Topics: Values Energy Equivalents Searchable Bibliography "
              "Background Constants Bibliography Constants, Units & Uncertainty home page speed of light in vacuum Numerical "
              "value 299 792 458 m s -1 Standard uncertainty (exact) Concise form 299 792 458 m s -1 Click here")


def _reader_misled_by_navigation(passage, claim):
    """As her reader read NIST's page live: the page whole, navigation and all, as a contradiction."""
    if "Energy Equivalents" in passage and "Searchable" in passage:
        return {"entailment": 0.04, "neutral": 0.02, "contradiction": 0.94}
    return {"entailment": 0.98, "neutral": 0.01, "contradiction": 0.01}


def test_a_data_page_is_read_where_it_states_the_claims_figure(monkeypatch):
    """LIVE 2026-11-12: NIST's page, read whole, came back a contradiction (0.94) of the claim it states; the words
    around its figure came back entailment (0.98)."""
    import core.epistemics.epistemic_reach as reach

    monkeypatch.setattr(reach, "_how_it_reads", _reader_misled_by_navigation)
    verdict = reach.EpistemicReachEngine.judge(CLAIM, _NIST_PAGE, "http://physics.nist.gov/cgi-bin/cuu/Value?c")
    assert verdict.verdict == reach.VERDICT_SUPPORTED and "around its figure" in verdict.reason


def test_a_figure_the_page_does_not_hold_gives_nothing_to_read(monkeypatch):
    import core.epistemics.epistemic_reach as reach

    monkeypatch.setattr(reach, "_how_it_reads", _reader_misled_by_navigation)
    wrong = CLAIM.replace("299,792,458", "300,000,000")
    verdict = reach.EpistemicReachEngine.judge(wrong, _NIST_PAGE, "http://physics.nist.gov/cgi-bin/cuu/Value?c")
    assert verdict.verdict != reach.VERDICT_SUPPORTED
