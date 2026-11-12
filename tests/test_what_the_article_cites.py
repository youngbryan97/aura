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
