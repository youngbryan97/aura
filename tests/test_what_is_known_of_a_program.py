"""What is known of a program is read in depth: the sentences that say what it does, the articles of what those rest
on, found by title and kept only when they say they are an algorithm, a format or the like and speak of the same
work, and what her model proposes taken only where her corpus bears it out."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from core.rebuilding.what_is_known_of_it import (
    behaviour_stated,
    deepened,
    known_of,
    what_she_would_look_up,
)

pytestmark = pytest.mark.unit


@dataclass
class _Hit:
    title: str
    doc_id: int
    snippet: str = ""
    source: str = "wikipedia"


class _Corpus:
    """Articles by title, searched as her corpus searches them: every word of the name in the title."""

    def __init__(self, articles: dict[str, str]):
        self.articles = list(articles.items())

    def search_titles(self, name, limit=5, **_):
        words = name.lower().replace("'", " ").split()
        found = [_Hit(t, i) for i, (t, _b) in enumerate(self.articles)
                 if all(w in t.lower().replace("'", " ") for w in words)]
        return sorted(found, key=lambda h: len(h.title))[:limit]

    def search(self, query, limit=5, **_):
        words = set(query.lower().split())
        return [_Hit(t, i) for i, (t, b) in enumerate(self.articles) if words <= set(b.lower().split())][:limit]

    def by_title(self, title, **_):
        return next((_Hit(t, i) for i, (t, _b) in enumerate(self.articles) if t == title), None)

    def body(self, doc_id, max_chars=20000):
        return self.articles[doc_id][1][:max_chars]


_ARTICLES = {
    "Bucket Painter": "Bucket Painter is a raster graphics editor. It allows users to fill an area with the BMP file "
                      "format colours. Images can be saved as BMP files. It was first released in 1990.",
    "BMP file format": "The BMP file format is a raster graphics image file format used to store bitmap images.",
    "BMP (company)": "BMP is a company that makes bicycles in a small town.",
    "Flood fill": "Flood fill is a flooding algorithm that determines the area connected to a given node in an array. "
                  "It is used in the bucket fill tool of paint programs to fill connected areas with a colour.",
    "Zoom (software)": "Zoom is a videotelephony software program developed by a company for video meetings.",
    "Undo": "Undo is an interaction technique which is implemented in many computer programs.",
    "Painting": "Painting may refer to: the practice of applying paint; a painting, the result.",
}


def test_a_sentence_says_what_a_program_does_by_its_shape():
    assert behaviour_stated("It allows users to fill an area with a colour.")
    assert behaviour_stated("Images can be saved as BMP files.")
    assert behaviour_stated("It can undo a change up to 10 times.")
    assert not behaviour_stated("It was first released in 1990.")


def test_what_the_program_does_leads_to_the_articles_it_rests_on_and_no_others():
    corpus = _Corpus(_ARTICLES)
    source = type("Source", (), {"title": "Bucket Painter", "text": _ARTICLES["Bucket Painter"]})()
    known = known_of("Bucket Painter", [source], corpus=corpus, passages=_NoPassages())
    assert [r.text for r in known.does] == ["It allows users to fill an area with the BMP file format colours.",
                                            "Images can be saved as BMP files."]
    assert list(known.rests_on) == ["BMP file format"]          # not the bicycle company, not a list of meanings
    assert known.of("save the picture as a BMP file")[0].about in ("", "BMP file format")


class _NoPassages:
    def search(self, *_a, **_k):
        return []


def test_her_models_names_are_taken_only_where_her_corpus_bears_them_out():
    corpus = _Corpus(_ARTICLES)
    source = type("Source", (), {"title": "Bucket Painter", "text": _ARTICLES["Bucket Painter"]})()
    known = known_of("Bucket Painter", [source], corpus=corpus, passages=_NoPassages())

    async def ask(prompt, schema, _tokens):
        assert "Bucket Painter" in prompt and "allows users to fill" in prompt
        return schema(names=["Flood fill", "Zoom", "Undo", "Spray tip algorithm", "Flood fill"])

    names = asyncio.run(what_she_would_look_up("Bucket Painter", known, ask))
    assert names == ["Flood fill", "Zoom", "Undo", "Spray tip algorithm"]
    deepened(known, names, corpus=corpus, passages=_NoPassages())
    assert "Flood fill" in known.rests_on and "Undo" in known.rests_on
    # A video call is not what a paint program zooms by; an algorithm no article names is her model's word alone.
    assert "Zoom (software)" not in known.rests_on
    assert known.unborne == ["Zoom", "Spray tip algorithm"]
    assert any("bucket fill tool" in r.text for r in known.rests_on["Flood fill"])
    assert known.rests_on["Flood fill"][0].where == "the article Flood fill, named by her model"
