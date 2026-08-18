"""The offline corpus must answer about a one-word subject, and pick the page.

Two defects, found 2026-08-18 while measuring corpus latency against the live
6.5M-page index.

The first made a whole class of question unanswerable. `_is_relevant` required
`max(2, ...)` matched terms, but "what is a semiconductor" reduces to the
single topical term "semiconductor" — a bar of two that one term can never
clear. Three good hits came back, all three were discarded, and the corpus
returned nothing for every one-word subject. It reported no error; it simply
had no passages, so the model answered unaided and the corpus might as well
not have been installed.

The second was ordering. The candidate pool equalled the number of passages
carried, so the three kept were whatever BM25 ranked first: "when did the
berlin wall fall" grounded on "The Berlin Wall (video game)", and "who was ada
lovelace" on "Ada Lovelace Award".

The rows below are the real hits the live index returned, in the order it
returned them, so the ordering these tests assert is the ordering that was
actually wrong. The negative controls are the two wrong-sense hits
`_is_relevant` was written for; widening the gate must not reopen them.
"""

from __future__ import annotations

import pytest

from core.knowledge.corpus_grounding import _is_relevant, corpus_grounding_for


class _Hit:
    def __init__(self, title: str, text: str) -> None:
        self.title = title
        self.text = text


class _Corpus:
    """A store that returns fixed rows, in BM25's unhelpful order."""

    def __init__(self, rows: list[tuple[str, str]]) -> None:
        self.rows = rows
        self.asked_limit = 0

    def search(self, query: str, *, limit: int, deadline_s: float) -> list[_Hit]:
        self.asked_limit = limit
        return [_Hit(t, x) for t, x in self.rows[:limit]]


_BERLIN = _Corpus([
    ("The Berlin Wall (video game)", "A 1991 platform game set at the Berlin Wall."),
    ("Thierry Noir", "A French artist who painted the Berlin Wall."),
    ("Beloved Berlin Wall", "A 2009 German film about the Berlin Wall."),
    ("Fall of the Berlin Wall", "The Berlin Wall fell on 9 November 1989."),
    ("List of Berlin Wall segments", "Segments of the Berlin Wall worldwide."),
])
_LOVELACE = _Corpus([
    ("Ada Lovelace Award", "An award named for Ada Lovelace."),
    ("Ada Lovelace Day", "An annual event celebrating Ada Lovelace."),
    ("Ada Lovelace", "Ada Lovelace was an English mathematician and writer."),
])
_SEMICONDUCTOR = _Corpus([
    ("Extrinsic semiconductor", "A semiconductor doped with impurities."),
    ("List of semiconductor companies", "Companies making semiconductor devices."),
    ("Semiconductor", "A semiconductor is a material between conductor and insulator."),
])
# The two live wrong-sense hits, with nothing better behind them.
_RUST = _Corpus([
    ("Pee-wee Herman", "A comic character who originally wrote and performed."),
])
_DJANGO = _Corpus([
    ("Django Haskins", "An American songwriter who released records."),
])


def _titles(query: str, store: _Corpus) -> list[str]:
    return [title for title, _ in corpus_grounding_for(query, store=store).passages]


@pytest.mark.parametrize(
    ("query", "store"),
    [
        ("what is a semiconductor", _SEMICONDUCTOR),
        ("who was ada lovelace", _LOVELACE),
        ("when did the berlin wall fall", _BERLIN),
    ],
)
def test_a_real_subject_gets_passages(query: str, store: _Corpus) -> None:
    assert _titles(query, store), f"corpus answered nothing for {query!r}"


def test_a_single_term_query_can_clear_the_relevance_bar() -> None:
    """The bar may not exceed what the query has to offer."""
    assert _is_relevant(
        "semiconductor", "Extrinsic semiconductor", "A semiconductor is a material."
    )


@pytest.mark.parametrize(
    ("query", "store", "expected"),
    [
        ("who was ada lovelace", _LOVELACE, "Ada Lovelace"),
        ("what is a semiconductor", _SEMICONDUCTOR, "Semiconductor"),
        ("when did the berlin wall fall", _BERLIN, "Fall of the Berlin Wall"),
    ],
)
def test_the_subject_page_outranks_its_neighbours(
    query: str, store: _Corpus, expected: str
) -> None:
    """"Ada Lovelace Award" and "The Berlin Wall (video game)" are not it."""
    titles = _titles(query, store)

    assert titles and titles[0] == expected, f"{query!r} -> {titles}"


def test_the_candidate_pool_is_wider_than_the_answer() -> None:
    """Ranking cannot improve on an ordering it is handed whole."""
    corpus = _Corpus(list(_BERLIN.rows))
    corpus_grounding_for("when did the berlin wall fall", limit=3, store=corpus)

    assert corpus.asked_limit > 3


@pytest.mark.parametrize(
    ("query", "store"),
    [
        ("wrote Rust borrow checker originally", _RUST),
        ("Django released 2005", _DJANGO),
    ],
)
def test_a_wrong_sense_hit_still_grounds_nothing(query: str, store: _Corpus) -> None:
    """The two live misses that put the floor there in the first place."""
    assert not _titles(query, store)
