"""Every paragraph of her corpus is found by itself, read back from the corpus, and the build resumes."""

from __future__ import annotations

from pathlib import Path

from core.knowledge.local_corpus import LocalCorpusStore
from core.knowledge.passage_index import PassageIndex, build, fused

ARTICLES = [
    ("HMS Endeavour",
     "HMS Endeavour was a Royal Navy research vessel of the eighteenth century.\n\n"
     "Her commander on the first voyage of discovery to the Pacific was James Cook, who charted New Zealand.",
     "wikipedia"),
    ("James Cook",
     "James Cook was a British explorer, navigator, cartographer and naval officer.\n\n"
     "He was born in Marton in Yorkshire in 1728 and joined the merchant navy as a teenager.", "wikipedia"),
    ("Marton", "Marton is a village in the borough of Middlesbrough in North Yorkshire, England.", "wikipedia"),
]


#: A test index needs one thread and a small writer heap.
SMALL = {"threads": 1, "heap_bytes": 50_000_000}


def _corpus(tmp_path: Path) -> Path:
    store = LocalCorpusStore(tmp_path / "corpus.db")
    store.add_documents(ARTICLES)
    return tmp_path / "corpus.db"


def test_a_paragraph_is_found_in_an_article_the_question_does_not_name(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    summary = build(corpus, tmp_path / "paragraphs", **SMALL)
    assert summary["articles"] == 3 and summary["paragraphs"] == 5
    index = PassageIndex(tmp_path / "paragraphs", corpus)
    # Asked about a captain's ship's voyage: the answer is in the ship's article, which the question does not name.
    found = index.search("Which vessel did the explorer command on his first voyage to the Pacific?", limit=2,
                         deadline_s=5.0)
    assert found[0].title == "HMS Endeavour" and "first voyage" in found[0].paragraph
    # Read back from the corpus by article and position, exactly as the careful lookup splits it.
    assert found[0].position == 1


def test_a_stopped_build_resumes_after_the_last_article_it_recorded(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    first = build(corpus, tmp_path / "paragraphs", until=1, **SMALL)
    rest = build(corpus, tmp_path / "paragraphs", **SMALL)
    assert (first["articles"], rest["articles"]) == (1, 2)
    assert first["paragraphs"] + rest["paragraphs"] == 5
    found = PassageIndex(tmp_path / "paragraphs", corpus).search("Marton village Middlesbrough", limit=5,
                                                                  deadline_s=5.0)
    assert [p.title for p in found][:1] == ["Marton"]


def test_a_question_with_no_content_words_and_a_missing_index_find_nothing(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    assert PassageIndex(tmp_path / "absent", corpus).search("Who was James Cook?", limit=3, deadline_s=5.0) == []
    build(corpus, tmp_path / "paragraphs", **SMALL)
    assert PassageIndex(tmp_path / "paragraphs", corpus).search("what is the", limit=3, deadline_s=5.0) == []


def test_fusion_ranks_what_both_lists_agree_on_first() -> None:
    a = [("A", "1"), ("B", "2"), ("C", "3")]
    b = [("B", "2"), ("D", "4"), ("A", "1")]
    assert fused([a, b], limit=4)[:2] == [("B", "2"), ("A", "1")]


def test_a_name_finds_its_article_by_the_words_said_with_it_in_one_search(tmp_path: Path) -> None:
    store = LocalCorpusStore(tmp_path / "corpus.db")
    store.add_documents([
        ("Kobe", "Kobe is a port city in Japan on Osaka Bay, known for its harbour and its beef.", "wikipedia"),
        ("Kobe Bryant", "Kobe Bryant was a basketball player who scored 81 points in one game.", "wikipedia"),
    ])
    build(tmp_path / "corpus.db", tmp_path / "paragraphs", **SMALL)
    index = PassageIndex(tmp_path / "paragraphs", tmp_path / "corpus.db")
    player = store.documents(index.articles_titled("Kobe", ["81", "points", "game"], limit=5))
    city = store.documents(index.articles_titled("Kobe", ["harbour", "beef", "ferry"], limit=5))
    assert [player[0].title, city[0].title] == ["Kobe Bryant", "Kobe"]
    assert index.articles_titled("Nagoya", ["harbour"], limit=5) == []
