"""A knowledge question's names lead to their articles, and the paragraphs carrying the question's rare words come back."""

from __future__ import annotations

from pathlib import Path

from core.knowledge.local_corpus import LocalCorpusStore
from core.knowledge.passages_about import named_spans, passages_about


def _store(tmp_path: Path) -> LocalCorpusStore:
    store = LocalCorpusStore(tmp_path / "corpus.db")
    store.add_documents([
        ("William B. Ittner",
         "William Butts Ittner was an American architect in St. Louis.\n"
         "Ittner designed over 430 school buildings in Missouri and other states.\n"
         "Ittner served as president of the Architectural League of America during 1903 to 1904.\n"
         "Ittner died in St. Louis in 1936 after a long career.", "wikipedia"),
        ("Skylines", "Skylines is a 2018 film about a city and the skyline of its towers.", "wikipedia"),
        ("Cities: Skylines", "Cities: Skylines is a 2015 city-building game by Colossal Order.\n"
         "The game was released for Google Stadia on May 17, 2022 after a long wait.", "wikipedia"),
        ("Salvador Dalí", "Salvador Dalí was a Spanish surrealist painter from Figueres in Catalonia.\n"
         "His uncle Anselm Domènech owned a bookshop in Barcelona that supplied the young painter.",
         "wikipedia"),
    ])
    return store


def test_names_are_read_with_their_joining_words_and_without_possessives() -> None:
    assert named_spans("Who served the Architectural League of America with William Butts Ittner?") == [
        "Architectural League of America", "William Butts Ittner"]
    assert named_spans("What is the nasab of Ahmad al-Kabīr al-Rifāʽī?") == ["Ahmad al-Kabīr al-Rifāʽī"]
    assert named_spans("What was Salvador Dalí's uncle called?") == ["Salvador Dalí"]
    assert named_spans("what is it?") == []


def test_the_answer_paragraph_comes_back_from_the_article_a_name_leads_to(tmp_path: Path) -> None:
    store = _store(tmp_path)
    got = passages_about("During which years did William Butts Ittner serve as president of the Architectural "
                         "League of America?", store=store, limit=1)
    assert got and got[0][0] == "William B. Ittner" and "1903 to 1904" in got[0][1]
    got = passages_about("On what date was Cities: Skylines released for Google Stadia?", store=store, limit=1)
    assert got and got[0][0] == "Cities: Skylines" and "May 17, 2022" in got[0][1]
    got = passages_about("What was the name of Salvador Dalí's uncle who owned a bookshop?", store=store, limit=1)
    assert got and "Anselm Domènech" in got[0][1]
    assert passages_about("what about it?", store=store) == []


def test_accented_names_reach_the_index_and_a_title_search_prefers_the_closest_title(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert any(hit.title == "Salvador Dalí" for hit in store.search("Dalí uncle", limit=3))
    titles = [hit.title for hit in store.search_titles("Cities Skylines")]
    assert titles == ["Cities: Skylines"]


def test_with_her_embedder_the_pool_is_ordered_by_meaning(tmp_path: Path) -> None:
    store = _store(tmp_path)
    question = "During which years did William Butts Ittner serve as president of the Architectural League of America?"
    lexical = passages_about(question, store=store, limit=4)
    assert "1903 to 1904" in lexical[0][1]

    def prefers_the_opening(asked, paragraphs):
        return [1.0 if "was an American architect" in p else 0.0 for p in paragraphs]

    reranked = passages_about(question, store=store, limit=1, rerank=prefers_the_opening)
    assert "was an American architect" in reranked[0][1]
    # Reranking orders the words' candidates; a paragraph sharing only the subject's name is not one.
    assert all("died" not in p for _t, p in passages_about(question, store=store, limit=4,
                                                           rerank=lambda a, ps: [float("died" in x) for x in ps]))


def test_a_question_about_named_things_is_asked_of_the_corpus_and_a_remark_to_her_is_not() -> None:
    from core.knowledge.corpus_grounding import asks_about_named_things

    assert asks_about_named_things("In what year did conductor David Randolph marry Mildred Greenberg?")
    assert asks_about_named_things("Which school did Lucas Radebe attend")
    for turn in ("Can you play Pong with me?", "Bryan here, did the new build work for you?",
                 "Is it going to rain in Seattle tomorrow?", "How was your conversation with Claude earlier?",
                 "open my notes", "hey Aura, how are you?", "what is it?"):
        assert not asks_about_named_things(turn), turn


def test_her_chat_falls_back_to_the_careful_lookup_when_its_own_search_finds_nothing(tmp_path: Path) -> None:
    from core.knowledge.corpus_grounding import corpus_grounding_for

    store = _store(tmp_path)
    grounding = corpus_grounding_for("In what years did William Butts Ittner serve as president of the "
                                     "Architectural League of America?", store=store)
    assert grounding.grounded and any("1903 to 1904" in text for _title, text in grounding.passages)
    assert not corpus_grounding_for("Can you play Pong with me?", store=store).grounded
