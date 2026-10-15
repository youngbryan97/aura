"""Her chat gets what her corpus says about whom a conversation is about, said or asked, and not about grammar."""

from __future__ import annotations

from pathlib import Path

from core.knowledge.corpus_grounding import corpus_grounding_for
from core.knowledge.local_corpus import LocalCorpusStore
from core.knowledge.talked_about import mentions, said_about, subjects

ARTICLES = [
    ("Kobe", "Kobe is a port city in Japan, the capital of Hyōgo Prefecture, on Osaka Bay.\n\n"
             "The city is known for its port, its beef and its harbour views from Mount Rokkō."),
    ("Kobe Bryant", "Kobe Bryant was an American professional basketball player for the Los Angeles Lakers.\n\n"
                    "In 2006 he scored 81 points in a game against the Toronto Raptors, the second-highest "
                    "total in a single game.\n\n"
                    "He won five championships and two Finals MVP awards with the Lakers."),
    ("Ada Lovelace", "Ada Lovelace was an English mathematician known for her work on the Analytical Engine.\n\n"
                     "Her notes contain what is considered the first published algorithm for a machine.\n\n"
                     "She was the only legitimate child of the poet Lord Byron."),
    ("Mercury", "Mercury may refer to:\n\nMercury (planet), the planet nearest the Sun\n\nMercury (element)"),
    ("Mercury (planet)", "Mercury is the first planet from the Sun and the smallest in the Solar System.\n\n"
                         "A year on Mercury lasts 88 Earth days."),
    ("List of NBA single-game scoring leaders", "This is a list of the highest point totals in a single game.\n\n"
                                                "The record for points in a single game is 100, scored by Wilt "
                                                "Chamberlain against the New York Knicks in 1962."),
]


def _store(tmp_path: Path) -> LocalCorpusStore:
    store = LocalCorpusStore(tmp_path / "corpus.db")
    store.add_documents([(title, body, "wikipedia") for title, body in ARTICLES])
    store.add_retained_document("Ada Lovelace notes", "Ada Lovelace notes she saved from a web search on her engine.",
                                artifact_id="a1")
    return store


def test_capitals_grammar_put_there_paths_and_her_own_name_are_not_names() -> None:
    turn = "Go to /Users/someone/game.html and play it. I'll watch, Aura. Then tell me about Ada Lovelace."
    assert mentions(turn, speakers=("Aura",)) == ["Ada Lovelace"]
    # A name ends at a comma, so a list of names is not one long name.
    assert set(mentions("We watched the Lakers, Raptors, Celtics games")) == {"Lakers", "Raptors", "Celtics"}
    # A sentence's first word is a name only when the person capitalises it elsewhere too.
    assert mentions("So… Kobe was better.") == []
    assert mentions("So… Kobe was better.", evidence=["I think Kobe was the best"]) == ["Kobe"]


def test_a_name_is_linked_by_the_words_around_it_and_never_to_a_list_of_namesakes(tmp_path: Path) -> None:
    store = _store(tmp_path)
    player = subjects("Was Kobe's 81 point game the best ever?", [], store=store, deadline_s=5.0, most=3)
    assert [hit.title for hit in player] == ["Kobe Bryant"]
    city = subjects("We took the ferry to Kobe for the harbour and the beef", [], store=store, deadline_s=5.0, most=3)
    assert [hit.title for hit in city] == ["Kobe"]
    # A disambiguation page is about no one; the page the words point to is.
    planet = subjects("How long is a year on Mercury compared with Earth?", [], store=store, deadline_s=5.0, most=3)
    assert [hit.title for hit in planet] == ["Mercury (planet)"]
    # A note she saved from the web is not an encyclopedia article.
    lovelace = subjects("I was reading about Ada Lovelace and her engine notes", [], store=store, deadline_s=5.0,
                        most=3)
    assert [hit.title for hit in lovelace] == ["Ada Lovelace"]


def test_he_she_or_they_carries_the_subject_of_the_persons_own_last_turn(tmp_path: Path) -> None:
    store = _store(tmp_path)
    before = [("user", "I have been reading about Ada Lovelace."),
              ("aura", "Lovely. The Analytical Engine sits next to Mount Rokkō in my head somehow, Kobe aside.")]
    carried = subjects("She was Byron's daughter, right?", before, store=store, deadline_s=5.0, most=3)
    assert [hit.title for hit in carried] == ["Ada Lovelace"]
    # No pronoun pointing back, nothing carried.
    assert subjects("That was a lovely evening.", before, store=store, deadline_s=5.0, most=3) == []


def test_the_opening_comes_first_then_the_paragraph_nearest_in_meaning(tmp_path: Path) -> None:
    store = _store(tmp_path)
    before = [("user", "Kobe Bryant was something else.")]

    def nearness(query: str, texts: list[str]) -> list[float]:
        # A stand-in for her embedder: the paragraph about championships is nearest to talk of rings.
        return [1.0 if "championships" in text else 0.0 for text in texts]

    found = said_about("He had a lot of rings though", before, store=store, limit=2, deadline_s=5.0, rerank=nearness)
    assert found[0][0] == "Kobe Bryant" and found[0][1].startswith("Kobe Bryant was")
    assert "five championships" in found[1][1]


def test_chat_grounds_a_statement_about_a_person_only_when_it_gives_the_conversation(tmp_path: Path) -> None:
    store = _store(tmp_path)
    said = "We don't rate Kobe Bryant around here, for good reason"
    assert not corpus_grounding_for(said, store=store).grounded
    grounded = corpus_grounding_for(said, store=store, conversation=[], speakers=("Aura",))
    assert grounded.grounded and grounded.passages[0][0] == "Kobe Bryant"
    # An action asked of her is not about the world, conversation or not.
    assert not corpus_grounding_for("Open Kobe Bryant's page", store=store, conversation=[]).grounded


def test_paragraphs_from_articles_the_conversation_does_not_name_join_its_subjects(tmp_path: Path) -> None:
    from core.knowledge.passage_index import build

    store = _store(tmp_path)
    build(tmp_path / "corpus.db", tmp_path / "passages.db")
    said = "Kobe Bryant's 81 was huge, though the record for points in a single game is someone else's"
    grounding = corpus_grounding_for(said, store=store, conversation=[], limit=3)
    titles = [title for title, _text in grounding.passages]
    assert titles[0] == "Kobe Bryant"
    # The record is in an article the turn never names.
    assert any("Wilt Chamberlain" in text for _title, text in grounding.passages), grounding.passages
