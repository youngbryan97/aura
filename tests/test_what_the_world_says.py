"""She reads what a turn names before answering it.

LIVE 2026-10-03: asked which was better, Bam's 83-point game or Kobe's 81, she
said the 83 never happened while her offline Wikipedia held the article; asked
to Google it, she searched for the film "It"; sent a Wikipedia link, she said
she could not open links.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from core.cognition import evidence_relevance
from core.conversation.what_the_world_says import (
    gather_world_evidence,
    passage_of,
    referent_phrases,
    wikipedia_title,
)
from core.language.search_request import read_search_request
from core.runtime.desktop_objective_intent import looks_like_desktop_objective

BAM = "On March 10, 2026, Bam Adebayo scored 83 points for the Miami Heat.\n\nBackground.\nHe was in his ninth season."
KOBE = "On January 22, 2006, Kobe Bryant scored 81 points for the Los Angeles Lakers."


@dataclass
class _Hit:
    title: str
    doc_id: int
    source: str = "wikipedia"
    snippet: str = ""


class _Corpus:
    """Two articles, found by any query naming them."""

    docs = {1: ("Bam Adebayo's 83-point game", BAM), 2: ("Kobe Bryant's 81-point game", KOBE)}

    def __init__(self):
        self.queries: list[str] = []

    def search(self, query, limit=5, **_):
        self.queries.append(query)
        return [_Hit(title, doc) for doc, (title, _text) in self.docs.items() if title.split()[0].rstrip("'s") in query][:limit]

    def body(self, doc_id, **_):
        return self.docs[doc_id][1]

    def by_title(self, title):
        return next((_Hit(t, d) for d, (t, _x) in self.docs.items() if t == title), None)


@pytest.fixture
def judge(monkeypatch):
    """Bears on the question when it shares a name with it; scores say how near."""

    def assess(question, passages):
        verdicts = []
        for passage in passages:
            shared = any(name in passage and name in question for name in ("Bam", "Kobe", "Heat"))
            score = 0.75 if shared else 0.55
            verdicts.append(evidence_relevance.EvidenceAlignment(score >= 0.5, True, score, 0.5, (), "stand-in"))
        return verdicts

    monkeypatch.setattr(evidence_relevance, "assess_evidence_alignments", assess)


def test_each_thing_named_is_its_own_query_and_keeps_its_numbers() -> None:
    assert referent_phrases("Which was better, Bam’s 83 point game or Kobe’s 81 point game?") == [
        "Bam 83 point game",
        "Kobe 81 point game",
    ]
    assert referent_phrases("How are you feeling tonight?") == []
    assert wikipedia_title("https://en.wikipedia.org/wiki/Bam_Adebayo%27s_83-point_game") == "Bam Adebayo's 83-point game"
    assert passage_of(BAM, ["ninth"]).startswith("On March 10, 2026")


def test_a_question_about_named_things_reads_them_from_the_offline_corpus(judge) -> None:
    corpus = _Corpus()
    evidence = asyncio.run(gather_world_evidence("Which was better, Bam’s 83 point game or Kobe’s 81 point game?", store=corpus))
    assert {source.title for source in evidence.sources} == {"Bam Adebayo's 83-point game", "Kobe Bryant's 81-point game"}
    assert evidence.searched == []
    rendered = evidence.render()
    assert "scored 83 points" in rendered and "offline Wikipedia" in rendered


def test_unrequested_evidence_must_score_where_matched_pairs_do(monkeypatch) -> None:
    """Between the judge's boundary and its matched floor: admitted only when asked for."""

    def assess(question, passages):
        return [
            evidence_relevance.EvidenceAlignment(True, True, 0.6 if "Kobe" in passage else 0.8, 0.5, (), "stand-in")
            for passage in passages
        ]

    monkeypatch.setattr(evidence_relevance, "assess_evidence_alignments", assess)
    corpus = _Corpus()
    unasked = asyncio.run(gather_world_evidence("Bam’s 83 or Kobe’s 81?", store=corpus))
    assert [source.title for source in unasked.sources] == ["Bam Adebayo's 83-point game"]

    async def search(query):
        return []

    asked = asyncio.run(gather_world_evidence("look up Bam’s 83 or Kobe’s 81", store=corpus, search=search))
    assert {source.title for source in asked.sources} == {"Bam Adebayo's 83-point game", "Kobe Bryant's 81-point game"}


def test_a_link_is_read_and_a_wikipedia_link_survives_the_network(judge) -> None:
    corpus = _Corpus()

    async def fetch(url):
        return {"title": "Bam Adebayo's 83-point game - Wikipedia", "url": url, "text": BAM}

    url = "https://en.wikipedia.org/wiki/Bam_Adebayo%27s_83-point_game"
    read = asyncio.run(gather_world_evidence(url, store=corpus, fetch=fetch))
    assert read.read == [url] and read.sources[0].origin == "the page you sent"

    async def offline(url):
        return None

    copy = asyncio.run(gather_world_evidence(url, store=corpus, fetch=offline))
    assert copy.sources[0].origin == "offline Wikipedia copy of the page you sent" and copy.unreadable == []
    gone = asyncio.run(gather_world_evidence("https://example.com/nothing", store=corpus, fetch=offline))
    assert gone.unreadable == ["https://example.com/nothing"] and "COULD NOT READ" in gone.render()


def test_google_it_searches_for_what_it_points_at(judge) -> None:
    corpus = _Corpus()
    asked: list[str] = []

    async def search(query):
        asked.append(query)
        return [{"title": "Heat's Bam Adebayo scores 83", "url": "https://news.example/bam", "text": "Bam Adebayo scored 83 points."}]

    evidence = asyncio.run(gather_world_evidence("Google it. Bam had an 83 point game", store=corpus, search=search))
    assert asked == ["Bam had an 83 point game"]
    assert evidence.searched == asked and evidence.read == ["https://news.example/bam"]
    asyncio.run(gather_world_evidence("look it up", previous_request="Did Bam score 83?", store=corpus, search=search))
    assert asked[-1] == "Did Bam score 83?"


def test_nothing_is_sent_to_a_search_engine_unasked(judge) -> None:
    corpus = _Corpus()
    asked: list[str] = []

    async def search(query):
        asked.append(query)
        return []

    asyncio.run(gather_world_evidence("My friend Sarah thinks Bam is overrated", store=corpus, search=search))
    assert asked == []


def test_finding_something_out_is_research_unless_an_app_or_her_own_place_is_named() -> None:
    assert read_search_request("Google Bam’s 83 point game. I’m telling you he had one").query == "Bam’s 83 point game"
    assert read_search_request("search my files for the budget").local_place == "file"
    assert not read_search_request("How are you?").asked
    assert not looks_like_desktop_objective("Google it. Bam had an 83 point game")
    assert looks_like_desktop_objective("open safari and google the weather")
    assert looks_like_desktop_objective("search my files for the budget spreadsheet")


def test_the_sources_travel_beside_the_message_and_reach_only_her_reply() -> None:
    """Appended to the message they were read as material the person supplied."""
    from core.brain.inference_gate_turn_serving import _ServesTheTurn
    from core.conversation.turn_evidence_custody import (
        bind_turn_evidence_custody,
        record_turn_world_evidence,
        turn_world_evidence,
    )

    async def blocks(visible: str) -> list[str]:
        _ambient, task = await _ServesTheTurn._generate_with_metadata_sink_task_grounding_blocks(
            {}, [], False, "", "", None, visible,
        )
        return task

    with bind_turn_evidence_custody(session_id="s", turn_id="t"):
        assert record_turn_world_evidence("[SOURCE 1: offline Wikipedia] Bam Adebayo scored 83 points.")
        assert turn_world_evidence() == ("[SOURCE 1: offline Wikipedia] Bam Adebayo scored 83 points.",)
        assert any("scored 83 points" in block for block in asyncio.run(blocks("Bam or Kobe?")))
        assert not any("scored 83 points" in block for block in asyncio.run(blocks("")))
    assert turn_world_evidence() == ()


def test_what_did_you_learn_reads_back_what_the_last_lookup_read(judge, monkeypatch) -> None:
    from core.conversation import what_the_world_says
    from core.conversation.what_the_world_says import (
        WorldEvidence,
        answer_from_earlier_reading,
        remember_reading,
    )

    monkeypatch.setattr(what_the_world_says, "_last_read", what_the_world_says.OrderedDict())
    looked_up = asyncio.run(gather_world_evidence("Did Bam score 83?", store=_Corpus()))
    remember_reading("session-a", looked_up)

    asked = WorldEvidence()
    assert answer_from_earlier_reading(asked, "What did you learn?", "session-a")
    assert asked.sources[0].origin == "read for an earlier turn (offline Wikipedia)"
    assert "scored 83 points" in asked.render()
    # Another conversation's reading, or a question about something else, is not carried.
    assert not answer_from_earlier_reading(WorldEvidence(), "What did you learn?", "session-b")
    assert not answer_from_earlier_reading(WorldEvidence(), "what did you learn about physics in school", "session-a")


def test_her_training_ledger_does_not_answer_a_question_about_a_lookup(monkeypatch) -> None:
    from core.conversation import turn_evidence_custody
    from core.learning import learning_selfreport

    report = learning_selfreport.LearningSelfReport()
    monkeypatch.setattr(report, "_build_block", lambda: "LEARNING STATUS")
    monkeypatch.setattr(turn_evidence_custody, "turn_world_evidence", lambda: ("source 1: offline Wikipedia",))
    assert report.get_context_injection("what did you learn") == ""
    assert report.get_context_injection("have you been training lately?") == "LEARNING STATUS"
    monkeypatch.setattr(turn_evidence_custody, "turn_world_evidence", lambda: ())
    assert report.get_context_injection("what did you learn") == "LEARNING STATUS"


def test_what_the_grammar_reads_exactly_teaches_the_learned_surfaces(judge, monkeypatch) -> None:
    from core.language import search_request

    for surface in (search_request._FINDS_OUT, search_request._ASKS_WHAT_WAS_FOUND):
        monkeypatch.setattr(surface, "positives", surface.positives)

    async def search(query):
        return [{"title": "Bam scores 83", "url": "https://news.example/bam", "text": "Bam Adebayo scored 83 points."}]

    asyncio.run(gather_world_evidence("Look up Bam's 83 point game", store=_Corpus(), search=search))
    assert "Look up Bam's 83 point game" in search_request._FINDS_OUT.positives
    search_request.teach_from_the_floor("so what did you find?")
    assert "so what did you find?" in search_request._ASKS_WHAT_WAS_FOUND.positives
    search_request.teach_from_the_floor("How are you feeling tonight?")
    assert "How are you feeling tonight?" not in search_request._FINDS_OUT.positives


def test_a_page_is_read_from_its_prose_not_its_menu() -> None:
    """LIVE 2026-10-03: the passage of a fetched Wikipedia page was its navigation."""
    menu = "\n".join(["Jump to content", "Main menu", "Navigation", "Main page", "Contents", "Search", "Log in"] * 3)
    caption = "The Kaseya Center in Miami, where the game took place."
    lead = (
        "On March 10, 2026, Bam Adebayo scored 83 points for the Miami Heat. "
        "It is the second-highest total in NBA history."
    )
    reaction = "After the game, Heat head coach Erik Spoelstra called it an absolutely surreal night."
    page = f"{menu}\n{caption}\n{lead}\nHeading\n{reaction}"
    bare = passage_of(page)
    assert "On March 10, 2026" in bare and "Main menu" not in bare and "Log in" not in bare
    assert "Spoelstra" in passage_of(page, ["Spoelstra"])
