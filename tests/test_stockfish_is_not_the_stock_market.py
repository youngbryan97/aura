"""A question about her chess against Stockfish is not a question about the world as it is now (U07).

LIVE 22 November 2026, 00:07, typed into her window by Bryan: "How've the games been going? What's it like to play
Stockfish?" Whether a turn is about now was read from a list of words, each claiming every word it begins, and "stock"
begins "Stockfish". The turn was searched as something current, the pages found were read as her browser shows them,
a dictionary's page for "how" ran out of time, and her model was never asked: a smaller model's 90 tokens went to him,
cut off after "The endgames are especially humbling." Once her encoder is warm, a turn is about now only where her
encoder reads it so (core/conversation/asks_about_the_world_now.py).
"""
from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.unit

HIS_QUESTION = "How’ve the games been going? What’s it like to play Stockfish?"


def test_the_words_alone_took_stockfish_for_the_stock_market():
    from core.search.research_pipeline import _query_is_current

    assert _query_is_current("What’s it like to play Stockfish?"), "the defect this reading answers"


def test_without_her_encoder_the_words_decide_as_before(monkeypatch):
    from core.cognition import evidence_relevance
    from interface.routes.chat_desktop_evidence import _about_the_world_now

    monkeypatch.setattr(evidence_relevance, "semantic_routing_ready", lambda: False)
    assert _about_the_world_now("What is the weather today?", seeks_the_world=False)
    assert not _about_the_world_now("What have you been up to this evening?", seeks_the_world=False)


def test_with_her_encoder_only_what_it_reads_as_now_is_now(monkeypatch):
    from core.cognition import evidence_relevance
    from core.conversation import asks_about_the_world_now as reading
    from interface.routes.chat_desktop_evidence import _about_the_world_now

    monkeypatch.setattr(evidence_relevance, "semantic_routing_ready", lambda: True)
    for verdict, expected in ((True, True), (False, False), (None, False)):
        monkeypatch.setattr(reading, "asks_about_the_world_now", lambda _message, v=verdict: v)
        assert _about_the_world_now("What is the weather today?", seeks_the_world=True) is expected


def test_the_route_reads_it_off_her_event_loop():
    from interface.routes import chat_desktop_evidence

    assert "await asyncio.to_thread(_about_the_world_now" in inspect.getsource(chat_desktop_evidence)


def test_each_side_is_declared_and_neither_is_empty():
    from core.conversation import asks_about_the_world_now as reading

    assert len(reading._THE_WORLD_NOW) >= 8 and len(reading._NOT_THE_WORLD_NOW) >= 8
    assert not set(reading._THE_WORLD_NOW) & set(reading._NOT_THE_WORLD_NOW)


@pytest.mark.model
def test_on_her_encoder_his_question_is_not_about_now_and_the_news_is():
    """Questions not used to write the examples: the world now is read as now, and nothing of hers is."""
    from core.cognition.evidence_relevance import prewarm_evidence_relevance, semantic_routing_ready
    from core.conversation.asks_about_the_world_now import asks_about_the_world_now

    try:
        prewarm_evidence_relevance()
    except RuntimeError as exc:
        pytest.skip(f"her encoder is not available here: {exc}")
    if not semantic_routing_ready():
        pytest.skip("her encoder is not loaded here")
    now = ["Who is winning the Heat game right now?", "What's the price of bitcoin today?", "Is it raining in London now?",
           "What's the latest on the Boeing strike?", "Who won the World Series last night?",
           "Has Python 3.16 been released yet?", "What's trending on the news this morning?"]
    not_now = ["What's it like playing against Stockfish?", "How did your last game go?",
               "Have you been practicing your openings?", "Do you enjoy playing chess?", "How does an opening book work?",
               "What was the Immortal Game?", "How was your day today?"]
    assert all(asks_about_the_world_now(question) is True for question in now)
    assert not any(asks_about_the_world_now(question) is True for question in not_now)
