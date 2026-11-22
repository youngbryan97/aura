"""Whether a question asks for a fact about something named is read by meaning, on her encoder (U07).

LIVE 22 November 2026: a question asking her to design something general from her chess play ("What would that look
like?", with Stockfish named in it) was read as a factual question about a named thing by its words, and searched on
the web whole. The reading is now her encoder's, against declared examples of each side, and where her encoder is not
loaded it says nothing (core/conversation/asks_for_a_fact.py).
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_without_her_encoder_it_says_nothing(monkeypatch):
    from core.conversation import asks_for_a_fact as reading
    from core.language import learned_matcher

    monkeypatch.setattr(learned_matcher, "her_encoder_only", lambda sentences: [])
    monkeypatch.setattr(reading, "_SURFACE", [])
    monkeypatch.setattr(reading, "_SURFACE_FAILED", [])
    assert reading.asks_for_a_fact("Who founded Hugging Face?") is None
    assert reading.asks_for_a_fact("") is None


def test_each_side_is_declared_and_neither_is_empty():
    from core.conversation import asks_for_a_fact as reading

    assert len(reading._A_FACT) >= 8 and len(reading._HER_WORKING_OUT) >= 8
    assert not set(reading._A_FACT) & set(reading._HER_WORKING_OUT)


@pytest.mark.model
def test_on_her_encoder_facts_are_told_from_her_working_out():
    """Unseen questions: facts are read as facts, and a request for her working-out is never read as one."""
    from core.cognition.evidence_relevance import prewarm_evidence_relevance
    from core.conversation.asks_for_a_fact import asks_for_a_fact

    prewarm_evidence_relevance()
    facts = ["When was the Louvre founded?", "Who is the CEO of Nvidia?", "How much did the James Webb telescope cost?",
             "What is the capital of Australia?", "Who wrote the novel Dune?"]
    working_out = ["Aura, I think we can work with this! I want this kind of reasoning to apply in any context, with "
                   "beating Stockfish as a proof of concept on the way. What would that look like?",
                   "Could you sketch how a bot that beats Stockfish would think?",
                   "What would it take to make this work for poker?",
                   "How should we structure the trajectory layer?"]
    if asks_for_a_fact(facts[0]) is None:
        pytest.skip("her encoder is not loaded here")
    assert all(asks_for_a_fact(question) is True for question in facts)
    assert not any(asks_for_a_fact(question) is True for question in working_out)
