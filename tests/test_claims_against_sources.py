"""A dated claim in her reply is read against what the turn's own sources say about the same thing.

LIVE 2026-10-04: "then FTX Arena after they bought the rights for $135 million
in 2022", from a turn whose Wikipedia page says "In March 2021, FTX acquired
the naming rights to the arena in a $135 million, 19-year agreement." Read
against the live page with the live encoder, that was the one claim of four
reported, against that sentence (0.795).
"""

from __future__ import annotations

import asyncio

import pytest

from core.cognition import evidence_relevance
from core.conversation.claims_against_sources import claims_with_years, disagreements

REPLY = (
    "December 31, 1999. New Year's Eve launch night with a Gloria Estefan concert. Groundbreaking was "
    "February 6, 1998, so it took about two years from dirt to doors open. And the naming history is "
    "messier than people remember: American Airlines Arena first (1999–2021), then FTX Arena after they "
    "bought the rights for $135 million in 2022 — a deal that lasted barely two months."
)
PAGE = {
    "origin": "web",
    "title": "Kaseya Center - Wikipedia",
    "location": "https://en.wikipedia.org/wiki/Kaseya_Center",
    "text": (
        "Kaseya Center is a multi-purpose arena on Biscayne Bay in Miami. It opened on December 31, 1999.\n"
        "Groundbreaking: February 6, 1998\n"
        "American Airlines Arena (1999–2021)\n"
        "In March 2021, FTX acquired the naming rights to the arena in a $135 million, 19-year agreement.\n"
        "FTX filed for bankruptcy in November 2022, and the county ended the agreement."
    ),
}


@pytest.fixture
def judge(monkeypatch):
    """Nearest by shared words, as the encoder would rank these; measured."""

    def assess(question, passages):
        asked = set(question.lower().split())
        out = []
        for passage in passages:
            shared = len(asked & set(passage.lower().split())) / max(1, len(asked))
            score = 0.6 + 0.4 * shared
            out.append(evidence_relevance.EvidenceAlignment(score >= 0.55, True, score, 0.5, (), "stand-in"))
        return out

    monkeypatch.setattr(evidence_relevance, "assess_evidence_alignments", assess)


def test_a_dated_clause_is_a_claim_and_a_date_alone_is_about_what_was_asked():
    claims = dict(claims_with_years(REPLY, "When did the Kaseya Center open?"))

    assert claims["December 31, 1999"] == {"kaseya", "center"}
    assert claims["then FTX Arena after they bought the rights for $135 million in 2022"] == {"ftx", "arena"}


def test_the_claim_its_source_dates_otherwise_is_the_one_reported(judge):
    found = disagreements(REPLY, [PAGE], "When did the Kaseya Center open?")

    assert [one.claim for one in found] == ["then FTX Arena after they bought the rights for $135 million in 2022"]
    assert found[0].says.startswith("In March 2021, FTX acquired the naming rights")
    assert found[0].location == PAGE["location"]


def test_a_sentence_that_only_bears_on_the_claim_does_not_date_it(monkeypatch):
    """LIVE 2026-10-05: the bankruptcy (2022) was paired at 0.516 with the naming deal (2021)."""
    monkeypatch.setattr(
        evidence_relevance,
        "assess_evidence_alignments",
        lambda question, passages: [
            evidence_relevance.EvidenceAlignment(True, True, 0.516, 0.5, (), "stand-in") for _ in passages
        ],
    )

    assert disagreements("FTX filed for bankruptcy in November 2022.", [PAGE], "names?") == []


def test_silence_is_not_disagreement(judge):
    assert disagreements("The Heat won the title in 2006.", [PAGE], "Who won?") == []


def test_nothing_is_reported_when_the_judge_has_not_measured(monkeypatch):
    monkeypatch.setattr(
        evidence_relevance,
        "assess_evidence_alignments",
        lambda question, passages: [
            evidence_relevance.EvidenceAlignment(False, False, None, None, (), "unmeasured") for _ in passages
        ],
    )

    assert disagreements(REPLY, [PAGE], "When did the Kaseya Center open?") == []


def test_a_disagreeing_draft_is_asked_again_with_the_sources_sentence(judge, monkeypatch):
    from core.conversation import turn_evidence_custody
    from interface.routes import chat_reply_checks

    shown: list[str] = []
    monkeypatch.setattr(turn_evidence_custody, "turn_world_sources", lambda: (PAGE,))
    monkeypatch.setattr(turn_evidence_custody, "record_turn_world_evidence", lambda text: shown.append(text) or True)
    monkeypatch.setattr(turn_evidence_custody, "record_turn_grounding", lambda text: True)
    asked_again: list[tuple[str, tuple[str, ...]]] = []

    async def retry(draft, reasons):
        asked_again.append((draft, tuple(reasons)))
        return "It opened on December 31, 1999. FTX bought the naming rights in March 2021."

    revised = asyncio.run(chat_reply_checks._against_its_sources("When did the Kaseya Center open?", REPLY, retry))

    assert revised.startswith("It opened on December 31, 1999.")
    assert asked_again == [(REPLY, (chat_reply_checks.DISAGREES_WITH_ITS_SOURCES,))]
    assert "In March 2021, FTX acquired the naming rights" in shown[0]


def test_a_draft_its_sources_agree_with_is_not_asked_again(judge, monkeypatch):
    from core.conversation import turn_evidence_custody
    from interface.routes import chat_reply_checks

    monkeypatch.setattr(turn_evidence_custody, "turn_world_sources", lambda: (PAGE,))

    async def retry(draft, reasons):
        raise AssertionError("asked again with nothing to correct")

    agreeing = "It opened on December 31, 1999. FTX Arena dates from March 2021."
    assert asyncio.run(chat_reply_checks._against_its_sources("When did it open?", agreeing, retry)) is None
