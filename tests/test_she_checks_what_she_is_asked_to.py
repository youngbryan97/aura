"""L10: asked to check a claim, she checks it before she answers, and keeps what she read.

LIVE 2026-10-20, typed into her chat: "Is it true that the speed of light in a vacuum is exactly 299,792,458 metres per
second? Can you check that?" She answered from what she remembered. Her check (core/epistemics/epistemic_reach.py)
ran only on her own felt doubt, nothing read the request as asking for one, and her store of what she has verified had
never been written in her app.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

CLAIM = "the Eiffel Tower opened in 1889 in Paris"


@pytest.mark.parametrize(("typed", "claim", "points_back"), [
    ("Is it true that the speed of light in a vacuum is exactly 299,792,458 metres per second? Can you check that?",
     "the speed of light in a vacuum is exactly 299,792,458 metres per second", False),
    ("Check whether Canberra is the capital of Australia.", "Canberra is the capital of Australia", False),
    ("The Eiffel Tower is 330 metres tall. Can you verify that for me?", "The Eiffel Tower is 330 metres tall", False),
    ("Is it right? I read that Sydney is the capital of Australia.", "Sydney is the capital of Australia", False),
    ("Can you check that?", "", True),
    ("is that true?", "", True),
])
def test_a_request_to_check_and_what_it_asks_about(typed, claim, points_back):
    from core.language.search_request import asks_to_check

    asked = asks_to_check(typed)
    assert asked.asked and asked.claim == claim and asked.points_back is points_back


@pytest.mark.parametrize("typed", ["check my email", "Can you check the oven timer on my screen?", "how are you?",
                                   "What is the capital of Australia?"])
def test_not_a_request_to_check(typed):
    from core.language.search_request import asks_to_check

    assert not asks_to_check(typed).asked


class _Source:
    def __init__(self, url: str, evidence: str) -> None:
        self.NAME, self.url, self.evidence, self.asked = url, url, evidence, 0

    def is_permitted(self, _policy) -> bool:
        return True

    async def lookup(self, _gateway, _terms):
        self.asked += 1
        return self.evidence, self.url


def _engine(sources, monkeypatch):
    from core.epistemics import epistemic_reach

    monkeypatch.setattr(epistemic_reach, "_how_it_reads", lambda evidence, claim: None)  # no reader model loaded here
    gateway = SimpleNamespace(policy=SimpleNamespace(read_hosts=frozenset(), mutate_hosts=frozenset()))
    return epistemic_reach.EpistemicReachEngine(gateway=gateway, sources=sources)


def test_a_claim_she_is_asked_to_check_is_read_kept_and_not_fetched_again(monkeypatch):
    from core.epistemics.epistemic_reach import check_asked
    from core.epistemics.what_she_has_verified import VERIFIED, her_verified_facts

    sources = [_Source("https://en.wikipedia.org/wiki/Eiffel_Tower", "The Eiffel Tower in Paris opened in 1889."),
               _Source("https://www.culture.gouv.fr/eiffel", "In Paris the Eiffel Tower opened to the public in 1889.")]
    engine = _engine(sources, monkeypatch)
    read = check_asked(CLAIM, timeout_s=30.0, reach=engine)
    assert [v.source_url for v in read] == [s.url for s in sources]
    assert her_verified_facts().get(CLAIM).status == VERIFIED
    again = check_asked(CLAIM, timeout_s=30.0, reach=engine)
    assert len(again) == 1 and "verified before" in again[0].reason and [s.asked for s in sources] == [1, 1]


def test_what_contradicts_a_claim_somebody_else_made_is_no_correction_of_hers(monkeypatch):
    from core.epistemics.epistemic_reach import check_asked

    engine = _engine([_Source("https://en.wikipedia.org/wiki/Australia", "The capital of Australia is Canberra, not Sydney.")],
                     monkeypatch)
    check_asked("Sydney is the capital of Australia", timeout_s=30.0, reach=engine)
    assert engine.pending_corrections() == 0


def test_her_answer_is_given_what_her_check_read(monkeypatch):
    from core.conversation import what_the_world_says
    from core.epistemics import epistemic_reach

    engine = _engine([_Source("https://en.wikipedia.org/wiki/Eiffel_Tower", "The Eiffel Tower in Paris opened in 1889.")],
                     monkeypatch)
    monkeypatch.setattr(epistemic_reach, "get_epistemic_reach", lambda: engine)
    found = asyncio.run(what_the_world_says.her_check_reads(f"Is it true that {CLAIM}? Can you check that?"))
    assert len(found) == 1 and found[0].location.endswith("Eiffel_Tower") and "1889" in found[0].text
    assert "read for her check" in found[0].origin
    assert asyncio.run(what_the_world_says.her_check_reads("How are you today?")) == []


def test_her_desktop_turn_is_given_what_was_read_for_it():
    """LIVE 2026-10-20: her check read Wikipedia's page before she answered, and her desktop turn, which builds its own
    messages, was never given it; her answer said "I can verify from what I know"."""
    from core.brain.what_she_brings_to_a_turn import grounding_she_brings
    from core.conversation.turn_evidence_custody import (
        bind_turn_evidence_custody,
        record_turn_world_evidence,
    )

    read = "source 1: read for her check, \"Speed of light\" ... exactly 299,792,458 metres per second"
    with bind_turn_evidence_custody(session_id="s", turn_id="t-check"):
        assert record_turn_world_evidence(read)
        assert read in grounding_she_brings({}, "Is it true that c is exactly 299,792,458 m/s?")
    assert read not in grounding_she_brings({}, "Is it true that c is exactly 299,792,458 m/s?")  # that turn's only
