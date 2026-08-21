"""The offline corpus answers timeless questions; the web answers current ones.

LIVE, 2026-08-10. Asked for a detail about Michael T. Wright's Antikythera
planetarium model, she correctly decided she did not know and looked it up —
capability token minted, intention declared, web_search dispatched — and spent
23,145ms on it. The same fact is in the local corpus (7,189,653 Wikipedia
pages), which answers that class of question in tens of milliseconds.

The corpus was already wired into this skill, but only as a DEGRADED fallback
for when the web is unreachable. So the fast, private, offline copy was
consulted only after the slow path had already failed.

Two defects fell out of fixing the ordering, both found by measuring:

  * document_count() was the availability guard — SELECT COUNT(*) over 7.19M
    rows in a 37GB table, ~6s on this host, paid before a search that then
    took 47ms. has_documents() already existed for exactly this and says so in
    its own docstring; the hot paths had never been switched to it.

  * _CURRENTNESS_TERMS held "today" and "now" but not the rest of the temporal
    deictic family, so "who won the election yesterday" was classified timeless
    and could have been answered from a dated snapshot.
"""

from __future__ import annotations

import pytest


def _wants_current(query: str) -> bool:
    from core.skills.web_search import EnhancedWebSearchSkill

    return EnhancedWebSearchSkill._query_wants_current_information(query)


@pytest.mark.parametrize(
    "query",
    [
        "who won the election yesterday",
        "what happened last night",
        "bitcoin price today",
        "what is the latest version of python",
        "breaking news on the strike",
        "who is winning right now",
        "what is the score at the moment",
        "the upcoming release date",
    ],
)
def test_time_sensitive_questions_go_to_the_network(query: str) -> None:
    assert _wants_current(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "Antikythera mechanism gear count",
        "Liskov substitution principle",
        "Kessler syndrome",
        "how does photosynthesis work",
        "who was Ada Lovelace",
    ],
)
def test_timeless_questions_prefer_the_offline_corpus(query: str) -> None:
    assert _wants_current(query) is False


def test_freshness_policy_is_shared_not_restated() -> None:
    """One definition of "needs to be current" in the runtime, not two.

    The router delegates to the research pipeline's own policy, so extending
    the term family fixes both callers at once — which is how "yesterday" was
    fixed for the pipeline and this skill in a single edit.
    """
    from core.search.research_pipeline import _CURRENTNESS_TERMS

    for term in ("today", "yesterday", "tonight", "last night", "right now"):
        assert term in _CURRENTNESS_TERMS, term


def test_availability_guard_is_the_o1_check_not_a_table_scan() -> None:
    """The 6-second guard in front of a 47-millisecond answer."""
    import inspect

    from core.skills import local_reference, web_search

    for module in (web_search, local_reference):
        source = inspect.getsource(module)
        assert "has_documents()" in source, module.__name__
        assert "document_count() <= 0" not in source, module.__name__


def test_local_first_is_skipped_for_current_queries(monkeypatch) -> None:
    """A live question must not be answered from a snapshot."""
    from core.skills.web_search import EnhancedWebSearchSkill as WebSearchSkill

    called: list[str] = []
    monkeypatch.setattr(
        WebSearchSkill,
        "_local_corpus_fallback",
        classmethod(lambda cls, q, n: called.append(q) or {"ok": True}),
    )

    assert WebSearchSkill._local_corpus_first("bitcoin price today", 3) is None
    assert called == []


def test_local_first_labels_its_provenance(monkeypatch) -> None:
    """A snapshot answer must never be presentable as live web evidence.

    This runtime has already made that mistake out loud — "I checked live web
    evidence" over a result that was not live.
    """
    from core.skills.web_search import EnhancedWebSearchSkill as WebSearchSkill

    monkeypatch.setattr(
        WebSearchSkill,
        "_local_corpus_fallback",
        classmethod(
            lambda cls, q, n: {
                "ok": True,
                "provenance": "local_corpus",
                "offline_fallback": True,
                "results": [{"title": "T", "snippet": "S"}],
                "summary": "old summary",
            }
        ),
    )

    answered = WebSearchSkill._local_corpus_first("Kessler syndrome", 3)

    assert answered is not None
    assert answered["provenance"] == "local_corpus"
    # It is a preference, not a degradation.
    assert answered["offline_fallback"] is False
    assert answered["offline_preferred"] is True
    assert "no network used" in answered["summary"]


def test_local_first_projects_snapshot_evidence_onto_skill_contract(monkeypatch) -> None:
    """Offline evidence remains evidence without being mislabeled as live web."""
    from core.skills.web_search import EnhancedWebSearchSkill as WebSearchSkill

    monkeypatch.setattr(
        WebSearchSkill,
        "_local_corpus_fallback",
        classmethod(
            lambda cls, q, n: {
                "ok": True,
                "provenance": "local_corpus",
                "results": [
                    {
                        "title": "Solaris",
                        "snippet": "Solaris is a novel by Stanislaw Lem.",
                        "source": "enwiki:Solaris_(novel)",
                        "provenance": "local_corpus",
                    }
                ],
            }
        ),
    )

    answered = WebSearchSkill._finalize_result(
        "who wrote Solaris",
        WebSearchSkill._local_corpus_first("who wrote Solaris", 3) or {},
    )

    assert answered["criteria_results"]["sources gathered"] is True
    assert answered["sources"] == [
        {
            "title": "Solaris",
            "snippet": "Solaris is a novel by Stanislaw Lem.",
            "source": "enwiki:Solaris_(novel)",
            "provenance": "local_corpus",
            "evidence_kind": "offline_reference_snapshot",
            "fetched": False,
        }
    ]
    assert "url" not in answered["sources"][0]


@pytest.mark.asyncio
async def test_local_first_result_passes_the_real_source_expectation(monkeypatch) -> None:
    """Reproduce the live research-cycle route through its actual verifier."""
    from core.capability_engine import CapabilityEngine
    from core.runtime.skill_contract import apply_action_expectation_payload
    from core.skills.web_search import EnhancedWebSearchSkill

    skill = EnhancedWebSearchSkill()
    monkeypatch.setattr(
        skill,
        "_local_corpus_first",
        lambda q, n: {
            "ok": True,
            "provenance": "local_corpus",
            "offline_preferred": True,
            "results": [
                {
                    "title": "Biocomputing",
                    "snippet": "A dated reference entry.",
                    "source": "enwiki:Biocomputing",
                    "provenance": "local_corpus",
                }
            ],
            "summary": "Answered from the local reference snapshot.",
        },
    )

    async def _network_must_not_run(*args, **kwargs):
        raise AssertionError("timeless local-first query reached the network")

    monkeypatch.setattr(skill.pipeline, "search", _network_must_not_run)
    params = {"query": "the history and future of biocomputing"}
    context = {"origin": "research_cycle", "requires_sources": True}

    payload = await skill.execute(params, context)
    expectation = CapabilityEngine.action_expectation_for("web_search", params, context)
    checked = apply_action_expectation_payload("web_search", payload, expectation)

    assert checked["ok"] is True
    assert checked["status"] == "success_verified"
    assert checked["expectation_verdict"]["passed"] is True
    assert checked["expectation_verdict"]["present_evidence"] == ["sources", "summary"]
    assert checked["sources"][0]["evidence_kind"] == "offline_reference_snapshot"


@pytest.mark.asyncio
async def test_raised_web_pipeline_fallback_keeps_sources_for_verification(monkeypatch) -> None:
    """A raising network lane cannot bypass result finalization."""
    from core.skills.web_search import EnhancedWebSearchSkill

    skill = EnhancedWebSearchSkill()

    async def _raise(*args, **kwargs):
        raise OSError("network unavailable")

    monkeypatch.setattr(skill.pipeline, "search", _raise)
    monkeypatch.setattr(
        skill,
        "_local_corpus_fallback",
        lambda q, n: {
            "ok": True,
            "provenance": "local_corpus",
            "offline_fallback": True,
            "results": [
                {
                    "title": "Biocomputing",
                    "snippet": "A dated reference entry.",
                    "source": "enwiki:Biocomputing",
                    "provenance": "local_corpus",
                }
            ],
            "summary": "Answered from the local reference snapshot.",
        },
    )

    result = await skill.execute(
        {"query": "research the history of biocomputing", "deep": False},
        {"origin": "research_cycle"},
    )

    assert result["ok"] is True
    assert result["web_error"] == "network unavailable"
    assert result["sources"][0]["source"] == "enwiki:Biocomputing"
    assert result["sources"][0]["evidence_kind"] == "offline_reference_snapshot"
    assert result["criteria_results"]["sources gathered"] is True


def test_an_empty_corpus_falls_through_to_the_web(monkeypatch) -> None:
    from core.skills.web_search import EnhancedWebSearchSkill as WebSearchSkill

    monkeypatch.setattr(
        WebSearchSkill, "_local_corpus_fallback", classmethod(lambda cls, q, n: None)
    )

    assert WebSearchSkill._local_corpus_first("Kessler syndrome", 3) is None


def test_unknown_freshness_prefers_the_network(monkeypatch) -> None:
    """If the policy cannot be consulted, the network is the safe answer."""
    import builtins

    from core.skills.web_search import EnhancedWebSearchSkill as WebSearchSkill

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "core.search.research_pipeline":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)

    assert WebSearchSkill._query_wants_current_information("Kessler syndrome") is True
