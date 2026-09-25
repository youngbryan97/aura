"""Four readers that asked the container for names nothing registers.

Each got None on every turn. The preferences she heard never reached her
beliefs, the coherence gate on self-modification never moved off 1.0, the
executive never saw her tasks end, and self-play's failures never reached the
distillation queue. Each now reads the owner, and none of them builds one.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.agency import task_commitment_verifier as verifiers
from core.cognition.knowledge_enrichment import KnowledgeEnricher
from core.epistemics.belief_revision import BeliefDomain
from core.self_modification import shadow_runtime as shadows

pytestmark = pytest.mark.unit


class _Beliefs:
    def __init__(self) -> None:
        self.claims: list[tuple[str, str, str, float]] = []

    async def process_new_claim(self, claim: str, domain: str, source: str, confidence: float = 0.5):
        self.claims.append((claim, domain, source, confidence))
        return {"ok": True}


class _Graph:
    def add_knowledge(self, **_: object) -> None:
        return None


def _hear_a_preference(monkeypatch, registered: object | None) -> KnowledgeEnricher:
    import core.container as container

    monkeypatch.setattr(
        container.ServiceContainer,
        "get",
        classmethod(lambda cls, name, default=None: registered if name == "belief_revision_engine" else default),
    )
    enricher = KnowledgeEnricher(knowledge_graph=_Graph(), brain=object())

    async def _extract(_excerpt: str) -> list[dict[str, str]]:
        return [{"type": "preference", "content": "likes green tea"}]

    enricher._extract = _extract  # type: ignore[method-assign]
    messages = [
        {"role": "user", "content": "I really like green tea in the mornings."},
        {"role": "assistant", "content": "Green tea it is, then."},
    ]
    asyncio.run(enricher.enrich_from_conversation(messages, force=True))
    return enricher


def test_a_preference_she_hears_reaches_her_belief_engine(monkeypatch) -> None:
    beliefs = _Beliefs()
    _hear_a_preference(monkeypatch, beliefs)
    assert beliefs.claims == [("The user likes green tea", BeliefDomain.USER, "conversation", 0.75)]


def test_before_boot_registers_the_engine_nothing_is_built(monkeypatch) -> None:
    import core.epistemics.belief_revision as belief_revision

    monkeypatch.setattr(belief_revision, "_instance", None)
    enricher = _hear_a_preference(monkeypatch, None)
    assert enricher._last_outcome != "completed_with_storage_errors"
    assert belief_revision._instance is None


def test_the_coherence_gate_reaches_the_shadow_runtime_self_modification_built(monkeypatch) -> None:
    monkeypatch.setattr(shadows, "_instance", None)
    assert shadows.existing_shadow_runtime() is None
    assert shadows._instance is None
    built = shadows.get_shadow_runtime(".")
    assert shadows.existing_shadow_runtime() is built
    built.set_coherence_gate(0.2)
    assert built._current_phi == pytest.approx(0.2)


def test_the_executive_reads_the_verifier_holding_her_tasks_without_building_one(monkeypatch) -> None:
    monkeypatch.setattr(verifiers, "_verifier", None)
    assert verifiers.existing_task_commitment_verifier() is None
    held = SimpleNamespace(get_all_active=lambda: [])
    monkeypatch.setattr(verifiers, "_verifier", held)
    assert verifiers.existing_task_commitment_verifier() is held
