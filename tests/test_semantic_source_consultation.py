"""Corpus, memory and model can suggest but cannot certify a law."""

import asyncio
from types import SimpleNamespace

from core.cognition.semantic_development import SemanticCase, SemanticDevelopment
from core.cognition.semantic_source_consultation import (
    HypothesisBatch,
    consult_semantic_sources,
)


class _Corpus:
    def search(self, query, limit, *, deadline_s):
        assert query == "signal transfer"
        assert deadline_s <= 0.25
        return [SimpleNamespace(doc_id=7, title="A relation", snippet="a possible pattern")]


class _Memory:
    def search_memories(self, query, top_k):
        return [{"id": "note-1", "text": "a remembered contrast"}]


class _Advisor:
    async def generate(self, prompt, *, is_background, deadline_s):
        assert is_background is True
        assert deadline_s <= 20
        assert "corpus:7" in prompt
        return HypothesisBatch.model_validate({"laws": [
            {"source_reference": "corpus:7", "predicates": [
                {"feature": "signal", "op": "==", "value": True}]},
            {"source_reference": "resident:model", "predicates": [
                {"feature": "unseen_answer", "op": "==", "value": True}]},
            {"source_reference": "missing:source", "predicates": [
                {"feature": "signal", "op": "==", "value": True}]},
        ]})


def test_local_sources_only_seed_fit_bounded_unaudited_proposals(tmp_path):
    engine = SemanticDevelopment(state_path=tmp_path / "semantic.json", min_support=4)
    for index in range(24):
        engine.observe(SemanticCase(
            f"source-{index}", "room", "transfer", index % 2 == 0,
            {"signal": index % 2 == 0, "noise": index % 3 == 0},
            observed_at=float(index)))
    result = asyncio.run(consult_semantic_sources(
        engine, outcome_name="transfer", query="signal transfer",
        corpus=_Corpus(), memory=_Memory(), advisor=_Advisor()))
    assert result["status"] == "proposed"
    assert result["leads"] == ("corpus:7", "memory:note-1")
    assert len(result["proposals"]) == 1
    proposal = engine.proposals[result["proposals"][0]]
    assert proposal.channel == "corpus"
    assert proposal.exposure_audited is False
    assert result["serving_authority"] is False
