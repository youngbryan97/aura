from types import SimpleNamespace

import pytest


class _AllowingAuthorityGateway:
    def authorize_belief_update_sync(self, *_args, **_kwargs):
        return SimpleNamespace(
            approved=True,
            reason="approved",
            outcome="approved",
            will_receipt_id="will-belief",
            substrate_receipt_id=None,
            executive_intent_id=None,
        )


def _patch_authority(monkeypatch):
    import core.executive.authority_gateway as authority

    monkeypatch.setattr(authority, "get_authority_gateway", lambda: _AllowingAuthorityGateway())


def test_belief_ops_degradation_audit_is_clean():
    from pathlib import Path

    from tools.audit_degradation import analyze_file

    assert analyze_file(Path("core/skills/belief_ops.py")) == []


@pytest.mark.asyncio
async def test_add_belief_rejects_invalid_raw_parameters():
    from core.skills.belief_ops import AddBeliefSkill

    result = await AddBeliefSkill().execute(
        {"source": "Bryan", "relation": "prefers", "target": "clarity", "confidence": 2.0},
        {},
    )

    assert result["ok"] is False
    assert "Invalid belief update parameters" in result["error"]


@pytest.mark.asyncio
async def test_add_belief_marks_memfs_only_write_as_non_causal(monkeypatch):
    from core.skills.belief_ops import AddBeliefSkill
    from core.skills.memory_ops import MemoryOpsSkill

    _patch_authority(monkeypatch)
    monkeypatch.setattr("core.container.ServiceContainer.get", lambda _name, default=None: default)

    async def _memfs_ok(self, payload, context):
        assert payload["action"] == "core_append"
        return {"ok": True}

    monkeypatch.setattr(MemoryOpsSkill, "execute", _memfs_ok)

    result = await AddBeliefSkill().execute(
        {"source": "Bryan", "relation": "prefers", "target": "operational truth"},
        {},
    )

    assert result["ok"] is True
    assert result["data"]["causal"] is False
    assert result["data"]["causal_layers"] == []
    assert result["data"]["archival_layers"] == ["memfs:user"]
    assert "not causally installed" in result["summary"]


@pytest.mark.asyncio
async def test_add_belief_continues_to_causal_layers_after_primary_failure(monkeypatch):
    from core.skills.belief_ops import AddBeliefSkill
    from core.skills.memory_ops import MemoryOpsSkill

    _patch_authority(monkeypatch)
    world_calls = []
    vector_calls = []

    class BrokenBeliefEngine:
        def add_belief(self, **_kwargs):
            error = RuntimeError("belief engine unavailable")
            raise error

    class WorldModel:
        def add_belief(self, **kwargs):
            world_calls.append(kwargs)

    class VectorMemory:
        async def store(self, **kwargs):
            vector_calls.append(kwargs)

    services = {
        "belief_revision_engine": BrokenBeliefEngine(),
        "epistemic_state": WorldModel(),
        "vector_memory_engine": VectorMemory(),
    }
    monkeypatch.setattr("core.container.ServiceContainer.get", lambda name, default=None: services.get(name, default))

    async def _memfs_failed(self, payload, context):
        return {"ok": False}

    monkeypatch.setattr(MemoryOpsSkill, "execute", _memfs_failed)

    result = await AddBeliefSkill().execute(
        {"source": "Bryan", "relation": "values", "target": "causal machinery"},
        {},
    )

    assert result["ok"] is True
    assert result["data"]["causal"] is True
    assert result["data"]["causal_layers"] == ["world_model", "vector_memory"]
    assert world_calls and world_calls[0]["predicate"] == "values"
    assert vector_calls and vector_calls[0]["source"] == "belief_ops"


@pytest.mark.asyncio
async def test_query_beliefs_continues_after_primary_query_failure(monkeypatch):
    from core.skills.belief_ops import QueryBeliefsSkill

    class BrokenBeliefEngine:
        @property
        def beliefs(self):
            error = RuntimeError("belief engine query unavailable")
            raise error

    class WorldModel:
        def get_relevant_beliefs(self, subject, n):
            assert subject == "Bryan"
            assert n == 5
            return [{"subject": "Bryan", "predicate": "values", "object": "rigor", "confidence": 0.92}]

    class VectorMemory:
        async def search(self, **kwargs):
            assert kwargs["query"] == "beliefs about Bryan"
            return [{"content": "Belief: Bryan values causal machinery"}]

    services = {
        "belief_revision_engine": BrokenBeliefEngine(),
        "epistemic_state": WorldModel(),
        "vector_memory_engine": VectorMemory(),
    }
    monkeypatch.setattr("core.container.ServiceContainer.get", lambda name, default=None: services.get(name, default))

    result = await QueryBeliefsSkill().execute({"subject": "Bryan", "limit": 5}, {})

    assert result["ok"] is True
    assert result["data"]["causal"] is True
    assert result["data"]["causal_sources"] == ["world_model", "vector_memory"]
    assert any("rigor" in belief for belief in result["data"]["beliefs"])
    assert any("causal machinery" in belief for belief in result["data"]["beliefs"])
