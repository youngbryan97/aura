from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.consciousness.dreaming import DreamingProcess
from core.dream_processor import DreamProcessor


def test_dream_reflection_is_lightweight_and_bounded():
    reflection = DreamingProcess._compose_reflection(
        "\n".join(
            [
                "Context: alpha cluster | Action: explored avenue | Outcome: insight gained",
                "Context: alpha cluster | Action: explored avenue | Outcome: insight gained",
                "Context: beta lane | Action: stabilized state | Outcome: continuity preserved",
                "Context: gamma lane | Action: audited drift | Outcome: coherence restored",
                "Context: delta lane | Action: ignored overflow | Outcome: should be trimmed",
            ]
        )
    )

    assert "integrating" in reflection
    assert "alpha cluster" in reflection
    assert "beta lane" in reflection
    assert "delta lane" not in reflection


@pytest.mark.asyncio
async def test_dream_cycle_avoids_brain_think_on_event_loop(service_container):
    orchestrator = SimpleNamespace(_last_user_interaction_time=0)
    dreamer = DreamingProcess(orchestrator, interval=0.1)
    recorded_growth = []

    dreamer._identity = SimpleNamespace(
        record_evolution=lambda **kwargs: recorded_growth.append(kwargs)
    )
    dreamer._narrator = object()

    async def _recent_summary():
        return (
            "Context: alpha cluster | Action: explored avenue | Outcome: insight gained\n"
            "Context: beta lane | Action: stabilized state | Outcome: continuity preserved"
        )

    dreamer._get_recent_summary = _recent_summary

    class _VectorMemory:
        def __init__(self):
            self.brains = []

        async def consolidate(self, brain=None):
            self.brains.append(brain)
            return 0

    vector_memory = _VectorMemory()
    brain = SimpleNamespace(
        think=AsyncMock(side_effect=AssertionError("dream cycle should not invoke brain.think"))
    )

    service_container.register_instance("vector_memory_engine", vector_memory, required=False)
    service_container.register_instance("cognitive_engine", brain, required=False)

    await dreamer.dream()

    assert vector_memory.brains == [None]
    assert recorded_growth
    assert "alpha cluster" in recorded_growth[0]["reflection"]
    brain.think.assert_not_called()


@pytest.mark.asyncio
async def test_legacy_dream_processor_uses_episodic_api_without_llm(monkeypatch):
    episodes = [
        SimpleNamespace(
            to_retrieval_text=lambda idx=idx: f"Context: event {idx} | Action: reflected | Outcome: retained"
        )
        for idx in range(12)
    ]
    episodic = SimpleNamespace(recall_recent_async=AsyncMock(return_value=episodes))
    memory = SimpleNamespace(episodic=episodic, add_memory=AsyncMock(return_value=True))
    brain = SimpleNamespace(think=AsyncMock(side_effect=AssertionError("legacy dream should stay bounded by default")))
    processor = DreamProcessor(memory, brain)
    processor._contract_graph = AsyncMock()

    monkeypatch.delenv("AURA_DREAM_PROCESSOR_USE_LLM", raising=False)

    await processor.dream()

    episodic.recall_recent_async.assert_awaited_once()
    memory.add_memory.assert_awaited_once()
    processor._contract_graph.assert_awaited_once()
    brain.think.assert_not_called()


@pytest.mark.asyncio
async def test_legacy_dream_processor_contract_graph_accepts_thought_objects(service_container, monkeypatch):
    class _KG:
        def __init__(self):
            self.edges = []

        def upsert_relationship(self, e1, rel, e2, weight=1.0):
            self.edges.append((e1, rel, e2, weight))

    kg = _KG()
    service_container.register_instance("knowledge_graph", kg, required=False)
    brain = SimpleNamespace(
        think=AsyncMock(
            return_value=SimpleNamespace(
                content="email | requires | content-aware reading\nreddit | records | blocked-login outcome"
            )
        )
    )
    processor = DreamProcessor(SimpleNamespace(), brain)

    monkeypatch.setenv("AURA_DREAM_PROCESSOR_USE_LLM", "1")
    await processor._contract_graph("Reflection: email and reddit autonomy")

    assert ("email", "requires", "content-aware reading", 1.5) in kg.edges
    assert ("reddit", "records", "blocked-login outcome", 1.5) in kg.edges


@pytest.mark.asyncio
async def test_legacy_dream_processor_reports_blocked_graph_writes(service_container, monkeypatch):
    class _BlockedKG:
        def __init__(self):
            self.attempts = []

        def upsert_relationship(self, e1, rel, e2, weight=1.0):
            self.attempts.append((e1, rel, e2, weight))
            return False

    kg = _BlockedKG()
    service_container.register_instance("knowledge_graph", kg, required=False)
    brain = SimpleNamespace(think=AsyncMock())
    processor = DreamProcessor(SimpleNamespace(), brain)

    monkeypatch.delenv("AURA_DREAM_PROCESSOR_USE_LLM", raising=False)
    committed = await processor._contract_graph("email reddit live chat degraded")

    assert committed == 0
    assert kg.attempts


@pytest.mark.asyncio
async def test_self_optimizer_is_opt_in_for_live_runtime(tmp_path, monkeypatch):
    from core.adaptation.self_optimizer import SelfOptimizer

    dataset = tmp_path / "lora.jsonl"
    dataset.write_text("{}\n" * 6)
    optimizer = SelfOptimizer(
        base_model_path=str(tmp_path / "model"),
        dataset_path=str(dataset),
        adapter_path=str(tmp_path / "adapter" / "adapters.safetensors"),
    )

    monkeypatch.delenv("AURA_SELF_OPTIMIZER_ENABLED", raising=False)
    result = await optimizer.optimize()

    assert result == {"ok": False, "error": "self_optimizer_disabled_for_live_runtime"}
