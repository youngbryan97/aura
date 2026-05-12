################################################################################

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.agency_core import AgencyCore, AgencyState, EngagementMode, SovereignSwarm
from core.behavior_controller import integrate_behavior_control
from core.brain.llm.structured_llm import StructuredLLM
from core.cognitive_integration_layer import CognitiveIntegrationLayer
from core.container import ServiceContainer
from core.memory.memory_facade import MemoryFacade
from core.moral_reasoning import MoralReasoningEngine
from core.schemas import ShardResponse
from core.self_model import SelfModel


@pytest.fixture(autouse=True)
def cleanup_container():
    ServiceContainer.reset()
    yield
    ServiceContainer.reset()

@pytest.mark.asyncio
async def test_memory_facade_hardening():
    """Verify MemoryFacade Pydantic status and setup."""
    # Mock sub-systems
    mock_episodic = MagicMock()
    ServiceContainer.register_instance("episodic_memory", mock_episodic)
    
    facade = MemoryFacade()
    facade.setup()
    
    assert facade.episodic == mock_episodic
    
    status = facade.get_status()
    assert status["episodic"] is True
    assert status["semantic"] is False
    assert "last_commit" in status

@pytest.mark.asyncio
async def test_agency_core_pydantic_state():
    """Verify AgencyCore Pydantic state and sync."""
    orch = MagicMock()
    # Mock liquid_state properly to avoid Pydantic float warnings
    mock_ls = MagicMock()
    mock_ls.current.energy = 0.9
    mock_ls.current.curiosity = 0.5
    mock_ls.current.frustration = 0.0
    orch.liquid_state = mock_ls
    
    orch.personality_engine = MagicMock()
    orch.personality_engine.traits = {"extraversion": 0.8}
    
    agency = AgencyCore(orchestrator=orch)
    assert isinstance(agency.state, AgencyState)
    assert agency.state.last_user_interaction > 0.0
    
    # Test Sync
    agency._sync_from_orchestrator()
    assert abs(agency.state.initiative_energy - 0.9) < 0.001
    assert agency.state.social_hunger > 0.3
    
    # Test Telemetry
    status = agency.get_status()
    assert status["engagement_mode"] == EngagementMode.ATTENTIVE_IDLE.value
    assert "pathways_active" in status

@pytest.mark.asyncio
async def test_cognitive_integration_segments():
    """Verify CognitiveIntegrationLayer initialization and greeting fast-path."""
    orch = MagicMock()
    orch.cognitive_engine = MagicMock()
    
    # Register dependencies that initialize() resolves
    mock_kernel = AsyncMock()
    mock_monologue = AsyncMock()
    mock_language = AsyncMock()
    mock_synth = AsyncMock()
    
    # Setup mock_kernel.evaluate to return a CognitiveBrief
    from core.cognitive_kernel import CognitiveBrief
    mock_kernel.evaluate = AsyncMock(return_value=CognitiveBrief(
        key_points=["Hello."],
        conviction=0.5
    ))
    
    # Setup mock_monologue.think to return a ThoughtPacket
    from core.inner_monologue import ThoughtPacket
    mock_monologue.think = AsyncMock(return_value=ThoughtPacket(
        stance="Hello.",
        primary_points=["Hello."]
    ))
    
    # Setup mock_language.express to return a string
    mock_language.express = AsyncMock(return_value="Hello.")
    
    ServiceContainer.register_instance("cognitive_kernel", mock_kernel)
    ServiceContainer.register_instance("inner_monologue", mock_monologue)
    ServiceContainer.register_instance("language_center", mock_language)
    ServiceContainer.register_instance("memory_synthesizer", mock_synth)
    
    cognition = CognitiveIntegrationLayer(orchestrator=orch)
    await cognition.initialize()
    
    assert cognition.is_active is True
    assert cognition.kernel == mock_kernel
    
    # Test greeting fast-path (doesn't need LLM)
    response = await cognition.process_turn("hello")
    assert response == "Hello."


@pytest.mark.asyncio
async def test_cognitive_integration_threads_history_into_reasoning_pipeline(monkeypatch):
    orch = MagicMock()
    orch.cognitive_engine = MagicMock()

    from core.cognitive_kernel import CognitiveBrief
    from core.inner_monologue import ThoughtPacket

    mock_kernel = AsyncMock()
    mock_kernel.evaluate = AsyncMock(return_value=CognitiveBrief(key_points=["Depth."], conviction=0.7))
    mock_monologue = AsyncMock()
    mock_monologue.think = AsyncMock(return_value=ThoughtPacket(stance="Depth.", primary_points=["Depth."]))
    mock_language = AsyncMock()
    mock_language.express = AsyncMock(return_value="Depth.")

    ServiceContainer.register_instance("cognitive_kernel", mock_kernel)
    ServiceContainer.register_instance("inner_monologue", mock_monologue)
    ServiceContainer.register_instance("language_center", mock_language)

    monkeypatch.setattr("core.cognitive_integration_layer.get_reflex", lambda: MagicMock(process=lambda _msg: None))

    cognition = CognitiveIntegrationLayer(orchestrator=orch)
    await cognition.initialize()

    context = {
        "history": [
            {"role": "user", "content": "Earlier"},
            {"role": "assistant", "content": "Later"},
        ]
    }
    response = await cognition.process_turn("Let's go deeper.", context=context)

    assert response == "Depth."
    assert mock_kernel.evaluate.await_args.kwargs["history"] == context["history"]
    assert mock_monologue.think.await_args.kwargs["history"] == context["history"]
    assert mock_language.express.await_args.kwargs["history"] == context["history"]


@pytest.mark.asyncio
async def test_agency_goal_genesis_awaits_moral_reasoning(monkeypatch):
    orch = MagicMock()
    orch._current_thought_task = None

    agency = AgencyCore(orchestrator=orch)
    agency.state.curiosity_pressure = 1.0
    agency.state.engagement_mode = EngagementMode.ATTENTIVE_IDLE
    agency.state.last_goal_genesis_time = 0.0

    moral = MagicMock()
    moral.reason_about_action = AsyncMock(return_value={"is_morally_acceptable": True})
    monkeypatch.setattr("core.moral_reasoning.get_moral_reasoning", lambda: moral)

    result = await agency._pathway_goal_genesis(now=1200.0, idle_seconds=601.0)

    assert result is not None
    moral.reason_about_action.assert_awaited_once()


@pytest.mark.asyncio
async def test_behavior_controller_pre_action_awaits_moral_reasoning():
    captured = {}
    orchestrator = MagicMock()
    orchestrator.moral_reasoning.reason_about_action = AsyncMock(
        return_value={"is_morally_acceptable": True}
    )
    orchestrator.hooks.register = lambda name, callback: captured.setdefault(name, callback)

    integrate_behavior_control(orchestrator)

    assert await captured["pre_action"]("read_file", {"command": ""}) is True
    orchestrator.moral_reasoning.reason_about_action.assert_awaited_once()


def test_self_model_accepts_long_term_goal_without_optional_logger_module():
    model = SelfModel(id="self-model-test")

    model.add_long_term_goal({"text": "Learn continuity repair"}, source="unit_test")


@pytest.mark.asyncio
async def test_moral_reasoning_accepts_self_model_identity_fallback():
    ServiceContainer.register_instance(
        "identity",
        SelfModel(id="moral-self-model", beliefs={"stance": "protect continuity"}),
    )

    moral = MoralReasoningEngine()
    assessment = await moral.reason_about_action(
        {"type": "autonomous_goal", "description": "Learn continuity repair"},
        {"affected_selves": ["self", "user"]},
    )

    assert assessment["identity_context"]["beliefs"] == ["protect continuity"]


@pytest.mark.asyncio
async def test_structured_llm_keeps_ghost_example_for_json_prompts():
    router = MagicMock()
    router.generate_with_metadata = AsyncMock(
        return_value={
            "text": (
                '{"analysis":"careful","action_type":"conclusion",'
                '"tools":[],"tool_name":null,"tool_payload":null,'
                '"conclusion":"done"}'
            )
        }
    )
    ServiceContainer.register_instance("llm_router", router)

    result = await StructuredLLM(ShardResponse, max_retries=1).generate(
        "Return valid JSON for this shard."
    )

    assert result is not None
    sent_prompt = router.generate_with_metadata.await_args.args[0]
    assert "GHOST EXAMPLE (Follow this structure exactly):" in sent_prompt
    assert '"tools": []' in sent_prompt


@pytest.mark.asyncio
async def test_structured_llm_treats_background_deferral_as_non_failure():
    router = MagicMock()
    router.generate_with_metadata = AsyncMock(
        return_value={"text": "", "error": "background_deferred:cortex_startup_quiet"}
    )
    ServiceContainer.register_instance("llm_router", router)

    structured = StructuredLLM(ShardResponse, max_retries=3)
    result = await structured.generate("Return JSON.")

    assert result is None
    assert router.generate_with_metadata.await_count == 1
    assert structured.last_defer_reason == "background_deferred:cortex_startup_quiet"


@pytest.mark.asyncio
async def test_social_reflection_awaits_async_memory_search():
    identity = MagicMock()
    identity.state.kinship = {"Bryan": "trusted"}
    ServiceContainer.register_instance("identity_service", identity)

    memory = MagicMock()
    memory.search = AsyncMock(return_value=[{"text": "Recent continuity note"}])
    ServiceContainer.register_instance("memory_facade", memory)

    agency = AgencyCore(orchestrator=MagicMock())
    agency.swarm = None
    agency._last_social_reflection = 0.0

    result = await agency._pathway_social_reflection(now=9999.0, idle_seconds=1801.0)

    assert result is not None
    memory.search.assert_awaited_once()
    identity.add_insight.assert_called_once()


@pytest.mark.asyncio
async def test_swarm_background_deferral_skips_formatting_collapse(monkeypatch):
    class DeferredStructuredLLM:
        def __init__(self, *_args, **_kwargs):
            self.last_defer_reason = "background_deferred:cortex_startup_quiet"

        async def generate(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(
        "core.brain.llm.structured_llm.StructuredLLM",
        DeferredStructuredLLM,
    )

    orchestrator = MagicMock()
    orchestrator.cognitive_engine = object()
    swarm = SovereignSwarm(orchestrator)
    swarm._active_self_repair_formatting = AsyncMock()

    await swarm._shard_wrapper("deferred goal", "context", shard_id="shard-test")

    swarm._active_self_repair_formatting.assert_not_called()

if __name__ == "__main__":
    pytest.main([__file__])


##
