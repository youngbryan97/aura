from types import SimpleNamespace

import pytest

from core.agency_core import AgencyCore, AgencyState, EngagementMode, SovereignSwarm
from core.autonomy.behavior_controller import integrate_behavior_control
from core.brain.llm.structured_llm import StructuredLLM
from core.cognition.cognitive_integration_layer import CognitiveIntegrationLayer
from core.container import ServiceContainer
from core.memory.memory_facade import MemoryFacade
from core.morality.moral_reasoning import MoralReasoningEngine
from core.schemas import ShardResponse
from core.self_model import SelfModel


class EpisodicMemoryProbe:
    pass


class LiquidCurrentProbe:
    energy = 0.9
    curiosity = 0.5
    frustration = 0.0


class LiquidStateProbe:
    current = LiquidCurrentProbe()


class PersonalityEngineProbe:
    traits = {"extraversion": 0.8}


class OrchestratorProbe:
    def __init__(self):
        self.liquid_state = LiquidStateProbe()
        self.personality_engine = PersonalityEngineProbe()
        self._current_thought_task = None
        self.hooks = HookRegistryProbe()
        self.moral_reasoning = MoralReasoningProbe({"is_morally_acceptable": True})
        self.cognitive_engine = object()


class KernelProbe:
    def __init__(self, brief):
        self.brief = brief
        self.started = False
        self.evaluate_calls = []

    async def start(self):
        self.started = True

    async def evaluate(self, message, **kwargs):
        self.evaluate_calls.append({"message": message, "kwargs": kwargs})
        return self.brief


class BeliefRevisionProbe:
    def __init__(self, events=None):
        self.events = events
        self.started = False

    async def start(self):
        self.started = True
        if self.events is not None:
            self.events.append("belief_started")


class MonologueProbe:
    def __init__(self, packet):
        self.packet = packet
        self.started = False
        self.think_calls = []

    async def start(self):
        self.started = True

    async def think(self, message, brief, **kwargs):
        self.think_calls.append({"message": message, "brief": brief, "kwargs": kwargs})
        return self.packet


class LanguageCenterProbe:
    def __init__(self, response):
        self.response = response
        self.started = False
        self.express_calls = []

    async def start(self):
        self.started = True

    async def express(self, packet, message, **kwargs):
        self.express_calls.append({"packet": packet, "message": message, "kwargs": kwargs})
        return self.response


class ReflexProbe:
    def __init__(self, response=None):
        self.response = response
        self.messages = []

    def process(self, message):
        self.messages.append(message)
        return self.response


class MoralReasoningProbe:
    def __init__(self, assessment):
        self.assessment = assessment
        self.calls = []

    async def reason_about_action(self, action, context):
        self.calls.append({"action": action, "context": context})
        return dict(self.assessment)


class HookRegistryProbe:
    def __init__(self):
        self.callbacks = {}

    def register(self, name, callback):
        self.callbacks[name] = callback


class RouterMetadataProbe:
    def __init__(self, metadata):
        self.metadata = metadata
        self.calls = []

    async def generate_with_metadata(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return dict(self.metadata)


class IdentityStateProbe:
    kinship = {"Bryan": "trusted"}


class IdentityServiceProbe:
    def __init__(self):
        self.state = IdentityStateProbe()
        self.insights = []

    def add_insight(self, insight, source=None):
        self.insights.append({"insight": insight, "source": source})


class MemorySearchProbe:
    def __init__(self, results):
        self.results = results
        self.calls = []

    async def search(self, query, limit=5):
        self.calls.append({"query": query, "limit": limit})
        return list(self.results)


class FormattingRepairTrap:
    def __init__(self):
        self.called = False

    async def __call__(self, *_args, **_kwargs):
        self.called = True


@pytest.fixture(autouse=True)
def cleanup_container():
    ServiceContainer.reset()
    ServiceContainer.register_instance("belief_revision_engine", BeliefRevisionProbe())
    yield
    ServiceContainer.reset()

@pytest.mark.asyncio
async def test_memory_facade_hardening():
    """Verify MemoryFacade Pydantic status and setup."""
    episodic = EpisodicMemoryProbe()
    ServiceContainer.register_instance("episodic_memory", episodic)
    
    facade = MemoryFacade()
    facade.setup()
    
    assert facade.episodic == episodic
    
    status = facade.get_status()
    assert status["episodic"] is True
    assert status["semantic"] is False
    assert "last_commit" in status

@pytest.mark.asyncio
async def test_agency_core_pydantic_state():
    """Verify AgencyCore Pydantic state and sync."""
    orch = OrchestratorProbe()
    
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
    from core.cognition.cognitive_kernel import CognitiveBrief
    from core.introspection.inner_monologue import ThoughtPacket

    orch = OrchestratorProbe()
    kernel = KernelProbe(CognitiveBrief(key_points=["Hello."], conviction=0.5))
    monologue = MonologueProbe(ThoughtPacket(stance="Hello.", primary_points=["Hello."]))
    language = LanguageCenterProbe("Hello.")

    ServiceContainer.register_instance("cognitive_kernel", kernel)
    ServiceContainer.register_instance("inner_monologue", monologue)
    ServiceContainer.register_instance("language_center", language)
    
    cognition = CognitiveIntegrationLayer(orchestrator=orch)
    await cognition.initialize()
    
    assert cognition.is_active is True
    assert cognition.kernel == kernel
    
    response = await cognition.process_turn("hello")
    assert response.startswith("Hello")


@pytest.mark.asyncio
async def test_cognitive_integration_starts_belief_dependency_before_kernel():
    events = []

    class OrderedKernel(KernelProbe):
        async def start(self):
            events.append("kernel_started")
            await super().start()

    belief = BeliefRevisionProbe(events)
    kernel = OrderedKernel(SimpleNamespace())
    monologue = MonologueProbe(SimpleNamespace())
    language = LanguageCenterProbe("ready")
    ServiceContainer.register_instance("belief_revision_engine", belief)
    ServiceContainer.register_instance("cognitive_kernel", kernel)
    ServiceContainer.register_instance("inner_monologue", monologue)
    ServiceContainer.register_instance("language_center", language)

    cognition = CognitiveIntegrationLayer(orchestrator=OrchestratorProbe())
    await cognition.initialize()

    assert events[:2] == ["belief_started", "kernel_started"]
    assert ServiceContainer.get("belief_revision_engine") is belief


@pytest.mark.asyncio
async def test_cognitive_integration_threads_history_into_reasoning_pipeline(monkeypatch):
    orch = OrchestratorProbe()

    from core.cognition.cognitive_kernel import CognitiveBrief
    from core.introspection.inner_monologue import ThoughtPacket

    kernel = KernelProbe(CognitiveBrief(key_points=["Depth."], conviction=0.7))
    monologue = MonologueProbe(ThoughtPacket(stance="Depth.", primary_points=["Depth."]))
    language = LanguageCenterProbe("Depth.")

    ServiceContainer.register_instance("cognitive_kernel", kernel)
    ServiceContainer.register_instance("inner_monologue", monologue)
    ServiceContainer.register_instance("language_center", language)

    monkeypatch.setattr("core.cognition.cognitive_integration_layer.get_reflex", lambda: ReflexProbe())

    cognition = CognitiveIntegrationLayer(orchestrator=orch)
    await cognition.initialize()

    context = {
        "origin": "curriculum",
        "history": [
            {"role": "user", "content": "Earlier"},
            {"role": "assistant", "content": "Later"},
        ]
    }
    response = await cognition.process_turn("Let's go deeper.", context=context)

    assert response.startswith("Depth")
    assert kernel.evaluate_calls[-1]["kwargs"]["history"] == context["history"]
    assert monologue.think_calls[-1]["kwargs"]["history"] == context["history"]
    assert language.express_calls[-1]["kwargs"]["history"] == context["history"]
    assert language.express_calls[-1]["kwargs"]["origin"] == "curriculum"


@pytest.mark.asyncio
async def test_agency_goal_genesis_awaits_moral_reasoning(monkeypatch):
    orch = OrchestratorProbe()

    agency = AgencyCore(orchestrator=orch)
    agency.state.curiosity_pressure = 1.0
    agency.state.engagement_mode = EngagementMode.ATTENTIVE_IDLE
    agency.state.last_goal_genesis_time = 0.0

    moral = MoralReasoningProbe({"is_morally_acceptable": True})
    monkeypatch.setattr("core.morality.moral_reasoning.get_moral_reasoning", lambda: moral)

    result = await agency._pathway_goal_genesis(now=1200.0, idle_seconds=601.0)

    assert result is not None
    assert len(moral.calls) == 1


@pytest.mark.asyncio
async def test_behavior_controller_pre_action_awaits_moral_reasoning():
    orchestrator = OrchestratorProbe()

    integrate_behavior_control(orchestrator)

    assert await orchestrator.hooks.callbacks["pre_action"]("read_file", {"command": ""}) is True
    assert len(orchestrator.moral_reasoning.calls) == 1


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


@pytest.fixture
def router_reachable(monkeypatch):
    """Let a structured generation actually reach the router.

    StructuredLLM defers before its first router call whenever the background
    policy says the runtime is busy, and ``proof_run_active()`` reports True for
    any of AURA_PROOF_RUN, AURA_AGI_MAX_TASKS or AURA_TESTING. AURA_TESTING is
    set by most of this repo's tooling, so the two tests below — which assert
    what StructuredLLM SENDS to the router — were unreachable in exactly the
    environment they run in, and failed on a deferral rather than on anything
    they were written to check.

    A test that asserts a call happens has to establish the preconditions for
    that call rather than inherit them from whoever exported what.
    """

    for name in ("AURA_PROOF_RUN", "AURA_AGI_MAX_TASKS", "AURA_TESTING"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.asyncio
async def test_structured_llm_keeps_ghost_example_for_json_prompts(router_reachable):
    router = RouterMetadataProbe(
        {
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
    sent_prompt = router.calls[0]["args"][0]
    assert "GHOST EXAMPLE (Follow this structure exactly):" in sent_prompt
    assert '"tools": []' in sent_prompt


@pytest.mark.asyncio
async def test_structured_llm_treats_background_deferral_as_non_failure(router_reachable):
    router = RouterMetadataProbe({"text": "", "error": "background_deferred:cortex_startup_quiet"})
    ServiceContainer.register_instance("llm_router", router)

    structured = StructuredLLM(ShardResponse, max_retries=3)
    result = await structured.generate("Return JSON.")

    assert result is None
    assert len(router.calls) == 1
    assert structured.last_defer_reason == "background_deferred:cortex_startup_quiet"


@pytest.mark.asyncio
async def test_social_reflection_awaits_async_memory_search():
    identity = IdentityServiceProbe()
    ServiceContainer.register_instance("identity_service", identity)

    memory = MemorySearchProbe([{"text": "Recent continuity note"}])
    ServiceContainer.register_instance("memory_facade", memory)

    agency = AgencyCore(orchestrator=OrchestratorProbe())
    agency.swarm = None
    agency._last_social_reflection = 0.0

    result = await agency._pathway_social_reflection(now=9999.0, idle_seconds=1801.0)

    assert result is not None
    assert memory.calls == [{"query": "Bryan user interaction highlights", "limit": 5}]

    # Reflecting is not believing, and a template is not a reflection: the
    # sentence is filled in by code, so it never enters durable identity, not
    # even for the winning proposal. On 18 October all nineteen such "insights"
    # in her identity were this template with a record pasted into it.
    assert identity.insights == []
    effects = dict(result["_deferred_effects"])
    await effects["social_reflection.commit"]()
    assert identity.insights == []


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

    orchestrator = OrchestratorProbe()
    swarm = SovereignSwarm(orchestrator)
    formatting_repair = FormattingRepairTrap()
    swarm._active_self_repair_formatting = formatting_repair

    await swarm._shard_wrapper("deferred goal", "context", shard_id="shard-test")
    assert formatting_repair.called is False

if __name__ == "__main__":
    pytest.main([__file__])


##
