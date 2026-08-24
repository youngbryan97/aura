from types import SimpleNamespace

import pytest

from core.brain.cognitive_engine import CognitiveEngine
from core.brain.latent_bridge import compute_inference_params
from core.brain.types import ThinkingMode
from core.cognitive.spiking_active_inference import (
    BoundedWorkingMemoryQueueModel,
    MultiCompartmentSpikeResponseModel,
    SoftmaxCompetitionStabilityProbe,
    SoftmaxODEStabilityProbe,
    SpikingActiveInferenceAdvisor,
    get_spiking_active_inference_advisor,
)
from core.container import ServiceContainer
from core.state.aura_state import AuraState


def teardown_function(_function=None):
    ServiceContainer.clear()


def test_spiking_active_inference_flags_governed_tools_without_executing_them():
    advisor = SpikingActiveInferenceAdvisor()

    advice = advisor.advise(
        "Open Chrome, search for three articles, create a document, and export it as a PDF.",
        context={"desktop_cognitive_engine_required": True},
        origin="desktop",
    )

    assert advice.routing_bias["use_tool_gateway"] is True
    assert advice.features["tool_pressure"] >= 0.58
    assert advice.governance["advisory_only"] is True
    assert advice.governance["executes_tools"] is False
    assert advice.governance["authority_gateway_required_for_effects"] is True


def test_spiking_active_inference_repair_pressure_changes_general_tendency():
    advisor = SpikingActiveInferenceAdvisor()

    advice = advisor.advise(
        "Aura crashed with a memory spike, Cortex unavailable, timeout, and broken tool loop.",
        origin="desktop",
    )

    assert advice.routing_bias["repair_first"] is True
    assert advice.features["error_pressure"] >= 0.70
    assert advice.sampling_bias["temperature_delta"] < 0.0
    assert advice.sampling_bias["repetition_penalty_delta"] > 0.0


def test_spiking_advisor_consumes_unified_memory_pressure(monkeypatch):
    from core.utils import memory_monitor

    monkeypatch.setattr(
        memory_monitor,
        "get_memory_pressure_snapshot",
        lambda: SimpleNamespace(
            pressure_pct=91.0,
            process_rss_gb=42.0,
            process_rss_limit_gb=40.0,
            level="critical",
        ),
    )
    advisor = SpikingActiveInferenceAdvisor()

    advice = advisor.advise(
        "Explain what external tools you can use.",
        context={"desktop_cognitive_engine_required": True},
        origin="desktop",
    )

    assert advice.features["memory_pressure"] >= 0.86
    assert advice.routing_bias["reduce_load"] is True
    assert advice.sampling_bias["max_tokens_factor"] < 0.62
    assert advice.features["error_pressure"] >= 0.60


def test_spiking_advisor_consumes_affective_driver_status():
    class Affect:
        def get_status(self):
            return {
                "confused": 100,
                "frustration": 80,
                "curiosity": 90,
                "longing": 60,
            }

    ServiceContainer.clear()
    ServiceContainer.register_instance("affect_engine", Affect(), required=False)
    advisor = SpikingActiveInferenceAdvisor()

    advice = advisor.advise(
        "Continue the conversation carefully.",
        context={"desktop_cognitive_engine_required": True},
        origin="desktop",
    )

    assert advice.features["clarity"] < 0.55
    assert advice.features["novelty"] >= 0.25
    assert advice.features["social_pressure"] >= 0.25
    assert advice.routing_bias["metacognition_depth"] >= 0.70


def test_bounded_working_memory_queue_defers_background_overload():
    queue = BoundedWorkingMemoryQueueModel(arrival_rate=7.0, service_rate=10.0, max_queue=3.0)
    high_pressure = {
        "clarity": 0.12,
        "energy": 0.30,
        "urgency": 1.0,
        "novelty": 0.9,
        "tool_pressure": 1.0,
        "error_pressure": 0.8,
        "social_pressure": 0.0,
        "memory_pressure": 0.8,
    }

    first = queue.observe(high_pressure, is_background=False)
    second = queue.observe(high_pressure, is_background=True)

    assert first["admitted"] is True
    assert first["admission"] in {"accept", "compress_foreground"}
    assert second["admitted"] is False
    assert second["admission"] == "defer_background"
    assert second["queue_load"] >= 0.9
    assert second["overload_pressure"] > 0.0


def test_spiking_advisor_turns_queue_pressure_into_load_shedding():
    advisor = SpikingActiveInferenceAdvisor()
    objective = (
        "Urgent: open apps, search, browse, create files, export PDFs, run tools, "
        "fix a crash, remember the result, and explain every step now."
    )

    advice = None
    for _ in range(4):
        advice = advisor.advise(
            objective,
            context={"desktop_cognitive_engine_required": True},
            origin="desktop",
        )

    assert advice is not None
    assert advice.working_memory["queue_load"] >= 0.9
    assert advice.working_memory["admission"] in {"compress_foreground", "accept"}
    assert advice.routing_bias["reduce_load"] is True
    assert advice.sampling_bias["max_tokens_factor"] < 0.62
    assert any("working memory admission" in item for item in advice.rationale)


def test_softmax_competition_stability_probe_is_bounded_and_causal():
    stable = SoftmaxCompetitionStabilityProbe.analyze(
        [8.0, 0.0, -2.0],
        [0.98, 0.015, 0.005],
        temperature=0.6,
    )
    unstable = SoftmaxCompetitionStabilityProbe.analyze(
        [0.1, 0.09, 0.08],
        [0.34, 0.33, 0.33],
        temperature=0.6,
    )

    assert 0.0 <= stable["spectral_radius"] <= 5.0
    assert "ode_spectral_abscissa" in stable
    assert 0.0 <= stable["bifurcation_pressure"] <= 1.0
    assert 0.0 <= unstable["decision_instability"] <= 1.0
    assert unstable["decision_instability"] > stable["decision_instability"]
    assert unstable["winner_margin"] < stable["winner_margin"]


def test_softmax_ode_stability_probe_is_local_and_bounded():
    quiet = SoftmaxODEStabilityProbe.analyze(
        [8.0, 0.0, -2.0],
        [0.98, 0.015, 0.005],
        temperature=0.6,
        features={"clarity": 0.95, "urgency": 0.2, "novelty": 0.1},
    )
    pressured = SoftmaxODEStabilityProbe.analyze(
        [0.1, 0.09, 0.08],
        [0.34, 0.33, 0.33],
        temperature=0.6,
        features={
            "clarity": 0.2,
            "urgency": 1.0,
            "novelty": 0.9,
            "error_pressure": 0.8,
            "memory_pressure": 0.7,
        },
    )

    assert -5.0 <= quiet["ode_spectral_abscissa"] <= 5.0
    assert 0.0 <= quiet["fixed_point_residual"] <= 1.0
    assert 0.0 <= pressured["bifurcation_pressure"] <= 1.0
    assert pressured["bifurcation_pressure"] >= quiet["bifurcation_pressure"]


def test_spiking_advisor_uses_stability_probe_for_metacognition():
    advisor = SpikingActiveInferenceAdvisor()

    advice = advisor.advise(
        "Maybe either path could work; perhaps it is ambiguous and I am not sure which one.",
        origin="desktop",
    )

    assert advice.stability["decision_instability"] >= 0.0
    assert advice.routing_bias["metacognition_depth"] >= 0.45
    assert "stability" in advice.to_dict()


def test_multi_compartment_srm_stays_bounded_under_repeated_pressure():
    model = MultiCompartmentSpikeResponseModel()

    summary = {}
    for _ in range(200):
        summary = model.tick([1.0, 0.2, 0.8, 0.9, 0.7, 0.9, 0.2, 0.4], modulation=1.8)

    assert 0.0 <= summary["spike_rate"] <= 1.0
    assert 0.0 <= summary["plateau_rate"] <= 1.0
    assert 0.0 <= summary["weight_mean"] <= 2.0
    assert 0.0 <= summary["threshold_mean"] <= 2.0


def test_spiking_advisor_reregisters_after_container_reset():
    ServiceContainer.clear()
    advisor = get_spiking_active_inference_advisor()
    assert ServiceContainer.get("spiking_active_inference", default=None) is advisor

    ServiceContainer.clear()
    same_advisor = get_spiking_active_inference_advisor()

    assert same_advisor is advisor
    assert ServiceContainer.get("spiking_active_inference", default=None) is advisor


def test_cognitive_engine_records_spiking_active_inference_on_state():
    ServiceContainer.clear()
    engine = CognitiveEngine()
    state = AuraState.default()

    context = engine._apply_spiking_active_inference(
        state,
        "Can you open my notes app and create a timestamped journal entry?",
        "desktop",
        {"desktop_cognitive_engine_required": True},
        is_background=False,
    )

    advice = state.response_modifiers["spiking_active_inference"]
    assert advice["governance"]["advisory_only"] is True
    assert state.response_modifiers["tool_governance_pressure"] is True
    assert "spiking_active_inference" in context
    assert context["spiking_active_inference"]["advice_id"] == advice["advice_id"]


def test_cognitive_engine_closes_feedback_loop_for_neurodynamic_advice():
    ServiceContainer.clear()
    engine = CognitiveEngine()
    state = AuraState.default()
    context = engine._apply_spiking_active_inference(
        state,
        "I am confused; reason carefully before acting.",
        "desktop",
        {"desktop_cognitive_engine_required": True},
        is_background=False,
    )

    feedback = engine._learn_spiking_active_inference_outcome(
        context,
        outcome="assistant_response",
        reward=1.0,
    )

    assert feedback is not None
    assert feedback["outcome"] == "assistant_response"
    assert feedback["action"] == context["spiking_active_inference"]["action"]
    assert "prediction_error" in feedback


def test_runtime_capabilities_expose_neurodynamic_status():
    ServiceContainer.clear()
    advisor = SpikingActiveInferenceAdvisor()
    advisor.advise(
        "Open a tool, verify the result, and remember the lesson.",
        context={"desktop_cognitive_engine_required": True},
        origin="desktop",
    )
    ServiceContainer.register_instance("spiking_active_inference", advisor, required=False)

    from interface.routes.system import _collect_runtime_capabilities

    payload = _collect_runtime_capabilities(
        {"conversation_ready": True, "state": "ready", "desired_model": "cortex"}
    )

    status = payload["neurodynamic_advisor"]
    assert status["status"] == "active"
    assert status["advisory_only"] is True
    assert status["authority_gateway_required_for_effects"] is True
    assert status["features"]["tool_pressure"] > 0.0
    assert status["features"]["memory_pressure"] > 0.0
    assert "working_memory" in status
    assert "queue_load" in status["working_memory"]
    assert "stability" in status
    assert "decision_instability" in status["stability"]
    assert "bifurcation_pressure" in status["stability"]


@pytest.mark.parametrize(
    ("admitted", "expected_mode"),
    [(False, "resident_systems"), (True, "distinct_specialist")],
)
def test_runtime_capabilities_distinguish_configured_from_admitted_solver(
    monkeypatch,
    admitted,
    expected_mode,
):
    from core.brain import inference_gate
    from core.brain.llm import model_registry
    from interface.routes.system import _collect_runtime_capabilities

    monkeypatch.setattr(model_registry, "deep_solver_is_distinctly_configured", lambda: True)
    monkeypatch.setattr(model_registry, "get_deep_model_name", lambda: "Local-Specialist")
    monkeypatch.setattr(inference_gate, "local_deep_solver_enabled", lambda: admitted)

    payload = _collect_runtime_capabilities(
        {"conversation_ready": True, "state": "ready", "desired_model": "cortex"}
    )

    assert payload["solver_configured"] is True
    assert payload["solver_active"] is admitted
    assert payload["deep_reasoning_mode"] == expected_mode
    assert payload["solver_model"] == "Local-Specialist"


@pytest.mark.asyncio
async def test_desktop_quick_path_consumes_neurodynamic_advisory():
    ServiceContainer.clear()
    captured = {}

    class Router:
        async def think(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return "I can help with that through the governed desktop tool path."

    ServiceContainer.register_instance("llm_router", Router(), required=False)
    engine = CognitiveEngine()
    advice = {
        "action": "use_governed_tools",
        "uncertainty": 0.42,
        "routing_bias": {
            "use_tool_gateway": True,
            "ask_clarification": False,
            "reduce_load": False,
            "repair_first": False,
        },
        "sampling_bias": {"max_tokens_factor": 0.50},
    }

    thought = await engine._direct_desktop_quick_reply(
        "Open a document and write a short summary.",
        ThinkingMode.FAST,
        "desktop",
        {
            "desktop_quick_reply_contract": True,
            "desktop_cognitive_engine_required": True,
            "spiking_active_inference": advice,
            "max_tokens": 512,
        },
        timeout_s=20.0,
    )

    assert thought is not None
    assert thought.metadata["spiking_active_inference"] == advice
    grounding = "\n".join(message["content"] for message in captured["messages"])
    assert "Neurodynamic advisory" in grounding
    assert captured["kwargs"]["protected_foreground_lane"] is True
    assert captured["kwargs"]["allow_cloud_fallback"] is False
    # The advisory IS consumed (asserted above via metadata + prompt injection),
    # but a plain conversational quick reply floors at 512 tokens so replies do
    # not truncate mid-sentence even after the 0.50 advisory reduction
    # (512 * 0.50 = 256 -> floored to 512; fix 23d341e97).
    assert captured["kwargs"]["max_tokens"] == 512


def test_latent_bridge_sampling_consumes_spiking_active_inference():
    ServiceContainer.clear()

    class Advisor:
        def snapshot(self):
            return {
                "uncertainty": 0.80,
                "features": {"tool_pressure": 0.70, "error_pressure": 0.65},
                "routing_bias": {"reduce_load": True, "seek_information": True},
            }

    base = compute_inference_params(base_max_tokens=1000, base_temperature=0.70)
    ServiceContainer.register_instance("spiking_active_inference", Advisor(), required=False)
    steered = compute_inference_params(base_max_tokens=1000, base_temperature=0.70)

    assert steered.temperature < base.temperature
    assert steered.top_p < base.top_p
    assert steered.max_tokens < base.max_tokens
    assert steered.presence_penalty > base.presence_penalty
    assert any("active_uncert" in item for item in steered.rationale)


def test_latent_bridge_sampling_consumes_causal_valenced_workspace():
    ServiceContainer.clear()

    class Vector:
        values = {
            "organismal_coherence": 0.22,
            "verification_need": 0.86,
            "governance_pressure": 0.74,
            "metabolic_budget": 0.48,
            "sentience_candidate_strength": 0.04,
        }

        def value(self, name, default=0.0):
            return self.values.get(name, default)

    ServiceContainer.register_instance(
        "being_runtime",
        SimpleNamespace(_last_causal_self_vector=Vector()),
        required=False,
    )

    steered = compute_inference_params(base_max_tokens=1000, base_temperature=0.70)

    ServiceContainer.clear()
    base = compute_inference_params(base_max_tokens=1000, base_temperature=0.70)

    assert steered.temperature < base.temperature
    assert steered.top_p < base.top_p
    assert steered.max_tokens < base.max_tokens
    assert steered.repetition_penalty > base.repetition_penalty
    assert any("cvw=" in item and "verify=" in item for item in steered.rationale)
