import asyncio
from contextvars import ContextVar
from types import SimpleNamespace

import pytest

from core.brain.generation_provenance import generation_metadata_of
from core.phases.response_generation import (
    ResponseGenerationPhase,
    _append_only_continuation_pending,
)
from core.state.aura_state import AuraState, CognitiveMode
from tests.support.amplifier_doubles import amplified_answer


class _Container:
    def __init__(self, services):
        self.services = services

    def get(self, name, default=None):
        return self.services.get(name, default)


class _Router:
    def __init__(self):
        self.calls = []

    async def think(self, **kwargs):
        self.calls.append(kwargs)
        return (
            "Thermal-safe response: I am keeping the architectural audit grounded, "
            "reducing the local load, and preserving the conversation thread."
        )

    def get_last_generation_metadata(self):
        return {
            "surface_control_receipt": {
                "enabled": True,
                "live_mind_controls_bound": True,
                "clean_user_surface_contract": True,
                "surface_validation_prompt_present": True,
                "surface_alpha_applied": 0.30,
                "surface_alpha_applied_ok": True,
                "recurrent_runtime_loops_applied": 2,
                "recurrent_runtime_loops_applied_ok": True,
                "surface_quality_gate_enabled": True,
                "surface_quality_gate_passed": True,
                "surface_quality_gate_attempts": 1,
                "surface_quality_gate_reasons": [],
                "applied": True,
            }
        }


class _MemoryAckRouter(_Router):
    async def think(self, **kwargs):
        self.calls.append(kwargs)
        return "I’ll remember that the blue lantern is under the desk for later in this conversation."


class _AttributedReceiptRouter(_Router):
    async def think(self, **kwargs):
        from core.brain.generation_provenance import attributed_text

        self.calls.append(kwargs)
        return attributed_text(
            "The retained draft is complete enough to continue from its exact state.",
            {
                "surface_control_receipt": {
                    **_Router.get_last_generation_metadata(self)[
                        "surface_control_receipt"
                    ],
                    "semantic_completion_contract": True,
                    "semantic_completion_satisfied": False,
                    "semantic_completion_incomplete": True,
                    "generation_stop_reason": "deadline_exceeded",
                    "continuation_resume_available": True,
                    "continuation_resume_handle": "a" * 32,
                }
            },
        )

    def get_last_generation_metadata(self):
        # A parent task cannot recover a child task's ContextVar snapshot. The
        # exact text still carries the receipt that produced it.
        return {}


class _StaleSinkFreshSnapshotRouter(_Router):
    """Reproduce a receipt amended after its first transport publication."""

    def __init__(self):
        super().__init__()
        self._fresh_metadata = {}

    async def think(self, **kwargs):
        self.calls.append(kwargs)
        stale = _Router.get_last_generation_metadata(self)
        sink = kwargs.get("_generation_metadata_sink")
        if isinstance(sink, dict):
            sink.update(stale)
        receipt = {
            **stale["surface_control_receipt"],
            "semantic_completion_contract": True,
            "semantic_completion_satisfied": False,
            "semantic_completion_incomplete": True,
            "surface_quality_gate_passed": False,
            "surface_quality_gate_reasons": ["truncated_tail"],
            "generation_stop_reason": "max_tokens",
            "continuation_resume_available": True,
            "continuation_resume_handle": "c" * 32,
        }
        self._fresh_metadata = {
            **stale,
            "surface_control_receipt": receipt,
            "post_generation_completion_evidence": ["truncated_tail"],
        }
        return "The evidence establishes the current release as"

    def get_last_generation_metadata(self):
        return dict(self._fresh_metadata)


def test_parent_completion_evidence_overrides_a_stale_positive_worker_receipt():
    assert _append_only_continuation_pending(
        {
            "surface_control_receipt": {
                "semantic_completion_contract": True,
                "semantic_completion_satisfied": True,
                "semantic_completion_incomplete": False,
            },
            "post_generation_completion_evidence": ["truncated_tail"],
        },
        clean_user_surface_contract=True,
    )


class _SearchCapability:
    def __init__(self):
        self.calls = []

    def resolve_skill_name(self, name):
        return str(name)

    async def execute(self, skill_name, params, context=None):
        self.calls.append((skill_name, dict(params), dict(context or {})))
        return {
            "ok": True,
            "query": params.get("query"),
            "answer": "NASA describes Europa as an icy moon of Jupiter with a subsurface ocean.",
            "results": [
                {
                    "title": "Europa: Jupiter's Ocean World",
                    "url": "https://science.nasa.gov/jupiter/moons/europa/",
                    "snippet": "Europa is one of Jupiter's moons and is a target in the search for habitable worlds.",
                }
            ],
            "source": "https://science.nasa.gov/jupiter/moons/europa/",
        }


class _EvidenceRouter(_Router):
    async def think(self, **kwargs):
        self.calls.append(kwargs)
        return (
            "I searched it live. Source title: Europa: Jupiter's Ocean World. "
            "NASA describes Europa as an icy moon of Jupiter with evidence for a subsurface ocean."
        )


class _FalseSearchInabilityRouter(_Router):
    async def think(self, **kwargs):
        self.calls.append(kwargs)
        return (
            "I can't execute web searches directly. But I know NASA has a Europa page."
        )


class _TimeoutSearchRouter(_Router):
    async def think(self, **kwargs):
        self.calls.append(kwargs)
        raise TimeoutError()


class _BlankSearchRouter(_Router):
    async def think(self, **kwargs):
        self.calls.append(kwargs)
        return ""


class _ConcurrentAmplifierRouter(_Router):
    def __init__(self):
        super().__init__()
        self.completion_order = []
        self._metadata = ContextVar(
            "test_amplifier_generation_metadata",
            default=None,
        )

    async def think(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][-1]["content"]
        candidate_id = int(prompt.rsplit("-", 1)[-1])
        await asyncio.sleep(0.03 if candidate_id == 0 else 0.005)
        self._metadata.set(
            {
                "candidate_id": candidate_id,
                "surface_control_receipt": {"applied": True},
            }
        )
        self.completion_order.append(candidate_id)
        return f"candidate answer {candidate_id}"

    def get_last_generation_metadata(self):
        return dict(self._metadata.get() or {})


class _LatentService:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def deep_reason(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class _TimedOutLatentService(_LatentService):
    async def deep_reason(self, **kwargs):
        self.calls.append(kwargs)
        raise TimeoutError("resident latent deadline")


class _AcquiringLatentService(_LatentService):
    async def deep_reason(self, **kwargs):
        raise AssertionError("live path bypassed the acquisition wrapper")

    async def deep_reason_with_acquisition(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def _live_latent_receipt(text: str, objective: str):
    from core.brain.llm.latent_cortex.output_quality import evaluate_latent_output

    quality = evaluate_latent_output(
        text,
        generated_tokens=96,
        termination="eos",
        objective=objective,
    )
    assert quality["passed"] is True, quality["reasons"]
    return {
        "episode_id": "episode-live-32b",
        "checkpoint_fingerprint": "a" * 64,
        "checkpoint_fingerprint_method": "sha256",
        "checkpoint_file_count": 32,
        "worker_boot_id": "b" * 32,
        "worker_pid": 4200,
        "worker_model_path": "/models/Aura-32B",
        "worker_model_parameter_count": 32_000_000_000,
        "worker_model_stored_parameter_element_count": 5_000_000_000,
        "worker_model_parameter_count_basis": "architecture_config_logical",
        "worker_source_sha256": "c" * 64,
        "worker_affective_steering_active": True,
        "worker_affective_steering_alpha": 0.30,
        "episode_affective_steering_applied": True,
        "episode_affective_steering_alpha": 0.30,
        "request_payload_sha256": "d" * 64,
        "input_tokens_sha256": "e" * 64,
        "input_token_count": 128,
        "steps_taken": 5,
        "decode_requested_tokens": 512,
        "decode_generated_tokens": 96,
        "decode_termination": "eos",
        "decode_temperature": 0.61,
        "decode_top_p": 0.87,
        "output_quality": quality,
        "runtime_identity": {
            "identity_bound": True,
            "launch_mode": "signed_app",
            "installed_app_required": True,
            "installed_app_verified": True,
            "source_verified": True,
            "source_commit": "f" * 40,
            "workspace_state_sha256": "1" * 64,
            "shell_assets_sha256": "2" * 64,
        },
    }


def _latent_context(objective):
    return {
        "desktop_cognitive_engine_required": True,
        "cognitive_engine_required": True,
        "visible_user_message": objective,
        "compact_desktop_chat_contract": False,
        "prompt_shape": {
            "question_parts": 2,
            "prefers_extended_answer": True,
            "requires_single_reply_coverage": True,
        },
        "max_tokens": 512,
        "live_mind_controls_bound": True,
        "live_mind_generation_controls": {
            "temperature": 0.61,
            "top_p": 0.87,
            "clean_user_surface_recurrent_loops": 2,
            "clean_user_surface_steering_alpha": 0.30,
        },
        "live_mind_snapshot_ready": True,
        "live_mind_required_subsystems_ok": True,
    }


@pytest.mark.asyncio
async def test_depth_worthy_live_turn_uses_the_acquisition_wrapper(monkeypatch):
    objective = (
        "Compare both failure modes, explain the causal tradeoff, and give one "
        "coherent implementation decision with its verification plan."
    )
    state = AuraState()
    state.cognition.current_objective = objective
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_mode = CognitiveMode.DELIBERATE
    answer = (
        "The first failure mode loses identity evidence, while the second duplicates "
        "generation after admission. I recommend the single-owner boundary because it "
        "prevents a late result from racing the proof ledger and makes every published "
        "answer attributable to one process; verify it with fault injection by cancelling "
        "an in-flight owner and asserting no successor publishes its stale result; force "
        "a timeout and assert the lease is fenced; then restart the worker and confirm "
        "exactly one new owner, one result, and one cleanup receipt before serving traffic."
    )
    latent = _AcquiringLatentService(
        {
            "ok": True,
            "text": answer,
            "receipt": _live_latent_receipt(answer, objective),
        }
    )
    phase = ResponseGenerationPhase(
        _Container({"llm_router": _Router(), "latent_cortex": latent})
    )
    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [{"role": "user", "content": objective}],
    )

    result = await phase.execute(state, context=_latent_context(objective))

    assert result.response_modifiers["latent_cortex_succeeded"] is True
    assert len(latent.calls) == 1
    assert latent.calls[0]["tenant_id"] == "local"
    assert latent.calls[0]["user_id"] == "owner"
    assert latent.calls[0]["session_id"] == "local"


@pytest.mark.asyncio
async def test_depth_worthy_desktop_turn_uses_one_latent_generation(monkeypatch):
    objective = (
        "Compare both failure modes, explain the causal tradeoff, and give one "
        "coherent implementation decision with its verification plan."
    )
    state = AuraState()
    state.cognition.current_objective = objective
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_mode = CognitiveMode.DELIBERATE
    router = _Router()
    latent_answer = (
        "The first failure mode loses identity evidence, while the second duplicates "
        "generation after admission. I recommend the single-owner boundary because it "
        "prevents a late result from racing the proof ledger and makes every published "
        "answer attributable to one process; verify it with fault injection by cancelling "
        "an in-flight owner and asserting no successor publishes its stale result; force "
        "a timeout and assert the lease is fenced; then restart the worker and confirm "
        "exactly one new owner, one result, and one cleanup receipt before serving traffic."
    )
    latent = _LatentService(
        {
            "ok": True,
            "text": latent_answer,
            "receipt": _live_latent_receipt(latent_answer, objective),
        }
    )
    phase = ResponseGenerationPhase(
        _Container({"llm_router": router, "latent_cortex": latent})
    )

    class _SplitVoice:
        def compile_profile(self, **_kwargs):
            return SimpleNamespace(
                word_budget=512,
                tone_override=None,
                multi_message=True,
                followup_probability=0.0,
            )

        def shape_response(self, text):
            first, rest = text.split(". ", 1)
            return [first + ".", rest]

        def decide_followup(self, **_kwargs):
            return SimpleNamespace(should_followup=False)

    monkeypatch.setattr(
        "core.voice.substrate_voice_engine.get_substrate_voice_engine",
        lambda: _SplitVoice(),
    )
    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [
            {"role": "system", "content": "full live context"},
            {"role": "user", "content": objective},
        ],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )

    result = await phase.execute(state, context=_latent_context(objective))

    assert len(latent.calls) == 1
    assert router.calls == []
    assert latent.calls[0]["messages"][-1]["content"] == objective
    assert latent.calls[0]["config_overrides"] == {
        "decode_max_tokens": 1024,
        "decode_temperature": 0.61,
        "decode_top_p": 0.87,
    }
    assert latent.calls[0]["runtime_controls"] == {
        "clean_user_surface_recurrent_loops": 2,
        "clean_user_surface_steering_alpha": 0.30,
    }
    assert result.response_modifiers["latent_cortex_succeeded"] is True
    assert result.response_modifiers["latent_cortex_fallback_used"] is False
    assert result.response_modifiers["latent_cortex_identity_bound"] is True
    assert result.response_modifiers["live_mind_controls_worker_applied"] is True
    delivered = result.cognition.working_memory[-1]["content"]
    assert "restart the worker" in delivered
    assert "one cleanup receipt before serving traffic." in delivered
    assert "queued_messages" not in result.response_modifiers


@pytest.mark.asyncio
async def test_final_latent_surface_is_regraded_and_cannot_start_second_owner(
    monkeypatch,
):
    objective = (
        "Compare early ownership with late deduplication, choose the stronger design, "
        "and explain how to verify cancellation and timeout faults."
    )
    raw_answer = (
        "Early ownership is stronger, whereas late deduplication permits competing "
        "generations to race. I recommend the early boundary because it keeps one "
        "publisher and one proof ledger entry. Verify it with fault injection: cancel "
        "the active owner and assert no stale publication, then force a timeout and "
        "check that the fenced successor publishes exactly one clean result."
    )
    malformed_final = (
        "<request> Compare early ownership with late deduplication, choose the stronger "
        "design, and explain how to verify cancellation and timeout faults. </request> "
        "Both designs process work."
    )
    state = AuraState()
    state.cognition.current_objective = objective
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_mode = CognitiveMode.DELIBERATE
    router = _Router()
    latent = _LatentService(
        {
            "ok": True,
            "text": raw_answer,
            "receipt": _live_latent_receipt(raw_answer, objective),
        }
    )
    phase = ResponseGenerationPhase(
        _Container({"llm_router": router, "latent_cortex": latent})
    )
    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [{"role": "user", "content": objective}],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )
    monkeypatch.setattr(phase, "_clean_response", lambda *_args, **_kwargs: malformed_final)

    result = await phase.execute(state, context=_latent_context(objective))

    assert result is not None
    assert len(latent.calls) == 1
    assert router.calls == []
    assert result.response_modifiers["latent_cortex_succeeded"] is False
    assert result.response_modifiers["model_retry_suppressed"] is True
    assert result.response_modifiers["response_path"] == (
        "cognitive_engine_latent_owner_exhausted"
    )
    quality = result.response_modifiers["latent_cortex_final_output_quality"]
    assert quality["passed"] is False
    assert "prompt_echo_contamination" in quality["reasons"]
    assert "protocol_artifact_leakage" in quality["reasons"]


@pytest.mark.asyncio
async def test_selected_latent_refusal_falls_back_to_exactly_one_generation(monkeypatch):
    objective = "Analyze both branches and recommend the safer architecture."
    state = AuraState()
    state.cognition.current_objective = objective
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_mode = CognitiveMode.DELIBERATE
    router = _Router()
    latent = _LatentService({"ok": False, "reason": "worker_not_ready"})
    phase = ResponseGenerationPhase(
        _Container({"llm_router": router, "latent_cortex": latent})
    )
    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [{"role": "user", "content": objective}],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )

    result = await phase.execute(state, context=_latent_context(objective))

    assert len(latent.calls) == 1
    assert len(router.calls) == 1
    assert result.response_modifiers["latent_cortex_succeeded"] is False
    assert result.response_modifiers["latent_cortex_fallback_used"] is True
    assert result.response_modifiers["latent_cortex_failure_reason"] == "worker_not_ready"


@pytest.mark.asyncio
async def test_selected_latent_refusal_uses_one_ordinary_fallback_without_second_amplifier(
    monkeypatch,
):
    objective = "Analyze both branches and recommend the safer architecture."
    state = AuraState()
    state.cognition.current_objective = objective
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_mode = CognitiveMode.DELIBERATE
    router = _Router()
    latent = _LatentService({"ok": False, "reason": "worker_not_ready"})
    phase = ResponseGenerationPhase(
        _Container({"llm_router": router, "latent_cortex": latent})
    )
    amplifier_calls = []

    async def capture_amplifier(**kwargs):
        amplifier_calls.append(dict(kwargs))
        return kwargs["draft"]

    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [{"role": "user", "content": objective}],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )
    monkeypatch.setattr(phase, "_maybe_amplify_response", capture_amplifier)

    result = await phase.execute(state, context=_latent_context(objective))

    assert result is not None
    assert len(latent.calls) == 1
    assert len(router.calls) == 1
    assert amplifier_calls == []
    assert result.response_modifiers["latent_cortex_selected"] is True
    assert result.response_modifiers["latent_cortex_succeeded"] is False


@pytest.mark.asyncio
async def test_successful_latent_answer_keeps_single_owner_and_skips_amplifier(monkeypatch):
    objective = "Explain why one component should own response publication."
    answer = (
        "The ownership-first branch is safer because one component owns admission, "
        "generation, and publication. It prevents a competing writer from publishing "
        "after cancellation. Verify that by cancelling the owner during generation "
        "and asserting that no successor response reaches the visible turn."
    )
    state = AuraState()
    state.cognition.current_objective = objective
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_mode = CognitiveMode.DELIBERATE
    router = _Router()
    latent = _LatentService(
        {"ok": True, "text": answer, "receipt": _live_latent_receipt(answer, objective)}
    )
    phase = ResponseGenerationPhase(
        _Container({"llm_router": router, "latent_cortex": latent})
    )

    async def amplifier_must_not_run(**_kwargs):
        raise AssertionError("successful latent answer opened a second model owner")

    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [{"role": "user", "content": objective}],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )
    monkeypatch.setattr(phase, "_maybe_amplify_response", amplifier_must_not_run)

    result = await phase.execute(state, context=_latent_context(objective))

    assert result is not None
    assert len(latent.calls) == 1
    assert router.calls == []
    assert result.response_modifiers["latent_cortex_succeeded"] is True


@pytest.mark.asyncio
async def test_materialized_latent_incumbent_is_served_without_second_generation(
    monkeypatch,
):
    objective = "Compare both architectures and recommend the safer design."
    incumbent = (
        "The ownership-first architecture is safer than late deduplication because "
        "one component owns the turn from admission through publication. Late "
        "deduplication permits two generators to consume compute and race before the "
        "publisher notices the conflict, whereas early ownership prevents the second "
        "writer from starting. I recommend ownership-first because it preserves a "
        "single causal receipt and makes cancellation authoritative. Verify it by "
        "cancelling the active owner and asserting that no stale successor can publish. "
        "Then force a deadline during generation and confirm that the materialized "
        "incumbent remains the only visible answer, with exactly one generation in the "
        "turn receipt and no second resident decode."
    )
    state = AuraState()
    state.cognition.current_objective = objective
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_mode = CognitiveMode.DELIBERATE
    router = _Router()
    receipt = _live_latent_receipt(incumbent, objective)
    receipt.update(
        {
            "honest_flags": [
                "vanilla_incumbent_captured_before_adaptation",
                "fallback_vanilla:ComputeBudgetUnaffordable",
                "fallback_reused_materialized_incumbent",
            ],
            "resident_owner_released": True,
            "resident_state_reusable": True,
            "last_stage": "latent_optimization",
        }
    )
    latent = _LatentService(
        {
            "ok": False,
            "reason": "receipt_contract_failed:latent_optimization_budget_exhausted",
            "text": incumbent,
            "receipt": receipt,
            "progress": {"stage": "failed", "elapsed_s": 151.0},
        }
    )
    phase = ResponseGenerationPhase(
        _Container({"llm_router": router, "latent_cortex": latent})
    )
    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [{"role": "user", "content": objective}],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )

    result = await phase.execute(state, context=_latent_context(objective))

    assert len(latent.calls) == 1
    assert router.calls == []
    assert result.cognition.working_memory[-1]["content"] == incumbent
    assert result.response_modifiers["latent_cortex_succeeded"] is False
    assert result.response_modifiers["latent_cortex_fallback_used"] is True
    assert result.response_modifiers["latent_cortex_incumbent_fallback_served"] is True
    assert result.response_modifiers["model_retry_suppressed"] is True
    assert result.response_modifiers["response_path"] == (
        "cognitive_engine_latent_incumbent_fallback"
    )


@pytest.mark.asyncio
async def test_latent_timeout_preserves_attempt_and_suppresses_second_model_owner(
    monkeypatch,
):
    objective = "Compare both architectures, then choose and verify the safer one."
    state = AuraState()
    state.cognition.current_objective = objective
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_mode = CognitiveMode.DELIBERATE
    router = _Router()
    latent = _TimedOutLatentService({})
    phase = ResponseGenerationPhase(
        _Container({"llm_router": router, "latent_cortex": latent})
    )
    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [{"role": "user", "content": objective}],
    )

    result = await phase.execute(state, context=_latent_context(objective))

    assert len(latent.calls) == 1
    assert router.calls == []
    assert result.response_modifiers["latent_cortex_selected"] is True
    assert result.response_modifiers["latent_cortex_attempted"] is True
    assert result.response_modifiers["latent_cortex_succeeded"] is False
    assert result.response_modifiers["model_retry_suppressed"] is True
    assert (
        result.response_modifiers["generation_failure_class"]
        == "latent_timeout:TimeoutError"
    )


@pytest.mark.asyncio
async def test_returned_latent_timeout_preserves_receipt_and_suppresses_fallback(
    monkeypatch,
):
    objective = "Compare both architectures, then choose and verify the safer one."
    state = AuraState()
    state.cognition.current_objective = objective
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_mode = CognitiveMode.DELIBERATE
    router = _Router()
    receipt = {
        "episode_id": "live-timeout",
        "input_token_count": 4096,
        "params_unchanged": True,
        "last_stage": "prefill",
        "stage_timings_s": {"prefill": 119.2},
    }
    progress = {
        "stage": "prefill",
        "elapsed_s": 119.2,
        "input_tokens": 4096,
    }
    latent = _LatentService(
        {
            "ok": False,
            "reason": "latent_timeout:cooperative_cancelled",
            "receipt": receipt,
            "progress": progress,
        }
    )
    phase = ResponseGenerationPhase(
        _Container({"llm_router": router, "latent_cortex": latent})
    )
    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [{"role": "user", "content": objective}],
    )

    result = await phase.execute(state, context=_latent_context(objective))

    assert len(latent.calls) == 1
    assert router.calls == []
    assert result.response_modifiers["model_retry_suppressed"] is True
    assert (
        result.response_modifiers["generation_failure_class"]
        == "latent_timeout:cooperative_cancelled"
    )
    assert result.response_modifiers["latent_cortex_receipt"] == receipt
    assert result.response_modifiers["latent_cortex_progress"] == progress
    assert (
        result.response_modifiers["response_path"]
        == "cognitive_engine_latent_owner_exhausted"
    )


@pytest.mark.asyncio
async def test_response_generation_downshifts_on_thermal_pressure(monkeypatch):
    state = AuraState()
    state.cognition.current_objective = "Perform a deep architectural audit"
    state.cognition.current_origin = "user"
    state.cognition.current_mode = CognitiveMode.DELIBERATE
    state.response_modifiers["model_tier"] = "secondary"
    state.response_modifiers["deep_handoff"] = True
    state.soma.hardware["temperature"] = 96.0
    state.soma.hardware["cpu_usage"] = 63.0

    router = _Router()
    container = _Container({"llm_router": router})
    phase = ResponseGenerationPhase(container)

    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [{"role": "system", "content": "context"}],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )

    new_state = await phase.execute(state)

    assert router.calls
    call = router.calls[0]
    assert call["prefer_tier"] == "tertiary"
    assert call["deep_handoff"] is False
    assert call["max_tokens"] < 6144
    assert new_state.response_modifiers["thermal_guard"] is True
    # Downstream voice shaping may add punctuation/styling, so verify content
    # presence rather than exact equality.
    assert "Thermal-safe response" in new_state.cognition.last_response


@pytest.mark.asyncio
async def test_response_generation_executes_required_search_before_answering(monkeypatch):
    state = AuraState()
    state.cognition.current_objective = (
        "Search the web for current NASA Europa page and tell me source title only."
    )
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_mode = CognitiveMode.REACTIVE
    state.response_modifiers["matched_skills"] = ["web_search"]

    router = _EvidenceRouter()
    capability = _SearchCapability()
    phase = ResponseGenerationPhase(
        _Container({"llm_router": router, "capability_engine": capability})
    )

    def _messages_from_state(state_arg, _objective):
        skill_blocks = [
            str(item.get("content") or "")
            for item in state_arg.cognition.working_memory
            if isinstance(item, dict)
            and (item.get("metadata") or {}).get("type") == "skill_result"
        ]
        return [{"role": "system", "content": "\n".join(["context", *skill_blocks])}]

    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        _messages_from_state,
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )

    new_state = await phase.execute(
        state,
        context={
            "desktop_cognitive_engine_required": True,
            "cognitive_engine_required": True,
            "visible_user_message": state.cognition.current_objective,
            "max_tokens": 512,
        },
    )

    assert capability.calls
    skill_name, params, context = capability.calls[0]
    assert skill_name == "web_search"
    assert params["query"] == "current NASA Europa page"
    assert params["retain"] is False
    assert context["effect_scope"] == "read_only_external_io"
    assert new_state.response_modifiers["last_skill_ok"] is True
    assert new_state.response_modifiers["last_skill_run"] == "web_search"
    assert "[SKILL RESULT: web_search]" in router.calls[0]["messages"][0]["content"]
    assert "Europa: Jupiter's Ocean World" in new_state.cognition.last_response


@pytest.mark.asyncio
async def test_response_generation_preserves_source_definition_search_tail(monkeypatch):
    state = AuraState()
    state.cognition.current_objective = (
        "Please search the web for one current NASA page about Europa. "
        "Tell me the source title and what NASA says Europa is."
    )
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_mode = CognitiveMode.REACTIVE
    state.response_modifiers["matched_skills"] = ["web_search"]

    router = _EvidenceRouter()
    capability = _SearchCapability()
    phase = ResponseGenerationPhase(
        _Container({"llm_router": router, "capability_engine": capability})
    )

    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda state_arg, _objective: [
            {
                "role": "system",
                "content": "\n".join(
                    str(item.get("content") or "")
                    for item in state_arg.cognition.working_memory
                    if isinstance(item, dict)
                ),
            }
        ],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )

    new_state = await phase.execute(
        state,
        context={
            "desktop_cognitive_engine_required": True,
            "cognitive_engine_required": True,
            "visible_user_message": state.cognition.current_objective,
            "max_tokens": 512,
        },
    )

    assert capability.calls
    _, params, _ = capability.calls[0]
    assert "one current NASA page about Europa" in params["query"]
    assert "what NASA says Europa is" in params["query"]
    assert "Tell me the source title" not in params["query"]
    assert new_state.response_modifiers["last_skill_ok"] is True


def test_required_search_cleaner_uses_original_request_inside_repair_prompt():
    repair_prompt = (
        "The prior draft for this same user turn did not satisfy the user-facing response contract.\n"
        "Observed problems: truncated_tail.\n\n"
        "Rewrite from scratch for the original user request below.\n\n"
        "Original user request:\n"
        "Please search the web for one current NASA page about Europa. "
        "Tell me the source title and what NASA says Europa is.\n\n"
        "Rejected draft for avoidance only:\n"
        "I've searched and found a relevant page from NASA. The source title is"
    )

    cleaned = ResponseGenerationPhase._clean_required_search_query(repair_prompt)

    assert "one current NASA page about Europa" in cleaned
    assert "what NASA says Europa is" in cleaned
    assert "Rejected draft" not in cleaned
    assert "I've searched" not in cleaned


@pytest.mark.asyncio
async def test_response_generation_required_search_uses_service_container_fallback(monkeypatch):
    state = AuraState()
    state.cognition.current_objective = (
        "Search the web for current NASA Europa page and tell me source title only."
    )
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_mode = CognitiveMode.REACTIVE
    state.response_modifiers["matched_skills"] = ["web_search"]

    router = _EvidenceRouter()
    capability = _SearchCapability()
    phase = ResponseGenerationPhase(
        _Container(
            {
                "llm_router": router,
                "composer_node": SimpleNamespace(
                    refine=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        AssertionError("composer reopened a foreground draft")
                    )
                ),
            }
        )
    )

    def _messages_from_state(state_arg, _objective):
        skill_blocks = [
            str(item.get("content") or "")
            for item in state_arg.cognition.working_memory
            if isinstance(item, dict)
            and (item.get("metadata") or {}).get("type") == "skill_result"
        ]
        return [{"role": "system", "content": "\n".join(["context", *skill_blocks])}]

    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        _messages_from_state,
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )
    monkeypatch.setattr(
        "core.phases.response_generation.ServiceContainer.get",
        lambda name, default=None: capability if name == "capability_engine" else default,
    )

    new_state = await phase.execute(
        state,
        context={
            "desktop_cognitive_engine_required": True,
            "cognitive_engine_required": True,
            "visible_user_message": state.cognition.current_objective,
            "max_tokens": 512,
        },
    )

    assert capability.calls
    assert len(router.calls) == 1
    assert "[SKILL RESULT: web_search]" in router.calls[0]["messages"][0]["content"]
    assert "Europa: Jupiter's Ocean World" in new_state.cognition.last_response


@pytest.mark.asyncio
async def test_response_generation_repairs_false_search_inability_after_evidence(monkeypatch):
    state = AuraState()
    state.cognition.current_objective = (
        "Search the web for one current NASA page about Europa, then answer with the source title and one sentence."
    )
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_mode = CognitiveMode.REACTIVE
    state.response_modifiers["matched_skills"] = ["web_search"]

    router = _FalseSearchInabilityRouter()
    capability = _SearchCapability()
    phase = ResponseGenerationPhase(
        _Container({"llm_router": router, "capability_engine": capability})
    )

    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda state_arg, _objective: [
            {
                "role": "system",
                "content": "\n".join(
                    str(item.get("content") or "")
                    for item in state_arg.cognition.working_memory
                    if isinstance(item, dict)
                ),
            }
        ],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )

    new_state = await phase.execute(
        state,
        context={
            "desktop_cognitive_engine_required": True,
            "cognitive_engine_required": True,
            "visible_user_message": state.cognition.current_objective,
            "max_tokens": 512,
        },
    )

    assert capability.calls
    assert new_state.response_modifiers["last_skill_ok"] is True
    assert (
        new_state.response_modifiers["required_tool_false_inability_repaired"]["skill"]
        == "web_search"
    )
    reply = new_state.cognition.last_response
    assert "Europa: Jupiter's Ocean World" in reply
    assert "science.nasa.gov/jupiter/moons/europa" in reply
    assert "can't execute web searches" not in reply.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("router_cls", "receipt_key"),
    [
        (_TimeoutSearchRouter, "required_tool_timeout_repaired"),
        (_BlankSearchRouter, "required_tool_empty_repaired"),
    ],
)
async def test_response_generation_answers_from_search_evidence_when_cortex_fails(
    monkeypatch,
    router_cls,
    receipt_key,
):
    state = AuraState()
    state.cognition.current_objective = (
        "Search the web for one current NASA page about Europa, then answer with the source title and one sentence."
    )
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_mode = CognitiveMode.REACTIVE
    state.response_modifiers["matched_skills"] = ["web_search"]

    router = router_cls()
    capability = _SearchCapability()
    phase = ResponseGenerationPhase(
        _Container({"llm_router": router, "capability_engine": capability})
    )

    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [{"role": "system", "content": "context"}],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )

    new_state = await phase.execute(
        state,
        context={
            "desktop_cognitive_engine_required": True,
            "cognitive_engine_required": True,
            "visible_user_message": state.cognition.current_objective,
            "max_tokens": 512,
        },
    )

    assert capability.calls
    assert new_state.response_modifiers["last_skill_ok"] is True
    assert new_state.response_modifiers[receipt_key]["skill"] == "web_search"
    reply = new_state.cognition.last_response
    assert "Europa: Jupiter's Ocean World" in reply
    assert "science.nasa.gov/jupiter/moons/europa" in reply
    mutation_stages = {
        item["stage"]
        for item in new_state.response_modifiers["live_mind_surface_control_receipt"][
            "text_mutations"
        ]
    }
    expected_stage = (
        "response_generation.required_tool_timeout_recovery"
        if router_cls is _TimeoutSearchRouter
        else "response_generation.required_tool_blank_recovery"
    )
    assert expected_stage in mutation_stages
    assert new_state.response_modifiers["live_mind_surface_control_receipt"][
        "deterministic_repair_applied"
    ] is True


@pytest.mark.asyncio
async def test_optional_stage_timeout_cannot_erase_resident_model_incumbent(
    monkeypatch,
):
    state = AuraState()
    state.cognition.current_objective = (
        "Explain why a verified search result matters in one complete response."
    )
    state.cognition.current_origin = "user"
    state.cognition.current_mode = CognitiveMode.REACTIVE

    router = _Router()
    phase = ResponseGenerationPhase(_Container({"llm_router": router}))

    async def optional_stage_times_out(**_kwargs):
        raise TimeoutError("optional stage exceeded the enclosing turn")

    monkeypatch.setattr(phase, "_maybe_amplify_response", optional_stage_times_out)
    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [{"role": "system", "content": "context"}],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )

    new_state = await phase.execute(
        state,
        context={
            "desktop_cognitive_engine_required": True,
            "cognitive_engine_required": True,
            "visible_user_message": state.cognition.current_objective,
            "max_tokens": 512,
        },
    )

    expected = (
        "Thermal-safe response: I am keeping the architectural audit grounded, "
        "reducing the local load, and preserving the conversation thread."
    )
    assert new_state.cognition.last_response == expected
    assert new_state.response_modifiers["optional_stage_timeout_repaired"] == {
        "method": "preserve_servable_incumbent",
        "completed_capabilities": [],
    }
    # The mutation ledger records changed bytes only. The incumbent was already
    # in ``response_text``, so preserving it is an ownership event rather than a
    # visible-text mutation.
    assert new_state.response_modifiers["live_mind_surface_control_receipt"][
        "text_mutations"
    ] == []


@pytest.mark.asyncio
async def test_response_generation_rejects_internal_cap_below_visible_surface(monkeypatch):
    state = AuraState()
    state.cognition.current_objective = "How would that uncertainty change your next decision?"
    state.cognition.current_origin = "user"
    state.cognition.current_mode = CognitiveMode.DELIBERATE
    state.response_modifiers["sampling_bias"] = {"max_tokens_factor": 1.25}
    state.response_modifiers["imagination_sampling_bias"] = {"max_tokens_factor": 1.25}
    state.response_modifiers["bicameral_sampling_bias"] = {"max_tokens_factor": 1.25}

    router = _Router()
    phase = ResponseGenerationPhase(
        _Container(
            {
                "llm_router": router,
                "composer_node": SimpleNamespace(
                    refine=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        AssertionError("composer reopened a foreground draft")
                    )
                ),
            }
        )
    )
    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [{"role": "system", "content": "context"}],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )

    await phase.execute(
        state,
        context={
            "desktop_cognitive_engine_required": True,
            "cognitive_engine_required": True,
            "visible_user_message": state.cognition.current_objective,
            "max_tokens": 1,
        },
    )

    assert router.calls
    assert router.calls[0]["max_tokens"] == 256


@pytest.mark.asyncio
async def test_response_generation_biases_cannot_erase_compound_answer_capacity(
    monkeypatch,
):
    state = AuraState()
    state.cognition.current_objective = (
        "Explain Dijkstra's algorithm in one complete response. Include: "
        "(1) the core invariant, (2) numbered pseudocode, (3) a worked example "
        "with five weighted edges, (4) binary-heap and array complexity, and "
        "(5) a negative-weight failure and the correct alternative."
    )
    state.cognition.current_origin = "user"
    state.cognition.current_mode = CognitiveMode.DELIBERATE
    state.response_modifiers["sampling_bias"] = {"max_tokens_factor": 0.25}
    state.response_modifiers["imagination_sampling_bias"] = {
        "max_tokens_factor": 0.25
    }

    router = _Router()
    phase = ResponseGenerationPhase(_Container({"llm_router": router}))
    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [{"role": "system", "content": "context"}],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )

    await phase.execute(
        state,
        context={
            "desktop_cognitive_engine_required": True,
            "cognitive_engine_required": True,
            "visible_user_message": state.cognition.current_objective,
            "max_tokens": 1536,
        },
    )

    assert router.calls
    assert router.calls[0]["max_tokens"] >= 2304
    assert router.calls[0]["user_surface_completion_floor"] == 2304


@pytest.mark.asyncio
async def test_response_generation_forwards_compiled_visible_output_contract(monkeypatch):
    state = AuraState()
    state.cognition.current_objective = 'Reply exactly: "yes"'
    state.cognition.current_origin = "user"
    state.cognition.current_mode = CognitiveMode.REACTIVE

    router = _Router()
    phase = ResponseGenerationPhase(_Container({"llm_router": router}))
    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [{"role": "system", "content": "context"}],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )

    result = await phase.execute(
        state,
        context={
            "visible_user_message": state.cognition.current_objective,
            "max_tokens": 4096,
        },
    )

    assert router.calls
    call = router.calls[0]
    assert call["max_tokens"] == len(b"yes") + 16
    assert call["requested_output_contract"]["kind"] == "exact_reply"
    assert call["semantic_output_token_cap"] == 8
    assert call["hard_output_token_ceiling"] == len(b"yes") + 16
    assert result.cognition.last_response == "yes"


@pytest.mark.asyncio
async def test_response_amplifier_adopts_selected_candidate_metadata_not_last_completion(
    monkeypatch,
):
    router = _ConcurrentAmplifierRouter()
    phase = ResponseGenerationPhase(_Container({"llm_router": router}))
    state = AuraState()

    async def fake_amplify_turn(_objective, generate, **_kwargs):
        candidates = await asyncio.gather(
            generate("candidate-0", 0.2),
            generate("candidate-1", 0.8),
        )
        winner = candidates[0]
        # The REAL AmplifiedAnswer/ReasoningReceipt, not a SimpleNamespace.
        # This used to hand-roll a receipt of {"winner": 0}, and when
        # promotion_authority became an adoption precondition the fake kept
        # omitting it — so the phase correctly kept the draft and the
        # assertions below were checking candidate selection on a path the
        # test never reached. A double built from the production dataclass
        # cannot drift from it silently.
        return amplified_answer(
            str(winner),
            promotion_authority="checked_verifier",
            generation_metadata=generation_metadata_of(winner),
            winning_candidate_id=0,
        )

    monkeypatch.setattr(
        "core.brain.reasoning_amplifier_v2.is_amplifiable",
        lambda _objective: "planning",
    )
    monkeypatch.setattr(
        "core.brain.reasoning_amplifier_v2.amplify_turn",
        fake_amplify_turn,
    )

    answer = await phase._maybe_amplify_response(
        objective="Plan a causally valid migration",
        draft="initial draft",
        router=router,
        state=state,
        request_timeout=30.0,
        origin="user",
        tier="primary",
        runtime_context={"visible_user_message": "Plan the migration"},
        is_user_facing=True,
        is_background=False,
        proof_or_benchmark=False,
    )

    assert answer == "candidate answer 0"
    assert router.completion_order == [1, 0]
    assert generation_metadata_of(answer)["candidate_id"] == 0


@pytest.mark.asyncio
async def test_response_generation_suppresses_background_identity_refresh_when_runtime_is_not_idle(monkeypatch):
    state = AuraState()
    state.cognition.current_objective = "[IDENTITY REFRESH: REMEMBER WHO YOU ARE]\nSummarize recent continuity."
    state.cognition.current_origin = "system"
    state.cognition.current_mode = CognitiveMode.REACTIVE

    router = _Router()
    container = _Container({"llm_router": router})
    phase = ResponseGenerationPhase(container)

    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        lambda *args, **kwargs: "failure_lockdown_0.20",
    )

    result = await phase.execute(state)

    assert result is state
    assert router.calls == []


@pytest.mark.asyncio
async def test_response_generation_suppresses_background_noise_objective(monkeypatch):
    state = AuraState()
    state.cognition.current_objective = "Task exception: database is locked while background cognitive state retries."
    state.cognition.current_origin = "autonomous_thought"
    state.cognition.current_mode = CognitiveMode.REACTIVE

    router = _Router()
    container = _Container({"llm_router": router})
    phase = ResponseGenerationPhase(container)

    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        lambda *args, **kwargs: "",
    )

    result = await phase.execute(state)

    assert result is state
    assert router.calls == []


@pytest.mark.asyncio
async def test_response_generation_treats_prefixed_user_origin_as_foreground(monkeypatch):
    state = AuraState()
    state.cognition.current_objective = "Hello Aura."
    state.cognition.current_origin = "routing_user"
    state.cognition.current_mode = CognitiveMode.REACTIVE

    router = _Router()
    container = _Container({"llm_router": router})
    phase = ResponseGenerationPhase(container)

    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [{"role": "system", "content": "context"}],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )

    background_gate_calls = []

    def _unexpected_background_gate(*_args, **_kwargs):
        background_gate_calls.append((_args, _kwargs))
        raise AssertionError("foreground origins should not consult background gating")

    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        _unexpected_background_gate,
    )

    result = await phase.execute(state)

    # The router must have been called as a foreground request.
    # We don't assert the exact response text because downstream voice shaping
    # (SubstrateVoiceEngine) may legitimately restyle it — but the routing
    # decision (foreground vs background) is what this test validates.
    assert background_gate_calls == []
    assert router.calls, "Router should have been called for a user-facing origin"
    assert router.calls[0]["is_background"] is False
    assert result.cognition.last_response, "A response should have been generated"


@pytest.mark.asyncio
async def test_response_generation_full_phase_injects_live_desktop_grounding(monkeypatch):
    state = AuraState()
    state.cognition.current_objective = "What tools can you use externally?"
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_mode = CognitiveMode.REACTIVE

    router = _Router()
    container = _Container({"llm_router": router})
    phase = ResponseGenerationPhase(container)

    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [
            {"role": "system", "content": "base live Aura context"},
            {"role": "user", "content": "What tools can you use externally?"},
        ],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )

    result = await phase.execute(
        state,
        context={
            "desktop_cognitive_engine_required": True,
            "cognitive_engine_required": True,
            "live_runtime_payload_required": True,
            "live_mind_context_required": True,
            "live_mind_context": {
                "required_for_live_desktop": True,
                "must_answer_from_full_mind_path": True,
                "required_subsystems_ok": True,
                "lane": {"state": "ready", "conversation_ready": True},
                "voice": {"mood": "steady"},
                "substrate": {"coherence": 0.91},
                "governance": {"legacy_fallback_allowed": False},
            },
            "mind_context_contract": "Use the live mind context as causal grounding.",
            "live_speech_grounding_frame": {
                "attention_focus": "Bryan's live desktop capability question",
                "dominant_action": "answer",
                "mood": "steady",
            },
            "grounded_capability_inventory_context": (
                "Aura can use governed desktop, browser, file, document, and terminal lanes "
                "only with authorization and effect receipts."
            ),
            "clean_user_surface_contract": True,
            "user_surface_validation_prompt": "What tools can you use externally?",
            "live_mind_controls_bound": True,
            "live_mind_generation_controls": {
                "temperature": 0.61,
                "top_p": 0.87,
                "clean_user_surface_recurrent_loops": 2,
                "clean_user_surface_steering_alpha": 0.30,
            },
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
        },
    )

    assert router.calls
    system_prompt = router.calls[0]["messages"][0]["content"]
    assert "LIVE MIND CONTEXT" in system_prompt
    assert "must_answer_from_full_mind_path" in system_prompt
    assert "LIVE SPEECH GROUNDING" in system_prompt
    assert "GOVERNED CAPABILITY INVENTORY EVIDENCE" in system_prompt
    call = router.calls[0]
    assert call["clean_user_surface_contract"] is True
    assert call["user_surface_validation_prompt"] == "What tools can you use externally?"
    assert call["live_mind_controls_bound"] is True
    assert call["live_mind_snapshot_ready"] is True
    assert call["live_mind_required_subsystems_ok"] is True
    assert call["clean_user_surface_recurrent_loops"] == 2
    assert call["clean_user_surface_steering_alpha"] == 0.30
    assert call["temperature"] == 0.61
    assert call["top_p"] == 0.87
    assert result.response_modifiers["live_mind_controls_worker_applied"] is True
    assert result.response_modifiers["live_mind_surface_control_receipt"]["applied"] is True


@pytest.mark.asyncio
async def test_response_generation_keeps_receipt_attached_to_exact_text(monkeypatch):
    state = AuraState()
    visible = "Explain what retained generation state means."
    state.cognition.current_objective = visible
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_mode = CognitiveMode.REACTIVE

    router = _AttributedReceiptRouter()
    phase = ResponseGenerationPhase(
        _Container(
            {
                "llm_router": router,
                "composer_node": SimpleNamespace(
                    refine=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        AssertionError("composer touched resumable draft")
                    )
                ),
            }
        )
    )

    async def _unexpected_amplifier(**_kwargs):
        raise AssertionError("amplifier touched resumable draft")

    async def _unexpected_dialogue_rewrite(*_args, **_kwargs):
        raise AssertionError("dialogue rewrite touched resumable draft")

    monkeypatch.setattr(phase, "_maybe_amplify_response", _unexpected_amplifier)
    monkeypatch.setattr(
        phase,
        "_clean_response",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cleanup touched resumable draft")
        ),
    )
    monkeypatch.setattr(
        phase,
        "_repair_false_required_tool_inability",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("tool repair touched resumable draft")
        ),
    )
    monkeypatch.setattr(
        "core.phases.response_generation.enforce_dialogue_contract",
        _unexpected_dialogue_rewrite,
    )
    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [
            {"role": "system", "content": "base live Aura context"},
            {"role": "user", "content": visible},
        ],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(
            align=lambda _text: (_ for _ in ()).throw(
                AssertionError("executive guard touched resumable draft")
            )
        ),
    )

    result = await phase.execute(
        state,
        context={
            "desktop_cognitive_engine_required": True,
            "cognitive_engine_required": True,
            "clean_user_surface_contract": True,
            "visible_user_message": visible,
            "user_surface_validation_prompt": visible,
            "live_mind_controls_bound": True,
            "live_mind_generation_controls": {
                "temperature": 0.55,
                "top_p": 0.88,
                "clean_user_surface_recurrent_loops": 2,
                "clean_user_surface_steering_alpha": 0.30,
            },
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
        },
    )

    receipt = result.response_modifiers["live_mind_surface_control_receipt"]
    assert receipt["continuation_resume_handle"] == "a" * 32
    assert receipt["generation_stop_reason"] == "deadline_exceeded"
    assert receipt["semantic_completion_incomplete"] is True
    assert router.calls and len(router.calls) == 1
    assert result.cognition.last_response == (
        "The retained draft is complete enough to continue from its exact state."
    )
    assert receipt["text_mutations"] == []


@pytest.mark.asyncio
async def test_fresh_completion_evidence_overrides_stale_transport_sink(monkeypatch):
    state = AuraState()
    visible = "Search the web and identify the current release with sources."
    state.cognition.current_objective = visible
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_mode = CognitiveMode.REACTIVE

    router = _StaleSinkFreshSnapshotRouter()
    phase = ResponseGenerationPhase(_Container({"llm_router": router}))

    async def _unexpected_amplifier(**_kwargs):
        raise AssertionError("amplifier reopened an incomplete owned draft")

    async def _unexpected_dialogue_rewrite(*_args, **_kwargs):
        raise AssertionError("phase reopened the route-owned continuation")

    monkeypatch.setattr(phase, "_maybe_amplify_response", _unexpected_amplifier)
    monkeypatch.setattr(
        "core.phases.response_generation.enforce_dialogue_contract",
        _unexpected_dialogue_rewrite,
    )
    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [
            {"role": "system", "content": "base live Aura context"},
            {"role": "user", "content": visible},
        ],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(
            align=lambda _text: (_ for _ in ()).throw(
                AssertionError("executive guard touched resumable draft")
            )
        ),
    )

    result = await phase.execute(
        state,
        context={
            "desktop_cognitive_engine_required": True,
            "cognitive_engine_required": True,
            "clean_user_surface_contract": True,
            "visible_user_message": visible,
            "user_surface_validation_prompt": visible,
            "live_mind_controls_bound": True,
            "live_mind_generation_controls": {
                "temperature": 0.55,
                "top_p": 0.88,
                "clean_user_surface_recurrent_loops": 2,
                "clean_user_surface_steering_alpha": 0.30,
            },
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
        },
    )

    receipt = result.response_modifiers["live_mind_surface_control_receipt"]
    assert len(router.calls) == 1
    assert receipt["semantic_completion_incomplete"] is True
    assert receipt["continuation_resume_handle"] == "c" * 32
    assert result.cognition.last_response.endswith("release as")


@pytest.mark.asyncio
async def test_desktop_owner_cannot_be_reopened_by_mislabeled_dialogue_contract(
    monkeypatch,
):
    from core.phases.response_contract import build_response_contract

    state = AuraState()
    visible = "Explain the result completely."
    state.cognition.current_objective = visible
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_mode = CognitiveMode.REACTIVE
    mislabeled = build_response_contract(state, visible, is_user_facing=False)
    retry_owners = []

    async def _capture_dialogue_owner(
        text,
        _contract,
        *,
        retry_generate=None,
        **_kwargs,
    ):
        retry_owners.append(retry_generate)
        return text, SimpleNamespace(to_dict=lambda: {}, selected_source="incumbent"), False

    router = _Router()
    phase = ResponseGenerationPhase(
        _Container(
            {
                "llm_router": router,
                "composer_node": SimpleNamespace(
                    refine=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        AssertionError("composer reopened a foreground draft")
                    )
                ),
            }
        )
    )
    monkeypatch.setattr(
        "core.phases.response_generation.build_response_contract",
        lambda *_args, **_kwargs: mislabeled,
    )
    monkeypatch.setattr(
        "core.phases.response_generation.enforce_dialogue_contract",
        _capture_dialogue_owner,
    )
    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [{"role": "user", "content": visible}],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )

    await phase.execute(
        state,
        context={
            "desktop_cognitive_engine_required": True,
            "clean_user_surface_contract": True,
            "visible_user_message": visible,
        },
    )

    assert len(router.calls) == 1
    assert retry_owners == [None]


@pytest.mark.asyncio
async def test_foreground_origin_derives_clean_completion_without_caller_labels(
    monkeypatch,
):
    from core.phases.response_contract import build_response_contract

    state = AuraState()
    visible = "Explain why exact cache ownership matters."
    state.cognition.current_objective = visible
    state.cognition.current_origin = "user"
    state.cognition.current_mode = CognitiveMode.REACTIVE
    mislabeled = build_response_contract(state, visible, is_user_facing=False)
    retry_owners = []

    async def _capture_dialogue_owner(
        text,
        _contract,
        *,
        retry_generate=None,
        **_kwargs,
    ):
        retry_owners.append(retry_generate)
        return text, SimpleNamespace(to_dict=lambda: {}, selected_source="incumbent"), False

    router = _Router()

    async def _single_owner_think(**kwargs):
        router.calls.append(kwargs)
        return "Exact cache ownership prevents one request from resuming another request's state."

    router.think = _single_owner_think
    phase = ResponseGenerationPhase(_Container({"llm_router": router}))
    monkeypatch.setattr(
        "core.phases.response_generation.build_response_contract",
        lambda *_args, **_kwargs: mislabeled,
    )
    monkeypatch.setattr(
        "core.phases.response_generation.enforce_dialogue_contract",
        _capture_dialogue_owner,
    )
    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [{"role": "user", "content": visible}],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )

    await phase.execute(state, context={})

    assert len(router.calls) == 1
    assert router.calls[0]["clean_user_surface_contract"] is True
    assert router.calls[0]["semantic_completion_contract"] is True
    assert retry_owners == [None]


@pytest.mark.asyncio
async def test_response_generation_quality_gate_uses_visible_desktop_prompt(monkeypatch):
    state = AuraState()
    visible = (
        "Remember this note for later in this conversation: "
        "the blue lantern is under the desk."
    )
    contract_wrapped = (
        f"{visible}\n\n[LIVE DESKTOP FULL-MIND CONTRACT]\n"
        "- Runtime path contract: governed tool and model lane status must remain available.\n"
        "[END LIVE DESKTOP FULL-MIND CONTRACT]"
    )
    state.cognition.current_objective = contract_wrapped
    state.cognition.current_origin = "desktop_quick_user"
    state.cognition.current_mode = CognitiveMode.REACTIVE

    router = _MemoryAckRouter()
    phase = ResponseGenerationPhase(_Container({"llm_router": router}))
    prompts_seen = []

    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [
            {"role": "system", "content": "base live Aura context"},
            {"role": "user", "content": contract_wrapped},
        ],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )

    from core.conversation.response_reliability import assess_user_facing_reply as real_assess

    def _record_assess(prompt, reply):
        prompts_seen.append(str(prompt))
        return real_assess(prompt, reply)

    monkeypatch.setattr(
        "core.phases.response_generation.assess_user_facing_reply",
        _record_assess,
    )

    result = await phase.execute(
        state,
        context={
            "desktop_cognitive_engine_required": True,
            "cognitive_engine_required": True,
            "live_runtime_payload_required": True,
            "clean_user_surface_contract": True,
            "visible_user_message": visible,
            "user_surface_validation_prompt": visible,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
        },
    )

    assert router.calls
    assert router.calls[0]["visible_user_message"] == visible
    assert router.calls[0]["user_surface_validation_prompt"] == visible
    assert prompts_seen
    assert all(prompt == visible for prompt in prompts_seen)
    assert "blue lantern" in result.cognition.last_response
    assert "Runtime path contract" not in result.cognition.last_response


@pytest.mark.asyncio
async def test_response_generation_leaves_user_surface_retry_to_route_owner(monkeypatch):
    state = AuraState()
    state.cognition.current_objective = "Explain how confusion changes your reasoning."
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_mode = CognitiveMode.REACTIVE

    router = _Router()
    async def _single_owner_think(**kwargs):
        router.calls.append(kwargs)
        return "Confusion increases uncertainty, so I slow down and check competing interpretations before answering."

    router.think = _single_owner_think
    phase = ResponseGenerationPhase(_Container({"llm_router": router}))

    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [
            {"role": "system", "content": "base live Aura context"},
            {"role": "user", "content": "Explain how confusion changes your reasoning."},
        ],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )

    async def _force_retry(
        response,
        contract,
        *,
        retry_generate,
        state,
        user_message=None,
    ):
        assert user_message == "Explain how confusion changes your reasoning."
        assert retry_generate is None
        return response, SimpleNamespace(to_dict=lambda: {"valid": False}), False

    monkeypatch.setattr(
        "core.phases.response_generation.enforce_dialogue_contract",
        _force_retry,
    )

    result = await phase.execute(
        state,
        context={
            "desktop_cognitive_engine_required": True,
            "cognitive_engine_required": True,
            "live_runtime_payload_required": True,
            "visible_user_message": "Explain how confusion changes your reasoning.",
            "clean_user_surface_contract": True,
            "live_mind_controls_bound": True,
            "live_mind_generation_controls": {
                "temperature": 0.49,
                "top_p": 0.81,
                "clean_user_surface_recurrent_loops": 2,
                "clean_user_surface_steering_alpha": 0.34,
            },
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
        },
    )

    assert len(router.calls) == 1
    assert router.calls[0]["desktop_cognitive_engine_required"] is True
    assert router.calls[0]["allow_cloud_fallback"] is False
    assert result.cognition.last_response.startswith("Confusion increases")


@pytest.mark.asyncio
async def test_user_facing_phase_never_opens_full_dialogue_regeneration(monkeypatch):
    state = AuraState()
    visible = (
        "Explain Dijkstra's invariant, give a worked example, include the "
        "binary-heap complexity, and name the negative-weight alternative."
    )
    state.cognition.current_objective = visible
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_mode = CognitiveMode.REACTIVE

    router = _Router()
    phase = ResponseGenerationPhase(_Container({"llm_router": router}))
    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [
            {"role": "system", "content": "base live Aura context"},
            {"role": "user", "content": visible},
        ],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda text: (text, False, [])),
    )

    async def _observe_retry_owner(
        response,
        contract,
        *,
        retry_generate,
        state,
        user_message=None,
    ):
        assert contract.is_user_facing is True
        assert retry_generate is None
        return response, SimpleNamespace(to_dict=lambda: {"valid": False}), False

    monkeypatch.setattr(
        "core.phases.response_generation.enforce_dialogue_contract",
        _observe_retry_owner,
    )

    await phase.execute(
        state,
        context={
            "desktop_cognitive_engine_required": True,
            "cognitive_engine_required": True,
            "visible_user_message": visible,
            "user_surface_validation_prompt": visible,
            # The ownership invariant does not depend on an optional caller
            # flag surviving every intermediate adapter.
            "clean_user_surface_contract": False,
        },
    )

    assert len(router.calls) == 1


@pytest.mark.asyncio
async def test_foreground_turn_does_not_redecode_when_alignment_empties_draft(
    monkeypatch,
):
    state = AuraState()
    visible = "Use the verified source evidence to answer this request."
    state.cognition.current_objective = visible
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_mode = CognitiveMode.REACTIVE

    router = _Router()
    phase = ResponseGenerationPhase(_Container({"llm_router": router}))
    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [
            {"role": "system", "content": "base live Aura context"},
            {"role": "user", "content": visible},
        ],
    )
    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=lambda _text: ("", True, ["test_alignment"])),
    )

    await phase.execute(
        state,
        context={
            "desktop_cognitive_engine_required": True,
            "cognitive_engine_required": True,
            "visible_user_message": visible,
        },
    )

    assert len(router.calls) == 1


@pytest.mark.asyncio
async def test_response_generation_ledgers_executive_guard_visible_replacement(monkeypatch):
    state = AuraState()
    state.cognition.current_objective = "Summarize the architectural audit."
    state.cognition.current_origin = "desktop_ui"
    state.cognition.current_mode = CognitiveMode.REACTIVE

    router = _Router()
    phase = ResponseGenerationPhase(_Container({"llm_router": router}))
    monkeypatch.setattr(
        "core.phases.response_generation.ContextAssembler.build_messages",
        lambda *_args, **_kwargs: [
            {"role": "system", "content": "base live Aura context"},
            {"role": "user", "content": "Summarize the architectural audit."},
        ],
    )

    def _align(text):
        return str(text).replace("Thermal-safe", "Identity-aligned"), True, [
            "identity_alignment"
        ]

    monkeypatch.setattr(
        "core.phases.response_generation.get_executive_guard",
        lambda: SimpleNamespace(align=_align),
    )

    result = await phase.execute(
        state,
        context={
            "visible_user_message": "Summarize the architectural audit.",
            "clean_user_surface_contract": True,
            "live_mind_controls_bound": True,
            "live_mind_generation_controls": {
                "temperature": 0.55,
                "top_p": 0.88,
                "clean_user_surface_recurrent_loops": 2,
                "clean_user_surface_steering_alpha": 0.30,
            },
        },
    )

    mutations = result.response_modifiers["live_mind_surface_control_receipt"][
        "text_mutations"
    ]
    assert any(
        item["stage"] == "response_generation.executive_guard"
        and item["deterministic"] is True
        for item in mutations
    )


def test_response_generation_timeout_fits_owning_cognitive_deadline(monkeypatch):
    monkeypatch.setattr("core.phases.response_generation.time.monotonic", lambda: 100.0)

    timeout = ResponseGenerationPhase._bounded_request_timeout(
        {"cognitive_cycle_deadline_monotonic": 130.0},
        180.0,
        reserve_s=8.0,
    )

    assert timeout == 22.0


def test_response_generation_timeout_keeps_requested_budget_without_deadline():
    assert (
        ResponseGenerationPhase._bounded_request_timeout({}, 180.0, reserve_s=8.0)
        == 180.0
    )
