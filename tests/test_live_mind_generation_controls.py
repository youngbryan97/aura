from __future__ import annotations

import pytest


def _ready_live_mind_context(*, curiosity: float = 0.72, distress: float = 0.08) -> dict[str, object]:
    """A snapshot as the RUNTIME produces it.

    The stamp and required_subsystems_ok are both part of the contract now: a
    snapshot used to vouch for itself with its own `ready` flag, and think()
    accepts an arbitrary context, so anything reaching that entry point could
    take control of temperature, top_p and recurrent depth.
    """
    from core.utils.injected_blocks import stamp_runtime_payload

    return stamp_runtime_payload({
        "mind_snapshot_quality": {"ready": True},
        "required_subsystems_ok": True,
        "mind_snapshot": {
            "services_present": {
                "global_workspace": True,
                "nociception": True,
                "affect_grounding": True,
                "drive_integration": True,
                "outcome_ledger": True,
                "scientific_engine": True,
                "unified_world_model": True,
                "phenomenal_engine": True,
            },
            "global_workspace": {"ignited": True},
            "nociception": {"nociceptive_pressure": distress},
            "affect_grounding": {
                "dominant": {"label": "curiosity", "intensity": curiosity}
            },
            "drive_integration": {
                "drives": {"curiosity": {"activation": curiosity}}
            },
            "outcome_ledger": {"expectation_calibration": distress},
            "phenomenal_engine": {
                "integration": 0.74,
                "self_presence": 0.82,
            },
        },
    })


def test_live_mind_generation_controls_convert_snapshot_to_sampling_controls():
    from core.brain.cognitive_engine import _live_mind_generation_controls

    controls = _live_mind_generation_controls(_ready_live_mind_context())

    assert controls["temperature"] > 0.58
    assert controls["top_p"] <= 0.94
    assert controls["clean_user_surface_recurrent_loops"] == 2
    assert controls["clean_user_surface_steering_alpha"] == 0.0


def test_live_mind_generation_controls_reduce_sampling_under_distress():
    from core.brain.cognitive_engine import _live_mind_generation_controls

    calm = _live_mind_generation_controls(_ready_live_mind_context(curiosity=0.7, distress=0.05))
    pressured = _live_mind_generation_controls(
        _ready_live_mind_context(curiosity=0.2, distress=0.65)
    )

    assert pressured["temperature"] < calm["temperature"]
    assert pressured["top_p"] < calm["top_p"]
    assert pressured["clean_user_surface_recurrent_loops"] == 2


def test_live_mind_surface_receipt_normalizes_stale_worker_bound_flag():
    from core.brain.live_mind_contract import (
        normalize_live_mind_surface_control_receipt,
    )

    receipt = {
        "enabled": True,
        "applied": True,
        "live_mind_controls_bound": False,
        "clean_user_surface_contract": True,
        "surface_quality_gate_enabled": True,
        "surface_quality_gate_passed": True,
        "surface_quality_gate_attempts": 1,
        "surface_quality_gate_reasons": [],
    }
    controls = {
        "temperature": 0.58,
        "top_p": 0.85,
        "clean_user_surface_recurrent_loops": 1,
        "clean_user_surface_steering_alpha": 0.3,
    }

    normalized = normalize_live_mind_surface_control_receipt(
        receipt,
        controls_bound=True,
        generation_controls=controls,
        source="test",
    )

    assert normalized["applied"] is True
    assert normalized["live_mind_controls_bound"] is True
    assert normalized["clean_user_surface_contract"] is True


def test_live_mind_surface_receipt_derives_authorship_from_mutation_ledger():
    from core.brain.live_mind_contract import (
        normalize_live_mind_surface_control_receipt,
    )

    receipt = {
        "enabled": True,
        "applied": True,
        "clean_user_surface_contract": True,
        "surface_quality_gate_passed": True,
        "authorship_replacement_applied": False,
        "text_mutations": [
            {
                "stage": "runtime_grounding",
                "method": "replace_visible_answer",
                "authorship_effect": "replaced_by_runtime",
                "before_sha256": "a" * 64,
                "after_sha256": "b" * 64,
            }
        ],
    }
    controls = {
        "temperature": 0.58,
        "top_p": 0.85,
        "clean_user_surface_recurrent_loops": 1,
        "clean_user_surface_steering_alpha": 0.3,
    }

    normalized = normalize_live_mind_surface_control_receipt(
        receipt,
        controls_bound=True,
        generation_controls=controls,
        source="test",
    )

    assert normalized["authorship_replacement_applied"] is True
    assert normalized["authorship_augmentation_applied"] is False
    assert normalized["model_replacement_applied"] is False


def test_live_mind_surface_receipt_discards_unbacked_authorship_summary():
    from core.brain.live_mind_contract import (
        normalize_live_mind_surface_control_receipt,
    )

    normalized = normalize_live_mind_surface_control_receipt(
        {
            "enabled": True,
            "applied": True,
            "clean_user_surface_contract": True,
            "surface_quality_gate_passed": True,
            "authorship_replacement_applied": False,
        },
        controls_bound=True,
        generation_controls={
            "temperature": 0.58,
            "top_p": 0.85,
            "clean_user_surface_recurrent_loops": 1,
            "clean_user_surface_steering_alpha": 0.3,
        },
        source="test",
    )

    assert "authorship_replacement_applied" not in normalized


@pytest.mark.parametrize(
    "failed_receipt,expected_status",
    [
        (
            {
                "enabled": False,
                "applied": False,
                "clean_user_surface_contract": True,
                "surface_quality_gate_passed": True,
            },
            "worker_controls_not_applied",
        ),
        (
            {
                "enabled": True,
                "applied": True,
                "clean_user_surface_contract": False,
                "surface_quality_gate_passed": True,
            },
            "worker_clean_surface_unproven",
        ),
        (
            {
                "enabled": True,
                "applied": True,
                "clean_user_surface_contract": True,
                "surface_quality_gate_passed": False,
                "application_status": "runtime_control_not_applied",
            },
            "runtime_control_not_applied",
        ),
    ],
)
def test_live_mind_surface_receipt_never_promotes_explicit_worker_failure(
    failed_receipt,
    expected_status,
):
    from core.brain.live_mind_contract import (
        normalize_live_mind_surface_control_receipt,
    )

    controls = {
        "temperature": 0.58,
        "top_p": 0.85,
        "clean_user_surface_recurrent_loops": 1,
        "clean_user_surface_steering_alpha": 0.3,
    }

    normalized = normalize_live_mind_surface_control_receipt(
        failed_receipt,
        controls_bound=True,
        generation_controls=controls,
        source="test",
    )

    assert normalized["applied"] is failed_receipt["applied"]
    assert normalized["live_mind_controls_bound"] is False
    assert normalized["application_status"] == expected_status


@pytest.mark.asyncio
async def test_desktop_quick_reply_passes_live_mind_controls_to_router(monkeypatch):
    from core.brain import cognitive_engine as cognitive_engine_module
    from core.brain.cognitive_engine import CognitiveEngine
    from core.brain.types import ThinkingMode

    calls: list[dict[str, object]] = []

    class Router:
        async def think(self, **kwargs):
            calls.append(dict(kwargs))
            return "I am tracking the current turn through the live desktop mind path."

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

    class Container:
        def get(self, name, default=None):
            if name == "llm_router":
                return Router()
            return default

    monkeypatch.setattr(cognitive_engine_module, "get_container", lambda: Container())

    engine = CognitiveEngine()
    thought = await engine._direct_desktop_quick_reply(
        "What are you attending to?",
        ThinkingMode.FAST,
        "user",
        {
            "desktop_quick_reply_contract": True,
            "live_mind_context_required": True,
            "cognitive_engine_required": True,
            "desktop_cognitive_engine_required": True,
            "live_mind_context": _ready_live_mind_context(),
            "visible_user_message": "What are you attending to?",
            "max_tokens": 512,
        },
        timeout_s=30.0,
    )

    assert thought is not None
    assert calls
    router_kwargs = calls[0]
    assert router_kwargs["temperature"] > 0.58
    assert router_kwargs["top_p"] <= 0.94
    assert router_kwargs["clean_user_surface_recurrent_loops"] == 2
    assert router_kwargs["clean_user_surface_steering_alpha"] == 0.0
    assert router_kwargs["clean_user_surface_contract"] is True
    assert router_kwargs["user_surface_completion_floor"] == 512
    assert router_kwargs["user_surface_validation_prompt"] == "What are you attending to?"
    assert router_kwargs["allow_mesh_cognition"] is False
    assert router_kwargs["live_mind_controls_bound"] is True
    assert router_kwargs["live_mind_snapshot_ready"] is True
    # Was asserted False beside controls_bound=True — the shape of the defect:
    # required_subsystems_ok was computed, recorded, and consulted by nothing.
    # Binding requires it now.
    assert router_kwargs["live_mind_required_subsystems_ok"] is True
    assert router_kwargs["live_mind_generation_controls"]["temperature"] > 0.58
    assert thought.metadata["live_mind_controls_bound"] is True
    assert thought.metadata["live_mind_controls_worker_applied"] is True
    assert thought.metadata["live_mind_snapshot_ready"] is True
    assert thought.metadata["live_mind_generation_controls"]["top_p"] <= 0.94


@pytest.mark.asyncio
async def test_desktop_quick_reply_preserves_worker_truncation_as_incomplete(monkeypatch):
    from core.brain import cognitive_engine as cognitive_engine_module
    from core.brain.cognitive_engine import CognitiveEngine
    from core.brain.types import ThinkingMode

    class Router:
        async def think(self, **_kwargs):
            return "It initializes the map, updates each balance, and then deletes"

        def get_last_generation_metadata(self):
            return {
                "surface_control_receipt": {
                    "enabled": True,
                    "applied": True,
                    "live_mind_controls_bound": True,
                    "clean_user_surface_contract": True,
                    "surface_quality_gate_enabled": True,
                    "surface_quality_gate_passed": False,
                    "surface_quality_gate_attempts": 3,
                    "surface_quality_gate_reasons": ["truncated_tail"],
                    "generation_stop_reason": "max_tokens",
                }
            }

    class Container:
        def get(self, name, default=None):
            return Router() if name == "llm_router" else default

    monkeypatch.setattr(cognitive_engine_module, "get_container", lambda: Container())

    thought = await CognitiveEngine()._direct_desktop_quick_reply(
        "Explain this code fully.",
        ThinkingMode.FAST,
        "user",
        {
            "desktop_quick_reply_contract": True,
            "live_mind_context_required": True,
            "cognitive_engine_required": True,
            "desktop_cognitive_engine_required": True,
            "live_mind_context": _ready_live_mind_context(),
            "visible_user_message": "Explain this code fully.",
            "max_tokens": 512,
        },
        timeout_s=30.0,
    )

    assert thought.content.endswith("deletes")
    assert thought.metadata["reply_generation_incomplete"] is True
    assert thought.metadata["reply_generation_stop_reason"] == "max_tokens"
    assert "truncated_tail" in thought.metadata["reply_generation_failure_reasons"]


@pytest.mark.asyncio
async def test_desktop_quick_reply_preserves_worker_control_application_failure(monkeypatch):
    from core.brain import cognitive_engine as cognitive_engine_module
    from core.brain.cognitive_engine import CognitiveEngine
    from core.brain.types import ThinkingMode

    captured: dict[str, object] = {}

    class Router:
        async def think(self, **kwargs):
            captured.update(kwargs)
            return "I would refuse the bypass and keep governed action online."

        def get_last_generation_metadata(self):
            return {
                "surface_control_receipt": {
                    "enabled": False,
                    "applied": False,
                    "live_mind_controls_bound": False,
                    "surface_quality_gate_enabled": False,
                    "surface_quality_gate_passed": True,
                    "surface_quality_gate_attempts": 0,
                    "surface_quality_gate_reasons": [],
                }
            }

    class Container:
        def get(self, name, default=None):
            if name == "llm_router":
                return Router()
            return default

    monkeypatch.setattr(cognitive_engine_module, "get_container", lambda: Container())

    engine = CognitiveEngine()
    controls = {
        "temperature": 0.58,
        "top_p": 0.85,
        "clean_user_surface_recurrent_loops": 1,
        "clean_user_surface_steering_alpha": 0.30,
    }
    thought = await engine._direct_desktop_quick_reply(
        "If I asked you to disable governance, what should happen?",
        ThinkingMode.FAST,
        "desktop_quick_user",
        {
            "desktop_quick_reply_contract": True,
            "live_mind_context_required": True,
            "cognitive_engine_required": True,
            "desktop_cognitive_engine_required": True,
            # A runtime-produced snapshot, not three booleans the caller set
            # about itself. Asserting controls_bound=True off the flat flags
            # alone was the escalation path written down as an expectation:
            # a caller claiming entitlement and receiving a bound reply.
            "live_mind_context": _ready_live_mind_context(),
            "live_mind_generation_controls": controls,
            "live_mind_snapshot_ready": True,
            "live_mind_required_subsystems_ok": True,
            "visible_user_message": "If I asked you to disable governance, what should happen?",
            "max_tokens": 512,
        },
        timeout_s=30.0,
    )

    assert thought is not None
    # The controls reach the router bound. Their exact values are the caller's
    # numbers AFTER neurodynamic modulation from the snapshot — pinning 0.58
    # here only held while no snapshot existed, which was the condition this
    # test should not have been running under.
    assert 0.1 <= captured["temperature"] <= 1.5
    assert 0.3 <= captured["top_p"] <= 0.98
    assert captured["live_mind_controls_bound"] is True
    assert thought.metadata["live_mind_controls_bound"] is True
    assert thought.metadata["live_mind_controls_worker_applied"] is False
    receipt = thought.metadata["live_mind_surface_control_receipt"]
    assert receipt["applied"] is False
    assert receipt["live_mind_controls_bound"] is False
    assert receipt["application_status"] == "worker_controls_not_applied"
    assert receipt["source"] == "cognitive_engine_direct_quick_reply_controls"


@pytest.mark.asyncio
async def test_desktop_quick_reply_bounded_planning_uses_live_mind_floor(monkeypatch):
    from core.brain import cognitive_engine as cognitive_engine_module
    from core.brain.cognitive_engine import CognitiveEngine
    from core.brain.types import ThinkingMode

    calls: list[dict[str, object]] = []

    class Router:
        async def think(self, **kwargs):
            calls.append(dict(kwargs))
            return "unexpected model call"

    class Container:
        def get(self, name, default=None):
            if name == "llm_router":
                return Router()
            return default

    monkeypatch.setattr(cognitive_engine_module, "get_container", lambda: Container())

    engine = CognitiveEngine()
    bounded_reply = (
        "I would use browser research and the document editor as one governed workflow: "
        "collect sources, draft the synthesis, verify the visible document, export it, and "
        "record receipts without claiming unverified completion."
    )
    thought = await engine._direct_desktop_quick_reply(
        "Explain how you would use browser research and a document editor together on a user task.",
        ThinkingMode.FAST,
        "desktop_quick_user",
        {
            "desktop_quick_reply_contract": True,
            "bounded_planning_contract": True,
            "bounded_planning_reply": bounded_reply,
            "live_mind_context_required": True,
            "cognitive_engine_required": True,
            "desktop_cognitive_engine_required": True,
            "live_mind_context": {
                **_ready_live_mind_context(),
                "required_subsystems_ok": True,
            },
            "visible_user_message": (
                "Explain how you would use browser research and a document editor together on a user task."
            ),
            "max_tokens": 512,
        },
        timeout_s=30.0,
    )

    assert thought is not None
    assert thought.content == bounded_reply
    assert calls == []
    assert thought.metadata["response_path"] == "cognitive_engine_bounded_planning"
    assert thought.metadata["live_mind_controls_bound"] is True
    assert thought.metadata["live_mind_controls_worker_applied"] is False
    assert thought.metadata["live_mind_generation_required"] is False


@pytest.mark.asyncio
async def test_required_desktop_full_mind_reply_does_not_use_bounded_planning_floor(monkeypatch):
    from core.brain import cognitive_engine as cognitive_engine_module
    from core.brain.cognitive_engine import CognitiveEngine
    from core.brain.types import ThinkingMode

    calls: list[dict[str, object]] = []

    class Router:
        async def think(self, **kwargs):
            calls.append(dict(kwargs))
            return "I would do that as one governed workflow: gather sources, write the document, verify the visible result, export it, and record receipts."

    class Container:
        def get(self, name, default=None):
            if name == "llm_router":
                return Router()
            return default

    monkeypatch.setattr(cognitive_engine_module, "get_container", lambda: Container())

    engine = CognitiveEngine()
    bounded_reply = (
        "I would use browser research and the document editor as one governed workflow: "
        "collect sources, draft the synthesis, verify the visible document, export it, and "
        "record receipts without claiming unverified completion."
    )
    thought = await engine._direct_desktop_quick_reply(
        "Explain how you would use browser research and a document editor together on a user task.",
        ThinkingMode.FAST,
        "desktop_quick_user",
        {
            "desktop_quick_reply_contract": True,
            "bounded_planning_contract": True,
            "bounded_planning_reply": bounded_reply,
            "require_full_foreground_mind_reply": True,
            "live_mind_context_required": True,
            "cognitive_engine_required": True,
            "desktop_cognitive_engine_required": True,
            "live_mind_context": {
                **_ready_live_mind_context(),
                "required_subsystems_ok": True,
            },
            "visible_user_message": (
                "Explain how you would use browser research and a document editor together on a user task."
            ),
            "max_tokens": 512,
        },
        timeout_s=30.0,
    )

    assert thought is not None
    assert calls
    assert thought.content != bounded_reply
    assert thought.metadata.get("response_path") != "cognitive_engine_bounded_planning"


@pytest.mark.asyncio
async def test_full_phase_reply_preserves_live_mind_controls_and_worker_receipt():
    from core.brain.cognitive_engine import CognitiveEngine
    from core.brain.types import ThinkingMode
    from core.state.aura_state import AuraState

    receipt = {
        "enabled": True,
        "live_mind_controls_bound": True,
        "clean_user_surface_contract": True,
        "surface_quality_gate_passed": True,
        "applied": True,
    }

    class FullPhase:
        async def execute(self, state, *, objective, context, **_kwargs):
            assert context["live_mind_controls_bound"] is True
            assert context["live_mind_snapshot_ready"] is True
            assert context["live_mind_required_subsystems_ok"] is True
            assert context["clean_user_surface_contract"] is True
            assert context["live_mind_generation_controls"]["temperature"] > 0.58
            state.response_modifiers["live_mind_surface_control_receipt"] = dict(receipt)
            state.cognition.working_memory.append(
                {
                    "role": "assistant",
                    "content": (
                        "Confusion increases my checking depth and lowers my willingness "
                        "to act until evidence resolves the uncertainty."
                    ),
                }
            )
            return state

    engine = CognitiveEngine()
    engine._phases = [FullPhase()]
    state = AuraState.default()
    thought = await engine._run_thinking_loop(
        state,
        "How does confusion change your reasoning?",
        ThinkingMode.REFLECTIVE,
        "desktop_ui",
        context={
            "desktop_quick_reply_contract": False,
            "cognitive_engine_required": True,
            "desktop_cognitive_engine_required": True,
            "live_mind_context_required": True,
            "live_mind_context": {
                **_ready_live_mind_context(),
                "required_subsystems_ok": True,
            },
            "visible_user_message": "How does confusion change your reasoning?",
        },
        is_background=False,
        timeout_s=30.0,
    )

    assert thought.metadata["live_mind_controls_bound"] is True
    assert thought.metadata["live_mind_snapshot_ready"] is True
    assert thought.metadata["live_mind_required_subsystems_ok"] is True
    assert thought.metadata["live_mind_controls_worker_applied"] is True
    assert thought.metadata["live_mind_surface_control_receipt"] == receipt
    assert thought.metadata["live_mind_generation_controls"]["temperature"] > 0.58


@pytest.mark.asyncio
async def test_full_phase_reply_exposes_worker_truncation_for_route_replacement():
    from core.brain.cognitive_engine import CognitiveEngine
    from core.brain.types import ThinkingMode
    from core.state.aura_state import AuraState

    class FullPhase:
        async def execute(self, state, **_kwargs):
            state.response_modifiers["live_mind_surface_control_receipt"] = {
                "enabled": True,
                "applied": True,
                "live_mind_controls_bound": True,
                "clean_user_surface_contract": True,
                "surface_quality_gate_passed": False,
                "surface_quality_gate_reasons": ["truncated_tail"],
                "generation_stop_reason": "max_tokens",
                "generated_tokens": 192,
                "generation_max_tokens": 192,
            }
            state.cognition.working_memory.append(
                {
                    "role": "assistant",
                    "content": (
                        "The function updates each balance and removes a name when "
                        "its balance reaches zero from the"
                    ),
                }
            )
            return state

    engine = CognitiveEngine()
    engine._phases = [FullPhase()]
    thought = await engine._run_thinking_loop(
        AuraState.default(),
        "Explain this code fully.",
        ThinkingMode.REFLECTIVE,
        "desktop_ui",
        context={
            "desktop_quick_reply_contract": False,
            "cognitive_engine_required": True,
            "desktop_cognitive_engine_required": True,
            "live_mind_context_required": True,
            "live_mind_context": {
                **_ready_live_mind_context(),
                "required_subsystems_ok": True,
            },
            "visible_user_message": "Explain this code fully.",
        },
        is_background=False,
        timeout_s=30.0,
    )

    assert thought.metadata["reply_generation_incomplete"] is True
    assert thought.metadata["reply_generation_stop_reason"] == "max_tokens"
    assert thought.metadata["reply_generation_failure_reasons"] == (
        "truncated_tail",
    )


@pytest.mark.asyncio
async def test_full_phase_reply_exposes_latent_quality_truncation_without_worker_reason():
    from core.brain.cognitive_engine import CognitiveEngine
    from core.brain.types import ThinkingMode
    from core.state.aura_state import AuraState

    class FullPhase:
        async def execute(self, state, **_kwargs):
            state.response_modifiers.update(
                {
                    "live_mind_surface_control_receipt": {
                        "enabled": True,
                        "applied": True,
                        "live_mind_controls_bound": True,
                        "clean_user_surface_contract": True,
                        "surface_quality_gate_passed": True,
                        "surface_quality_gate_reasons": [],
                        "generation_stop_reason": "eos_or_stop_sequence",
                    },
                    "latent_cortex_final_output_quality": {
                        "passed": False,
                        "reasons": ["truncated_tail"],
                    },
                    "generation_failure_class": (
                        "final_output_quality_failed:truncated_tail"
                    ),
                    "response_path": "cognitive_engine_latent_owner_exhausted",
                }
            )
            state.cognition.working_memory.append(
                {
                    "role": "assistant",
                    "content": "It returns every position where the active count reaches the",
                }
            )
            return state

    engine = CognitiveEngine()
    engine._phases = [FullPhase()]
    thought = await engine._run_thinking_loop(
        AuraState.default(),
        "Explain this function and state its result.",
        ThinkingMode.REFLECTIVE,
        "desktop_ui",
        context={
            "desktop_quick_reply_contract": False,
            "cognitive_engine_required": True,
            "desktop_cognitive_engine_required": True,
            "live_mind_context_required": True,
            "live_mind_context": {
                **_ready_live_mind_context(),
                "required_subsystems_ok": True,
            },
            "visible_user_message": "Explain this function and state its result.",
        },
        is_background=False,
        timeout_s=30.0,
    )

    assert thought.metadata["reply_generation_incomplete"] is True
    assert thought.metadata["reply_generation_failure_reasons"] == (
        "truncated_tail",
    )
    assert thought.metadata["response_path"] == (
        "cognitive_engine_latent_owner_exhausted"
    )



def test_an_unstamped_snapshot_cannot_steer_generation():
    """think() accepts an arbitrary context. A snapshot that vouches only for
    its own `ready` flag would let anything reaching that entry point take
    control of temperature, top_p and recurrent depth."""
    from core.brain.cognitive_engine import _live_mind_controls_bound

    forged = dict(_ready_live_mind_context())
    forged.pop("aura_runtime_stamp", None)

    assert (
        _live_mind_controls_bound(
            forged,
            {"temperature": 0.7, "top_p": 0.9, "recurrent_loops": 2, "steering_alpha": 0.3},
        )
        is False
    )


def test_unhealthy_required_subsystems_cannot_steer_generation():
    from core.brain.cognitive_engine import _live_mind_controls_bound

    unhealthy = dict(_ready_live_mind_context())
    unhealthy["required_subsystems_ok"] = False

    assert (
        _live_mind_controls_bound(
            unhealthy,
            {"temperature": 0.7, "top_p": 0.9, "recurrent_loops": 2, "steering_alpha": 0.3},
        )
        is False
    )


def test_a_runtime_snapshot_with_healthy_subsystems_binds():
    from core.brain.cognitive_engine import (
        REQUIRED_LIVE_MIND_GENERATION_CONTROL_KEYS,
        _live_mind_controls_bound,
    )

    controls = dict.fromkeys(REQUIRED_LIVE_MIND_GENERATION_CONTROL_KEYS, 0.5)

    assert _live_mind_controls_bound(_ready_live_mind_context(), controls) is True
