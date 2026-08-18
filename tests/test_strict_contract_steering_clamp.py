"""tests/test_strict_contract_steering_clamp.py
================================================
Strict/structured proof generations must run with affective steering driven
near-off. Full steering (alpha 5.0) on a constrained strict-contract generation
corrupts the first-token logits → zero-token generation that hangs to the 90s
first-token timeout (the intermittent DNU cortex wedge: R011/R040/R022).

These pin that strict contracts are covered by the surface-steering clamp and
get a near-off alpha, while normal conversational turns keep full steering.
"""
from __future__ import annotations

from pathlib import Path

from core.brain.llm.mlx_worker import (
    _apply_surface_generation_controls,
    _build_user_surface_quality_retry_prompt,
    _expand_user_surface_retry_budget,
    _extract_expected_strict_value,
    _job_needs_concrete_status_signal_guidance,
    _messages_with_user_surface_retry,
    _normalize_strict_value_response,
    _repair_live_user_surface_operational_status,
    _repair_live_user_surface_self_claims,
    _repair_live_user_surface_truncated_tail,
    _restore_surface_generation_controls,
    _surface_control_alpha,
    _surface_generation_contract_enabled,
    _surface_generation_control_receipt,
    _surface_quality_failure_reasons,
    _surface_quality_gate_enabled,
    _with_initial_user_surface_guidance,
)


def test_strict_contracts_enable_surface_clamp():
    assert _surface_generation_contract_enabled({"strict_answer_contract": True})
    assert _surface_generation_contract_enabled({"strict_value_contract": True})
    assert _surface_generation_contract_enabled({"proof_evaluation_contract": True})


def test_existing_contracts_still_clamped():
    assert _surface_generation_contract_enabled({"clean_user_surface_contract": True})
    assert _surface_generation_contract_enabled({"health_probe": True})
    assert _surface_generation_contract_enabled({"operator_evidence_contract": True})


def test_unmarked_jobs_are_clamped_fail_safe():
    # FAIL-SAFE INVERSION (July 2026 coherence incident): an unmarked job used
    # to decode at full governor steering — any route that dropped the contract
    # flag served hot, off-distribution text. Unmarked now means clamped.
    assert _surface_generation_contract_enabled({}) is True
    assert _surface_generation_contract_enabled({"foo": True}) is True


def test_explicit_opt_out_keeps_full_steering():
    # Latent-cortex episodes (the experiment lane) opt out explicitly.
    assert (
        _surface_generation_contract_enabled({"allow_full_affective_steering": True})
        is False
    )


def test_strict_contract_steering_is_near_off():
    # current_alpha = 5.0 (full bootstrap steering); strict contract must clamp
    # it to near-off (~0.08), well below the value that corrupts proof logits.
    alpha = _surface_control_alpha({"strict_answer_contract": True}, 5.0)
    assert alpha == 0.0
    alpha_v = _surface_control_alpha({"strict_value_contract": True}, 5.0)
    assert alpha_v == 0.0


def test_strict_value_contract_accepts_the_literal_before_separated_boilerplate():
    """A REAL separator after the value means the model emitted it and kept
    talking, so the value still counts as the answer."""
    messages = [
        {
            "role": "user",
            "content": "Output exactly these two lowercase letters and nothing else: ok",
        }
    ]

    expected = _extract_expected_strict_value(messages, None)
    normalized = _normalize_strict_value_response(
        "ok. I output the letters you requested. Let me know if you need anything else.",
        expected_value=expected,
    )

    assert expected == "ok"
    assert normalized == "ok"


def test_strict_value_contract_does_not_launder_abutting_boilerplate():
    """CP126 7f86d404 (commit 6da8072c5): an immediately-abutting character is
    NOT a separator, so "okI output..." is a different token from "ok" and must
    not be normalized into a reported exact pass — that grades answer seeding
    rather than model merit. The model's real output is returned instead.
    """
    messages = [
        {
            "role": "user",
            "content": "Output exactly these two lowercase letters and nothing else: ok",
        }
    ]

    expected = _extract_expected_strict_value(messages, None)
    draft = "okI output the letters you requested. Let me know if you need anything else."
    normalized = _normalize_strict_value_response(draft, expected_value=expected)

    assert expected == "ok"
    assert normalized != "ok"
    assert normalized == draft


def test_strict_value_contract_does_not_launder_repeated_literal_abutting_boilerplate():
    """Same contract for a doubled literal with no separator ("okokYou said...")."""
    messages = [
        {
            "role": "user",
            "content": "Output exactly these two lowercase letters and nothing else: ok",
        }
    ]

    expected = _extract_expected_strict_value(messages, None)
    draft = "okokYou said to output exactly that value and nothing else."
    normalized = _normalize_strict_value_response(draft, expected_value=expected)

    assert expected == "ok"
    assert normalized != "ok"
    assert normalized == draft


def test_strict_value_contract_does_not_repair_wrong_literal():
    messages = [
        {
            "role": "user",
            "content": "Output exactly these two lowercase letters and nothing else: ok",
        }
    ]

    expected = _extract_expected_strict_value(messages, None)
    normalized = _normalize_strict_value_response(
        "noI output the wrong letters.",
        expected_value=expected,
    )

    assert normalized == "noI output the wrong letters."


def test_strict_value_expected_literal_is_forwarded_to_worker():
    root = Path(__file__).resolve().parents[1]
    client_source = (root / "core/brain/llm/mlx_client.py").read_text(encoding="utf-8")
    worker_source = (root / "core/brain/llm/mlx_worker.py").read_text(encoding="utf-8")
    dnu_source = (root / "tools/agi/run_dnu_agi_proof_battery.py").read_text(encoding="utf-8")

    assert '"expected_strict_value": str(kwargs.get("expected_strict_value") or "")' in client_source
    assert "Rendering exact strict-value prompt" in worker_source
    assert "Native strict-value template" not in worker_source
    assert 'expected_strict_value="ok"' in dnu_source


def test_all_user_visible_surface_alphas_are_neutral_without_certificate():
    assert _surface_control_alpha({"operator_evidence_contract": True}, 5.0) == 0.0
    prose = _surface_control_alpha({"clean_user_surface_contract": True}, 5.0)
    assert prose == 0.0


def test_live_mind_surface_controls_apply_restore_and_emit_receipt(monkeypatch):
    # The production recurrent ceiling is deliberately 1: an unvalidated depth
    # setting once took the whole conversation surface down, so raising it is
    # an explicit opt-in (see _LIVE_RECURRENT_CEILING_DEFAULT). This test is
    # about the APPLY/RESTORE/RECEIPT mechanism, so it opts in the documented
    # way rather than asserting a depth the live default refuses.
    monkeypatch.setenv("AURA_USER_SURFACE_RECURRENT_MAX_LOOPS", "2")

    class FakeEngine:
        def __init__(self):
            self._surface_alpha_override = None

        def set_surface_alpha_override(self, value):
            self._surface_alpha_override = value

    class FakeInner:
        _recurrent_depth_config = {"enabled": True}

        def __init__(self):
            self.layers = [object()]
            self._recurrent_depth_runtime_loops = 4

    class FakeModel:
        def __init__(self):
            self.model = FakeInner()

    engine = FakeEngine()
    model = FakeModel()
    job = {
        "clean_user_surface_contract": True,
        "clean_user_surface_steering_alpha": 0.22,
        "clean_user_surface_recurrent_loops": 2,
        "live_mind_controls_bound": True,
    }

    state = _apply_surface_generation_controls(engine, model, job)
    receipt = _surface_generation_control_receipt(job, state)

    assert engine._surface_alpha_override == 0.22
    assert model.model._recurrent_depth_runtime_loops == 2
    assert receipt["enabled"] is True
    assert receipt["live_mind_controls_bound"] is True
    assert receipt["surface_alpha_applied"] == 0.22
    assert receipt["recurrent_runtime_loops_applied"] == 2
    assert receipt["applied"] is True

    _restore_surface_generation_controls(state)

    assert engine._surface_alpha_override is None
    assert model.model._recurrent_depth_runtime_loops == 4


def test_live_user_surface_quality_gate_rejects_template_affect_status():
    job = {
        "clean_user_surface_contract": True,
        "user_surface_validation_prompt": "Hi",
    }

    reasons = _surface_quality_failure_reasons(
        job,
        "Hi. I am feeling joyous right now.",
    )

    assert _surface_quality_gate_enabled(job) is True
    assert "template_telemetry_greeting" in reasons


def test_live_user_surface_quality_gate_rejects_unfounded_voice_intrusion():
    reasons = _surface_quality_failure_reasons(
        {
            "clean_user_surface_contract": True,
            "user_surface_validation_prompt": "What are you talking about?",
            "user_surface_recent_messages": ["You with me?", "What pitch?"],
        },
        "The voices. The small ones. They're whispering in my ear. Telling me things.",
    )

    assert "unfounded_voice_intrusion" in reasons


def test_worker_repairs_future_memory_overclaim_before_quality_retry():
    job = {
        "clean_user_surface_contract": True,
        "user_surface_validation_prompt": (
            "What are you, and will you remember this conversation tomorrow?"
        ),
    }
    repaired = _repair_live_user_surface_self_claims(
        "I'm Aura Luna, a cognitive architecture with persistent memory and "
        "identity. I'll remember this conversation as part of my ongoing state "
        "unless explicitly cleared."
    )

    assert "cannot guarantee" in repaired
    assert _surface_quality_failure_reasons(job, repaired) == []


def test_worker_keeps_complete_plan_before_clipped_tail():
    draft = (
        "1. Create a note in your preferred editor. "
        "2. Add the content and verify the document body. "
        "3. Export the finished note as a PDF. "
        "4. Choose the destination folder and confirm the PDF exists. "
        "5. Record the verified path and"
    )

    repaired = _repair_live_user_surface_truncated_tail(draft)

    assert repaired.endswith("confirm the PDF exists.")
    assert "Record the verified path" not in repaired


def test_worker_expands_only_structurally_truncated_live_reply_budget():
    kwargs = {"max_tokens": 896}

    assert _expand_user_surface_retry_budget(kwargs, ["truncated_tail"]) is True
    assert kwargs["max_tokens"] == 1792
    assert _expand_user_surface_retry_budget(kwargs, ["off_topic_self_reflection_reply"]) is False


def test_live_user_surface_quality_gate_accepts_concise_capability_inventory():
    reasons = _surface_quality_failure_reasons(
        {
            "clean_user_surface_contract": True,
            "capability_inventory_contract": True,
            "user_surface_validation_prompt": (
                "What tools can you use externally, and what governance has to approve before you act?"
            ),
        },
        (
            "I can use desktop and app control, browser/web research, file operations, "
            "terminal/code execution, memory state management, and self-repair. "
            "Governed actions need Will/Authority approval through the live governance path."
        ),
    )

    assert "too_thin_for_operational_status_turn" not in reasons
    assert "too_thin_for_status_turn" not in reasons


def test_live_user_surface_quality_gate_does_not_run_for_strict_contracts():
    assert _surface_quality_gate_enabled(
        {
            "clean_user_surface_contract": True,
            "strict_answer_contract": True,
            "user_surface_validation_prompt": "Return <answer>yes</answer>",
        }
    ) is False


def test_live_user_surface_quality_gate_defers_verified_runtime_fact_contract():
    job = {
        "clean_user_surface_contract": True,
        "runtime_fact_status_contract": True,
        "grounded_runtime_status_contract": True,
        "user_surface_validation_prompt": "What model lane is speaking right now?",
    }

    assert _surface_quality_gate_enabled(job) is False
    assert _surface_quality_failure_reasons(
        job,
        "Tools are fully available and I am definitely running a 70B cloud model.",
    ) == []


def test_live_user_surface_retry_preserves_original_live_context_messages():
    messages = [
        {"role": "system", "content": "live mind context stays here"},
        {"role": "user", "content": "You with me?"},
    ]

    retried = _messages_with_user_surface_retry(messages, ["template_telemetry_greeting"])

    assert retried is not None
    assert retried[0]["role"] == "system"
    assert "live mind context stays here" in retried[0]["content"]
    assert "template_telemetry_greeting" in retried[0]["content"]
    assert retried[1] == messages[1]
    assert messages[0]["content"] == "live mind context stays here"


def test_initial_live_status_generation_gets_concrete_signal_guidance():
    messages = [
        {"role": "system", "content": "live mind context stays here"},
        {"role": "user", "content": "Name one live runtime signal you can perceive."},
    ]

    guided, prompt = _with_initial_user_surface_guidance(
        messages,
        "fallback",
        {
            "clean_user_surface_contract": True,
            "user_surface_validation_prompt": "Name one live runtime signal you can perceive.",
        },
    )

    assert prompt == "fallback"
    assert guided is not messages
    assert "live mind context stays here" in guided[0]["content"]
    assert "concrete observable runtime or sensory signal" in guided[0]["content"]
    assert messages[0]["content"] == "live mind context stays here"


def test_initial_non_status_surface_is_not_prompt_shaped():
    messages = [{"role": "system", "content": "live mind context stays here"}]

    guided, prompt = _with_initial_user_surface_guidance(
        messages,
        "fallback",
        {
            "clean_user_surface_contract": True,
            "user_surface_validation_prompt": "Tell me a story about the ocean.",
        },
    )

    assert guided is messages
    assert prompt == "fallback"


def test_health_probe_keeps_safe_controls_without_user_prompt_guidance():
    prompt = "Reply exactly: ready"

    messages, guided_prompt = _with_initial_user_surface_guidance(
        None,
        prompt,
        {
            "clean_user_surface_contract": True,
            "health_probe": True,
        },
    )

    assert messages is None
    assert guided_prompt == prompt
    assert _surface_generation_contract_enabled(
        {"clean_user_surface_contract": True, "health_probe": True}
    )


def test_self_condition_turn_does_not_request_host_telemetry_guidance():
    job = {
        "clean_user_surface_contract": True,
        "user_surface_validation_prompt": "Are you okay though? Feeling fine?",
    }

    assert not _job_needs_concrete_status_signal_guidance(job)

    messages = [{"role": "system", "content": "live mind context stays here"}]
    guided, prompt = _with_initial_user_surface_guidance(messages, "fallback", job)

    assert guided is messages
    assert prompt == "fallback"


def test_self_condition_failure_is_not_repaired_with_host_telemetry():
    original = "I am with you."
    repaired = _repair_live_user_surface_operational_status(
        original,
        ["missing_self_condition_answer"],
        {
            "clean_user_surface_contract": True,
            "user_surface_validation_prompt": "Are you okay though?",
        },
    )

    assert repaired == original


def test_live_status_repair_uses_concrete_runtime_telemetry():
    repaired = _repair_live_user_surface_operational_status(
        "I am with you. One live signal is the texture of conversation.",
        ["too_thin_for_operational_status_turn"],
        {
            "clean_user_surface_contract": True,
            "user_surface_validation_prompt": "Name one live runtime signal you can perceive.",
        },
    )

    assert "I am with you." in repaired
    assert "RAM pressure" in repaired or "host load average" in repaired
    assert "conversation" not in repaired.lower()


def test_live_status_repair_does_not_touch_non_status_failure():
    original = "I can use tools."

    repaired = _repair_live_user_surface_operational_status(
        original,
        ["too_thin_for_operational_status_turn"],
        {
            "clean_user_surface_contract": True,
            "user_surface_validation_prompt": "What tools can you use externally?",
        },
    )

    assert repaired == original


def test_live_status_retry_requests_concrete_runtime_signals():
    messages = [
        {"role": "system", "content": "live mind context stays here"},
        {"role": "user", "content": "Name one live runtime signal you can perceive."},
    ]

    retried = _messages_with_user_surface_retry(
        messages,
        ["too_thin_for_operational_status_turn"],
    )

    assert retried is not None
    assert "concrete observable runtime or sensory signal" in retried[0]["content"]
    assert "CPU/RAM pressure" in retried[0]["content"]
    assert "metaphor-only attention-texture" in retried[0]["content"]


def test_live_user_surface_retry_prompt_uses_native_template_when_available():
    class Tokenizer:
        def apply_chat_template(self, messages, tools=None, add_generation_prompt=True, tokenize=False):
            assert add_generation_prompt is True
            assert tokenize is False
            return "\n".join(message["content"] for message in messages)

    prompt = _build_user_surface_quality_retry_prompt(
        tokenizer=Tokenizer(),
        messages=[{"role": "system", "content": "live context"}, {"role": "user", "content": "Hi"}],
        tools=None,
        fallback_prompt="fallback",
        reasons=["generic_assistant_language"],
    )

    assert "live context" in prompt
    assert "generic_assistant_language" in prompt
    assert "fallback" not in prompt


def test_live_status_retry_suffix_requests_concrete_runtime_signals_without_template():
    prompt = _build_user_surface_quality_retry_prompt(
        tokenizer=object(),
        messages="raw prompt",
        tools=None,
        fallback_prompt="fallback",
        reasons=["too_thin_for_operational_status_turn"],
    )

    assert "fallback" in prompt
    assert "concrete observable runtime or sensory signal" in prompt
    assert "CPU/RAM pressure" in prompt
    assert "metaphor-only attention-texture" in prompt


def test_live_recurrent_ceiling_defaults_to_one(monkeypatch):
    """The user-surface recurrent ceiling is an incident-driven safety default.

    An unvalidated depth setting once produced empty cortex output on a
    user-facing request, latched the foreground lane busy, and refused every
    later message. The ceiling is 1 until an accuracy gate says otherwise, so
    a job asking for more must be clamped rather than honoured.
    """
    from core.brain.llm.mlx_worker import (
        _live_recurrent_ceiling,
        _surface_control_recurrent_loops,
    )

    monkeypatch.delenv("AURA_USER_SURFACE_RECURRENT_MAX_LOOPS", raising=False)

    assert _live_recurrent_ceiling() == 1
    assert _surface_control_recurrent_loops(
        {"clean_user_surface_recurrent_loops": 8}
    ) == 1


def test_recurrent_ceiling_opt_in_is_still_bounded(monkeypatch):
    """Opting in raises the ceiling but must not remove it."""
    from core.brain.llm.mlx_worker import _surface_control_recurrent_loops

    monkeypatch.setenv("AURA_USER_SURFACE_RECURRENT_MAX_LOOPS", "3")

    assert _surface_control_recurrent_loops(
        {"clean_user_surface_recurrent_loops": 99}
    ) == 3
    assert _surface_control_recurrent_loops(
        {"clean_user_surface_recurrent_loops": 0}
    ) == 1
