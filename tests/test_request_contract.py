"""The request context is typed, and says what it threw away.

The defect these cover is not exotic. Every policy flag reaching the
inference gate is read with a bare truthiness test, so any string turns it
ON — including the strings that mean OFF.
"""
from __future__ import annotations

import pytest

from core.brain.request_contract import (
    POLICY_FIELDS,
    REQUEST_FIELDS,
    normalize_tier,
    project_user_surface_resume_capability,
    strict_bool,
    validate_request_context,
)


@pytest.mark.parametrize("token", ["false", "False", "FALSE", "no", "off", "0", "n"])
def test_strings_that_mean_false_are_false(token):
    """bool('false') is True. That is the whole bug."""
    assert bool(token) is True, "the built-in behaviour this replaces"
    assert strict_bool(token) is False


@pytest.mark.parametrize("token", ["true", "True", "yes", "on", "1", "y"])
def test_strings_that_mean_true_are_true(token):
    assert strict_bool(token) is True


@pytest.mark.parametrize("value", ["maybe", "", "  ", "2", None, [], {}, object()])
def test_ambiguous_values_are_not_guessed(value):
    assert strict_bool(value) is None


def test_real_booleans_pass_through():
    assert strict_bool(True) is True
    assert strict_bool(False) is False


def test_retired_remote_fallback_flag_is_always_disabled():
    result = validate_request_context({"allow_cloud_fallback": "false"})
    assert result.context["allow_cloud_fallback"] is False
    assert not result.rejected

    attempted_enable = validate_request_context({"allow_cloud_fallback": True})
    assert attempted_enable.context["allow_cloud_fallback"] is False
    assert "allow_cloud_fallback" not in POLICY_FIELDS


def test_clearing_a_proof_requirement_actually_clears_it():
    result = validate_request_context({"proof_primary_lane_required": "no"})
    assert result.context["proof_primary_lane_required"] is False


def test_an_unreadable_flag_is_dropped_and_reported():
    result = validate_request_context({"benchmark_request": "sometimes"})
    assert "benchmark_request" not in result.context
    assert "benchmark_request" in result.rejected
    assert not result.clean


def test_tier_aliases_resolve_and_junk_is_rejected():
    assert normalize_tier("deep") == "secondary"
    assert normalize_tier("local") == "primary"
    assert normalize_tier("LOCAL_FAST") == "tertiary"
    assert normalize_tier("turbo") is None
    result = validate_request_context({"prefer_tier": "turbo"})
    assert "prefer_tier" not in result.context
    assert "prefer_tier" in result.rejected


def test_non_finite_sampling_values_are_rejected_not_clamped():
    result = validate_request_context(
        {"temperature": float("nan"), "top_p": float("inf")}
    )
    assert "temperature" in result.rejected
    assert "top_p" in result.rejected


def test_out_of_domain_floats_clamp_within_their_declared_range():
    result = validate_request_context({"top_p": 1.7, "min_p": -0.4})
    assert result.context["top_p"] == 1.0
    assert result.context["min_p"] == 0.0


def test_nonsense_integers_are_rejected_rather_than_clamped():
    """max_tokens=-5 clamped to 1 is a one-token answer, not a safe default."""
    result = validate_request_context({"max_tokens": -5, "top_k": 0})
    assert "max_tokens" not in result.context
    assert "top_k" not in result.context
    assert set(result.rejected) == {"max_tokens", "top_k"}


def test_numeric_strings_are_accepted():
    result = validate_request_context(
        {
            "temperature": "0.8",
            "max_tokens": "512",
            "user_surface_completion_floor": "384",
        }
    )
    assert result.context["temperature"] == pytest.approx(0.8)
    assert result.context["max_tokens"] == 512
    assert result.context["user_surface_completion_floor"] == 384


def test_undeclared_keys_are_reported_not_forwarded():
    result = validate_request_context({"secret_backdoor_flag": True})
    assert result.context == {}
    assert result.unknown == ["secret_backdoor_flag"]


def test_a_bare_string_stop_sequence_becomes_a_list():
    assert validate_request_context({"stop_sequences": "END"}).context[
        "stop_sequences"
    ] == ["END"]


def test_a_stop_sequence_list_with_non_strings_is_rejected():
    assert "stop_sequences" in validate_request_context(
        {"stop_sequences": ["ok", 7]}
    ).rejected


def test_messages_must_be_a_sequence():
    assert "messages" in validate_request_context({"messages": "not a list"}).rejected
    assert validate_request_context({"messages": [{"role": "user"}]}).clean


def test_completed_capability_evidence_crosses_as_opaque_evidence():
    evidence = {"schema": "aura.completed_capability_evidence.v1", "ok": True}

    result = validate_request_context({"completed_capability_evidence": evidence})

    assert result.clean
    assert result.context["completed_capability_evidence"] is evidence


def test_continuation_resume_handle_is_a_typed_bearer_capability():
    handle = "A9" * 16

    accepted = validate_request_context(
        {"user_surface_continuation_resume_handle": handle}
    )
    malformed = validate_request_context(
        {"user_surface_continuation_resume_handle": "resume-latest"}
    )

    assert accepted.context["user_surface_continuation_resume_handle"] == handle.lower()
    assert "user_surface_continuation_resume_handle" in malformed.rejected


def test_conversation_resume_handle_is_a_typed_bearer_capability():
    handle = "B7" * 16

    accepted = validate_request_context(
        {"user_surface_conversation_resume_handle": handle}
    )
    malformed = validate_request_context(
        {"user_surface_conversation_resume_handle": "most-recent-turn"}
    )

    assert accepted.context["user_surface_conversation_resume_handle"] == handle.lower()
    assert "user_surface_conversation_resume_handle" in malformed.rejected


def test_exact_resume_projection_selects_only_the_transaction_owner():
    continuation = "C1" * 16
    conversation = "D2" * 16

    ordinary = project_user_surface_resume_capability(
        {"user_surface_conversation_resume_handle": conversation},
        continuation_contract=False,
    )
    retry = project_user_surface_resume_capability(
        {"user_surface_continuation_resume_handle": continuation},
        continuation_contract=True,
    )

    assert ordinary.context == {
        "user_surface_conversation_resume_handle": conversation.lower()
    }
    assert retry.context == {
        "user_surface_continuation_resume_handle": continuation.lower()
    }


def test_exact_resume_projection_rejects_conflicts_wrong_transactions_and_backgrounds():
    continuation = "e3" * 16
    conversation = "f4" * 16

    conflict = project_user_surface_resume_capability(
        {
            "user_surface_continuation_resume_handle": continuation,
            "user_surface_conversation_resume_handle": conversation,
        },
        continuation_contract=False,
    )
    wrong_transaction = project_user_surface_resume_capability(
        {"user_surface_continuation_resume_handle": continuation},
        continuation_contract=False,
    )
    background = project_user_surface_resume_capability(
        {"user_surface_conversation_resume_handle": conversation},
        continuation_contract=False,
        user_surface=False,
    )

    assert conflict.context == {}
    assert set(conflict.rejected) == {
        "user_surface_continuation_resume_handle",
        "user_surface_conversation_resume_handle",
    }
    assert wrong_transaction.context == {}
    assert "user_surface_continuation_resume_handle" in wrong_transaction.rejected
    assert background.context == {}
    assert "user_surface_conversation_resume_handle" in background.rejected


def test_a_clean_context_reports_clean():
    result = validate_request_context(
        {"origin": "user", "max_tokens": 2048, "allow_cloud_fallback": False}
    )
    assert result.clean
    assert result.to_dict() == {"rejected": {}, "unknown": []}


def test_cognitive_mode_is_a_closed_typed_request_field():
    for mode in ("reactive", "deliberate", "dreaming", "dormant", "deep"):
        result = validate_request_context({"cognitive_mode": mode})
        assert result.clean
        assert result.context["cognitive_mode"] == mode

    rejected = validate_request_context({"cognitive_mode": "make-it-smart"})
    assert "cognitive_mode" in rejected.rejected


def test_rejection_callback_sees_every_rejection():
    seen = []
    validate_request_context(
        {"benchmark_request": "huh", "prefer_tier": "turbo"},
        on_rejection=lambda key, value, why: seen.append(key),
    )
    assert sorted(seen) == ["benchmark_request", "prefer_tier"]


def test_policy_fields_are_a_subset_of_declared_fields():
    assert POLICY_FIELDS <= set(REQUEST_FIELDS)
    assert POLICY_FIELDS, "the authority-relevant set must not be empty"


def test_every_policy_field_is_a_bool_or_a_tier():
    """A policy key with a free-form type is one an authority check cannot bind."""
    from core.brain.request_contract import Kind

    for name in POLICY_FIELDS:
        assert REQUEST_FIELDS[name].kind in (Kind.BOOL, Kind.TIER), name


def test_none_context_is_handled():
    assert validate_request_context(None).clean
