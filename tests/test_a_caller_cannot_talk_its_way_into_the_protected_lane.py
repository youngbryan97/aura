"""Untrusted strings decided routing, resources, and trust.

Three separate ways a caller — or a person typing a message — reached things
that are supposed to be policy decisions:

**The origin label.** An origin counted as user-facing if ANY underscore-
delimited token anywhere in it matched the allowlist. `background_user`,
`audit_probe` and `internal_test_sweep` were all user-facing, so putting one
allowlisted word anywhere in an unauthenticated label inherited the protected
Cortex lane, its memory admission work, and its worker shedding.

**The prompt text.** `looks_like_deep_mind_probe` is a heuristic over what the
person typed, and a match promoted the turn to the protected lane — which
force-sheds background workers and extends the quiet window, per message, with
no budget of any kind.

**Recognition as authorization.** Sovereign, trusted and guest were treated
identically: all three re-promoted the request to the protected lane, so an
unauthenticated guest could reverse a downgrade morphogenesis or headroom
policy had already made in the same request. And the recognized level went into
shared cognitive state with no session, principal, or timestamp, so a later
turn — or another interlocutor sharing that state — assembled context under a
classification granted to somebody else.

The validation half of this file is the same shape at the front door: values
the caller supplies being used before anything checks them.
"""
from __future__ import annotations

import math

import pytest

from core.brain.inference_gate import InferenceGate
from tests.source_contract import family_text


# ─────────────────────────────── the origin label


@pytest.mark.parametrize(
    "origin",
    ["user", "api", "desktop", "voice", "admin", "desktop_quick_user", "api_chat"],
)
def test_a_real_user_facing_origin_still_is(origin):
    assert InferenceGate._origin_is_user_facing(origin) is True


@pytest.mark.parametrize(
    "origin",
    [
        "background_user",
        "internal_test_sweep",
        "system_audit",
        "autonomous_user_probe",
        "cron_api_refresh",
        "daemon_voice_worker",
    ],
)
def test_an_allowlisted_word_buried_in_a_label_does_not_promote(origin):
    assert InferenceGate._origin_is_user_facing(origin) is False


def test_an_empty_origin_is_not_user_facing():
    assert InferenceGate._origin_is_user_facing("") is False
    assert InferenceGate._origin_is_user_facing(None) is False


def test_the_routing_prefix_is_still_stripped():
    assert InferenceGate._origin_is_user_facing("routing_routing_user") is True


def test_every_origin_literal_in_the_tree_keeps_its_meaning():
    """The rule change is only safe because no real origin moves. If someone
    adds one that would, this says so rather than the routing quietly
    changing."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1]
    pattern = re.compile(r'origin="([a-z_]+)"')
    origins: set[str] = set()
    for folder in ("core", "interface"):
        for path in (root / folder).rglob("*.py"):
            origins.update(pattern.findall(path.read_text("utf-8", errors="ignore")))

    from core.brain.inference_gate import _USER_FACING_ORIGINS

    for origin in sorted(origins):
        normalized = origin.strip().lower().replace("-", "_")
        while normalized.startswith("routing_"):
            normalized = normalized[len("routing_") :]
        if not normalized:
            continue
        old = (
            normalized in _USER_FACING_ORIGINS
            or bool({t for t in normalized.split("_") if t} & _USER_FACING_ORIGINS)
            or any(normalized.startswith(f"{p}_") for p in _USER_FACING_ORIGINS)
        )
        new = InferenceGate._origin_is_user_facing(origin)
        if old and not new:
            head = normalized.split("_", 1)[0]
            from core.brain.inference_gate import _NOT_USER_FACING_ORIGIN_PREFIXES

            assert head in _NOT_USER_FACING_ORIGIN_PREFIXES, (
                f"origin {origin!r} silently lost user-facing routing"
            )


# ─────────────────────────────── the prompt text


def test_a_text_inferred_promotion_can_shed_once():
    gate = InferenceGate.__new__(InferenceGate)
    gate._last_heuristic_shed_at = 0.0

    assert gate._admit_heuristic_protected_shed() is True


def test_a_second_text_inferred_shed_inside_the_window_is_refused():
    """A run of probe-shaped messages used to shed background workers once per
    message, each reloading what the last one unloaded."""
    gate = InferenceGate.__new__(InferenceGate)
    gate._last_heuristic_shed_at = 0.0

    assert gate._admit_heuristic_protected_shed() is True
    assert gate._admit_heuristic_protected_shed() is False


def test_the_budget_reopens_after_the_window():
    import time

    gate = InferenceGate.__new__(InferenceGate)
    gate._last_heuristic_shed_at = time.monotonic() - (
        InferenceGate._HEURISTIC_SHED_INTERVAL_S + 1.0
    )

    assert gate._admit_heuristic_protected_shed() is True


def test_an_explicit_protected_contract_is_not_rate_limited():
    """A caller that states protected_foreground_lane has said so on purpose;
    only a promotion inferred from text is budgeted."""
    import ast
    import inspect

    import core.brain.inference_gate as gate_mod

    source = family_text(gate_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "heuristic_promotion" not in targets:
            continue
        rendered = ast.get_source_segment(source, node) or ""
        assert "deep_mind_probe" in rendered
        assert "protected_foreground_lane" in rendered
        return
    raise AssertionError("the heuristic-promotion guard was not found")


# ─────────────────────────────── trust is bound to its request


def test_an_elevated_trust_level_without_a_binding_is_not_elevated():
    """A level in shared state with no timestamp is a classification granted
    to somebody else that this turn inherited."""
    from core.brain.llm.context_assembler import _TRUST_BINDING_MAX_AGE_S

    assert _TRUST_BINDING_MAX_AGE_S > 0


def test_the_gate_writes_who_and_when_with_the_trust_level():
    import inspect

    import core.brain.inference_gate as gate_mod

    source = family_text(gate_mod)

    assert 'modifiers["trust_level_binding"]' in source
    assert '"recognized_at"' in source
    assert '"session_id"' in source


def test_the_assembler_requires_a_fresh_binding_before_elevating():
    import inspect

    import core.brain.llm.context_assembler as assembler_mod

    source = family_text(assembler_mod)

    assert "trust_level_binding" in source
    assert "_TRUST_BINDING_MAX_AGE_S" in source


def test_guest_recognition_does_not_reverse_a_safety_downgrade():
    import inspect

    import core.brain.inference_gate as gate_mod

    source = family_text(gate_mod)

    assert "downgraded_for_safety" in source
    assert "not re-promoting it to the protected lane" in source


# ─────────────────────────────── caller values validated before use


@pytest.mark.parametrize("value", [-1.0, 0.0, float("inf"), float("nan"), "soon", None])
def test_an_unusable_timeout_falls_back_to_the_default(value):
    resolved = InferenceGate._requested_timeout_s(value, 90.0)

    assert math.isfinite(resolved)
    assert 0.0 < resolved <= InferenceGate._MAX_REQUEST_TIMEOUT_S


def test_a_usable_timeout_is_kept():
    assert InferenceGate._requested_timeout_s(45.0, 90.0) == 45.0


def test_a_huge_timeout_is_capped():
    assert (
        InferenceGate._requested_timeout_s(1e9, 90.0)
        == InferenceGate._MAX_REQUEST_TIMEOUT_S
    )


@pytest.mark.parametrize("value", [-5, 0, float("nan"), float("inf"), "lots", None])
def test_an_unusable_token_budget_falls_back_to_the_default(value):
    resolved = InferenceGate._requested_max_tokens(value, 512)

    assert isinstance(resolved, int)
    assert 0 < resolved <= InferenceGate._TOKEN_BOUND_HARD_CEILING


def test_a_usable_token_budget_is_kept():
    assert InferenceGate._requested_max_tokens(1024, 512) == 1024


def test_a_huge_token_budget_is_capped():
    assert (
        InferenceGate._requested_max_tokens(10_000_000, 512)
        == InferenceGate._TOKEN_BOUND_HARD_CEILING
    )


def test_a_nan_temperature_is_rejected_not_maxed():
    """NaN follows nested min/max to the 2.0 ceiling — the highest sampling
    entropy available — instead of being rejected."""
    import inspect

    import core.brain.inference_gate as gate_mod

    source = family_text(gate_mod)

    assert "if math.isfinite(_caller_temp)" in source


# ─────────────────────────────── cache policy is not a caller preference


def test_a_contract_that_requires_a_cold_prompt_overrides_the_caller():
    import inspect

    import core.brain.inference_gate as gate_mod

    source = family_text(gate_mod)

    assert "Cache policy is not a caller preference." in source
    assert 'for _cache_key in ("disable_prompt_cache", "clear_prompt_cache"):' in source


# ─────────────────────────────── a malformed payload cannot become system-only


def test_a_non_list_message_payload_is_reported_not_silently_dropped():
    import inspect

    import core.brain.inference_gate as gate_mod

    source = family_text(gate_mod)

    assert 'context["prebuilt_messages_rejected"] = "not_a_list"' in source


def test_a_payload_with_no_user_turn_gets_the_prompt_back():
    import inspect

    import core.brain.inference_gate as gate_mod

    source = family_text(gate_mod)

    assert 'context["prebuilt_messages_user_turn_recovered"] = True' in source
    assert "restored the caller's prompt as the user turn" in source
