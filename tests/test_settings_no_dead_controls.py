"""tests/test_settings_no_dead_controls.py
==============================================
Backstop for the dead-control finding (see docs/SETTINGS_WIRING_AUDIT.md): the
settings panel persists every toggle but the runtime read almost none, so most
controls looked functional and did nothing. This test forces a conscious
classification of EVERY settings key — wired, frontend-only, or known-dead
(tracked debt) — so a NEW setting cannot be added as a silent no-op, and the
known-dead set cannot grow unnoticed.

When you wire a known-dead setting, move it from KNOWN_DEAD to WIRED. When you
add a new setting, classify it here (and wire it, or justify it as frontend-only)
or this test fails.
"""
from __future__ import annotations

from interface.routes.settings import _RUNTIME_MODE_KEYS, SCHEMA

# Settings the runtime actually enforces today (verified by other tests).
WIRED = {
    "model.local_path",
    "model.deep_path",
    "voice.input_enabled",
    "voice.output_enabled",
    "voice.output_rate",
    "permissions.camera",
    "permissions.screen",
    "autonomy.actions_enabled",
    "autonomy.level",
    "autonomy.proactive_messaging",
    "autonomy.self_modification",
    "governance.approval_mode",
    "learning.auto_enrichment_enabled",
    "learning.reflection_enabled",
    "memory.retention_days",
    "privacy.mode",
    "safety.safe_mode",
    "dev.developer_mode",
    "notify.enabled",
    "notify.quiet_hours_start",
    "notify.quiet_hours_end",
}

# Legitimately client-side only — the desktop/web shell renders these; there is
# nothing for the Python runtime to enforce.
FRONTEND_ONLY = {
    "theme.mode",
    "theme.reduced_motion",
    "voice.auto_listen",
    # The first-run wizard sets it when it finishes and the desktop reads it
    # to decide whether to offer setup. Nothing in the Python runtime enforces
    # it, and nothing should: it is a fact about what this person has already
    # been shown.
    "onboarding.completed",
}

# Tracked debt: persisted but NOT yet enforced by the runtime. Each needs a
# bridge+consumer+test (see docs/SETTINGS_WIRING_AUDIT.md). Shrink this set as
# settings get wired; do NOT grow it without a deliberate decision.
KNOWN_DEAD = {
    "permissions.files_workspace",
    "memory.review_window",
    "dev.diagnostics_enabled",
}


def test_every_setting_is_classified():
    """No setting may be unclassified — a new one must be wired or justified."""
    classified = WIRED | FRONTEND_ONLY | KNOWN_DEAD
    schema_keys = {s.key for s in SCHEMA}
    unclassified = schema_keys - classified
    assert not unclassified, (
        f"New/unclassified settings: {sorted(unclassified)}. Wire them (preferred), "
        "mark frontend-only, or add to KNOWN_DEAD in docs/SETTINGS_WIRING_AUDIT.md."
    )
    # And no stale classifications pointing at removed keys.
    stale = classified - schema_keys
    assert not stale, f"Classifications for keys no longer in SCHEMA: {sorted(stale)}"


def test_classification_buckets_are_disjoint():
    assert not (WIRED & FRONTEND_ONLY)
    assert not (WIRED & KNOWN_DEAD)
    assert not (FRONTEND_ONLY & KNOWN_DEAD)


def test_wired_safety_controls_are_actually_bridged():
    # safe_mode is the emergency containment switch: mutable, and therefore it
    # MUST reconfigure the resident orchestrator immediately rather than at the
    # next tick. That is what the runtime-mode bridge is for.
    assert "safety.safe_mode" in _RUNTIME_MODE_KEYS
    for key in _RUNTIME_MODE_KEYS:
        assert key in WIRED, f"{key} drives runtime but isn't marked WIRED"


def test_the_agency_invariant_is_immutable_rather_than_bridged():
    """autonomy.level is not a kill switch, and must not become one.

    This test used to require autonomy.level in the runtime-mode bridge, from
    when it was conceived as the second kill switch. The design deliberately
    changed: it is declared ``mutable=False`` and read at the authority gate,
    so operators can constrain particular external effects or enter emergency
    safe mode, but cannot convert Aura's agency into a user-selected operating
    level.

    Bridging it would mean it could be set at runtime, which is precisely what
    immutability forbids — so asserting immutability is the STRONGER contract,
    not a relaxation. It is also not a dead control: read_at_gate is its wiring.
    """
    level = next(s for s in SCHEMA if s.key == "autonomy.level")
    assert level.mutable is False, "the agency invariant became operator-settable"
    assert level.apply_mode == "read_at_gate"
    assert "autonomy.level" not in _RUNTIME_MODE_KEYS, (
        "an immutable invariant must not be in the runtime-mode bridge"
    )
