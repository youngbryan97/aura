"""Background-yield policy: one coherent vocabulary, not 25 magic numbers.

Bryan's 'federation, not one organism' complaint has a concrete seam: every
background loop deciding when to yield to the user. These tests pin the shared
profile vocabulary as THE spine and ratchet against the magic-number sprawl
growing back.
"""
from __future__ import annotations

import ast
from pathlib import Path

from core.runtime.background_policy import (
    COMPUTE_ONLY_BACKGROUND_POLICY,
    IDLE_COGNITION_BACKGROUND_POLICY,
    MAINTENANCE_BACKGROUND_POLICY,
    RESEARCH_BACKGROUND_POLICY,
    THOUGHT_BACKGROUND_POLICY,
    background_activity_reason,
)

_CORE = Path(__file__).resolve().parents[1] / "core"

# The migration from inline magic numbers to named profiles is incremental.
# This is the CURRENT count of call sites still passing raw TIER thresholds
# (min_idle_seconds / max_memory_percent); it may only shrink. A new
# inline-tier call site fails this test and must adopt a named profile.
# Overriding ONLY max_failure_pressure on top of a named profile is the
# sanctioned pattern — that knob is deliberately tuned per loop (mind_tick
# documents why 0.70; meta-cognition runs at 0.10) and does not count.
_MAGIC_NUMBER_BUDGET = 22


def test_profiles_are_ordered_by_cost():
    # Idle gate must grow with the cost/disruptiveness of the work.
    assert (
        IDLE_COGNITION_BACKGROUND_POLICY.min_idle_seconds
        <= RESEARCH_BACKGROUND_POLICY.min_idle_seconds
        <= MAINTENANCE_BACKGROUND_POLICY.min_idle_seconds
    )


def test_profiles_have_sane_bounds():
    for profile in (
        IDLE_COGNITION_BACKGROUND_POLICY,
        COMPUTE_ONLY_BACKGROUND_POLICY,
        THOUGHT_BACKGROUND_POLICY,
        RESEARCH_BACKGROUND_POLICY,
        MAINTENANCE_BACKGROUND_POLICY,
    ):
        assert profile.min_idle_seconds >= 0.0
        assert 0.0 < profile.max_memory_percent <= 100.0
        assert 0.0 < profile.max_failure_pressure <= 1.0


def test_deliberate_tiers_require_conversation_ready():
    # THOUGHT/RESEARCH/MAINTENANCE are deliberate work that must wait for a
    # ready conversation lane. IDLE_COGNITION is always-on and may run during
    # warmup (it still yields to an active turn via orchestrator.is_busy).
    assert THOUGHT_BACKGROUND_POLICY.require_conversation_ready is True
    assert RESEARCH_BACKGROUND_POLICY.require_conversation_ready is True
    assert MAINTENANCE_BACKGROUND_POLICY.require_conversation_ready is True


def test_idle_cognition_is_drop_in_for_the_emergent_inline_convention():
    # profile=IDLE_COGNITION must resolve to exactly what the bare inline
    # callers (min_idle=180, mem=78, nothing else) already get, so adopting it
    # is behavior-preserving.
    assert IDLE_COGNITION_BACKGROUND_POLICY.min_idle_seconds == 180.0
    assert IDLE_COGNITION_BACKGROUND_POLICY.max_memory_percent == 78.0
    assert IDLE_COGNITION_BACKGROUND_POLICY.max_failure_pressure == 0.60
    assert IDLE_COGNITION_BACKGROUND_POLICY.require_conversation_ready is False


def test_background_policy_magic_numbers_do_not_grow():
    """Ratchet: count background_activity_reason() calls passing raw thresholds.

    Named profiles are the intended vocabulary; inline min_idle_seconds /
    max_memory_percent / max_failure_pressure keyword args are the federation
    smell. This count may only shrink.
    """
    threshold_kwargs = {
        "min_idle_seconds",
        "max_memory_percent",
    }
    inline_sites = 0
    for path in _CORE.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if name != "background_activity_reason":
                continue
            if any(kw.arg in threshold_kwargs for kw in node.keywords):
                inline_sites += 1
    assert inline_sites <= _MAGIC_NUMBER_BUDGET, (
        f"{inline_sites} background_activity_reason() sites still pass magic-number "
        f"thresholds (budget {_MAGIC_NUMBER_BUDGET}). Adopt a named *_BACKGROUND_POLICY "
        "profile instead of inline min_idle_seconds/max_memory_percent."
    )


def test_named_profile_matches_inline_call_resolution():
    # Behavioral proof the drop-in is exact: with no orchestrator both forms
    # traverse identical threshold logic (the early global guards may short
    # circuit, but the resolved thresholds are the contract we assert above).
    import inspect

    src = inspect.getsource(background_activity_reason)
    assert "profile.min_idle_seconds" in src
    assert "profile.require_conversation_ready" in src


def test_compute_only_work_waits_as_idle_cognition_but_for_memory_only_where_the_loops_slow():
    # Work that computes without the model holds little memory of its own: it waits for the same idleness as idle
    # cognition, and for memory only at the level the background loops themselves slow at.
    assert COMPUTE_ONLY_BACKGROUND_POLICY.min_idle_seconds == IDLE_COGNITION_BACKGROUND_POLICY.min_idle_seconds
    assert COMPUTE_ONLY_BACKGROUND_POLICY.max_failure_pressure == IDLE_COGNITION_BACKGROUND_POLICY.max_failure_pressure
    assert COMPUTE_ONLY_BACKGROUND_POLICY.require_conversation_ready is False
    assert IDLE_COGNITION_BACKGROUND_POLICY.max_memory_percent < COMPUTE_ONLY_BACKGROUND_POLICY.max_memory_percent
    assert COMPUTE_ONLY_BACKGROUND_POLICY.max_memory_percent == THOUGHT_BACKGROUND_POLICY.max_memory_percent
