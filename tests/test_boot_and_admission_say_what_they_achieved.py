"""Three things reported more than they had done.

`_initialized = True` meant setup RAN. It is True after a deferred boot and
after a RAM-guarded one, where no generation lane exists at all, and after an
eager boot whose warmup did not finish. Anything reading it as "inference
service ready" was reading a flag that never meant that.

A warmup deferral was logged and nothing else. Recording it as a degradation
would be wrong — the ladder declining to load a tier is designed backpressure,
and on this fail-closed subsystem a degradation escalates to CRITICAL (52 of
those came out of one soak). But a runtime that cannot warm its primary lane is
something health should be able to see, and a coalesced log line is not
evidence.

The affect snapshot read four values off two subsystems with bare `float()` and
an assumed percentage scale. A subsystem reporting 0..1 instead of 0..100
silently became 0.005; a NaN reached CRSM, the hedonic gradient and the
higher-order thought engine at once; and a partial failure left a mix of
observed values and constructor defaults that nothing downstream could tell
apart.
"""
from __future__ import annotations

import pytest

from core.brain.inference_gate import InferenceGate
from tests.source_contract import family_text


def _gate():
    gate = InferenceGate.__new__(InferenceGate)
    gate._warmup_deferral_counts = {}
    gate._last_cortex_warmup_deferral_log_at = 0.0
    return gate


# ─────────────────────────────── deferrals are counted, not just logged


def test_every_deferral_is_counted_even_when_the_log_is_coalesced():
    gate = _gate()

    for _ in range(5):
        gate._log_cortex_warmup_deferral("memory_pressure", context="foreground")

    receipt = gate.warmup_deferral_receipt()
    assert receipt["foreground:memory_pressure"]["count"] == 5


def test_causes_are_counted_separately():
    gate = _gate()

    gate._log_cortex_warmup_deferral("memory_pressure", context="foreground")
    gate._log_cortex_warmup_deferral("warmup_backoff", context="recovery")

    receipt = gate.warmup_deferral_receipt()
    assert set(receipt) == {"foreground:memory_pressure", "recovery:warmup_backoff"}


def test_a_deferral_records_when_it_first_and_last_happened():
    gate = _gate()

    gate._log_cortex_warmup_deferral("memory_pressure", context="foreground")
    gate._log_cortex_warmup_deferral("memory_pressure", context="foreground")

    entry = gate.warmup_deferral_receipt()["foreground:memory_pressure"]
    assert entry["first_at"] > 0
    assert entry["last_at"] >= entry["first_at"]


def test_the_receipt_is_a_copy():
    gate = _gate()
    gate._log_cortex_warmup_deferral("memory_pressure", context="foreground")

    gate.warmup_deferral_receipt()["foreground:memory_pressure"]["count"] = 999

    assert gate.warmup_deferral_receipt()["foreground:memory_pressure"]["count"] == 1


def test_deferrals_reach_the_status_snapshot():
    """A log line scrolls. Health reads this."""
    import inspect

    import core.brain.inference_gate as gate_mod

    source = inspect.getsource(gate_mod.InferenceGate.get_conversation_status)

    assert '"warmup_deferrals"' in source


def test_a_deferral_is_still_not_a_degradation():
    """Escalating designed backpressure to CRITICAL on a fail-closed subsystem
    is the noise this counter exists to avoid re-creating."""
    import ast
    import inspect

    import core.brain.inference_gate as gate_mod

    source = inspect.getsource(gate_mod.InferenceGate._log_cortex_warmup_deferral)
    tree = ast.parse(source.lstrip())

    assert not [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_record_inference_degradation"
    ]


# ─────────────────────────────── boot says which boot happened


def test_the_initialization_receipt_names_the_mode():

    import core.brain.inference_gate as gate_mod

    source = family_text(gate_mod)

    for mode in ("eager_warmup", "deferred_prewarm", "ram_admitted"):
        assert f'"{mode}"' in source

    assert "def initialization_receipt(" in source


def test_an_uninitialised_gate_has_an_empty_receipt():
    gate = InferenceGate.__new__(InferenceGate)

    assert gate.initialization_receipt() == {}


def test_initialize_uses_a_lockdep_visible_lock():
    """asyncio.Lock is invisible to lockdep, and boot is where an ABBA
    deadlock costs a runtime rather than a turn."""
    import inspect

    import core.brain.inference_gate as gate_mod

    source = inspect.getsource(gate_mod.InferenceGate.initialize)

    assert "checked_async_lock(" in source
    assert "asyncio.Lock()" not in source


# ─────────────────────────────── affect axes are validated


@pytest.mark.parametrize("raw", [float("nan"), float("inf"), "warm", None, [], {}])
def test_an_unusable_affect_axis_is_none_not_a_default(raw):
    """None is what lets the caller tell a measured value from a default."""
    assert InferenceGate._bounded_affect(raw, low=0.0, high=1.0) is None


def test_a_usable_axis_is_clamped_into_its_range():
    assert InferenceGate._bounded_affect(5.0, low=0.0, high=1.0) == 1.0
    assert InferenceGate._bounded_affect(-5.0, low=-1.0, high=1.0) == -1.0
    assert InferenceGate._bounded_affect(0.42, low=0.0, high=1.0) == 0.42


def test_the_public_reader_is_preferred_over_the_private_one():
    """Reaching past a public accessor into a subsystem's internals is how a
    rename becomes an outage."""

    import core.brain.inference_gate as gate_mod

    source = family_text(gate_mod)
    public_at = source.index('_circ and hasattr(_circ, "get_llm_params")')
    private_at = source.index('hasattr(_circ, "_sample_raw_axes")')

    assert public_at < private_at
    # And the private read only runs when the public one produced nothing.
    assert 'not _affect_observed["valence"]' in source


def test_defaulted_axes_are_named_in_the_receipt():

    import core.brain.inference_gate as gate_mod

    source = family_text(gate_mod)

    assert 'segments.omit(\n                "affect_axes",' in source
    assert "defaults used for: " in source


# ─────────────────────────────── mesh self-reports wait for trust


def test_only_neutral_mesh_outcomes_may_answer_before_trust():
    from core.brain.inference_gate import _MESH_PRE_TRUST_RATIONALES

    assert _MESH_PRE_TRUST_RATIONALES == {"acknowledgement", "resource_hold"}
    assert "self_report_from_state" not in _MESH_PRE_TRUST_RATIONALES


def test_a_deferred_mesh_self_report_is_recorded():

    import core.brain.inference_gate as gate_mod

    source = family_text(gate_mod)

    assert 'context["mesh_deferred_for_trust"]' in source


def test_a_raised_mesh_path_is_not_called_a_decline():
    """`handled=False` is the design working; an exception is a broken organism
    path, and one debug line called both the same thing."""

    import core.brain.inference_gate as gate_mod

    source = family_text(gate_mod)

    assert 'context["mesh_cognition_error"]' in source
    assert "fell through to the LLM path after the mesh cognition path raised" in source
