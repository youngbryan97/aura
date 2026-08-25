"""`generate` returned None for four different things, and None said nothing.

A background admission deferral, a proof contract that names Cortex, RAM
admission deferring a cold load, and critical memory pressure all exited with
`return None` after a log line. That is the same value the model returns when
it produces no text, so a caller could not tell "policy refused this" from
"the model said nothing" — and the two want opposite handling: one is retry
later, the other is a failure to investigate.

The return contract stays `str | None`, because every caller depends on it.
What changed is that None is no longer the whole message: a typed receipt goes
to the caller's own context, to the gate for health, and to the turn ledger,
whose terminal status is then a refusal rather than a turn that mysteriously
held no answer.

The second half of this file is the request clock. Every attempt used to open
a window of its own — the primary attempt, each repair at 30-60s, the
brainstem, the reflex, APIAdapter at 30s, then HealthRouter at another 30 — so
a caller asking for 45 seconds could wait several minutes while every
individual `wait_for` was "within budget".
"""
from __future__ import annotations

import ast
import inspect

import pytest

from core.brain.inference_gate import InferenceGate
from core.utils.deadlines import get_deadline


def _gate():
    gate = InferenceGate.__new__(InferenceGate)
    gate._last_refusal_receipt = {}
    return gate


# ────────────────────────────── the refusal is typed and reaches the caller


def test_a_refusal_reaches_the_caller_through_its_own_context():
    gate = _gate()
    context: dict = {}

    result = gate._refuse_generation(
        InferenceGate.REFUSAL_RESOURCE,
        "critical_memory_pressure",
        context=context,
        origin="desktop_user",
    )

    assert result is None, "the return contract changed"
    receipt = context["inference_refusal"]
    assert receipt["kind"] == InferenceGate.REFUSAL_RESOURCE
    assert receipt["reason"] == "critical_memory_pressure"
    assert receipt["origin"] == "desktop_user"


def test_the_gate_keeps_the_last_refusal_for_health():
    gate = _gate()

    gate._refuse_generation(
        InferenceGate.REFUSAL_DEFERRED, "foreground_quiet_window", context=None
    )

    assert gate.last_refusal_receipt()["reason"] == "foreground_quiet_window"


def test_the_receipt_is_a_copy_not_the_live_record():
    gate = _gate()
    gate._refuse_generation(InferenceGate.REFUSAL_DEFERRED, "reason", context=None)

    gate.last_refusal_receipt()["reason"] = "tampered"

    assert gate.last_refusal_receipt()["reason"] == "reason"


def test_an_unknown_retry_time_is_omitted_rather_than_zeroed():
    """A zero would read as "retry immediately", which is a claim nobody made."""
    gate = _gate()
    context: dict = {}

    gate._refuse_generation(
        InferenceGate.REFUSAL_DEFERRED, "reason", context=context, origin="x"
    )

    assert "retry_after_s" not in context["inference_refusal"]


def test_a_known_retry_time_is_carried():
    gate = _gate()
    context: dict = {}

    gate._refuse_generation(
        InferenceGate.REFUSAL_DEFERRED,
        "reason",
        context=context,
        retry_after_s=12.5,
    )

    assert context["inference_refusal"]["retry_after_s"] == 12.5


def test_the_turn_ledger_records_a_refusal_not_a_missing_answer():
    from core.runtime.turn_outcome import TurnOutcome, bind_turn

    gate = _gate()
    outcome = TurnOutcome(origin="user_chat")

    with bind_turn(outcome):
        gate._refuse_generation(
            InferenceGate.REFUSAL_PROOF_LANE, "primary_required", context=None
        )

    receipt = outcome.finalize()
    assert receipt.status.value == "refused"
    assert any(r["kind"] == "inference_refusal" for r in receipt.causal_receipts)


def test_no_turn_bound_is_not_an_error():
    """Background work and tools run with no turn; a refusal must still work."""
    gate = _gate()

    assert gate._refuse_generation(InferenceGate.REFUSAL_DEFERRED, "r", context=None) is None


def test_every_policy_exit_goes_through_the_refusal_receipt():
    """A bare `return None` added later would put the ambiguity straight back."""
    import core.brain.inference_gate as gate_mod

    source = inspect.getsource(gate_mod)
    tree = ast.parse(source)

    generate = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "generate"
        and node.args.args
        and [arg.arg for arg in node.args.args][:2] == ["self", "prompt"]
    )

    bare = [
        node.lineno
        for node in ast.walk(generate)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Constant)
        and node.value.value is None
    ]

    assert not bare, (
        f"generate still returns a bare None — indistinguishable from an empty "
        f"model answer — at line(s) {bare}"
    )


# ────────────────────────────── one clock for the whole request


def test_an_attempt_never_gets_more_than_the_request_has_left():
    deadline = get_deadline(10.0)

    assert InferenceGate._window_within(deadline, 30.0) <= 10.0


def test_an_attempt_asking_for_less_keeps_its_own_window():
    deadline = get_deadline(100.0)

    assert InferenceGate._window_within(deadline, 5.0) == 5.0


def test_an_exhausted_request_gets_no_window_at_all():
    """Not a small window — none. An attempt that starts after the caller's
    deadline cannot deliver anything the caller is still waiting for, and it
    holds the model lane while it fails."""
    deadline = get_deadline(0.0)

    assert InferenceGate._window_within(deadline, 30.0) == 0.0


def test_an_unbounded_request_leaves_the_attempt_alone():
    deadline = get_deadline(None)

    assert InferenceGate._window_within(deadline, 30.0) == 30.0


def test_a_missing_deadline_object_does_not_break_dispatch():
    assert InferenceGate._window_within(None, 30.0) == 30.0


@pytest.mark.parametrize(
    "variable",
    ["primary_deadline", "retry_deadline", "fallback_deadline", "reflex_deadline"],
)
def test_every_local_attempt_is_capped_by_the_request_deadline(variable):
    import core.brain.inference_gate as gate_mod

    source = inspect.getsource(gate_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if variable not in targets:
            continue
        rendered = ast.get_source_segment(source, node) or ""
        assert "_window_within" in rendered, (
            f"{variable} opens a window outside the caller's budget"
        )
        return
    raise AssertionError(f"{variable} was not found")


def test_no_attempt_runs_on_its_own_clock():
    """Every attempt window comes out of the caller's budget.

    This used to name cloud_window_s and router_window_s. The remote
    inference provider was removed, so both identifiers went with it and the
    test pinned two names instead of the property they existed to protect —
    that no attempt opens a clock the caller never agreed to.
    """
    import core.brain.inference_gate as gate_mod

    source = inspect.getsource(gate_mod)

    assert "def _window_within(" in source, (
        "the helper that bounds an attempt by the request deadline is gone"
    )
    assert source.count("_window_within(") > 1, (
        "nothing calls the helper that bounds an attempt"
    )
    # No hard budget survives anywhere in the gate.
    assert "timeout=30.0," not in source, (
        "an attempt still runs on its own thirty-second clock"
    )


def test_the_remote_inference_provider_is_gone():
    """It was removed; nothing should be able to reintroduce it quietly."""
    import core.brain.inference_gate as gate_mod

    source = inspect.getsource(gate_mod)
    for name in ("cloud_window_s", "router_window_s", "_call_cloud"):
        assert name not in source, f"{name} came back without a window"


# ────────────────────────── a draft handed downstream is not yet an answer


def test_a_repairable_draft_opens_an_obligation_with_the_draft_hash():
    """The endpoint was recorded as if the answer came from it, with no id, no
    hash, no acceptance and no postcondition behind the promise of repair."""
    gate = InferenceGate.__new__(InferenceGate)
    gate._repair_obligations = {}

    obligation_id = gate._open_repair_obligation(
        label="Cortex", draft="half an answer", reasons=("truncated_tail",)
    )

    assert obligation_id.startswith("repair_")
    [record] = gate.open_repair_obligations()
    assert record["endpoint"] == "Cortex"
    assert record["draft_chars"] == len("half an answer")
    assert record["reasons"] == ["truncated_tail"]
    assert len(record["draft_sha256"]) == 64


def test_the_endpoint_attribution_is_marked_provisional():
    gate = InferenceGate.__new__(InferenceGate)
    gate._last_user_generation_provisional = False

    gate._record_user_generation_endpoint("Cortex", provisional=True)

    assert gate._last_user_generation_provisional is True


def test_an_ordinary_answer_is_not_provisional():
    gate = InferenceGate.__new__(InferenceGate)
    gate._last_user_generation_provisional = True

    gate._record_user_generation_endpoint("Cortex")

    assert gate._last_user_generation_provisional is False


def test_discharging_an_unknown_obligation_reports_false():
    """Something claiming to have repaired a draft this gate never handed out
    is worth knowing about."""
    gate = InferenceGate.__new__(InferenceGate)
    gate._repair_obligations = {}

    assert (
        gate.discharge_repair_obligation("repair_nope", repaired_text="x", accepted=True)
        is False
    )


def test_a_discharged_obligation_records_whether_anything_changed():
    gate = InferenceGate.__new__(InferenceGate)
    gate._repair_obligations = {}
    obligation_id = gate._open_repair_obligation(
        label="Cortex", draft="half an answer", reasons=()
    )

    assert gate.discharge_repair_obligation(
        obligation_id, repaired_text="a whole answer", accepted=True
    )
    assert gate.open_repair_obligations() == []


def test_serving_the_turn_closes_the_obligation_with_a_postcondition():
    """The ledger holds the draft hash and the served text; comparing them IS
    the postcondition the promise of repair was missing."""
    from core.runtime.turn_outcome import TurnOutcome

    gate = InferenceGate.__new__(InferenceGate)
    gate._repair_obligations = {}
    outcome = TurnOutcome(origin="user_chat")

    from core.runtime.turn_outcome import bind_turn

    with bind_turn(outcome):
        gate._open_repair_obligation(label="Cortex", draft="half an answer", reasons=())

    outcome.mark_served("a whole, repaired answer")
    receipt = outcome.finalize()

    discharged = [r for r in receipt.causal_receipts if r["kind"] == "repair_discharged"]
    assert discharged, "the repair promise was never closed"
    assert discharged[0]["payload"]["changed"] is True


def test_an_unrepaired_draft_that_goes_out_unchanged_is_recorded_as_such():
    from core.runtime.turn_outcome import TurnOutcome, bind_turn

    gate = InferenceGate.__new__(InferenceGate)
    gate._repair_obligations = {}
    outcome = TurnOutcome(origin="user_chat")

    with bind_turn(outcome):
        gate._open_repair_obligation(label="Cortex", draft="half an answer", reasons=())

    outcome.mark_served("half an answer")
    receipt = outcome.finalize()

    discharged = [r for r in receipt.causal_receipts if r["kind"] == "repair_discharged"]
    assert discharged[0]["payload"]["changed"] is False, (
        "a flawed draft went out unchanged and the record could not say so"
    )


# ────────────────────────── a refusal undoes the temporal anchor


def test_a_refusal_restores_the_temporal_anchor():
    """on_inference_start anchors while parameters are still being assembled,
    so a later refusal had moved temporal state for an inference that never
    ran."""
    from core.consciousness.temporal_continuity import TemporalContinuityEngine

    continuity = TemporalContinuityEngine()
    original = continuity._anchor_time

    continuity.on_inference_start()
    assert continuity._anchor_time != original

    assert continuity.on_inference_abandoned("refused") is True
    assert continuity._anchor_time == original
    assert continuity.abandoned_inference_count() == 1


def test_abandoning_twice_restores_once():
    from core.consciousness.temporal_continuity import TemporalContinuityEngine

    continuity = TemporalContinuityEngine()
    continuity.on_inference_start()

    assert continuity.on_inference_abandoned("refused") is True
    assert continuity.on_inference_abandoned("refused again") is False


def test_a_completed_inference_cannot_be_abandoned():
    from core.consciousness.temporal_continuity import TemporalContinuityEngine

    continuity = TemporalContinuityEngine()
    before_start = continuity._anchor_time
    continuity.on_inference_start()
    # on_inference_complete re-anchors, which is its job.
    continuity.on_inference_complete()
    after_complete = continuity._anchor_time

    assert continuity.on_inference_abandoned("too late") is False
    assert continuity._anchor_time == after_complete
    assert continuity._anchor_time != before_start


def test_the_refusal_path_restores_the_anchor():
    import ast
    import inspect

    from core.brain.inference_gate import InferenceGate

    source = inspect.getsource(InferenceGate._refuse_generation)
    tree = ast.parse(source.lstrip())

    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "on_inference_abandoned"
        for node in ast.walk(tree)
    ), "a refusal leaves the temporal anchor advanced for an inference that never ran"
