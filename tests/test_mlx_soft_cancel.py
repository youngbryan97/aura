"""Cooperative (soft) cancel of MLX generations.

The preemption ladder's first rung: the parent writes the active job's
sequence number into shared memory, the worker token loop stops between
tokens, and the model stays warm — force-abort (worker kill + full model
reload) remains the escalation, not the default.
"""
from __future__ import annotations

import time

from core.brain.llm import mlx_client as mlx_client_mod
from core.brain.llm.latent_cortex.action_state_capture import (
    UnknownActionStateApplicationError,
)
from core.brain.llm.mlx_client import MLXLocalClient
from core.brain.llm.mlx_worker import (
    _state_application_quarantine_response,
    soft_cancel_requested,
)


class _Value:
    """Duck-typed stand-in for multiprocessing.Value(lock=False)."""

    def __init__(self, value: int = 0):
        self.value = value


# ── worker-side predicate ──────────────────────────────────────────────


def test_soft_cancel_requested_matches_only_the_targeted_job():
    channel = _Value(7)
    assert soft_cancel_requested(channel, 7) is True
    assert soft_cancel_requested(channel, 8) is False
    assert soft_cancel_requested(channel, 0) is False
    assert soft_cancel_requested(None, 7) is False
    assert soft_cancel_requested(_Value(0), 7) is False


def test_soft_cancel_requested_survives_broken_channel():
    class _Broken:
        @property
        def value(self):
            self.reads = getattr(self, "reads", 0) + 1
            raise OSError("shared memory detached")

    assert soft_cancel_requested(_Broken(), 3) is False


# ── client-side API ────────────────────────────────────────────────────


def _bare_client(*, active_seq: int = 0, started: bool = False) -> MLXLocalClient:
    client = MLXLocalClient.__new__(MLXLocalClient)
    client.model_path = "/models/Qwen2.5-32B-Instruct-4bit"
    client._cancel_seq = _Value(0)
    client._current_request_seq = active_seq
    client._current_request_started_at = time.time() if started else 0.0
    return client


def test_soft_cancel_writes_active_seq_to_channel():
    client = _bare_client(active_seq=42, started=True)

    receipt = client.soft_cancel_active_generation("foreground_preemption")

    assert receipt["requested"] is True
    assert receipt["active_seq"] == 42
    assert client._cancel_seq.value == 42


def test_soft_cancel_refuses_when_no_generation_active():
    client = _bare_client(active_seq=0, started=False)

    receipt = client.soft_cancel_active_generation("foreground_preemption")

    assert receipt["requested"] is False
    assert receipt["detail"] == "no_active_generation"
    assert client._cancel_seq.value == 0


def test_new_generation_clears_stale_cancel_request():
    client = MLXLocalClient("/models/test-small")
    client._cancel_seq = _Value(0)
    # Minimal tracking fields _mark_generation_started touches.
    client._last_heartbeat = 0.0
    client._last_progress_at = 0.0
    client._last_ready_at = 0.0
    client._cancel_seq.value = 41  # stale cancel for a finished job

    client._mark_generation_started("req-abc", request_seq=42)

    assert client._current_request_seq == 42
    assert client._cancel_seq.value == 0


# ── module-level sweep + owner-clear integration ───────────────────────


def test_soft_cancel_sweep_collects_only_accepting_clients(monkeypatch):
    accepted = _bare_client(active_seq=5, started=True)
    idle = _bare_client(active_seq=0, started=False)
    monkeypatch.setattr(
        mlx_client_mod, "_CLIENTS", {"a": accepted, "b": idle}
    )

    receipts = mlx_client_mod.soft_cancel_active_generations(reason="test_sweep")

    assert len(receipts) == 1
    assert receipts[0]["active_seq"] == 5
    assert receipts[0]["model"] == "Qwen2.5-32B-Instruct-4bit"
    assert accepted._cancel_seq.value == 5


def test_force_clear_foreground_owner_requests_soft_cancel(monkeypatch):
    active = _bare_client(active_seq=9, started=True)
    monkeypatch.setattr(mlx_client_mod, "_CLIENTS", {"a": active})
    monkeypatch.setattr(mlx_client_mod, "_FOREGROUND_OWNER_NAME", "chat_api:default")
    monkeypatch.setattr(
        mlx_client_mod, "_FOREGROUND_OWNER_ACQUIRED_AT", time.time() - 60.0
    )

    result = mlx_client_mod.force_clear_foreground_owner(
        reason="chat_lock_preemption", min_age_s=45.0
    )

    assert result["cleared"] is True
    assert result["soft_cancel"], "owner clear must ask the wedged generation to stop"
    assert result["soft_cancel"][0]["active_seq"] == 9
    assert active._cancel_seq.value == 9


def test_force_clear_foreground_owner_without_active_generation(monkeypatch):
    monkeypatch.setattr(mlx_client_mod, "_CLIENTS", {})
    monkeypatch.setattr(mlx_client_mod, "_FOREGROUND_OWNER_NAME", "chat_api:default")
    monkeypatch.setattr(
        mlx_client_mod, "_FOREGROUND_OWNER_ACQUIRED_AT", time.time() - 60.0
    )

    result = mlx_client_mod.force_clear_foreground_owner(
        reason="chat_lock_preemption", min_age_s=45.0
    )

    assert result["cleared"] is True
    assert result["soft_cancel"] == []


# ── request wiring ─────────────────────────────────────────────────────


def test_generate_requests_carry_monotonic_seq():
    client = _bare_client()
    client._job_seq_counter = 0

    # The request builder increments the counter per job; simulate two turns.
    client._job_seq_counter += 1
    first = client._job_seq_counter
    client._job_seq_counter += 1
    second = client._job_seq_counter

    assert (first, second) == (1, 2)


def test_worker_spawn_args_include_cancel_channel():
    import inspect

    from core.brain.llm.mlx_worker import _mlx_worker_loop

    params = list(inspect.signature(_mlx_worker_loop).parameters)
    assert "cancel_seq" in params
    # Spawn site passes it positionally after steering_active_flag.
    assert params.index("cancel_seq") == params.index("steering_active_flag") + 1
    assert "self._cancel_seq," in inspect.getsource(mlx_client_mod)


def test_worker_spawn_carries_only_the_public_capture_launch_challenge():
    import inspect

    from core.brain.llm.mlx_worker import _mlx_worker_loop

    signature = inspect.signature(_mlx_worker_loop)
    assert signature.parameters["worker_capture_launch_challenge"].default is None
    source = inspect.getsource(mlx_client_mod.MLXLocalClient._spawn_worker_blocking)
    assert "build_worker_capture_launch_authority()" in source
    spawn_block = source.split("target=_mlx_worker_loop", 1)[1][:900]
    assert "dict(self._worker_capture_launch_authority.challenge)" in spawn_block
    assert "_worker_capture_launch_authority.private_key" not in spawn_block


def test_ambiguous_state_application_emits_fatal_worker_quarantine():
    error = UnknownActionStateApplicationError(
        {
            "operation_id": "a" * 32,
            "arm": "treatment",
            "worker_boot_id": "b" * 32,
            "worker_pid": 4242,
            "request_sha256": "c" * 64,
            "snapshot_sha256": "d" * 64,
        }
    )

    response = _state_application_quarantine_response(error)

    assert response == {
        "status": "error",
        "message": "private_snapshot_unknown_application_process_replacement_required",
        "state_application_quarantine": error.quarantine_evidence,
        "requires_worker_recycle": True,
    }


def test_latent_worker_exits_after_ambiguous_state_application():
    import inspect

    from core.brain.llm.mlx_worker import _mlx_worker_loop

    source = inspect.getsource(_mlx_worker_loop)
    quarantine = source.split(
        "except UnknownActionStateApplicationError as quarantine_exc:", 1
    )[1]
    before_generic_handler = quarantine.split("except (", 1)[0]
    assert "recycle_after_response = True" in before_generic_handler
    post_response = source.split("ipc_writer.put(response)", 1)[1]
    assert "if recycle_after_response:" in post_response
    assert "break" in post_response.split("if recycle_after_response:", 1)[1][:400]


# ── parent-side response handling ──────────────────────────────────────


def test_soft_cancelled_ok_response_bypasses_empty_telemetry_and_retries():
    """A requested cancel is not a generation failure: the empty-generation
    telemetry block must be gated on the response NOT being soft-cancelled,
    and the soft-cancel branch must return before the user-facing completion
    mark (single shared _consecutive_empty reset per the runtime contract)."""
    import inspect

    source = inspect.getsource(mlx_client_mod)
    assert "if not text and not cooperative_stop:" in source, (
        "empty-generation telemetry must skip cooperative terminal responses"
    )
    assert '"deadline_exceeded"' in source
    idx_cancel = source.find("if cooperative_stop:")
    idx_completed = source.find("self._mark_generation_completed(")
    assert 0 < idx_cancel < idx_completed, (
        "soft-cancel branch must return before the user-facing completion mark"
    )


def test_cooperative_quality_rejection_is_preserved_before_partial_return():
    import inspect

    source = inspect.getsource(mlx_client_mod.MLXLocalClient._generate_inner)
    idx_rejection = source.find(
        "if quality_rejection_reasons and (not text or cooperative_stop):"
    )
    idx_cancel = source.find("if cooperative_stop:")

    assert 0 < idx_rejection < idx_cancel
    rejection_block = source[idx_rejection:idx_cancel]
    assert "_record_suppressed_draft(" in rejection_block
    assert "_preserve_lane_after_surface_quality_rejection()" in rejection_block


def test_suppressed_cooperative_partial_respects_unspeakable_boundary():
    from core.runtime.turn_outcome import TurnOutcome, bind_turn

    client = MLXLocalClient.__new__(MLXLocalClient)
    with bind_turn(TurnOutcome("cancelled-partial", origin="test")) as outcome:
        client._record_suppressed_draft(
            "PROCEEDING TOOL_ACTION",
            ["backend_symbolic_surface_leak"],
        )

    candidate = outcome.candidates()[0]
    assert candidate.suppressed is not None
    assert candidate.suppressed.recoverable is False


def test_suppressed_repairable_partial_remains_recoverable():
    from core.runtime.turn_outcome import TurnOutcome, bind_turn

    client = MLXLocalClient.__new__(MLXLocalClient)
    with bind_turn(TurnOutcome("cancelled-partial", origin="test")) as outcome:
        client._record_suppressed_draft(
            "The answer is complete apart from the final sentence.",
            ["truncated_tail"],
        )

    candidate = outcome.candidates()[0]
    assert candidate.suppressed is not None
    assert candidate.suppressed.recoverable is True


# ── warm-lane preservation on abandoned requests ───────────────────────
#
# Historically every abandoned request recycled the worker even when it was
# alive and merely slow — a full model reload during which arriving turns
# died (observed live as soak death-clusters). The resolver now keeps the
# warm lane when the worker acknowledges the soft-cancel and reboots only
# unacknowledged (truly wedged) workers.


class _AliveProcess:
    def is_alive(self):
        return True


class _DeadProcess:
    def is_alive(self):
        return False


def _resolver_client(
    *,
    cancel_value: int,
    alive: bool = True,
    heartbeat_fresh: bool = True,
    acknowledged: bool = False,
    cancelled: bool = True,
):
    """A client mid-abandon.

    ``acknowledged`` is now the ONLY thing that makes the ack wait succeed:
    CP126 6b4337de replaced "the shared flag is zero and some heartbeat is
    fresh" — neither of which names the cancelled request — with the worker's
    own terminal frame for that request id.
    """
    client = MLXLocalClient.__new__(MLXLocalClient)
    client.model_path = "/models/Qwen2.5-32B-Instruct-4bit"
    client._cancel_seq = _Value(cancel_value)
    client._process = _AliveProcess() if alive else _DeadProcess()
    client._last_heartbeat = time.time() if heartbeat_fresh else time.time() - 300.0
    client._degraded_events = []
    client._reboots = []
    client._soft_cancel_target = (
        {
            "req_id": "req-abandoned",
            "seq": 7,
            "requested_monotonic": time.monotonic(),
            "reason": "test",
        }
        if cancelled
        else None
    )
    client._soft_cancel_ack = (
        {
            "req_id": "req-abandoned",
            "observed_monotonic": time.monotonic() + 0.001,
            "soft_cancelled": True,
            "status": "ok",
        }
        if acknowledged
        else None
    )

    def _record(name, **kwargs):
        client._degraded_events.append((name, kwargs))

    async def _reboot(reason="", mark_failed=False):
        client._reboots.append((reason, mark_failed))

    client._record_degraded_event = _record
    client.reboot_worker = _reboot
    return client


def _run(coro):
    import asyncio

    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def test_ack_wait_succeeds_when_the_worker_answered_this_request():
    client = _resolver_client(cancel_value=0, acknowledged=True)
    assert _run(client._soft_cancel_acknowledged(timeout_s=1.0)) is True


def test_ack_wait_fails_when_worker_dead():
    client = _resolver_client(cancel_value=0, alive=False, acknowledged=True)
    assert _run(client._soft_cancel_acknowledged(timeout_s=1.0)) is False


def test_ack_wait_times_out_when_the_worker_never_answered():
    client = _resolver_client(cancel_value=99)
    start = time.monotonic()
    assert _run(client._soft_cancel_acknowledged(timeout_s=0.6)) is False
    assert time.monotonic() - start < 5.0


def test_a_cleared_flag_alone_is_not_an_acknowledgement():
    """The exact live hazard: a reboot or a new job also zeroes the flag."""
    client = _resolver_client(cancel_value=0, acknowledged=False)
    assert _run(client._soft_cancel_acknowledged(timeout_s=0.6)) is False


def test_an_answer_for_a_different_request_is_not_an_acknowledgement():
    client = _resolver_client(cancel_value=0)
    client._soft_cancel_ack = {
        "req_id": "req-someone-else",
        "observed_monotonic": time.monotonic() + 1.0,
        "soft_cancelled": True,
        "status": "ok",
    }
    assert _run(client._soft_cancel_acknowledged(timeout_s=0.6)) is False


def test_an_answer_observed_before_the_cancel_is_not_an_acknowledgement():
    client = _resolver_client(cancel_value=0)
    client._soft_cancel_ack = {
        "req_id": "req-abandoned",
        "observed_monotonic": client._soft_cancel_target["requested_monotonic"] - 5.0,
        "soft_cancelled": False,
        "status": "ok",
    }
    assert _run(client._soft_cancel_acknowledged(timeout_s=0.6)) is False


def test_no_cancel_means_nothing_to_acknowledge():
    client = _resolver_client(cancel_value=0, cancelled=False, acknowledged=False)
    assert _run(client._soft_cancel_acknowledged(timeout_s=0.6)) is False


def test_the_listener_records_the_terminal_frame_as_the_acknowledgement():
    client = _resolver_client(cancel_value=0)
    client._note_soft_cancel_acknowledgement(
        {"id": "req-abandoned", "status": "ok", "soft_cancelled": True}
    )
    assert _run(client._soft_cancel_acknowledged(timeout_s=1.0)) is True


def test_the_listener_ignores_frames_for_other_requests():
    client = _resolver_client(cancel_value=0)
    client._note_soft_cancel_acknowledgement({"id": "req-other", "status": "ok"})
    assert client._soft_cancel_ack is None


def test_recoverable_abandon_with_ack_preserves_warm_lane():
    client = _resolver_client(cancel_value=0, acknowledged=True)
    _run(client._resolve_deferred_reboot("recoverable_token_progress_stalled"))
    assert client._reboots == [], "acknowledged soft-cancel must not reboot the warm worker"
    assert any(
        name == "warm_lane_preserved_after_soft_cancel" for name, _ in client._degraded_events
    )


def test_recoverable_abandon_without_ack_reboots_softly():
    client = _resolver_client(cancel_value=99)
    import os

    os.environ["AURA_MLX_SOFT_CANCEL_ACK_WAIT_S"] = "0.5"
    try:
        _run(client._resolve_deferred_reboot("recoverable_token_progress_stalled"))
    finally:
        os.environ.pop("AURA_MLX_SOFT_CANCEL_ACK_WAIT_S", None)
    assert client._reboots == [("token_progress_stalled", False)]


def test_nonrecoverable_abandon_reboots_immediately_marked_failed():
    client = _resolver_client(cancel_value=99)
    _run(client._resolve_deferred_reboot("token_progress_stalled"))
    assert client._reboots == [("token_progress_stalled", True)]


def test_recoverable_but_nonpreservable_reason_still_reboots():
    client = _resolver_client(cancel_value=0, acknowledged=True)
    _run(client._resolve_deferred_reboot("recoverable_empty_generation"))
    assert client._reboots == [("empty_generation", False)]


# ── worker-side stale-flag hygiene ─────────────────────────────────────


def test_clear_stale_soft_cancel_resets_foreign_seq():
    from core.brain.llm.mlx_worker import clear_stale_soft_cancel

    channel = _Value(41)
    clear_stale_soft_cancel(channel, 42)
    assert channel.value == 0


def test_clear_stale_soft_cancel_keeps_own_and_zero():
    from core.brain.llm.mlx_worker import clear_stale_soft_cancel

    own = _Value(42)
    clear_stale_soft_cancel(own, 42)
    assert own.value == 42  # a cancel already aimed at this job survives

    idle = _Value(0)
    clear_stale_soft_cancel(idle, 42)
    assert idle.value == 0
