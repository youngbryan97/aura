"""Every number on the inference health path came off another process.

`is_inference_ready` reads a lane-status payload the MLX worker wrote, then
converts fields with bare `int()` and `float()` and compares the results to
grace windows an operator sets by environment variable. Three ways that went
wrong, all of them the health endpoint believing arithmetic it should have
refused:

- `float("nan")` survives `float()`. `max(0.0, now - nan)` is `nan`, and every
  comparison against `nan` is False, so a NaN timestamp could not be caught by
  the staleness check it was supposed to fail.
- A timestamp from the future produced a clamped age of zero, which reads as
  the freshest possible progress. A skewed clock or a corrupt payload was
  therefore the strongest evidence of health the endpoint could receive.
- Grace windows were clamped at the minimum only. `AURA_INFERENCE_ACTIVE_
  STARTUP_GRACE_S=1e12` kept a permanently stalled generation classified as
  operational.

And a malformed payload — a string where a number belongs, a list where the
lane belongs — raised out of the endpoint instead of returning false with the
reason recorded.
"""
from __future__ import annotations

import time

import pytest

from core.brain.inference_gate import (
    _HEALTH_CLOCK_SKEW_TOLERANCE_S,
    _MAX_HEALTH_WINDOW_S,
    InferenceGate,
    _elapsed_since,
    _finite,
    _health_window_s,
)


# ─────────────────────────── the coercions refuse what they cannot use


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf"), "not a number", None, [], {}],
)
def test_unusable_values_are_missing_evidence_not_numbers(value):
    assert _finite(value) is None


def test_finite_numbers_pass_through():
    assert _finite("12.5") == 12.5
    assert _finite(3) == 3.0


def test_a_future_timestamp_is_not_fresh_progress():
    now = time.time()
    far_future = now + 3600.0

    assert _elapsed_since(far_future, now=now) is None, (
        "a future timestamp produced an age of zero, which reads as the "
        "freshest possible progress"
    )


def test_small_clock_skew_is_still_tolerated():
    """Another process's clock can sit slightly ahead. That is skew, not a
    broken payload, and it must not flip health to unhealthy."""
    now = time.time()

    age = _elapsed_since(now + (_HEALTH_CLOCK_SKEW_TOLERANCE_S / 2), now=now)

    assert age == 0.0


def test_a_nan_timestamp_proves_nothing():
    assert _elapsed_since(float("nan"), now=time.time()) is None


def test_absent_and_zero_timestamps_prove_nothing():
    now = time.time()

    assert _elapsed_since(0.0, now=now) is None
    assert _elapsed_since(None, now=now) is None


# ─────────────────────────── the windows are bounded at both ends


def test_a_huge_operator_window_is_capped():
    assert _health_window_s("1e12", default=120.0, minimum=15.0) == _MAX_HEALTH_WINDOW_S


def test_an_infinite_operator_window_falls_back_to_the_default():
    assert _health_window_s("inf", default=120.0, minimum=15.0) == 120.0


def test_a_tiny_operator_window_is_floored():
    assert _health_window_s("0.001", default=120.0, minimum=15.0) == 15.0


def test_an_unparseable_window_uses_the_default():
    assert _health_window_s("soon", default=45.0, minimum=10.0) == 45.0


# ─────────────────────────── a malformed lane returns false, never raises


def _lane(**overrides):
    lane = {
        "active_generations": 1,
        "foreground_owned": True,
        "current_request_started_at": time.time() - 1.0,
    }
    lane.update(overrides)
    return lane


def test_a_progressing_generation_is_operational():
    assert InferenceGate._active_generation_is_progressing(_lane()) is True


@pytest.mark.parametrize(
    "lane",
    [
        None,
        [],
        "ready",
        _lane(active_generations="lots"),
        _lane(active_generations=float("nan")),
        _lane(current_request_started_at="just now"),
        _lane(current_request_started_at=float("inf")),
        _lane(last_token_progress_at=float("nan")),
    ],
)
def test_a_malformed_lane_is_not_progressing_and_does_not_raise(lane):
    assert InferenceGate._active_generation_is_progressing(lane) is False


def test_a_stalled_generation_is_not_progressing():
    stalled = _lane(
        current_request_started_at=time.time() - 10_000.0,
        last_token_progress_at=time.time() - 10_000.0,
    )

    assert InferenceGate._active_generation_is_progressing(stalled) is False


def test_a_future_token_timestamp_does_not_certify_progress():
    """The clamp turned a future timestamp into a zero age — the strongest
    possible evidence of progress — so a broken clock read as perfect health."""
    lane = _lane(
        current_request_started_at=time.time() - 10_000.0,
        last_token_progress_at=time.time() + 10_000.0,
    )

    assert InferenceGate._active_generation_is_progressing(lane) is False


def test_a_lane_that_owns_nothing_is_not_progressing():
    assert InferenceGate._active_generation_is_progressing(_lane(foreground_owned=False)) is False


def test_long_prefill_is_operational_only_while_current_request_advances():
    now = time.time()
    lane = _lane(current_request_started_at=now - 300, last_prefill_progress_at=now - 1)
    assert InferenceGate._active_generation_is_progressing(lane) is True
    for stamp in (now - 301, now - 100, now + 10_000, float("nan"), "bad"):
        lane["last_prefill_progress_at"] = stamp
        assert InferenceGate._active_generation_is_progressing(lane) is False


def test_prefill_cannot_hide_a_stalled_decode():
    now = time.time()
    lane = _lane(current_request_started_at=now - 300,
                 last_prefill_progress_at=now - 1, last_token_progress_at=now - 100)
    assert InferenceGate._active_generation_is_progressing(lane) is False


# ─────────────────────────── one proof decision per answer


def test_proof_policy_is_read_once_per_readiness_answer(monkeypatch):
    """It was read at entry and again further down. A config change or a
    transient error between the two mixed proof rules with ordinary rules
    inside one result."""
    gate = InferenceGate()
    gate._initialized = True
    gate._mlx_client = None

    reads: list[str] = []

    def _counted(**_kwargs):
        reads.append("proof_run_active")
        return False

    monkeypatch.setattr("core.runtime.proof_policy.proof_run_active", _counted)
    monkeypatch.setattr(gate, "_iter_local_clients", lambda: {})

    gate.inference_readiness()

    assert len(reads) == 1, f"proof policy was read {len(reads)} times in one answer"


def test_an_unreadable_proof_policy_fails_closed(monkeypatch):
    gate = InferenceGate()
    gate._initialized = True
    gate._mlx_client = None

    def _broken(**_kwargs):
        raise RuntimeError("proof policy store is unreadable")

    monkeypatch.setattr("core.runtime.proof_policy.proof_run_active", _broken)
    monkeypatch.setattr(gate, "_iter_local_clients", lambda: {})

    ready, reason = gate.inference_readiness()

    assert ready is False
    assert reason == "proof_policy_unknown"


def test_a_proof_primary_run_refuses_a_lower_tier(monkeypatch):
    from types import SimpleNamespace

    gate = InferenceGate()
    gate._initialized = True
    gate._mlx_client = None

    monkeypatch.setattr("core.runtime.proof_policy.proof_run_active", lambda **_kw: True)
    monkeypatch.setattr("core.runtime.proof_policy.proof_model_tier", lambda: "primary")
    monkeypatch.setattr(
        gate,
        "_iter_local_clients",
        lambda: {
            "fallback": SimpleNamespace(
                is_alive=lambda: True,
                get_lane_status=lambda: {"conversation_ready": True},
            )
        },
    )

    ready, reason = gate.inference_readiness()

    assert ready is False
    assert reason == "proof_primary_requires_cortex"


# ─────────────────────────── liveness says which kind of alive it means


def test_deferred_boot_is_not_the_same_alive_as_a_running_worker(monkeypatch):
    gate = InferenceGate()
    gate._initialized = True
    gate._mlx_client = None
    monkeypatch.setattr(InferenceGate, "_desktop_safe_boot_enabled", staticmethod(lambda: True))

    assert gate.liveness_state() == "deferred"
    # Still "alive" — the gate can cold-start — but the caller can now tell.
    assert gate.is_alive() is True
    assert gate.is_inference_ready() is False


def test_a_running_worker_reports_backend_live():
    from types import SimpleNamespace

    gate = InferenceGate()
    gate._initialized = True
    gate._mlx_client = SimpleNamespace(is_alive=lambda: True)

    assert gate.liveness_state() == "backend_live"


def test_an_uninitialized_gate_is_down():
    gate = InferenceGate()
    gate._initialized = False

    assert gate.liveness_state() == "uninitialized"
    assert gate.is_alive() is False


def test_the_health_contract_binds_to_the_strict_check():
    """`is_alive` is true during deferred boot. The runtime health contract
    must not be reading that one."""
    from core.runtime import health_contract

    source = health_contract.__file__
    with open(source, encoding="utf-8") as handle:
        text = handle.read()

    assert 'liveness_check="is_inference_ready"' in text
