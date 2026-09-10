"""A probe missing its wait budget once is the design working.

The boot-health probe is a singleflight: an HTTP caller waits up to 2.5s and
is then handed fresh cached evidence while the probe finishes in its own time.
Missing the budget under load is what that mechanism is for.

It logged a warning every time. Live on 2026-09-07 that was generations 20
through 24, five consecutive warnings, while a person was being answered — and
the guide is explicit that expected backpressure logs at info and becomes a
degradation only when it is persistent.

So the run decides. One miss is information; a run of them is a probe that is
not coming back.
"""

from __future__ import annotations

import pytest

from interface.routes import system as system_routes


@pytest.fixture(autouse=True)
def _fresh_probe_state():
    with system_routes._HEALTH_PROBE_STATE_LOCK:
        before = dict(system_routes._HEALTH_PROBE_STATE)
        system_routes._HEALTH_PROBE_STATE.pop("timeout_recorded_generation", None)
        system_routes._HEALTH_PROBE_STATE.pop("consecutive_timeouts", None)
    yield
    with system_routes._HEALTH_PROBE_STATE_LOCK:
        system_routes._HEALTH_PROBE_STATE.clear()
        system_routes._HEALTH_PROBE_STATE.update(before)


def _miss(generation: int) -> int:
    state, recorded = system_routes._record_health_probe_wait_timeout(generation)
    assert recorded
    return int(state["consecutive_timeouts"])


def test_consecutive_generations_extend_the_run() -> None:
    assert _miss(20) == 1
    assert _miss(21) == 2
    assert _miss(22) == 3
    assert _miss(23) == 4
    assert _miss(24) == 5


def test_a_generation_that_came_back_resets_the_run() -> None:
    """A busy stretch is one run, not a fresh alarm each time."""

    assert _miss(20) == 1
    assert _miss(21) == 2
    # 22 answered inside its budget; 23 misses.
    assert _miss(23) == 1


def test_the_run_is_on_the_health_surface() -> None:
    """A reader asking 'backpressure or fault' wants the run, not the total."""

    _miss(7)
    snapshot = system_routes._health_probe_state_snapshot()
    assert "consecutive_timeouts" in snapshot
    assert "total_timeouts" in snapshot


def test_the_threshold_is_named_and_the_log_reads_it() -> None:
    import inspect

    assert system_routes._HEALTH_PROBE_TIMEOUTS_BEFORE_A_WARNING >= 2
    body = inspect.getsource(system_routes)
    marker = body.index("exceeded the %.1fs HTTP wait budget")
    before = body[marker - 700 : marker]
    assert "_HEALTH_PROBE_TIMEOUTS_BEFORE_A_WARNING" in before
    assert "logger.info" in before
    assert "generation == 1" not in before, (
        "the old special case decided by generation number, not by whether "
        "the condition had persisted"
    )


def test_the_perceptual_pump_uses_the_same_rule() -> None:
    """The same shape, in the other place that warned on every first miss.

    The substrate injection warned when a streak began, so an intermittently
    slow host warned continuously — every live line read "(overruns=N,
    streak=1)". A single overrun costs this frame's freshness, off the pump
    thread; a run is a substrate not keeping up.
    """

    import inspect

    from core.perception.perceptual_pump import PerceptualPump

    assert PerceptualPump.SUBSTRATE_OVERRUNS_BEFORE_A_WARNING >= 2
    body = inspect.getsource(PerceptualPump)
    marker = body.index("Perceptual substrate transaction exceeded budget: ")
    before = body[marker - 900 : marker]
    assert "SUBSTRATE_OVERRUNS_BEFORE_A_WARNING" in before
    assert "_substrate_injection_overrun_streak == 1\n" not in before, (
        "the streak-of-one branch still decides the warning"
    )
    assert "logger.info" in body[marker : marker + 1200], (
        "an isolated overrun should still be reported, at info"
    )


def test_a_signed_steering_detachment_is_not_a_fault() -> None:
    """The worker decides it, logs it at info, and now says why to the parent.

    The parent was sent the boolean and not the reason, so it warned about a
    deliberate detachment on every call — live on 2026-09-07, "Liveness flag
    CLEAR for this worker (origin=api_stabilizer, gen=0)" beside the worker's
    own "remains detached under signed migration disposition:
    steering_generation_deferred".
    """

    import inspect

    from core.brain.llm import mlx_client as mc

    assert "steering_generation_deferred" in mc._EXPECTED_STEERING_DETACHMENTS
    assert "steering_generation_retired" in mc._EXPECTED_STEERING_DETACHMENTS

    body = inspect.getsource(mc)
    # The log line itself, not the comment above it that quotes it.
    marker = body.index('"⚠️ [STEERING] Liveness flag CLEAR for this worker')
    branch = body[marker - 1800 : marker]
    # The disposition decides which of the two lines is taken, and both are
    # here: an expected detachment reports at info, an unexplained one warns.
    assert "_EXPECTED_STEERING_DETACHMENTS" in branch
    assert "Detached under a signed disposition" in branch
    assert "logger.info(" in branch
    assert "logger.warning(" in body[marker - 220 : marker]


def test_the_worker_sends_the_disposition() -> None:
    from pathlib import Path

    worker = Path(__file__).resolve().parents[1] / "core/brain/llm/mlx_worker.py"
    body = worker.read_text("utf-8")
    assert '"steering_disposition": _steering_disposition,' in body
