"""A lane still coming up is skipped, and said so quietly.

In one live session a third of her warnings were "Circuit OPEN ... lane_not_ready:
cold" and "no endpoints matched routing plan" for lanes that were only
warming. They filled the panel's buffer and rotated the warnings worth reading
out of it. Skipping a warming lane is the design; saying it as a fault is not.
"""

from __future__ import annotations

import logging

from core.brain.llm_health_router import EndpointHealth, _only_warming


def test_a_lane_that_is_coming_up_is_only_warming():
    assert _only_warming("lane_not_ready:cold")
    assert _only_warming("lane_not_ready:handshaking")
    assert not _only_warming("client_returned_no_text")
    assert not _only_warming("mlx_runtime_unavailable:crashed")


def test_skipping_a_warming_lane_is_said_quietly(caplog):
    lane = EndpointHealth(name="Reflex", url="local://reflex", model="a small model")
    with caplog.at_level(logging.INFO, logger="Brain.HealthRouter"):
        lane.trip_temporarily("lane_not_ready:cold")
    said = [record for record in caplog.records if "Circuit OPEN" in record.getMessage()]
    assert said and all(record.levelno == logging.INFO for record in said)
    assert not lane.is_available()


def test_a_lane_that_failed_is_still_a_warning(caplog):
    lane = EndpointHealth(name="Reflex", url="local://reflex", model="a small model")
    with caplog.at_level(logging.INFO, logger="Brain.HealthRouter"):
        lane.trip_temporarily("mlx_runtime_unavailable:crashed")
    said = [record for record in caplog.records if "Circuit OPEN" in record.getMessage()]
    assert said and said[0].levelno == logging.WARNING


def _a_router_with(*lanes: EndpointHealth):
    from core.brain.llm_health_router import HealthAwareLLMRouter

    router = HealthAwareLLMRouter.__new__(HealthAwareLLMRouter)
    router.endpoints = {lane.name: lane for lane in lanes}
    router._last_fallback_warning_at = 0.0
    return router


def _the_fallback_line(caplog, router):
    with caplog.at_level(logging.INFO, logger="Brain.HealthRouter"):
        router._generate_core_part_6([], False, [], "curiosity", "tertiary")
    return [r for r in caplog.records if "no endpoints matched routing plan" in r.getMessage()]


def test_lanes_cooling_down_from_a_transient_trip_are_a_wait(caplog):
    """LIVE 2026-09-20: fifty warnings in one uptime while both background
    lanes cooled down between budget timeouts under host load. A transient
    trip leaves the failure streak alone by design; the line about it is
    not a warning either."""
    brainstem = EndpointHealth(name="Brainstem", url="local://brainstem", model="b")
    reflex = EndpointHealth(name="Reflex", url="local://reflex", model="r")
    brainstem.trip_temporarily("endpoint_timeout")
    reflex.trip_temporarily("lane_not_ready:handshaking")
    said = _the_fallback_line(caplog, _a_router_with(brainstem, reflex))
    assert said and all(r.levelno == logging.INFO for r in said)


def test_a_circuit_opened_by_counted_failures_is_still_a_warning(caplog):
    brainstem = EndpointHealth(name="Brainstem", url="local://brainstem", model="b")
    for _ in range(brainstem.failure_threshold):
        brainstem.record_failure("mlx_runtime_unavailable:crashed")
    assert not brainstem.is_available()
    said = _the_fallback_line(caplog, _a_router_with(brainstem))
    assert said and said[0].levelno == logging.WARNING


def test_an_admission_refusal_is_not_the_endpoints_failure(caplog):
    """LIVE 2026-09-20: `Circuit OPEN for Brainstem after 3 failures. Reason:
    event_loop_lag_1.0s` — admission had said "not now" for the host's loop
    lag, and the router counted it against the endpoint. The reasons
    admission produces are declared where it produces them, and the router
    reads them as waits."""
    from core.brain.llm_health_router import (
        _background_error_is_quiet,
        _is_transient_local_runtime_failure,
    )
    from core.runtime.control_plane import ADMISSION_REASON_PREFIXES, is_an_admission_reason

    for said in ("event_loop_lag_1.083s", "resource_busy", "moderate_memory_pressure_81.0", "candidate_worker_not_ready"):
        assert is_an_admission_reason(said), said
        assert _is_transient_local_runtime_failure(said), said
        assert _background_error_is_quiet(said), said
    # a worker that started and then died is a real event and stays loud
    assert not is_an_admission_reason("worker_died_during_generation")
    assert not _is_transient_local_runtime_failure("worker_died_during_generation")
    # The list is the controller's own: every reason it refuses with starts
    # with one. Each pressure it refuses on, against every kind of work.
    for said in _every_reason_the_controller_refuses_with():
        assert said.startswith(ADMISSION_REASON_PREFIXES), said
        assert _is_transient_local_runtime_failure(said), said
        assert _background_error_is_quiet(said), said


def _every_reason_the_controller_refuses_with() -> set[str]:
    """Each pressure branch driven, and a lane that is busy, and what came back."""
    import asyncio

    from core.runtime.control_plane import (
        AdmissionPriority,
        AdmissionRequest,
        PressureSnapshot,
        ResourceAdmissionController,
        WorkClass,
    )

    pressures = (
        PressureSnapshot(shutdown_requested=True),
        PressureSnapshot(suspended_capabilities=("background_exploration", "large_model_cortex")),
        PressureSnapshot(memory_percent=95.0),
        PressureSnapshot(memory_percent=88.0),
        PressureSnapshot(thermal_level=3),
        PressureSnapshot(thermal_level=2),
        PressureSnapshot(loop_monitor_running=False),
        PressureSnapshot(loop_lag_s=1.5),
        PressureSnapshot(red_zones=("pressure_provider_unavailable",)),
    )
    reasons = {
        block[0]
        for pressure in pressures
        for work in WorkClass
        for priority in AdmissionPriority
        if (
            block := ResourceAdmissionController._pressure_block_reason(
                AdmissionRequest(owner="test", work_class=work, priority=priority), pressure
            )
        )
    }
    assert reasons == {
        "runtime_shutdown_requested",
        "background_capability_suspended",
        "large_model_capability_suspended",
        "critical_memory_pressure_95.0",
        "moderate_memory_pressure_88.0",
        "critical_thermal_pressure_3",
        "serious_thermal_pressure_2",
        "event_loop_signal_unavailable",
        "event_loop_lag_1.500s",
        "pressure_provider_unavailable",
    }, "a pressure branch refused with a reason this test does not know"

    controller = ResourceAdmissionController(pressure_provider=lambda: PressureSnapshot())

    async def busy() -> str:
        held = await controller.acquire(
            AdmissionRequest(owner="holder", work_class=WorkClass.INFERENCE, lane="brainstem")
        )
        assert held.outcome.value == "admitted", held.reason
        waiting = await controller.acquire(
            AdmissionRequest(
                owner="waiter", work_class=WorkClass.INFERENCE, lane="brainstem", timeout_s=0
            )
        )
        return waiting.reason

    reasons.add(asyncio.run(busy()))
    return reasons


def test_a_lane_on_its_way_up_is_said_at_info():
    """Routed round by design, so not a warning (eleven in one boot, 2026-09-23)."""
    from core.brain.llm_health_router_endpoint_call import _a_lane_still_coming_up

    for state in ("recovering", "spawning", "handshaking", "warming"):
        assert _a_lane_still_coming_up(f"lane_not_ready:{state}")
    assert not _a_lane_still_coming_up("lane_not_ready:cold")
    assert not _a_lane_still_coming_up("worker_died_during_generation")


def test_the_fallback_line_never_takes_a_lanes_probe(caplog):
    """The line read every lane through the admission check, which grants an
    open circuit's one half-open probe once its cooldown has passed. A log
    line held the probe, and no caller could test the lane until the lease
    ran out."""
    from core.brain.llm_health_router import CircuitState

    brainstem = EndpointHealth(name="Brainstem", url="local://brainstem", model="b")
    for _ in range(brainstem.failure_threshold):
        brainstem.record_failure("mlx_runtime_unavailable:crashed")
    brainstem.cooldown_until_monotonic = 0.0  # its cooldown has passed

    _the_fallback_line(caplog, _a_router_with(brainstem))

    assert brainstem.state is CircuitState.OPEN
    assert brainstem.half_open_probe_at_monotonic == 0.0
    assert brainstem.is_available(), "the first real caller did not get the probe"


def test_the_fallback_line_names_each_lane_out_and_why(caplog):
    reflex = EndpointHealth(name="Reflex", url="local://reflex", model="r")
    for _ in range(reflex.failure_threshold):
        reflex.record_failure("mlx_runtime_unavailable:crashed")

    said = _the_fallback_line(caplog, _a_router_with(reflex))[0].getMessage()

    assert "Reflex (open: mlx_runtime_unavailable:crashed)" in said, said
    assert "No lane of this tier" in _the_fallback_line(caplog, _a_router_with())[-1].getMessage()
