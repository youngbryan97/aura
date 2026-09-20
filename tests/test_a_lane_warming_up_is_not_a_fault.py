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
