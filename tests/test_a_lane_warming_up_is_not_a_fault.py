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
