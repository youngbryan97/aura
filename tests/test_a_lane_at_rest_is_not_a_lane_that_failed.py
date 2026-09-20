"""Cold is where a demand-loaded lane rests, not a state it failed into.

LIVE 2026-09-19, over and over on one boot:

    Endpoint Brainstem failed validation: lane_not_ready:cold
    Circuit OPEN for Brainstem on transient runtime failure.

`_local_client_failure_reason` listed `cold` beside `recovering`,
`spawning`, `handshaking` and `warming` as states a lane must not be given
work in. For the resident foreground lane that is right, and the comment
there says why: a foreground generation handed to a warming Cortex blocks
the caller for 105 seconds while warmup contends for the single worker,
which in July took a busy Aura's 32B away permanently.

For a lane that loads on demand it is the opposite. Cold is its resting
state, sending it work is how it stops being cold, and refusing it for
being at rest opened a circuit that then prevented the load that would have
cleared it.

The gate's own tier-health sweep already answers this the other way: its
transitional set is ("spawning", "handshaking", "warming", "recovering") —
without `cold` — and it calls a cold brainstem `standby`, `cold_by_policy`.
Two places, one question, two answers.
"""
from __future__ import annotations

import pytest

from core.brain import llm_health_router as router
from core.brain.llm.model_registry import BRAINSTEM_ENDPOINT, PRIMARY_ENDPOINT


class _LaneIn:
    def __init__(self, state: str) -> None:
        self._state = state

    def get_lane_status(self) -> dict[str, object]:
        return {"state": self._state, "conversation_ready": False}


def test_a_cold_resident_lane_is_still_refused():
    """The protection this rule was written for is untouched."""
    assert (
        router._local_client_failure_reason(_LaneIn("cold"), cold_is_standby=False)
        == "lane_not_ready:cold"
    )


def test_a_cold_demand_loaded_lane_is_available():
    """Nothing refuses it, so something can wake it."""
    assert router._local_client_failure_reason(
        _LaneIn("cold"), cold_is_standby=True
    ) == ""


@pytest.mark.parametrize("state", ["recovering", "spawning", "handshaking", "warming"])
def test_a_lane_actually_coming_up_is_refused_either_way(state):
    """Cold is the only state that changes. The rest are transitional on
    every lane, and a lane mid-warmup still must not be handed work."""
    for standby in (True, False):
        assert router._local_client_failure_reason(
            _LaneIn(state), cold_is_standby=standby
        ) == f"lane_not_ready:{state}"


def test_a_failed_lane_is_failed_whatever_it_rests_at():
    """`failed` is not a resting state and is not reached by this rule."""
    class _Failed:
        def get_lane_status(self) -> dict[str, object]:
            return {"state": "failed", "last_error": "worker died"}

    assert (
        router._local_client_failure_reason(_Failed(), cold_is_standby=True)
        == "worker died"
    )


def test_only_the_resident_lane_is_held_to_the_resident_rule():
    """The call sites decide by the one constant that names the resident lane,
    rather than by a list of the lanes that are not it."""
    from tests.source_contract import family_text

    # the call sites are in the endpoint-call mixin now
    source = family_text(router)
    assert 'cold_is_standby=str(getattr(ep, "name", ""))' in source
    assert "!= PRIMARY_ENDPOINT" in source
    assert PRIMARY_ENDPOINT != BRAINSTEM_ENDPOINT
