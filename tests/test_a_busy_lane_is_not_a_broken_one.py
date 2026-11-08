"""A loaded lane kept from conversation only by a generation under way on it is busy, not broken: the turn waits.

LIVE 8 November 2026, dress rehearsal: the person's question while a remark of hers was being written was refused as
"lanes_exhausted" twice, its lane "blocked by active_generation_in_flight".
"""

from core.brain.inference_gate import InferenceGate


def _lane(blockers, state="ready", **more):
    return {"state": state, "readiness_blockers": list(blockers), **more}


def test_a_generation_in_flight_is_only_busy():
    assert InferenceGate._lane_only_busy(_lane(["active_generation_in_flight"]))
    assert InferenceGate._lane_only_busy(_lane(["foreground_generation_active", "visible_conversation_probe_missing"]))


def test_warming_failed_or_otherwise_blocked_is_not():
    assert not InferenceGate._lane_only_busy(_lane([]))
    assert not InferenceGate._lane_only_busy(_lane(["active_generation_in_flight"], state="warming"))
    assert not InferenceGate._lane_only_busy(_lane(["active_generation_in_flight"], warmup_in_flight=True))
    assert not InferenceGate._lane_only_busy(_lane(["active_generation_in_flight", "warmup_in_flight"]))
    assert not InferenceGate._lane_only_busy(_lane(["warmup_foreground_owner"]))
    assert not InferenceGate._lane_only_busy(None)
