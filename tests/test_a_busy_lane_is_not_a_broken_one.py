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


async def test_a_turn_waits_its_turn_on_a_busy_lane_and_warms_nothing(monkeypatch):
    """LIVE 24 November 2026: a question asked while a program was being written warmed a lane that was serving, waited
    15 s for it behind the writing, and was answered from a second 27B loaded for it."""
    gate = InferenceGate.__new__(InferenceGate)
    warmed, reset = [], []

    class Client:
        async def warmup(self):
            warmed.append(True)

    gate._mlx_client = Client()
    busy = _lane(["active_generation_in_flight"], last_ready_at=5.0, conversation_ready=False)
    monkeypatch.setattr(gate, "get_conversation_status", lambda: busy, raising=False)
    monkeypatch.setattr(gate, "_reset_cortex_warmup_backoff", lambda: reset.append(True), raising=False)
    assert await gate.ensure_foreground_ready(timeout=20.0) is busy
    assert warmed == [] and reset == [True]
