"""Her goal pass takes her model one generation at a time, and a person's turn waiting at the gate gets it in a gap.

LIVE 11 November 2026, 01:16 to 01:17: Bryan spoke to her three times in voice mode while a goal pass of hers read her
source. The pass held one generation lease for the whole of it, up to ten minutes and twelve generations. Each of his
turns waited at the gate; the router's soft cancel stopped the generation running, and the pass began the next one on
the same lease. Every turn ended "without establishing what happened", and the voice lane covered the silence with
"hmm" and "give me a moment".
"""
from __future__ import annotations

import asyncio

import core.agency.her_goal_at_work as work

GOAL = "Find something productive to work on"


def _a_pass(monkeypatch, steps, *, a_person_after=None):
    import core.brain.llm_health_router as router
    import core.runtime.service_access as access

    taken, given_back, asked = [], [], []
    waiting = [False]

    class Client:
        async def think_and_act(self, objective, **kwargs):
            asked.append(kwargs)
            step = steps[len(asked) - 1]
            if a_person_after is not None and len(asked) == a_person_after:
                waiting[0] = True  # a person's turn reaches the gate during this generation
            return step

    async def acquire(*, owner, timeout_s, wait_s):
        taken.append(owner)
        return len(taken)

    monkeypatch.setattr(access, "resolve_inference_gate", lambda default=None: type("Gate", (), {"_mlx_client": Client()})())
    monkeypatch.setattr(router, "acquire_external_generation_gate_lease", acquire)
    monkeypatch.setattr(router, "release_external_generation_gate_lease", given_back.append)
    monkeypatch.setattr(router, "a_person_is_waiting_for_her_model", lambda: waiting[0])
    result = asyncio.run(work.HerCortex().think_and_act(GOAL, tools={}, max_turns=12, prior_turns=[]))
    return result, taken, given_back, asked


def _call(n):
    return {"tool_calls": [{"tool": "read_source", "args": {"path": f"core/f{n}.py"}, "result": f"line {n}"}]}


def test_each_generation_has_its_own_lease_and_the_calls_so_far_go_into_the_next(monkeypatch):
    steps = [_call(1), _call(2), {"content": "The connections are never closed.", "tool_calls": []}]
    result, taken, given_back, asked = _a_pass(monkeypatch, steps)
    assert len(taken) == 3 and given_back == [1, 2, 3], "a lease per generation, each handed back"
    assert [c["args"]["path"] for c in result["tool_calls"]] == ["core/f1.py", "core/f2.py"]
    assert result["content"] == "The connections are never closed." and result["turns"] == 3
    assert [len(kwargs["prior_turns"]) for kwargs in asked] == [0, 2, 4], "each call and its result go in"
    ids = [m["tool_call_id"] for m in asked[2]["prior_turns"] if m["role"] == "tool"]
    assert len(set(ids)) == len(ids) == 2


def test_a_person_waiting_ends_the_pass_at_the_next_gap_and_it_keeps_what_it_did(monkeypatch, caplog):
    import logging

    steps = [_call(1), _call(2), _call(3)]
    with caplog.at_level(logging.INFO, logger="Aura.HerGoalAtWork"):
        result, taken, given_back, _asked = _a_pass(monkeypatch, steps, a_person_after=1)
    assert len(taken) == 1 and given_back == [1], "no lease taken after a person began to wait"
    assert [c["args"]["path"] for c in result["tool_calls"]] == ["core/f1.py"]
    assert any("a person is talking with her" in r.getMessage() for r in caplog.records)


def test_the_router_counts_a_person_only_while_their_turn_waits():
    from core.brain.llm_health_router import _a_person_waits, a_person_is_waiting_for_her_model

    assert a_person_is_waiting_for_her_model() is False
    with _a_person_waits():
        assert a_person_is_waiting_for_her_model() is True
        with _a_person_waits():
            assert a_person_is_waiting_for_her_model() is True
        assert a_person_is_waiting_for_her_model() is True
    assert a_person_is_waiting_for_her_model() is False


def test_a_person_s_turn_is_counted_while_it_waits_at_the_gate(monkeypatch):
    """Through the router itself: the foreground ladder's wait is the one counted."""
    import threading

    from core.brain import llm_health_router as router_module
    from core.brain.llm_health_router import HealthAwareLLMRouter

    monkeypatch.setattr(router_module, "_GENERATION_GATE", threading.BoundedSemaphore(1))
    monkeypatch.setattr(router_module, "_GENERATION_GATE_ACTIVE_LEASES", {})
    monkeypatch.setattr(router_module, "_GENERATION_GATE_LEASE_DEADLINES", {})
    monkeypatch.setattr(router_module, "_GENERATION_GATE_FORCED_LEASES", set())
    monkeypatch.setattr(router_module, "_GENERATION_GATE_FOREGROUND_LEASES", set())
    seen = []

    async def watching(wait_s):
        seen.append(router_module.a_person_is_waiting_for_her_model())
        return True

    router = HealthAwareLLMRouter()

    async def gated(*_a, **_k):
        return {"ok": True, "text": "reply", "endpoint": "local", "tokens": 1}

    monkeypatch.setattr(router, "_generate_with_metadata_gated", gated)
    monkeypatch.setattr(router, "_background_suppression_result", lambda **_kw: None)
    monkeypatch.setattr(router_module, "_acquire_generation_gate_slot", watching)
    result = asyncio.run(router.generate_with_metadata(
        "hello", origin="user", purpose="response_generation_user", foreground_request=True))
    assert result["ok"] is True
    assert seen == [True], "counted while it waited"
    assert router_module.a_person_is_waiting_for_her_model() is False, "and not after"
