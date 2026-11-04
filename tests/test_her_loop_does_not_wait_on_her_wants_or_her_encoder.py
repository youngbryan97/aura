"""Two blocking locks lockdep caught held on her event loop thread, on the day runs of 1 and 3 November.

- `conation.what_she_wants`, 425 ms, from the overt action loop's coroutine: every change to what she wants copies all
  she keeps under its lock.
- `vector_memory_engine.encode`, 55 ms, from the inference gate's coroutine: the check of whether a draft reply is for
  downstream repair encodes it.

Both are now made off the loop. The action loop's record keeping goes to a worker thread; the Will's leave stays on the
loop. The gate assesses the draft on a worker thread and hands the assessment in.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


def test_the_action_loop_keeps_her_record_off_the_loop():
    from core.runtime import overt_action_loop

    source = inspect.getsource(overt_action_loop.OvertActionLoop.run_once)
    for helper in ("_decide_for_later", "_she_is_free_again", "_begin_doing", "_end_doing"):
        assert f"asyncio.to_thread({helper}" in source or f"asyncio.to_thread(_end_doing" in source, helper
    assert "asyncio.shield(asyncio.to_thread(_end_doing" in source, "a cancelled act still ends what it began"
    # The Will's leave is asked on the loop, where the Will lives.
    assert "_what_she_decided(self)" in source
    assert "asyncio.to_thread(_what_she_decided" not in source
    assert "asyncio.to_thread(_note_want" in inspect.getsource(overt_action_loop)


def test_a_draft_assessed_off_the_loop_is_not_assessed_again(monkeypatch):
    from core.brain import inference_gate

    def refuse(*_args, **_kwargs):
        raise AssertionError("assessed on the loop again")

    monkeypatch.setattr(inference_gate, "assess_user_facing_reply", refuse)
    gate = object.__new__(inference_gate.InferenceGate)
    kept = gate._repairable_user_facing_draft_for_downstream(
        "a draft", "a prompt", assessment=SimpleNamespace(retryable=False, reasons=())
    )
    assert kept is None


def test_the_gate_assesses_the_draft_on_a_worker_thread():
    from core.brain import inference_gate

    source = inspect.getsource(inference_gate.InferenceGate._generate_with_metadata_sink)
    assert "await asyncio.to_thread(\n" in source and "assess_user_facing_reply, visible_user_prompt" in source
