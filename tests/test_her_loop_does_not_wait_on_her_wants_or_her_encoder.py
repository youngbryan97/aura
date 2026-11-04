"""Two blocking locks lockdep caught held on her event loop thread, on the day runs of 1 and 3 November.

- `conation.what_she_wants`, 425 ms, from the overt action loop's coroutine: every change to what she wants copies all
  she keeps under its lock.
- `vector_memory_engine.encode`, 55 ms, from the inference gate's coroutine: the check of whether a draft reply is for
  downstream repair encodes it.

The action loop's record keeping now goes to a worker thread; the Will's leave stays on the loop. The gate's check
stays on the loop: moved to a worker thread, the encode outlived a cancelled turn holding the model lane's lock file
(the regression of 3 November caught it as a leaked open file), which costs more than 55 ms once.
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
