"""A first-token deadline must not fire while the prompt is still being read.

Before the first token can exist, the whole prompt has to be read. Measured on
this host at about 720 tokens a second, a two-thousand-token prompt spends
nearly three seconds in prefill by design — so a caller whose budget is four
seconds cancels the request at the moment prefill finishes, every time, for
reasons that have nothing to do with the worker.

LIVE 2026-08-26: every decision she made while playing was cancelled this
way. "her reasoning produced nothing (no text came back)", over and over, so
she chose her moves from the consequence record alone and never held a plan.
From outside that looks exactly like a mind that is not thinking.
"""
from __future__ import annotations

import inspect

from core.brain.llm import mlx_client


def _cancel_block() -> str:
    source = inspect.getsource(mlx_client)
    where = source.index("prefilling = (")
    return source[where - 200 : where + 900]


def test_prefill_in_flight_lifts_the_callers_first_token_ceiling():
    block = _cancel_block()
    assert "_current_prefill_tokens_processed" in block
    assert "_current_prefill_tokens_total" in block
    assert "hard_first_token_ceiling = livelock_ceiling" in block


def test_a_finished_prefill_does_not_lift_it():
    """The lift is for work in flight, not for any request that once had a
    prompt: processed must still be short of total."""
    block = _cancel_block()
    assert "< self._current_prefill_tokens_total" in block


def test_a_stalled_prefill_does_not_lift_it_forever():
    """Progress has to be recent. A prefill that stopped advancing is not
    reading anything."""
    block = _cancel_block()
    assert "_last_progress_at" in block


def test_the_livelock_ceiling_still_applies():
    """A genuinely wedged prefill is still caught — the lift raises the
    caller's deadline to the livelock ceiling, never past it."""
    block = _cancel_block()
    assert "elapsed_without_token <= livelock_ceiling" in block
