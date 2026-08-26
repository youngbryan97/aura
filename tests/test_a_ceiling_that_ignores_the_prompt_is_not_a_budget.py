"""A first-token deadline has to know how much prompt there is.

LIVE 2026-08-26, two lines apart in the same log:

    first-token ceiling 90.0s for a 5-char prompt
    first-token ceiling  4.0s for a 3431-char prompt

Ninety seconds to read five characters; four to read nine hundred tokens. The
number had no relationship to the work, and every decision she made while
playing was cancelled by it — so she chose her moves from the consequence
record alone and never held a plan, which from outside looks exactly like a
mind that is not thinking.
"""
from __future__ import annotations

from core.brain.llm import mlx_client


class _Worker:
    """Just the parts of the client this arithmetic touches."""

    _prefill_tokens_per_s = 0.0

    _measured_prefill_rate = mlx_client.MLXLocalClient._measured_prefill_rate
    _prefill_floor_seconds = mlx_client.MLXLocalClient._prefill_floor_seconds


def test_an_unmeasured_worker_is_assumed_slow_rather_than_fast():
    """Being generous with an unmeasured worker costs a little latency. Being
    mean with it costs the answer."""
    worker = _Worker()
    assert worker._measured_prefill_rate() == mlx_client._UNMEASURED_PREFILL_RATE
    assert mlx_client._UNMEASURED_PREFILL_RATE < 716, "the rate observed on this host"


def test_a_longer_prompt_needs_longer():
    worker = _Worker()
    short = worker._prefill_floor_seconds(20)
    long = worker._prefill_floor_seconds(3431)
    assert long > short > 0.0
    # The 3431-char prompt that was being cancelled at four seconds.
    assert long > 4.0


def test_an_empty_prompt_needs_nothing():
    assert _Worker()._prefill_floor_seconds(0) == 0.0
    assert _Worker()._prefill_floor_seconds(-5) == 0.0


def test_a_measured_worker_uses_its_own_rate():
    worker = _Worker()
    worker._prefill_tokens_per_s = 720.0
    assert worker._measured_prefill_rate() == 720.0
    # Faster worker, smaller floor.
    slow = _Worker()._prefill_floor_seconds(3431)
    assert worker._prefill_floor_seconds(3431) < slow


def test_the_rate_is_measured_from_prefill_progress():
    import inspect

    source = inspect.getsource(mlx_client.MLXLocalClient._mark_prefill_progress)
    assert "_prefill_tokens_per_s" in source
    # Averaged, so one slow chunk under contention does not become the rule.
    assert "previous * 0.7 + observed * 0.3" in source


def test_the_floor_only_ever_raises_a_ceiling():
    import inspect

    source = inspect.getsource(mlx_client.MLXLocalClient)
    where = source.index("needed = self._prefill_floor_seconds")
    block = source[where : where + 400]
    assert "0.0 < self._current_first_token_hard_ceiling_s < needed" in block
