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
    _worker_measured_prefill_tps = 0.0

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


def test_a_measured_worker_uses_the_rate_the_worker_reported():
    """The worker's own clock, not the rate its progress messages arrive at.

    This used to set ``_prefill_tokens_per_s`` and expect the deadline to be
    built from it. That number is the interval between prefill PROGRESS
    MESSAGES, which cross an IPC queue onto a busy event loop, so it measures
    how often the parent got told. Live on 2026-09-07 the worker logged 410 to
    990 tok/s while this side had learned 56, and the deadline built from 56
    said a 52,020-character prompt would take 698 seconds to read against about
    thirty in fact.
    """
    worker = _Worker()
    worker._worker_measured_prefill_tps = 720.0
    assert worker._measured_prefill_rate() == 720.0
    # Faster worker, smaller floor.
    slow = _Worker()._prefill_floor_seconds(3431)
    assert worker._prefill_floor_seconds(3431) < slow


def test_the_progress_interval_estimate_sizes_no_deadline():
    """It stays for in-flight liveness and must never size a ceiling again."""
    worker = _Worker()
    worker._prefill_tokens_per_s = 720.0
    assert worker._measured_prefill_rate() == mlx_client._UNMEASURED_PREFILL_RATE


def test_the_rate_is_measured_between_observations_not_since_the_request_began():
    """Measuring from the request start folds in queueing and admission: it
    reported 4 tok/s on a worker doing 720, and asked for a ten-minute
    ceiling on a prompt that takes a second and a half to read."""
    import inspect

    source = inspect.getsource(mlx_client.MLXLocalClient._mark_prefill_progress)
    assert "_prefill_observed_at" in source
    assert "(done - last_done) / spent" in source
    # Averaged, so one slow chunk under contention does not become the rule.
    assert "previous * 0.7 + observed * 0.3" in source


def test_the_floor_only_ever_raises_a_ceiling():
    import inspect

    source = inspect.getsource(mlx_client.MLXLocalClient)
    where = source.index("needed = min(")
    block = source[where : where + 700]
    assert "0.0 < self._current_first_token_hard_ceiling_s < needed" in block
    # And never past the ceiling that catches a wedged worker.
    assert "_first_token_hard_ceiling(foreground_request=True)" in block
