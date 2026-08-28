"""Progress decides whether a generation lives, not elapsed time.

A deadline that cancels a generation which is still producing tokens is not
protecting anything on this runtime. There is one person, one laptop, no queue
behind the turn and no bill. What a stopwatch was catching was a turn taking
longer than somebody guessed, and what it cost was the end of the answer.

What genuinely needs stopping still is: a worker that has gone silent, a decode
looping forever, and anything pathological enough to reach the absolute bound.
None of those are measured in elapsed time — the first two are measured in
tokens arriving and in what the tokens say.
"""

from __future__ import annotations

import time

from core.brain.inference_gate import InferenceGate
from core.brain.llm.mlx_client import MLXLocalClient


class _Client:
    """Just the progress bookkeeping the wait loop consults."""

    _still_producing = MLXLocalClient._still_producing

    def __init__(self, last_token_at: float) -> None:
        self._last_token_progress_at = last_token_at


def test_a_generation_still_emitting_tokens_is_alive() -> None:
    live = _Client(time.time())
    assert live._still_producing(within_s=20.0, foreground_request=True) is True


def test_a_generation_that_has_gone_quiet_is_not() -> None:
    """The case a deadline was standing in for, decided on the real signal."""

    quiet = _Client(time.time() - 60.0)
    assert quiet._still_producing(within_s=20.0, foreground_request=True) is False


def test_a_generation_that_never_started_is_not_a_slow_one() -> None:
    """Nothing has arrived, so this is silence and the first-token ceiling owns it."""

    never = _Client(0.0)
    assert never._still_producing(within_s=20.0, foreground_request=True) is False


def test_background_work_keeps_its_deadline() -> None:
    """One GPU. A dream cycle does not get to hold it while a person waits."""

    live = _Client(time.time())
    assert live._still_producing(within_s=20.0, foreground_request=False) is False


def test_the_background_budget_control_is_for_background() -> None:
    """It said "background requests only" and checked no such thing.

    The signal it scales on is registered under the name
    "background_token_budget", and it was trimming the answers people were
    waiting for.
    """

    import inspect

    source = inspect.getsource(InferenceGate)
    marker = "phi_scale = max(0.6, 0.6 + 0.4 * (phi_val / 0.8))"
    assert marker in source
    condition = source[source.rindex("if (", 0, source.index(marker)) : source.index(marker)]
    assert "is_background" in condition, condition


def test_a_user_facing_turn_is_allowed_a_full_reply() -> None:
    """A ceiling, not a reservation: the model stops when it has finished."""

    assert (
        InferenceGate._default_max_tokens_for_request(
            "desktop_quick_user", "primary", deep_handoff=False, is_background=False
        )
        == 4096
    )
    assert (
        InferenceGate._default_max_tokens_for_request(
            "dream_cycle", "primary", deep_handoff=False, is_background=True
        )
        == 384
    )
