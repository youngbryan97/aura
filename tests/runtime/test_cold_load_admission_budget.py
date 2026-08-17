"""A cold cortex load must not be charged for the answer it cannot yet give.

LIVE 2026-08-17, measured four consecutive times: the first message typed after
launch died at 15-16 seconds with "the live answer lane could not finish
preparing before a reasoning turn began." Ten seconds later the same message
served normally.

The arithmetic, on the live runtime: the foreground timeout is ~80s, and the
admission budget is the turn's remaining time minus a reserve of
_DESKTOP_COGNITIVE_MIN_REQUIRED_BUDGET_S (60s) plus the response reserve (4s).
That leaves ~16s for admission — while a 32B cold load needs well over a
minute. The first turn after any launch could not have succeeded at any point,
and the message it produced reads as a fault rather than as a model loading.

Three earlier fixes this session aimed at plausible upstream causes (boot-phase
detection, foreign lane ownership, deferral handling). All three were real bugs
and none of them was THIS one, which is why the number never moved off 15s.
"""

from __future__ import annotations

from interface.routes.chat import _cortex_is_cold_loading


def test_a_cold_load_is_recognised() -> None:
    """Never ready, never served: the weights are still coming up."""
    lane = {
        "conversation_ready": False,
        "has_generated_successfully": False,
        "last_ready_at": 0.0,
    }

    assert _cortex_is_cold_loading(lane) is True


def test_a_ready_lane_is_not_cold() -> None:
    assert _cortex_is_cold_loading({"conversation_ready": True}) is False


def test_a_lane_that_already_served_is_a_recovery_not_a_cold_load() -> None:
    """Recovery keeps the short cap on purpose; it must not borrow this one."""
    lane = {
        "conversation_ready": False,
        "has_generated_successfully": True,
        "last_ready_at": 0.0,
    }

    assert _cortex_is_cold_loading(lane) is False


def test_a_lane_ready_earlier_is_not_cold() -> None:
    lane = {
        "conversation_ready": False,
        "has_generated_successfully": False,
        "last_ready_at": 1787002426.4,
    }

    assert _cortex_is_cold_loading(lane) is False


def test_garbage_is_not_cold() -> None:
    for value in (None, "", 0, [], "warming"):
        assert _cortex_is_cold_loading(value) is False


def test_unreadable_fields_do_not_raise() -> None:
    assert _cortex_is_cold_loading({"last_ready_at": "not-a-number"}) is False
