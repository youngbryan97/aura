"""A lease that a clock change can revoke, and a wedge nobody proved.

Two CP126 findings in foreground ownership — the thing that decides whether a
background worker may respawn while a person's turn is running.

6595b0e1 — acquisition timestamps and ages used time.time(). An NTP step or a
sleep/wake made a healthy 32B cold load look minutes stale and cleared it
mid-load, which is the 'cortex warming forever' deadlock; meanwhile a genuinely
wedged task held until a guessed age, because nothing measured progress.

8f772011 — the force-clear API checked age and an owner prefix. Age is not
evidence of a wedge: a cold load is legitimately slow.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from core.brain.llm import mlx_client

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _restore_owner():
    fields = (
        "_FOREGROUND_OWNER_NAME",
        "_FOREGROUND_OWNER_ACQUIRED_AT",
        "_FOREGROUND_OWNER_STALE_AFTER",
        "_FOREGROUND_OWNER_ACQUIRED_MONOTONIC",
        "_FOREGROUND_OWNER_HEARTBEAT_MONOTONIC",
        "_FOREGROUND_OWNER_IS_USER_FACING",
    )
    saved = {name: getattr(mlx_client, name) for name in fields}
    yield
    for name, value in saved.items():
        setattr(mlx_client, name, value)


def _own(seconds_ago: float, name: str = "chat_api:default") -> None:
    mlx_client._FOREGROUND_OWNER_NAME = name
    mlx_client._stamp_foreground_owner(time.time() - seconds_ago)


# --- the age is monotonic (6595b0e1) ------------------------------------


def test_a_wall_clock_step_does_not_age_the_owner(monkeypatch):
    """An NTP step used to clear a healthy cold load mid-load."""
    _own(5.0)
    before = mlx_client._foreground_owner_age()

    # The wall clock jumps an hour forward; the monotonic clock does not.
    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() + 3600.0)

    assert mlx_client._foreground_owner_age() == pytest.approx(before, abs=1.0)


def test_the_age_still_grows_with_real_elapsed_time():
    _own(120.0)

    assert mlx_client._foreground_owner_age() >= 119.0


def test_no_owner_has_no_age():
    mlx_client._FOREGROUND_OWNER_NAME = None
    mlx_client._stamp_foreground_owner(0.0)

    assert mlx_client._foreground_owner_age() == 0.0


# --- silence, not age, is what a wedge looks like (8f772011) ------------


def test_a_working_owner_reports_no_silence():
    _own(120.0)
    mlx_client.note_foreground_owner_progress()

    assert mlx_client._foreground_owner_silence() < 1.0
    assert mlx_client._foreground_owner_age() >= 119.0


def test_a_silent_owner_accumulates_silence():
    _own(120.0)

    assert mlx_client._foreground_owner_silence() >= 119.0


def test_status_cleanup_preserves_an_old_owner_reporting_progress():
    _own(600.0)
    mlx_client.note_foreground_owner_progress()

    assert mlx_client._clear_stale_foreground_owner() is None
    assert mlx_client._FOREGROUND_OWNER_NAME == "chat_api:default"


def test_status_cleanup_releases_an_old_silent_owner():
    _own(600.0)

    assert mlx_client._clear_stale_foreground_owner() == "chat_api:default"
    assert mlx_client._FOREGROUND_OWNER_NAME is None


def test_waiting_client_cannot_expire_a_progressing_owner(monkeypatch):
    _own(600.0)
    mlx_client._FOREGROUND_OWNER_STALE_AFTER = 200.0
    mlx_client.note_foreground_owner_progress()
    monkeypatch.setattr(mlx_client, "_foreground_owner_wait_budget", lambda *a, **k: 0.01)

    async def contend():
        with pytest.raises(TimeoutError):
            async with mlx_client._foreground_owner_context("other", foreground_request=False):
                pytest.fail("stole a progressing owner")

    asyncio.run(contend())
    assert mlx_client._FOREGROUND_OWNER_NAME == "chat_api:default"


def test_a_slow_but_working_owner_is_not_force_cleared():
    """A 32B cold load is legitimately slow; clearing it mid-load is the
    deadlock this guard exists to avoid."""
    _own(300.0)
    mlx_client.note_foreground_owner_progress()

    result = mlx_client.force_clear_foreground_owner(reason="test", min_age_s=45.0)

    assert result["cleared"] is False
    assert result["detail"] == "owner_still_reporting_progress"
    assert mlx_client._FOREGROUND_OWNER_NAME == "chat_api:default"


def test_late_release_preserves_successor_with_same_label():
    mlx_client._FOREGROUND_OWNER_NAME = None
    mlx_client._stamp_foreground_owner(0.0)

    async def replace():
        async with mlx_client._foreground_owner_context("chat_api:default"):
            # Reclamation followed by reacquisition can reuse the client label.
            previous = mlx_client._FOREGROUND_OWNER_ACQUIRED_MONOTONIC
            mlx_client._FOREGROUND_OWNER_ACQUIRED_MONOTONIC = previous + 1.0
            mlx_client._FOREGROUND_OWNER_HEARTBEAT_MONOTONIC = previous + 1.0
        assert mlx_client._FOREGROUND_OWNER_NAME == "chat_api:default"
        assert mlx_client._FOREGROUND_OWNER_ACQUIRED_MONOTONIC == previous + 1.0

    asyncio.run(replace())


def test_a_silent_owner_is_force_cleared():
    _own(300.0)

    result = mlx_client.force_clear_foreground_owner(reason="test", min_age_s=45.0)

    assert result["cleared"] is True
    assert mlx_client._FOREGROUND_OWNER_NAME is None


def test_a_caller_with_its_own_proof_can_override():
    """A desktop HTTP timeout has already observed the turn fail; it does not
    need the heartbeat to agree."""
    _own(300.0)
    mlx_client.note_foreground_owner_progress()

    result = mlx_client.force_clear_foreground_owner(
        reason="http_timeout", min_age_s=45.0, require_silence=False
    )

    assert result["cleared"] is True


def test_a_young_owner_is_never_cleared():
    _own(2.0)

    result = mlx_client.force_clear_foreground_owner(reason="test", min_age_s=45.0)

    assert result["cleared"] is False
    assert result["detail"] == "owner_younger_than_min_age"


def test_the_heartbeat_only_applies_to_a_live_owner():
    mlx_client._FOREGROUND_OWNER_NAME = None
    mlx_client._stamp_foreground_owner(0.0)
    before = mlx_client._FOREGROUND_OWNER_HEARTBEAT_MONOTONIC

    mlx_client.note_foreground_owner_progress()

    assert mlx_client._FOREGROUND_OWNER_HEARTBEAT_MONOTONIC == before


def test_the_stamps_stay_coherent():
    """Three views of one fact. Setting one and not the others is how a stale
    monotonic stamp made an old owner look freshly acquired."""
    _own(90.0)

    assert mlx_client._FOREGROUND_OWNER_ACQUIRED_MONOTONIC > 0.0
    assert mlx_client._FOREGROUND_OWNER_HEARTBEAT_MONOTONIC == (
        mlx_client._FOREGROUND_OWNER_ACQUIRED_MONOTONIC
    )
    assert mlx_client._foreground_owner_age() == pytest.approx(90.0, abs=2.0)
